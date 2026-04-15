from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency in runtime
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = Path(__file__).resolve().parent


PROBE_WORDS = {
    "personas": ["hombre", "mujer", "padre", "madre", "nino", "nina"],
    "tiempo_vida": ["mundo", "vida", "tiempo", "ano", "dia", "noche"],
    "verbos": ["dijo", "fue", "era", "tiene", "hace", "esta"],
}


@dataclass
class TrainConfig:
    batch_size: int = 16
    block_size: int = 128
    vocab_size: int = 65536
    n_embd: int = 768 # 192
    n_head: int = 24 # 6
    n_layer: int = 4

    max_iters: int = 100000 # 25000
    eval_interval: int = 500 # 500
    eval_batches: int = 20
    save_interval: int = 500 # 500
    grad_log_interval: int = 20
    umap_interval: int = 2000
    umap_sample_size: int = 2000
    umap_random_state: int = 42

    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_iters: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    train_bin: str = str(REPO_ROOT / "train.bin")
    val_bin: str = str(REPO_ROOT / "val.bin")
    checkpoint_path: str = str(REPO_ROOT / "pequellm_pesado_checkpoint.pth")
    tokenizer_path: str = str(REPO_ROOT / "tokenizer-culturax-es-hf.json")
    output_root: str = str(EMB_DIR / "artifacts_gpt2")

    resume: bool = True
    device: str = "auto"
    precision: str = "auto"
    seed: int = 42
    generate_tokens: int = 100
    run_name: str = ""
    auto_presentation: bool = True


class Head(nn.Module):
    def __init__(self, n_embd: int, block_size: int, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        k = self.key(x)
        q = self.query(x)
        att = q @ k.transpose(-2, -1) * (channels ** -0.5)
        att = att.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        return att @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, block_size: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd=n_embd, block_size=block_size, head_size=head_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, block_size: int, n_head: int):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd=n_embd, block_size=block_size, n_head=n_head)
        self.ffwd = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPTModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, cfg.block_size, cfg.n_head) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = idx.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class BinTokenDataset:
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset bin not found: {self.path}")
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")

    def get_batch(self, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - block_size - 1
        if max_start <= 0:
            raise ValueError(f"Dataset {self.path} is too short for block_size={block_size}")
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i : i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1 : i + block_size + 1]).astype(np.int64)) for i in ix])
        return x, y


def select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class AmpSettings:
    enabled: bool
    device_type: str
    dtype: torch.dtype
    use_grad_scaler: bool


def resolve_amp_settings(cfg: TrainConfig, device: str) -> AmpSettings:
    requested = cfg.precision.lower().strip()
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        requested = "auto"

    # AMP is mainly useful on CUDA; keep CPU/MPS in fp32 for stability.
    if device != "cuda":
        return AmpSettings(enabled=False, device_type=device, dtype=torch.float32, use_grad_scaler=False)

    if requested == "fp32":
        return AmpSettings(enabled=False, device_type="cuda", dtype=torch.float32, use_grad_scaler=False)

    if requested == "bf16":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return AmpSettings(enabled=True, device_type="cuda", dtype=torch.bfloat16, use_grad_scaler=False)
        # fall back to fp16 if bf16 unsupported
        return AmpSettings(enabled=True, device_type="cuda", dtype=torch.float16, use_grad_scaler=True)

    if requested == "fp16":
        return AmpSettings(enabled=True, device_type="cuda", dtype=torch.float16, use_grad_scaler=True)

    # auto
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return AmpSettings(enabled=True, device_type="cuda", dtype=torch.bfloat16, use_grad_scaler=False)
    return AmpSettings(enabled=True, device_type="cuda", dtype=torch.float16, use_grad_scaler=True)


def get_lr(step: int, cfg: TrainConfig) -> float:
    if cfg.max_iters <= 0:
        return cfg.lr_min
    if step < cfg.warmup_iters:
        return cfg.lr_max * float(step + 1) / float(max(1, cfg.warmup_iters))
    if step >= cfg.max_iters:
        return cfg.lr_min
    decay_ratio = (step - cfg.warmup_iters) / float(max(1, cfg.max_iters - cfg.warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.lr_min + coeff * (cfg.lr_max - cfg.lr_min)


def configure_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embedding" not in name and "ln" not in name and not name.endswith("bias"):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=cfg.lr_max, betas=(cfg.beta1, cfg.beta2))


def evaluate_loss(model: GPTModel, ds: BinTokenDataset, cfg: TrainConfig, device: str, amp: AmpSettings) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            xb, yb = ds.get_batch(cfg.block_size, cfg.batch_size)
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.autocast(
                device_type=amp.device_type,
                dtype=amp.dtype,
                enabled=amp.enabled,
            ):
                _, loss = model(xb, yb)
            if loss is not None:
                losses.append(float(loss.item()))
    model.train()
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def collect_layer_grad_norms(model: nn.Module) -> Dict[str, float]:
    layer_sumsq: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        layer_name = name.rsplit(".", 1)[0] if "." in name else name
        grad = param.grad.detach().float()
        layer_sumsq[layer_name] = layer_sumsq.get(layer_name, 0.0) + float(torch.sum(grad * grad).item())
    return {name: math.sqrt(value) for name, value in layer_sumsq.items()}


def reduce_to_2d(vectors: np.ndarray, random_state: int) -> Tuple[np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.05,
            metric="cosine",
            random_state=random_state,
        )
        return reducer.fit_transform(vectors), "umap"
    except Exception:
        try:
            from sklearn.decomposition import PCA

            return PCA(n_components=2, random_state=random_state).fit_transform(vectors), "pca_fallback"
        except Exception:
            if vectors.shape[1] >= 2:
                return vectors[:, :2], "raw2d_fallback"
            pad = np.zeros((vectors.shape[0], 2), dtype=vectors.dtype)
            pad[:, : vectors.shape[1]] = vectors
            return pad, "raw2d_fallback"


def save_umap_snapshot(model: GPTModel, sample_ids: np.ndarray, iteration: int, out_dir: Path, random_state: int) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = model.token_embedding_table.weight.detach().cpu().numpy()
    sampled = emb[sample_ids]
    coords, method = reduce_to_2d(sampled, random_state=random_state)

    csv_path = out_dir / f"umap_tokens_iter_{iteration:06d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "x", "y", "method"])
        for idx, (x, y) in zip(sample_ids.tolist(), coords.tolist()):
            writer.writerow([idx, x, y, method])

    if plt is not None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.65)
        ax.set_title(f"Token embedding projection at iter={iteration} ({method})")
        ax.set_xlabel("axis_1")
        ax.set_ylabel("axis_2")
        fig.tight_layout()
        fig.savefig(out_dir / f"umap_tokens_iter_{iteration:06d}.png", dpi=160)
        plt.close(fig)
    return method


def compute_probe_metrics(model: GPTModel, tokenizer: Tokenizer | None) -> Dict[str, float]:
    if tokenizer is None:
        return {"probe_intra_cos": float("nan"), "probe_inter_cos": float("nan"), "probe_gap": float("nan"), "probe_knn_purity": float("nan")}

    emb_table = model.token_embedding_table.weight.detach().cpu()
    vectors = []
    labels = []
    for category, words in PROBE_WORDS.items():
        for word in words:
            ids = tokenizer.encode(word).ids
            if not ids:
                continue
            vec = emb_table[ids].mean(dim=0)
            vectors.append(vec.numpy())
            labels.append(category)

    if len(vectors) < 4:
        return {"probe_intra_cos": float("nan"), "probe_inter_cos": float("nan"), "probe_gap": float("nan"), "probe_knn_purity": float("nan")}

    mat = np.vstack(vectors)
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norm
    cos = mat @ mat.T

    intra_vals = []
    inter_vals = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_vals.append(cos[i, j])
            else:
                inter_vals.append(cos[i, j])

    intra = float(np.mean(intra_vals)) if intra_vals else float("nan")
    inter = float(np.mean(inter_vals)) if inter_vals else float("nan")
    gap = intra - inter if not (math.isnan(intra) or math.isnan(inter)) else float("nan")

    knn_hits = 0
    for i in range(len(labels)):
        sim = cos[i].copy()
        sim[i] = -1.0
        nn_idx = np.argsort(sim)[-3:]
        nn_labels = [labels[k] for k in nn_idx]
        hits = sum(1 for lbl in nn_labels if lbl == labels[i])
        if hits >= 2:
            knn_hits += 1
    purity = knn_hits / float(len(labels))

    return {
        "probe_intra_cos": intra,
        "probe_inter_cos": inter,
        "probe_gap": gap,
        "probe_knn_purity": purity,
    }


def detect_grokking_like(history: List[Dict[str, float]], max_iters: int) -> Dict[str, float | bool]:
    if len(history) < 8:
        return {"grokking_like": False, "val_improvement_after_train_low": 0.0, "delay_iters": 0.0}

    train_losses = np.array([row["train_loss"] for row in history], dtype=np.float64)
    val_losses = np.array([row["val_loss"] for row in history], dtype=np.float64)
    probe_gaps = np.array([row["probe_gap"] for row in history], dtype=np.float64)
    iters = np.array([row["iter"] for row in history], dtype=np.float64)

    if np.isnan(train_losses).any() or np.isnan(val_losses).any():
        return {"grokking_like": False, "val_improvement_after_train_low": 0.0, "delay_iters": 0.0}

    low_threshold = np.percentile(train_losses, 25)
    low_idx = np.where(train_losses <= low_threshold)[0]
    if len(low_idx) == 0:
        return {"grokking_like": False, "val_improvement_after_train_low": 0.0, "delay_iters": 0.0}
    start = int(low_idx[0])
    if start >= len(train_losses) - 2:
        return {"grokking_like": False, "val_improvement_after_train_low": 0.0, "delay_iters": 0.0}

    val_at_start = float(val_losses[start])
    best_after_idx = start + int(np.argmin(val_losses[start:]))
    best_after_val = float(val_losses[best_after_idx])
    delay_iters = float(iters[best_after_idx] - iters[start])
    val_improvement = (val_at_start - best_after_val) / max(1e-8, val_at_start)

    gap_start = float(probe_gaps[start]) if not math.isnan(float(probe_gaps[start])) else 0.0
    gap_end = float(probe_gaps[best_after_idx]) if not math.isnan(float(probe_gaps[best_after_idx])) else gap_start
    gap_gain = gap_end - gap_start

    enough_delay = delay_iters > 0.1 * max(1, max_iters)
    enough_val_drop = val_improvement > 0.15
    growing_semantics = gap_gain > 0.0
    grokking_like = bool(enough_delay and enough_val_drop and growing_semantics)

    return {
        "grokking_like": grokking_like,
        "val_improvement_after_train_low": float(val_improvement),
        "delay_iters": delay_iters,
        "probe_gap_gain": float(gap_gain),
    }


def validate_embedding_size(cfg: TrainConfig) -> Dict[str, str]:
    checks: Dict[str, str] = {}
    checks["n_embd_multiple_of_n_head"] = "ok" if cfg.n_embd % cfg.n_head == 0 else "fail"
    head_dim = cfg.n_embd // cfg.n_head if cfg.n_head > 0 else -1
    checks["head_dim"] = str(head_dim)
    checks["head_dim_range"] = "ok" if 16 <= head_dim <= 128 else "warn"

    emb_params = cfg.vocab_size * cfg.n_embd
    emb_mem_mb = emb_params * 4 / (1024**2)
    checks["token_embedding_params"] = str(emb_params)
    checks["token_embedding_memory_mb_fp32"] = f"{emb_mem_mb:.2f}"
    checks["token_embedding_memory_mb_fp16"] = f"{emb_mem_mb / 2.0:.2f}"

    if emb_mem_mb > 4096:
        checks["embedding_size_comment"] = "very_large"
    elif emb_mem_mb > 1024:
        checks["embedding_size_comment"] = "large"
    else:
        checks["embedding_size_comment"] = "reasonable_for_single_gpu_or_cpu_batches"
    return checks


def write_parameter_explainer(cfg: TrainConfig, validation: Dict[str, str], out_path: Path) -> None:
    reasons = {
        "batch_size": "Controla estabilidad de gradiente vs memoria. 16 es un punto medio.",
        "block_size": "Contexto maximo visible por token. 128 permite dependencias mas largas.",
        "vocab_size": "Debe cubrir el rango de IDs en train.bin (uint16 -> hasta 65535).",
        "n_embd": "Capacidad de representacion por token. 192 es mas expresivo que 32.",
        "n_head": "Subespacios de atencion en paralelo. 6 deja head_dim=32.",
        "n_layer": "Profundidad del Transformer. 4 mejora abstraccion sin costo extremo.",
        "precision": "Precision numerica para entrenamiento: auto intenta bf16/fp16 en CUDA y conserva estabilidad.",
        "lr_max/lr_min": "Warmup + cosine decay para convergencia mas estable.",
        "weight_decay": "Regulariza matrices y reduce sobreajuste.",
        "grad_clip": "Evita exploding gradients.",
        "eval_interval": "Frecuencia para observar train/val y posible grokking.",
        "umap_interval": "Frecuencia de snapshots para ver evolucion geometrica.",
    }

    lines = ["# GPT2 experiment parameter guide", "", "## Why these values", ""]
    for key, value in reasons.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Embedding size validation", ""])
    for key, value in validation.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Raw config", ""])
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), indent=2))
    lines.append("```")
    out_path.write_text("\n".join(lines), encoding="utf-8")


class MetricsLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.train_path = self.run_dir / "train_metrics.csv"
        self.grad_path = self.run_dir / "grad_norms_by_layer.csv"
        self.validation_path = self.run_dir / "config_validation.json"

        with self.train_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iter",
                    "train_loss",
                    "val_loss",
                    "lr",
                    "global_grad_norm",
                    "probe_intra_cos",
                    "probe_inter_cos",
                    "probe_gap",
                    "probe_knn_purity",
                    "umap_method",
                ]
            )

        with self.grad_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "layer_name", "grad_norm", "delta_vs_prev"])

    def write_validation(self, validation: Dict[str, str]) -> None:
        self.validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    def log_train(
        self,
        iteration: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        global_grad_norm: float,
        probe: Dict[str, float],
        umap_method: str,
    ) -> None:
        with self.train_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    iteration,
                    train_loss,
                    val_loss,
                    lr,
                    global_grad_norm,
                    probe["probe_intra_cos"],
                    probe["probe_inter_cos"],
                    probe["probe_gap"],
                    probe["probe_knn_purity"],
                    umap_method,
                ]
            )

    def log_gradients(self, iteration: int, current: Dict[str, float], previous: Dict[str, float]) -> None:
        with self.grad_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for layer_name in sorted(current.keys()):
                grad_norm = current[layer_name]
                delta = grad_norm - previous.get(layer_name, 0.0)
                writer.writerow([iteration, layer_name, grad_norm, delta])


def save_checkpoint(path: Path, model: GPTModel, optimizer: torch.optim.Optimizer, iteration: int, run_dir: Path, cfg: TrainConfig) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "run_dir": str(run_dir),
        "config": asdict(cfg),
    }
    torch.save(payload, path)


def maybe_load_checkpoint(
    path: Path, model: GPTModel, optimizer: torch.optim.Optimizer, resume: bool, device: str
) -> Tuple[int, Path | None]:
    if not resume or not path.exists():
        return 0, None

    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model" in raw:
        model.load_state_dict(raw["model"])
        if "optimizer" in raw:
            optimizer.load_state_dict(raw["optimizer"])
        start_iter = int(raw.get("iteration", 0))
        run_dir = Path(raw["run_dir"]) if "run_dir" in raw else None
        return start_iter, run_dir

    # Backward compatibility: old checkpoint only with state_dict
    model.load_state_dict(raw)
    return 0, None


def build_run_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        return output_root / run_name
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return output_root / f"run_{stamp}"


def train(cfg: TrainConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = select_device(cfg.device)
    amp = resolve_amp_settings(cfg, device)
    print(f"[INFO] device = {device}")
    print(
        f"[INFO] precision = requested:{cfg.precision} "
        f"amp_enabled:{amp.enabled} dtype:{amp.dtype} grad_scaler:{amp.use_grad_scaler}"
    )

    train_ds = BinTokenDataset(cfg.train_bin)
    val_ds = BinTokenDataset(cfg.val_bin)

    model = GPTModel(cfg).to(device)
    optimizer = configure_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp.use_grad_scaler and device == "cuda"))

    checkpoint_path = Path(cfg.checkpoint_path)
    start_iter, restored_run = maybe_load_checkpoint(checkpoint_path, model, optimizer, cfg.resume, device)
    run_dir = restored_run if restored_run is not None else build_run_dir(Path(cfg.output_root), cfg.run_name)
    logger = MetricsLogger(run_dir)

    validation = validate_embedding_size(cfg)
    logger.write_validation(validation)
    write_parameter_explainer(cfg, validation, run_dir / "parameter_guide.md")

    total_params = sum(param.numel() for param in model.parameters())
    print(f"[INFO] params = {total_params / 1e6:.2f}M")
    print(f"[INFO] run_dir = {run_dir}")
    print(f"[INFO] start_iter = {start_iter}")
    print(f"[INFO] embedding_validation = {validation}")

    tokenizer = None
    try:
        tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    except Exception as exc:
        print(f"[WARN] tokenizer could not be loaded ({cfg.tokenizer_path}): {exc}")

    sample_size = min(cfg.umap_sample_size, cfg.vocab_size)
    sample_ids = np.random.default_rng(cfg.seed).choice(cfg.vocab_size, size=sample_size, replace=False)
    prev_grad_norms: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    last_train_loss = float("nan")
    last_global_grad_norm = float("nan")

    model.train()
    timer = time.time()
    for iteration in range(start_iter, cfg.max_iters):
        xb, yb = train_ds.get_batch(cfg.block_size, cfg.batch_size)
        xb = xb.to(device)
        yb = yb.to(device)

        lr = get_lr(iteration, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=amp.device_type,
            dtype=amp.dtype,
            enabled=amp.enabled,
        ):
            _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Loss became None during training.")

        if amp.use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            global_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            global_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimizer.step()

        last_train_loss = float(loss.item())
        last_global_grad_norm = global_grad_norm

        if iteration % cfg.grad_log_interval == 0:
            grad_norms = collect_layer_grad_norms(model)
            logger.log_gradients(iteration, grad_norms, prev_grad_norms)
            prev_grad_norms = grad_norms

        should_eval = (iteration % cfg.eval_interval == 0) or (iteration == cfg.max_iters - 1)
        if should_eval:
            val_loss = evaluate_loss(model, val_ds, cfg, device=device, amp=amp)
            probe_metrics = compute_probe_metrics(model, tokenizer)
            umap_method = "not_run"

            if cfg.umap_interval > 0 and (iteration % cfg.umap_interval == 0):
                umap_method = save_umap_snapshot(
                    model=model,
                    sample_ids=sample_ids,
                    iteration=iteration,
                    out_dir=run_dir / "umap_snapshots",
                    random_state=cfg.umap_random_state,
                )

            logger.log_train(
                iteration=iteration,
                train_loss=last_train_loss,
                val_loss=val_loss,
                lr=lr,
                global_grad_norm=last_global_grad_norm,
                probe=probe_metrics,
                umap_method=umap_method,
            )
            history.append(
                {
                    "iter": float(iteration),
                    "train_loss": last_train_loss,
                    "val_loss": val_loss,
                    "probe_gap": probe_metrics["probe_gap"],
                }
            )

            elapsed = time.time() - timer
            timer = time.time()
            print(
                f"[iter {iteration:06d}] "
                f"train_loss={last_train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"lr={lr:.2e} "
                f"global_grad_norm={last_global_grad_norm:.4f} "
                f"probe_gap={probe_metrics['probe_gap']:.4f} "
                f"umap={umap_method} "
                f"dt={elapsed:.2f}s"
            )

        if iteration > 0 and (iteration % cfg.save_interval == 0):
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                run_dir=run_dir,
                cfg=cfg,
            )
            print(f"[INFO] checkpoint saved at iter={iteration} -> {checkpoint_path}")

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        iteration=max(0, cfg.max_iters - 1),
        run_dir=run_dir,
        cfg=cfg,
    )
    print(f"[INFO] final checkpoint saved -> {checkpoint_path}")

    grok = detect_grokking_like(history, max_iters=cfg.max_iters)
    (run_dir / "grokking_heuristic.json").write_text(json.dumps(grok, indent=2), encoding="utf-8")
    print(f"[INFO] grokking heuristic -> {grok}")

    if tokenizer is not None and cfg.generate_tokens > 0:
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            generated_ids = model.generate(context, cfg.generate_tokens)[0].tolist()
        text = tokenizer.decode(generated_ids)
        (run_dir / "sample_generation.txt").write_text(text, encoding="utf-8")
        print(f"[INFO] sample generation saved -> {run_dir / 'sample_generation.txt'}")

    if cfg.auto_presentation:
        try:
            from presentation_report import generate_presentation

            outputs = generate_presentation(run_dir=run_dir, run_name=run_dir.name)
            print(f"[INFO] presentation generated -> {outputs.get('pdf')}")
            print(f"[INFO] presentation summary -> {outputs.get('markdown')}")
        except Exception as exc:
            print(f"[WARN] could not generate presentation automatically: {exc}")


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Instrumented GPT2-style training with gradient and UMAP diagnostics.")
    parser.add_argument("--max-iters", type=int, default=cfg.max_iters)
    parser.add_argument("--eval-interval", type=int, default=cfg.eval_interval)
    parser.add_argument("--save-interval", type=int, default=cfg.save_interval)
    parser.add_argument("--eval-batches", type=int, default=cfg.eval_batches)
    parser.add_argument("--grad-log-interval", type=int, default=cfg.grad_log_interval)
    parser.add_argument("--umap-interval", type=int, default=cfg.umap_interval)
    parser.add_argument("--umap-sample-size", type=int, default=cfg.umap_sample_size)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--block-size", type=int, default=cfg.block_size)
    parser.add_argument("--n-embd", type=int, default=cfg.n_embd)
    parser.add_argument("--n-head", type=int, default=cfg.n_head)
    parser.add_argument("--n-layer", type=int, default=cfg.n_layer)
    parser.add_argument("--lr-max", type=float, default=cfg.lr_max)
    parser.add_argument("--lr-min", type=float, default=cfg.lr_min)
    parser.add_argument("--warmup-iters", type=int, default=cfg.warmup_iters)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--precision", type=str, default=cfg.precision, help="auto|fp32|fp16|bf16")
    parser.add_argument("--run-name", type=str, default=cfg.run_name)
    parser.add_argument("--checkpoint-path", type=str, default=cfg.checkpoint_path)
    parser.add_argument("--train-bin", type=str, default=cfg.train_bin)
    parser.add_argument("--val-bin", type=str, default=cfg.val_bin)
    parser.add_argument("--tokenizer-path", type=str, default=cfg.tokenizer_path)
    parser.add_argument("--output-root", type=str, default=cfg.output_root)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-presentation", action="store_true")
    args = parser.parse_args()

    cfg.max_iters = args.max_iters
    cfg.eval_interval = args.eval_interval
    cfg.save_interval = args.save_interval
    cfg.eval_batches = args.eval_batches
    cfg.grad_log_interval = args.grad_log_interval
    cfg.umap_interval = args.umap_interval
    cfg.umap_sample_size = args.umap_sample_size
    cfg.batch_size = args.batch_size
    cfg.block_size = args.block_size
    cfg.n_embd = args.n_embd
    cfg.n_head = args.n_head
    cfg.n_layer = args.n_layer
    cfg.lr_max = args.lr_max
    cfg.lr_min = args.lr_min
    cfg.warmup_iters = args.warmup_iters
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.device = args.device
    cfg.precision = args.precision
    cfg.run_name = args.run_name
    cfg.checkpoint_path = args.checkpoint_path
    cfg.train_bin = args.train_bin
    cfg.val_bin = args.val_bin
    cfg.tokenizer_path = args.tokenizer_path
    cfg.output_root = args.output_root
    cfg.resume = not args.no_resume
    cfg.auto_presentation = not args.skip_presentation
    return cfg

def generate_sample(model: GPTModel, tokenizer: Tokenizer, device: str, cfg: TrainConfig, run_dir: Path) -> None:
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        generated_ids = model.generate(context, cfg.generate_tokens)[0].tolist()
    text = tokenizer.decode(generated_ids)
    print(f"[INFO] sample generation saved -> {run_dir / 'sample_generation.txt'}")


if __name__ == "__main__":
    config = parse_args()
    train(config)
