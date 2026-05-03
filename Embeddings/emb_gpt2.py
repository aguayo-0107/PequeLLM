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
    "personas":      ["hombre", "mujer", "padre", "madre", "hijo", "hija",
                      "niño", "niña", "abuelo", "abuela", "hermano", "hermana"],
    "verbos_ser":    ["es", "era", "fue", "sido", "sera", "siendo",
                      "esta", "estaba", "estuvo", "estara"],
    "verbos_accion": ["dijo", "hace", "tiene", "puede", "quiere", "sabe",
                      "viene", "va", "da", "pone", "toma", "lleva"],
    "numeros":       ["uno", "dos", "tres", "cuatro", "cinco",
                      "seis", "siete", "ocho", "nueve", "diez", "cien", "mil"],
    "colores":       ["rojo", "azul", "verde", "negro", "blanco",
                      "amarillo", "gris", "morado", "naranja", "rosa"],
    "lugares":       ["ciudad", "pais", "casa", "calle", "plaza",
                      "europa", "america", "mexico", "españa", "mundo"],
    "tiempo":        ["hoy", "ayer", "mañana", "año", "mes", "dia",
                      "hora", "semana", "siglo", "momento", "ahora", "antes"],
    "emociones":     ["amor", "miedo", "alegria", "tristeza", "enojo",
                      "feliz", "triste", "solo", "contento", "furioso", "calma"],
    "naturaleza":    ["agua", "fuego", "tierra", "aire", "sol", "luna",
                      "mar", "rio", "montaña", "bosque", "cielo", "viento"],
    "abstractos":    ["verdad", "vida", "muerte", "dios", "alma",
                      "poder", "bien", "mal", "libertad", "justicia", "paz"],
}


@dataclass
class TrainConfig:
    batch_size: int = 16
    block_size: int = 128
    vocab_size: int = 65536
    n_embd: int = 768 # 192
    n_head: int = 12 #24 # 6
    n_layer: int = 12

    dropout: float = 0.1
    weight_tying: bool = True
    
    max_iters: int = 100000 # 25000
    eval_interval: int = 500 # 500
    eval_batches: int = 50 #20
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
    checkpoint_path: str = str(REPO_ROOT / "pequellm_gpt2small_checkpoint.pth")
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
    def __init__(self, n_embd: int, block_size: int, head_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        k = self.key(x)
        q = self.query(x)
        att = q @ k.transpose(-2, -1) * (channels ** -0.5)
        att = att.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)  # dropout sobre los pesos de atención
        return att @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, block_size: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd=n_embd, block_size=block_size, head_size=head_size, dropout=dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(torch.cat([head(x) for head in self.heads], dim=-1))
        return self.resid_dropout(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd: int, block_size: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd=n_embd, block_size=block_size, n_head=n_head, dropout=dropout)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
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
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[
            Block(cfg.n_embd, cfg.block_size, cfg.n_head, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.lm_head.weight = self.token_embedding_table.weight
            print("[INFO] Weight tying activado: lm_head comparte pesos con token_embedding")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = idx.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = self.emb_dropout(tok_emb + pos_emb)  # dropout en embeddings
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
            idx_cond = idx[:, -self.cfg.block_size:]
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
        x = torch.stack([torch.from_numpy((self.data[i: i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1: i + block_size + 1]).astype(np.int64)) for i in ix])
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

    if device != "cuda":
        return AmpSettings(enabled=False, device_type=device, dtype=torch.float32, use_grad_scaler=False)

    if requested == "fp32":
        return AmpSettings(enabled=False, device_type="cuda", dtype=torch.float32, use_grad_scaler=False)

    if requested == "bf16":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return AmpSettings(enabled=True, device_type="cuda", dtype=torch.bfloat16, use_grad_scaler=False)
        return AmpSettings(enabled=True, device_type="cuda", dtype=torch.float16, use_grad_scaler=True)

    if requested == "fp16":
        return AmpSettings(enabled=True, device_type="cuda", dtype=torch.float16, use_grad_scaler=True)

    # auto
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return AmpSettings(enabled=True, device_type="cuda", dtype=torch.bfloat16, use_grad_scaler=False)
    return AmpSettings(enabled=True, device_type="cuda", dtype=torch.float16, use_grad_scaler=True)


def get_lr(step: int, cfg: TrainConfig, start_iter: int = 0) -> float:
    total_iters = cfg.max_iters
    if total_iters <= 0:
        return cfg.lr_min

    # Warmup relativo al inicio del entrenamiento actual
    relative_step = step - start_iter
    warmup_end = start_iter + cfg.warmup_iters

    if step < warmup_end:
        progress = float(relative_step + 1) / float(max(1, cfg.warmup_iters))
        return cfg.lr_max * progress

    if step >= total_iters:
        return cfg.lr_min

    decay_ratio = (step - warmup_end) / float(max(1, total_iters - warmup_end))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.lr_min + coeff * (cfg.lr_max - cfg.lr_min)


def configure_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    seen_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))

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

# Métricas adicionales para replicar experimento GPT-2
def compute_perplexity(loss: float) -> float:
    """Perplexity = e^loss. Métrica estándar en NLP. Más interpretable que el loss crudo.
    Un modelo que predice al azar sobre 65536 tokens tendría perplexity=65536.
    El GPT-2 original logró ~35 en WebText en inglés."""
    if math.isnan(loss):
        return float("nan")
    return math.exp(min(loss, 20.0))  # cap para evitar overflow con losses muy altos


def compute_bits_per_character(loss: float, chars_per_token: float = 4.5) -> float:
    """Bits por caracter = loss / ln(2) / chars_per_token.
    Mide qué tan bien comprime el modelo el texto.
    GPT-2 logró ~0.86 bpc en inglés. En español esperamos algo más alto.
    chars_per_token=4.5 es una estimación razonable para español con tokenizador BPE de 50k."""
    if math.isnan(loss):
        return float("nan")
    return loss / math.log(2) / chars_per_token


def compute_tokens_per_second(elapsed_s: float, batch_size: int, block_size: int) -> float:
    """Tokens procesados por segundo durante el intervalo de eval.
    Útil para comparar eficiencia entre corridas y detectar throttling de GPU."""
    if elapsed_s <= 0:
        return float("nan")
    return (batch_size * block_size) / elapsed_s


def compute_param_norm(model: nn.Module) -> float:
    """Norma L2 total de todos los parámetros del modelo.
    Si crece sin control puede indicar que el modelo está 'explotando' silenciosamente
    a pesar del grad_clip (que solo controla gradientes, no los pesos mismos)."""
    total_sq = 0.0
    for param in model.parameters():
        total_sq += float(param.detach().float().pow(2).sum().item())
    return math.sqrt(total_sq)


def compute_grad_norm_ratio(current_norm: float, norm_history: List[float]) -> float:
    """Ratio entre el grad_norm actual y el promedio histórico de las últimas N evals.
    Un ratio > 3 indica un spike de gradiente que puede desestabilizar el entrenamiento.
    Útil para detectar problemas que grad_clip no alcanza a suavizar."""
    if not norm_history or math.isnan(current_norm):
        return float("nan")
    avg = float(np.mean([n for n in norm_history if not math.isnan(n)]))
    if avg < 1e-8:
        return float("nan")
    return current_norm / avg


def compute_embedding_stats(model: GPTModel) -> Dict[str, float]:
    """Norma promedio y desviación estándar de todos los vectores de embedding.
    - norm_mean: debe crecer gradualmente conforme el modelo aprende.
      Si se queda plano, los embeddings no se están actualizando.
    - norm_std: si es muy alta, algunos tokens dominan el espacio vectorial
      y otros son ignorados (problema de tokens raros)."""
    emb = model.token_embedding_table.weight.detach().float()
    norms = emb.norm(dim=1)  # norma de cada vector de token
    return {
        "emb_norm_mean": float(norms.mean().item()),
        "emb_norm_std": float(norms.std().item()),
        "emb_norm_min": float(norms.min().item()),
        "emb_norm_max": float(norms.max().item()),
    }


def count_loss_spikes(history: List[Dict[str, float]], threshold: float = 0.05) -> int:
    """Cuenta cuántas veces el val_loss subió más de `threshold` entre dos evals consecutivas.
    Spikes frecuentes indican inestabilidad en el entrenamiento."""
    spikes = 0
    for i in range(1, len(history)):
        prev = history[i - 1]["val_loss"]
        curr = history[i]["val_loss"]
        if not math.isnan(prev) and not math.isnan(curr):
            if (curr - prev) / max(abs(prev), 1e-8) > threshold:
                spikes += 1
    return spikes

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
    except Exception as e:
        print(f"[WARN] UMAP falló ({e}), cayendo a PCA.")
        try:
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=random_state).fit_transform(vectors), "pca_fallback"
        except Exception as e2:
            print(f"[WARN] PCA también falló ({e2}), usando primeras 2 dimensiones.")
            if vectors.shape[1] >= 2:
                return vectors[:, :2], "raw2d_fallback"
            pad = np.zeros((vectors.shape[0], 2), dtype=vectors.dtype)
            pad[:, : vectors.shape[1]] = vectors
            return pad, "raw2d_fallback"


def save_umap_snapshot(
    model: GPTModel,
    sample_ids: np.ndarray,
    iteration: int,
    out_dir: Path,
    random_state: int,
    tokenizer: Tokenizer | None = None,
) -> str:
    """
    Genera un snapshot UMAP de los embeddings.

    Mejoras vs versión original:
    - Usa PROBE_WORDS como anclas etiquetadas y coloreadas por categoría semántica.
    - Rellena el resto con tokens del sample aleatorio pintados en gris (fondo).
    - Guarda columnas 'category' y 'word' en el CSV para análisis posterior.
    - Si el tokenizer no está disponible, cae al comportamiento original.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = model.token_embedding_table.weight.detach().cpu().numpy()

    # --- Construir anclas etiquetadas desde PROBE_WORDS ---
    labeled_ids: List[int] = []
    labeled_categories: List[str] = []
    labeled_words: List[str] = []

    if tokenizer is not None:
        for category, words in PROBE_WORDS.items():
            for word in words:
                ids = tokenizer.encode(word).ids
                if ids:
                    # Usar solo el primer token de la palabra
                    labeled_ids.append(ids[0])
                    labeled_categories.append(category)
                    labeled_words.append(word)

    if not labeled_ids:
        # Sin tokenizer o sin matches: comportamiento original sin etiquetas
        print(f"[WARN] UMAP: no se encontraron tokens de PROBE_WORDS, usando solo sample aleatorio.")

    # --- Combinar: anclas etiquetadas + fondo aleatorio (sin duplicar) ---
    labeled_set = set(labeled_ids)
    background_ids = [int(sid) for sid in sample_ids if int(sid) not in labeled_set]

    all_ids = labeled_ids + background_ids
    all_categories = labeled_categories + ["otros"] * len(background_ids)
    all_words = labeled_words + [""] * len(background_ids)

    sampled = emb[all_ids]
    coords, method = reduce_to_2d(sampled, random_state=random_state)

    # --- Guardar CSV con etiquetas ---
    csv_path = out_dir / f"umap_tokens_iter_{iteration:06d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "x", "y", "method", "category", "word"])
        for tid, cat, word, (x, y) in zip(all_ids, all_categories, all_words, coords.tolist()):
            writer.writerow([tid, x, y, method, cat, word])

    # --- Graficar con colores por categoría ---
    if plt is not None:
        CATEGORY_COLORS = {
            "personas":    "#e74c3c",   # rojo
            "tiempo_vida": "#2ecc71",   # verde
            "verbos":      "#3498db",   # azul
            "otros":       "#bdc3c7",   # gris claro
        }

        coords_arr = np.array(coords) if not isinstance(coords, np.ndarray) else coords
        fig, ax = plt.subplots(figsize=(10, 8))

        # Primero el fondo gris para que no tape las anclas
        bg_mask = [i for i, c in enumerate(all_categories) if c == "otros"]
        if bg_mask:
            bg_coords = coords_arr[bg_mask]
            ax.scatter(bg_coords[:, 0], bg_coords[:, 1],
                       s=6, alpha=0.25, color=CATEGORY_COLORS["otros"], label="otros", zorder=1)

        # Luego las categorías etiquetadas encima
        for category in PROBE_WORDS.keys():
            cat_mask = [i for i, c in enumerate(all_categories) if c == category]
            if not cat_mask:
                continue
            cat_coords = coords_arr[cat_mask]
            color = CATEGORY_COLORS.get(category, "#9b59b6")
            ax.scatter(cat_coords[:, 0], cat_coords[:, 1],
                       s=80, alpha=0.95, color=color, label=category,
                       edgecolors="white", linewidths=0.5, zorder=5)
            # Anotar cada palabra
            for i, mask_i in enumerate(cat_mask):
                ax.annotate(
                    all_words[mask_i],
                    (coords_arr[mask_i, 0], coords_arr[mask_i, 1]),
                    fontsize=7, ha="center", va="bottom",
                    color=color, fontweight="bold",
                )

        ax.set_title(f"Token embedding projection — iter={iteration} ({method})", fontsize=13)
        ax.set_xlabel("axis_1")
        ax.set_ylabel("axis_2")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"umap_tokens_iter_{iteration:06d}.png", dpi=160)
        plt.close(fig)

    n_labeled = len(labeled_ids)
    print(f"[INFO] UMAP snapshot guardado en iter={iteration} método={method} anclas={n_labeled}")
    return method


def compute_probe_metrics(model: GPTModel, tokenizer: Tokenizer | None) -> Dict[str, float]:
    if tokenizer is None:
        return {
            "probe_intra_cos": float("nan"),
            "probe_inter_cos": float("nan"),
            "probe_gap": float("nan"),
            "probe_knn_purity": float("nan"),
        }

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
        return {
            "probe_intra_cos": float("nan"),
            "probe_inter_cos": float("nan"),
            "probe_gap": float("nan"),
            "probe_knn_purity": float("nan"),
        }

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
    emb_mem_mb = emb_params * 4 / (1024 ** 2)
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

        # Solo escribe el header si el archivo no existe todavía
        if not self.train_path.exists():
            with self.train_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iter", "train_loss", "val_loss", "lr", "global_grad_norm",
                    "probe_intra_cos", "probe_inter_cos", "probe_gap", "probe_knn_purity",
                    "umap_method",
                    # Nuevas métricas GPT-2
                    "perplexity", "bits_per_char", "tokens_per_sec",
                    "param_norm", "grad_norm_ratio", "loss_spike_count",
                    "emb_norm_mean", "emb_norm_std", "emb_norm_min", "emb_norm_max",
                ])
        else:
            print(f"[INFO] MetricsLogger: train_metrics.csv existente, continuando en modo append.")

        if not self.grad_path.exists():
            with self.grad_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["iter", "layer_name", "grad_norm", "delta_vs_prev"])
        else:
            print(f"[INFO] MetricsLogger: grad_norms_by_layer.csv existente, continuando en modo append.")

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
        extra: Dict[str, float] | None = None,
    ) -> None:
        # Siempre append — el header ya fue escrito en __init__
        ex = extra or {}
        with self.train_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, train_loss, val_loss, lr, global_grad_norm,
                probe["probe_intra_cos"], probe["probe_inter_cos"],
                probe["probe_gap"], probe["probe_knn_purity"], umap_method,
                ex.get("perplexity", float("nan")),
                ex.get("bits_per_char", float("nan")),
                ex.get("tokens_per_sec", float("nan")),
                ex.get("param_norm", float("nan")),
                ex.get("grad_norm_ratio", float("nan")),
                ex.get("loss_spike_count", float("nan")),
                ex.get("emb_norm_mean", float("nan")),
                ex.get("emb_norm_std", float("nan")),
                ex.get("emb_norm_min", float("nan")),
                ex.get("emb_norm_max", float("nan")),
            ])

    def log_gradients(self, iteration: int, current: Dict[str, float], previous: Dict[str, float]) -> None:
        with self.grad_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for layer_name in sorted(current.keys()):
                grad_norm = current[layer_name]
                delta = grad_norm - previous.get(layer_name, 0.0)
                writer.writerow([iteration, layer_name, grad_norm, delta])


def save_checkpoint(
    path: Path, model: GPTModel, optimizer: torch.optim.Optimizer,
    iteration: int, run_dir: Path, cfg: TrainConfig
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "run_dir": str(run_dir),
        "config": asdict(cfg),
    }
    torch.save(payload, path)


def maybe_load_checkpoint(
    path: Path, model: GPTModel, optimizer: torch.optim.Optimizer,
    resume: bool, device: str
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

    # Backward compatibility: checkpoint antiguo solo con state_dict
    model.load_state_dict(raw)
    return 0, None


def load_history_from_csv(train_path: Path) -> List[Dict[str, float]]:
    """Reconstruye el historial de métricas desde el CSV al resumir.
    Esto permite que detect_grokking_like tenga el historial completo
    aunque el proceso haya sido interrumpido y reiniciado varias veces."""
    if not train_path.exists():
        return []
    history = []
    with train_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                history.append({
                    "iter": float(row["iter"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "probe_gap": float(row.get("probe_gap", "nan")),
                })
            except (KeyError, ValueError):
                continue
    if history:
        print(f"[INFO] Historial recargado desde CSV: {len(history)} entradas (iter {int(history[0]['iter'])}→{int(history[-1]['iter'])})")
    return history


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

    if start_iter > 0:
        cfg.max_iters = start_iter + cfg.max_iters
        print(f"[INFO] Resume detectado: max_iters ajustado a {cfg.max_iters} (start={start_iter})")

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
    grad_norm_history: List[float] = []  # para compute_grad_norm_ratio

    # Recargar historial desde CSV si estamos resumiendo
    history: List[Dict[str, float]] = load_history_from_csv(logger.train_path)

    last_train_loss = float("nan")
    last_global_grad_norm = float("nan")

    model.train()
    timer = time.time()

    for iteration in range(start_iter, cfg.max_iters):
        xb, yb = train_ds.get_batch(cfg.block_size, cfg.batch_size)
        xb = xb.to(device)
        yb = yb.to(device)

        lr = get_lr(iteration, cfg, start_iter=start_iter)
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

            if iteration % cfg.grad_log_interval == 0:
                grad_norms = collect_layer_grad_norms(model)
                logger.log_gradients(iteration, grad_norms, prev_grad_norms)
                prev_grad_norms = grad_norms

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            global_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))

            if iteration % cfg.grad_log_interval == 0:
                grad_norms = collect_layer_grad_norms(model)
                logger.log_gradients(iteration, grad_norms, prev_grad_norms)
                prev_grad_norms = grad_norms

            optimizer.step()

        last_train_loss = float(loss.item())
        last_global_grad_norm = global_grad_norm

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
                    tokenizer=tokenizer,
                )

            # --- Nuevas métricas ---
            perplexity      = compute_perplexity(val_loss)
            bits_per_char   = compute_bits_per_character(val_loss)
            param_norm      = compute_param_norm(model)
            emb_stats       = compute_embedding_stats(model)
            grad_norm_ratio = compute_grad_norm_ratio(last_global_grad_norm, grad_norm_history)
            grad_norm_history.append(last_global_grad_norm)
            if len(grad_norm_history) > 50:        # ventana de 50 evals
                grad_norm_history.pop(0)

            # Actualizar historial ANTES de contar spikes para incluir este punto
            history.append({
                "iter": float(iteration),
                "train_loss": last_train_loss,
                "val_loss": val_loss,
                "probe_gap": probe_metrics["probe_gap"],
            })
            spike_count = count_loss_spikes(history)

            # tokens/sec: cuántos tokens procesó la GPU en el intervalo de eval
            elapsed = time.time() - timer
            tps = compute_tokens_per_second(elapsed, cfg.batch_size, cfg.block_size)
            timer = time.time()

            extra_metrics = {
                "perplexity":      perplexity,
                "bits_per_char":   bits_per_char,
                "tokens_per_sec":  tps,
                "param_norm":      param_norm,
                "grad_norm_ratio": grad_norm_ratio,
                "loss_spike_count": float(spike_count),
                **emb_stats,
            }

            logger.log_train(
                iteration=iteration,
                train_loss=last_train_loss,
                val_loss=val_loss,
                lr=lr,
                global_grad_norm=last_global_grad_norm,
                probe=probe_metrics,
                umap_method=umap_method,
                extra=extra_metrics,
            )

            print(
                f"[iter {iteration:06d}] "
                f"train_loss={last_train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"ppl={perplexity:.1f} "
                f"bpc={bits_per_char:.4f} "
                f"lr={lr:.2e} "
                f"grad_norm={last_global_grad_norm:.4f} "
                f"grad_ratio={grad_norm_ratio:.2f} "
                f"param_norm={param_norm:.1f} "
                f"emb_mean={emb_stats['emb_norm_mean']:.4f} "
                f"emb_std={emb_stats['emb_norm_std']:.4f} "
                f"spikes={spike_count} "
                f"probe_gap={probe_metrics['probe_gap']:.4f} "
                f"tps={tps:.0f} "
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
        gen_path = run_dir / "sample_generation.txt"
        gen_path.write_text(text, encoding="utf-8")
        print(f"[INFO] sample generation saved -> {gen_path}")
        print(f"[INFO] sample: {text[:200]}")

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
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--no-weight-tying", action="store_true")
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
    cfg.dropout = args.dropout
    cfg.weight_tying = not args.no_weight_tying
    cfg.resume = not args.no_resume
    cfg.auto_presentation = not args.skip_presentation
    return cfg


if __name__ == "__main__":
    config = parse_args()
    train(config)
