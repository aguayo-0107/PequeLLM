from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class HFGPT2InstructionConfig:
    model_name: str = "gpt2"
    train_json: str = str(REPO_ROOT / "FineTuning" / "data" / "instruction_train.json")
    val_json: str = str(REPO_ROOT / "FineTuning" / "data" / "instruction_val.json")
    test_json: str = str(REPO_ROOT / "FineTuning" / "data" / "instruction_test.json")
    output_root: str = str(REPO_ROOT / "FineTuning" / "artifacts_instruction_hf")
    run_name: str = ""

    batch_size: int = 1
    max_length: int = 256
    max_epochs: int = 1
    eval_interval: int = 50
    eval_batches: int = 5
    lr: float = 5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    precision: str = "auto"

    freeze_base: bool = False
    mask_prompt_tokens: bool = False
    generate_samples: int = 5
    generate_tokens: int = 80


def select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_settings(device: str, precision: str) -> tuple[bool, str, torch.dtype]:
    requested = precision.lower().strip()
    if device != "cuda" or requested == "fp32":
        return False, device, torch.float32
    if requested == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return True, "cuda", torch.bfloat16
    if requested == "auto" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return True, "cuda", torch.bfloat16
    return True, "cuda", torch.float16


def format_input(entry: Dict[str, str]) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def format_response(entry: Dict[str, str]) -> str:
    return f"\n\n### Response:\n{entry['output']}"


def read_instruction_json(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    for idx, entry in enumerate(data):
        for key in ("instruction", "input", "output"):
            if key not in entry:
                raise ValueError(f"{path} entry {idx} is missing key '{key}'")
    return data


class InstructionDataset(Dataset):
    def __init__(self, entries: Sequence[Dict[str, str]], tokenizer, max_length: int):
        self.entries = list(entries)
        self.items: List[Tuple[List[int], int]] = []
        for entry in self.entries:
            prompt_ids = tokenizer.encode(format_input(entry), add_special_tokens=False)
            full_ids = tokenizer.encode(format_input(entry) + format_response(entry), add_special_tokens=False)
            full_ids.append(tokenizer.eos_token_id)
            full_ids = full_ids[:max_length]
            if len(full_ids) < 2:
                continue
            self.items.append((full_ids, min(len(prompt_ids), len(full_ids))))
        if not self.items:
            raise ValueError("Instruction dataset has no usable examples after tokenization/truncation")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.items[idx]


def collate_batch(batch: Sequence[Tuple[List[int], int]], pad_token_id: int, mask_prompt_tokens: bool):
    max_len = max(len(ids) for ids, _ in batch)
    input_ids = []
    attention_masks = []
    prompt_lengths = []
    for ids, prompt_len in batch:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)
        prompt_lengths.append(prompt_len)

    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_tensor = torch.tensor(attention_masks, dtype=torch.long)
    labels = input_tensor.clone()
    labels = labels.masked_fill(attention_tensor == 0, -100)

    if mask_prompt_tokens:
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, :prompt_len] = -100
    return input_tensor, attention_tensor, labels


def configure_trainable_params(model, freeze_base: bool) -> None:
    if not freeze_base:
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False
    transformer_blocks = model.transformer.h
    for param in transformer_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.transformer.ln_f.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True


def configure_optimizer(model, cfg: HFGPT2InstructionConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.endswith("bias") and "ln" not in name and "wte" not in name and "wpe" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: str, amp_enabled: bool, amp_device: str, amp_dtype: torch.dtype, max_batches: int) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += float(output.loss.item())
        total_batches += 1
        if 0 < max_batches <= total_batches:
            break
    model.train()
    return total_loss / max(1, total_batches)


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "lr"])


def append_metric(path: Path, epoch: int, step: int, train_loss: float, val_loss: float, lr: float) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([epoch, step, train_loss, val_loss, lr])


def build_run_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        return output_root / run_name
    return output_root / f"hf_gpt2_instruction_{time.strftime('%Y%m%d-%H%M%S')}"


@torch.no_grad()
def generate_response(model, tokenizer, entry: Dict[str, str], device: str, max_new_tokens: int) -> str:
    model.eval()
    prompt = format_input(entry) + "\n\n### Response:\n"
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text[len(prompt) :].strip() if text.startswith(prompt) else text


def save_sample_responses(model, tokenizer, entries: Sequence[Dict[str, str]], run_dir: Path, device: str, max_samples: int, max_new_tokens: int) -> None:
    samples = []
    for entry in list(entries)[:max_samples]:
        samples.append(
            {
                "instruction": entry["instruction"],
                "input": entry.get("input", ""),
                "expected_output": entry["output"],
                "model_response": generate_response(model, tokenizer, entry, device, max_new_tokens),
            }
        )
    (run_dir / "sample_responses.json").write_text(json.dumps(samples, ensure_ascii=True, indent=2), encoding="utf-8")


def train_instruction_hf(cfg: HFGPT2InstructionConfig) -> Path:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = select_device(cfg.device)
    amp_enabled, amp_device, amp_dtype = autocast_settings(device, cfg.precision)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    configure_trainable_params(model, cfg.freeze_base)
    model.to(device)

    train_entries = read_instruction_json(Path(cfg.train_json))
    val_entries = read_instruction_json(Path(cfg.val_json))
    test_entries = read_instruction_json(Path(cfg.test_json))
    train_ds = InstructionDataset(train_entries, tokenizer, cfg.max_length)
    val_ds = InstructionDataset(val_entries, tokenizer, cfg.max_length)

    collate_fn = lambda batch: collate_batch(batch, tokenizer.pad_token_id, cfg.mask_prompt_tokens)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = configure_optimizer(model, cfg)
    use_scaler = amp_enabled and amp_dtype == torch.float16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    run_dir = build_run_dir(Path(cfg.output_root), cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    write_metrics_header(metrics_path)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "hf_gpt2_instruction_config": asdict(cfg),
                "device": device,
                "amp_enabled": amp_enabled,
                "amp_dtype": str(amp_dtype),
                "train_examples": len(train_ds),
                "val_examples": len(val_ds),
                "test_examples": len(test_entries),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"[INFO] model_name={cfg.model_name}")
    print(f"[INFO] device={device} precision={cfg.precision} amp={amp_enabled} dtype={amp_dtype}")
    print(f"[INFO] trainable_params={trainable:,} total_params={total:,}")
    print(f"[INFO] examples train={len(train_ds)} val={len(val_ds)} test={len(test_entries)}")
    print(f"[INFO] run_dir={run_dir}")

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            global_step += 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            if global_step % cfg.eval_interval == 0:
                val_loss = evaluate_loss(model, val_loader, device, amp_enabled, amp_device, amp_dtype, cfg.eval_batches)
                append_metric(metrics_path, epoch, global_step, float(loss.item()), val_loss, cfg.lr)
                print(
                    f"[epoch {epoch:02d} step {global_step:05d}] "
                    f"train_loss={float(loss.item()):.4f} val_loss={val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_dir = run_dir / "best_hf_gpt2_instruction"
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)

        val_loss = evaluate_loss(model, val_loader, device, amp_enabled, amp_device, amp_dtype, cfg.eval_batches)
        append_metric(metrics_path, epoch, global_step, float("nan"), val_loss, cfg.lr)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = run_dir / "best_hf_gpt2_instruction"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        print(f"[epoch {epoch:02d} done] val_loss={val_loss:.4f}")

    (run_dir / "final_metrics.json").write_text(json.dumps({"best_val_loss": best_val_loss}, indent=2), encoding="utf-8")
    if cfg.generate_samples > 0:
        save_sample_responses(model, tokenizer, test_entries, run_dir, device, cfg.generate_samples, cfg.generate_tokens)
    print(f"[INFO] best_val_loss={best_val_loss:.4f}")
    return run_dir


def parse_args() -> HFGPT2InstructionConfig:
    cfg = HFGPT2InstructionConfig()
    parser = argparse.ArgumentParser(description="Instruction fine-tune official Hugging Face GPT-2.")
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--train-json", default=cfg.train_json)
    parser.add_argument("--val-json", default=cfg.val_json)
    parser.add_argument("--test-json", default=cfg.test_json)
    parser.add_argument("--output-root", default=cfg.output_root)
    parser.add_argument("--run-name", default=cfg.run_name)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--max-epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--eval-interval", type=int, default=cfg.eval_interval)
    parser.add_argument("--eval-batches", type=int, default=cfg.eval_batches)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--precision", default=cfg.precision)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--mask-prompt-tokens", action="store_true")
    parser.add_argument("--generate-samples", type=int, default=cfg.generate_samples)
    parser.add_argument("--generate-tokens", type=int, default=cfg.generate_tokens)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


if __name__ == "__main__":
    train_instruction_hf(parse_args())
