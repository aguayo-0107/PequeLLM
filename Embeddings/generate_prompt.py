from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = Path(__file__).resolve().parent
if str(EMB_DIR) not in sys.path:
    sys.path.insert(0, str(EMB_DIR))

from emb_gpt2 import GPTModel, TrainConfig, select_device  # noqa: E402


def state_dict_from_checkpoint(raw: object) -> Tuple[Dict[str, torch.Tensor], dict | None]:
    if isinstance(raw, dict) and "model" in raw:
        return raw["model"], raw.get("config") if isinstance(raw.get("config"), dict) else None
    if isinstance(raw, dict):
        return raw, None
    raise ValueError("Unsupported checkpoint format. Expected a state_dict or a dict with key 'model'.")


def infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> TrainConfig:
    cfg = TrainConfig()

    token_emb = state_dict.get("token_embedding_table.weight")
    pos_emb = state_dict.get("position_embedding_table.weight")
    if token_emb is None or pos_emb is None:
        raise ValueError("Checkpoint is missing embedding tables; cannot infer GPT config.")

    cfg.vocab_size = int(token_emb.shape[0])
    cfg.n_embd = int(token_emb.shape[1])
    cfg.block_size = int(pos_emb.shape[0])

    layer_ids = []
    head_ids = []
    for name in state_dict:
        layer_match = re.match(r"blocks\.(\d+)\.", name)
        if layer_match:
            layer_ids.append(int(layer_match.group(1)))
        head_match = re.match(r"blocks\.0\.sa\.heads\.(\d+)\.", name)
        if head_match:
            head_ids.append(int(head_match.group(1)))

    cfg.n_layer = max(layer_ids) + 1 if layer_ids else cfg.n_layer
    cfg.n_head = max(head_ids) + 1 if head_ids else cfg.n_head
    return cfg


def build_config(raw_config: dict | None, state_dict: Dict[str, torch.Tensor]) -> TrainConfig:
    if raw_config is None:
        return infer_config_from_state_dict(state_dict)

    cfg = TrainConfig()
    for key, value in raw_config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_model(checkpoint_path: Path, device: str) -> GPTModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device)
    state_dict, raw_config = state_dict_from_checkpoint(raw)
    cfg = build_config(raw_config, state_dict)
    model = GPTModel(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] checkpoint = {checkpoint_path}")
    print(
        "[INFO] model_config = "
        f"vocab_size:{cfg.vocab_size} block_size:{cfg.block_size} "
        f"n_embd:{cfg.n_embd} n_head:{cfg.n_head} n_layer:{cfg.n_layer}"
    )
    return model


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        cutoff = values[:, [-1]]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_completion(
    model: GPTModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> list[int]:
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        idx = torch.cat((idx, next_token), dim=1)
    return idx[0].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prompt continuations with a PequeLLM checkpoint.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hola, soy un modelo de inteligencia artificial",
        help="Text prompt to continue.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=str(REPO_ROOT / "pequellm_pesado_checkpoint.pth"))
    parser.add_argument("--tokenizer-path", type=str, default=str(REPO_ROOT / "tokenizer-culturax-es-hf.json"))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-file", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        raise ValueError("Prompt produced no token IDs. Try a non-empty prompt.")

    model = load_model(Path(args.checkpoint_path), device=device)
    if max(prompt_ids) >= model.cfg.vocab_size:
        raise ValueError(
            f"Prompt contains token id {max(prompt_ids)} but model vocab_size is {model.cfg.vocab_size}. "
            "Use the tokenizer that matches the checkpoint."
        )

    print(f"[INFO] device = {device}")
    print(f"[INFO] prompt_tokens = {len(prompt_ids)}")
    print(
        f"[INFO] samples = {args.num_samples} "
        f"max_new_tokens = {args.max_new_tokens} temperature = {args.temperature} top_k = {args.top_k}"
    )

    results = []
    for sample_idx in range(1, args.num_samples + 1):
        output_ids = generate_completion(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        text = tokenizer.decode(output_ids)
        results.append({"sample": sample_idx, "text": text})
        print(f"\n--- Muestra {sample_idx} ---")
        print(text)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "checkpoint_path": args.checkpoint_path,
                    "tokenizer_path": args.tokenizer_path,
                    "num_samples": args.num_samples,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "results": results,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n[INFO] saved generations -> {out_path}")


if __name__ == "__main__":
    main()
