from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "Embeddings"
FT_DIR = REPO_ROOT / "FineTuning"
for path in (EMB_DIR, FT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from emb_gpt2 import GPTModel, TrainConfig, select_device  # noqa: E402
from finetune_classifier import GPTForClassification  # noqa: E402


def build_train_config(raw: Dict) -> TrainConfig:
    cfg = TrainConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def encode_text(tokenizer: Tokenizer, text: str, max_length: int, pad_token_id: int, eos_token_id: int | None) -> torch.Tensor:
    token_ids = tokenizer.encode(text).ids
    if eos_token_id is not None:
        token_ids.append(eos_token_id)
    token_ids = token_ids[:max_length]
    if not token_ids:
        token_ids = [pad_token_id]
    if len(token_ids) < max_length:
        token_ids = token_ids + [pad_token_id] * (max_length - len(token_ids))
    return torch.tensor([token_ids], dtype=torch.long)


def load_classifier(checkpoint_path: Path, device: str) -> tuple[GPTForClassification, Dict[str, int], int]:
    raw = torch.load(checkpoint_path, map_location=device)
    base_cfg = build_train_config(raw["base_config"])
    fine_cfg = raw["fine_tune_config"]
    label_to_id = raw["label_to_id"]
    base_model = GPTModel(base_cfg)
    model = GPTForClassification(base_model, num_classes=len(label_to_id))
    model.load_state_dict(raw["model"])
    model.to(device)
    model.eval()
    return model, label_to_id, int(fine_cfg["max_length"])


@torch.no_grad()
def predict_one(
    model: GPTForClassification,
    tokenizer: Tokenizer,
    text: str,
    id_to_label: Dict[int, str],
    max_length: int,
    pad_token_id: int,
    eos_token_id: int | None,
    device: str,
) -> Dict[str, object]:
    xb = encode_text(tokenizer, text, max_length, pad_token_id, eos_token_id).to(device)
    logits, _ = model(xb)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_id = int(probs.argmax().item())
    return {
        "text": text,
        "predicted_label": id_to_label[pred_id],
        "confidence": float(probs[pred_id].item()),
        "probabilities": {id_to_label[i]: float(probs[i].item()) for i in range(len(id_to_label))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Use a PequeLLM classification fine-tuning checkpoint.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-path", default=str(REPO_ROOT / "tokenizer-culturax-es-hf.json"))
    parser.add_argument("--text", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    model, label_to_id, max_length = load_classifier(Path(args.checkpoint_path), device)
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        pad_token_id = tokenizer.token_to_id("</s>")
    if pad_token_id is None:
        pad_token_id = 0
    eos_token_id = tokenizer.token_to_id("</s>")

    results: List[Dict[str, object]] = []
    if args.text:
        results.append(predict_one(model, tokenizer, args.text, id_to_label, max_length, pad_token_id, eos_token_id, device))

    if args.csv:
        with Path(args.csv).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if args.text_column not in (reader.fieldnames or []):
                raise ValueError(f"{args.csv} is missing text column '{args.text_column}'")
            for row in reader:
                results.append(
                    predict_one(
                        model,
                        tokenizer,
                        row[args.text_column],
                        id_to_label,
                        max_length,
                        pad_token_id,
                        eos_token_id,
                        device,
                    )
                )

    if not results:
        raise ValueError("Provide --text or --csv")

    if args.output_csv:
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "predicted_label", "confidence", "probabilities_json"])
            for result in results:
                writer.writerow(
                    [
                        result["text"],
                        result["predicted_label"],
                        result["confidence"],
                        json.dumps(result["probabilities"], ensure_ascii=True),
                    ]
                )

    for result in results:
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
