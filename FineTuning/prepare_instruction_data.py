from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
    "main/ch07/01_main-chapter-code/instruction-data.json"
)


def load_or_download(path: Path, url: str) -> List[Dict[str, str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            path.write_text(response.read().decode("utf-8"), encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    for idx, entry in enumerate(data):
        for key in ("instruction", "input", "output"):
            if key not in entry:
                raise ValueError(f"Entry {idx} is missing required key '{key}'")
    return data


def write_splits(data: List[Dict[str, str]], out_dir: Path) -> Dict:
    # Same order-preserving split as chapter 7 in the book.
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    splits = {
        "train": data[:train_portion],
        "test": data[train_portion : train_portion + test_portion],
        "val": data[train_portion + test_portion :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (out_dir / f"instruction_{split}.json").write_text(
            json.dumps(rows, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    summary = {
        "source": "Raschka LLMs-from-scratch chapter 7 instruction-data.json",
        "source_url": DEFAULT_URL,
        "split_strategy": "book chapter 7 order-preserving 85/10/5 split: train/test/val",
        "total_examples": len(data),
        "splits": {
            split: {
                "examples": len(rows),
                "path": str(out_dir / f"instruction_{split}.json"),
            }
            for split, rows in splits.items()
        },
        "fields": ["instruction", "input", "output"],
    }
    (out_dir / "instruction_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Raschka chapter-7 instruction data splits.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-json", default=str(REPO_ROOT / "FineTuning" / "data" / "instruction-data.json"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "FineTuning" / "data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_or_download(Path(args.raw_json), args.url)
    summary = write_splits(data, Path(args.out_dir))
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
