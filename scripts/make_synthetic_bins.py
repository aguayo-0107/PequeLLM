"""Genera train.bin y val.bin sinteticos para verificar el pipeline sin red.

Los tokens son uint16 uniformemente aleatorios en [0, 65535], que es
exactamente lo que BinTokenDataset (Embeddings/emb_gpt2.py) espera leer
con np.memmap(..., dtype=np.uint16). El loss resultante NO tiene
significado (el modelo no puede aprender ruido uniforme); el unico
proposito es probar que: container OK -> GPU OK -> training loop OK ->
checkpoint OK.

Uso:
    python scripts/make_synthetic_bins.py                       # defaults
    python scripts/make_synthetic_bins.py --out-dir /tmp/foo
    python scripts/make_synthetic_bins.py --train-tokens 1000000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_TRAIN_TOKENS = 4_000_000   # ~8 MB en uint16
DEFAULT_VAL_TOKENS = 500_000       # ~1 MB en uint16
DEFAULT_SEED = 42


def _resolve_default_out_dir() -> Path:
    workspace_data = Path("/workspace/data")
    if workspace_data.is_dir():
        return workspace_data
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directorio destino. Por defecto /workspace/data si existe, si no la raiz del repo.",
    )
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_VAL_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def write_bin(path: Path, n_tokens: int, rng: np.random.Generator) -> int:
    arr = rng.integers(0, 65535, size=n_tokens, dtype=np.uint16, endpoint=False)
    arr.tofile(path)
    return os.path.getsize(path)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else _resolve_default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[SYNTHETIC] AVISO: estos bins son ruido uint16 uniforme. "
        "El loss convergira hacia log(65536) ~ 11.1 y no tiene "
        "significado linguistico. Solo sirven para verificar el pipeline.",
        file=sys.stderr,
    )

    rng = np.random.default_rng(args.seed)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    train_bytes = write_bin(train_path, args.train_tokens, rng)
    val_bytes = write_bin(val_path, args.val_tokens, rng)

    print(f"[SYNTHETIC] {train_path}  ({args.train_tokens:>10} tokens, {train_bytes / 1024**2:6.2f} MB)")
    print(f"[SYNTHETIC] {val_path}    ({args.val_tokens:>10} tokens, {val_bytes / 1024**2:6.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
