"""Verifica que el KV cache produce la MISMA salida que el camino original.

Genera de forma *greedy* (determinista, sin muestreo) desde un mismo prompt
dos veces — una con KV cache y otra sin él — y compara token a token.

Si la arquitectura del cache es correcta, ambas secuencias deben ser idénticas
(salvo, como mucho, diferencias de punto flotante que en greedy casi nunca
cambian el token elegido). También imprime tokens/seg de cada ruta para ver la
aceleración.

Uso (dentro del contenedor ROCm en renna):
    python /workspace/repo/scripts/verify_kv_cache.py \
        --checkpoint-path /workspace/data/pequellm_medium_checkpoint.pth \
        --prompt "Hola, soy un modelo de inteligencia artificial" \
        --max-new-tokens 80

Salida: una línea final "RESULTADO: IDÉNTICO ✓" o "RESULTADO: DIFIEREN ✗"
con detalle de en qué token divergen.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "Embeddings"
if str(EMB_DIR) not in sys.path:
    sys.path.insert(0, str(EMB_DIR))

from emb_gpt2 import select_device  # noqa: E402
from generate_prompt import load_model  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def greedy_no_cache(model, prompt_ids, max_new_tokens, device):
    block_size = model.cfg.block_size
    idx = torch.tensor([prompt_ids[-block_size:]], dtype=torch.long, device=device)
    generated = []
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond, last_token_only=True)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)
        generated.append(next_id)
        if idx.shape[1] >= block_size:
            break
    _sync(device)
    return generated, time.perf_counter() - t0


@torch.no_grad()
def greedy_with_cache(model, prompt_ids, max_new_tokens, device):
    block_size = model.cfg.block_size
    cond = torch.tensor([prompt_ids[-block_size:]], dtype=torch.long, device=device)
    generated = []
    _sync(device)
    t0 = time.perf_counter()
    logits, _, past = model(cond, last_token_only=True, use_cache=True)
    cur_len = cond.shape[1]
    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_id)
        if cur_len >= block_size:
            break
        next_token = torch.tensor([[next_id]], device=device)
        logits, _, past = model(next_token, last_token_only=True, past=past, use_cache=True)
        cur_len += 1
    _sync(device)
    return generated, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica equivalencia del KV cache (greedy).")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-path", default=str(REPO_ROOT / "tokenizer-culturax-es-hf.json"))
    parser.add_argument("--prompt", default="Hola, soy un modelo de inteligencia artificial")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        raise ValueError("El prompt no produjo tokens.")

    model = load_model(Path(args.checkpoint_path), device=device)

    print(f"[INFO] device = {device}")
    print(f"[INFO] prompt_tokens = {len(prompt_ids)} max_new_tokens = {args.max_new_tokens}")

    ids_no_cache, t_no = greedy_no_cache(model, prompt_ids, args.max_new_tokens, device)
    ids_cache, t_yes = greedy_with_cache(model, prompt_ids, args.max_new_tokens, device)

    n = min(len(ids_no_cache), len(ids_cache))
    first_diff = next((i for i in range(n) if ids_no_cache[i] != ids_cache[i]), None)
    same_len = len(ids_no_cache) == len(ids_cache)
    identical = first_diff is None and same_len

    def tps(n_tok, t):
        return (n_tok / t) if t > 0 else float("nan")

    print(f"\n[SIN cache] {len(ids_no_cache)} tokens en {t_no:.3f}s -> {tps(len(ids_no_cache), t_no):.1f} tok/s")
    print(f"[CON cache] {len(ids_cache)} tokens en {t_yes:.3f}s -> {tps(len(ids_cache), t_yes):.1f} tok/s")
    if t_yes > 0:
        print(f"[SPEEDUP]   {t_no / t_yes:.2f}x más rápido con cache")

    print("\n--- Texto SIN cache ---")
    print(tokenizer.decode(ids_no_cache))
    print("\n--- Texto CON cache ---")
    print(tokenizer.decode(ids_cache))

    if identical:
        print("\nRESULTADO: IDÉNTICO ✓  (el KV cache no cambia la salida)")
    else:
        if not same_len:
            print(f"\n[WARN] longitudes distintas: sin={len(ids_no_cache)} con={len(ids_cache)}")
        if first_diff is not None:
            print(f"[WARN] primer token que difiere: índice {first_diff} "
                  f"(sin={ids_no_cache[first_diff]} con={ids_cache[first_diff]})")
        print("\nRESULTADO: DIFIEREN ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
