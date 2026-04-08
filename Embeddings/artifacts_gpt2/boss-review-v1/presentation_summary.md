# PequeLLM Training Presentation Summary

- PDF presentation: `Embeddings\artifacts_gpt2\boss-review-v1\boss-review-v1_presentation.pdf`
- Last iter: `24999`
- Train loss: `6.2340`
- Val loss: `6.0957`
- Global grad norm: `1.8743`
- Probe gap: `0.0302`

## Embedding validation
- `n_embd_multiple_of_n_head`: `ok`
- `head_dim`: `32`
- `head_dim_range`: `ok`
- `token_embedding_params`: `12582912`
- `token_embedding_memory_mb_fp32`: `48.00`
- `token_embedding_memory_mb_fp16`: `24.00`
- `embedding_size_comment`: `reasonable_for_single_gpu_or_cpu_batches`

## Grokking heuristic
- `grokking_like`: `False`
- `val_improvement_after_train_low`: `0.042502462364531854`
- `delay_iters`: `15000.0`
- `probe_gap_gain`: `0.0013580690138041973`