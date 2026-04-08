# GPT2 experiment parameter guide

## Why these values

- `batch_size`: Controla estabilidad de gradiente vs memoria. 16 es un punto medio.
- `block_size`: Contexto maximo visible por token. 128 permite dependencias mas largas.
- `vocab_size`: Debe cubrir el rango de IDs en train.bin (uint16 -> hasta 65535).
- `n_embd`: Capacidad de representacion por token. 192 es mas expresivo que 32.
- `n_head`: Subespacios de atencion en paralelo. 6 deja head_dim=32.
- `n_layer`: Profundidad del Transformer. 4 mejora abstraccion sin costo extremo.
- `lr_max/lr_min`: Warmup + cosine decay para convergencia mas estable.
- `weight_decay`: Regulariza matrices y reduce sobreajuste.
- `grad_clip`: Evita exploding gradients.
- `eval_interval`: Frecuencia para observar train/val y posible grokking.
- `umap_interval`: Frecuencia de snapshots para ver evolucion geometrica.

## Embedding size validation

- `n_embd_multiple_of_n_head`: `ok`
- `head_dim`: `32`
- `head_dim_range`: `ok`
- `token_embedding_params`: `12582912`
- `token_embedding_memory_mb_fp32`: `48.00`
- `token_embedding_memory_mb_fp16`: `24.00`
- `embedding_size_comment`: `reasonable_for_single_gpu_or_cpu_batches`

## Raw config

```json
{
  "batch_size": 16,
  "block_size": 128,
  "vocab_size": 65536,
  "n_embd": 192,
  "n_head": 6,
  "n_layer": 4,
  "max_iters": 25000,
  "eval_interval": 500,
  "eval_batches": 20,
  "save_interval": 5000,
  "grad_log_interval": 20,
  "umap_interval": 2000,
  "umap_sample_size": 2000,
  "umap_random_state": 42,
  "lr_max": 0.0003,
  "lr_min": 3e-05,
  "warmup_iters": 500,
  "weight_decay": 0.1,
  "beta1": 0.9,
  "beta2": 0.95,
  "grad_clip": 1.0,
  "train_bin": "C:\\Repos\\PequeLLM\\train.bin",
  "val_bin": "C:\\Repos\\PequeLLM\\val.bin",
  "checkpoint_path": "C:\\Repos\\PequeLLM\\pequellm_v3_checkpoint.pth",
  "tokenizer_path": "C:\\Repos\\PequeLLM\\tokenizer-culturax-es-hf.json",
  "output_root": "C:\\Repos\\PequeLLM\\Embeddings\\artifacts_gpt2",
  "resume": true,
  "device": "auto",
  "seed": 42,
  "generate_tokens": 100,
  "run_name": "boss-review-v1"
}
```