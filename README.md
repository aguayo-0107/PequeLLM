# PequeLLM

Proyecto **educativo** para construir, entrenar y hacer fine-tuning de un modelo de lenguaje pequeño tipo GPT **desde cero**, en español. Está diseñado para correr en el homelab `renna` (GPU AMD Strix Halo / Radeon 8060S, 96 GB de memoria unificada) sobre un contenedor ROCm/PyTorch.

En `renna` logramos entrenar dos modelos: **GPT-2 Small** ([Embeddings/emb_gpt2.py](Embeddings/emb_gpt2.py)) y **GPT-2 Medium** ([Embeddings/Emb_gptMed.py](Embeddings/Emb_gptMed.py)). El fine-tuning no se alcanzó a hacer sobre nuestro modelo; el código queda como referencia en [FineTuning/](FineTuning/).

## Arquitectura de los modelos

Ambos modelos comparten exactamente la misma arquitectura tipo GPT — solo cambian las dimensiones:

- **Tipo**: Transformer *decoder-only* (estilo GPT-2), *pre-norm*.
- **Atención**: multi-head causal, `head_dim = 64` en ambos (igual que el GPT-2 original).
- **FeedForward**: `Linear(n_embd → 4·n_embd) → GELU → Linear(4·n_embd → n_embd)` (expansión 4×).
- **Bloque Transformer**: conexiones residuales alrededor de atención y FFN, con 2 capas `LayerNorm`.
- **Embeddings**: tabla de tokens + tabla de posiciones (aprendida), con `dropout`.
- **`weight_tying = True`**: la `lm_head` comparte pesos con la tabla de embeddings de tokens.
- **`vocab_size = 65536`** (límite uint16) y **`dropout = 0.1`** en ambos.

### Comparación de specs

| Parámetro | **GPT-2 Small** | **GPT-2 Medium** |
|---|---|---|
| Archivo | [Embeddings/emb_gpt2.py](Embeddings/emb_gpt2.py) | [Embeddings/Emb_gptMed.py](Embeddings/Emb_gptMed.py) |
| `n_embd` (dim. embedding) | **768** | **1024** |
| `n_head` (cabezas de atención) | **12** | **16** |
| `head_dim` | 64 | 64 |
| `n_layer` (bloques Transformer) | **12** | **24** |
| `block_size` (contexto) | **128** | **256** |
| `vocab_size` | 65536 | 65536 |
| `batch_size` | 16 | 8 |
| **Parámetros totales (aprox.)** | **~135.5 M** | **~369.6 M** |
| — Transformer (sin embeddings) | ~85 M | ~302 M |
| — embedding de tokens (compartido vía *weight tying*) | 50.3 M | 67.1 M |

### Hiperparámetros de entrenamiento

| Config | GPT-2 Small | GPT-2 Medium |
|---|---|---|
| Optimizador | AdamW (β₁=0.9, β₂=0.95) | AdamW (β₁=0.9, β₂=0.95) |
| `lr_max` → `lr_min` | 3e-4 → 3e-5 | 2e-4 → 2e-5 (reducido por profundidad) |
| Schedule de LR | warmup + cosine decay | warmup + cosine decay |
| `warmup_iters` | 500 | 1000 (más warmup por ser más profundo) |
| `weight_decay` | 0.1 (solo en matrices ≥2D) | 0.1 |
| `grad_clip` | 1.0 | 1.0 |
| Criterio de parada | `max_iters = 100000` (límite duro) | `max_iters = -1` → **hasta convergencia** con early stopping (`patience=30`, `min_delta=0.001`) |
| `lr_horizon` | = `max_iters` | 600000 (decay largo y suave) |
| Pérdida | Cross-entropy + InfoNCE contrastiva opcional (`contrastive_weight=0.005`) | Igual |
| Checkpoint | `pequellm_gpt2small_checkpoint.pth` | `pequellm_gpt2medium_checkpoint.pth` |

### Diferencias notables del Medium

Además de ser más grande, el Medium añade refinamientos de estabilidad que el Small no incluye:

- **Inicialización escalada de los residuales** por `1/√(2·n_layer)` ([Emb_gptMed.py:288](Embeddings/Emb_gptMed.py#L288)) — clave para entrenar 24 capas sin que exploten los gradientes.
- **Early stopping** real por convergencia, en lugar de un número fijo de iteraciones.
- **Aviso de grad-norm** (`grad_norm_warn_threshold=5.0`) para detectar inestabilidad durante el entrenamiento.

> **Nota:** estos no son los GPT-2 "oficiales" de OpenAI (124 M / 350 M). Los nuestros pesan más (~135 M / ~370 M) porque el **vocabulario es de 65536 tokens** (vs. 50257 del GPT-2 original), lo que infla la tabla de embeddings. La parte Transformer pura sí coincide con las proporciones Small/Medium de OpenAI.
