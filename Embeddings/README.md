# Embeddings - guia de archivos

Esta carpeta contiene una progresion didactica para construir un mini GPT desde componentes basicos.
Tambien incluye una version mas grande (V2.0) y scripts de comparacion semantica contra un modelo profesional.
Todos los scripts usan PyTorch y leen datos desde `train.bin` con `np.memmap`.

## Requisitos

- Tener `train.bin` en la raiz del repo (`C:\Repos\PequeLLM\train.bin`).
- Tener instaladas dependencias de `requirements.txt` (al menos `torch`, `numpy`, `tokenizers`).
- Ejecutar desde la raiz del proyecto para que la ruta `DATA_PATH = "train.bin"` funcione.
- Para scripts de comparacion: instalar `sentence-transformers`, `scikit-learn` y `matplotlib` si no estan ya en el entorno.

## Orden recomendado de estudio

1. `emb_ingenuo.py`
2. `emb_attention.py`
3. `emb_multi-attention.py`
4. `emb_bucle.py`
5. `emb_bb.py`
6. `emb_gpt2.py`
7. `comparacion_1.py`
8. `comparacion_2.py`

## Explicacion de cada archivo

### `emb_ingenuo.py`

- Que hace:
  - Implementa un modelo de lenguaje muy basico con:
    - embedding de token
    - embedding posicional
    - capa lineal de salida (`lm_head`)
  - No usa atencion todavia.
  - Hace un solo `forward pass` y reporta `loss`.
- Para que sirve:
  - Entender el pipeline minimo `x -> embeddings -> logits -> cross_entropy`.
- Nota importante:
  - Tiene `VOCAB_SIZE = 50000`, pero tus bins actuales llegan hasta `50256`, asi que puede dar `IndexError` en algunos lotes.

### `emb_attention.py`

- Que hace:
  - Agrega una cabeza de self-attention causal (`Head`) sobre los embeddings.
  - Mantiene un modelo simple (`GPTBaseModel`) con una sola cabeza.
  - Ejecuta prueba unica de `forward` y `loss`.
- Para que sirve:
  - Entender como un token "atiende" a tokens previos con mascara triangular causal.
- Nota importante:
  - Igual que el anterior, usa `VOCAB_SIZE = 50000`.

### `emb_multi-attention.py`

- Que hace:
  - Escala a una arquitectura tipo Transformer:
    - `MultiHeadAttention`
    - `FeedForward`
    - `Block` con residuales + `LayerNorm`
  - Arma un `GPTModel` con 1 bloque y prueba un `forward`.
- Para que sirve:
  - Ver el salto desde "una cabeza" a un bloque Transformer mas cercano a GPT.
- Nota importante:
  - Sigue con `VOCAB_SIZE = 50000`.

### `emb_bucle.py`

- Que hace:
  - Usa arquitectura tipo GPT (similar a `emb_multi-attention.py`) pero ya con bucle de entrenamiento real.
  - Detecta dispositivo (`cuda`/`cpu`), crea `optimizer`, corre `max_iters`, e imprime loss periodicamente.
- Para que sirve:
  - Primer script de la carpeta que entrena de punta a punta.
- Punto fuerte:
  - Corrige el vocabulario a `VOCAB_SIZE = 65536` para compatibilidad con datos `uint16`.

### `emb_bb.py`

- Que hace:
  - Parte del modelo de `emb_bucle.py` y agrega `generate()` para inferencia autoregresiva.
  - Al final del entrenamiento intenta decodificar texto con `tokenizer-culturax-es-hf.json`.
- Para que sirve:
  - Cerrar el ciclo completo: entrenamiento + generacion de texto.
- Punto fuerte:
  - Tambien usa `VOCAB_SIZE = 65536`.

### `emb_gpt2.py`

- Que hace:
  - Implementa una version V2.0 del modelo, mas grande que los anteriores:
    - `N_EMBD = 192`
    - `BLOCK_SIZE = 128`
    - `N_LAYER = 4`
    - `N_HEAD = 6`
  - Incluye entrenamiento largo (`MAX_ITERS = 25000`) con guardado de checkpoints.
  - Reanuda automaticamente si existe `pequellm_v2_checkpoint.pth`.
  - Al final genera texto con `generate()`.
- Para que sirve:
  - Pasar de un prototipo didactico a un entrenamiento mas serio y reproducible.
- Archivos relacionados:
  - Checkpoint esperado: `pequellm_v2_checkpoint.pth`
  - Tokenizador esperado: `tokenizer-culturax-es-hf.json`

### `comparacion_1.py`

- Que hace:
  - Carga un checkpoint "undertrained" (`mi_modelo_undertrained.pth`).
  - Extrae embeddings de palabras frecuentes de 3 categorias semanticas.
  - Reduce dimensionalidad con PCA y grafica esos vectores.
  - Compara lado a lado contra embeddings de un modelo profesional (`paraphrase-multilingual-MiniLM-L12-v2`).
- Para que sirve:
  - Visualizar si el espacio semantico de su modelo ya muestra estructura o sigue caotico.

### `comparacion_2.py`

- Que hace:
  - Repite la misma idea de comparacion, pero para la arquitectura V2.0.
  - Carga `pequellm_v2_checkpoint.pth`.
  - Grafica PCA del modelo de ustedes vs el modelo profesional.
- Para que sirve:
  - Medir visualmente el avance semantico entre version inicial y version V2.0.

### `Comp1.png` y `Comp2.png`

- Que son:
  - Imagenes de salida de las comparaciones (graficas PCA).
- Para que sirven:
  - Documentar resultados para reportes/presentaciones sin tener que volver a correr scripts.

## Ejecucion rapida

Desde `C:\Repos\PequeLLM`:

```powershell
python Embeddings/emb_ingenuo.py
python Embeddings/emb_attention.py
python Embeddings/emb_multi-attention.py
python Embeddings/emb_bucle.py
python Embeddings/emb_bb.py
python Embeddings/emb_gpt2.py
python Embeddings/comparacion_1.py
python Embeddings/comparacion_2.py
```

## Observaciones tecnicas

- Todos los scripts comparten la misma funcion `get_batch` basada en `np.memmap`, lo cual es eficiente en RAM.
- Para consistencia con tus datos actuales, conviene unificar todos a `VOCAB_SIZE = 65536` o derivarlo automaticamente del tokenizador/dataset.
- Estos scripts son excelentes para aprendizaje incremental: cada archivo agrega exactamente una idea nueva.
- Las comparaciones (`comparacion_1.py` y `comparacion_2.py`) dependen de checkpoints ya entrenados; sin esos archivos solo mostraran error de carga.
