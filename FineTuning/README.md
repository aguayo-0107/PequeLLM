# Fine-tuning de PequeLLM

Esta carpeta inicia la etapa de fine-tuning siguiendo el capitulo 6 de
`Build a Large Language Model (From Scratch)` de Sebastian Raschka.

El objetivo de esta primera fase no es todavia crear un asistente conversacional.
Primero hacemos fine-tuning supervisado para clasificacion: dado un texto, el modelo
debe predecir una etiqueta.

## Relacion con el libro

El capitulo 6 propone este flujo:

1. Preparar un dataset supervisado con columnas de texto y etiqueta.
2. Tokenizar cada texto.
3. Convertir los ejemplos a tensores con longitud fija.
4. Cargar un GPT preentrenado.
5. Reemplazar la cabeza generativa por una cabeza de clasificacion.
6. Congelar la mayor parte del modelo base.
7. Entrenar la cabeza nueva y, opcionalmente, el ultimo bloque Transformer.
8. Medir loss y accuracy en validation/test.

Eso es exactamente lo que implementa:

```text
FineTuning/finetune_classifier.py
```

## Idea central

Durante pretraining, PequeLLM aprende a predecir el siguiente token:

```text
tokens -> GPT -> logits sobre vocabulario
```

En clasificacion, reutilizamos el mismo GPT como extractor de representaciones:

```text
tokens -> GPT -> vector final -> classification_head -> logits sobre etiquetas
```

La diferencia clave es la cabeza final:

- Pretraining usa `lm_head`, que predice uno de `vocab_size` tokens.
- Fine-tuning usa `classification_head`, que predice una de `num_classes` etiquetas.

## Formato del dataset

El script espera archivos CSV con al menos dos columnas:

```csv
text,label
"este mensaje es normal","ham"
"gana dinero rapido aqui","spam"
```

Por default busca:

```text
FineTuning/data/classification_demo_train.csv
FineTuning/data/classification_demo_val.csv
FineTuning/data/classification_demo_test.csv
```

Puedes usar otros nombres con:

```powershell
python FineTuning/finetune_classifier.py `
  --train-csv ruta/train.csv `
  --val-csv ruta/val.csv `
  --test-csv ruta/test.csv
```

Si tus columnas se llaman distinto:

```powershell
python FineTuning/finetune_classifier.py `
  --text-column mensaje `
  --label-column categoria
```

## Prueba rapida con datos demo

Para generar un dataset pequeno de prueba y validar que el pipeline funciona:

```powershell
python FineTuning/finetune_classifier.py `
  --create-demo-data `
  --base-checkpoint-path FineTuning/data/no_checkpoint_for_demo.pth `
  --n-embd 32 `
  --n-head 4 `
  --n-layer 1 `
  --block-size 32 `
  --max-length 32 `
  --batch-size 2 `
  --max-epochs 1 `
  --eval-interval 1
```

Esta prueba no mide calidad real. Solo verifica:

- carga del tokenizador;
- construccion del dataset;
- carga del modelo base si existe;
- reemplazo de cabeza;
- forward/backward;
- evaluacion;
- guardado de outputs.

Usamos un modelo mini aleatorio en esta prueba para evitar cargar un checkpoint pesado
solo para validar la logica del codigo.

## Uso con el checkpoint real

Cuando tu companera termine o tenga un checkpoint estable del pretraining:

```powershell
python FineTuning/finetune_classifier.py `
  --base-checkpoint-path pequellm_pesado_checkpoint.pth `
  --train-csv FineTuning/data/mi_train.csv `
  --val-csv FineTuning/data/mi_val.csv `
  --test-csv FineTuning/data/mi_test.csv `
  --run-name spam_v1 `
  --precision auto
```

Si el checkpoint es antiguo y no guarda su configuracion interna, debes pasar la misma
arquitectura con la que fue entrenado:

```powershell
python FineTuning/finetune_classifier.py `
  --base-checkpoint-path pequellm_v2_checkpoint.pth `
  --n-embd 192 `
  --n-head 6 `
  --n-layer 4 `
  --block-size 128
```

En `renna`, dentro del contenedor, normalmente el checkpoint vive en el volumen de datos.
Un ejemplo de ruta podria ser:

```bash
python FineTuning/finetune_classifier.py \
  --base-checkpoint-path /workspace/data/pequellm_pesado_checkpoint.pth \
  --train-csv FineTuning/data/mi_train.csv \
  --val-csv FineTuning/data/mi_val.csv \
  --test-csv FineTuning/data/mi_test.csv \
  --run-name spam_v1 \
  --precision auto
```

## Parametros importantes

`--max-length`

Longitud maxima de tokens por texto. Debe ser menor o igual que el `block_size` del
modelo base. Si el modelo se entreno con `block_size=128`, entonces `max_length=128`
es el limite natural.

`--batch-size`

Numero de ejemplos por paso de entrenamiento. Si falta memoria, bajarlo.

`--lr`

Learning rate del fine-tuning. El default `5e-5` es pequeno porque partimos de un
modelo preentrenado y no queremos destruir sus pesos.

`--train-full-model`

Entrena todos los pesos del GPT base. Es mas costoso y tiene mas riesgo de overfitting.
Por default no se usa.

`--freeze-all-base`

Congela todo el GPT base y entrena solo la cabeza de clasificacion. Es la opcion mas
barata, pero puede quedarse corta si el dataset necesita adaptar representaciones.

## Que capas se entrenan por default

Por default el script:

1. Congela el modelo base.
2. Descongela el ultimo bloque Transformer.
3. Descongela la normalizacion final `ln_f`.
4. Entrena siempre la nueva `classification_head`.

Esto sigue la idea del libro: aprovechar lo aprendido en pretraining, pero permitir
una adaptacion pequena para la tarea supervisada.

## Archivos de salida

Cada corrida crea una carpeta:

```text
FineTuning/artifacts_classifier/<run_name>/
```

Dentro se generan:

```text
config.json
label_to_id.json
metrics.csv
best_classifier_checkpoint.pth
test_metrics.json
test_predictions.csv
```

`config.json`

Guarda la configuracion del fine-tuning, la configuracion del modelo base y si se cargo
un checkpoint real.

`label_to_id.json`

Mapea etiquetas de texto a enteros. Por ejemplo:

```json
{
  "ham": 0,
  "spam": 1
}
```

`metrics.csv`

Registra loss de entrenamiento, loss de validation y accuracy de validation.

`best_classifier_checkpoint.pth`

Guarda el mejor modelo segun validation accuracy.

`test_metrics.json`

Resume la evaluacion final sobre test.

`test_predictions.csv`

Permite inspeccionar ejemplo por ejemplo:

```text
texto, etiqueta real, etiqueta predicha, confianza
```

## Usar el clasificador entrenado

Despues de entrenar, puedes clasificar un texto nuevo con:

```powershell
python FineTuning/predict_classifier.py `
  --checkpoint-path FineTuning/artifacts_classifier/spam_v1/best_classifier_checkpoint.pth `
  --text "reclama tu premio ahora"
```

Tambien puedes clasificar un CSV completo:

```powershell
python FineTuning/predict_classifier.py `
  --checkpoint-path FineTuning/artifacts_classifier/spam_v1/best_classifier_checkpoint.pth `
  --csv FineTuning/data/mensajes_nuevos.csv `
  --text-column text `
  --output-csv FineTuning/artifacts_classifier/spam_v1/predicciones_nuevas.csv
```

El script imprime JSON con:

```json
{
  "text": "reclama tu premio ahora",
  "predicted_label": "spam",
  "confidence": 0.91,
  "probabilities": {
    "ham": 0.09,
    "spam": 0.91
  }
}
```

## Explicacion del codigo

### `FineTuneConfig`

Agrupa todos los parametros del experimento: rutas, batch size, learning rate,
precision, dispositivo y opciones de congelamiento.

### `ClassificationDataset`

Convierte cada fila del CSV en:

```text
input_ids: tensor de tokens con longitud fija
label_id: entero con la clase correcta
```

Pasos internos:

1. Tokeniza el texto.
2. Agrega `</s>` si el tokenizador lo tiene.
3. Recorta a `max_length`.
4. Rellena con `<pad>` o `</s>`.

### `GPTForClassification`

Envuelve el `GPTModel` de `Embeddings/emb_gpt2.py`.

En vez de usar `lm_head`, calcula los estados internos del GPT y usa el vector de la
ultima posicion como resumen de la secuencia. Ese vector pasa por:

```python
nn.Linear(n_embd, num_classes)
```

### `load_base_model`

Carga el checkpoint del pretraining. Si el checkpoint contiene la configuracion original
del modelo, la usa para reconstruir exactamente la misma arquitectura.

Esto es importante porque no podemos cargar pesos de un modelo `n_embd=192` dentro de
un modelo `n_embd=768`, ni cambiar el numero de capas/cabezas sin romper shapes.

### `configure_trainable_layers`

Controla que se entrena:

- con default: ultimo bloque + `ln_f` + cabeza;
- con `--freeze-all-base`: solo cabeza;
- con `--train-full-model`: todo el modelo.

### `evaluate`

Calcula:

- `loss`: que tan equivocadas estan las probabilidades;
- `accuracy`: proporcion de ejemplos clasificados correctamente.

### `predict_classifier.py`

Carga `best_classifier_checkpoint.pth`, reconstruye la arquitectura, tokeniza nuevos
textos y devuelve la etiqueta mas probable con su confianza.

## Recomendaciones para el proyecto

1. Usar primero un dataset pequeno para validar que todo corre.
2. Despues usar un checkpoint base estable del pretraining.
3. Mantener `max_length` igual o menor al `block_size` del pretraining.
4. Empezar con el default congelado antes de entrenar todo el modelo.
5. Comparar `--freeze-all-base`, default y `--train-full-model` con el mismo dataset.
6. No versionar checkpoints ni artifacts pesados.

## Siguiente etapa

Despues de clasificacion, el siguiente paso del libro es instruction fine-tuning:

```text
instruction + input -> response
```

Eso corresponde al capitulo 7 y debe implementarse aparte, cuando tengamos un modelo base
o un checkpoint suficientemente bueno.
