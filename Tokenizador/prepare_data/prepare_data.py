import os
from pathlib import Path
import sys
import numpy as np
from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from security_utils import load_dataset_secure, validate_uint16_ids

# 1. Configuración
TOKENIZER_PATH = str(Path(__file__).resolve().parents[2] / "tokenizer-culturax-es-hf.json" # Tu tokenizador de 50k
MAX_TOKENS = 250_000_000  # ~500MB en disco (2 bytes por token)
TRAIN_BIN_PATH = "train.bin"
VAL_BIN_PATH = "val.bin"

def prepare():
    # Cargar tokenizador
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    print("Conectando con CulturaX para preparación binaria...")
    dataset = load_dataset_secure("uonlp/CulturaX", "es", split="train", streaming=True)
    
    # Listas temporales para acumular tokens
    all_tokens = []
    total_tokens = 0
    
    print(f"Iniciando tokenización. Meta: {MAX_TOKENS} tokens.")
    
    eos_id = tokenizer.token_to_id("</s>")
    if eos_id is None:
        raise ValueError("El tokenizador no contiene </s>; no se puede preparar el corpus de entrenamiento.")

    try:
        for i, item in enumerate(dataset):
            # Tokenizar el texto
            text = item["text"]
            ids = tokenizer.encode(text).ids
            
            # Añadir el token de final de secuencia (opcional pero recomendado)
            ids.append(eos_id)
            validate_uint16_ids(ids, source_name=f"fila {i}")
            
            all_tokens.extend(ids)
            total_tokens += len(ids)
            
            if i % 1000 == 0:
                print(f"Procesadas {i} filas... Tokens acumulados: {total_tokens:,}")
            
            if total_tokens >= MAX_TOKENS:
                break
                
    except KeyboardInterrupt:
        print("\nProceso interrumpido. Guardando lo que llevamos...")

    # 2. Convertir a Numpy (uint16 es clave para ahorrar espacio)
    print("Convirtiendo a formato binario...")
    arr = np.array(all_tokens, dtype=np.uint16)
    
    # 3. Dividir en Entrenamiento (95%) y Validación (5%)
    n = len(arr)
    train_data = arr[:int(n*0.95)]
    val_data = arr[int(n*0.95):]
    
    # 4. Guardar en disco
    train_data.tofile(TRAIN_BIN_PATH)
    val_data.tofile(VAL_BIN_PATH)
    
    print(f"\n--- ¡Preparación Completada! ---")
    print(f"Archivo de entrenamiento: {TRAIN_BIN_PATH} ({os.path.getsize(TRAIN_BIN_PATH) / 1024**2:.2f} MB)")
    print(f"Archivo de validación: {VAL_BIN_PATH} ({os.path.getsize(VAL_BIN_PATH) / 1024**2:.2f} MB)")
    print(f"Total de tokens procesados: {len(arr):,}")

if __name__ == "__main__":
    prepare()
