import numpy as np
from tokenizers import Tokenizer

# Rutas estándar dentro de tu contenedor
bin_path = "/workspace/data/train.bin"
tok_path = "/workspace/repo/tokenizer-culturax-es-hf.json"

# 1. Mapear el archivo binario a memoria sin cargarlo todo (eficiencia pura)
data = np.memmap(bin_path, dtype=np.uint16, mode='r')
tokens_totales = len(data)

print(f"\n--- ESTADÍSTICAS ---")
print(f"Archivo: {bin_path}")
print(f"Total de tokens: {tokens_totales:,}")

# 2. Cargar el tokenizador
try:
    tokenizer = Tokenizer.from_file(tok_path)
    
    # 3. Extraer y decodificar una rebanada (slice) del arreglo
    # Cambia los índices si quieres ver otra parte del dataset
    muestra_tokens = data[0:100].tolist() 
    texto_decodificado = tokenizer.decode(muestra_tokens)

    print(f"\n--- MUESTRA DE TEXTO (Primeros 100 tokens) ---")
    print(texto_decodificado)
    print(f"----------------------------------------------\n")
    
except Exception as e:
    print(f"Error cargando el tokenizador: {e}")