from tokenizers import Tokenizer

# 1. Cargar el tokenizador que guardaste
# No funciona con mi_tokenizador_es.json por el formato, utilizar el test_manualTokenizer.py

tokenizer = Tokenizer.from_file("tokenizer-culturax-es-hf.json")

def probar_frase(texto):
    # 2. Codificar
    output = tokenizer.encode(texto)
    
    print(f"\n--- Análisis de Frase ---")
    print(f"Texto original: {texto}")
    print(f"Número de tokens: {len(output.ids)}")
    print(f"IDs: {output.ids}")
    
    # Ver los "pedazos" (subwords) que el tokenizador creó
    # El símbolo 'Ġ' representa un espacio antes de la palabra
    print(f"Tokens: {output.tokens}")
    
    # 3. Decodificar (debe ser idéntico al original)
    decodificado = tokenizer.decode(output.ids)
    print(f"Decodificado: {decodificado}")
    
    # Calcular ratio de compresión (Bytes vs Tokens)
    bytes_originales = len(texto.encode('utf-8'))
    ratio = bytes_originales / len(output.ids)
    print(f"Ratio de compresión: {ratio:.2f} bytes por token")

# --- PRUEBAS ---
# Prueba 1: Una frase normal
#probar_frase("La inteligencia artificial es una herramienta poderosa.")

# Prueba 2: Una palabra compleja o técnica (para ver cómo la divide)
#probar_frase("Electroencefalografista")

# Prueba 3: Algo que no existe (para ver si falla o usa bytes)
#probar_frase("Xyzyth-99 es un alienígena.")

frase = input("Prueba una frase tuya: ")
probar_frase(frase)