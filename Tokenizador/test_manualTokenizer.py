from tokenizer import MyTokenizer

def probar_tokenizador_manual():
    # 1. Instanciar y cargar
    tokenizer = MyTokenizer()
    archivo_modelo = "mi_tokenizador_es.json" # El nombre que le pusimos al manual
    
    try:
        tokenizer.load(archivo_modelo)
        print(f"Modelo cargado exitosamente desde {archivo_modelo}")
    except Exception as e:
        print(f"Error al cargar: {e}")
        return

    # 2. Definir frases de prueba
    frases = [
        "Hola, esta es una prueba del tokenizador manual.",
        "La inteligencia artificial es fascinante.",
        "Electroencefalografista",
        "Aguayo huele terriblemente mal. 🐈🐈"
    ]

    for texto in frases:
        # 3. Codificar
        ids = tokenizer.encode(texto)
        
        # 4. Decodificar
        decodificado = tokenizer.decode(ids)
        
        print(f"\n--- Resultado ---")
        print(f"Texto: {texto}")
        print(f"IDs ({len(ids)}): {ids}")
        print(f"¿Coincide?: {'✅ Sí' if texto == decodificado else '❌ No'}")
        
        # Mostrar cómo se descompone (ver los bytes de cada ID)
        fragmentos = [tokenizer.vocab[idx].decode('utf-8', errors='replace') for idx in ids]
        print(f"Fragmentos: {fragmentos}")

if __name__ == "__main__":
    probar_tokenizador_manual()