from datasets import load_dataset
from tokenizer import MyTokenizer
from huggingface_hub import login
import time

login()

def ejecutar_entrenamiento():
    MODEL_FILE = "mi_tokenizador_es.json"
    VOCAB_GOAL = 50257  # Prueba primero con 1024, luego sube si quieres
    
    # 1. Cargar o inicializar tokenizador
    tokenizer = MyTokenizer()
    tokenizer.load(MODEL_FILE)

    # 2. Cargar CulturaX (Español) con Streaming
    print("Conectando con CulturaX (es)...")
    dataset = load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)
    
    # 3. Tomar una muestra de texto significativa
    # Nota: BPE necesita ver mucho texto. Vamos a tomar las primeras 1000 filas.
    print("Recopilando muestras del dataset...")
    corpus = ""
    for i, item in enumerate(dataset):
        corpus += item['text'] + "\n"
        if i >= 1500: break # Puedes aumentar este número si tienes mucha RAM (og. 1000)

    # 4. Entrenar
    try:
        print("Iniciando fase de entrenamiento. Presiona Ctrl+C para detener y guardar.")
        tokenizer.train(corpus, VOCAB_GOAL, verbose=True)
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    
    # 5. Guardar siempre al final
    tokenizer.save(MODEL_FILE)

    # 6. Prueba rápida
    prueba = "El entrenamiento de modelos de lenguaje es fascinante."
    tokens = tokenizer.encode(prueba)
    print(f"\nPrueba: {prueba}")
    print(f"Tokens: {tokens}")
    print(f"Decodificado: {tokenizer.decode(tokens)}")

if __name__ == "__main__":
    ejecutar_entrenamiento()