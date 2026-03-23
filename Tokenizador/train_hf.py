from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os

# Desactiva el paralelismo de las librerías de Hugging Face para evitar conflictos de sockets en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_professional_tokenizer():
    # 1. Configuración del modelo BPE (Byte Level para evitar errores de caracteres desconocidos)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # El pre-tokenizador de ByteLevel es lo que usa GPT-2/GPT-4
    # Divide por espacios pero mantiene la información de los bytes.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 2. Configurar el Entrenador
    # Aquí definimos el tamaño de 50,257 y los tokens especiales
    trainer = trainers.BpeTrainer(
        vocab_size=50257,
        min_frequency=2, # Ignora combinaciones que aparecen solo una vez
        special_tokens=["<s>", "<pad>", "</s>", "[UNK]", "<mask>"],
        show_progress=True
    )

    # 3. Generador de datos (Streaming de CulturaX)
    def batch_iterator(batch_size=500): # Se puede subir a 10,000 pero no funciona en la red del itam
        print("Conectando con CulturaX (es)...")
        try:
            # Añadimos un trust_remote_code por seguridad si el dataset lo requiere
            dataset = load_dataset(
                "uonlp/CulturaX", 
                "es", 
                split="train", 
                streaming=True,
                trust_remote_code=True
            )
            
            batch = []
            for i, item in enumerate(dataset):
                batch.append(item["text"])
                if len(batch) == batch_size:
                    yield batch
                    batch = []
                
                if i >= 150000: # Subimos a 100k para que valga la pena el entrenamiento
                    break
            if batch:
                yield batch
        except Exception as e:
            print(f"Error durante el streaming: {e}")

    # 4. Iniciar entrenamiento (Esto será MUCHO más rápido)
    print("Iniciando entrenamiento con Rust backend...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # 5. Configurar el Post-procesador y Decoder (para que el texto se vea bien al decodificar)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 6. Guardar el modelo
    tokenizer.save("tokenizer-culturax-es-hf.json")
    print("¡Tokenizador guardado exitosamente!")

    # 7. Prueba rápida
    test_text = "El entrenamiento en Rust es increíblemente rápido."
    encoded = tokenizer.encode(test_text)
    print(f"\nTexto: {test_text}")
    print(f"IDs: {encoded.ids}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Decodificado: {tokenizer.decode(encoded.ids)}")

if __name__ == "__main__":
    train_professional_tokenizer()