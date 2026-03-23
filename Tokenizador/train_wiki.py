import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# 1. Limpieza de avisos y configuración de Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_wikipedia_tokenizer():
    # 2. Configuración del modelo BPE
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=50257,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "[UNK]", "<mask>"],
        show_progress=True
    )

    # 3. Iterador con el nuevo dataset y permiso de ejecución
    def batch_iterator():
        print("Conectando con Wikipedia en español (versión moderna)...")
        # Cambiamos a 'wikimedia/wikipedia' y añadimos trust_remote_code=True
        dataset = load_dataset(
            "wikimedia/wikipedia", 
            "20231101.es", # Fecha actualizada
            split="train", 
            streaming=True,
            trust_remote_code=True # <-- ESTO CORRIGE TU ERROR
        )
        
        batch = []
        for i, item in enumerate(dataset):
            batch.append(item["text"])
            if len(batch) == 1000:
                yield batch
                batch = []
            
            # Para 50k tokens, 100k filas es un entrenamiento muy sólido
            if i >= 100000: 
                break
        if batch:
            yield batch

    # 4. Entrenamiento
    print("Iniciando entrenamiento. Esto usará todos tus núcleos de CPU...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # 5. Guardar el resultado
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save("tokenizer-wiki-es-50k.json")
    print("\n¡Éxito! Tokenizador guardado como 'tokenizer-wiki-es-50k.json'")

if __name__ == "__main__":
    train_wikipedia_tokenizer()