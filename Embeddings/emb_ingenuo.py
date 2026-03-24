import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# --- 1. Hiperparámetros de la Arquitectura ---
# Para esta prueba usaremos números pequeños para que corra rápido en tu laptop
BATCH_SIZE = 4       # Cuántas secuencias procesamos en paralelo
BLOCK_SIZE = 8       # Longitud de contexto (cuántos tokens ve hacia atrás)
VOCAB_SIZE = 50000   # El tamaño de tu tokenizador de CulturaX
N_EMBD = 32          # Dimensión del vector (en GPT-2 real es 768, aquí usamos 32 para probar)
DATA_PATH = "train.bin" # La ruta a tu archivo binario

# --- 2. Función para extraer lotes (Batches) ---
def get_batch(data_path, block_size, batch_size):
    # memmap permite leer el archivo directamente del disco sin cargarlo todo a la RAM
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    # Elegimos índices aleatorios
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    
    # x: Contexto (Entradas)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # y: Objetivos (El siguiente token desplazado una posición)
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    
    return x, y

# --- 3. El Modelo "Ingenuo" ---
class NaiveLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        # Cimientos: Matrices de Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # (Aquí es donde construiremos los bloques Transformer más adelante)
        
        # Cabezal de salida: Proyección de vuelta al tamaño del vocabulario
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        b, t = idx.shape
        # Vector de posiciones: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        # Obtener los vectores y sumarlos
        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channels)
        pos_emb = self.position_embedding_table(pos) # (Time, Channels)
        x = tok_emb + pos_emb 
        
        # Calcular los logits (puntuaciones de predicción)
        logits = self.lm_head(x) # (Batch, Time, Vocab_Size)
        
        # Calcular el error si tenemos etiquetas
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

# --- 4. Prueba de Ejecución (Unit Test) ---
if __name__ == "__main__":
    print("--- Iniciando prueba de tuberías (Pipeline Test) ---")
    
    # 1. Extraer datos
    x, y = get_batch(DATA_PATH, BLOCK_SIZE, BATCH_SIZE)
    print(f"Forma de x (Input): {x.shape} -> (Batch_Size, Block_Size)")
    print(f"Forma de y (Target): {y.shape} -> (Batch_Size, Block_Size)")
    
    # 2. Instanciar el modelo
    m = NaiveLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE)
    
    # 3. Pasar los datos por el modelo (Forward Pass)
    logits, loss = m(x, y)
    
    print(f"\nForma de los Logits: {logits.shape} -> (Batch_Size * Block_Size, Vocab_Size)")
    print(f"Pérdida inicial (Loss): {loss.item():.4f}")