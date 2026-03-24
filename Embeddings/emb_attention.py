import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# --- 1. Hiperparámetros de la Arquitectura ---
BATCH_SIZE = 4       # Secuencias procesadas en paralelo
BLOCK_SIZE = 8       # Longitud de contexto máximo
VOCAB_SIZE = 50000   # Tamaño de tu tokenizador
N_EMBD = 32          # Dimensión del vector de embedding
DATA_PATH = "train.bin" # Ruta a tu corpus preparado

# --- 2. Extracción de Datos (Pipeline) ---
def get_batch(data_path, block_size, batch_size):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    
    return x, y

# --- 3. El Motor: Cabeza de Self-Attention ---
class Head(nn.Module):
    """ Una única cabeza de Self-Attention con enmascaramiento causal """
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Máscara triangular para evitar que el modelo "vea el futuro"
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Producto escalar escalado para calcular las afinidades
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Ponderación de los valores
        v = self.value(x)
        out = wei @ v
        return out

# --- 4. El Modelo: GPT Base ---
class GPTBaseModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        # Cimientos: Embeddings de token y de posición
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Cerebro: Una cabeza de atención (por ahora)
        self.sa_head = Head(head_size=n_embd, n_embd=n_embd, block_size=block_size)
        
        # Salida: Proyección a los logits del vocabulario
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        # Fase 1: Creación del embedding rico
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(pos) # (T, C)
        x = tok_emb + pos_emb 
        
        # Fase 2: Comunicación entre tokens
        x = self.sa_head(x) 
        
        # Fase 3: Predicción
        logits = self.lm_head(x) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

# --- 5. Prueba de Ejecución (Main) ---
if __name__ == "__main__":
    print("--- Iniciando prueba con Self-Attention ---")
    
    # 1. Cargar tensores
    x, y = get_batch(DATA_PATH, BLOCK_SIZE, BATCH_SIZE)
    print(f"Forma de entrada (x): {x.shape}")
    
    # 2. Instanciar modelo
    m = GPTBaseModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE)
    
    # 3. Forward pass
    logits, loss = m(x, y)
    
    print(f"Forma de los Logits: {logits.shape}")
    print(f"Pérdida (Loss) con Atención: {loss.item():.4f}")