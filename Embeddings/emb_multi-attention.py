import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# --- 1. Hiperparámetros de la Arquitectura ---
BATCH_SIZE = 4       
BLOCK_SIZE = 8       
VOCAB_SIZE = 65536  
N_EMBD = 32          
N_HEAD = 4           # ¡NUEVO! Cantidad de cabezas de atención trabajando en paralelo
DATA_PATH = "train.bin" 

# --- 2. Extracción de Datos ---
def get_batch(data_path, block_size, batch_size):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    return x, y

# --- 3. Componentes del Transformer ---

class Head(nn.Module):
    """ Una única cabeza de Self-Attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Múltiples cabezas de atención en paralelo """
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Creamos una lista de cabezas
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Proyección final para mezclar los descubrimientos de todas las cabezas
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        # Concatenamos la salida de todas las cabezas en la dimensión de los canales (-1)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ Una capa lineal simple para que los tokens 'reflexionen' sobre lo que aprendieron en la atención """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expandimos la dimensión temporalmente (estándar en GPT)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Comprimimos de vuelta
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Un bloque Transformer completo: Comunicación (Atención) + Reflexión (FeedForward) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # ¡Conexiones Residuales (x + ...) y Layer Normalization (ln)!
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 4. El Modelo: GPT Completo ---
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        # Insertamos nuestro Bloque Transformer
        self.blocks = nn.Sequential(
            Block(N_EMBD, n_head=N_HEAD)
        )
        self.ln_f = nn.LayerNorm(N_EMBD) # LayerNorm final antes del clasificador
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        
    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(pos) 
        x = tok_emb + pos_emb 
        
        x = self.blocks(x) # Pasa por las cabezas de atención y el feed-forward
        x = self.ln_f(x)
        
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
    print("--- Iniciando prueba de Arquitectura GPT Completa ---")
    
    x, y = get_batch(DATA_PATH, BLOCK_SIZE, BATCH_SIZE)
    m = GPTModel()
    
    logits, loss = m(x, y)
    
    print(f"Forma de entrada (x): {x.shape}")
    print(f"Forma de los Logits: {logits.shape}")
    print(f"Pérdida Inicial del Transformer: {loss.item():.4f}")
    print(f"Parámetros totales del modelo: {sum(p.numel() for p in m.parameters())/1e6:.2f} Millones")