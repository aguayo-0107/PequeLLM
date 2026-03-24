import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tokenizers import Tokenizer # Importamos tu tokenizador original

# --- 1. Hiperparámetros de la Arquitectura ---
BATCH_SIZE = 4       
BLOCK_SIZE = 8       
VOCAB_SIZE = 65536   
N_EMBD = 32          
N_HEAD = 4           
DATA_PATH = "train.bin" 
TOKENIZER_PATH = "tokenizer-culturax-es-hf.json" # Ruta a tu tokenizador

# --- 2. Extracción de Datos ---
def get_batch(data_path, block_size, batch_size):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    return x, y

# --- 3. Componentes del Transformer ---
class Head(nn.Module):
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
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD) 

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 4. El Modelo: GPT Completo ---
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        self.blocks = nn.Sequential(
            Block(N_EMBD, n_head=N_HEAD)
        )
        self.ln_f = nn.LayerNorm(N_EMBD) 
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        
    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(pos) 
        x = tok_emb + pos_emb 
        
        x = self.blocks(x) 
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

    # ¡NUEVA FUNCIÓN!: El motor de generación
    def generate(self, idx, max_new_tokens):
        # idx es un tensor de índices de contexto (Batch, Time)
        for _ in range(max_new_tokens):
            # Recortar el contexto para que nunca exceda el BLOCK_SIZE
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            # Obtener las predicciones del modelo
            logits, _ = self(idx_cond)
            
            # Enfocarse solo en el último paso de tiempo
            logits = logits[:, -1, :] # Se vuelve (Batch, Vocab_Size)
            
            # Aplicar Softmax para obtener probabilidades
            probs = F.softmax(logits, dim=-1)
            
            # Muestrear de la distribución (tirar los dados)
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            
            # Pegar el nuevo token a la secuencia de contexto
            idx = torch.cat((idx, idx_next), dim=1) # (Batch, Time+1)
            
        return idx

# --- 5. Bucle de Entrenamiento e Inferencia (Main) ---
if __name__ == "__main__":
    print("--- Iniciando el Bucle de Entrenamiento ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    m = GPTModel().to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    
    max_iters = 1000 # Lo mantenemos corto para que termine rápido
    eval_interval = 100 
    
    for iter in range(max_iters):
        xb, yb = get_batch(DATA_PATH, BLOCK_SIZE, BATCH_SIZE)
        xb, yb = xb.to(device), yb.to(device)
        
        logits, loss = m(xb, yb)
        
        optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        optimizer.step()
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"Iteración {iter:04d} | Pérdida: {loss.item():.4f}")
            
    print("\n--- ¡Entrenamiento completado! Iniciando Generación de Texto ---")
    
    # 1. Cargar el tokenizador para decodificar los números a texto
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        
        # 2. Crear un contexto inicial (Ej. El token ID de la palabra "El" o solo un cero)
        # Usamos el ID 0 como un "disparador" inicial arbitrario
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        # 3. Pedirle al modelo que genere 50 tokens nuevos
        print("El modelo dice:")
        generated_ids = m.generate(context, max_new_tokens=50)[0].tolist()
        
        # 4. Decodificar e imprimir
        texto_generado = tokenizer.decode(generated_ids)
        print(f"\n[ {texto_generado} ]\n")
        
    except Exception as e:
        print(f"\nError al intentar decodificar (¿está bien la ruta del tokenizador?): {e}")