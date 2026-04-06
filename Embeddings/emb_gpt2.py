import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time 
import os
from tokenizers import Tokenizer # Importación necesaria para que hable al final

# --- 1. HIPERPARÁMETROS V2.0 (Maratón Nocturno) ---
BATCH_SIZE = 16      
BLOCK_SIZE = 128     # Memoria a corto plazo expandida
VOCAB_SIZE = 65536   
N_EMBD = 192         
N_HEAD = 6           
N_LAYER = 4          # Red profunda de 4 capas
MAX_ITERS = 25000    
EVAL_INTERVAL = 500  
SAVE_INTERVAL = 5000
DATA_PATH = "train.bin" 
CHECKPOINT_PATH = "pequellm_v2_checkpoint.pth" 
TOKENIZER_PATH = "tokenizer-culturax-es-hf.json" # Ruta para que pueda hablar

# --- 2. EXTRACCIÓN DE DATOS ---
def get_batch(data_path, block_size, batch_size):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    return x, y

# --- 3. COMPONENTES DEL TRANSFORMER ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        return F.softmax(wei, dim=-1) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

# --- 4. EL MODELO ---
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
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

    # ¡LA FUNCIÓN DE GENERACIÓN ESTÁ DE VUELTA!
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Recortar el contexto para no exceder la nueva memoria de 128
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# --- 5. BUCLE DE ENTRENAMIENTO ---
if __name__ == "__main__":
    print("="*50)
    print(" INICIANDO ENTRENAMIENTO PEQUELLM V2.0")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Hardware asignado: {device.upper()}")
    
    m = GPTModel().to(device)
    
    # --- ¡NUEVO!: Lógica de Reanudación de Checkpoint ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n[!] Checkpoint encontrado. Restaurando cerebro desde: {CHECKPOINT_PATH}")
        # map_location=device asegura que si lo entrenaste en GPU, se cargue bien en GPU (o CPU)
        m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("[!] ¡Cerebro restaurado con éxito! Continuando aprendizaje...\n")
    else:
        print("\n[!] No se encontró checkpoint previo. Iniciando con un cerebro en blanco...\n")
        
    print(f"Parámetros totales: {sum(p.numel() for p in m.parameters())/1e6:.2f} Millones")
    
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    t0 = time.time() 
    
    for iter in range(MAX_ITERS):
        xb, yb = get_batch(DATA_PATH, BLOCK_SIZE, BATCH_SIZE)
        xb, yb = xb.to(device), yb.to(device)
        
        logits, loss = m(xb, yb)
        
        optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        optimizer.step()
        
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            t1 = time.time()
            dt = t1 - t0
            print(f"Iteración {iter:05d}/{MAX_ITERS} | Loss: {loss.item():.4f} | Tiempo: {dt:.2f}s")
            t0 = time.time() 
            
        if iter > 0 and iter % SAVE_INTERVAL == 0:
            print(f" ---> Guardando Checkpoint en la iteración {iter}...")
            torch.save(m.state_dict(), CHECKPOINT_PATH)
            print(" ---> ¡Guardado exitoso!")
            
    # Guardado Final
    torch.save(m.state_dict(), CHECKPOINT_PATH)
    print("\n" + "="*50)
    print(f" ¡Entrenamiento V2.0 Completado! Pesos finales guardados en {CHECKPOINT_PATH}")
    print("="*50)

    # --- 6. LA GRAN REVELACIÓN (Generación de texto post-entrenamiento) ---
    print("\n--- ¡Despertando al modelo para que hable! ---")
    try:
        # Ponemos el modelo en modo evaluación (buenas prácticas)
        m.eval()
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        
        # Le damos un token vacío para arrancar (un lienzo en blanco)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        print("\nGenerando 100 tokens...\n")
        # Generamos texto y evitamos que calcule gradientes para ahorrar memoria
        with torch.no_grad():
            generated_ids = m.generate(context, max_new_tokens=100)[0].tolist()
            
        texto_generado = tokenizer.decode(generated_ids)
        print("El modelo V2.0 dice:")
        print(f"\n[ {texto_generado} ]\n")
        
    except Exception as e:
        print(f"\nOcurrió un error en la generación final: {e}")