import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# --- 1. HIPERPARÁMETROS V2.0 (Deben coincidir EXACTAMENTE con tu entrenamiento) ---
VOCAB_SIZE = 65536   
N_EMBD = 192         # Cerebro más ancho
BLOCK_SIZE = 128     # Memoria expandida
N_HEAD = 6           # 6 cabezas
N_LAYER = 4          # 4 capas profundas
TOKENIZER_PATH = "tokenizer-culturax-es-hf.json"
PESOS_PATH = "pequellm_v2_checkpoint.pth" # Tu nuevo checkpoint
MODELO_PRO_NOMBRE = 'paraphrase-multilingual-MiniLM-L12-v2'

# Palabras de altísima frecuencia en español narrativo
palabras_prueba = {
    "Personas": ["hombre", "mujer", "padre", "madre", "niño", "niña"],
    "Tiempo_y_Vida": ["mundo", "vida", "tiempo", "año", "día", "noche"],
    "Verbos": ["dijo", "fue", "era", "tiene", "hace", "está"]
}
colores_categoria = {"Personas": "blue", "Tiempo_y_Vida": "green", "Verbos": "red"}

# --- 2. ARQUITECTURA V2.0 (Los planos para cargar los 27M de parámetros) ---
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
    def forward(self, x): return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
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

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD) 
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        
# --- 3. EXTRACCIÓN Y PCA ---
def obtener_vectores_peque(modelo_instancia, tokenizer, palabras_dict):
    print("Mapeando palabras a los nuevos embeddings de 192D...")
    todas_palabras = []
    vectores = []
    colores = []
    
    modelo_instancia.to('cpu')
    embedding_table = modelo_instancia.token_embedding_table.weight.data
    
    for categoria, lista in palabras_dict.items():
        for palabra in lista:
            ids = tokenizer.encode(palabra).ids
            vectores_tokens = embedding_table[ids]
            # Promediamos los tokens de la palabra
            word_vector = vectores_tokens.mean(dim=0).numpy()
            
            todas_palabras.append(palabra)
            vectores.append(word_vector)
            colores.append(colores_categoria[categoria])
            
    return np.array(vectores), todas_palabras, colores

# --- 4. EJECUCIÓN (Main) ---
if __name__ == "__main__":
    print("--- Iniciando Duelo Semántico V2.0 ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    leyenda_patches = [mpatches.Patch(color=color, label=cat) for cat, color in colores_categoria.items()]

    # --- SUBPLOT 1: TU MODELO V2.0 ---
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        m = GPTModel()
        m.load_state_dict(torch.load(PESOS_PATH, map_location='cpu'))
        print(f"Pesos cargados exitosamente desde {PESOS_PATH}")
        
        vectores_peque, nombres_peque, colores_peque = obtener_vectores_peque(m, tokenizer, palabras_prueba)
        
        # PCA de 192D a 2D
        pca_peque = PCA(n_components=2)
        vectores_2d_peque = pca_peque.fit_transform(vectores_peque)
        
        ax1.scatter(vectores_2d_peque[:, 0], vectores_2d_peque[:, 1], c=colores_peque, s=100, alpha=0.6)
        for i, txt in enumerate(nombres_peque):
            ax1.annotate(txt, (vectores_2d_peque[i, 0] + 0.05, vectores_2d_peque[i, 1] + 0.05), fontsize=8)
        
        ax1.set_title("A. Nuestro Modelo V2.0 (25k Iters - N_EMBD=192)\nBuscando el Orden Semántico", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(handles=leyenda_patches)
        
    except FileNotFoundError:
        ax1.text(0.5, 0.5, f"Error:\nArchivo '{PESOS_PATH}' no encontrado.", ha='center', va='center')
        ax1.set_title("A. Error en tu Modelo")

    # --- SUBPLOT 2: MODELO PROFESIONAL ---
    print("\nCargando modelo profesional de Microsoft...")
    modelo_pro = SentenceTransformer(MODELO_PRO_NOMBRE)
    
    todas_palabras_pro = []
    colores_pro = []
    for categoria, lista in palabras_prueba.items():
        todas_palabras_pro.extend(lista)
        colores_pro.extend([colores_categoria[categoria]] * len(lista))
        
    vectores_pro = modelo_pro.encode(todas_palabras_pro)
    
    # PCA de 384D a 2D
    pca_pro = PCA(n_components=2)
    vectores_2d_pro = pca_pro.fit_transform(vectores_pro)
    
    ax2.scatter(vectores_2d_pro[:, 0], vectores_2d_pro[:, 1], c=colores_pro, s=100, alpha=0.6)
    for i, txt in enumerate(todas_palabras_pro):
        ax2.annotate(txt, (vectores_2d_pro[i, 0] + 0.05, vectores_2d_pro[i, 1] + 0.05), fontsize=8)
    
    ax2.set_title(f"B. Modelo Pro: MiniLM Semántico (N_EMBD=384)\nEstructura Ordenada", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=leyenda_patches)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    print("\n--- ¡Comparación finalizada! Revisa las ventanas de tu PC. ---")
    plt.show()