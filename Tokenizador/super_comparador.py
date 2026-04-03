import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer

def obtener_datos_completos(tokenizer_obj, textos, tipo="hf_library"):
    """
    Procesa textos y devuelve frecuencias con texto real y ratio de compresión.
    Tipos: 'hf_library' (el tuyo de 50k), 'transformers' (GPT2/Llama), 'manual' (el primero)
    """
    tokens_texto = []
    total_bytes = 0
    
    for t in textos:
        total_bytes += len(t.encode('utf-8'))
        
        if tipo == "hf_library": # Tu Wiki-50k
            ids = tokenizer_obj.encode(t).ids
            tokens_texto.extend(tokenizer_obj.encode(t).tokens)
        elif tipo == "transformers": # GPT2 o Llama2
            tokens = tokenizer_obj.tokenize(t)
            tokens_texto.extend(tokens)
        elif tipo == "manual": # Tu primer tokenizador
            ids = tokenizer_obj.encode(t)
            # Decodificamos cada ID a su pedazo de texto
            tokens_texto.extend([tokenizer_obj.vocab[i].decode('utf-8', errors='replace') for i in ids])
            
    frecuencias = Counter(tokens_texto)
    ratio = total_bytes / len(tokens_texto) if tokens_texto else 0
    return frecuencias, len(tokens_texto), ratio

def graficar_frecuencias_texto(frecuencias, titulo, color):
    """Histograma usando los segmentos de texto en el eje X."""
    top_20 = frecuencias.most_common(20)
    labels, counts = zip(*top_20)
    
    # Limpiamos visualmente los caracteres de control (como Ġ o  )
    labels = [label.replace('Ġ', ' ').replace(' ', ' ') for label in labels]
    
    plt.figure(figsize=(12, 5))
    sns.barplot(x=list(labels), y=list(counts), color=color)
    plt.title(f"Top 20 Segmentos más frecuentes: {titulo}")
    plt.xticks(rotation=45)
    plt.show()

from tokenizer import MyTokenizer # Tu manual
from tokenizers import Tokenizer as LibTokenizer # Tu Wiki-50k
from security_utils import load_dataset_secure

# --- CARGA DE MODELOS ---
print("Cargando tokenizadores...")
tk_manual = MyTokenizer()
tk_manual.load("mi_tokenizador_es.json")

tk_wiki50k = LibTokenizer.from_file("tokenizer-wiki-es-50k.json")

# Cargamos GPT-2 y Llama-2 de Hugging Face
tk_gpt2 = AutoTokenizer.from_pretrained("gpt2")
tk_llama = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

# --- MUESTRA DE DATOS ---
dataset = load_dataset_secure("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
muestra = [item["text"] for _, item in zip(range(50), dataset)]

# --- PROCESAMIENTO ---
modelos = [
    (tk_manual, "Manual", "salmon", "manual"),
    (tk_wiki50k, "Wiki-50k", "teal", "hf_library"),
    (tk_gpt2, "GPT-2 (OpenAI)", "skyblue", "transformers"),
    (tk_llama, "Llama-2 (Meta)", "orange", "transformers")
]

resultados = []

for obj, nombre, color, tipo in modelos:
    print(f"Analizando {nombre}...")
    frec, total_t, ratio = obtener_datos_completos(obj, muestra, tipo)
    resultados.append({"Modelo": nombre, "Tokens Totales": total_t, "Compresión (B/T)": round(ratio, 2)})
    graficar_frecuencias_texto(frec, nombre, color)

# --- REPORTE FINAL ---
df_final = pd.DataFrame(resultados)
print("\n" + "="*40)
print("REPORTE TÉCNICO DE TOKENIZACIÓN")
print("="*40)
print(df_final)
