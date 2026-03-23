import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np

def obtener_estadisticas(tokenizer, textos, es_huggingface=True):
    """Calcula frecuencias de tokens y ratio de compresión."""
    todos_los_ids = []
    total_bytes = 0
    
    for t in textos:
        total_bytes += len(t.encode('utf-8'))
        if es_huggingface:
            ids = tokenizer.encode(t).ids
        else:
            ids = tokenizer.encode(t)
        todos_los_ids.extend(ids)
    
    frecuencias = Counter(todos_los_ids)
    total_tokens = len(todos_los_ids)
    
    # Cálculo de ratio de compresión
    ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    
    return frecuencias, total_tokens, ratio

def graficar_histograma(frecuencias, titulo, color="skyblue"):
    """Genera un histograma de los 30 tokens más comunes."""
    # Convertir IDs a string para el eje X
    top_30 = frecuencias.most_common(30)
    labels, counts = zip(*top_30)
    labels = [str(l) for l in labels]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels, y=counts, color=color)
    plt.title(f"Top 30 Tokens más frecuentes - {titulo}")
    plt.xticks(rotation=45)
    plt.ylabel("Frecuencia")
    plt.xlabel("ID del Token")
    plt.show()

def comparar_tokenizadores(stats_manual, stats_hf):
    """Crea una tabla comparativa de métricas clave."""
    data = {
        "Métrica": ["Total Tokens", "Ratio Compresión (Bytes/Token)", "Vocabulario Real Usado"],
        "Manual (Small Vocab)": [stats_manual[1], f"{stats_manual[2]:.2f}", len(stats_manual[0])],
        "HuggingFace (50k)": [stats_hf[1], f"{stats_hf[2]:.2f}", len(stats_hf[0])]
    }
    return pd.DataFrame(data)