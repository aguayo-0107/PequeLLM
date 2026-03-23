from tokenizer import MyTokenizer # Tu primer tokenizador
from tokenizers import Tokenizer as HFTokenizer # El de HuggingFace
from datasets import load_dataset
import tokenizer_analytics as ana

# 1. Cargar ambos modelos
tk_manual = MyTokenizer()
tk_manual.load("mi_tokenizador_es.json")

tk_hf = HFTokenizer.from_file("tokenizer-culturax-es-hf.json")

# 2. Obtener una muestra de datos para analizar
print("Cargando muestra de datos...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True, trust_remote_code=True)
muestra_textos = [item["text"] for _, item in zip(range(100), dataset)]
print(len(muestra_textos))

# 3. Ejecutar análisis
print("Analizando Tokenizador Manual...")
frec_m, total_m, ratio_m = ana.obtener_estadisticas(tk_manual, muestra_textos, es_huggingface=False)

print("Analizando Tokenizador Culturax...")
frec_h, total_h, ratio_h = ana.obtener_estadisticas(tk_hf, muestra_textos, es_huggingface=True)

# 4. Visualizar
ana.graficar_histograma(frec_m, "Manual BPE", color="salmon")
ana.graficar_histograma(frec_h, "HuggingFace 50k", color="teal")

# 5. Tabla comparativa
df_comp = ana.comparar_tokenizadores((frec_m, total_m, ratio_m), (frec_h, total_h, ratio_h))
print("\n--- REPORTE COMPARATIVO ---")
print(df_comp)

def inspeccionar_tokens(tokenizer, texto, es_hf=True):
    if es_hf:
        tokens = tokenizer.encode(texto).tokens
    else:
        # Para el manual, convertimos IDs a texto para ver qué son
        ids = tokenizer.encode(texto)
        tokens = [tokenizer.vocab[i].decode('utf-8', errors='replace') for i in ids]
    
    print(f"\nInterpretación ({'HF' if es_hf else 'Manual'}):")
    print(f"Tokens: {tokens}")

test = "El conocimiento es poder."
inspeccionar_tokens(tk_manual, test, es_hf=False)
inspeccionar_tokens(tk_hf, test, es_hf=True)