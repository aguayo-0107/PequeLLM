import json
import os

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class MyTokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)} # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # 1. Preparar IDs iniciales (bytes)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # 2. Determinar cuántos merges faltan
        current_vocab_size = 256 + len(self.merges)
        num_merges = vocab_size - current_vocab_size
        
        if num_merges <= 0:
            print("El vocabulario ya alcanzó el tamaño deseado.")
            return

        # Si ya teníamos merges previos, debemos pre-procesar los IDs
        if len(self.merges) > 0:
            print("Re-aplicando merges previos al texto...")
            for pair, idx in self.merges.items():
                ids = merge(ids, pair, idx)

        print(f"Iniciando entrenamiento desde {current_vocab_size} hasta {vocab_size}...")

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats: break
            
            pair = max(stats, key=stats.get)
            idx = 256 + len(self.merges) # El nuevo ID es el siguiente disponible
            
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Merge {current_vocab_size + i + 1}/{vocab_size} completado.")

    def save(self, filename):
        """Guarda los merges en un JSON para no perder el progreso."""
        # JSON no acepta tuplas como llaves, convertimos a string "id1,id2"
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "merges": serializable_merges,
            "vocab_size": len(self.vocab)
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Modelo guardado en {filename}")

    def load(self, filename):
        """Carga los merges desde un archivo."""
        if not os.path.exists(filename):
            print("No se encontró archivo de guardado previo.")
            return
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruir merges (convertir "1,2" de nuevo a tupla (1,2))
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data["merges"].items()}
        # Reconstruir vocabulario
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        print(f"Modelo cargado: Vocabulario de {len(self.vocab)} tokens.")

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        partes_bytes = [self.vocab[idx] for idx in ids]
        texto_bytes = b"".join(partes_bytes)
        return texto_bytes.decode("utf-8", errors="replace")