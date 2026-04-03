import json
import os
from pathlib import Path

from security_utils import atomic_write_json

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
        merge_ids = list(self.merges.values())
        if len(merge_ids) != len(set(merge_ids)):
            raise ValueError("Tokenizer merges contain duplicate token IDs; refusing to save a malformed model.")
        if merge_ids and (min(merge_ids) < 256 or max(merge_ids) > 65535):
            raise ValueError("Tokenizer token IDs must stay within the uint16 range before saving.")

        serializable_merges = {
            f"{k[0]},{k[1]}": v
            for k, v in sorted(self.merges.items(), key=lambda item: item[1])
        }
        data = {
            "merges": serializable_merges,
            "vocab_size": len(self.vocab)
        }
        atomic_write_json(Path(filename), data)
        print(f"Modelo guardado en {filename}")

    def load(self, filename):
        """Carga los merges desde un archivo."""
        if not os.path.exists(filename):
            print("No se encontró archivo de guardado previo.")
            return
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("El archivo del tokenizador debe contener un objeto JSON válido.")
        raw_merges = data.get("merges")
        if not isinstance(raw_merges, dict):
            raise ValueError("El archivo del tokenizador no contiene una tabla 'merges' válida.")

        # Reconstruir merges (convertir "1,2" de nuevo a tupla (1,2))
        parsed_merges = {}
        seen_ids = set()
        for raw_pair, idx in raw_merges.items():
            if not isinstance(raw_pair, str) or "," not in raw_pair:
                raise ValueError(f"Par de merge inválido: {raw_pair!r}")
            if not isinstance(idx, int):
                raise ValueError(f"ID de merge inválido para {raw_pair!r}: {idx!r}")
            if idx < 256 or idx > 65535:
                raise ValueError(f"ID de merge fuera de rango uint16: {idx}")

            p0, p1 = map(int, raw_pair.split(',', 1))
            parsed_merges[(p0, p1)] = idx
            if idx in seen_ids:
                raise ValueError(f"ID de merge duplicado detectado: {idx}")
            seen_ids.add(idx)

        self.merges = parsed_merges
        # Reconstruir vocabulario
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda item: item[1]):
            if p0 not in self.vocab or p1 not in self.vocab:
                raise ValueError(
                    f"El archivo del tokenizador referencia un merge dependiente inexistente: {(p0, p1)} -> {idx}"
                )
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        expected_vocab_size = data.get("vocab_size")
        if expected_vocab_size is not None and expected_vocab_size != len(self.vocab):
            raise ValueError(
                f"vocab_size inconsistente: archivo={expected_vocab_size}, reconstruido={len(self.vocab)}"
            )
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
