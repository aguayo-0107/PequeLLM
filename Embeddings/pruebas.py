from emb_gpt2 import (
    TrainConfig,
    GPTModel,
    select_device,
    REPO_ROOT,
)
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from dataclasses import fields

def completar(
    prompt: str,
    checkpoint_path: str = "/workspace/data/pequellm_medium_checkpoint.pth",
    tokenizer_path: str = str(REPO_ROOT / "tokenizer-culturax-es-hf.json"),
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "auto",
) -> str:
    """
    Carga el checkpoint del PequeLLM y completa el texto a partir de un prompt.

    Args:
        prompt:           Texto inicial (ej: "El cielo es")
        checkpoint_path:  Ruta al .pth guardado por save_checkpoint()
        tokenizer_path:   Ruta al tokenizer HuggingFace JSON
        max_new_tokens:   Cuántos tokens generar
        temperature:      > 1 más aleatorio, < 1 más determinista, 1 = normal
        top_k:            Muestrea solo entre los top-k tokens más probables (0 = desactivado)
        device:           "auto", "cpu", "cuda", "mps"

    Returns:
        El prompt original + la continuación generada.
    """
    device = select_device(device)

    # ── Cargar tokenizer ──────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # ── Cargar checkpoint ─────────────────────────────────────────────────────
    raw = torch.load(checkpoint_path, map_location=device)
    if not isinstance(raw, dict) or "config" not in raw:
        raise ValueError("El checkpoint no tiene el formato esperado (falta 'config').")

    # ── Filtrar configuración antigua ─────────────────────────────────────────
    config_guardada = raw["config"]
    claves_validas = {f.name for f in fields(TrainConfig)}
    config_filtrada = {k: v for k, v in config_guardada.items() if k in claves_validas}
    
    cfg = TrainConfig(**config_filtrada)
    model = GPTModel(cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    print(f"[INFO] Modelo cargado desde iter={raw.get('iteration', '?')} | device={device}")

    # ── Tokenizar el prompt ───────────────────────────────────────────────────
    input_ids = tokenizer.encode(prompt).ids
    if not input_ids:
        raise ValueError(f"El tokenizer no produjo tokens para el prompt: '{prompt}'")

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    # ── Generación con temperature + top-k ───────────────────────────────────
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]                       # (1, vocab_size)

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

    # ── Decodificar ───────────────────────────────────────────────────────────
    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)

if __name__ == "__main__":
    print("PequeLLM — Completar texto\n")

    # ── 1. Selección de Modelo ────────────────────────────────────────────────
    print("¿Qué modelo deseas cargar?")
    print("  [1] Modelo Medium (predeterminado - pequellm_medium_checkpoint.pth)")
    print("  [2] Modelo Small (pequellm_pesado_checkpoint.pth)")
    
    opcion = input("Elige una opción [1/2]: ").strip()
    
    if opcion == "2":
        modelo_seleccionado = "/workspace/data/pequellm_pesado_checkpoint.pth"
    else:
        # Aquí ponemos la ruta del modelo medium que configuramos en el paso anterior
        modelo_seleccionado = "/workspace/data/pequellm_medium_checkpoint.pth"

    print(f"\n[INFO] Se utilizará el modelo ubicado en: {modelo_seleccionado}\n")

    # ── 2. Configuración de Inferencia ────────────────────────────────────────
    prompt = input("Escribe tu prompt: ").strip()
    if not prompt:
        print("Prompt vacío, usando ejemplo.")
        prompt = "El sol salía sobre las montañas"

    tokens_input = input("Tokens a generar [100]: ").strip()
    tokens = int(tokens_input) if tokens_input else 100
    
    temp_input = input("Temperature 0.1–2.0 [1.0]: ").strip()
    temp = float(temp_input) if temp_input else 1.0

    print("\n--- Generando... ---\n")
    
    # ── 3. Llamada a la función con el modelo seleccionado ────────────────────
    # Nota: Es crucial pasar explícitamente el argumento 'checkpoint_path' 
    # para sobreescribir el valor por defecto que tiene la función.
    resultado = completar(
        prompt=prompt, 
        checkpoint_path=modelo_seleccionado, 
        max_new_tokens=tokens, 
        temperature=temp
    )
    
    print(resultado)
    