"""Dashboard Streamlit para "conversar" con un checkpoint de PequeLLM.

El modelo no conversa de verdad: hace *completación de tokens*. Esta app lo
envuelve en una interfaz tipo chat para poder demostrarlo desde cualquier lado
(exponiendo el puerto 8501 con cloudflared o ngrok).

Reutiliza la lógica de inferencia que ya existe en
``Embeddings/generate_prompt.py`` — en particular ``load_model`` (que infiere la
configuración del checkpoint, así que sirve igual para Small y Medium) y
``sample_next_token``.

IMPORTANTE: esta app es SOLO-LECTURA sobre el checkpoint. Nunca escribe el .pth
ni llama a save_checkpoint; solo carga el modelo en modo eval y genera texto.

Uso (dentro del contenedor ROCm en renna):
    streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501

Normalmente se lanza con ``./run.sh dashboard`` (ver dashboard/README.md).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List

import streamlit as st
import torch
from tokenizers import Tokenizer

# ── Rutas e imports del repo ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "Embeddings"
for _p in (str(EMB_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emb_gpt2 import select_device  # noqa: E402
from generate_prompt import load_model, sample_next_token  # noqa: E402


DEFAULT_TOKENIZER = str(REPO_ROOT / "tokenizer-culturax-es-hf.json")
# Carpeta donde el entrenamiento guarda los checkpoints dentro del contenedor.
DATA_DIR = Path(os.environ.get("PEQUELLM_DATA_DIR", "/workspace/data"))


# ── Descubrimiento de checkpoints ───────────────────────────────────────────
def discover_checkpoints() -> Dict[str, str]:
    """Mapea 'etiqueta visible' -> ruta absoluta de cada .pth encontrado."""
    found: Dict[str, str] = {}
    for directory in (DATA_DIR, REPO_ROOT):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pth")):
            name = path.name.lower()
            if "medium" in name or "med" in name:
                label = f"GPT-2 Medium — {path.name}"
            elif "small" in name:
                label = f"GPT-2 Small — {path.name}"
            else:
                label = path.name
            found.setdefault(label, str(path))
    return found


# ── Carga (cacheada) de modelo + tokenizer ──────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo en memoria…")
def get_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(Path(checkpoint_path), device=device)
    return model, tokenizer


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Generación con streaming token-a-token ──────────────────────────────────
@torch.no_grad()
def stream_generate(
    model,
    tokenizer: Tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> Iterator[str]:
    """Genera tokens uno por uno y emite el texto nuevo decodificado.

    Decodifica sólo la parte generada (no el prompt) para que en el chat se vea
    únicamente la respuesta del modelo.
    """
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: List[int] = []
    prev_text = ""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        idx = torch.cat((idx, next_token), dim=1)
        generated.append(int(next_token.item()))
        text = tokenizer.decode(generated)
        delta = text[len(prev_text):]
        if delta:
            prev_text = text
            yield delta


def build_prompt(messages: List[Dict[str, str]], include_history: bool) -> str:
    """Construye el texto que se le da al modelo.

    - Sin historial: sólo el último mensaje del usuario (completación pura).
    - Con historial: concatena los turnos previos como texto plano. El modelo
      no tiene memoria real; esto es sólo más contexto para la completación y
      la ventana se recorta sola por block_size.
    """
    if not include_history:
        return messages[-1]["content"]
    return "\n".join(m["content"] for m in messages)


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PequeLLM Chat", page_icon="💬", layout="centered")
st.title("💬 PequeLLM")
st.caption(
    "Demo: el modelo **no conversa**, hace *completación de tokens*. "
    "Escribe un inicio de texto y el modelo lo continúa."
)

device = select_device("auto")
checkpoints = discover_checkpoints()

with st.sidebar:
    st.header("⚙️ Configuración")

    if checkpoints:
        etiqueta = st.selectbox("Modelo (checkpoint)", list(checkpoints.keys()))
        checkpoint_path = checkpoints[etiqueta]
    else:
        st.warning(
            f"No se encontraron checkpoints (.pth) en {DATA_DIR} ni en el repo. "
            "Indica la ruta manualmente."
        )
        checkpoint_path = st.text_input(
            "Ruta del checkpoint", value=str(DATA_DIR / "pequellm_medium_checkpoint.pth")
        )

    tokenizer_path = st.text_input("Tokenizer", value=DEFAULT_TOKENIZER)

    st.divider()
    max_new_tokens = st.slider("Tokens a generar", 16, 400, 120, step=8)
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.9, step=0.05)
    top_k = st.slider("top-k (0 = desactivado)", 0, 200, 50, step=5)
    include_history = st.checkbox(
        "Incluir conversación como contexto", value=False,
        help="El modelo no tiene memoria; esto sólo concatena los turnos previos como texto.",
    )

    st.divider()
    if st.button("🧹 Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

# Cargar el modelo seleccionado (cacheado por ruta).
try:
    model, tokenizer = get_model_and_tokenizer(checkpoint_path, tokenizer_path, device)
except FileNotFoundError as exc:
    st.error(f"No se pudo cargar el modelo: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 — mostrar cualquier fallo de carga en la UI
    st.error(f"Error cargando modelo/tokenizer: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("ℹ️ Modelo cargado")
    cfg = model.cfg
    st.markdown(
        f"- **device**: `{device}`\n"
        f"- **parámetros**: ~{count_parameters(model) / 1e6:.1f} M\n"
        f"- **n_layer**: {cfg.n_layer} · **n_head**: {cfg.n_head}\n"
        f"- **n_embd**: {cfg.n_embd} · **block_size**: {cfg.block_size}\n"
        f"- **vocab_size**: {cfg.vocab_size}"
    )

# ── Estado e historial del chat ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Escribe un inicio de texto…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_text = build_prompt(st.session_state.messages, include_history)
    prompt_ids = tokenizer.encode(prompt_text).ids

    with st.chat_message("assistant"):
        if not prompt_ids:
            respuesta = "_(El prompt no produjo tokens. Intenta con otro texto.)_"
            st.markdown(respuesta)
        elif max(prompt_ids) >= model.cfg.vocab_size:
            respuesta = (
                f"⚠️ El prompt contiene el token id {max(prompt_ids)} pero el "
                f"vocab_size del modelo es {model.cfg.vocab_size}. "
                "Usa el tokenizer que corresponde al checkpoint."
            )
            st.markdown(respuesta)
        else:
            with st.spinner("PequeLLM está escribiendo…"):
                respuesta = "".join(
                    stream_generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        device=device,
                    )
                )
            st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
