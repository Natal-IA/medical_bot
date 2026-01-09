#_____TEMPORAL__________________________________________-
import os
from pathlib import Path
import streamlit as st

st.write("CWD:", os.getcwd())
st.write("Files in repo root:", [p.name for p in Path(".").iterdir()][:30])

chroma_path = Path("vector_store/chroma")
st.write("CHROMA exists?:", chroma_path.exists())
st.write("CHROMA absolute:", str(chroma_path.resolve()))
if chroma_path.exists():
    st.write("CHROMA files sample:", [p.name for p in chroma_path.iterdir()][:20])
#_______________________________________________-



import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
# --- Cargar entorno ---
# --- Ajustar PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # sube hasta medical_bot
sys.path.append(str(PROJECT_ROOT))

# --- Cargar entorno ---
load_dotenv(PROJECT_ROOT / ".env")  # local; en Streamlit usa secrets

from src.app.services.router import RouterService

st.set_page_config(page_title="Asistente Urología", page_icon="🩺")

st.title("🩺 Asistente de la clínica de urología")
st.caption("⚠️ Este asistente no sustituye una consulta médica. Para urgencias, acude a urgencias o llama al 112.")

# Inicializar router una vez
@st.cache_resource
def load_router():
    return RouterService()

router = load_router()

# Historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
if prompt := st.chat_input("Escribe tu pregunta"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        out = router.handle(prompt, top_k=5)
        st.markdown(out.answer)

        # (Opcional) mostrar fuentes
        if out.sources:
            with st.expander("Fuentes"):
                for s in out.sources:
                    st.write(f"- {s['source']} (dist={s['distance']:.3f})")

    st.session_state.messages.append(
        {"role": "assistant", "content": out.answer}
    )
