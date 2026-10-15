# app.py — MedAI Streamlit app (deploy with main file path: app.py)
import streamlit as st
import torch

from src import config
from src.inference import build_prompt, load_inference_model, load_retriever, stream_response

st.set_page_config(page_title="MedAI — Medical QA", page_icon="🩺", layout="centered")
torch.set_num_threads(2)


@st.cache_resource(show_spinner="Loading the fine-tuned model (first start can take a few minutes)…")
def get_model_and_tokenizer():
    return load_inference_model()


@st.cache_resource(show_spinner="Loading the FAISS retrieval index…")
def get_retriever():
    return load_retriever()


st.title("🩺 MedAI — Medical Question Answering")
st.caption("Qwen2.5 fine-tuned with QLoRA on MedQuAD · RAG with bge-small embeddings + FAISS")

with st.sidebar:
    st.header("Settings")
    use_rag = st.toggle("Use RAG (retrieve MedQuAD passages)", value=True)
    top_k = st.slider("Top-K retrieved chunks", 1, 5, config.TOP_K)
    max_new = st.slider("Max answer length (tokens)", 64, 400, config.MAX_NEW_TOKENS, step=16)
    temp = st.slider("Temperature (0 = greedy)", 0.0, 1.0, config.TEMPERATURE, step=0.05)
    if st.button("Clear chat"):
        st.session_state.messages = []
    st.divider()
    st.warning("Educational use only — not medical advice. MedAI cannot diagnose or treat. "
               "Consult a qualified healthcare professional. In an emergency call 112 (India).")
    st.caption("Runs on a free shared CPU, so answers can take 20–60 seconds.")

model, tokenizer = get_model_and_tokenizer()
retriever = get_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources retrieved from MedQuAD (FAISS)"):
                st.markdown("\n".join(f"- {s}" for s in m["sources"]))

if question := st.chat_input("Ask a medical question, e.g. What are the symptoms of iron deficiency anemia?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1][-2:]]
    messages, sources = build_prompt(question, retriever if use_rag else None, top_k, history)

    with st.chat_message("assistant"):
        answer = st.write_stream(stream_response(model, tokenizer, messages, max_new, temp))
        if sources:
            with st.expander("Sources retrieved from MedQuAD (FAISS)"):
                st.markdown("\n".join(f"- {s}" for s in sources))
        elif use_rag:
            st.caption("No sufficiently similar passage found — answered from the model alone.")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
