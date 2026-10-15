# src/inference.py
"""Model loading, retrieval and generation for the MedAI app.

Uses the SAME prompt format the model was fine-tuned on (Qwen chat template + MedAI system prompt),
and the SAME retrieval code as training/evaluation (space/rag.py).
"""
import threading

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from space import rag
from .config import ADAPTER_ID, INDEX_REPO, MODEL_ID


def load_inference_model():
    """Load the fine-tuned model and its tokenizer."""
    use_gpu = torch.cuda.is_available()
    dtype = torch.float16 if use_gpu else torch.bfloat16   # bf16 halves CPU memory vs float32

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID or MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, low_cpu_mem_usage=True)

    if ADAPTER_ID:                                   # base model + LoRA adapter -> merge W + BA
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_ID).merge_and_unload()

    if use_gpu:
        model.to("cuda")
    model.eval()
    return model, tokenizer


def load_retriever():
    """Download the prebuilt FAISS index from the Hugging Face Hub and load it."""
    index_dir = snapshot_download(INDEX_REPO, repo_type="dataset")
    return rag.Retriever(index_dir, embedder=rag.load_embedder(device="cpu"))


def build_prompt(question, retriever=None, top_k=3, history=None):
    """Retrieve context (optional) and build chat messages. Returns (messages, sources)."""
    contexts = retriever.search(question, k=top_k) if retriever is not None else []
    messages = rag.build_messages(question, contexts, history)
    sources = [f"{c['source_question']} (similarity {c['score']:.2f})" for c in contexts]
    return messages, sources


def stream_response(model, tokenizer, messages, max_new_tokens=200, temperature=0.2):
    """Yield the answer piece by piece (for st.write_stream)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens), repetition_penalty=1.05)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    yield from streamer


def generate_response(model, tokenizer, question, retriever=None, top_k=3, max_new_tokens=200, temperature=0.2):
    """Non-streaming helper: returns (answer, sources)."""
    messages, sources = build_prompt(question, retriever, top_k)
    answer = "".join(stream_response(model, tokenizer, messages, max_new_tokens, temperature)).strip()
    return answer, sources
