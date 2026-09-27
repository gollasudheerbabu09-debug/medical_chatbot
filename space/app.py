"""
app.py — MedAI Gradio app (Hugging Face Spaces).

Online inference flow (PDF Section 27 "ONLINE INFERENCE"):
    user question -> query embedding -> FAISS search -> top-K chunks
    -> question + context -> Qwen2.5 + LoRA -> streamed answer -> Gradio UI

Configuration via Space "Variables" (Settings -> Variables and secrets):
    BASE_MODEL  base model id          (default: Qwen/Qwen2.5-7B-Instruct)
    ADAPTER_ID  your LoRA adapter repo (e.g. your-username/qwen2.5-7b-medquad-qlora)
    INDEX_DIR   folder with medquad.faiss + chunks.jsonl (default: rag_index)
"""
import os
import threading

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import rag

try:                                   # present on ZeroGPU Spaces
    import spaces
    gpu = spaces.GPU(duration=90)
except ImportError:                    # local / CPU: no-op decorator
    def gpu(fn):
        return fn

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_ID = os.getenv("ADAPTER_ID", "").strip()
INDEX_DIR = os.getenv("INDEX_DIR", "rag_index")
SOURCES_MARK = "\n\n---\n**Sources"

# ---------------------------------------------------------------- retrieval
embedder = rag.load_embedder(device="cpu")
if not os.path.exists(os.path.join(INDEX_DIR, "medquad.faiss")):
    print("No prebuilt index found — building it from MedQuAD (one-time, a few minutes)...")
    train_df, _, _ = rag.load_splits()
    rag.build_index(rag.build_chunks(train_df), embedder, INDEX_DIR)
retriever = rag.Retriever(INDEX_DIR, embedder=embedder)
print(f"FAISS index loaded: {retriever.index.ntotal} chunks")

# ---------------------------------------------------------------- model
use_gpu = torch.cuda.is_available() or bool(os.getenv("SPACES_ZERO_GPU"))
dtype = torch.bfloat16 if use_gpu else torch.float32
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID or BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
if ADAPTER_ID:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model = model.merge_and_unload()   # fold LoRA (W + BA) into the weights for faster inference
    print(f"Loaded LoRA adapter: {ADAPTER_ID}")
else:
    print("WARNING: ADAPTER_ID not set — running the base model without fine-tuning.")
model.eval()
if use_gpu:
    model.to("cuda")


@gpu
def stream_generate(messages, max_new_tokens, temperature):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens),
                  repetition_penalty=1.05)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    threading.Thread(target=model.generate, kwargs=kwargs).start()
    out = ""
    for piece in streamer:
        out += piece
        yield out


def _text(content):
    """Gradio 5 gives strings, Gradio 6 may give a list of parts — normalise to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def respond(message, history, use_rag, top_k, max_new_tokens, temperature):
    question = _text(message).strip()
    if not question:
        yield "Please type a medical question."
        return

    # keep the last 2 exchanges, without the appended source lists
    past = []
    for turn in (history or [])[-4:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            past.append({"role": turn["role"], "content": _text(turn["content"]).split(SOURCES_MARK)[0]})

    contexts = retriever.search(question, k=int(top_k)) if use_rag else []
    messages = rag.build_messages(question, contexts, past)

    sources = ""
    if contexts:
        lines = [f"{i + 1}. {c['source_question']} (similarity {c['score']:.2f})"
                 for i, c in enumerate(contexts)]
        sources = SOURCES_MARK + " retrieved from MedQuAD (FAISS):**\n" + "\n".join(lines)
    elif use_rag:
        sources = SOURCES_MARK + ":** no sufficiently similar passage found; answered from the model alone."

    answer = ""
    for answer in stream_generate(messages, max_new_tokens, temperature):
        yield answer
    yield answer + sources


DESCRIPTION = """
**MedAI** answers general medical questions using **Qwen2.5-Instruct fine-tuned with QLoRA on MedQuAD**,
grounded by **retrieval-augmented generation (sentence embeddings + FAISS)**.

⚠️ *Educational use only. MedAI is not a doctor and cannot diagnose or treat. For symptoms or
treatment decisions, consult a qualified healthcare professional. In an emergency, call your local
emergency number (112 in India).*
"""

EXAMPLES = [
    "What are the symptoms of iron deficiency anemia?",
    "Who is at risk for Lymphocytic Choriomeningitis (LCM)?",
    "How is type 2 diabetes diagnosed?",
    "What causes glaucoma and can it be prevented?",
]

chat_kwargs = {"type": "messages"} if int(gr.__version__.split(".")[0]) < 6 else {}

with gr.Blocks(title="MedAI — Medical QA") as demo:
    gr.Markdown("# 🩺 MedAI — Medical Question Answering")
    gr.Markdown(DESCRIPTION)
    with gr.Accordion("Settings", open=False):
        use_rag = gr.Checkbox(value=True, label="Use RAG (retrieve MedQuAD passages)")
        top_k = gr.Slider(1, 8, value=4, step=1, label="Top-K retrieved chunks")
        max_new = gr.Slider(64, 768, value=384, step=32, label="Max new tokens")
        temp = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature (0 = greedy)")
    gr.ChatInterface(
        respond,
        additional_inputs=[use_rag, top_k, max_new, temp],
        examples=[[q, True, 4, 384, 0.3] for q in EXAMPLES],
        cache_examples=False,
        **chat_kwargs,
    )

if __name__ == "__main__":
    demo.queue().launch()
