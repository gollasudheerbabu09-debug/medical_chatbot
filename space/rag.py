"""
rag.py — MedAI retrieval pipeline (shared by the training notebook and the Gradio app).

Flow (matches the project PDF, Steps 20-27):
    MedQuAD answers -> cleaning -> chunking -> embedding model -> FAISS index
    user query -> query embedding -> similarity search -> top-K chunks -> prompt context

Heavy libraries (faiss, sentence-transformers, datasets) are imported lazily so that
the pure-Python parts (cleaning, chunking) can be tested without a GPU.
"""
import json
import os
import re

DATASET_ID = "prithvi1029/medquad-medical-qa"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # 384-dim, fast on CPU
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
SEED = 42


# ----------------------------------------------------------------------------
# 1. Data loading, cleaning and the fixed train/val/test split
# ----------------------------------------------------------------------------
def clean_df(df):
    """Null/empty removal, whitespace cleanup, de-duplication (PDF Step 2)."""
    df = df.dropna(subset=["Question", "Answer"]).copy()
    df["Question"] = df["Question"].astype(str).str.strip()
    df["Answer"] = df["Answer"].astype(str).str.strip()
    # collapse the long runs of spaces/newlines MedQuAD has inside answers
    df["Answer"] = df["Answer"].str.replace(r"[ \t]+", " ", regex=True)
    df["Answer"] = df["Answer"].str.replace(r"\n\s*\n+", "\n", regex=True)
    df = df[(df["Question"] != "") & (df["Answer"] != "")]
    df = df.drop_duplicates(subset=["Question", "Answer"]).reset_index(drop=True)
    return df


def split_df(df, seed=SEED):
    """80 / 10 / 10 split — identical everywhere because the seed is fixed (PDF Step 4)."""
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def load_splits():
    from datasets import load_dataset
    df = load_dataset(DATASET_ID)["train"].to_pandas()
    return split_df(clean_df(df))


# ----------------------------------------------------------------------------
# 2. Chunking (PDF Step 21)
# ----------------------------------------------------------------------------
def chunk_text(text, chunk_words=180, overlap=40):
    """Split text into overlapping word windows. Short texts stay as one chunk."""
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if len(piece) < 30 and chunks:          # tiny tail -> merge into previous chunk
            chunks[-1] = chunks[-1] + " " + " ".join(piece)
            break
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def build_chunks(train_df, exclude_answers=None, chunk_words=180, overlap=40):
    """
    Turn training-split answers into retrieval chunks with metadata.
    `exclude_answers` removes any answer that also appears in the test split, so
    RAG evaluation is not inflated by retrieving the exact reference answer.
    """
    exclude = set(exclude_answers or [])
    chunks, seen = [], set()
    for _, row in train_df.iterrows():
        ans = row["Answer"]
        if ans in exclude or ans in seen:        # same answer text appears under many questions
            continue
        seen.add(ans)
        for j, piece in enumerate(chunk_text(ans, chunk_words, overlap)):
            chunks.append({
                "id": len(chunks),
                "source_question": row["Question"],
                "qtype": str(row.get("qtype", "")),
                "part": j,
                # the question is prepended so each chunk says what it is about
                "text": f"{row['Question']}\n{piece}",
            })
    return chunks


# ----------------------------------------------------------------------------
# 3. Embeddings + FAISS (PDF Steps 22-23)
# ----------------------------------------------------------------------------
def load_embedder(device=None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device=device)


def build_index(chunks, embedder, out_dir, batch_size=64):
    import faiss
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    vecs = embedder.encode([c["text"] for c in chunks], batch_size=batch_size,
                           normalize_embeddings=True, show_progress_bar=True,
                           convert_to_numpy=True).astype(np.float32)
    # normalized vectors + inner product == cosine similarity
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, os.path.join(out_dir, "medquad.faiss"))
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return index


# ----------------------------------------------------------------------------
# 4. Retrieval (PDF Steps 24-26)
# ----------------------------------------------------------------------------
class Retriever:
    def __init__(self, index_dir, embedder=None):
        import faiss
        self.index = faiss.read_index(os.path.join(index_dir, "medquad.faiss"))
        with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.embedder = embedder or load_embedder()

    def search(self, query, k=4, min_score=0.35):
        import numpy as np
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)
        scores, ids = self.index.search(q, k)          # FAISS returns IDs + scores
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1 or score < min_score:
                continue
            c = dict(self.chunks[idx])                  # map ID -> original chunk text
            c["score"] = float(score)
            results.append(c)
        return results


# ----------------------------------------------------------------------------
# 5. Prompt construction (PDF Step 27)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are MedAI, a careful medical information assistant. "
    "Answer the user's medical question accurately and concisely. "
    "When context passages are provided, base your answer on them and do not invent facts "
    "that are not supported. If the context does not contain the answer, say so briefly and "
    "give general guidance. You do not diagnose; recommend seeing a healthcare professional "
    "when appropriate."
)


def build_messages(question, contexts=None, history=None):
    """Return chat messages for Qwen's chat template: system + (history) + context-augmented question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append(turn)
    if contexts:
        ctx = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))
        user = f"Context:\n{ctx}\n\nQuestion: {question}"
    else:
        user = question
    messages.append({"role": "user", "content": user})
    return messages

