# src/config.py
"""Central settings for MedAI. Every value can be overridden with an environment variable."""
import os

# ---- Model used by the app -------------------------------------------------
# Streamlit Community Cloud (CPU, ~2.7 GB RAM): the 0.5B model with LoRA already merged in.
MODEL_ID = os.getenv("MODEL_ID", "Sudheer2002/qwen2.5-0.5b-medquad-merged")
# Optional: set ADAPTER_ID to load base model + LoRA adapter instead (GPU / local use), e.g.
#   MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"  ADAPTER_ID="Sudheer2002/qwen2.5-1.5b-medquad-qlora"
ADAPTER_ID = os.getenv("ADAPTER_ID", "") or None

# ---- RAG -------------------------------------------------------------------
INDEX_REPO = os.getenv("INDEX_REPO", "Sudheer2002/medquad-faiss-index")  # HF dataset with medquad.faiss + chunks.jsonl
TOP_K = 3

# ---- Generation ------------------------------------------------------------
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.2

# ---- Training (used by the notebook-style scripts) -------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/medquad_clean.csv"   # optional local CSV; if missing, MedQuAD is downloaded from the HF Hub
MAX_LEN = 512                # tokens per training example (prompt + answer)
MAX_TRAIN_SAMPLES = 2000     # None = all ~13k training examples
N_VAL = 200                  # validation examples for eval loss

LORA_CONFIG = dict(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# fp16/bf16 are chosen automatically for the GPU in chatbot.py
TRAINING_ARGUMENTS = dict(
    output_dir="qwen-medical-qlora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    eval_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=50, save_total_limit=2,
    report_to="none",
    disable_tqdm=True,
)
