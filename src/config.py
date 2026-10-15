# src/config.py

import torch
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

# Model and Tokenizer Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

# Training Arguments Configuration
TRAINING_ARGUMENTS = TrainingArguments(
    output_dir="./qwen2.5_0.5B_medical",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="results/runs",        # for TensorBoard
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,                # keep only the last 2 checkpoints
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="tensorboard"
)

# Data Configuration
DATA_PATH = "data/medquad_clean.csv"