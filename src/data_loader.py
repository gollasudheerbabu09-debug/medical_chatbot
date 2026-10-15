# src/data_loader.py
"""Load MedQuAD, clean it, split it and tokenize it exactly like the Kaggle notebook."""
import os

import pandas as pd
from datasets import Dataset

from space import rag


def to_messages(question, answer=None):
    """Qwen chat format used for training AND inference (same system prompt as the app)."""
    messages = [{"role": "system", "content": rag.SYSTEM_PROMPT}, {"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def load_splits(data_path=None):
    """Return (train_df, val_df, test_df) with columns Question / Answer."""
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        from datasets import load_dataset
        df = load_dataset(rag.DATASET_ID)["train"].to_pandas()
    # accept question/answer in any letter case
    df = df.rename(columns={c: c.capitalize() for c in df.columns if c.lower() in ("question", "answer")})
    return rag.split_df(rag.clean_df(df))


def _tokenize(tokenizer, max_len):
    def fn(ex):
        # loss only on the answer: prompt positions get label -100
        prompt = tokenizer.apply_chat_template(to_messages(ex["Question"]), tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(to_messages(ex["Question"], ex["Answer"]), tokenize=False)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]   # answer + <|im_end|>
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    return fn


def load_and_prepare_data(data_path, tokenizer, max_len=512, max_train_samples=None, n_val=200):
    """Return (tokenized_train, tokenized_val, test_df)."""
    train_df, val_df, test_df = load_splits(data_path)
    if max_train_samples:
        train_df = train_df.sample(min(max_train_samples, len(train_df)), random_state=42)
    val_df = val_df.sample(min(n_val, len(val_df)), random_state=42)

    cols = ["Question", "Answer"]
    fn = _tokenize(tokenizer, max_len)
    train_ds = Dataset.from_pandas(train_df[cols].reset_index(drop=True)).map(fn, remove_columns=cols)
    val_ds = Dataset.from_pandas(val_df[cols].reset_index(drop=True)).map(fn, remove_columns=cols)
    return train_ds, val_ds, test_df
