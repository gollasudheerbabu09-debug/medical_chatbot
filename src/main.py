# src/main.py
"""Train MedAI from the command line (needs a GPU). Run from the repository root:

    pip install -r requirements-train.txt
    python -m src.main
"""
from .chatbot import MedicalChatbot
from .config import (DATA_PATH, LORA_CONFIG, MAX_LEN, MAX_TRAIN_SAMPLES, MODEL_NAME,
                     N_VAL, TRAINING_ARGUMENTS)
from .data_loader import load_and_prepare_data


def main():
    chatbot = MedicalChatbot(MODEL_NAME, LORA_CONFIG, TRAINING_ARGUMENTS)

    train_ds, val_ds, _ = load_and_prepare_data(DATA_PATH, chatbot.tokenizer, MAX_LEN, MAX_TRAIN_SAMPLES, N_VAL)
    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    chatbot.fine_tune(train_ds, val_ds)
    chatbot.save(TRAINING_ARGUMENTS["output_dir"])

    print("\n--- Sample answer ---")
    print(chatbot.generate_response("What are some early symptoms of cholera?"))


if __name__ == "__main__":
    main()
