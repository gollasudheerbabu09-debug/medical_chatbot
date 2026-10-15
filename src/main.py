# src/main.py

from config import MODEL_NAME, LORA_CONFIG, TRAINING_ARGUMENTS, DATA_PATH
from data_loader import load_and_prepare_data
from chatbot import MedicalChatbot

def main():
    # Load and prepare data
    train_dataset = load_and_prepare_data(DATA_PATH)

    # Initialize the chatbot
    chatbot = MedicalChatbot(
        model_name=MODEL_NAME,
        lora_config=LORA_CONFIG,
        training_args=TRAINING_ARGUMENTS
    )

    # Fine-tune the chatbot
    chatbot.fine_tune(train_dataset)


    # Format the instruction to match the training data format
    instruction = """### Instruction:
                    You are a medical chatbot. Your task is to answer the user's medical question.

                    ### Question:
                    What are some early symptoms of cholera?

                    ### Response:
                    """

    print("Generating response...")
    response = chatbot.generate_response(instruction)
    print("\n--- Model Response ---")
    print(response)
    print("--------------------")


if __name__ == "__main__":
    main()