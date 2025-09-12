# src/data_loader.py

from datasets import load_dataset

def load_and_prepare_data(data_path):
    """
    Loads the dataset from a CSV file and formats it for training.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        Dataset: The formatted dataset.
    """
    dataset = load_dataset("csv", data_files=data_path, split="train")

    def format_instruction(sample):
        return f"""### Instruction:
You are a medical chatbot. Your task is to answer the user's medical question based on the provided context.

### Context:
{sample['answer']}

### Question:
{sample['question']}

### Response:
{sample['answer']}
"""

    return dataset