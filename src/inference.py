import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import your model configuration
from .config import MODEL_NAME

# Path to your saved LoRA adapter
ADAPTER_PATH = "./qwen_medical_finetuned/qwen_medical_finetuned"


def load_inference_model():
    """
    Loads the base model and merges it with the fine-tuned LoRA adapter.
    """
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA adapter and merging...")
    # Load the LoRA adapter and merge it with the base model
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload() # This is the key step!

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model and tokenizer ready for inference.")
    return model, tokenizer

def generate_response(model, tokenizer, question):
    """
    Generates a response to a given question using the fine-tuned model.
    """
    # Format the prompt using the same template as in training
    prompt = f"""### Instruction:
You are a medical chatbot. Your task is to answer the user's medical question.

### Question:
{question}

### Response:
"""

    # Tokenize the input and generate the output
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=250, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the response to only show the actual answer
    # This splits the full text at "### Response:" and takes the part after it.
    try:
        return response.split("### Response:")[1].strip()
    except IndexError:
        return "Could not generate a valid response."