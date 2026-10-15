
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import MODEL_NAME

# Hugging Face adapter repository
ADAPTER_PATH = "Sudheer2002/qwen2.5-1.5b-medquad-qlora"


def load_inference_model():
    """Load the base model and apply the LoRA adapter."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, question):
    """Generate a response using the fine-tuned model."""

    device = next(model.parameters()).device

    prompt = f"""### Instruction:
You are a medical chatbot. Your task is to answer the user's medical question.

### Question:
{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "### Response:" in response:
        return response.split("### Response:", 1)[1].strip()

    return response
