
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import MODEL_NAME

# Base model and fine-tuned LoRA adapter
ADAPTER_PATH = "Sudheer2002/qwen2.5-1.5b-medquad-qlora"


def load_inference_model():
    """Load the base model and apply the trained LoRA adapter."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading base model: {MODEL_NAME}")
    print(f"Device: {device}")

    # Load the same base model used during fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print(f"Loading LoRA adapter: {ADAPTER_PATH}")

    # Download and apply the adapter from Hugging Face
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH
    )

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()

    model.to(device)
    model.eval()

    # Load the base model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and adapter loaded successfully.")

    return model, tokenizer


def generate_response(model, tokenizer, question):
    """Generate an answer using the fine-tuned model."""

    device = next(model.parameters()).device

    prompt = f"""### Instruction:
You are a medical chatbot. Answer the user's medical question
with clear, helpful, and medically cautious information.

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
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated answer
    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response
