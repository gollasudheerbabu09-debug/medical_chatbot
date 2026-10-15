# src/chatbot.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model
from trl import SFTTrainer

class MedicalChatbot:
    """
    A medical chatbot that can be fine-tuned on a given dataset.
    """
    def __init__(self, model_name, lora_config, training_args):
        self.model_name = model_name
        self.lora_config = lora_config
        self.training_args = training_args
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()

    def _setup_tokenizer(self):
        """Initializes the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _setup_model(self):
        """Initializes the model with PEFT configuration."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return get_peft_model(model, self.lora_config)

    def fine_tune(self, train_dataset):
        """
        Fine-tunes the model on the provided training dataset.

        Args:
            train_dataset (Dataset): The dataset to train on.
        """
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            peft_config=self.lora_config,
            args=self.training_args,
            tokenizer=self.tokenizer,
            packing=True,
            dataset_text_field="text" 
        )
        trainer.train()

    def generate_response(self, instruction):
        """
        Generates a response to a given instruction.

        Args:
            instruction (str): The instruction to respond to.

        Returns:
            str: The generated response.
        """
        device = "cuda:0"
        inputs = self.tokenizer(instruction, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)