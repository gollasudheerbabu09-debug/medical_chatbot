# src/chatbot.py
"""QLoRA fine-tuning wrapper: 4-bit NF4 frozen Qwen + trainable LoRA adapters."""
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from .data_loader import to_messages


class MedicalChatbot:
    def __init__(self, model_name, lora_config, training_args):
        self.model_name = model_name
        self.training_args = dict(training_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)   # Qwen has its own pad token: keep it
        self.model = self._setup_model(LoraConfig(**lora_config))

    def _setup_model(self, lora_config):
        use_bf16 = torch.cuda.is_bf16_supported()                     # False on T4 -> fp16
        self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.training_args.update(bf16=use_bf16, fp16=not use_bf16)

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=self.compute_dtype, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb,
                                                     dtype=self.compute_dtype, device_map={"": 0})
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora_config)                    # LoRA applied once, here
        model.print_trainable_parameters()
        return model

    def fine_tune(self, train_dataset, eval_dataset):
        collator = DataCollatorForSeq2Seq(self.tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
        trainer = Trainer(model=self.model, args=TrainingArguments(**self.training_args),
                          train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collator)
        trainer.train()
        metrics = trainer.evaluate()
        print(f"Validation loss {metrics['eval_loss']:.4f} | perplexity {torch.exp(torch.tensor(metrics['eval_loss'])):.2f}")
        return metrics

    def save(self, path):
        self.model.save_pretrained(path)          # saves only the LoRA adapter
        self.tokenizer.save_pretrained(path)

    @torch.no_grad()
    def generate_response(self, question, max_new_tokens=200):
        self.model.eval()
        text = self.tokenizer.apply_chat_template(to_messages(question), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  repetition_penalty=1.05, use_cache=True)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
