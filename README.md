# 🩺 MedAI — Medical QA with Qwen2.5 + QLoRA + RAG (FAISS)

A medical question-answering assistant built by fine-tuning **Qwen2.5-1.5B-Instruct** with **QLoRA** on the
**MedQuAD** dataset and grounding answers with **retrieval-augmented generation** (sentence embeddings + FAISS),
with a **Gradio** chat interface.

> Educational project — not medical advice.

**Fine-tuned LoRA adapter (Hugging Face Hub):** https://huggingface.co/Sudheer2002/qwen2.5-1.5b-medquad-qlora




https://github.com/user-attachments/assets/7c1e06ef-ef44-4779-97ed-3646f746ec07


## Architecture
```
TRAINING   MedQuAD (16,359 QA pairs after cleaning) -> Qwen chat template -> train/val/test (80/10/10)
           -> Qwen tokenizer -> 4-bit NF4 frozen Qwen2.5-1.5B + LoRA (r=16, attention + MLP, 1.2% trainable)
           -> cross-entropy on answer tokens only -> backprop -> paged 8-bit AdamW -> LoRA adapter

RAG INDEX  train-split answers -> 180-word chunks (40 overlap) -> BAAI/bge-small-en-v1.5
           -> FAISS IndexFlatIP (cosine), 22,449 chunks

INFERENCE  question -> query embedding -> FAISS top-4 -> question + context
           -> Qwen2.5 + merged LoRA -> streamed answer + sources -> Gradio
```

## Results
Training: 2,000 MedQuAD examples, 1 epoch (125 steps), single Kaggle T4 GPU.

| Metric | Value |
|---|---|
| Validation loss | 1.213 |
| Validation perplexity | 3.36 |

Generation quality on 50 held-out test questions:

| Model | ROUGE-L | BERTScore-F1 |
|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 0.140 | 0.773 |
| + QLoRA fine-tuning | 0.240 | **0.807** |
| + QLoRA + RAG | **0.263** | 0.800 |


<img width="613" height="393" alt="loss_curve" src="https://github.com/user-attachments/assets/b26ba4ff-df4f-4091-bf0e-1570b4d76de7" />


**Takeaways**
- QLoRA fine-tuning improved ROUGE-L by ~71% over the base model (0.140 → 0.240) and BERTScore-F1 from 0.773 to 0.807.
- RAG raised lexical overlap further (ROUGE-L 0.263) while BERTScore stayed about the same.
- Limitations: the evaluation set is small (n = 50), and manual review shows the 1.5B model can still state
  incorrect medical facts (e.g. inheritance patterns) even with retrieved context. Larger models, a
  stricter grounding prompt, and faithfulness evaluation are natural next steps.

## Repository layout
```
notebooks/MedAI_QLoRA_RAG_pipeline.ipynb   end-to-end: data -> QLoRA -> eval -> FAISS -> deploy
space/app.py            Gradio app (streaming, shows retrieved sources, ZeroGPU-ready)
space/rag.py            cleaning, split, chunking, embeddings, FAISS, prompt building
space/requirements.txt  app dependencies
docs/                   project flow notes, deployment guide, loss curve
```

## Run the demo
The Gradio app runs on any GPU notebook (e.g. Kaggle T4):
```python
!pip install -q -U transformers peft accelerate sentence-transformers faiss-cpu datasets gradio
!git clone https://github.com/gollasudheerbabu09-debug/Medical_ChatBot_002.git
%cd Medical_ChatBot_002/space
!sed -i 's/torch.bfloat16 if use_gpu/torch.float16 if use_gpu/' app.py
import os
os.environ["BASE_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"
os.environ["ADAPTER_ID"] = "Sudheer2002/qwen2.5-1.5b-medquad-qlora"
import app
app.demo.queue().launch(share=True)
```

## Design notes
- **Loss on answers only**: prompt tokens are masked with `-100`, and Qwen's own pad token is kept so the model learns the `<|im_end|>` stop token.
- **Dynamic padding at 512 tokens** instead of padding to 2048 cut training tokens by ~6.7×.
- **No test leakage in RAG**: answers that appear in the test split are excluded from the FAISS index.
