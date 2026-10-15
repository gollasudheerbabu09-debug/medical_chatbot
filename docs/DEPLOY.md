# Deployment guide (GitHub + Hugging Face)

## 1. Push the code to GitHub
Create an empty **public** repo named `medai` on GitHub (no README), then on your computer:
```bash
unzip medai.zip && cd medai
git init
git add .
git commit -m "MedAI: Qwen2.5 QLoRA + FAISS RAG medical QA"
git branch -M main
git remote add origin https://github.com/<your-github-username>/medai.git
git push -u origin main
```
(GitHub asks for a Personal Access Token instead of your password: Settings → Developer settings → Tokens.)

## 2. Train, evaluate and deploy (GPU notebook)
1. Kaggle: New Notebook → File → Import Notebook → upload `notebooks/MedAI_QLoRA_RAG_pipeline.ipynb`.
   Settings → Accelerator **GPU T4**, Internet **On**. (Colab: Runtime → Change runtime type → T4 GPU.)
2. Create a Hugging Face token with **Write** access: https://huggingface.co/settings/tokens
3. In the CONFIG cell set `GITHUB_REPO_URL` and `HF_USERNAME`.
4. Run all. Rough timings on a free T4: 7B with 4,000 examples ≈ 1–2 h training; the Trainer progress bar
   shows the real ETA. `MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"` is several times faster.
5. The last cells push the LoRA adapter to the Hub, create the Space `<you>/MedAI`, upload the app and
   FAISS index, and set the `BASE_MODEL` / `ADAPTER_ID` variables.

## 3. Space hardware
- **7B model → needs a GPU.** Choose **ZeroGPU** in the Space Settings. HF currently lets free accounts in
  good standing (verified email, account > 30 days old) host a small number of ZeroGPU Spaces; PRO accounts
  get more. The app already uses `@spaces.GPU`.
- **Free CPU basic (2 vCPU / 16 GB)** cannot run 7B. Retrain with `Qwen/Qwen2.5-1.5B-Instruct` (or 0.5B)
  and it will run on CPU, slowly (several seconds per answer).
- First start downloads the model (~15 GB for 7B), so allow 5–10 minutes. Check the **Logs** tab if it fails.

## 4. After it works
- Put the real metrics from `eval_results.csv` and the Space link into `README.md` and push again.
- Add `loss_curve.png` to the repo if you like (`docs/`).
