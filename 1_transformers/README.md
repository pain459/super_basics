# Mini Transformer (Sentiment) — Step‑by‑Step

This tiny project helps you **understand Transformers** by building and training a **mini Transformer encoder** for **sentiment classification** (positive/negative). It’s CPU‑friendly and focuses on the core ideas: **embeddings, positional encodings, multi‑head self‑attention, residuals/LayerNorm, and a [CLS] classifier**.

---

## ✅ What You’ll Do (in ~10 minutes)
1. **Create a Python env**
2. **Install PyTorch (CPU)**
3. **Run the script** to train a tiny model
4. **See predictions + attention** over tokens for a few test sentences

---

## 1) Create & Activate a Virtual Env
```bash
# Linux / macOS (Python ≥3.10 recommended)
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Tip: If you use `pyenv`, pick a local version first: `pyenv local 3.11.13` then make the venv.

---

## 2) Install Dependencies (CPU‑only)
**Option A — quick (PyTorch CPU wheels):**
```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Option B — generic (let pip resolve):**
```bash
pip install torch
```

> GPU is not needed. This runs fast on CPU.

---

## 3) Run the Mini Transformer
From the project folder:
```bash
python mini_transformer_sentiment.py
```

You’ll see training logs like:
```
Vocab size: 93 | Train: 23 | Val: 5
[epoch 01] loss=0.6921
[epoch 03] loss=0.6204
...
[epoch 30] loss=0.1350
Validation accuracy: 5/5 = 1.00
```

Then per test sentence, you’ll get a prediction and an **attention bar** showing how the **[CLS] token attends to each token** in the last encoder layer (head‑averaged), e.g.:
```
Sentence: delightful and funny experience
Tokens:   <cls> | delightful | and | funny | experience | <pad> | ...

Pred: POSITIVE (confidence 0.98)
Attention from [CLS] to tokens (last layer, head-avg):
       <cls>  ████████████████████
   delightful  █████████████
          and  ███
        funny  ███████████
   experience  █████████
         <pad>  
         <pad>  
```

---

## 4) Tweak & Rerun (to learn by doing)
Try changing these:
```bash
# Fewer epochs (faster), smaller model
python mini_transformer_sentiment.py --epochs 15 --d_model 48 --heads 3 --layers 1

# Larger model (still small) and more training
python mini_transformer_sentiment.py --epochs 40 --d_model 96 --heads 4 --layers 2
```

You can also **edit the dataset** at the top of the script — add more positive/negative sentences and rerun. The model should adapt quickly.

---

## How it Works (short version)
- **Tokenizer**: simple lowercase & alphanumeric split
- **Vocab**: includes special tokens: `<pad>`, `<unk>`, `<cls>`
- **Embeddings + Positional Encoding**: learn token vectors; add **sinusoidal** positions
- **Transformer blocks** (×L): each does **Multi‑Head Self‑Attention → Add & Norm → Feed‑Forward → Add & Norm**
- **Classification**: use the **[CLS]** token’s final hidden state → a linear layer → softmax
- **Attention Inspection**: we collect per‑head attention from the last layer and print how **[CLS]** distributes attention across tokens

---

## Troubleshooting
- **Torch install errors**: try the CPU index URL above; or `pip install --upgrade pip wheel setuptools`
- **Slow run**: reduce `--epochs` to 10‑15 and set `--layers 1`
- **Weird predictions**: the dataset is tiny; add more lines for better generalization

---

## Next Steps
- Swap the simple classifier head with **mean pooling** over tokens to compare behaviors
- Log attention as a **heatmap** (matplotlib) or print **top‑k attended tokens**
- Replace our blocks with `torch.nn.TransformerEncoderLayer` and compare results
- Try a **Hugging Face** quick‑start: fine‑tune `distilbert-base-uncased` on a larger dataset

Have fun poking at attention!
