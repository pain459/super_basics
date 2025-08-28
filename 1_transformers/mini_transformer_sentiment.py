#!/usr/bin/env python3
"""
Mini Transformer for Sentiment Classification (from scratch-ish, in PyTorch)

What you get:
- Tiny dataset (you can add more lines easily)
- Tokenizer + vocab
- Sinusoidal positional encodings
- A 2-layer Transformer encoder (using nn.MultiheadAttention so you can inspect attention weights)
- Simple training loop (CPU-friendly, finishes in seconds on a small dataset)
- At the end: predictions + printed attention heat for each test sentence

Run:
  python mini_transformer_sentiment.py

Optional args:
  --epochs 30         # change training epochs
  --d_model 64        # embedding size
  --heads 4           # number of attention heads
  --layers 2          # transformer layers
  --max_len 24        # sequence length
  --lr 0.002          # learning rate
  --seed 42           # reproducibility
"""
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) Tiny dataset
# -----------------------------
RAW_DATA: List[Tuple[str, int]] = [
    # positive (1)
    ("I loved this movie", 1),
    ("An excellent watch", 1),
    ("Absolutely fantastic, would recommend", 1),
    ("Great acting and a solid story", 1),
    ("The film was delightful and heartwarming", 1),
    ("What a brilliant performance", 1),
    ("I enjoyed every minute of it", 1),
    ("A charming, uplifting experience", 1),
    ("Wonderful direction and pacing", 1),
    ("This was a beautiful surprise", 1),
    ("It made me smile throughout", 1),
    ("The soundtrack was amazing", 1),
    ("Pretty good and well executed", 1),
    ("Smart, funny, and touching", 1),

    # negative (0)
    ("This was terrible", 0),
    ("Not worth the time", 0),
    ("I hated this film", 0),
    ("Boring and predictable plot", 0),
    ("A complete waste of time", 0),
    ("Awful acting and weak script", 0),
    ("Painfully dull and messy", 0),
    ("I regret watching this", 0),
    ("Disappointing and underwhelming", 0),
    ("Poor pacing and direction", 0),
    ("It made me yawn repeatedly", 0),
    ("Flat characters and no depth", 0),
    ("Terrible editing ruined it", 0),
    ("Bad, loud, and confusing", 0),
]

TEST_SENTENCES = [
    "i loved the acting and story",
    "this was a complete waste of time",
    "delightful and funny experience",
    "boring, dull and predictable",
]

# -----------------------------
# 2) Tokenizer / Vocab
# -----------------------------
PAD = "<pad>"
UNK = "<unk>"
CLS = "<cls>"

def simple_tokenize(text: str) -> List[str]:
    # lowercase, keep letters/numbers, split on spaces & punctuation
    text = text.lower()
    # replace punctuation with space
    out = []
    token = []
    for ch in text:
        if ch.isalnum():
            token.append(ch)
        else:
            if token:
                out.append("".join(token))
                token = []
    if token:
        out.append("".join(token))
    return out

def build_vocab(pairs: List[Tuple[str, int]], min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for text, _ in pairs:
        counter.update(simple_tokenize(text))
    # special tokens first
    itos = [PAD, UNK, CLS]
    for tok, freq in counter.items():
        if freq >= min_freq and tok not in itos:
            itos.append(tok)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi

def encode_sentence(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    toks = [CLS] + simple_tokenize(text)
    ids = [vocab.get(t, vocab[UNK]) for t in toks[:max_len]]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids

# -----------------------------
# 3) Positional Encoding
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# -----------------------------
# 4) Transformer Block (w/ attention weights)
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None, need_attn=False):
        # Self-attention
        attn_out, attn_weights = self.mha(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=need_attn,
            average_attn_weights=False  # keep per-head weights
        )
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights

# -----------------------------
# 5) The Model
# -----------------------------
class MiniTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, heads: int, layers: int, num_classes: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, dim_ff=4*d_model, dropout=dropout)
            for _ in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, key_padding_mask=None, return_attn=False):
        # x: (batch, seq_len) -> (batch, seq_len, d_model)
        h = self.emb(x)
        h = self.pos(h)
        attn_collector = []
        for i, block in enumerate(self.blocks):
            need_attn = return_attn  # request weights on all layers if asked
            h, attn = block(h, key_padding_mask=key_padding_mask, need_attn=need_attn)
            if return_attn:
                attn_collector.append(attn)  # shape: (batch, heads, tgt_len, src_len)
        # Use the CLS token representation (position 0) for classification
        cls_repr = h[:, 0, :]  # (batch, d_model)
        logits = self.cls_head(self.dropout(cls_repr))
        if return_attn:
            return logits, attn_collector
        return logits

# -----------------------------
# 6) Utilities: batching, training
# -----------------------------
@dataclass
class Batch:
    x: torch.Tensor         # (B, T)
    y: torch.Tensor         # (B, )
    pad_mask: torch.Tensor  # (B, T) True where PAD

def make_batches(pairs: List[Tuple[str, int]], vocab: Dict[str, int], max_len: int, batch_size: int) -> List[Batch]:
    random.shuffle(pairs)
    batches = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        xs, ys = [], []
        for text, label in chunk:
            ids = encode_sentence(text, vocab, max_len)
            xs.append(ids)
            ys.append(label)
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        pad_mask = (x == vocab[PAD])  # True where padding
        batches.append(Batch(x, y, pad_mask))
    return batches

def train(model, train_pairs, vocab, epochs=30, max_len=24, batch_size=8, lr=2e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        batches = make_batches(train_pairs, vocab, max_len, batch_size)
        for b in batches:
            x = b.x.to(device)
            y = b.y.to(device)
            pad_mask = b.pad_mask.to(device)
            opt.zero_grad()
            logits = model(x, key_padding_mask=pad_mask, return_attn=False)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(train_pairs)
        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"[epoch {ep:02d}] loss={avg:.4f}")
    return model

# -----------------------------
# 7) Pretty-print attention
# -----------------------------
def print_attention_for_sentence(model, sentence: str, vocab: Dict[str, int], max_len: int, device="cpu"):
    model.eval()
    with torch.no_grad():
        ids = encode_sentence(sentence, vocab, max_len)
        x = torch.tensor([ids], dtype=torch.long).to(device)
        pad_mask = (x == vocab[PAD]).to(device)
        logits, attn_layers = model(x, key_padding_mask=pad_mask, return_attn=True)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        label_name = "POSITIVE" if pred == 1 else "NEGATIVE"
        conf = probs[pred].item()

        # tokens used
        inv_vocab = {i:t for t,i in vocab.items()}
        tokens = [inv_vocab.get(i, UNK) for i in ids]

        print("\nSentence:", sentence)
        print("Tokens:  ", " | ".join(tokens))
        print(f"Pred: {label_name}  (confidence {conf:.2f})")

        # Show attention from the last layer, head-averaged, from CLS to all tokens
        last_attn = attn_layers[-1]  # (1, heads, T, T)
        attn_avg = last_attn.mean(dim=1).squeeze(0)  # (T, T)
        cls_to_all = attn_avg[0]  # attention distribution from CLS
        # Normalize for pretty printing
        cls_to_all = (cls_to_all / (cls_to_all.max() + 1e-9)).cpu().tolist()

        print("Attention from [CLS] to tokens (last layer, head-avg):")
        bars = []
        for tok, w in zip(tokens, cls_to_all):
            bar = "█" * int(w * 20)
            bars.append(f"{tok:>10s}  {bar}")
        print("\n".join(bars))

# -----------------------------
# 8) Train & Run
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build vocab
    vocab = build_vocab(RAW_DATA, min_freq=1)

    # Split train/val (simple)
    data = RAW_DATA[:]
    random.shuffle(data)
    split = int(0.85 * len(data))
    train_pairs = data[:split]
    val_pairs = data[split:]

    model = MiniTransformerClassifier(
        vocab_size=len(vocab),
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        num_classes=2,
        max_len=args.max_len,
        dropout=0.1
    )

    print(f"Vocab size: {len(vocab)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    train(model, train_pairs, vocab, epochs=args.epochs, max_len=args.max_len, batch_size=8, lr=args.lr, device='cpu')

    # Quick validation accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for text, label in val_pairs:
            ids = encode_sentence(text, vocab, args.max_len)
            x = torch.tensor([ids], dtype=torch.long)
            pad_mask = (x == vocab[PAD])
            logits = model(x, key_padding_mask=pad_mask, return_attn=False)
            pred = int(torch.argmax(logits, dim=-1).item())
            correct += int(pred == label)
    if len(val_pairs) > 0:
        print(f"Validation accuracy: {correct}/{len(val_pairs)} = {correct/len(val_pairs):.2f}")

    # Demo on test sentences with attention print
    for s in TEST_SENTENCES:
        print_attention_for_sentence(model, s, vocab, args.max_len, device="cpu")


if __name__ == "__main__":
    main()
