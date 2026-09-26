# Capstone Project 03 — Understand Transformers (Implement a Mini-GPT)

> **Deliverable:** A tiny GPT-style character-level language model, ~1M parameters, trained on the works of Shakespeare (or any text corpus you like). Implemented in PyTorch from scratch — every submodule visible: token embedding, positional encoding, multi-head self-attention, MLP block, residual + layer-norm. Generates readable Shakespeare-ish text after a few minutes on a laptop.
>
> **Time:** 6–8 hours; understanding is more important than novelty.
>
> **What you'll be able to say:** "I built a GPT from scratch. I understand what Q, K, V are, why we divide by √d_k, what causal masking does, why residual connections matter for training stability, and what layer norm does that batch norm can't."

---

## 1. Project Overview

Follow Karpathy's nanoGPT / "Let's build GPT" walkthrough, but write every piece yourself with heavy annotation. Train on `tinyshakespeare.txt` (1MB, character-level).

We build up in five stages:
1. Bigram model (baseline)
2. Add position embedding
3. Add single-head self-attention
4. Add multi-head + MLP + residuals + layer norm = one transformer block
5. Stack N blocks = GPT

### Architecture
![IMG-CAP03-01](/7%20-%20Projects/images/IMG-CAP03-01.jpg)


### Prerequisites

- Python 3.11+, PyTorch
- A GPU is helpful (Colab free tier is fine); CPU works but slower
- `tinyshakespeare.txt` from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

---

## 2. Step-by-Step Build

### Step 1 — Data & tokenizer (character-level)

```python
"""minigpt.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download once:
# curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = Path("input.txt").read_text()
chars = sorted(set(text))
vocab_size = len(chars)  # ~65 for Shakespeare
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

**Why character-level?** Simplest possible tokenizer. Real GPTs use BPE tokens (~50k pieces of subwords), but the math is identical.

### Step 2 — Batching function

```python
BLOCK_SIZE = 128  # context length
BATCH_SIZE = 64

def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)
```

**Why the offset?** The model predicts the next character at every position. If input is `"the ca"`, target is `"he cat"`. Every position in the sequence contributes a training signal — this is why transformers train so efficiently.

### Step 3 — Bigram baseline

```python
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Just a lookup: given char i, what's the next-char distribution?
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding(idx)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
```

Trains to loss ~2.5 (baseline). Generated text is gibberish but has right letter frequencies.

### Step 4 — Single-Head Self-Attention (the heart of it)

```python
class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, head_size, block_size):
        super().__init__()
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        # Register as buffer — moves with .to(device) but isn't a parameter
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)                         # (B, T, head_size)
        Q = self.query(x)                       # (B, T, head_size)
        V = self.value(x)                       # (B, T, head_size)

        # Attention scores: (B, T, T)
        scores = Q @ K.transpose(-2, -1) / (C ** 0.5)

        # Causal mask — position i can only attend to positions ≤ i
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)     # rows sum to 1

        return weights @ V                       # (B, T, head_size)
```

**The mental model:**
- **Query** = "what am I looking for?"
- **Key** = "what do I offer as a match?"
- **Value** = "what do I actually contribute if you pick me?"
- Q·K^T = compatibility scores; softmax turns into weights; · V is the weighted sum.

**Why divide by √d_k?** As `d_k` grows, dot products grow proportionally, pushing softmax into saturation (nearly one-hot). Dividing by √d_k keeps the pre-softmax scores in a "soft" regime where gradients flow well. Without it, deeper transformers train poorly.

**Why the causal mask?** For a language model, position `i` predicting the next token must not "see" the future — otherwise it's just copying. The lower-triangular mask forces attention to be autoregressive.

### Step 5 — Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(d_model, head_size, block_size)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_size, d_model)

    def forward(self, x):
        # Concat heads along the last dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
```

**Why multi-head?** Each head can specialize: one attends to previous-noun agreement, another to punctuation, another to long-range topic. Concatenating gives the next layer a richer representation than one head could produce. The final `proj` mixes heads back into `d_model`.

Production transformers implement this as a single matmul with reshaping (much faster), but the semantics are identical.

### Step 6 — MLP block

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_mult * d_model),
            nn.GELU(),
            nn.Linear(hidden_mult * d_model, d_model),
        )
    def forward(self, x):
        return self.net(x)
```

**Why 4× expansion?** Standard from the original paper; empirically the sweet spot. The MLP is where the model does per-token nonlinear processing on the aggregated attention output.

**Why GELU over ReLU?** Smooth (differentiable everywhere), retains small negative activations rather than hard-zeroing them. Empirically better for transformers.

### Step 7 — Transformer block (Attention + MLP + LayerNorm + Residuals)

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        head_size = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_heads, d_model, head_size, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x):
        # Pre-norm architecture: x + f(norm(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

**Why residual connections?** Deep networks suffer from **degradation** — beyond ~10 layers, training loss stops decreasing not from overfitting but because gradients don't flow. Residual (`x + f(x)`) means at initialization each block computes the identity plus a small perturbation; gradients flow directly along the residual path. Enables 50+ layer nets.

**Why LayerNorm not BatchNorm?** BatchNorm normalizes across the batch dimension — problematic for language (variable sequence lengths, non-IID batches). LayerNorm normalizes across the feature dimension per-token — batch-independent and works cleanly with the causal mask.

**Why pre-norm vs post-norm?** Original paper used post-norm (`norm(x + f(x))`). Modern practice is pre-norm — more stable training, works without careful warm-up. Almost every modern LLM uses pre-norm.

### Step 8 — Full GPT

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_heads=6, n_layers=6, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embed(idx)                         # (B, T, d_model)
        pos = self.pos_embed(torch.arange(T, device=idx.device))  # (T, d_model)
        x = tok + pos                                        # (B, T, d_model)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # truncate to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
```

**Why positional embedding?** Attention itself is permutation-invariant — it treats tokens as a set. Adding a learned position vector gives the model a sense of "this token is 3rd in the sequence" so it can learn word order. Modern GPTs use RoPE (rotary) or ALiBi; learned absolute is the original.

**Why `token_embed` and `head` weights aren't tied here?** Weight tying (same matrix for input embedding and output projection) saves parameters and slightly helps small models. It's optional; add if you want. GPT-2 uses tying.

### Step 9 — Training loop

```python
def estimate_loss(model, iters=100):
    model.eval()
    losses = {"train": [], "val": []}
    for split in losses:
        for _ in range(iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[split].append(loss.item())
    model.train()
    return {k: sum(v)/len(v) for k, v in losses.items()}

model = MiniGPT(vocab_size).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

for step in range(5000):
    X, Y = get_batch("train")
    _, loss = model(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % 500 == 0:
        est = estimate_loss(model)
        print(f"step {step}: train {est['train']:.3f}, val {est['val']:.3f}")

# Generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
```

Expected losses: train ~1.5, val ~1.7 after 5000 steps (~10 min on a laptop GPU).

Sample generation after training:

```
DUKE VINCENTIO:
Well, but he is a plague to me.

FLORIZEL:
I am ashamed of it, and yet I would
Not have him hate me for my hatred.
```

It's not writing coherent plays, but it's clearly learned Shakespeare's line structure, capitalized character names, iambic-ish rhythm. That's a ~1M-parameter model doing that from scratch.

---

## 3. What Each Component Actually Does

`[IMG-CAP03-02]` — *Prompt: A visualization of self-attention weights during one forward pass. Show a 20×20 grid representing the attention weight matrix for one head on the sentence "The cat sat on the mat because it was tired." Cells are colored by weight magnitude (dark = low, bright yellow = high). Highlight that "it" (row 8) has bright cells at "cat" (col 1) — the head has learned coreference. Diagonal shows self-attention. Lower triangle only (causal mask); upper triangle is dark. Include a colorbar and title "Head 3, Layer 4: coreference-like attention pattern."*

| Component | Role |
|-----------|------|
| **Token embedding** | Map discrete token IDs to dense vectors |
| **Position embedding** | Inject sequence order (attention is permutation-invariant otherwise) |
| **Multi-head attention** | Each token gathers information from other positions weighted by learned relevance |
| **Causal mask** | Enforce autoregressive: position i can only see positions ≤ i |
| **√d_k scaling** | Keep softmax pre-activations from saturating |
| **MLP block** | Per-token nonlinear processing on aggregated info |
| **LayerNorm** | Stabilize activations without depending on batch stats |
| **Residual connections** | Enable gradient flow through many layers |
| **Final projection** | Map d_model back to vocab_size for next-token prediction |

---

## 4. The KV Cache (What Makes Inference Fast)

Not built into this project but critical to know:

During autoregressive generation, at each step you re-compute attention. Naive: O(T²) work per step, O(T³) total. **KV cache**: keys and values for past tokens don't change; cache them and only compute Q for the new token. Reduces to O(T) per step, O(T²) total — the difference between a viable chatbot and an unresponsive one.

```python
# Sketch (not integrated with the above)
def generate_with_kv_cache(model, idx, max_new_tokens):
    # Run once to populate cache
    logits, cache = model(idx, use_cache=True)
    for _ in range(max_new_tokens):
        next_idx = sample_from(logits[:, -1, :])
        idx = torch.cat([idx, next_idx], dim=1)
        # Only forward the new token, past K/V from cache
        logits, cache = model(next_idx, cache=cache, use_cache=True)
    return idx
```

---

## 5. Extensions

- **Byte-Pair Encoding tokenizer** — replace char-level with a small BPE using tiktoken. Model becomes vastly more efficient.
- **RoPE positional encoding** — replace learned position embeddings with rotary; the modern default.
- **Grouped-query attention (GQA)** — fewer K/V heads than Q heads; standard in modern models (Llama, Claude, Gemma).
- **Flash Attention** — memory-efficient attention; requires GPU with proper library.
- **Fine-tune on your own text corpus** — a blog, an email archive, chat logs. Watch it learn your idioms.
- **Scale up** — 6 layers → 12; d_model 192 → 384; watch quality improve.

---

## 6. Common Confusions

- **"Attention is what makes transformers powerful."** Partial. Residuals, LayerNorm, and MLPs matter equally. Attention lets tokens *communicate*; MLPs let them *think per-token*.
- **"Q, K, V come from three separate inputs."** No — same input `x` projected through three different learned matrices.
- **"Positional embedding is added, not concatenated."** Yes, added. Concatenation would waste dimensions.
- **"Self-attention is O(n) per token."** Naive attention is O(n²) per token because each token looks at all others. Flash Attention makes it memory-linear but the compute is still O(n²).
- **"Causal mask is only during training."** It's used during both training and generation.

---

## 7. Interview Talking Points

Common transformer questions:

1. **"Walk me through self-attention."** Explain Q, K, V, dot product for compatibility, softmax over past positions, weighted sum of V. Mention √d_k scaling and causal mask.
2. **"Why LayerNorm and not BatchNorm?"** Batch stats don't work well for variable-length sequences; LayerNorm is batch-independent.
3. **"What does the residual connection do?"** Enables gradient flow through deep stacks; at init, each block is near-identity so training starts stable.
4. **"Why multi-head?"** Different heads specialize in different attention patterns (syntax, coreference, long-range topic).
5. **"What's the KV cache?"** During generation, past keys/values don't change; caching them turns O(n²) per-step into O(n).
6. **"What's positional encoding for?"** Attention is permutation-invariant; we must inject order.
7. **"Explain how a modern LLM differs from what you built."** Bigger (billions of params), BPE tokenizer, RoPE, GQA, Flash Attention, mixture-of-experts optional, RLHF/DPO post-training, safety fine-tuning.

---

## 8. References

- Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out" (YouTube; the definitive walkthrough)
- Vaswani et al., "Attention Is All You Need" (2017) — the original paper; short and worth reading
- Karpathy's nanoGPT — https://github.com/karpathy/nanoGPT
- 3Blue1Brown "Attention in Transformers" video (excellent geometric intuition)
- Lilian Weng — "The Transformer Family Version 2.0" (blog post overview)
- FlashAttention paper — Dao et al., 2022
