# Transformers — Master Study Guide

> **Track:** Deep Learning · **Module:** 10
> **Prerequisites:** Modules 01–09 (especially 09 — Attention).
> **Time budget:** ~12–15 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** The transformer is the architecture that ate the world. Since "Attention Is All You Need" (Vaswani et al., 2017), transformers have replaced:

- RNNs for sequence modeling.
- CNNs for image classification (Vision Transformer).
- Task-specific models for translation, summarization, question answering.
- Handcrafted features for tabular data (TabTransformer, FT-Transformer).
- Everything for LLMs (GPT-3/4/5, Claude, Gemini, LLaMA, Mixtral, and every foundation model).

The transformer is a **general-purpose sequence-to-sequence architecture** built from stacked self-attention + feedforward blocks with residuals and normalization. Its scalability (parallelism, long context, transferable pretraining) is why we're in the era of foundation models.

**Fundamental principles you must own:**

1. **The transformer block** = multi-head self-attention + feedforward network, each with a residual and normalization.
2. **Depth (layers) + width (model dim) + heads + FFN size** — the four capacity knobs.
3. **Encoder** processes the input bidirectionally (BERT-style); **decoder** generates autoregressively (GPT-style); **encoder-decoder** does both (T5, original transformer).
4. **Pre-norm vs post-norm** — pre-norm (LN before sub-layer) is the modern default; more stable.
5. **The FFN is bigger than attention** — usually 4× wider than $d_{\text{model}}$, holds ~2/3 of the parameters.
6. **Scaling laws** show performance improves predictably with more parameters, more data, more compute.

If you retain nothing else: **a transformer is stacks of (attention + FFN) blocks with residuals and normalization — a scalable, parallelizable sequence architecture.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Transformer Block

The atomic unit of a transformer, in its **pre-norm** form (modern standard):

$$\mathbf{x}' = \mathbf{x} + \text{MultiHeadSelfAttn}(\text{LayerNorm}(\mathbf{x}))$$
$$\mathbf{y} = \mathbf{x}' + \text{FFN}(\text{LayerNorm}(\mathbf{x}'))$$

Where:
- **LayerNorm** is applied **before** each sub-layer (pre-norm).
- **Residual connection** wraps each sub-layer — this is what makes training stable at depth.
- **FFN** is the position-wise feedforward network (details below).

**Post-norm** (original transformer, "Attention Is All You Need"):

$$\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{MultiHeadSelfAttn}(\mathbf{x}))$$
$$\mathbf{y} = \text{LayerNorm}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))$$

Pre-norm is empirically more stable — training doesn't need as much warmup, deeper models are more trainable. Almost every modern LLM (GPT-2/3/4, LLaMA) uses pre-norm.

**The FFN sub-layer** (per-position, applied identically to every token):

$$\text{FFN}(\mathbf{x}) = W_2 \, \text{GELU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

Where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.

Standard: $d_{\text{ff}} = 4 \cdot d_{\text{model}}$. This is where most of the parameters live.

**Modern FFN variants:**

- **GELU** activation (BERT, GPT-2/3).
- **SwiGLU** or **GeGLU** — used in LLaMA, PaLM, T5-v1.1:

  $$\text{SwiGLU}(\mathbf{x}) = (W_1 \mathbf{x} \odot \text{SiLU}(W_g \mathbf{x})) W_2$$

  Adds a third projection $W_g$, elementwise-gated. Empirically slightly better.

**Parameter count per block** (approx, ignoring biases):

- **Attention:** $4 \cdot d_{\text{model}}^2$ (three projections $Q, K, V$ + output $W_O$).
- **FFN:** $2 \cdot d_{\text{model}} \cdot d_{\text{ff}} = 8 d_{\text{model}}^2$ (with $d_{\text{ff}} = 4 d_{\text{model}}$).
- **Total per block:** $\approx 12 d_{\text{model}}^2$.

For an $L$-layer model, **total transformer parameters $\approx 12 L d_{\text{model}}^2$** — you can quickly estimate model size in your head.

For GPT-3 175B: $L = 96$, $d_{\text{model}} = 12288$, so $12 \cdot 96 \cdot 12288^2 \approx 174$ billion. Add embeddings and heads: 175 B. Rough math works.

---

### 2.2 Encoder, Decoder, Encoder-Decoder

Three variants of the transformer architecture, each suited to different tasks:

**Encoder-only** (BERT-style):
- Bidirectional self-attention — every token attends to every other token in both directions.
- Trained with **masked language modeling** (predict masked tokens from context).
- Good for classification, embeddings, retrieval.
- Examples: BERT, RoBERTa, DeBERTa, sentence-transformers.

**Decoder-only** (GPT-style):
- Causal (autoregressive) self-attention — each token only attends to previous tokens.
- Trained with **next-token prediction** (autoregressive language modeling).
- Good for generation, chat, code, few-shot learning.
- Examples: GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Claude.

**Encoder-decoder** (T5, original transformer):
- Encoder processes input bidirectionally.
- Decoder generates output autoregressively, cross-attending to encoder outputs.
- Good for sequence-to-sequence tasks — translation, summarization.
- Examples: original transformer, T5, BART, Pegasus, Whisper.

**Which is winning?** Decoder-only. It turns out that big enough decoder-only models trained on next-token prediction can do everything — classification, translation, retrieval — via prompting. This is why modern LLMs are dominantly decoder-only.

**Cross-attention** (in encoder-decoder):

$$Q_{\text{dec}} = W_Q \mathbf{y}, \quad K = W_K \mathbf{H}^{\text{enc}}, \quad V = W_V \mathbf{H}^{\text{enc}}$$

Decoder positions query encoder positions.

---

### 2.3 Building a Transformer Block in PyTorch

**A skeleton implementation** for clarity (real production code uses fused kernels):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.causal = causal

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # (B, H, T, d_head)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal
        )                                                     # (B, H, T, d_head)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(attn)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, causal=False):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

**A minimal GPT-like model:**

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, causal=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Optional: tie weights between embedding and head
        self.head.weight = self.tok_emb.weight

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)          # (B, T, vocab_size)
        return logits
```

**Training loop for language modeling:**

```python
tokens = ...     # (B, T)
logits = model(tokens[:, :-1])                   # predict tokens[:, 1:]
loss   = F.cross_entropy(logits.view(-1, V), tokens[:, 1:].reshape(-1))
loss.backward()
optimizer.step()
```

That's Karpathy's `nanoGPT` in ~100 lines. You can train tiny GPT models on a single GPU in an evening.

---

### 2.4 Positional Encoding, Normalization, and Key Design Choices

**Position** — see Module 09.5. Modern choices:
- **Sinusoidal** — original transformer.
- **Learned position embeddings** — BERT, GPT-2.
- **RoPE** (Rotary) — LLaMA-family, GPT-NeoX. Most common in 2023+ models.
- **ALiBi** — BLOOM, MPT. Extrapolates well to longer sequences.

**Normalization:**
- **LayerNorm** — standard. Applied per-token.
- **RMSNorm** — a simpler variant that skips the mean-subtraction step; LLaMA uses it. Faster with equivalent quality.
- **Pre-norm** vs post-norm — pre-norm is the modern default.

**Weight tying** — share the input embedding matrix with the output projection matrix. Saves parameters ($V \cdot d_{\text{model}}$ can be huge for large vocab), sometimes improves quality.

**Bias in linear layers:**
- Modern LLMs (LLaMA, PaLM) drop biases in most linear layers — small savings, marginal quality difference.

**Attention variants:**

- **Multi-Query Attention (MQA)** — one K and V per layer, shared across heads. Speeds up inference dramatically. Used by PaLM, Falcon.
- **Grouped-Query Attention (GQA)** — a middle ground: groups of heads share K and V. Used by LLaMA-2, Mixtral. Best speed/quality trade-off.

**Sparse and efficient attention:**

- **Sliding-window attention** — attend only to a local window (Longformer, Mistral). Enables long context.
- **Global + local attention** — some tokens (special ones) attend globally; others locally.
- **Flash Attention** — an I/O-optimized exact attention implementation. Same math, much faster & less memory.
- **Linear attention** — approximates softmax with $O(n)$ cost (Performer, RWKV). Trades exactness for speed.

**Mixture of Experts (MoE)** — replace the FFN with a sparse gating: only a few "expert" FFNs are active per token. Enables trillion-parameter models with sub-linear compute. Used by Mixtral, GPT-4 (reportedly), Gemini.

---

### 2.5 Scaling, Context Length, and Modern Recipes

**Scaling laws** (Kaplan et al., 2020; Hoffmann et al., 2022 — "Chinchilla"):

- Loss decreases as a power law in each of (parameters, data, compute).
- For a fixed compute budget, there's an optimal balance of parameters and training data (Chinchilla: roughly 20 tokens per parameter).
- Overtraining a small model beats undertraining a big one.

**Modern LLM training recipe (2024–2026 era):**

1. **Pretraining** — trillions of tokens of internet text, code, books.
2. **Instruction fine-tuning** — supervised fine-tuning on human-labeled instruction-response pairs.
3. **RLHF or DPO** — align model to human preferences via reinforcement learning or direct preference optimization.

More on this in Module 11.

**Long context.** Getting attention to work well over 100k+ tokens is a research-and-engineering challenge:

- **Positional encoding extrapolation** — RoPE with position interpolation, YaRN, NTK scaling.
- **Sparse attention** — sliding windows, chunked attention.
- **Memory-efficient attention** — Flash Attention, ring attention.
- **Retrieval augmentation (RAG)** — for very long context, retrieve relevant chunks instead of feeding everything.

**Inference efficiency:**

- **KV cache** — cache K, V from previous tokens during autoregressive generation. Turns $O(n^2)$ per-token compute into $O(n)$.
- **Quantization** — int8/int4 weights. 4× or 8× smaller memory, minimal quality loss.
- **Speculative decoding** — a small model proposes tokens; a large model verifies. Faster with same quality.
- **Batching** — server many requests in parallel (dynamic batching).

**Training efficiency:**

- **Data parallelism** — same model on N GPUs, each with a batch shard.
- **Tensor parallelism** — split individual layers across GPUs.
- **Pipeline parallelism** — different layers on different GPUs, forward/backward pipelined.
- **ZeRO / FSDP** — shard optimizer states, gradients, and parameters across GPUs.

Frontier LLM training uses all four simultaneously across thousands of GPUs.

---

## 3. Mental Models & Analogies

### 3.1 The "Message Passing on a Complete Graph" Model

At each transformer block, every token sends a **message** to every other token. The message has three parts:

- **Query** — "here's what I'm looking for."
- **Key** — "here's what I've got to offer."
- **Value** — "here's the actual payload if you want to consume mine."

Attention says: for each token, average the payloads from all other tokens, weighting by how well their offer matches your query.

The **FFN** is a per-token post-processor — after receiving your weighted mail, you sit down and privately update your state via a nonlinear function.

Then the whole thing repeats — token states now include information from every other token. In the next layer, the messages become richer because each token's state is now a summary of the whole sequence.

- **1 layer** = every token can see every other token once.
- **L layers** = every token can see every other token through $L$ rounds of message-passing, allowing for compositional reasoning.
- **Attention as message passing** frames transformers as **graph neural networks on a fully-connected graph** — a useful abstraction.

### 3.2 The "Working Memory" Model

Think of a transformer processing a document as a **worker with an enormous working memory**:

- All tokens are simultaneously in memory (unlike an RNN which forgets as it reads).
- At each processing step (layer), the worker updates their view of each token by consulting all other tokens (attention) and thinking privately about the result (FFN).
- After many steps, each token's representation has absorbed context from the entire document.
- The **residual connection** is a "return to your notes" — you always have your previous thoughts, and you add new insights on top rather than replacing.

**Why residuals + LayerNorm are essential**: without residuals, deep transformers lose gradient signal and fail to train. Without LayerNorm, activations drift into pathological ranges and training destabilizes.

**Why attention scales so well**: unlike RNNs (which must be processed step-by-step), attention is a single big matrix multiply — a **parallelizable** operation. Modern GPUs love matrix multiplies.

![IMG-TFM-01](/3%20—%20Deep%20Learning/images/IMG-TFM-01.jpg)

> **Caption:** A pre-norm transformer block: attention and FFN each wrapped by LayerNorm and a residual connection.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Transformer Is Just Attention"

Attention is the star, but a transformer is **attention + FFN + residuals + LayerNorm** — remove any component and it fails to train or performs badly. The FFN is where 2/3 of the parameters live; residuals give gradients a highway; LayerNorm keeps activations in stable ranges. Andrej Karpathy's "attention is a communication mechanism; FFN is a computation mechanism" is the right framing.

### 4.2 "Bigger Transformers Are Always Better"

Chinchilla's insight (Hoffmann et al., 2022): for a fixed compute budget, **smaller models trained on more data outperform larger under-trained ones**. LLaMA showed you can beat GPT-3 (175B) with a 65B model trained on more tokens. Naïve "make it bigger" is inefficient; you need to also scale data proportionally.

### 4.3 "Transformers Can Handle Any Sequence Length"

Vanilla attention is $O(n^2)$ in both time and memory. On a 40 GB A100, a naive transformer runs out of memory somewhere around $n = 32k$–$64k$ depending on model size. Extending to 100k+ tokens requires:

- **Sparse or windowed attention**.
- **Flash Attention** (memory-efficient exact attention).
- **Position encoding that extrapolates** (RoPE with tricks, ALiBi).
- **Retrieval augmentation** — for very long documents, retrieve relevant chunks.

Long context is a solvable engineering problem, but it's not free.

---

## 5. Self-Assessment Bank (Transformers)

### Questions

**Q1 (Short answer).** Write the two-line update for a pre-norm transformer block.

**Q2 (Multiple choice).** In a transformer block, the FFN operates:
A. Across the sequence dimension (mixes tokens).
B. Per-position, identically applied to each token independently.
C. Only on the [CLS] token.
D. Only during training.

**Q3 (Short answer).** Approximately how many parameters does a transformer block have in terms of $d_{\text{model}}$?

**Q4 (Multiple choice).** Which best describes the differences between encoder-only, decoder-only, and encoder-decoder transformers?
A. All three are architecturally identical.
B. Encoder uses bidirectional self-attention; decoder uses causal (masked) self-attention; encoder-decoder combines both with cross-attention.
C. Encoders don't have FFNs.
D. Decoders don't have attention.

**Q5 (Short answer).** Explain the difference between pre-norm and post-norm, and which is now standard for LLMs.

**Q6 (Multiple choice).** The main advantage of transformers over RNNs for training is:
A. Fewer parameters.
B. Better long-range modeling combined with full parallelism across the sequence.
C. Cheaper inference.
D. Simpler code.

**Q7 (Short answer).** Why is the KV cache important for autoregressive transformer inference?

**Q8 (Multiple choice).** Grouped-Query Attention (GQA):
A. Increases number of heads.
B. Reduces memory for K, V during inference by grouping heads to share KV, at similar quality to full MHA.
C. Removes attention.
D. Only works for training.

**Q9 (Short answer).** In one paragraph, summarize why "next-token prediction" is enough of a training objective to produce a model that can answer questions, translate, and generate code.

**Q10 (Multiple choice).** Chinchilla scaling laws (Hoffmann et al., 2022) suggest that:
A. Bigger models always win.
B. For a fixed compute budget, there's an optimal balance of model size and training tokens (roughly 20 tokens per parameter).
C. Data size doesn't matter.
D. Depth is more important than width.

---

### Answer Key & Detailed Explanations

**A1.** $\mathbf{x}' = \mathbf{x} + \text{MHSA}(\text{LN}(\mathbf{x}))$; $\mathbf{y} = \mathbf{x}' + \text{FFN}(\text{LN}(\mathbf{x}'))$. LayerNorm before each sub-layer, residual from before-LN input.

**A2. B.** The FFN is applied identically and independently at every sequence position — a per-token nonlinear transform. It doesn't mix information across tokens (that's attention's job). The FFN's per-position nature is what makes it parameter-efficient.

**A3.** ≈ $12 d_{\text{model}}^2$: attention has $4 d_{\text{model}}^2$ (Q/K/V/output projections), FFN has $8 d_{\text{model}}^2$ ($d_{\text{ff}} = 4 d_{\text{model}}$, two matrices). Total transformer ≈ $12 L d_{\text{model}}^2$ where $L$ is the number of layers.

**A4. B.** Encoder-only (BERT): bidirectional self-attention; good for classification and embeddings. Decoder-only (GPT): causal self-attention; good for generation. Encoder-decoder (T5, original transformer): encoder handles the input bidirectionally, decoder generates autoregressively with cross-attention to encoder outputs.

**A5.** **Post-norm** applies LayerNorm *after* the sub-layer and residual: $\text{LN}(x + \text{Sublayer}(x))$. **Pre-norm** applies LayerNorm *before* the sub-layer, keeping the residual outside: $x + \text{Sublayer}(\text{LN}(x))$. Pre-norm is more stable — the residual path is unnormalized, giving gradients a clean highway. Every major LLM (GPT-2/3/4, LLaMA, Mistral) uses pre-norm; original transformer (2017) used post-norm.

**A6. B.** Transformers handle long-range dependencies via direct attention (any token can look at any other in one step), and their forward and backward passes are fully parallelized across the sequence dimension. RNNs must be processed sequentially — token $t$ depends on token $t-1$. This makes transformers trainable on massive datasets that would be impractical for RNNs.

**A7.** During autoregressive generation, tokens are produced one at a time. Without caching, each new token requires re-computing all keys and values for every previous token — quadratic cost per token, cubic cost for a full sequence. The **KV cache** stores K and V for all previous tokens; when generating token $t+1$, only its own Q/K/V need to be computed and appended. This drops per-token inference cost from $O(t^2)$ to $O(t)$, and total generation from $O(n^3)$ to $O(n^2)$. Essential for real-time LLM inference.

**A8. B.** GQA groups $h$ heads into $g$ groups; each group shares one K and V. Reduces KV cache memory by $h/g$ (typically 8× or so), speeding up inference. LLaMA-2 70B uses GQA with $g = 8$. Quality is close to full multi-head attention.

**A9.** Next-token prediction on a massive corpus of natural text and code forces the model to learn everything that helps predict the next token: syntax, grammar, world knowledge, factual associations, code structure, question-answer patterns, translation pairs (both languages appear in the training set), etc. Because the training data contains examples of every task humans do in text (Q&A, translations, code generation, chain-of-thought reasoning), a sufficiently large model trained to predict next tokens implicitly learns all those tasks. Prompt engineering then elicits the relevant capability. This is the "language model is a task-agnostic learner" insight of GPT-2/3 papers.

**A10. B.** Chinchilla's finding: for a fixed compute budget, ~20 tokens per parameter is optimal. GPT-3 (175B params, 300B tokens = 1.7 tokens/param) was severely undertrained. Chinchilla (70B params, 1.4T tokens = 20 tokens/param) matched or beat GPT-3 with much less compute per inference. This changed the field's approach to LLM training.

---

## 6. Practice Prompts

1. **NanoGPT.** Build Karpathy-style nanoGPT: single-file GPT-like model, trained on tinyshakespeare. ~200 lines of PyTorch.
2. **From-scratch transformer block.** Implement the transformer block above without any `nn.MultiheadAttention` or `nn.TransformerEncoderLayer` — plain matmuls only. Verify on a small task.
3. **Pre-norm vs post-norm.** Train two small transformers, one pre-norm and one post-norm, on the same task. Compare training stability without warmup.
4. **KV cache demo.** Modify your GPT to use a KV cache during generation. Benchmark tokens-per-second vs no cache for a 200-token generation.
5. **Position encoding comparison.** Implement sinusoidal, learned, and RoPE. Train identical models on WikiText. Compare test perplexity and extrapolation to longer sequences.

---

## 7. References

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017).
- Devlin et al., ["BERT: Pre-training of Deep Bidirectional Transformers"](https://arxiv.org/abs/1810.04805) (2018).
- Radford et al., ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) — GPT-2.
- Brown et al., ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (2020) — GPT-3.
- Touvron et al., ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971) (2023).
- Hoffmann et al., ["Training Compute-Optimal Large Language Models"](https://arxiv.org/abs/2203.15556) (2022) — Chinchilla.
- Andrej Karpathy, ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT).
