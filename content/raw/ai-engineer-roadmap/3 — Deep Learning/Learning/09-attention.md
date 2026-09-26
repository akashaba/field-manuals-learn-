# Attention — Master Study Guide

> **Track:** Deep Learning · **Module:** 09
> **Prerequisites:** Modules 01–08.
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Attention is the single most important architectural idea of the last decade. It solved the RNN's biggest weakness (the fixed-size context bottleneck) and became the foundation for every modern large model — BERT, GPT, T5, LLaMA, CLIP, Whisper, Stable Diffusion, Sora. Once you truly grasp attention, transformers, LLMs, and multimodal models stop feeling mysterious.

The core idea is deceptively simple: **at each position, look at all other positions, and compute a weighted average of their values, where the weights come from how relevant each other position is to me.** That's it. The rest is engineering.

**Fundamental principles you must own:**

1. **Attention is a weighted average.** The weights are learned from similarity scores.
2. **Query, Key, Value** — three projections of the input, each playing a specific role.
3. **Scaled dot-product attention** is the specific form used in transformers.
4. **Self-attention** = queries, keys, and values all come from the same source.
5. **Multi-head attention** runs several attention computations in parallel, each with its own subspace.
6. **Attention is $O(n^2)$ in sequence length** — the main scaling bottleneck for transformers.

If you retain nothing else: **attention is a learned lookup that lets every position pull information from every other position.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 From "Fixed Context" to "Weighted Lookup"

**Original attention** (Bahdanau et al., 2014) — solved the RNN encoder-decoder bottleneck. Instead of the decoder using only the encoder's final hidden state, it looks at all encoder hidden states and computes a weighted sum:

$$\mathbf{c}_t = \sum_{i=1}^T \alpha_{ti} \mathbf{h}_i^{\text{enc}}$$

Where $\alpha_{ti}$ is the attention weight — how much the decoder at step $t$ attends to encoder step $i$. Weights sum to 1 (softmax over positions) and depend on the current decoder state.

**Luong attention** (2015) — variants: dot, general, concat scoring functions.

**Self-attention** (Vaswani et al., 2017, "Attention Is All You Need") — throws out the RNN entirely; uses attention over the sequence itself, applied in parallel to every position. This is the transformer's key innovation.

**The generic attention recipe:**

1. From each input position, produce a **query** $\mathbf{q}$, a **key** $\mathbf{k}$, and a **value** $\mathbf{v}$.
2. Compute a **compatibility score** between every query and every key.
3. Softmax the scores to get **attention weights**.
4. Output = weighted sum of values.

The whole rest of this module is filling in that recipe.

---

### 2.2 Scaled Dot-Product Attention

The most important formula in modern ML:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Where:
- **$Q \in \mathbb{R}^{n \times d_k}$** — matrix of $n$ query vectors, each of dimension $d_k$.
- **$K \in \mathbb{R}^{m \times d_k}$** — $m$ key vectors.
- **$V \in \mathbb{R}^{m \times d_v}$** — $m$ value vectors.
- **$d_k$** = the key dimension.

Step by step:

1. **Scores** $S = QK^\top \in \mathbb{R}^{n \times m}$. Each entry $S_{ij}$ is the dot product $\mathbf{q}_i \cdot \mathbf{k}_j$ — a similarity score.
2. **Scale** by $\sqrt{d_k}$ — prevents dot products from growing too large for high-dimensional $Q, K$, which would push softmax into saturating regions.
3. **Softmax** across the key dimension (per query row) — probabilities summing to 1.
4. **Weighted sum** — the output at position $i$ is $\sum_j \text{softmax}_j \cdot V_j$.

**Output shape** = $(n, d_v)$. Every input query gets an output vector.

**The scaling factor $\sqrt{d_k}$.** For $d_k = 64$ and random unit-variance $Q, K$, $QK^\top$ elements have variance $d_k = 64$; standard deviation 8. That would push softmax into extreme concentration on the top few keys, killing gradients elsewhere. Dividing by $\sqrt{d_k}$ keeps scores unit-variance and softmax well-behaved.

**Self-attention** = $Q, K, V$ all derived from the same input $X$ by three learned linear projections:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$$\text{SelfAttention}(X) = \text{softmax}\!\left(\frac{X W_Q W_K^\top X^\top}{\sqrt{d_k}}\right) X W_V$$

Every position ends up as a weighted mixture of every other position's value vector, where the mixing coefficient is a learned similarity.

**Cross-attention** = queries from one source (decoder's positions), keys and values from another (encoder's positions). Used in encoder-decoder transformers for translation.

---

### 2.3 Multi-Head Attention

One attention computation has one "pattern of attention." **Multi-head attention** runs $h$ attention computations in parallel, each with its own $W_Q, W_K, W_V$ subspaces:

$$\text{head}_i = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)})$$

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

Where:
- Each head has $d_k = d_v = d_{\text{model}} / h$ typically.
- $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ projects the concatenated heads back to the model dimension.

**Why multi-head:**

- **Different heads can attend to different patterns** — one head for syntactic dependencies, another for semantic relations, another for long-range references, etc.
- **Parameter-cost neutral** — each head is smaller, so total parameters are the same as one big head.
- **Empirically much better** than single-head with the same total dimension.

**Typical settings** — number of heads $h$ and per-head dim $d_k$:

- **BERT-base:** $h = 12$, $d_k = 64$, $d_{\text{model}} = 768$.
- **GPT-2 small:** $h = 12$, $d_k = 64$, $d_{\text{model}} = 768$.
- **GPT-3 175B:** $h = 96$, $d_k = 128$, $d_{\text{model}} = 12288$.

**PyTorch:** `nn.MultiheadAttention(embed_dim, num_heads)`. Or the newer `F.scaled_dot_product_attention` (uses Flash Attention when available).

---

### 2.4 Masking, Causality, and Padding

**Padding masks.** Sequences in a batch are padded to the same length; masked-out (padded) positions should get zero attention weight. Applied *before* softmax by setting corresponding score entries to $-\infty$:

$$S_{ij} = \begin{cases}\mathbf{q}_i \cdot \mathbf{k}_j & \text{key } j \text{ is real} \\ -\infty & \text{key } j \text{ is padding}\end{cases}$$

After softmax, $-\infty \to 0$; those positions contribute nothing.

**Causal (autoregressive) mask.** For a decoder generating one token at a time, each position must **only attend to previous positions** — no peeking at future tokens. Apply a lower-triangular mask:

$$S_{ij} = \begin{cases}\mathbf{q}_i \cdot \mathbf{k}_j & j \leq i \\ -\infty & j > i\end{cases}$$

This is what makes GPT autoregressive. During training, you can compute the loss for every position in parallel (thanks to the mask), and the model still cannot cheat. During inference, you generate one token at a time, and each new token attends to all previous ones.

**Cross-attention masks.** Encoder outputs may be padded; decoder queries attend to encoder keys/values with a padding mask.

**Attention mask as a tensor.** Standard convention:
- **Additive mask:** the mask is a matrix added to scores; $0$ for allowed, $-\infty$ for blocked. Common in PyTorch.
- **Boolean mask:** $\text{True}$ where allowed. Some APIs expect this.

---

### 2.5 Positional Encoding

Attention is **permutation-invariant** — shuffling the input rows produces shuffled output rows (with attention weights permuted). Nothing in the operation encodes position. But for sequences, **order matters**. So we add position information to the input:

**Sinusoidal positional encoding** (Vaswani et al., 2017) — deterministic:

$$\text{PE}(\text{pos}, 2i) = \sin(\text{pos} / 10000^{2i/d_{\text{model}}})$$
$$\text{PE}(\text{pos}, 2i+1) = \cos(\text{pos} / 10000^{2i/d_{\text{model}}})$$

- Different frequencies at different dimensions.
- Can extrapolate beyond training-time lengths (in principle).
- **Added** to input embeddings before the first attention layer.

**Learned positional embedding** — a learnable $\mathbb{R}^{L_{\max} \times d_{\text{model}}}$ table indexed by position. Simpler; standard in BERT and GPT-2.

**Rotary Position Embedding (RoPE)** (Su et al., 2021) — rotates query and key vectors by an angle proportional to position. **The modern default** in LLaMA, GPT-NeoX, PaLM, and most 2023+ open-weights LLMs.

- Applied inside the attention computation (not added to embeddings).
- Naturally encodes **relative** position.
- Better long-context extrapolation properties.

**ALiBi (Attention with Linear Biases)** (Press et al., 2021) — instead of embeddings, add a linear penalty to attention scores based on distance:

$$S_{ij} \leftarrow S_{ij} - m \cdot |i - j|$$

Great extrapolation, simple to implement. Used in BLOOM, MPT.

**Position matters more than most people realize.** Poor positional encoding is a leading cause of long-context failure. Modern LLMs use RoPE with tricks (like YaRN, position interpolation) to extend context.

---

## 3. Mental Models & Analogies

### 3.1 The "Database Lookup" Model

Attention is a **soft, differentiable database lookup**:

- Every position submits a **query**: "I'm interested in X."
- Every position advertises a **key**: "I'm about Y."
- Every position holds a **value**: "if you match me, here's what I've got."
- The **attention weight** is the softmax of (query · key) — how well the query matches this key.
- The **output** is a weighted average of the matched values.

Unlike a hard database lookup (SQL, hash tables), attention is **soft** — every key contributes some weight, and the softmax makes the weights differentiable. That means you can backprop through attention and *learn* what to attend to.

Practically:
- In self-attention on a sentence, each word queries every other word. "The" might attend strongly to the noun it modifies. "It" might attend to its antecedent.
- In cross-attention (translation), each target-language token queries every source-language token, effectively looking up which source words matter.
- In vision transformers, each patch attends to every other patch — no fixed neighborhood assumption.

The magic is that **each attention head learns its own query/key/value projections** — so each head becomes a specialized lookup pattern.

### 3.2 The "Committee of Advisors" Model

Imagine you're making a decision at every position of a sequence. You've assembled $h$ committees of advisors, each specializing in a different topic:

- **Committee 1**: syntactic relations. It answers "which words grammatically depend on this one?"
- **Committee 2**: semantic similarity. "Which words carry related meaning?"
- **Committee 3**: long-range references. "Which distant words matter?"
- ... (up to $h$ committees)

At each position, all $h$ committees advise in parallel. Each committee's advice is a weighted average of information from the other positions (weighted by that committee's own similarity metric). Then you **concatenate** all committee reports and apply a final linear layer to synthesize them.

That's multi-head attention.

The key point: no single committee could handle every kind of relation. Multi-head architecture lets **different heads specialize**. Empirically we see heads learn distinct patterns — head A attends to "the same word later in the sentence", head B attends to "verbs that share a subject with this noun", etc.

![IMG-ATT-01](/3%20—%20Deep%20Learning/images/IMG-ATT-01.jpg)
> **Caption:** Each row of the attention weight matrix is the softmax over how much this position attends to every other position.
> **Placement:** Section 2.2 / Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Multi-Head Attention Is Just Many Attentions Averaged"

They're not averaged — they're **concatenated** and then projected. Averaging would waste the head diversity; concatenation preserves each head's specialized signal, and the output projection $W_O$ learns how to combine them. This distinction matters: if you tried averaging, you'd lose the interpretability and expressiveness benefits.

### 4.2 "Attention Is $O(n)$ Because It's Just a Softmax"

Attention is $O(n^2)$ in both time and memory, because you compute $QK^\top \in \mathbb{R}^{n \times n}$. For a sequence of $n = 10{,}000$, that's a $100$-million-entry matrix per head per layer per batch. This quadratic cost is the primary reason long-context LLMs are expensive. Optimizations exist (**Flash Attention** — kernel-fused; **linear attention** — approximations; **sparse attention** — only some pairs), but naïve implementation is $O(n^2)$.

### 4.3 "Attention Is Interpretable — Look at the Weights"

Popular wisdom, but subtle. Attention weights show **where a position looks**, not necessarily **why the model made a decision**. A head might attend heavily to "the" for boring computational reasons (a "no-op" attention pattern). Multi-head + residuals + MLPs mean the final prediction depends on many components, not just attention weights. Interpretability research (probing, attribution, sparse autoencoders) suggests attention weights are one lens among many, not a definitive explanation.

---

## 5. Self-Assessment Bank (Attention)

### Questions

**Q1 (Short answer).** Write scaled dot-product attention as one formula, and label $Q$, $K$, $V$, $d_k$.

**Q2 (Multiple choice).** Why divide by $\sqrt{d_k}$ before the softmax?
A. To normalize probabilities.
B. To prevent large dot products from pushing softmax into saturating regions.
C. To make the operation faster.
D. It's a legacy convention.

**Q3 (Short answer).** Explain the difference between self-attention and cross-attention.

**Q4 (Multiple choice).** In multi-head attention with $h = 8$ heads and $d_{\text{model}} = 512$, the per-head key dimension is:
A. 512
B. 128
C. 64
D. 8

**Q5 (Short answer).** Why is a causal mask necessary for training an autoregressive language model like GPT? How is it implemented mathematically?

**Q6 (Multiple choice).** Attention is:
A. $O(n)$ in sequence length.
B. $O(n \log n)$.
C. $O(n^2)$ in both time and memory.
D. $O(n^3)$.

**Q7 (Short answer).** Why do we need positional encoding in transformers? Give two options and one modern default.

**Q8 (Multiple choice).** In the multi-head attention output formula $\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$, the final $W_O$:
A. Is optional.
B. Combines information across heads and projects back to $d_{\text{model}}$.
C. Applies dropout.
D. Normalizes attention weights.

**Q9 (Short answer).** What does a padding mask do in attention, and how is it implemented on the scores?

**Q10 (Multiple choice).** Attention weights show that head 7 puts most of its mass on position 3 regardless of the query. This most likely means:
A. Head 7 discovered a critical dependency on position 3.
B. Head 7 has learned a "no-op" pattern; it's using attention as a bias or constant lookup.
C. Head 7 is broken.
D. Positional encoding failed.

---

### Answer Key & Detailed Explanations

**A1.** $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$. $Q \in \mathbb{R}^{n \times d_k}$ = queries; $K \in \mathbb{R}^{m \times d_k}$ = keys; $V \in \mathbb{R}^{m \times d_v}$ = values; $d_k$ = key dimension.

**A2. B.** For high-dimensional $Q, K$ with random unit-variance entries, dot products $QK^\top$ have standard deviation $\sqrt{d_k}$. Without scaling, this pushes softmax into a very peaky regime (one-hot-like), giving near-zero gradients to non-max entries and destabilizing training. Dividing by $\sqrt{d_k}$ keeps scores unit-variance.

**A3.** **Self-attention:** queries, keys, and values all come from the same input tensor (they're each a learned linear projection of the same $X$). Every position attends to every other position of the same sequence. **Cross-attention:** queries come from one source (e.g., decoder positions), keys and values come from another (e.g., encoder positions). Decoder tokens attend to encoder tokens — this is how translation models "read" the source sentence when generating the target.

**A4. C.** $d_k = d_{\text{model}} / h = 512 / 8 = 64$. Each head operates in a 64-dimensional subspace; concatenating 8 heads restores 512 dimensions.

**A5.** In autoregressive generation, each token must be predicted based only on previous tokens. During training, we want to compute the loss for **every** position in parallel — but each position's attention must not include future tokens, or the model would trivially copy the future to predict itself. **Causal mask:** set score entries $S_{ij}$ to $-\infty$ for $j > i$ (positions after $i$). Softmax turns $-\infty$ into 0. Now every position's output depends only on itself and earlier positions.

**A6. C.** The score matrix $QK^\top$ is $\mathbb{R}^{n \times n}$ — $n^2$ entries in memory, $O(n^2 d_k)$ compute for the matrix multiply plus $O(n^2 d_v)$ for the value-weighted sum. Flash Attention reduces memory footprint via kernel fusion but the total work is still $O(n^2)$.

**A7.** Attention is permutation-invariant — shuffling input rows just shuffles output rows correspondingly. For a sequence, order matters, so we inject position information. Options: (1) **sinusoidal positional encoding** — deterministic sines/cosines; (2) **learned positional embedding** — a learnable table indexed by position; (3) **RoPE** (Rotary) — rotates $Q$ and $K$ by angle proportional to position; (4) **ALiBi** — linear-distance bias added to attention scores. **Modern default: RoPE**, used in LLaMA-family models.

**A8. B.** After concatenating all heads' outputs, the resulting tensor is $d_{\text{model}}$-dimensional. $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ is a learned linear projection that mixes information across heads and shapes the output for the next layer. Removing it would leave the heads as siloed components without opportunity to combine.

**A9.** A padding mask marks positions in the input that are padding tokens (not real content). In attention, we set the score $S_{ij}$ to $-\infty$ whenever key $j$ is a padding token. After softmax, those positions have zero attention weight — they contribute nothing to any query's output. Ensures the model doesn't learn to attend to padding.

**A10. B.** A head whose attention pattern is essentially constant regardless of query is playing a **no-op** role — it's using attention as a fixed lookup, essentially adding a constant bias to the residual. This is a real phenomenon in transformers (many heads are prunable without hurting performance). It doesn't mean the head is broken; it just isn't doing "real" attention.

---

## 6. Practice Prompts

1. **From scratch.** Implement scaled dot-product attention in PyTorch (no `nn.MultiheadAttention`). Verify output matches `F.scaled_dot_product_attention`.
2. **Multi-head from scratch.** Implement multi-head attention with $h = 4$ heads. Include a causal mask option.
3. **Attention viz.** Train a tiny transformer on a character-level task (predict next character in Shakespeare). Visualize attention weights — do you see interpretable patterns?
4. **Cross-attention seq2seq.** Build a small encoder-decoder transformer for a translation task. Explicitly separate self-attention (in encoder), self-attention (in decoder, causal), and cross-attention (decoder queries, encoder keys/values).
5. **Positional encoding comparison.** On the same task, compare sinusoidal, learned, and RoPE. Which extrapolates best beyond training-time lengths?

---

## 7. References

- Bahdanau, Cho & Bengio, ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473) (2014) — original attention.
- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017) — the transformer.
- Su et al., ["RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864) (2021).
- Press, Smith & Lewis, ["Train Short, Test Long: Attention with Linear Biases"](https://arxiv.org/abs/2108.12409) (2021) — ALiBi.
- Dao et al., ["FlashAttention"](https://arxiv.org/abs/2205.14135) (2022) — I/O-aware attention.
- Jay Alammar, ["The Illustrated Transformer"](http://jalammar.github.io/illustrated-transformer/) — the canonical visual explainer.
- Andrej Karpathy, ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the canonical code-through.
