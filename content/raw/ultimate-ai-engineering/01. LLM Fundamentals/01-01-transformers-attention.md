# 01.01 — Transformers & Attention

> **Module 1: LLM Fundamentals** · Subtopic 1 of 4
> **Prerequisites:** linear algebra (matrix products, projections), softmax and cross-entropy, backprop, PyTorch tensors.
> **Outcome:** you can derive, implement, profile, and debug a modern decoder-only transformer (Llama/Qwen/Mistral-class). You can also reason about its memory and FLOP budget at the level of a serving capacity plan.

---

## 1. The Modern Decoder-Only Block (what actually ships)

The 2017 transformer was an encoder-decoder with post-norm, learned or sinusoidal positions, ReLU FFN, and full multi-head attention. Almost every production LLM today is a **decoder-only, pre-norm, RoPE, RMSNorm, SwiGLU, GQA** stack. Many also use **MoE** FFNs.

```
                 token ids (B, T)
                       │
               ┌───────▼────────┐
               │ Embedding  E   │  (V × d)
               └───────┬────────┘
                       │  h⁰
      ┌────────────────▼─────────────────┐   × L layers
      │  ┌──────────┐                     │
      │  │ RMSNorm  │                     │
      │  └────┬─────┘                     │
      │       ▼                           │
      │  Q = xW_Q   K = xW_K   V = xW_V   │  Q: n_h heads, K/V: n_kv heads (GQA)
      │       │ RoPE(Q), RoPE(K)          │
      │       ▼                           │
      │  softmax(QKᵀ/√d_h + M_causal) V   │  ← KV cache lives here
      │       │  W_O                      │
      │  h ← h + Attn(·)   (residual)     │
      │  ┌──────────┐                     │
      │  │ RMSNorm  │                     │
      │  └────┬─────┘                     │
      │       ▼                           │
      │  SwiGLU FFN  (or MoE: router→top-k experts)
      │  h ← h + FFN(·)    (residual)     │
      └────────────────┬─────────────────┘
                       ▼
               ┌────────────────┐
               │ Final RMSNorm  │
               │ LM head  (d×V) │  often tied to Eᵀ in small models
               └───────┬────────┘
                       ▼
                 logits (B, T, V)
```

The **residual stream** is the central abstraction. Every sublayer *reads* from it (after normalization) and *writes* an additive update. Interpretability work (Elhage et al., 2021) treats it as a shared communication bus. Attention heads move information *between positions*. MLPs transform information *within a position*.

![IMG-CAP10-01](/01.%20LLM%20Fundamentals/images/IMG-CAP10-01.jpg)


---

## 2. Scaled Dot-Product Attention — The Math You Must Own

### 2.1 Definition

For queries $Q \in \mathbb{R}^{T_q \times d_k}$, keys $K \in \mathbb{R}^{T_k \times d_k}$, values $V \in \mathbb{R}^{T_k \times d_v}$:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
\qquad
M_{ij} = \begin{cases} 0 & j \le i + (T_k - T_q) \\ -\infty & \text{otherwise} \end{cases}
$$

The mask $M$ above is the **bottom-right-aligned causal mask**. It is the correct one whenever $T_q \neq T_k$, which covers decode with a KV cache, chunked prefill, and speculative verification. Many bugs come from using a top-left-aligned mask here (see §8).

### 2.2 Why $\sqrt{d_k}$?

Assume the components of $q$ and $k$ are i.i.d. with mean $0$ and variance $1$. Then

$$
\mathbb{E}[q\cdot k] = 0, \qquad \mathrm{Var}(q\cdot k) = \sum_{i=1}^{d_k}\mathrm{Var}(q_i k_i) = d_k .
$$

Without scaling, logits have standard deviation $\sqrt{d_k}$ (≈ 11 for $d_k=128$). Softmax then saturates toward one-hot, and its Jacobian $\mathrm{diag}(p) - pp^\top$ collapses toward zero, which gives vanishing gradients. Dividing by $\sqrt{d_k}$ restores unit variance at initialization. Some architectures replace this with a learned or fixed scale plus **QK-norm** (RMSNorm on $q$ and $k$), which bounds logits by construction.

### 2.3 Softmax numerics

Always compute $\mathrm{softmax}(s)_j = \dfrac{e^{s_j - m}}{\sum_k e^{s_k - m}}$ with $m=\max_k s_k$. In fp16, $e^{x}$ overflows for $x > 11.09$. bf16 has fp32's exponent range but only 8 bits of mantissa. Accumulate softmax denominators and the $PV$ product in **fp32** even when inputs are bf16.

### 2.4 Online softmax (the core of FlashAttention)

Suppose you process keys in blocks and want the exact result without ever materialising the full $T\times T$ matrix. Keep a running max $m$, normaliser $\ell$, and unnormalised output $o$. For each new block of scores $s^{(b)}$ with values $V^{(b)}$:

$$
\begin{aligned}
m' &= \max\big(m,\ \max_j s^{(b)}_j\big) \\
\ell' &= e^{m - m'}\,\ell + \sum_j e^{s^{(b)}_j - m'} \\
o' &= e^{m - m'}\,o + \sum_j e^{s^{(b)}_j - m'}\, V^{(b)}_j
\end{aligned}
\qquad\Longrightarrow\qquad \text{output} = o_{\text{final}} / \ell_{\text{final}}
$$

This recurrence (Milakov & Gimelshein, 2018) is exact. FlashAttention tiles $Q$, $K$, and $V$ into on-chip SRAM, runs the recurrence per tile, and writes only $O$ (plus the log-sum-exp for backward) to HBM. Attention memory drops from $O(T^2)$ to $O(T)$. Wall-clock time also falls, because naive attention is **memory-bandwidth-bound**, not FLOP-bound.

![LLM-02](/01.%20LLM%20Fundamentals/images/LLM-02.jpg)

---

## 3. Multi-Head, Multi-Query, Grouped-Query, and Latent Attention

### 3.1 Multi-head attention (MHA)

$$
\mathrm{head}_i = \mathrm{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}),\qquad
\mathrm{MHA}(X) = [\mathrm{head}_1;\dots;\mathrm{head}_{n_h}]\,W_O
$$

Each head has $d_h = d/n_h$. Heads specialise: previous-token heads, induction heads, syntactic heads, and so on.

### 3.2 The KV cache is the real cost

During autoregressive decode you cache $K$ and $V$ for every past token, in every layer and every KV head:

$$
\boxed{\ \text{KV bytes} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot b_{\text{dtype}} \cdot T \cdot B\ }
$$

**Worked example** (a 70B-class config: $L=80$, $n_h=64$, $n_{kv}=8$, $d_h=128$, bf16):

| Variant | Per-token KV | 128k-token context (1 seq) |
|---|---|---|
| MHA ($n_{kv}=64$) | $2\cdot80\cdot64\cdot128\cdot2 = 2.62$ MB | **~320 GiB** |
| GQA ($n_{kv}=8$) | $327{,}680$ B ≈ 320 KiB | **~40 GiB** |
| MQA ($n_{kv}=1$) | 40 KiB | ~5 GiB |

This single table explains why GQA is the default. At long context or high batch, the KV cache, not the weights, sets how many requests fit on a GPU.

### 3.3 MQA and GQA

- **MQA** (Shazeer, 2019): all query heads share one K/V head. Maximum savings, but a measurable quality loss and some training instability.
- **GQA** (Ainslie et al., 2023): $n_h$ query heads are divided into $g = n_{kv}$ groups, and each group shares a K/V head. Quality is close to MHA. You can **uptrain** an MHA checkpoint to GQA by mean-pooling K/V heads within each group, then continuing training on about 5% of the original compute.

### 3.4 Multi-head Latent Attention (MLA, DeepSeek-V2/V3)

MLA compresses K and V into a shared low-rank latent vector $c_t = W_{DKV}\,h_t \in \mathbb{R}^{d_c}$, and caches only $c_t$ plus a small decoupled RoPE key. Keys and values are reconstructed as $k = W_{UK}c$ and $v = W_{UV}c$. At inference, $W_{UK}$ can be **absorbed** into $W_Q$ (and $W_{UV}$ into $W_O$), so attention runs directly against the latent vectors. RoPE is not linear-absorbable, which is why a separate RoPE sub-key is kept. The result is a KV footprint below GQA with MHA-level quality. The cost is more complex kernels.

---

## 4. Positional Information

Attention is **permutation-equivariant**, so position has to be injected somehow.

### 4.1 Sinusoidal (original)

$PE_{(p,2i)} = \sin(p/10000^{2i/d})$ and $PE_{(p,2i+1)} = \cos(p/10000^{2i/d})$, added to the embeddings. Extrapolation is poor in practice.

### 4.2 RoPE (Rotary Position Embedding)

Split $q$ and $k$ into 2-D pairs and rotate pair $i$ at position $m$ by angle $m\theta_i$, with $\theta_i = b^{-2i/d_h}$ (usually $b = 10^4$ or larger):

$$
R_{m}^{(i)} = \begin{pmatrix}\cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i\end{pmatrix},
\qquad
\langle R_m q,\; R_n k\rangle = \langle q,\; R_{n-m}\,k\rangle .
$$

The dot product depends only on the **relative offset** $n-m$, even though the rotation is applied to absolute positions. Low-$i$ pairs rotate fast (local, high-frequency signal). High-$i$ pairs rotate slowly (long-range signal). Context-extension methods (PI, NTK, YaRN; see §01.04) all work by manipulating the $\theta_i$.

> ⚠️ **Convention trap:** the original RoPE pairs dimensions *(0,1), (2,3), …* (interleaved). HF Llama pairs *(i, i + d/2)* using `rotate_half`. The two are equivalent only if weights were permuted to match. Loading Meta weights into an interleaved implementation without the permutation produces a model that runs but outputs garbage.


![LLM-03](/01.%20LLM%20Fundamentals/images/LLM-03.jpg)


### 4.3 ALiBi

ALiBi adds no embedding. Instead it adds a head-specific linear bias $-m_h\cdot(i-j)$ to the attention logits, with slopes $m_h$ forming a geometric sequence. It extrapolates reasonably well, but RoPE plus scaling has won in practice.

---

## 5. Normalization and the FFN

### 5.1 RMSNorm vs LayerNorm

$$
\mathrm{LN}(x) = \gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,
\qquad
\mathrm{RMSNorm}(x) = \gamma\odot\frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2+\epsilon}}
$$

RMSNorm drops mean-centering and bias. It is cheaper and empirically just as good.

### 5.2 Pre-norm vs post-norm

**Post-norm** computes $x + f(x)$ and then normalizes. **Pre-norm** computes $x + f(\mathrm{Norm}(x))$. With post-norm, gradients near the output are large at initialization (Xiong et al., 2020), so it needs learning-rate warmup and becomes unstable in deep stacks. Pre-norm keeps an identity path through the residual stream and trains stably without careful warmup. The trade-off is that the residual-stream norm grows with depth, which is one reason for the final norm before the LM head.

### 5.3 SwiGLU FFN

$$
\mathrm{FFN}_{\text{SwiGLU}}(x) = \big(\mathrm{SiLU}(xW_1)\odot xW_3\big)W_2,
\qquad \mathrm{SiLU}(z)=z\,\sigma(z)
$$

This uses three matrices instead of two. To keep parameters equal to a $4d$ ReLU FFN, set $d_{ff}\approx \tfrac{8}{3}d$, rounded to a hardware-friendly multiple (e.g. 256).

### 5.4 Parameter and FLOP accounting

Per layer, ignoring norms:

$$
P_{\text{attn}} = d\,(n_h d_h) + 2\,d\,(n_{kv} d_h) + (n_h d_h)\,d, \qquad
P_{\text{ffn}} = 3\,d\,d_{ff}
$$

For forward FLOPs per token (Kaplan et al., 2020), with $N$ the non-embedding parameter count:

$$
C_{\text{fwd}} \approx 2N + 2\,L\,T_{\text{ctx}}\,d_{\text{attn}},\qquad C_{\text{train}} \approx 6N \text{ per token}
$$

The second term is attention over context. It is small at short context and dominant at long context (derivation in §01.04).

---

## 6. Mixture-of-Experts (MoE)

Replace the dense FFN with $E$ expert FFNs and a router $g(x) = \mathrm{softmax}(xW_r)$. Each token is sent to its top-$k$ experts:

$$
y = \sum_{e \in \mathrm{TopK}(g(x),k)} \tilde g_e(x)\,\mathrm{FFN}_e(x)
$$

Here $\tilde g$ is the gate renormalised over the selected experts. The design gives **total** parameters of roughly $E$× the FFN, but **active** parameters per token of only $k$×.

**Load balancing.** A router left alone collapses onto a few experts. The Switch Transformer auxiliary loss is

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i\, P_i
$$

where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ is the mean router probability for expert $i$. Newer designs (DeepSeek-V3) use **auxiliary-loss-free** balancing: a per-expert bias is added to routing scores and adjusted online. They also use fine-grained experts plus always-on shared experts.

**Production realities.** Serving an MoE is limited by memory (all experts must be resident) and communication (expert-parallel all-to-all). The kernels are grouped GEMMs. Batch composition changes routing, which affects determinism (see §01.02).

---

## 7. Reference Implementation (PyTorch)

A GQA attention layer with RoPE and a preallocated KV cache. It handles prefill, single-token decode, and **chunked prefill**, which is where most hand-rolled implementations fail.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(max_len: int, head_dim: int, base: float = 10_000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (T, D/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair convention. x: (B, H, T, D); cos/sin: (T, D/2)."""
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class GQAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h, self.kv_h, self.hd = n_heads, n_kv_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None, start_pos: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv_h, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv_h, self.hd).transpose(1, 2)

        pos_cos, pos_sin = cos[start_pos:start_pos + T], sin[start_pos:start_pos + T]
        q, k = apply_rope(q, pos_cos, pos_sin), apply_rope(k, pos_cos, pos_sin)

        if kv_cache is not None:                      # preallocated (B, kv_h, T_max, hd)
            k_cache, v_cache = kv_cache
            k_cache[:, :, start_pos:start_pos + T] = k
            v_cache[:, :, start_pos:start_pos + T] = v
            k, v = k_cache[:, :, :start_pos + T], v_cache[:, :, :start_pos + T]

        S = k.shape[2]
        # Bottom-right-aligned causal mask: query i may see keys j <= start_pos + i.
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)

        rep = self.h // self.kv_h                     # broadcast KV heads to query groups
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


def naive_attention(q, k, v, mask):
    """Reference for testing: materialises the full score matrix."""
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s.float(), dim=-1).type_as(q) @ v


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), GQAttention(d, n_heads, n_kv_heads)
        self.n2, self.ffn = RMSNorm(d), SwiGLU(d, d_ff)

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        x = x + self.attn(self.n1(x), cos, sin, kv_cache, start_pos)
        return x + self.ffn(self.n2(x))
```

**Notes that matter in production:**

- `repeat_interleave` materialises the expanded K/V. Real kernels (FlashAttention, `scaled_dot_product_attention(..., enable_gqa=True)` in recent PyTorch, vLLM/SGLang kernels) index the shared head instead of copying it.
- Passing an explicit boolean mask can force SDPA off the FlashAttention backend onto the slower "efficient" or math backend. Use `is_causal=True` for square prefill, and handle the offset case in a kernel that supports bottom-right causal alignment (FlashAttention does; so does PyTorch **FlexAttention** via a `mask_mod`).
- Always test against a naive reference, and assert **cached decode == uncached full forward** logits within tolerance.

---

## 8. Production Challenges & How to Solve Them

| Problem | Symptom | Root cause | Fix |
|---|---|---|---|
| **Wrong causal alignment** | Correct output on prefill, degraded output on decode or chunked prefill | Top-left mask when $T_q \ne T_k$ | Bottom-right mask (§2.1). Add a unit test: chunked prefill equals single-pass prefill. |
| **RoPE convention mismatch** | Model loads, output is fluent nonsense | Interleaved vs half-split pairing | Match the checkpoint's convention or permute $W_Q$/$W_K$ rows. Compare layer-0 attention outputs against the reference implementation. |
| **fp16 overflow / NaNs** | NaN loss spikes, `inf` in logits | Large activations, softmax in fp16 | bf16 training, fp32 softmax accumulation, QK-norm, z-loss $\lambda(\log Z)^2$, **logit soft-capping** $c\cdot\tanh(z/c)$ |
| **Loss spikes at scale** | Divergence mid-run | Attention-logit growth, outlier features | QK-norm, lower β₂, gradient clipping, skip bad batches, restart from a checkpoint with data reordered |
| **Attention sinks** | Model dumps attention mass on token 0; evicting it breaks generation | Softmax must sum to 1, so "no-op" heads park mass on the first tokens | Never evict the first few tokens (StreamingLLM), or train with a learned sink or gated attention |
| **Quadratic prefill** | TTFT grows superlinearly with prompt length | $O(T^2)$ attention | FlashAttention kernels, chunked prefill, prefix caching, context-parallelism (§01.04) |
| **KV memory ceiling** | Low max batch, OOM at long context | §3.2 formula | GQA/MLA models, KV quantization (FP8/INT4), paged allocation (§01.02) |
| **Outlier activations** | INT8 quantization destroys quality | A few massive channels in the residual stream | Mixed precision for outlier channels (LLM.int8()), SmoothQuant migration, FP8 with per-channel scales |
| **MoE imbalance** | Expert OOM, low utilisation, token dropping | Router collapse, skewed batches | Aux or bias-based balancing, capacity factor, expert replication for hot experts |

---

## 9. Hands-On Projects

### Project 1 — Build and Verify a Llama-Style Decoder from Scratch

**User stories**
- *As an ML engineer*, I want a minimal decoder (GQA + RoPE + RMSNorm + SwiGLU) that I fully understand, so that I can debug production models by analogy.
- *As a reviewer*, I want tests proving that cached and uncached inference agree, so that I trust the implementation.

**Acceptance criteria**
1. Model config is parametrised by `(L, d, n_h, n_kv, d_ff, vocab, max_len)`. Setting `n_kv = n_h` reproduces MHA exactly, and `n_kv = 1` gives MQA.
2. A unit test proves the RoPE relative property: $\langle R_m q, R_n k\rangle = \langle R_{m+c} q, R_{n+c} k\rangle$ to within 1e-5 (fp32).
3. Greedy generation with the KV cache produces **identical tokens** to generation without it for 200 tokens. Logits match within 1e-4 in fp32.
4. Chunked prefill (chunks of 64) matches single-pass prefill logits within 1e-4.
5. Trains on TinyStories (or a similar small corpus) at ≤ 30M parameters, with a monotonic validation-loss curve and samples that are coherent short stories.
6. A parameter-count function matches `sum(p.numel())` exactly, and a FLOP estimator matches `torch.utils.flop_counter` within 5%.

**Step-by-step**
1. **Scaffold.** Create `model.py` with the §7 modules, an `Embedding`, a final `RMSNorm`, and an LM head (optionally tied).
2. **Data.** Tokenize TinyStories with an existing tokenizer (e.g. GPT-2 BPE via `tiktoken`). Pack into a contiguous `uint16` memmap and sample random windows of length `T`.
3. **Training loop.** Use AdamW (β = 0.9, 0.95; wd = 0.1), cosine LR with warmup, bf16 autocast, grad-clip 1.0, and `torch.compile`. Log train and validation loss every N steps.
4. **Tests.** Write `pytest` cases for criteria 1–4. Use fp32 and `torch.manual_seed`.
5. **KV cache.** Preallocate per-layer `(B, n_kv, T_max, d_h)` tensors and thread `start_pos` through `forward`.
6. **Accounting.** Implement `count_params(config)` from §5.4 and compare it to the real count. Profile a forward pass with `torch.profiler` and identify which ops dominate at T=256 vs T=4096.
7. **Report.** Write a README covering the loss curve, samples, a KV-memory table for your config, and one bug you hit and how the tests caught it.

---

### Project 2 — Attention Kernel Benchmark & Tiled Online-Softmax Implementation

**User stories**
- *As a performance engineer*, I want measured latency and memory curves for naive attention, PyTorch SDPA backends, and FlashAttention across sequence lengths, so that I can choose kernels on evidence.
- *As a learner*, I want to implement tiled online-softmax attention myself, so that I understand why FlashAttention is fast.

**Acceptance criteria**
1. Benchmarks sequence lengths {512, 1k, 2k, 4k, 8k, 16k, 32k} with a fixed batch and head config, in bf16 on one GPU. Runs forward-only and forward+backward.
2. Reports median latency (CUDA events, warmup excluded) and peak memory (`torch.cuda.max_memory_allocated`). Naive attention is shown hitting OOM or exploding quadratically in memory.
3. Includes a custom **Triton** (or blocked pure-PyTorch) kernel implementing the §2.4 recurrence. It matches the SDPA output within `atol=2e-2` in bf16 (fp32 accumulators) and supports causal masking.
4. A written analysis computes the arithmetic intensity (FLOPs / HBM bytes) of naive vs tiled attention and relates the measured speedup to your GPU's roofline.

**Step-by-step**
1. Write `bench.py` with a harness: `torch.cuda.synchronize()`, 10 warmup iterations, then 50 timed with `torch.cuda.Event`. Report the median.
2. Implement the variants: `naive_attention`, SDPA with forced backends (`torch.nn.attention.sdpa_kernel`), and `flash_attn_func` from the `flash-attn` package.
3. Write a blocked PyTorch reference first: loop over K/V blocks and keep `m`, `l`, `o` per query row. Validate it against naive at small T.
4. Port the reference to Triton. Assign one program per (batch·head, Q-block), loop over K/V blocks, use `tl.dot`, and keep the online-softmax state in registers. Start from the official Triton fused-attention tutorial *structure*, but write the kernel yourself.
5. Plot latency and memory vs T on log-log axes and annotate slopes: 2 for quadratic, 1 for linear.
6. Profile with Nsight Compute on one configuration. Compare achieved bandwidth and FLOP/s against peak.

---

### Project 3 — Find and Ablate Induction Heads (Mechanistic Interpretability)

**User stories**
- *As an AI engineer*, I want to locate the attention heads that implement in-context copying, so that I understand the mechanism behind in-context learning and long-context retrieval failures.

**Acceptance criteria**
1. Using GPT-2 small (or Pythia-160M), computes an **induction score** for every (layer, head) on sequences of random tokens repeated twice (e.g. 2 × 50 tokens). The score is the mean attention from position $t$ in the second copy to position $t - 50 + 1$.
2. Produces a layer × head heatmap and identifies the top-k heads (induction heads usually appear in middle layers).
3. **Zero- or mean-ablating** the top-k induction heads raises loss on the repeated half by a margin much larger than ablating k random heads (report the mean ± std over 20 random draws).
4. Shows at least one head's attention pattern visualised on a real sentence with repeated names.

**Step-by-step**
1. `pip install transformer_lens` and load `HookedTransformer.from_pretrained("gpt2")`.
2. Generate `rep = torch.cat([bos, r, r], dim=1)` with random token ids `r`.
3. Run `model.run_with_cache(rep)` and read `cache["pattern", layer]` with shape `(B, H, T, T)`.
4. Compute the induction score as the mean of `pattern[..., 50+i, i+1]` along the offset diagonal.
5. Ablate with a hook on `hook_z` (per-head output) that zeroes the selected heads. Compare per-token loss on the second half.
6. Write up how the previous-token-head → induction-head circuit works (the K-composition described by Olsson et al., 2022).

---

## 10. Foundational Papers (exact titles)

**Core architecture**
- Vaswani et al., 2017 — *Attention Is All You Need*
- Radford et al., 2019 — *Language Models are Unsupervised Multitask Learners*
- Xiong et al., 2020 — *On Layer Normalization in the Transformer Architecture*
- Zhang & Sennrich, 2019 — *Root Mean Square Layer Normalization*
- Shazeer, 2020 — *GLU Variants Improve Transformer*
- Touvron et al., 2023 — *LLaMA: Open and Efficient Foundation Language Models*
- Grattafiori et al. (Llama Team), 2024 — *The Llama 3 Herd of Models*

**Attention variants and positions**
- Shazeer, 2019 — *Fast Transformer Decoding: One Write-Head is All You Need*
- Ainslie et al., 2023 — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- DeepSeek-AI, 2024 — *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*
- Su et al., 2021 — *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Press et al., 2021 — *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*

**Efficient kernels**
- Milakov & Gimelshein, 2018 — *Online normalizer calculation for softmax*
- Dao et al., 2022 — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- Dao, 2023 — *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- Shah et al., 2024 — *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*

**Mixture-of-Experts**
- Shazeer et al., 2017 — *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*
- Fedus et al., 2021 — *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*
- DeepSeek-AI, 2024 — *DeepSeek-V3 Technical Report*

**Scaling and stability**
- Kaplan et al., 2020 — *Scaling Laws for Neural Language Models*
- Hoffmann et al., 2022 — *Training Compute-Optimal Large Language Models*
- Dehghani et al., 2023 — *Scaling Vision Transformers to 22 Billion Parameters* (QK-norm for stability)
- Xiao et al., 2023 — *Efficient Streaming Language Models with Attention Sinks*

**Interpretability**
- Elhage et al., 2021 — *A Mathematical Framework for Transformer Circuits*
- Olsson et al., 2022 — *In-context Learning and Induction Heads*

## 11. Essential Tooling

| Tool | Why |
|---|---|
| **PyTorch** (`scaled_dot_product_attention`, **FlexAttention**, `torch.compile`, `torch.profiler`) | Reference implementations, custom masks without writing kernels |
| **flash-attn** | Production attention kernels, varlen/packed sequences |
| **Triton** | Writing your own fused GPU kernels in Python |
| **nanoGPT / llm.c** (Karpathy) | Minimal, readable training references |
| **Hugging Face `transformers`** (`modeling_llama.py`) | Canonical source for weight layouts and conventions |
| **TransformerLens** | Hooks, caching, and ablation for interpretability |
| **Nsight Systems / Nsight Compute** | Timeline and kernel-level GPU profiling |
| **Weights & Biases / TensorBoard** | Experiment tracking |
