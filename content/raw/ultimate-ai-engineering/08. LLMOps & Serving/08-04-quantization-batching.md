# 08.04 — Quantization & Batching (Making Every GPU-Second Count)

> **Module 8: LLMOps & Serving** · Subtopic 4 of 5
> **Prerequisites:** **01.02 §2, §3, §7 (required: roofline view of prefill/decode, KV cache, PagedAttention, continuous batching, the quantization intro)**, 01.01 §3.2 (KV sizing), 06.01 (paired bootstrap), 08.02 (unit economics), 08.03 (vLLM flags, capacity math).
> **Outcome:** you can choose a number format and quantization algorithm for a given GPU and workload, produce and validate a quantized checkpoint with slice-level quality gates, and tune batching for maximum **goodput** (throughput delivered within SLO), with a roofline model that predicts where each optimisation helps.

> **Status check (September 2026).**
> - **LLM Compressor** (`vllm-project/llm-compressor`) is the standard way to produce vLLM-ready checkpoints (the `compressed-tensors` format). It supports:
>   - **W8A8** (INT8 and FP8), **W4A16**, **W8A16**, **W4AFP8**;
>   - microscaling **NVFP4 / MXFP4 / MXFP8**;
>   - **FP8 and NVFP4 KV-cache/attention** quantization;
>   - the algorithms **GPTQ, AWQ, SmoothQuant, AutoRound**, rotations (**SpinQuant, QuIP**), and simple PTQ.
> - **Hardware sets the menu:**
>   - Ampere (A100) has no FP8 tensor cores, so use INT8 or W4A16 there.
>   - Hopper (H100/H200) and Ada add FP8.
>   - Blackwell (B200/GB200/RTX 50xx) adds native FP4 (NVFP4/MXFP4).
>   - Always verify kernel support for your exact (GPU, scheme, vLLM version) triple.

---

## 1. Why Quantize: Decode Is a Bandwidth Problem

Recall 01.02 §2. Each decode step must stream **every weight** (plus each sequence's KV cache) from HBM.

For batch size $B$, parameters $P$, bytes per weight $b_w$, KV bytes per sequence $b_{kv}$, peak compute $F$, and bandwidth $BW$:

$$
t_{\text{step}}(B) \;\approx\; \max\!\left(\underbrace{\frac{2PB}{F}}_{\text{compute}},\; \underbrace{\frac{P\,b_w + B\,b_{kv}}{BW}}_{\text{memory}}\right),
\qquad
\text{tok/s} = \frac{B}{t_{\text{step}}(B)} .
$$

Ignoring KV, the weight GEMM stays **memory-bound** while its arithmetic intensity $2B/b_w$ is below the GPU's **ridge point** $F/BW$:

$$
B^{*} \;=\; \frac{F}{BW}\cdot\frac{b_w}{2}.
$$

Three consequences drive every decision in this subtopic:

1. **Weight-only quantization (W4A16, W8A16) speeds up small-batch, latency-bound serving** almost in proportion to $1/b_w$. Its crossover batch $B^*$ is lower, however. With INT4 on an H100 it is ≈ 74 versus ≈ 295 for BF16 (§5.1). Past that point you are compute-bound on BF16 math plus dequantisation overhead, and W4A16 can end up **slower** than FP8.
2. **Weight+activation quantization (W8A8-FP8, W4A4-NVFP4)** also doubles or quadruples the tensor-core FLOPs. It is the choice for **high-batch throughput** serving.
3. **At long context and large batch, KV traffic dominates weights.** An 8B model at 2k context and $B=256$ reads about 67 GB of KV per step versus 16 GB of weights (§5.1). There, **KV-cache quantization** (FP8 KV) and prefix caching buy more than weight quantization does. KV quantization also doubles KV capacity (08.03 §3).

---

## 2. Number Formats

| Format | Bits (S/E/M) | Range / grid | Scaling granularity | HW (native math) | Typical use |
|---|---|---|---|---|---|
| **BF16** | 1/8/7 | FP32 range, ~3 significant digits | none | A100+ | Baseline serving, training |
| **FP8 E4M3** | 1/4/3 | ±448, 3 mantissa bits | per-tensor, per-channel, per-token, or **block (e.g. 128×128)** | H100/H200, Ada, Blackwell | **W8A8 inference default on Hopper**, KV cache |
| **FP8 E5M2** | 1/5/2 | ±57344, 2 mantissa bits | per-tensor | same | Gradients (training) |
| **INT8** | 8-bit two's complement | 256 uniform levels | per-channel W, per-token A | A100+ (IMMA) | W8A8 on Ampere, SmoothQuant |
| **INT4** | 4-bit | 16 uniform levels | **group-wise (g = 32/64/128)** + FP16 scales (± zero points) | dequant-to-FP16 kernels (Marlin/Machete etc.) | **W4A16** (GPTQ/AWQ) for memory-bound serving |
| **MXFP4** (OCP MX) | E2M1 elements | {0, .5, 1, 1.5, 2, 3, 4, 6} × scale | **32-element blocks, E8M0 (power-of-two) scale** | Blackwell | Open standard FP4 |
| **NVFP4** | E2M1 elements | same grid | **16-element blocks, FP8 E4M3 scale + FP32 per-tensor scale** | Blackwell | W4A4 / W4A16 on NVIDIA, FP4 KV |

**Uniform quantization.** With step $\Delta$ and zero point $z$:

$$
q = \mathrm{clamp}\!\left(\mathrm{round}\!\left(\tfrac{x}{\Delta}\right) + z,\; q_{\min},\, q_{\max}\right),
\qquad \hat{x} = \Delta\,(q - z).
$$

- **Symmetric:** $\Delta = \max|x| / (2^{b-1}-1)$ and $z=0$.
- **Asymmetric:** $\Delta = (\max x - \min x)/(2^b - 1)$.
- For values spread uniformly inside the clipping range, the rounding error is uniform on $[-\Delta/2, \Delta/2]$, which gives:

$$
\mathbb{E}\big[(x-\hat x)^2\big] = \frac{\Delta^2}{12},
\qquad
\text{SQNR} \approx 6.02\,b + 1.76 \ \text{dB (full-scale sinusoid)}.
$$

Each bit removed costs about 6 dB. The real enemy is not the bit count, though. It is **$\Delta$ being set by outliers**: one value 20× larger than the rest inflates $\Delta$ 20× for everything sharing its scale. Every modern method is a way to shrink the set of values that share a scale (groups, blocks), to move outliers somewhere harmless (SmoothQuant, rotations), or to compensate the rounding error (GPTQ).

**Floating-point formats** give relative (log-spaced) precision. The error scales with $|x|$ rather than with the global max, which is why FP8 E4M3 per-tensor beats INT8 per-tensor on outlier-heavy tensors (below). **Microscaling** formats (MX/NVFP4) combine a tiny FP element with a per-block scale, so small blocks isolate outliers. NVFP4's smaller blocks and finer (E4M3 rather than power-of-two) scales give it lower error than MXFP4.

```python
import numpy as np

def quant_int(x, bits=4, axis=None, group=None, symmetric=True):
    """Fake-quantize (quantize->dequantize). axis=None: per-tensor; group: group-wise along last dim."""
    x = np.asarray(x, dtype=np.float64)
    shp = x.shape
    if group is not None:
        assert shp[-1] % group == 0
        x = x.reshape(*shp[:-1], shp[-1] // group, group)
        axis = -1
    red = dict(axis=axis, keepdims=True) if axis is not None else dict()
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = np.max(np.abs(x), **red) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        out = np.clip(np.round(x / scale), -qmax - 1, qmax) * scale
    else:
        lo, hi = np.min(x, **red), np.max(x, **red)
        scale = (hi - lo) / (2 ** bits - 1)
        scale = np.where(scale == 0, 1.0, scale)
        z = np.round(-lo / scale)
        q = np.clip(np.round(x / scale) + z, 0, 2 ** bits - 1)
        out = (q - z) * scale
    return out.reshape(shp)

def fp_round(x, man_bits, max_val, min_exp):
    """Round to a sign/exponent/mantissa grid (saturating; subnormals below 2**min_exp)."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    e = np.maximum(np.floor(np.log2(np.maximum(ax, 1e-300))), min_exp)
    step = 2.0 ** (e - man_bits)
    return np.sign(x) * np.minimum(np.round(ax / step) * step, max_val)

E4M3 = dict(man_bits=3, max_val=448.0, min_exp=-6)    # FP8 E4M3 ("fn" variant)
E2M1 = dict(man_bits=1, max_val=6.0, min_exp=0)       # FP4 element grid {0,.5,1,1.5,2,3,4,6}

def quant_fp8(x, fmt=E4M3):
    s = np.max(np.abs(x)) / fmt["max_val"]             # per-tensor scale: amax -> format max
    return fp_round(x / s, **fmt) * s

def quant_nvfp4(x):
    """NVFP4 sim: 16-element blocks, E4M3 block scale, FP32 per-tensor scale."""
    x = np.asarray(x, dtype=np.float64)
    b = x.reshape(-1, 16)
    g = np.max(np.abs(b)) / (6.0 * 448.0)
    s = fp_round(np.max(np.abs(b), axis=1, keepdims=True) / 6.0 / g, **E4M3)
    s = np.where(s == 0, 1.0, s)
    return (fp_round(b / (s * g), **E2M1) * s * g).reshape(x.shape)

def quant_mxfp4(x):
    """MXFP4 sim (OCP MX): 32-element blocks, power-of-two (E8M0) shared scale."""
    x = np.asarray(x, dtype=np.float64)
    b = x.reshape(-1, 32)
    amax = np.maximum(np.max(np.abs(b), axis=1, keepdims=True), 1e-30)
    s = 2.0 ** (np.floor(np.log2(amax)) - 2)           # emax of E2M1 = 2
    return (fp_round(b / s, **E2M1) * s).reshape(x.shape)

def rel_err(x, xq):
    return float(np.linalg.norm(x - xq) / np.linalg.norm(x))

rng = np.random.default_rng(0)
W = rng.standard_normal((256, 1024)) * 0.02
W[rng.random(W.shape) < 0.001] *= 20                    # sparse large-magnitude outliers
for name, f in [("INT8 per-tensor", lambda w: quant_int(w, 8)),
                ("FP8 E4M3 per-tensor", quant_fp8),
                ("INT4 per-tensor", lambda w: quant_int(w, 4)),
                ("INT4 per-channel", lambda w: quant_int(w, 4, axis=-1)),
                ("INT4 group-128", lambda w: quant_int(w, 4, group=128)),
                ("MXFP4 (32, E8M0)", quant_mxfp4),
                ("NVFP4 (16, E4M3)", quant_nvfp4)]:
    print(f"{name:22s} rel_err={rel_err(W, f(W)):.4f}")

grid = sorted({float(v) for v in np.abs(fp_round(np.linspace(-7, 7, 2001), **E2M1)).round(3)})
assert grid == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
x = rng.uniform(-1, 1, 200_000)
print("asym INT8 MSE", np.mean((x - quant_int(x, 8, symmetric=False)) ** 2), "theory", (2 / 255) ** 2 / 12)
```

Measured output (outlier-bearing weights):

| Scheme | Relative error |
|---|---|
| INT8 per-tensor | 0.116 |
| **FP8 E4M3 per-tensor** | **0.026** |
| INT4 per-tensor | 0.817 |
| INT4 per-channel | 0.506 |
| INT4 group-128 | 0.216 |
| MXFP4 | 0.145 |
| **NVFP4** | **0.093** |

The uniform-noise MSE matches $\Delta^2/12$ to 0.1%.

Reading the numbers:
- FP8's relative precision makes it 4× better than INT8 when outliers set the scale.
- Group size is the main lever for INT4.
- Block-scaled FP4 beats group-128 INT4 at the same nominal 4 bits. The effective bits are ≈ 4.5 for NVFP4: 4 bits plus an 8-bit scale per 16 elements.

**Effective bits per weight:**

$$
b_{\text{eff}} = b + \frac{b_{\text{scale}} (+ b_{\text{zero}})}{g}.
$$

This gives, for example, 4 + 16/128 = 4.125 for INT4 g128 with FP16 scales, 4.5 for NVFP4, and 4.25 for MXFP4. Use $b_{\text{eff}}$, not the nominal bit count, in memory math.

---

## 3. Algorithms

### 3.1 Taxonomy

```
                    ┌── weight-only (W4A16, W8A16) ── RTN · GPTQ · AWQ · AutoRound  → memory-bound wins
 Post-training ─────┼── weight+activation (W8A8, W4A4) ── SmoothQuant · rotations (QuaRot/SpinQuant) · FP8 dynamic
 quantization (PTQ) └── KV cache (FP8 / NVFP4 / INT4-2bit research: KIVI) → capacity + attention bandwidth
 Quantization-aware training (QAT) / QLoRA fine-tuning ── when PTQ loses too much (≤ 4-bit activations, small models)
```

| Method | Idea | Calibration | Cost | Strength | Watch out |
|---|---|---|---|---|---|
| **RTN** (round-to-nearest) | Round with per-group scales | none | seconds | FP8 dynamic, INT8 W8A16 are near-lossless | Poor at W4 without small groups |
| **GPTQ** | Second-order error compensation, column by column, using $H = 2X^\top X$ | ~128–512 samples | minutes–hours | Strong W4/W3 | **Overfits calibration domain**; act-order + groups need kernel support |
| **AWQ** | Protect salient channels (large activation magnitude) by per-channel scaling before quantizing; grid-search $\alpha$ | small | fast | Robust W4A16, less calibration-sensitive | Weight-only |
| **SmoothQuant** | Migrate activation outliers into weights: $X' = X\,\mathrm{diag}(s)^{-1}$, $W' = \mathrm{diag}(s)\,W$ | small | fast | Makes **W8A8 INT8** work | α tuning per model |
| **QuaRot / SpinQuant** | Orthogonal (Hadamard / learned) rotations spread outliers; exact in FP | small (SpinQuant learns $R$) | moderate | **W4A4 and 4-bit KV** feasible | Needs rotation-aware kernels / fused Hadamards |
| **FP8 dynamic** | Per-token activation scales computed at runtime; static per-channel weight scales | none | seconds | **Default first move on Hopper**; near-lossless for most models | Needs FP8 HW |
| **NVFP4 / MXFP4** | Microscaled FP4 with calibrated global scales | small | fast | Blackwell 4-bit compute | Accuracy varies by model; validate |
| **QAT / QLoRA** | Train with fake-quant in the loop / LoRA on frozen 4-bit base | training data | hours–days | Recovers accuracy at low bits | Engineering + compute cost |

### 3.2 GPTQ — the math that matters

GPTQ is used for a linear layer $Y = XW^\top$, where $W \in \mathbb{R}^{d_{out}\times d_{in}}$ and calibration inputs are $X \in \mathbb{R}^{n\times d_{in}}$. It minimizes $\lVert XW^\top - X\hat W^\top\rVert_F^2$ row by row. The loss is quadratic with Hessian $H = 2X^\top X$, shared across rows.

When column $i$ is quantized to $q_i$, the **Optimal Brain Surgeon** update redistributes the error optimally over the remaining columns:

$$
\delta_{\,i+1:} \;=\; -\,\frac{w_i - q_i}{[H^{-1}]_{ii}}\;[H^{-1}]_{i,\,i+1:} .
$$

GPTQ's engineering tricks turn this into an $O(d_{in}^3)$-per-layer algorithm that runs on 100B+ models:

- process columns in a **fixed order**, so one Cholesky factor of $H^{-1}$ serves all rows;
- apply **lazy batch updates**;
- use **dampening** $H + \lambda\,\overline{\mathrm{diag}(H)}\,I$.

"Act-order" quantizes high-$H_{ii}$ columns first.

```python
import numpy as np

def rtn_rowwise(W, bits=4, group=None):
    """Round-to-nearest, symmetric, per-output-row scales (optionally group-wise along input dim)."""
    qmax = 2 ** (bits - 1) - 1
    out, g = np.empty_like(W), group or W.shape[1]
    for j in range(0, W.shape[1], g):
        blk = W[:, j:j + g]
        s = np.maximum(np.abs(blk).max(axis=1, keepdims=True), 1e-12) / qmax
        out[:, j:j + g] = np.clip(np.round(blk / s), -qmax - 1, qmax) * s
    return out

def gptq(W, X, bits=4, group=None, damp=0.01):
    """GPTQ: quantize input columns in order; push each column's error onto the not-yet-quantized
    columns via the upper Cholesky factor of H^-1, H = 2 X^T X.  W: (out, in); X: (n, in)."""
    W = W.astype(np.float64).copy()
    d = W.shape[1]
    H = 2.0 * X.T @ X
    H += damp * np.mean(np.diag(H)) * np.eye(d)
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    qmax, g = 2 ** (bits - 1) - 1, group or d
    Q = np.zeros_like(W)
    for i in range(d):
        if i % g == 0:                                   # group scales from the *updated* weights
            s = np.maximum(np.abs(W[:, i:i + g]).max(axis=1), 1e-12) / qmax
        q = np.clip(np.round(W[:, i] / s), -qmax - 1, qmax) * s
        Q[:, i] = q
        err = (W[:, i] - q) / U[i, i]
        W[:, i + 1:] -= np.outer(err, U[i, i + 1:])
    return Q

rng = np.random.default_rng(1)
n, d_in, d_out = 4096, 256, 128
A = rng.standard_normal((d_in, d_in)) / np.sqrt(d_in)
mix = np.eye(d_in) + 0.9 * A                              # correlated input features
X = rng.standard_normal((n, d_in)) @ mix;   X[:, :4] *= 8.0
Xte = rng.standard_normal((n, d_in)) @ mix; Xte[:, :4] *= 8.0   # held-out activations
W = rng.standard_normal((d_out, d_in)) * 0.05

def out_err(Wq):
    return float(np.linalg.norm(Xte @ W.T - Xte @ Wq.T) / np.linalg.norm(Xte @ W.T))

for bits, grp in [(4, None), (4, 64), (3, 64)]:
    e_rtn, e_gptq = out_err(rtn_rowwise(W, bits, grp)), out_err(gptq(W, X, bits, grp))
    print(f"W{bits} group={grp}: RTN {e_rtn:.4f}  GPTQ {e_gptq:.4f}  ratio {e_gptq / e_rtn:.2f}")
    assert e_gptq < e_rtn
```

**Result on held-out activations:**

| Setting | RTN output error | GPTQ output error |
|---|---|---|
| W4 per-channel | 0.128 | 0.081 |
| W4 g64 | 0.109 | 0.070 |
| W3 g64 | 0.256 | 0.179 |

GPTQ gives **30–37% lower output error**. The gain comes from correlated inputs: GPTQ exploits $H$'s off-diagonal structure, which RTN ignores. The same mechanism is why GPTQ **overfits** when the calibration set does not match production. **Calibrate on your traffic**, e.g. bills, statutes, and agent transcripts rather than generic web text, and keep a held-out domain eval.

### 3.3 SmoothQuant and rotations — taming activation outliers

LLM activations have a few **systematic outlier channels**, 10–100× larger than the rest, that recur across tokens. Per-tensor or per-token activation scales are then dominated by them. Weights are easy to quantize by comparison.

**SmoothQuant** moves difficulty from $X$ to $W$ with an exact reparameterization:

$$
Y = XW^\top = \big(X\,\mathrm{diag}(s)^{-1}\big)\big(W\,\mathrm{diag}(s)\big)^\top,
\qquad
s_j = \frac{\max|X_{:,j}|^{\alpha}}{\max|W_{:,j}|^{\,1-\alpha}} .
$$

Here α≈0.5 balances the difficulty, and $s$ folds into the preceding LayerNorm or linear layer offline.

**Rotations** (QuaRot, SpinQuant) apply an orthogonal $R$ such that $Y = (XR)(WR)^\top$. A Hadamard $R$ mixes every channel into every other, so outliers are spread out and the distribution becomes near-Gaussian. This is the enabler for **W4A4 and 4-bit KV**.

```python
import numpy as np
from scipy.linalg import hadamard
from scipy.stats import kurtosis

def q8_tensor(x):                                         # per-tensor activation scale
    s = np.abs(x).max() / 127
    return np.clip(np.round(x / s), -128, 127) * s

def q8_rows(w):                                           # per-output-channel weight scales
    s = np.abs(w).max(axis=1, keepdims=True) / 127
    return np.clip(np.round(w / s), -128, 127) * s

def smoothquant_scales(X, W, alpha=0.5):
    ax, aw = np.abs(X).max(axis=0), np.abs(W).max(axis=0)
    return np.maximum(ax, 1e-8) ** alpha / np.maximum(aw, 1e-8) ** (1 - alpha)

def w8a8_err(X, W, s=None):
    Y = X @ W.T
    if s is not None:
        X, W = X / s, W * s
    return float(np.linalg.norm(Y - q8_tensor(X) @ q8_rows(W).T) / np.linalg.norm(Y))

rng = np.random.default_rng(2)
n, d, o = 2048, 512, 256
X = rng.standard_normal((n, d))
X[:, rng.choice(d, 6, replace=False)] *= 60.0             # systematic outlier channels
W = rng.standard_normal((o, d)) * 0.02

print(f"W8A8 naive            : {w8a8_err(X, W):.4f}")
for a in (0.3, 0.5, 0.7, 0.9):
    print(f"W8A8 SmoothQuant a={a} : {w8a8_err(X, W, smoothquant_scales(X, W, a)):.4f}")

Hm = hadamard(d) / np.sqrt(d)                              # orthonormal Walsh-Hadamard
Xr, Wr = X @ Hm, W @ Hm
assert np.allclose(X @ W.T, Xr @ Wr.T)                    # exact in full precision
print(f"activation kurtosis   : raw {kurtosis(X.ravel()):.1f} -> rotated {kurtosis(Xr.ravel()):.2f}")
print(f"W8A8 after rotation   : {w8a8_err(Xr, Wr):.4f}")
```

**Results.**
- Naive W8A8 error: 0.087.
- SmoothQuant: 0.027 at α=0.3, **0.015 at α=0.5–0.7**, and 0.026 at α=0.9. The U-shape in α is why the parameter is tuned per model.
- Hadamard rotation: excess kurtosis drops from **245 to 0.05**, which is Gaussian, and W8A8 error falls to **0.014**. Rotation needs no calibration here.

### 3.4 KV-cache quantization

Quantizing the KV cache to **FP8** (`--kv-cache-dtype fp8`; see 08.03 §3):
- halves KV memory, doubling concurrent tokens (≈214k → 428k in the 70B example there);
- halves attention bandwidth at long context.

Use calibrated per-tensor or per-head scales (LLM Compressor can produce them) rather than scale = 1.0 defaults. Validate **long-context retrieval** (01.04 needle and multi-hop tests) specifically, because KV error accumulates over positions. 4-bit and 2-bit KV (KIVI, NVFP4 KV, rotated KV) are viable only with validation per model.

### 3.5 What to pick

```
Is the GPU Blackwell? ──yes──► try NVFP4 (W4A4) or FP8; gate on evals; keep FP8 as the fallback
        │ no
Hopper/Ada? ──yes──► FP8 dynamic W8A8 + FP8 KV   (first move, near-lossless, minutes to produce)
        │               └─ memory-bound / single-GPU fit needed? add W4A16 (AWQ/GPTQ) candidate
        │ no (Ampere / older / consumer)
        └────────► W4A16 (AWQ or GPTQ, g128) for fit/latency · W8A8-INT8 (SmoothQuant) for throughput
Local/edge (CPU, Apple): GGUF k-quants (llama.cpp) or MLX 4-bit
Never skip: slice-level quality gates (§4) + a perf benchmark at YOUR batch/context profile (§5)
```

### 3.6 Producing a checkpoint with LLM Compressor

This is the canonical `oneshot` flow. Scheme names and arguments evolve between releases, so pin the version.

```python
# pip install llmcompressor   (pin the version; run on a GPU box)
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier

MODEL_ID = "your-org/bill-summarizer-8b"          # fine-tuned model from 08.01's bake-off
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# (a) FP8 dynamic: no calibration data, near-lossless on Hopper
fp8 = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=fp8)
model.save_pretrained("out/bill-8b-fp8", save_compressed=True); tok.save_pretrained("out/bill-8b-fp8")

# (b) W4A16 GPTQ: calibrate on *domain* text (bills, statutes, fiscal notes), not generic web data
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")   # reload fresh
# gptq = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
# oneshot(model=model, dataset=calib_ds, recipe=gptq, max_seq_length=4096, num_calibration_samples=512)
```

Serve it with `vllm serve out/bill-8b-fp8 --kv-cache-dtype fp8 ...`. vLLM reads the quantization config from the checkpoint. Record the scheme, the calibration-set hash, and the tool version in the **bundle manifest** (08.05 §2).

---

## 4. Validating a Quantized Model (the Part Teams Skip)

**Perplexity is necessary but not sufficient.** Quantization damage concentrates in:
- long-context retrieval;
- arithmetic and fiscal numbers;
- structured output validity;
- tool-call arguments;
- rare languages and domain jargon;
- reasoning chains, where small per-token drift compounds.

The ladder is:

1. **Teacher-forced divergence.** On identical inputs, compute $\mathrm{KL}(p_{ref}\,\Vert\,p_{q})$ per token and the **greedy flip rate**. This is cheap, sensitive, and catches broken kernels and bad scales immediately.
2. **Task evals by slice**, paired against the BF16 baseline on the *same items*, with the paired bootstrap from 06.01. Pass means the **lower CI bound of the delta ≥ −ε per slice**, not just in aggregate.
3. **Hard invariants:** JSON/schema validity, tool-call parse rate, refusal and guardrail behaviour (07.01). These must not regress at all.
4. **Performance at your profile.** Measure TTFT, ITL, and max QPS at SLO (08.03 §5) at your real prompt/output length distribution and concurrency. A scheme that is faster at batch 1 can be slower at batch 128 (§1).
5. **Shadow and canary** (08.05 §4) with online metrics (06.03).

```python
import numpy as np

def _log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def token_divergence(ref_logits, q_logits):
    """Teacher-forced: mean/p99 KL(ref || quant) per position and greedy top-1 flip rate."""
    lp, lq = _log_softmax(ref_logits), _log_softmax(q_logits)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    flips = ref_logits.argmax(-1) != q_logits.argmax(-1)
    return dict(mean_kl=float(kl.mean()), p99_kl=float(np.percentile(kl, 99)), flip_rate=float(flips.mean()))

def quant_gate(base, cand, slices, max_drop=0.01, n_boot=4000, alpha=0.05, seed=0, hard_checks=None):
    """Paired bootstrap per slice: PASS iff lower CI bound of mean(cand - base) >= -max_drop,
    on the SAME items. hard_checks: name -> (value, minimum) invariants (e.g. JSON validity)."""
    rng = np.random.default_rng(seed)
    base, cand, slices = map(np.asarray, (base, cand, slices))
    report, ok = {}, True
    for s in ["ALL", *sorted(set(slices.tolist()))]:
        m = np.ones_like(base, bool) if s == "ALL" else slices == s
        d = cand[m] - base[m]
        idx = rng.integers(0, len(d), (n_boot, len(d)))
        lo = float(np.percentile(d[idx].mean(1), 100 * alpha / 2))
        report[s] = dict(n=int(m.sum()), delta=float(d.mean()), ci_lo=lo, pass_=bool(lo >= -max_drop))
        ok &= report[s]["pass_"]
    for name, (value, limit) in (hard_checks or {}).items():
        report[name] = dict(value=value, limit=limit, pass_=value >= limit)
        ok &= report[name]["pass_"]
    return bool(ok), report

rng = np.random.default_rng(3)
V, T = 32000, 512
ref = rng.standard_normal((T, V)) * 3
print("small noise:", token_divergence(ref, ref + rng.standard_normal((T, V)) * 0.05))
print("large noise:", token_divergence(ref, ref + rng.standard_normal((T, V)) * 0.6))

n = 1200
sl = rng.choice(["summaries", "statute_qa", "fiscal_notes"], n, p=[0.45, 0.45, 0.10])
base = (rng.random(n) < 0.86).astype(float)
cand = base.copy()
cand[(sl == "fiscal_notes") & (rng.random(n) < 0.15)] = 0.0     # regression hidden in a small slice
ok, rep = quant_gate(base, cand, sl, max_drop=0.02, hard_checks={"json_valid_rate": (0.997, 0.995)})
for k, v in rep.items():
    print(k, v)
print("GATE:", "PASS" if ok else "FAIL")
assert rep["ALL"]["pass_"] and not rep["fiscal_notes"]["pass_"] and not ok
```

The planted regression makes the point. The **aggregate passes** (Δ = −1.3 pts, CI lower bound −2.0), but the **fiscal-notes slice fails** (Δ = −13.9 pts). That slice holds the number-heavy text quantization tends to hurt, and an aggregate-only gate would have shipped it.

---

## 5. Batching: Throughput, Latency, and Goodput

### 5.1 The roofline in numbers

```python
def decode_step_roofline(B, params_b, bytes_per_w, peak_tflops, bw_tbs, kv_bytes_per_seq=0.0):
    """One decode step at batch B: max(compute, memory). Returns (ms, regime, tok/s)."""
    flops = 2.0 * params_b * 1e9 * B
    bytes_ = params_b * 1e9 * bytes_per_w + B * kv_bytes_per_seq
    t_c, t_m = flops / (peak_tflops * 1e12), bytes_ / (bw_tbs * 1e12)
    t = max(t_c, t_m)
    return t * 1e3, ("compute" if t_c > t_m else "memory"), B / t

def crossover_batch(bytes_per_w, peak_tflops, bw_tbs):
    return (peak_tflops / bw_tbs) * bytes_per_w / 2.0

# H100 SXM, approximate dense specs: ~989 TFLOPS BF16, ~1979 FP8, 3.35 TB/s
for name, bpw, tf in [("BF16", 2.0, 989), ("FP8 W8A8", 1.0, 1979), ("INT4 W4A16", 0.5, 989)]:
    print(f"{name:10s} crossover B ~ {crossover_batch(bpw, tf, 3.35):5.0f}", end=" | ")
    for B in (1, 32, 256):
        ms, reg, tps = decode_step_roofline(B, 8, bpw, tf, 3.35)
        print(f"B={B}: {ms:4.2f} ms {reg[:3]} {tps:7.0f} tok/s", end="; ")
    print()

kv_seq = 2 * 32 * 8 * 128 * 2 * 2000       # Llama-3-8B-like: 128 KiB/token x 2k context
for B in (32, 256):
    ms, reg, tps = decode_step_roofline(B, 8, 2.0, 989, 3.35, kv_bytes_per_seq=kv_seq)
    print(f"BF16 + KV@2k B={B}: {ms:5.2f} ms ({reg}) {tps:6.0f} tok/s; KV/step {B * kv_seq / 1e9:.1f} GB")
```

Idealised upper bounds for an 8B model on one H100 (real kernels reach perhaps 60–80% of these):

| Scheme | Crossover $B^*$ | B=1 | B=256 |
|---|---|---|---|
| BF16 | ≈ 295 | 4.8 ms/step | still memory-bound |
| FP8 W8A8 | ≈ 295 (FP8 doubles both FLOPs and bytes/elt) | 2× BF16 | 2× BF16 |
| INT4 W4A16 | **≈ 74** | **4× BF16** | **compute-bound** at 4.1 ms, and **loses to FP8** at 2.4 ms |

**With KV at 2k context,** $B=256$ reads **67 GB of KV per step vs 16 GB of weights**. The step grows to 24.8 ms and throughput collapses from 53.6k to 10.3k tok/s. Long-context, high-concurrency workloads are **KV-bound**, so FP8 KV, prefix caching (08.02 §2), GQA/MLA models, and context budgets (01.04) are the levers.

### 5.2 Queueing: Little's law and the step-time model

**Little's law** in steady state: $L = \lambda W$. Here $L$ is the mean number of in-flight requests, $\lambda$ the arrival rate, and $W$ the mean time in system.

If the engine runs $L$ concurrent sequences with step time $t(B) = t_0 + kB$, then per-sequence ITL is $t(L)$ and decode throughput is $L/t(L)$. For mean output length $\bar{n}_{out}$ the system is stable only if

$$
\lambda\,\bar n_{\text{out}} \;<\; \max_{B \le B_{\max}} \frac{B}{t_0 + kB} \;=\; \frac{B_{\max}}{t_0 + k B_{\max}},
$$

and ITL SLO compliance requires $t_0 + kB \le \text{ITL}_{\text{SLO}}$, i.e. $B \le (\text{ITL}_{\text{SLO}} - t_0)/k$. **Goodput** = requests/s completed *within* TTFT and ITL SLOs. It is the metric to maximize, not raw tok/s (DistServe).

### 5.3 Static vs continuous batching — simulated

```python
import numpy as np

def simulate(policy, lam, n_req=3000, max_batch=64, t0=0.012, k=0.00025,
             prefill_per_tok=0.00002, seed=0):
    """Poisson arrivals; step time = t0 + k*B + prefill cost of newly admitted prompts.
    'continuous' admits into free slots every step (iteration-level scheduling, Orca/vLLM);
    'static' admits a new batch only when the current one fully drains."""
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1 / lam, n_req))
    out_len = np.clip(rng.lognormal(np.log(150), 0.8, n_req).astype(int), 5, 2000)
    in_len = np.clip(rng.lognormal(np.log(800), 0.6, n_req).astype(int), 20, 8000)
    t, nxt, running, queue, done, first_tok = 0.0, 0, [], [], {}, {}
    while len(done) < n_req:
        while nxt < n_req and arrivals[nxt] <= t:
            queue.append(nxt); nxt += 1
        admit = []
        if policy == "continuous" or not running:
            while queue and len(running) + len(admit) < max_batch:
                admit.append(queue.pop(0))
        running += [[i, out_len[i]] for i in admit]
        if not running:
            t = arrivals[nxt]; continue
        t += t0 + k * len(running) + prefill_per_tok * sum(in_len[i] for i in admit)
        for r in running:
            first_tok.setdefault(r[0], t)
            r[1] -= 1
            if r[1] == 0:
                done[r[0]] = t
        running = [r for r in running if r[1] > 0]
    lat = np.array([done[i] - arrivals[i] for i in range(n_req)])
    ttft = np.array([first_tok[i] - arrivals[i] for i in range(n_req)])
    return dict(tok_s=out_len.sum() / (max(done.values()) - arrivals[0]),
                p50_ttft=float(np.percentile(ttft, 50)), p95_ttft=float(np.percentile(ttft, 95)),
                p95_e2e=float(np.percentile(lat, 95)))

for lam in (4, 8, 12):
    for pol in ("static", "continuous"):
        r = simulate(pol, lam)
        print(f"lam={lam:2d} {pol:10s} tok/s={r['tok_s']:5.0f} p50 TTFT={r['p50_ttft']:6.2f}s "
              f"p95 TTFT={r['p95_ttft']:6.2f}s p95 e2e={r['p95_e2e']:6.2f}s")
assert simulate("continuous", 8)["p95_ttft"] < simulate("static", 8)["p95_ttft"]
```

| λ (req/s) | Static: tok/s · p95 TTFT | Continuous: tok/s · p95 TTFT |
|---|---|---|
| 4 | 765 · 45.9 s | 821 · **0.08 s** |
| 8 | 758 · 404 s (**unstable**) | **1,577** · 0.20 s |
| 12 | 774 · 507 s | 1,800 · 71 s (**past capacity**) |

What the table shows:
- Static batching caps out near 760 tok/s. Every batch waits for its **longest** member (lognormal output lengths), so slots sit idle, and the queue diverges already at λ=4 (demand ≈ 825 tok/s).
- Continuous batching **doubles capacity** and keeps TTFT sub-second until it approaches its own limit.
- At λ=12 it also saturates, which you would detect with `num_requests_waiting` (08.03 §6.3) and fix with replicas or shorter outputs.

### 5.4 Batching knobs and their trade-offs (vLLM terms, see 08.03 §4)

| Knob | ↑ increases | ↓ decreases | Guidance |
|---|---|---|---|
| `--max-num-seqs` (max B) | throughput, KV pressure | ITL (worse), preemptions | Set from the ITL SLO: $B \le (\text{ITL}_{SLO}-t_0)/k$ fitted from a sweep |
| `--max-num-batched-tokens` (token budget per step) | prefill throughput, TTFT for long prompts | ITL stability (big prefills stall decodes) | With **chunked prefill**, 2k–8k is typical for interactive; larger for offline |
| Chunked prefill | ITL p99 stability | slight TTFT increase | On for interactive mixes (Sarathi-Serve) |
| Priority scheduling | SLO for interactive tier | batch/offline latency | Two classes: interactive vs bulk |
| Prefill/decode **disaggregation** | independent scaling of TTFT vs ITL | complexity, KV transfer | At scale (llm-d / Dynamo, 08.03 §6.2); DistServe/Splitwise |
| Speculative decoding | ITL at low B | gains vanish at high B (compute-bound) | Enable for latency tier only (01.02 §5) |

### 5.5 Offline (bulk) batch inference

Examples of bulk work: nightly re-summarisation of all amended bills, back-filling embeddings after a model change (03.01), and eval runs. For these, **latency is irrelevant and cost per token is everything.** The recipe:

1. **Separate pool or priority class**, so bulk never shares an SLO with interactive traffic.
2. Use **vLLM offline** `LLM.generate(prompts, sampling_params)`, which schedules the whole list with maximum batch, or a provider **Batch API** (often ≈50% discount, 24 h window; 08.02 §4.2).
3. **Sort or group by shared prefix** (same system prompt and schema) so prefix caching hits. Sort by prompt length to reduce padding in any static stages.
4. Feed work from **Kafka** with idempotent keys (`bill_id:version:task`), commit offsets only after the result is persisted, and route failures to a dead-letter topic with bounded retries.
5. Checkpoint progress and emit throughput and cost per 1k items to the FinOps dashboard (08.05 §7).

```python
# Offline bulk generation with vLLM (GPU box). Assumes prompts share a system prompt -> prefix-cache hits.
from vllm import LLM, SamplingParams

llm = LLM(model="out/bill-8b-fp8", kv_cache_dtype="fp8", enable_prefix_caching=True,
          max_num_seqs=256, max_num_batched_tokens=16384)
params = SamplingParams(temperature=0.0, max_tokens=400)
SYSTEM = "You summarize Montana bills for legislators. Output JSON {summary, fiscal_impact, sections}."
bills = load_amended_bills()                               # your data access
prompts = [f"{SYSTEM}\n\nBILL {b.id} v{b.version}:\n{b.text}" for b in sorted(bills, key=lambda b: len(b.text))]
for b, out in zip(sorted(bills, key=lambda b: len(b.text)), llm.generate(prompts, params)):
    persist(b.id, b.version, out.outputs[0].text)          # idempotent upsert keyed on (id, version)
```

---

## 6. Production Challenges and Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **Quality cliff in one slice** | Aggregate evals flat; users report wrong fiscal numbers | Slice-level paired gates (§4); number-heavy and long-context slices mandatory; calibrate on domain data |
| **Calibration overfit (GPTQ)** | Great on the calib-like eval, worse on real traffic | Calibrate on sampled production prompts (PII-scrubbed; 07.01), hold out a domain eval; prefer AWQ/FP8 when unsure |
| **"Quantized is slower"** | W4A16 regresses throughput at high concurrency | Roofline check (§5.1): past $B^*$ use FP8/W8A8; benchmark at real concurrency |
| **Kernel not supported** | vLLM falls back to a slow path or fails to load | Match (GPU arch, scheme, group size, act-order) to supported kernels for the pinned vLLM version; CI load test |
| **Long-context degradation with FP8 KV** | Needle/multi-hop recall drops at 32k+ | Calibrated KV scales; keep BF16 KV for the long-context tier; test by position |
| **Structured-output / tool-call breakage** | JSON validity or argument accuracy dips | Hard invariants in the gate; constrained decoding (02.04) masks malformed tokens but not wrong values |
| **Non-determinism confusion** | Different outputs at different batch sizes | Batch-invariance is not guaranteed (batched kernels change reduction order); evaluate distributionally, pin seeds only for debugging |
| **MoE quantization** | Router or expert gates destabilise | Keep routers/gates and `lm_head` in higher precision (`ignore=[...]`); check expert load balance post-quant |
| **Tail latency from big prefills** | ITL p99 spikes when long prompts arrive | Chunked prefill + token budget; priority classes; disaggregation at scale |
| **Checkpoint provenance** | Can't reproduce which calib set/scheme is in prod | Record scheme, tool version, calib hash, eval report in the bundle manifest (08.05) |

---

## 7. Hands-On Projects

### Project 1 — Quantization Bake-Off for the Bill-Summarizer

**User stories**
- *As the platform owner*, I want the cheapest checkpoint that serves bill summaries within SLO without measurable quality loss, so that session-peak traffic fits on our GPU budget.
- *As a fiscal analyst*, I need fiscal numbers in summaries to be exactly right, whatever precision the model runs at.

**Acceptance criteria**
- Candidates are BF16 (baseline), FP8-dynamic + FP8 KV, W4A16-AWQ, and W4A16-GPTQ (domain calibration), plus NVFP4 if Blackwell is available. All are produced with a pinned LLM Compressor.
- Teacher-forced KL and flip rate are reported for each. Each candidate goes through `quant_gate` with slices {summaries, statute QA, fiscal notes, long bills > 16k tokens} and ε = 1 pt. JSON validity must stay ≥ baseline − 0.2 pt.
- Performance: max QPS at SLO (TTFT p95 ≤ 1.5 s, ITL p95 ≤ 60 ms) at the production length distribution, using 08.03's harness.
- A decision memo gives $/1M output tokens at SLO for each candidate (08.02 `call_cost` / 08.01 TCO) and names a winner plus a fallback.

**Step-by-step**
1. Sample 2k PII-scrubbed production prompts: 512 for calibration, the rest for eval. Freeze the eval set with a hash (06.02).
2. Produce the checkpoints (§3.6) and log scheme, calibration hash, and tool version.
3. Dump logits on 200 held-out prompts for each checkpoint and run `token_divergence`. Any candidate with a flip rate far above FP8's is investigated first; it may be a kernel or scale bug.
4. Run the task evals, with LLM-as-judge plus exact-match on extracted fiscal figures, then run `quant_gate`.
5. Sweep concurrency on each passing candidate and fit $t(B)=t_0+kB$. Compute max QPS at SLO and cost.
6. Write the memo. Include the roofline explanation of why the winner wins at your concurrency.

### Project 2 — GPTQ/AWQ From Scratch, Verified Against the Library

**User stories**
- *As an ML engineer*, I want to understand exactly what the quantizer does to our weights, so I can debug accuracy regressions rather than guess.

**Acceptance criteria**
- Implement per-layer GPTQ (from §3.2, extended with act-order and lazy block updates) and AWQ scale search in PyTorch for one decoder layer of a small open model (≤ 1.5B).
- Layer output error is within 10% of LLM Compressor's on the same calibration batch.
- The full model is quantized with your implementation. Its perplexity on a domain corpus is within 0.2 of the library's, and both are reported against BF16.
- An ablation shows the effect of group size (32/64/128), act-order on/off, dampening (0.001–0.1), and calibration domain (web vs legislative) on domain perplexity.

**Step-by-step**
1. Hook the inputs of each `nn.Linear` layer and accumulate $H = 2X^\top X$ over the calibration batches (in float64 for stability).
2. Port `gptq()` to torch. Add act-order (permute columns by $\mathrm{diag}(H)$ descending, and un-permute when saving) and block updates (128 columns).
3. AWQ: for each linear layer, grid-search $\alpha \in [0,1]$ with $s = \bar{|x|}^{\alpha}$, and minimize $\lVert Q(W\,\mathrm{diag}(s))\,(\mathrm{diag}(s)^{-1}x) - Wx\rVert$.
4. Pack to the `compressed-tensors` W4A16 format, or evaluate with fake-quant weights.
5. Run the ablation grid and plot error against effective bits. Write up the calibration-domain finding.

### Project 3 — Goodput Tuning and a Bulk Pipeline for Session Peaks

**User stories**
- *As the on-call engineer*, I want interactive chat to hold its SLO while nightly bulk summarisation runs on the same cluster during the legislative session.
- *As FinOps*, I want bulk jobs to run at minimum cost per item.

**Acceptance criteria**
- A concurrency sweep produces a fitted $t(B)$ model. `--max-num-seqs` and `--max-num-batched-tokens` are chosen analytically (§5.2) and validated by load test. Interactive goodput is ≥ 95% of requests within SLO at the forecast session peak λ.
- The bulk pipeline is Kafka → vLLM offline workers (or a provider Batch API), with idempotent keys, a dead-letter queue, and resume-from-checkpoint. Throughput and cost per 1k bills are reported, and the prefix-cache hit rate is ≥ 80%.
- In a chaos test with the bulk job at full load, interactive p95 TTFT degrades by ≤ 10%. This is achieved with priority classes or separate pools.
- A dashboard shows goodput, `num_requests_waiting`, KV usage, preemptions, and cost per 1k items.

**Step-by-step**
1. Replay production traces (lengths and inter-arrivals) with 08.03's load generator. Sweep B and fit $t_0$ and $k$.
2. Derive the B cap from the ITL SLO and a token budget from TTFT targets. Validate with the simulator (§5.3), then with real load.
3. Build the bulk consumer: sort by prefix and length, use `LLM.generate` in chunks of about 2k prompts, upsert results idempotently, commit offsets after persisting, and send failures to a DLQ.
4. Isolate traffic with either (a) a separate node pool with KEDA scaling to zero outside batch windows (08.03 §6.2), or (b) priority scheduling on a shared pool. Measure both options.
5. Run the chaos test, then tune and document the chosen isolation strategy.

---

## 8. Foundational Papers and Tooling

**Papers (exact titles)**
- Frantar et al., 2022 — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- Frantar & Alistarh, 2022 — *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning*
- Lin et al., 2023 — *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
- Xiao et al., 2022 — *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*
- Dettmers et al., 2022 — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*
- Dettmers et al., 2023 — *QLoRA: Efficient Finetuning of Quantized LLMs*
- Ashkboos et al., 2024 — *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*
- Liu et al., 2024 — *SpinQuant: LLM Quantization with Learned Rotations*
- Micikevicius et al., 2022 — *FP8 Formats for Deep Learning*
- Rouhani et al., 2023 — *Microscaling Data Formats for Deep Learning*
- Liu et al., 2024 — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*
- Kurtic et al., 2024 — *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization*
- Yu et al., 2022 — *Orca: A Distributed Serving System for Transformer-Based Generative Models*
- Kwon et al., 2023 — *Efficient Memory Management for Large Language Model Serving with PagedAttention*
- Agrawal et al., 2024 — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- Zhong et al., 2024 — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*
- Williams, Waterman & Patterson, 2009 — *Roofline: An Insightful Visual Performance Model for Multicore Architectures*

**Tooling**
- **Producing checkpoints:** LLM Compressor + `compressed-tensors` (vLLM), NVIDIA TensorRT Model Optimizer (ModelOpt), AutoAWQ/AutoGPTQ (legacy; prefer LLM Compressor), Intel AutoRound, bitsandbytes (QLoRA / on-the-fly 4/8-bit), llama.cpp GGUF quantizers, MLX.
- **Kernels and engines:** vLLM (Marlin/Machete W4A16, FP8/FP4 CUTLASS kernels), TensorRT-LLM, SGLang.
- **Evaluation:** lm-evaluation-harness, your 06.01 eval harness, vLLM `benchmarks/` and GuideLLM for load, NVIDIA Nsight Systems for profiling.
- **Batch and bulk:** vLLM offline `LLM` API, Ray Data + vLLM, provider Batch APIs, Kafka consumers with DLQ.

**Image prompt (roofline figure):** *"Clean technical roofline chart on white background: log-log axes, x = arithmetic intensity (FLOP/byte), y = attainable TFLOPS. A sloped memory-bandwidth line meets a flat compute ceiling at a labeled 'ridge point ≈ 295 FLOP/byte (H100 BF16)'. Plot three dots per format (BF16, FP8, INT4-W4A16) at batch sizes 1, 32, 256, colour-coded; INT4 dots cross the ridge earliest. Annotate 'decode = memory-bound', 'large-batch prefill = compute-bound'. Minimal flat style, sans-serif labels."*

**Image prompt (continuous batching):** *"Side-by-side timeline diagram. Left 'static batching': 4 GPU slot rows with requests of varying length as coloured bars; after short requests finish, grey idle gaps until the longest finishes; queued requests wait outside. Right 'continuous batching': same slots, new requests slot in immediately as others finish, no gaps. Labels: 'iteration-level scheduling', 'idle slots', 'queue'. Flat vector style, white background."*
