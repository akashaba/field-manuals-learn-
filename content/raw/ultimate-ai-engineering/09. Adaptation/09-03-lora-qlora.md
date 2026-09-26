# 09.03 — LoRA / QLoRA (Low-Rank Adaptation and 4-Bit Fine-Tuning)

> **Module 9: Adaptation** · Subtopic 3 of 4
> **Prerequisites:** 09.01 (SFT, memory math), 09.02 (PEFT families and adapter operations), 08.04 §2 (number formats, blockwise quantization).
> **Outcome:** you can derive LoRA's forward pass, gradients, and scaling rules. You can choose rank, α, target modules and learning rate from evidence, pick among LoRA variants (rsLoRA, DoRA, PiSSA, LoRA+), and implement NF4 and double quantization from first principles. You can run QLoRA on a single GPU, and decide how to merge and serve a QLoRA adapter without silent quality loss.

---

## 1. LoRA, Precisely

For a frozen linear layer $W_0 \in \mathbb{R}^{d_{out}\times d_{in}}$, LoRA (Hu et al., 2021) learns a low-rank update:

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\,B A\,x,
\qquad A \in \mathbb{R}^{r\times d_{in}},\; B \in \mathbb{R}^{d_{out}\times r},\; r \ll \min(d_{in}, d_{out}).
$$

**Initialisation:** $A \sim \mathcal{N}(0, \sigma^2)$ (Kaiming-uniform in PEFT) and $B = 0$. So $\Delta W = 0$ at step 0: **training starts exactly at the base model**, and early gradients flow only into $B$.

**Gradients.** With $G = \partial\mathcal{L}/\partial h$ (batch $\times d_{out}$) and $s = \alpha/r$:

$$
\frac{\partial \mathcal{L}}{\partial B} = s\,G^{\top} (A x)^{\top},\qquad
\frac{\partial \mathcal{L}}{\partial A} = s\,B^{\top} G^{\top} x^{\top}.
$$

Gradients and Adam state exist only for $r(d_{in}+d_{out})$ parameters instead of $d_{in} d_{out}$. The activations needed for the backward pass barely change, because the input activation $x$ is stored anyway.

**Merging.** $W' = W_0 + sBA$ gives zero inference overhead, and unmerging is exact in FP32. In BF16, merging rounds $W'$, which is usually harmless; check it with an eval (09.02 §6).

**Why low rank suffices.** Fine-tuning updates have low **intrinsic dimension** (Aghajanyan et al.): task adaptation lives in a small subspace. LoRA enforces that prior as a hard constraint, which is also why it **"learns less and forgets less"** (Biderman et al., 2024). It changes the base less, and absorbs less of large new-knowledge datasets.

---

## 2. Scaling: α, Rank, and Learning Rate Interact

The scale $s$ multiplies the update, so **rank, α, and learning rate are coupled**. Two conventions exist:
- **Standard LoRA** uses $s = \alpha / r$. It keeps the *initial function change* comparable across ranks under SGD-style analysis.
- **rsLoRA** (Kalajdzievski, 2023) uses $s = \alpha / \sqrt{r}$. It argues that with $\alpha/r$ the updates shrink as $r$ grows, so high ranks "collapse" and learn no faster than low ranks.

Measure it rather than trust either claim. The experiment below uses Adam on a regression toward a rank-32 target shift at a fixed learning rate:

```python
import torch

torch.set_num_threads(1)

def lora_probe(r, scaling, steps=1, lr=1e-3, d=512, alpha=16, seed=0, n=256):
    """Returns (||s B A||_F, loss) after `steps` Adam steps; target = W0 + rank-32 shift."""
    g = torch.Generator().manual_seed(seed)
    W0 = torch.randn(d, d, generator=g) / d ** 0.5
    Wt = W0 + torch.randn(d, 32, generator=g) @ torch.randn(32, d, generator=g) / d
    X = torch.randn(n, d, generator=g)
    A = (torch.randn(r, d, generator=g) / d ** 0.5).requires_grad_(True)
    B = torch.zeros(d, r, requires_grad=True)
    s = alpha / r if scaling == "alpha/r" else alpha / r ** 0.5
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(steps):
        loss = ((X @ (W0 + s * B @ A).T - X @ Wt.T) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((s * B @ A).norm()), float(loss)

ranks = (4, 16, 64, 256)
res = {sc: [(lora_probe(r, sc)[0], lora_probe(r, sc, steps=20)[1]) for r in ranks] for sc in ("alpha/r", "alpha/sqrt(r)")}
for sc, rows in res.items():
    print(f"{sc:14s}", "  ".join(f"r={r}: |dW|1={n:.3f} loss@20={l:.4f}" for r, (n, l) in zip(ranks, rows)))
assert res["alpha/r"][0][0] > 5 * res["alpha/r"][-1][0]                  # alpha/r: first-step update shrinks with r
assert max(n for n, _ in res["alpha/sqrt(r)"]) < 1.5 * min(n for n, _ in res["alpha/sqrt(r)"])
assert res["alpha/sqrt(r)"][-1][1] < res["alpha/r"][-1][1]               # high rank learns faster with rsLoRA
```

| Scaling | First-step $\lVert\Delta W\rVert_F$, r = 4 → 256 | Loss after 20 steps, r = 4 → 256 (start 0.062) |
|---|---|---|
| $\alpha/r$ | 0.176 → 0.091 → 0.047 → **0.026** (∝ $1/\sqrt r$) | 0.056 → 0.048 → 0.044 → **0.042** |
| $\alpha/\sqrt r$ (rsLoRA) | 0.35 → 0.36 → 0.38 → 0.42 (≈ constant) | 0.056 → 0.042 → 0.018 → **0.0045** |

**Interpretation.** At a fixed learning rate, $\alpha/r$ makes extra rank almost useless early in training. The update magnitude falls as $1/\sqrt r$ under Adam, so a higher rank needs a proportionally higher LR. rsLoRA removes that coupling.

Thinking Machines' *LoRA Without Regret* (Schulman et al., 2025) reports that in LLM SFT with $\alpha/r$ the **optimal LR is approximately rank-independent** over a full run. Their regime (large models, long training, tuned α) differs from this toy. **The practical rule: when you change rank, either use rsLoRA or re-sweep LR, and never compare ranks at a single untuned LR.**

---

## 3. Evidence-Based Defaults

The consolidated findings come from LoRA Without Regret (Thinking Machines, Sept 2025; summarised in TRL's docs), Biderman et al. (2024), and the QLoRA and DoRA papers:

| Knob | Recommendation | Evidence / reason |
|---|---|---|
| **Target modules** | **All linear layers** (`"all-linear"`), MLP included | Attention-only LoRA underperforms *even at matched parameter count*; the MLP holds ~2/3 of the budget and most of the benefit |
| **Rank (SFT)** | 16–64 for narrow tasks; **up to 256** for post-training-scale data | Capacity must match the information in the dataset; LoRA falls behind full FT when data exceeds capacity |
| **Rank (RL / GRPO)** | **1–32** | Policy-gradient RL conveys roughly one bit per episode, so tiny ranks suffice |
| **Learning rate** | **≈10× the full-FT LR** (e.g. 1e-4–2e-4 for 7–8B SFT) | LoRA's optimal LR is consistently higher |
| **Batch size** | Keep the effective batch **modest (< 32 sequences)** where feasible | LoRA tolerates large batches worse than full FT, and higher rank doesn't fix it |
| **α** | 16–32 with $\alpha/r$ (or rsLoRA); treat α and LR as one knob | See §2 |
| **Dropout** | 0–0.05 | Small datasets may benefit; often unnecessary |
| **Epochs** | 1–3 | Watch the task metric; LoRA overfits small data too |

---

## 4. LoRA Variants Worth Knowing

| Variant | Change | When it helps | PEFT flag |
|---|---|---|---|
| **rsLoRA** | $s=\alpha/\sqrt r$ | High ranks (≥ 64) | `use_rslora=True` |
| **DoRA** | $W' = m \odot \dfrac{W_0 + BA}{\lVert W_0 + BA\rVert_{c}}$: learn the magnitude $m$ (per column) and the direction separately | Low ranks, closer to full-FT learning dynamics; ~10–30% slower training | `use_dora=True` |
| **PiSSA** | Initialise $A,B$ from the **top-r singular vectors** of $W_0$; residual $W_0 - BA$ is frozen | Faster convergence; with quantization, the residual quantizes better | `init_lora_weights="pissa"` |
| **OLoRA / CorDA / EVA** | Orthonormal (QR), context-oriented, or data-driven (activation SVD) initialisation | Faster or more stable starts; EVA adapts ranks to the data | `init_lora_weights="olora"` / `"corda"` / `"eva"` |
| **LoRA+** | Separate LR for $B$: $\eta_B = \lambda\,\eta_A$ with $\lambda \approx 16$ | Speeds up training at wide layers | optimizer param groups (PEFT `create_loraplus_optimizer`) |
| **LoftQ** | Initialise LoRA to **compensate quantization error**: alternate $Q = \mathrm{quant}(W - BA)$ and SVD of $W - Q$ | QLoRA at 2–4 bits | `init_lora_weights="loftq"` |
| **AdaLoRA** | SVD-parameterised; prune singular values to reallocate rank across layers | Tight parameter budgets | `AdaLoraConfig` |

Every variant must start at (numerically) the base model's function. This is checked below on a small model. PiSSA and OLoRA move weight into the adapter and subtract it from the base, so their outputs match to about 1e-7 rather than exactly.

```python
import copy, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

BASE_ID = "meta-llama/Llama-3.1-8B-Instruct"
torch.manual_seed(0)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.float32).eval()
x = torch.randint(0, base.config.vocab_size, (2, 24))
with torch.no_grad():
    ref = base(x).logits

for name, kw in {"lora (B=0)": {}, "rslora": dict(use_rslora=True), "dora": dict(use_dora=True),
                 "pissa": dict(init_lora_weights="pissa"), "olora": dict(init_lora_weights="olora")}.items():
    m = get_peft_model(copy.deepcopy(base), LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", **kw)).eval()
    with torch.no_grad():
        d = float((m(x).logits - ref).abs().max())
    print(f"{name:12s} trainable={m.get_nb_trainable_parameters()[0]:7d}  max|out - base| at init = {d:.2e}")
    assert d < 1e-3, name
```

DoRA adds one magnitude vector per target layer, which is visible as the small increase in trainable parameters.

---

## 5. QLoRA: Fine-Tuning on a 4-Bit Frozen Base

QLoRA (Dettmers et al., 2023) freezes the base in **4-bit NormalFloat (NF4)**. It dequantizes blocks on the fly to BF16 for each matmul and trains BF16 LoRA adapters on top:

$$
h = \mathrm{dequant}\!\big(W_0^{\text{NF4}}\big)\,x + \tfrac{\alpha}{r} B A x .
$$

Three ingredients make it work:

1. **NF4.** A 4-bit codebook built from **quantiles of $\mathcal{N}(0,1)$**, which is information-theoretically well matched to normally distributed weights. It is asymmetric so that **0 is exactly representable**, and used with absmax scaling per block of 64.
2. **Double quantization.** The per-block FP32 absmax constants (32/64 = 0.5 bits/param) are themselves quantized to 8 bits in blocks of 256. The cost becomes $8/64 + 32/(64\cdot256) \approx 0.127$ bits/param, **4.127 bits total**.
3. **Paged optimizers.** Optimizer state lives in CUDA unified memory and is paged to CPU during memory spikes (long sequences), avoiding OOM crashes.

```python
import numpy as np
from scipy.stats import norm

def nf4_codebook(offset=0.9677083):
    """NormalFloat4: N(0,1) quantiles, 8 positive + 7 negative levels + exact zero, normalised to [-1, 1]."""
    pos = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])
    neg = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
    v = np.sort(np.concatenate([pos, [0.0], neg]))
    return v / np.abs(v).max()

def quant_blockwise(w, codebook, block=64):
    """Absmax-scaled blockwise quantization to the nearest codebook value. Returns (dequantized, absmax)."""
    b = w.reshape(-1, block)
    absmax = np.abs(b).max(axis=1, keepdims=True)
    idx = np.abs((b / absmax)[..., None] - codebook).argmin(-1)
    return (codebook[idx] * absmax).reshape(w.shape), absmax.ravel()

def bits_per_param(block=64, scale_bits=32, dq_block=None, dq_bits=8):
    if dq_block is None:
        return 4 + scale_bits / block
    return 4 + dq_bits / block + 32 / (block * dq_block)       # double quantization of the absmax constants

NF4 = nf4_codebook()
BNB_NF4 = np.array([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0])   # bitsandbytes table
assert np.allclose(NF4, BNB_NF4, atol=1e-4)
INT4 = np.arange(-7, 8) / 7.0
FP4 = np.array(sorted({s * v for s in (-1, 1) for v in (0, .5, 1, 1.5, 2, 3, 4, 6)})) / 6.0   # E2M1

rng = np.random.default_rng(0)
W = rng.standard_normal((4096, 1024)) * 0.02
errs = {}
for name, cb in (("INT4", INT4), ("FP4 E2M1", FP4), ("NF4", NF4)):
    errs[name] = np.mean((W - quant_blockwise(W, cb)[0]) ** 2) / np.mean(W ** 2)
    print(f"{name:9s} block-64 relative MSE = {errs[name]:.5f}")
print("bits/param: plain", bits_per_param(), "| double-quant", round(bits_per_param(dq_block=256), 3))
assert errs["NF4"] < errs["FP4 E2M1"] < errs["INT4"] and abs(bits_per_param(dq_block=256) - 4.127) < 1e-3
```

The reconstruction **matches bitsandbytes' NF4 table to 1e-4**. On Gaussian weights with block-64 absmax scaling, relative MSE is:

| Format | Relative MSE |
|---|---|
| INT4 | 0.0116 |
| FP4 E2M1 | 0.0112 |
| **NF4** | **0.0085** (27% below INT4) |

Double quantization gives exactly **4.127 bits/param**, the constant used in 09.01 §3's memory model.

### 5.1 What fits where

| Model | LoRA (BF16 base) static | QLoRA static + r=16 all-linear adapters + activations (4k ctx, checkpointing, chunked logits) | Single GPU |
|---|---|---|---|
| 8B | 16.7 GB | ≈ 6.5 GB | 16–24 GB card |
| 70B | 141 GB | 36.4 GB (NF4) + 3.3 GB (207M adapter params × 16 B) + 6.5 GB ≈ **46 GB** | **1× 80 GB** (48 GB is tight) |

**Speed.** QLoRA is typically **~30–50% slower per step** than BF16 LoRA because of dequantization, and single-GPU runs are slower than a sharded multi-GPU setup. When BF16 LoRA fits, use it. Choose QLoRA when memory is the constraint.

### 5.2 Running QLoRA

```python
# SKIP-TEST: requires CUDA + bitsandbytes
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"                 # pin the revision; check the licence
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2", device_map={"": 0})
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)   # casts norms, enables input grads
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
args = SFTConfig(output_dir="out/qlora-70b-bill", assistant_only_loss=True, max_length=4096, packing=True,
                 learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=0.03, num_train_epochs=2,
                 per_device_train_batch_size=1, gradient_accumulation_steps=16, gradient_checkpointing=True,
                 optim="paged_adamw_8bit", bf16=True, logging_steps=10, report_to="none")
SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, peft_config=peft_cfg).train()
```

### 5.3 Merging and serving a QLoRA adapter: evaluate the artefact you ship

The adapter was trained against the *quantized* base $\tilde W_0 = \mathrm{NF4}(W_0)$, but you have three serving options:
- **(a) Unmerged on the 4-bit base.** This is exactly what was trained, and slow-ish unless the engine has fast 4-bit + LoRA kernels.
- **(b) Merged into the original BF16 weights,** $W_0 + sBA$.
- **(c) Merged, then re-quantized** for serving (e.g. to NF4, AWQ, or FP8; 08.04).

These are **different models**. The experiment below trains a LoRA against an NF4 base on a task whose true update is rank 4, then scores all three:

```python
import numpy as np, torch
from scipy.stats import norm

torch.set_num_threads(1)

def nf4_codebook(offset=0.9677083):
    pos = norm.ppf(np.linspace(offset, 0.5, 9)[:-1]); neg = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
    v = np.sort(np.concatenate([pos, [0.0], neg])); return v / np.abs(v).max()

def nf4(w, block=64):
    cb = nf4_codebook(); b = w.reshape(-1, block); a = np.abs(b).max(1, keepdims=True)
    return (cb[np.abs((b / a)[..., None] - cb).argmin(-1)] * a).reshape(w.shape)

rng = np.random.default_rng(0); torch.manual_seed(0)
d, r, n = 256, 16, 2048
W0 = rng.standard_normal((d, d)) * 0.05
dW = rng.standard_normal((d, 4)) @ rng.standard_normal((4, d)) * 0.01       # true task update (rank 4)
Q = nf4(W0)                                                                  # frozen 4-bit base (dequantized view)
X, Xte = torch.randn(n, d), torch.randn(n, d)
T = lambda a: torch.tensor(a, dtype=torch.float32)
Yt, Yte = X @ T(W0 + dW).T, Xte @ T(W0 + dW).T

A = (torch.randn(r, d) / d ** 0.5).requires_grad_(True); B = torch.zeros(d, r, requires_grad=True); s = 2.0
opt = torch.optim.Adam([A, B], lr=3e-3)
for _ in range(1500):                                                        # QLoRA: train LoRA on the NF4 base
    loss = ((X @ (T(Q) + s * B @ A).T - Yt) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
dL = (s * B @ A).detach().numpy()

def rel_err(W):
    with torch.no_grad():
        return float(((Xte @ T(W).T - Yte) ** 2).mean() / (Yte ** 2).mean())

res = {"4-bit base, no adapter": rel_err(Q), "(a) unmerged: NF4(W0) + BA": rel_err(Q + dL),
       "(b) merged into BF16: W0 + BA": rel_err(W0 + dL), "(c) re-quantized: NF4(NF4(W0) + BA)": rel_err(nf4(Q + dL))}
for k, v in res.items():
    print(f"{k:38s} relative error {v:.5f}")
assert res["(b) merged into BF16: W0 + BA"] < res["(a) unmerged: NF4(W0) + BA"] < res["(c) re-quantized: NF4(NF4(W0) + BA)"]
```

| Serving artefact | Relative error |
|---|---|
| 4-bit base, no adapter | 0.155 |
| (a) Unmerged on NF4 base (as trained) | 0.0059 |
| **(b) Merged into original BF16** | **0.0013** |
| (c) Merged, then re-quantized to NF4 | 0.0146 |

Reading the results:
- **(b) wins here.** The adapter learned the task update, and a rank-16 adapter can't absorb NF4's full-rank quantization error. Merging into the original weights removes that error entirely.
- **(c) is worst.** Re-quantizing introduces *new* quantization error that the adapter never saw.

Real models can behave differently, e.g. when an adapter partially compensated the low-rank structure of the quantization error. So the rule is operational, not theoretical: **evaluate the exact artefact you serve**, with 08.04 §4's slice-level gates. If you must serve 4-bit, prefer quantizing the merged BF16 model with a proper method (AWQ, GPTQ, FP8, NVFP4; 08.04) and calibration data, then gate it. Don't naively re-round to NF4.

---

## 6. Production Challenges and Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **Attention-only targets** | LoRA underperforms full FT | `target_modules="all-linear"` |
| **Rank ↑ doesn't help** | r=128 no better than r=16 | LR too low for $\alpha/r$ at high rank: use rsLoRA or re-sweep LR (§2); or the task simply needs little capacity |
| **LoRA ≪ full FT on a big dataset** | Plateau above full-FT loss | Capacity-limited: raise rank (256+), or switch to full FT |
| **Large-batch degradation** | Worse results at big effective batches | Smaller effective batch, more steps |
| **QLoRA OOM on long sequences** | Spikes at long examples | Paged optimizers, gradient checkpointing, chunked loss, length-bucketed batches |
| **Slow QLoRA** | 2× the step time of LoRA | Use BF16 LoRA if it fits; FlashAttention, packing, Unsloth-style fused kernels |
| **Merged model differs from adapter eval** | Quality drop after merge or quantization | Evaluate the served artefact; §5.3 options; gate with slices |
| **New special tokens not learned** | Tool or format tokens ignored | Train embeddings/`lm_head` via `modules_to_save` or `trainable_token_indices` |
| **fp16 overflow** | NaN losses | BF16 compute dtype; avoid fp16 on modern GPUs |
| **Adapter on wrong base revision** | Mysterious degradation | Base digest in adapter metadata; checked at load (09.02 §6) |

---

## 7. Hands-On Projects

### Project 1 — Reproduce "LoRA vs Full FT" on Your Data

**User stories**
- *As the ML lead*, I want to know, for our legislative tasks, when LoRA matches full fine-tuning and what rank, LR, and targets to standardise on.

**Acceptance criteria**
- Two datasets: a narrow task (1–3k bill summaries) and a broad one (≥ 50k mixed instruction examples from legislative services workflows plus general data).
- The grid covers:
  - full FT (LR sweep);
  - LoRA r ∈ {4, 16, 64, 256}, each with its own LR sweep, under both $\alpha/r$ and rsLoRA;
  - attention-only vs all-linear;
  - effective batch {8, 32, 128}.
- Deliverables: loss-vs-steps curves, a task metric with 95% CIs, and a finding for each claim in §3 (confirmed or refuted on your data).
- The standard config per task class goes into a shared template.

**Step-by-step**
1. Freeze the datasets and eval sets (06.02), and build the TRL training script with a config grid (Hydra/YAML).
2. Run the LR sweeps first (short runs), then full runs at the best LR per configuration.
3. Plot final metric against trainable parameters, and learning curves against steps.
4. Test each §3 claim explicitly, then write up the results and templates.

### Project 2 — QLoRA a 70B Model on One GPU, and Ship It

**User stories**
- *As the platform owner*, I want to adapt a 70B open model for statute Q&A with our budget of a single 80 GB GPU, then serve it efficiently.

**Acceptance criteria**
- QLoRA training fits on 1×80 GB (NF4, double quant, paged 8-bit AdamW, 4k context), with a measured peak memory within 15% of the §5.1 estimate.
- The tuned model beats the prompted base on the held-out statute-QA eval (with RAG context, RAFT-style; 09.01 §1) with CI excluding 0.
- The three serving artefacts from §5.3 (unmerged; merged BF16; merged then AWQ/FP8 via LLM Compressor) are evaluated with `quant_gate` (08.04 §4). The chosen artefact passes all slices and hard invariants.
- It is deployed on vLLM (08.03) with a bundle manifest recording the base digest, adapter hash, merge and quant method, and eval reports.

**Step-by-step**
1. Build the dataset with retrieved contexts from the production retriever, including distractors.
2. Run QLoRA training (§5.2) and log memory and throughput.
3. Evaluate the unmerged adapter, then merge into the BF16 weights (on a large-RAM CPU box or across GPUs).
4. Quantize the merged model with LLM Compressor (08.04 §3.6) and evaluate all three artefacts.
5. Choose, gate, deploy, and document the result.

### Project 3 — LoRA and NF4 from Scratch, Verified Against the Libraries

**User stories**
- *As an ML engineer*, I want to own the implementation details so I can debug numerical issues and evaluate new variants.

**Acceptance criteria**
- A pure-PyTorch `LoRALinear` with merge/unmerge, rsLoRA, and DoRA. Outputs match PEFT to within 1e-5 for identical weights.
- An NF4 quantize/dequantize implementation with double quantization, a "QLoRA linear" (dequantize-then-matmul plus LoRA), and gradients checked by `torch.autograd.gradcheck` on small shapes.
- The NF4 codebook matches bitsandbytes. Dequantized weights match `bitsandbytes.functional.dequantize_4bit` to within 1e-3 relative error on a real checkpoint layer (on a GPU box).
- A mini fine-tune of a ≤ 1.5B model with your QLoRA layer reaches within 2% of the PEFT+bitsandbytes eval loss.

**Step-by-step**
1. Implement and unit-test `LoRALinear`, including merge exactness in FP32.
2. Implement NF4 and double quantization, then compare the storage bits with the §5 math.
3. Write a custom `autograd.Function` for dequantize-matmul, run gradcheck, then swap it into the model's linears.
4. Train a small model and compare against the libraries.
5. Write up the discrepancies and fixes.

---

## 8. Foundational Papers (exact titles)

- Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*
- Dettmers et al., 2023 — *QLoRA: Efficient Finetuning of Quantized LLMs*
- Kalajdzievski, 2023 — *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (rsLoRA)
- Liu et al., 2024 — *DoRA: Weight-Decomposed Low-Rank Adaptation*
- Meng, Wang & Zhang, 2024 — *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models*
- Hayou, Ghosh & Yu, 2024 — *LoRA+: Efficient Low Rank Adaptation of Large Models*
- Li et al., 2023 — *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models*
- Zhang et al., 2023 — *Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning* (AdaLoRA)
- Biderman et al., 2024 — *LoRA Learns Less and Forgets Less*
- Aghajanyan, Zettlemoyer & Gupta, 2020 — *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*
- Dettmers et al., 2022 — *8-bit Optimizers via Block-wise Quantization*
- Schulman & Thinking Machines Lab, 2025 — *LoRA Without Regret* (blog post / technical report)

## 9. Essential Tooling

| Tool | Use |
|---|---|
| **PEFT 0.21** | LoRA and its variants, init schemes (PiSSA/OLoRA/EVA/CorDA/LoftQ), merge/unload |
| **bitsandbytes** | NF4/FP4 quantization, 8-bit and paged optimizers |
| **TRL 1.13** | `peft_config` in every trainer; QLoRA via `quantization_config` |
| **Unsloth** | Fused kernels for fast, memory-lean LoRA/QLoRA |
| **Axolotl, LLaMA-Factory, torchtune** | Recipe-driven LoRA/QLoRA training |
| **vLLM** (`--enable-lora`), LoRAX | Adapter serving |
| **LLM Compressor** (08.04) | Quantizing merged models properly for serving |

**Image prompt (LoRA):** *"Technical diagram: input vector x enters two parallel paths — top path a large grey frozen matrix W0 (d×d, snowflake icon), bottom path a narrow orange matrix A (r×d) then orange matrix B (d×r), scaled by α/r; outputs summed into h. Annotations: 'B initialised to 0', 'trainable r(d_in+d_out)', 'merge: W0 + (α/r)BA'. Clean flat vector, white background."*

**Image prompt (NF4):** *"Chart: standard normal bell curve with 16 vertical lines at NF4 quantile levels (denser near zero, sparse in tails), compared below with 16 evenly spaced INT4 levels; caption 'NF4 places levels where weights are'. Minimal technical style."*
