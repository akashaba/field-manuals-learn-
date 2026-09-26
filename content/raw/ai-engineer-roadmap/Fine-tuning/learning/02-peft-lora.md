# 02 — Parameter-Efficient Fine-Tuning: LoRA and QLoRA

> **Module goal:** Understand why nobody full-parameter fine-tunes 8B+ models anymore. Master LoRA and QLoRA at the level of "I can explain the rank, alpha, and target-modules hyperparameters in an interview," and know exactly how QLoRA fits a 70B model on 48GB of consumer GPU memory.

---

## 1. Executive Summary & Core Concepts

**Full-parameter fine-tuning** on modern LLMs is a bad economic choice for almost every practical use case. For an 8B model:

- Full SFT needs ~120 GB VRAM (weights + gradients + optimizer state + activations at fp16 with Adam)
- Requires an H100 or 4× A100
- Costs $50-200+ per training run
- Produces a full-size checkpoint (~16 GB) per experiment
- Actively risks catastrophic forgetting

**LoRA (Low-Rank Adaptation)** solves this by *freezing* the base model and inserting small trainable adapter matrices next to specific weight matrices. Instead of updating a 4096×4096 matrix (16.7M params), you train two matrices $A$ (4096×8) and $B$ (8×4096) — 65k params total. The update to the weight matrix is $\Delta W = BA$, always low-rank (rank 8 in this example).

Result:
- Trainable params: ~0.1-1% of the base model
- VRAM: ~30-50% of full SFT (still holds base weights, but no gradients/optimizer state on them)
- Fits an 8B fine-tune on a single 24GB GPU
- Adapters are ~50-200 MB — dozens fit on a single deployed base model

**QLoRA** goes further: the frozen base model is **quantized to 4-bit** (from fp16). Base weights take ~1/4 the memory. Adapters remain in higher precision (bf16). Training math still works because gradients only flow through the fp16/bf16 adapters.

QLoRA result:
- Llama 3.1 8B fine-tunable on a single 16 GB GPU (RTX 3080/4060 Ti 16GB)
- Llama 3.1 70B fine-tunable on a single 48-80 GB GPU (A100 40GB, RTX 6000 Ada, or two 24 GB GPUs)
- Quality within 1-2% of full-precision LoRA on most tasks

**The takeaway:** since 2023, "fine-tuning" almost always means "QLoRA on the largest base model that fits." Full-parameter fine-tuning is reserved for foundation-model developers (Meta, Mistral, Qwen) and edge cases requiring deep behavior changes.

### When Full-Parameter Still Wins

- Very small models (<1B) where LoRA's rank constraint bites
- Domain adaptation where the base model's *knowledge* itself needs modification (rare, expensive, usually still a bad idea)
- Research on new architectures

For everything else covered in this track: **use QLoRA.**

---

## 2. Deep-Dive Breakdown

### 2.1 The Math of LoRA

A linear layer in a transformer computes:

$$
y = Wx + b, \quad W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
$$

Full fine-tuning updates $W$ directly: $W' = W + \Delta W$, where $\Delta W$ has the same shape and requires full gradients.

LoRA factorizes $\Delta W$ as a product of two low-rank matrices:

$$
\Delta W = BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}
$$

where $r \ll \min(d_{\text{in}}, d_{\text{out}})$ — typically $r \in \{4, 8, 16, 32, 64\}$.

The forward pass with LoRA becomes:

$$
y = Wx + \frac{\alpha}{r} BAx + b
$$

Two things to note:
1. **$W$ is frozen** — gradients don't flow to it. All learning happens in $A$ and $B$.
2. **Scaling factor $\alpha/r$**: `alpha` is a hyperparameter (often $\alpha = 2r$ or $\alpha = r$) that controls how much the LoRA update contributes at forward pass. Bigger $\alpha$ = larger effective update. Doesn't affect the number of trainable parameters.

Initialization:
- $A$ initialized with a small Gaussian (Kaiming uniform typically)
- $B$ initialized to **zero**

Why? So that at step 0, $BA = 0$ and the model output is identical to the base model. Training then gradually introduces the adapter's effect. This is the "no perturbation at init" property — crucial for stable training.

### 2.2 Which Modules to Target

Modern LoRA doesn't attach to *every* linear layer. Attention matrices ($Q$, $K$, $V$, $O$ projections) are the most common targets; feed-forward layers (gate, up, down projections) are increasingly included in 2025+ practice.

The `target_modules` argument in PEFT / LoRA config controls this:

```python
from peft import LoraConfig

# Minimal — attention only
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # classic; Q and V only
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Broader — attention + MLP (recommended for 2024+)
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# "All linear" — the current default in many frameworks
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",           # PEFT >= 0.11 supports this
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Rule of thumb:** target more modules → better quality, more parameters. On modern models the "all-linear" default is a strong baseline. Reduce to `q_proj, v_proj` only if VRAM is very tight.

### 2.3 Choosing Rank and Alpha

**Rank $r$** — the "capacity" of the adapter.

| Rank | Params (7B model, all-linear) | Typical use |
|------|-------------------------------|-------------|
| 4    | ~5M   | Very light — style tweak |
| 8    | ~10M  | Baseline for most tasks |
| 16   | ~20M  | Better for larger datasets / more complex tasks |
| 32   | ~40M  | Diminishing returns beyond this for most tasks |
| 64   | ~80M  | Rarely necessary; approaches full SFT |

Higher $r$ = more capacity to learn, but also more overfitting risk and more compute. **Start with $r = 16$**. Move up if val loss plateaus early; down if you overfit.

**Alpha $\alpha$** — the effective learning rate multiplier for the adapter.

Common conventions:
- $\alpha = 2r$ (Hugging Face TRL default) — moderate scaling
- $\alpha = r$ — 1:1 (some setups)
- $\alpha = 16$ regardless of $r$ (some legacy code, less common now)

The scaling $\alpha/r$ appears in the forward pass. If you double $r$ and keep $\alpha$ fixed, the effective scale drops — you may need to adjust LR. If you use the $\alpha = 2r$ convention, the scale is always 2 and things behave more predictably.

### 2.4 QLoRA — The Key Innovations

QLoRA (Dettmers et al., 2023) is not just "LoRA on a quantized base." It introduced three specific techniques:

**1. 4-bit NormalFloat (NF4) quantization.**
Standard 4-bit quantization uses linear or logarithmic quantization levels. NF4 uses quantization levels drawn from the theoretical distribution of neural network weights (approximately Gaussian). This preserves quality better than naive int4.

**2. Double quantization.**
The quantization constants themselves (one per weight block) are quantized. Squeezes an additional ~0.5 bits per parameter.

**3. Paged optimizer state.**
Uses NVIDIA unified memory to page optimizer state (Adam moments) between GPU and CPU RAM. This avoids OOM on long training runs.

Result: A 70B model weighs ~35 GB in NF4 (vs 140 GB in fp16); can fine-tune on a single 48 GB GPU with adapters in bf16.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### 2.5 Merging Adapters for Deployment

After training, you have two artifacts:
1. The frozen base model
2. The LoRA adapter (small, ~50-200 MB)

For serving, you have two options:

**A) Load base + adapter separately (recommended for A/B/canary):**

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base, "./my-adapter/")
```

The adapter is applied dynamically at forward pass. Small overhead (~2-5%). Lets you swap adapters, keep multiple adapters per base, A/B test cheaply.

**B) Merge adapter into base (recommended for production single-purpose):**

```python
merged = model.merge_and_unload()  # merges BA into W, drops adapter
merged.save_pretrained("./merged-model/")
```

Now the merged model is a plain fp16 model, indistinguishable from the base architecturally. No adapter overhead. Ideal for vLLM serving (Module 3 in self-hosting).

Trade-off:
- Unmerged: flexibility, cheap experimentation, can serve N adapters from 1 base
- Merged: max throughput, standard model artifact, simpler ops

`[IMG-FT02-01]` — *Prompt: A diagram showing LoRA's low-rank decomposition. Left: a large weight matrix W (say 4096x4096) shown as a big blue square, labeled "frozen". Middle: two small matrices A (4096x8) and B (8x4096) shown as thin rectangles, labeled "trainable" in orange. Arrow showing "delta_W = B * A", with the resulting matrix shown as a subtle overlay on W. Below, indicate parameter counts: W has 16.7M params (frozen), A+B have 65k params (trainable) — a 250x reduction. Right side: a mini bar chart comparing VRAM required for full SFT (large red bar) vs LoRA (medium orange bar) vs QLoRA (small green bar) — with labels like "Llama 3.1 8B: 120 GB / 40 GB / 12 GB". Clean textbook illustration style.*

---

## 3. Mental Models & Analogies

### Model 1: Sticky Notes on a Contract

Imagine a 500-page legal contract (the base model). You want to modify how it behaves in a specific situation. You have two options:

**Full fine-tuning** = rewrite the whole contract from scratch. Expensive, error-prone, and someone might accidentally change other clauses they didn't mean to.

**LoRA** = attach sticky notes at specific paragraphs: "when this paragraph applies, do X instead." The original text is preserved. The notes are cheap to write, small to store, easy to remove or modify. If you want a different behavior for another situation, you use a different sticky-note pack — same contract, different notes.

**QLoRA** = the contract is stored on microfilm to save space (4-bit quantization), but the sticky notes are still hand-written on regular paper (bf16). You lose a bit of legibility on the original text, but the sticky-note experience is unchanged.

This is why serving-time LoRA is so flexible: you have one big frozen base model and dozens of adapter packs for different tasks/customers. Load whichever one you need.

### Model 2: The Musician's Practice Room

Every serious musician has: (1) a broad musical vocabulary from years of practice (the base model's pretraining), (2) refined technique from focused conservatory training (SFT/instruction-tuning), and (3) role-specific interpretive choices for a given piece (LoRA).

A jazz trio might have three interpretation packs for the same set list — one intimate club vibe, one energetic festival vibe, one recorded-album polish. The musicians themselves haven't retrained; they're the same trio. But each interpretation pack is a small collection of decisions (tempo, dynamics, ornamentation) that reshape the same underlying technique for the moment.

That's LoRA in production: a single expensive-to-produce base model, plus lightweight per-context adapters that reshape it for specific customers, use cases, or moments — without retraining the musicians.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — "LoRA is a lower-quality fine-tune."**
Widely believed, generally wrong. In practice LoRA with sensible hyperparameters (rank 16-32, all-linear targets, standard alpha) matches full SFT quality on almost every downstream task. This has been replicated in the LIMA paper, the QLoRA paper, and countless industry benchmarks. There are edge cases (very small models, very domain-shifted continued pretraining) where full SFT eeks out a small win, but for typical instruct-model fine-tuning, LoRA is not a quality compromise — it's a cost optimization with equivalent output. Interviewers hearing "LoRA is lower quality" will (correctly) file you as a year behind the field.

**Pitfall #2 — Setting rank absurdly high "just to be safe."**
"Rank 128 must be better than rank 16, right?" No. Higher rank means more parameters to overfit on your (usually small) dataset. On a 2000-example SFT set, rank 128 will memorize aggressively and generalize worse than rank 16. Start at rank 8-16 and only increase if evaluation says you're underfitting (val loss still descending at end of training).

Corollary: don't naïvely target every module either. `all-linear` is a sensible default; going wider (embeddings, layer norms) is rarely worth it and can destabilize.

**Pitfall #3 — Forgetting to disable dropout at inference.**
LoRA config often includes `lora_dropout=0.05` for regularization during training. At inference the framework should switch to eval mode (which disables dropout) — but if you're doing anything custom or exporting weights weirdly, verify. Runs where the dropout stays on at inference are ~1-3% worse than they should be and the bug is hard to catch. Every `model.eval()` call needs to happen before serving.

Related: **merging adapters with dropout still enabled** produces subtly wrong merged weights. Merge with `model.eval()` first.

---

## 5. Self-Assessment Bank

**Q1 (MC):** In LoRA, matrix $B$ is initialized to:
A) Random Gaussian
B) All zeros
C) Identity matrix
D) A copy of $W$

**Q2 (short):** Explain why the LoRA scaling factor $\alpha/r$ appears in the forward pass and what happens if you double $r$ while keeping $\alpha$ fixed.

**Q3 (MC):** QLoRA's "NF4" quantization is:
A) A generic 4-bit integer quantization
B) A 4-bit floating-point format that maps to normal-distribution-shaped weight statistics
C) A 4-bit format specific to NVIDIA GPUs
D) A 4-bit encoding of the *outputs* only

**Q4 (short):** You want to fine-tune Llama 3.1 70B on a single 48 GB GPU. What technique makes this possible, and what are the three key mechanisms?

**Q5 (MC):** Which LoRA target-modules configuration is a strong default in 2025+?
A) `["q_proj", "v_proj"]` only
B) `all-linear` (or equivalent — all attention + MLP linear layers)
C) `["embed_tokens", "lm_head"]`
D) `["layer_norm"]`

**Q6 (short):** Your trained LoRA adapter is 200 MB and works well. You want to deploy to a vLLM instance for max throughput. What are the two deployment options and their trade-offs?

**Q7 (MC):** After doubling rank $r$ from 16 to 32 with $\alpha$ fixed at 32, the effective update scale $\alpha/r$:
A) Doubles
B) Halves
C) Stays the same
D) Quadruples

**Q8 (short):** Give a concrete scenario where a team would run *full-parameter* fine-tuning instead of LoRA/QLoRA, and one where they would not.

**Q9 (MC):** Which of the following is most likely to overfit a small SFT dataset?
A) LoRA, rank 8, 1 epoch
B) LoRA, rank 128, 5 epochs
C) LoRA, rank 16, 2 epochs
D) LoRA, rank 32, 1 epoch

**Q10 (short):** How does QLoRA's "paged optimizer state" work and what problem does it solve?

---

### Answer Key

**A1: B.** $B$ is zero-initialized. This ensures $BA = 0$ at step 0, so the model behaves identically to the base at initialization. Training gradually introduces the adapter's contribution. $A$ is typically Kaiming-uniform (or small Gaussian). Both being zero would mean no gradient flow; both being random would mean the model starts perturbed from the base (bad for stable training).

**A2:** The factor $\alpha/r$ scales the LoRA update contribution: $y = Wx + (\alpha/r) BAx$. It exists because $\Delta W = BA$'s magnitude depends on $r$ — a larger rank produces a larger raw magnitude just from more accumulated inner products. Scaling by $\alpha/r$ decouples the effective learning rate from the rank choice. If you double $r$ (say 16 → 32) but keep $\alpha = 32$, then $\alpha/r$ drops from 2 to 1: the adapter contributes half as much per forward pass, so effective updates are halved. To keep behavior similar, double $\alpha$ alongside $r$ (the "$\alpha = 2r$" convention makes this automatic and is why it's popular).

**A3: B.** NF4 (4-bit NormalFloat) is a 4-bit floating-point format whose quantization levels are drawn from a normal distribution — matching the empirical distribution of neural network weights (which is roughly Gaussian). This is why it preserves quality better than uniform int4: quantization error is smaller for the most common weight values. Not GPU-specific (works anywhere bitsandbytes runs); quantizes weights only, not outputs.

**A4:** **QLoRA** makes this possible. Three mechanisms: (1) **4-bit NF4 quantization** of the frozen base model, reducing 70B model from ~140 GB (fp16) to ~35 GB; (2) **double quantization** of the block-wise quantization constants, saving an additional ~0.5 bits/param; (3) **paged optimizer state** using NVIDIA unified memory so optimizer moments (Adam m and v) can spill to CPU RAM when GPU is tight, preventing OOM. The trainable LoRA adapters remain in bf16 (~0.5 GB), plus activation memory, plus optimizer state — total fits under 48 GB.

**A5: B.** `all-linear` (attention Q/K/V/O + MLP gate/up/down) is the current strong default. It gives the adapter access to more of the model's capacity than attention-only, without the additional cost of embedding/norm targeting. Attention-only (`q_proj, v_proj`) is the historical default; still fine for extreme VRAM constraints but leaves quality on the table. Embedding-only and norm-only don't work in practice for behavioral fine-tuning.

**A6:** Two deployment options:
1. **Load base + adapter separately at serve time.** Pros: swap adapters without redeploying the base; run N adapters against 1 base (multi-tenant), A/B test cheap. Cons: ~2-5% forward-pass overhead; not all serving frameworks support this well (vLLM does via `--enable-lora`).
2. **Merge adapter into base ("merge and unload"), then save the merged model.** Pros: standard model artifact, max throughput, works with any inference server. Cons: no more flexibility — the fine-tune is now baked in; you need a full re-save for each variant; storage per variant is full-model-size.

Typical production choice: merge for single-purpose serving; unmerged for platforms serving many customer-specific fine-tunes.

**A7: B.** $\alpha/r = 32/32 = 1$, down from $32/16 = 2$. So the effective update scale halves. Practical implication: if you double rank and don't want the effective scale to change, also double alpha (or use a framework that follows the `alpha = 2r` convention automatically).

**A8:** **Full-parameter examples:** (i) You're the base-model developer (Meta, Mistral) training a new checkpoint; (ii) domain-adapting a very small model (say 500M params) where LoRA's rank constraint bites and you have compute to spare; (iii) research on architecture changes that require modifying all weights.

**Won't do full-parameter:** (i) Behavioral SFT on any 7B+ model in an applied engineering team — LoRA equivalent quality at 1% of the cost; (ii) fine-tuning with < 10k examples — full SFT overfits fast and offers no benefit; (iii) any budget-constrained context (startups, consulting, indie devs) — QLoRA halves the hardware bill.

The overwhelming majority of applied fine-tuning is LoRA/QLoRA. Full SFT in an applied team in 2026 is a red flag: usually indicates either bad hyperparameter search or resistance to modern tooling.

**A9: B.** Rank 128 + 5 epochs is the most overfit-prone: high capacity + long training = memorization. Rank 8 with 1 epoch (A) is the safest; the others are moderate. Overfitting on LLM SFT typically manifests as: training loss near zero, val loss climbing, model regurgitating training examples verbatim on similar prompts, general capability tanking.

**A10:** Adam optimizer maintains two moment estimates ($m$ and $v$) per parameter — for a 70B fine-tune, even with LoRA that's still significant (though much less than full). On tight GPUs, this optimizer state alone can OOM the run. **Paged optimizer state** uses NVIDIA's unified memory (UM) to transparently page these tensors between GPU VRAM and CPU RAM: when they're not actively being updated, they can live in host memory. Similar to OS-level virtual memory paging. Adds minor throughput overhead but prevents the OOMs that would otherwise kill long training runs on borderline hardware. Enabled in QLoRA via `optim="paged_adamw_8bit"` or `paged_adamw_32bit` in the trainer args.

---

**Related modules:**
- `learning/01-sft-fundamentals.md` — the base SFT concepts LoRA builds on
- `learning/05-fine-tuning-ops.md` — actually running a QLoRA training
- `builds/01-fine-tune-legal-classifier.md` — hands-on QLoRA
- `../self-hosting/learning/01-vllm-serving.md` — serving LoRA adapters via vLLM

**Practice prompts:**
1. Compute the trainable-parameter count for Llama 3.1 8B with LoRA rank 16, all-linear targets. Show the arithmetic.
2. Load a small model (`microsoft/Phi-3-mini`) and run one training step with a LoRA config. Print the diff between `param.grad.norm()` for a base weight vs a LoRA weight.
3. Compare rank 8 vs rank 32 on a small SFT dataset (e.g., a subset of Alpaca). Which converges faster? Which reaches lower val loss? Explain what you see.

**References:**
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — the foundational paper
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- Hu et al. update (2024) — "LoRA Learns Less and Forgets Less" — surprising result on LoRA's regularization effect
- Hugging Face PEFT library — https://github.com/huggingface/peft
- bitsandbytes — https://github.com/TimDettmers/bitsandbytes
