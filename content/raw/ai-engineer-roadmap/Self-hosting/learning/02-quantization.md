# 02 — Quantization

> **Module goal:** Understand why the same 8B model takes 16 GB, 8 GB, or 4 GB depending on your choices — and what you lose (or don't lose) at each step down. Master the four dominant quantization schemes (bnb-int8, GPTQ, AWQ, GGUF Q4_K_M) and know which to pick for which deployment.

---

## 1. Executive Summary & Core Concepts

**Quantization** = representing model weights (and sometimes activations) in fewer bits than their trained precision (fp16/bf16). Instead of each weight being 16 bits (~2 bytes), quantized weights take 8, 4, 3, or even 2 bits.

Why quantize:
1. **Fit larger models on smaller hardware.** A 70B model in bf16 is ~140 GB; in 4-bit AWQ it's ~35 GB — fits on a single 48 GB card.
2. **Faster inference.** Fewer bytes to move through memory → higher throughput on memory-bandwidth-bound workloads (which most LLM inference is).
3. **Lower serving cost.** Smaller VRAM → cheaper GPU tier.

Quality cost: modest, if done right. A well-quantized 4-bit model retains 95-99% of the bf16 baseline's quality on most benchmarks. Below 4-bit, quality drops noticeably.

**The taxonomy that matters in 2026:**

| Format | Bits | Tool | Best for |
|--------|------|------|----------|
| **bf16 / fp16** | 16 | native | Reference; training; when VRAM allows |
| **bnb-int8** | 8 | bitsandbytes | Quick 2× VRAM reduction; slow inference |
| **GPTQ** | 4 (or 3, 2) | AutoGPTQ / vLLM | Server inference (good throughput) |
| **AWQ** | 4 | vLLM / AutoAWQ | Server inference (~fastest 4-bit) |
| **GGUF Q4_K_M** | ~4 (mixed) | llama.cpp / Ollama | Desktop / edge; CPU + GPU offload |
| **NF4** | 4 | bitsandbytes | Training (QLoRA — Module 02 of fine-tuning); not primarily a serving format |

**The rule of thumb:**
- On production GPU serving with vLLM: **AWQ** (fastest) or **GPTQ** (widely available)
- On consumer hardware / Mac / CPU serving: **GGUF Q4_K_M**
- For quick experiments where speed doesn't matter: **bnb-int8**
- For training-time quantization: **NF4 via QLoRA**

**Below 4-bit** (Q3, Q2, INT2) exists but quality drops meaningfully; consider only for extreme resource constraints and always benchmark.

---

## 2. Deep-Dive Breakdown

### 2.1 The Basics — What "Quantization" Actually Is

A weight tensor is a bunch of floating-point numbers. Quantization maps them to a discrete grid of integers, storing:
- The quantized integer per weight (say, 4 bits)
- A **scale** factor (fp16/fp32) per group of weights (say, per row or per 128-weight block) so we can approximately recover the original: $w \approx s \cdot q$

At load / inference time:
- Dequantize back to bf16 in tiles (**dequant-on-the-fly**), then run the matmul
- Or use a specialized kernel that does INT × BF16 → BF16 without full dequant (faster)

Symmetric vs asymmetric:
- **Symmetric**: range is $[-a, +a]$; quantization is `w / s → q`. Simple, fast.
- **Asymmetric**: range is $[a_{\min}, a_{\max}]$; extra `zero_point` parameter per group. Slightly better fit for skewed distributions; slower.

Per-tensor vs per-row vs per-block-of-N:
- **Per-tensor**: one scale for the whole weight matrix. Fastest, worst fit.
- **Per-row/column**: one scale per output channel. Better.
- **Per-block (group_size)**: one scale per group of ~64-128 weights. Best fit, more overhead.

Modern schemes (GPTQ, AWQ) use per-block with group sizes of 64 or 128.

### 2.2 GPTQ

**GPTQ (Frantar et al., 2022)** was the first widely-adopted 4-bit LLM quantization scheme. Key insight: don't just round each weight independently; use **calibration data** (a small dataset of representative inputs) and adjust remaining weights to compensate for each rounding error, minimizing overall reconstruction loss.

Algorithm sketch:
1. Take a small calibration set (~128-1000 samples of typical prompts)
2. For each weight matrix, process columns left-to-right:
   - Quantize the current column
   - Compute the resulting error
   - Adjust unquantized columns to compensate (Hessian-based update)
3. Store quantized weights + scales

Result: 4-bit models very close to bf16 quality (usually within 1% on MMLU / HellaSwag).

Practical use:
```bash
# Load a pre-quantized GPTQ model in vLLM
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-Chat-GPTQ \
    --quantization gptq
```

Pros: mature, wide model coverage on Hugging Face (TheBloke, hugging-quants, LoneStriker, and modern QuantFactory repos), well-supported.

Cons: needs calibration data to produce good quality; the calibration step is offline (produces the quantized checkpoint, not a runtime option).

### 2.3 AWQ (Activation-aware Weight Quantization)

**AWQ (Lin et al., 2023)** observes that not all weights matter equally — some are much more important because they correspond to *large activations*. Quantizing critical weights more carefully protects quality.

Algorithm sketch:
1. Take calibration data; run inference; measure activation magnitudes
2. Identify the top ~1% "salient" weight channels (those with big activations)
3. Scale down those channels' inputs (equivalent to protecting those weights)
4. Then quantize all weights to 4-bit uniformly

Result: comparable quality to GPTQ, often *slightly faster* inference kernels, and simpler to implement per-model.

Practical use:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-3.1-70B-Instruct-AWQ \
    --quantization awq
```

Or `--quantization awq_marlin` on newer vLLM for the fastest AWQ kernel path (requires Ampere+).

**GPTQ vs AWQ:** in benchmarks they're comparable in quality. AWQ tends to have slightly faster inference kernels in vLLM. Use whichever pre-quantized checkpoint exists for your model; if both, prefer AWQ for serving throughput.

### 2.4 GGUF and llama.cpp

**GGUF** (GGML Universal Format) is the model format used by **llama.cpp**, the C++ inference engine optimized for CPU and consumer GPU inference. Ollama (the popular desktop LLM tool) wraps llama.cpp.

GGUF supports many quantization levels:

| GGUF quant | Effective bits | Quality | Speed | Size (7B model) |
|-----------|---------------|---------|-------|-----------------|
| F16 / BF16 | 16 | Reference | Slowest | ~13 GB |
| Q8_0 | 8 | Excellent | Fast | ~7 GB |
| Q6_K | ~6.5 | Very good | Fast | ~5.5 GB |
| Q5_K_M | ~5.5 | Very good | Fast | ~5 GB |
| **Q4_K_M** | ~4.5 | Good (sweet spot) | Very fast | ~4.4 GB |
| Q4_K_S | ~4.3 | Slightly worse than Q4_K_M | Faster | ~4 GB |
| Q3_K_M | ~3.3 | Noticeable drop | Fastest | ~3.5 GB |
| Q2_K | ~2.6 | Quality drops significantly | Fastest | ~2.7 GB |

**Q4_K_M is the widely-agreed-on sweet spot** for consumer / edge deployment. Ollama's default is often Q4_K_M or Q5_K_M for popular models.

Convert Hugging Face → GGUF:
```bash
# From llama.cpp repo
python convert_hf_to_gguf.py ./merged/my-model --outtype bf16
./llama-quantize ./merged/my-model/ggml-model-bf16.gguf ./my-model-Q4_K_M.gguf Q4_K_M
```

Serve via Ollama:
```bash
# Create a Modelfile pointing at the GGUF
echo 'FROM ./my-model-Q4_K_M.gguf' > Modelfile
ollama create my-model -f Modelfile
ollama run my-model
```

Or via llama.cpp's server directly:
```bash
./llama-server -m my-model-Q4_K_M.gguf --host 0.0.0.0 --port 8080
```

GGUF's kill features:
- **Memory-mapped loading** — model doesn't fully load into RAM; pages in on demand
- **CPU + GPU offload** — offload N layers to GPU, rest run on CPU. Enables serving models larger than your VRAM.
- **Mac Metal support** — best-in-class performance on Apple Silicon (M1/M2/M3/M4)
- **Rock-solid single-user desktop UX** — Ollama is what most developers "just try" LLMs on

Downside: **not competitive with vLLM for high-throughput multi-user server workloads.** Use GGUF for local/edge; use AWQ/GPTQ in vLLM for server.

### 2.5 bnb-int8 (bitsandbytes)

The oldest and simplest option: `load_in_8bit=True` via bitsandbytes. Converts weights to int8 at load time (no calibration required).

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

Pros: zero prep work, one config flag.

Cons: **slow inference**. bitsandbytes' int8 kernels are unoptimized compared to GPTQ/AWQ/GGUF specialized kernels. Fine for VRAM reduction in experiments; **do not use for production serving.**

### 2.6 Choosing Quantization: A Decision Tree

```
Are you serving on a server GPU with vLLM/TGI?
├── Yes → AWQ (preferred) or GPTQ (if only that exists). NEVER bnb-int8 for prod serving.
│         What VRAM do you have?
│         ├── >= 2× model_size_bf16 → serve bf16 (no quantization needed)
│         ├── ~= 1× model_size_bf16 → 8-bit if you must; but consider 4-bit for headroom
│         └── << 1× model_size_bf16 → 4-bit (AWQ / GPTQ); 3-bit only if forced
│
├── Local / desktop / Mac?
│   └── GGUF Q4_K_M (Ollama or llama.cpp)
│
├── Training (fine-tuning)?
│   └── NF4 via QLoRA (see fine-tuning/learning/02-peft-lora.md)
│
└── Just experimenting for size reduction, don't care about latency?
    └── bnb-int8 for quick iteration
```

### 2.7 Measuring Quality Loss

You *must* eval every quantized model against its bf16 baseline before shipping. Standard checks:

1. **Perplexity on a text corpus** (WikiText-2 is the traditional benchmark). Perplexity ratio quantized/base should be within 1-3%.
2. **MMLU / HellaSwag / GSM8k sample**. Aggregate score drop < 2 percentage points is typically acceptable.
3. **Your task-specific golden set** (Month 6 Module 08). This is the one that matters — some tasks are more sensitive to quantization than others (math and structured output particularly so).
4. **Long-context behavior**. Some quantization schemes degrade at long context; test if your workload has long prompts.

Rule: **never ship a quantized model without measuring against the bf16 reference on your task.** "AWQ is usually within 1%" is not a shipping justification.

`[IMG-SH02-01]` — *Prompt: A four-panel comparison chart of quantization formats. Panel 1: bar chart of VRAM requirement for Llama 3.1 70B at fp16 (140 GB, red), int8 (70 GB, orange), int4 AWQ (35 GB, green), Q4_K_M GGUF (~38 GB, green-yellow). Panel 2: bar chart of MMLU score preservation (all clustered around baseline ~85%, with tiny drops shown at ~-0.5% to ~-1.5%). Panel 3: bar chart of throughput (tokens/sec) at concurrency 32, with int4 AWQ topping, bf16 next, int8 significantly behind. Panel 4: decision tree flowchart from "Are you serving on GPU?" leading to AWQ/GPTQ vs GGUF vs bnb-int8. Clean data-vis style, muted palette, clearly labeled.*

---

## 3. Mental Models & Analogies

### Model 1: JPEG for Model Weights

Everyone knows JPEG compression: it drops the highest-frequency detail in an image, saving 90%+ of the bytes with imperceptible quality loss on most images (and terrible artifacts on some — text, sharp edges, gradients).

Quantization is JPEG for LLM weights. Same trade: much smaller, mostly imperceptible loss, catastrophic on some inputs. And like JPEG, the choice of quality setting matters: Q4_K_M is the "JPEG quality 90" — great default. Q2_K is "JPEG quality 30" — visibly bad on many images. GPTQ vs AWQ is roughly "JPEG vs WebP" — different encoders with slightly different quality/speed trade-offs but similar core idea.

Corollary that saves lives: never assume compression is free. **Test it on your workload.** JPEG artifacts destroy text-heavy images; quantization artifacts destroy math and structured-output tasks disproportionately.

### Model 2: Selective Note-Taking

Imagine you're a student summarizing a 500-page textbook.

- **bf16 = photocopying every page.** Lossless, big, slow.
- **int8 = writing detailed notes on every chapter.** Half the size, mostly the same information.
- **int4 AWQ = highlighting the important passages and paraphrasing the rest.** Quarter the size; you know which parts you focused on protecting.
- **int2 = one-sentence chapter summaries.** Barely useful for anything specific.

AWQ's "activation-aware" is the highlighter: it decides which weights *matter most* (based on activation magnitudes on real data) and protects them more carefully. GPTQ's approach is more like taking notes column-by-column and correcting mistakes as you go. Both work. Both are compression by prioritization.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — "8-bit will halve my VRAM and cost me nothing."**
Partly true (halves VRAM), partly wrong (costs you a lot in serving throughput if you use bitsandbytes). The bitsandbytes int8 path in inference is much slower than either bf16 native or AWQ int4. If you need 8-bit for VRAM reasons, prefer GGUF Q8_0 (llama.cpp path) or find an AWQ int4 (which is smaller, faster, and comparable quality) instead. **Never deploy `load_in_8bit=True` to production; it's a debugging tool.**

**Pitfall #2 — Skipping the quality evaluation.**
"AWQ is usually within 1%" is a paper-average statement. Your task may be one of the sensitivity outliers. Common cases where int4 quantization visibly hurts:
- Math (GSM8k, MATH benchmarks)
- Structured JSON output at long lengths
- Long-context QA (128k+)
- Multi-step tool calling

Always run: your golden set + a general-capability check + a long-context stress test before shipping. If quality drops > 3 percentage points on a critical metric, back off to 8-bit or bf16.

**Pitfall #3 — Confusing training-time and inference-time quantization.**
QLoRA's NF4 (fine-tuning Module 02) is used during **training** — the base model is quantized while frozen; adapters train in bf16. That's a very different setting from serving a pre-quantized checkpoint at inference time. Common bug: someone hears "QLoRA uses 4-bit" and tries to load a QLoRA-trained model directly for inference in 4-bit NF4 — which mostly works but isn't the same as an AWQ/GPTQ-quantized model designed for serving. For production serving of a fine-tune: merge the adapter into fp16/bf16 base, then quantize that with AWQ or GPTQ for the deployment.

---

## 5. Self-Assessment Bank

**Q1 (MC):** For high-throughput server-side serving of Llama 3.1 70B, which quantization is most appropriate?
A) bnb-int8
B) AWQ (int4)
C) GGUF Q4_K_M
D) fp32

**Q2 (short):** Explain the core idea of AWQ (Activation-aware Weight Quantization) in two sentences.

**Q3 (MC):** GGUF is most appropriate for:
A) High-throughput multi-tenant server serving
B) Consumer / desktop / edge inference, especially on Mac
C) Model training
D) Never — it's deprecated

**Q4 (short):** Your Llama 3.1 70B in bf16 is ~140 GB. You have a single 48 GB GPU. What are your options for fitting this?

**Q5 (MC):** Which of these should you NOT do in production?
A) Load an AWQ int4 model in vLLM
B) Use `load_in_8bit=True` (bitsandbytes) for high-throughput serving
C) Serve GGUF Q4_K_M via llama.cpp for a desktop app
D) Merge a LoRA and quantize to AWQ int4 for serving

**Q6 (short):** You quantize your production model to AWQ int4 and MMLU drops from 68% to 64%. What do you do?

**Q7 (MC):** The "K_M" in GGUF's `Q4_K_M` refers to:
A) A specific compression algorithm
B) A quantization scheme using K-quants at Medium mixed precision
C) A model size (K = thousand, M = million)
D) A benchmark name

**Q8 (short):** Compare GPTQ and AWQ: what's the same, what's different, and which would you pick for a new vLLM deployment today?

**Q9 (MC):** Below what bit-precision does quality typically start dropping meaningfully?
A) 8 bits
B) 6 bits
C) 4 bits (sweet spot)
D) 3 bits or lower

**Q10 (short):** A colleague argues "let's use int2 to save more VRAM." How do you respond?

---

### Answer Key

**A1: B.** AWQ int4 is the production sweet spot for GPU serving: near-bf16 quality, dramatic VRAM savings, fast inference kernels (especially with `awq_marlin` on Ampere+ GPUs). bnb-int8 is slow. GGUF is for consumer/edge. fp32 wastes VRAM without quality gains (models were trained in bf16/fp16 — fp32 doesn't add information).

**A2:** AWQ observes that not all weights are equally important: the ~1% associated with large activations disproportionately affect quality. AWQ uses calibration data to identify these "salient" channels and applies scaling that protects them during otherwise-uniform 4-bit quantization, yielding better quality than naive uniform quantization at the same bit budget.

**A3: B.** GGUF's strengths — memory-mapped loading, CPU + GPU offload, Mac Metal support, mature single-user UX (Ollama) — target consumer / desktop / edge inference. For server high-throughput workloads, vLLM with AWQ or GPTQ is faster and more memory-efficient. GGUF is very much alive and actively developed; it just isn't the right tool for server serving.

**A4:** Options:
1. **Quantize to AWQ int4** — model shrinks to ~35 GB, fits on the 48 GB GPU with KV cache room
2. **Quantize to GPTQ int4** — similar result
3. **GGUF Q4_K_M via llama.cpp** — fits, but throughput will be much lower than AWQ in vLLM
4. **Tensor parallelism across two smaller GPUs** — instead of one 48 GB, use 2× 40 GB (A100) or 2× 24 GB (RTX 4090) with `--tensor-parallel-size 2`
5. **Rent bigger hardware** — one H100 80GB or A100 80GB fits bf16 with headroom
6. **CPU + GPU offload via llama.cpp** — technically fits but throughput plummets; not a serving solution

For a production deployment on that 48 GB card: option 1 (AWQ int4) is the standard choice.

**A5: B.** `load_in_8bit=True` via bitsandbytes has painfully slow inference kernels; it's an experimentation tool, not a production serving option. Use AWQ, GPTQ, or GGUF for production quantization. All the other options are legitimate production deployments.

**A6:** 4-percentage-point drop is substantial. Options:
1. **Try GPTQ instead** — sometimes marginally different quality profile
2. **Try 8-bit** — AWQ int8 or GPTQ int8; less VRAM saving but usually near-lossless
3. **Try a different calibration dataset for the AWQ conversion** — if calibration data was mismatched to your target distribution, results improve when you use in-domain calibration
4. **Investigate WHICH questions are failing.** Often quantization degrades certain question types (math, long-context) more than others. If your production workload avoids those types, the aggregate MMLU drop overstates real-world impact.
5. **If none of the above helps: don't quantize.** Serve bf16 on a bigger GPU. Cost-vs-quality is a business decision, not a quantization requirement.

**A7: B.** In GGUF nomenclature, "K" refers to the "K-quants" quantization family (Kawrakow's methods), and "M" is the mixed-precision variant (some weights kept at higher precision than others). K_S is small, K_M is medium (recommended), K_L is large. The mixed-precision approach spends more bits on important layers.

**A8:** **Same:** both target int4, both use per-block scaling, both rely on calibration data, both are widely available on HuggingFace, both produce near-bf16 quality on most tasks.

**Different:** GPTQ processes columns sequentially with error compensation using a Hessian; AWQ uses activation magnitudes to identify salient channels and scale-protects them. GPTQ came first (2022) and has broader model coverage. AWQ (2023) has slightly faster inference kernels in vLLM (especially `awq_marlin` on Ampere+).

**Pick for a new deployment today:** AWQ if available (best throughput), GPTQ as a fallback if only that pre-quantized checkpoint exists for your model. Very close in quality; the tie-breaker is kernel speed.

**A9: D.** Down to 4 bits quality is preserved well by modern methods (GPTQ, AWQ, K-quants). At 3 bits quality starts dropping visibly; at 2 bits it drops significantly. Some newer schemes (SmoothQuant, EETQ, 1-bit LLMs) push the boundary lower but with quality trade-offs that make them niche.

**A10:** "Two bits saves ~2× the VRAM vs 4-bit but comes with substantial quality loss on most tasks — often 5-15 percentage points on reasoning benchmarks. Before we do it we'd need to (a) benchmark on our task-specific golden set and confirm the drop is acceptable, (b) test long-context and structured-output cases which quantize particularly poorly, (c) rule out that the same VRAM savings could come from serving a smaller *higher-quality* model instead (Llama 3.1 8B at bf16 vs 70B at int2 — often the smaller model in higher precision wins). Two-bit is legitimate for research and extreme edge deployments, but as a default it's rarely the right call. Let's see the workload numbers first."

That's the shape of a good pushback: acknowledge the intent, name the specific risks, propose the check, offer alternatives.

---

**Related modules:**
- `learning/01-vllm-serving.md` — where quantized checkpoints get served
- `learning/03-openai-compatible-api.md` — API layer sits on top
- `builds/01-self-host-llama-api.md` — hands-on including quantization
- `../fine-tuning/learning/02-peft-lora.md` — QLoRA's training-time quantization (NF4)

**Practice prompts:**
1. Download an AWQ-quantized 8B model from HuggingFace (search "Llama-3.1-8B-Instruct-AWQ"). Launch in vLLM. Benchmark tokens/sec vs bf16 baseline.
2. Convert a small fine-tuned model to GGUF Q4_K_M. Serve via Ollama. Note the disk size and load time compared to the fp16 original.
3. Run MMLU (a subset — 200 questions) on the same model at bf16, AWQ int4, and GPTQ int4. Report the deltas.
4. Read the AWQ paper (Lin et al., 2023). Explain in one paragraph what "salient" means and how it's identified.

**References:**
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)
- llama.cpp README and quantization docs — https://github.com/ggerganov/llama.cpp
- Neural Magic's SparseML / deepsparse write-ups
- TheBloke / QuantFactory / hugging-quants — canonical repositories of pre-quantized model checkpoints
