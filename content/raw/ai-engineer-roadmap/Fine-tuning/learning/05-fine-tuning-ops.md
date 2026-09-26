# 05 — Fine-Tuning Ops: Training, Evaluation, Deployment

> **Module goal:** Actually run a fine-tune end-to-end. Choose infra, launch training with the right frameworks, read the loss curve honestly, evaluate against the base, and deploy the adapter or merged model to production. This is the operational glue that connects the previous four modules to a shipping model.

---

## 1. Executive Summary & Core Concepts

Modules 01-04 are the theory. This is the plumbing.

Five operational steps:

1. **Choose infra** — Colab, Modal, Runpod, your own GPU, or a managed provider
2. **Choose framework** — TRL, Axolotl, torchtune, Unsloth
3. **Configure and launch training** — hyperparameters, monitoring, checkpoints
4. **Evaluate** — task metric + general-capability sanity check + cost + latency
5. **Deploy** — adapter vs merged; A/B against base in shadow traffic; monitor in prod

**The framework choice matrix:**

| Framework | Best for | Trade-off |
|-----------|----------|-----------|
| **Hugging Face TRL** | Learning the mechanics; small-scale runs | Verbose; you write more code |
| **Axolotl** | Production runs from a YAML config | Batteries-included; slight abstraction leak |
| **torchtune** | PyTorch-native; larger scale; multi-GPU | Newer, smaller ecosystem |
| **Unsloth** | Fastest single-GPU fine-tuning (~2× TRL); memory-optimal | Restricted to certain model families |
| **Provider APIs** (OpenAI, Anthropic) | Can't/won't run infra | Vendor lock-in; less control |

**Recommendation for this track:** learn on TRL (understanding), then switch to Axolotl or Unsloth for production runs.

**GPU choice:** for 7-8B QLoRA fine-tuning, a single 24 GB card (RTX 3090/4090, A10, L4) is plenty. For 13-14B, 24 GB is still doable with careful config. For 70B QLoRA, 48 GB (RTX 6000 Ada, A100 40GB with tight config) or 2× 24 GB.

---

## 2. Deep-Dive Breakdown

### 2.1 Infra: Cheapest to Most Managed

**Colab Pro / Pro+** ($10-50/mo) — easiest onramp; A100 access is not guaranteed. Good for learning; tolerable for occasional runs.

**Runpod / Vast.ai / Modal** ($0.30-1.00/hr on-demand for consumer-tier GPUs; $1-3/hr for H100/A100) — the sweet spot for individuals and small teams. Persistent volumes; SSH; scripted launches.

**Modal specifically** — Python-native, has a nice programmatic model for launching training jobs. Charges by GPU-second.

```python
# modal_finetune.py — sketch
import modal

app = modal.App("finetune")
image = modal.Image.debian_slim().pip_install("transformers", "peft", "trl", "bitsandbytes")

@app.function(gpu="A100-40GB", timeout=3600 * 4, image=image)
def train():
    # ... training code here ...
    pass

if __name__ == "__main__":
    with app.run():
        train.remote()
```

**Local GPU (RTX 3090/4090)** — best long-term value if you fine-tune regularly. Upfront cost $1200-2000 but no hourly bill. Nightly runs come for free.

**Managed provider APIs** (Anthropic Claude, OpenAI GPT-4 fine-tuning) — costs $5-50 per million training tokens; opinionated defaults; no infra work. Trade-offs: vendor lock-in, restricted model choice, limited hyperparameter control. Good for teams without GPU expertise or for validating that fine-tuning helps at all before investing in open-model infra.

### 2.2 A Concrete QLoRA Training Script (TRL)

This is roughly what a working fine-tune script looks like. Study it — you'll write variations of this dozens of times.

```python
# train_qlora.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- 1. Load quantized base model ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # if available
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 2. LoRA config ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 3. Load dataset (assumes messages format) ---
dataset = load_dataset("json", data_files="data/train.jsonl")["train"]
eval_dataset = load_dataset("json", data_files="data/val.jsonl")["train"]

# --- 4. Training config ---
training_args = SFTConfig(
    output_dir="./runs/llama-3.1-8b-legal-classifier",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,      # effective batch 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.0,
    max_seq_length=2048,
    packing=True,                        # pack short examples
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to=["mlflow", "tensorboard"],
    optim="paged_adamw_8bit",
    seed=42,
)

# --- 5. Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer,
)

trainer.train()

# Save adapter + tokenizer
trainer.model.save_pretrained("./runs/llama-3.1-8b-legal-classifier/final")
tokenizer.save_pretrained("./runs/llama-3.1-8b-legal-classifier/final")
```

Notes on the choices:
- `bnb_4bit_compute_dtype=bfloat16`: compute in bf16 (higher precision than fp16 for training stability); weights stored in 4-bit
- `attn_implementation="flash_attention_2"`: massive speedup on modern GPUs (Ampere+)
- `gradient_accumulation_steps=8, per_device_train_batch_size=4`: effective batch 32; increase gradient accumulation instead of batch size when VRAM-constrained
- `learning_rate=2e-4`: standard for LoRA; would be 1-5e-5 for full SFT
- `packing=True`: throughput win
- `paged_adamw_8bit`: QLoRA's paged optimizer
- `load_best_model_at_end`: use validation loss to select checkpoint, not last-step

### 2.3 Reading the Loss Curve

You'll be staring at loss curves a lot. Interpret them honestly.

**Healthy curve:**
- Train loss descends smoothly from ~2.0 to ~1.0 over the first 100-500 steps
- Eval loss follows within a small gap (say 0.05-0.15 gap by end)
- Both plateau near the end

**Overfitting:**
- Train loss keeps descending
- Eval loss plateaus, then climbs
- Gap between train and eval widens

**Underfitting:**
- Train loss stays high (> 1.5) throughout
- Eval loss high too
- LR probably too low or dataset too small

**Instability (LR too high):**
- Train loss spikes upward or NaN
- Occasional large steps
- Usually irrecoverable — restart with lower LR

**Common mystery patterns:**
- Train loss suddenly drops at end: probably packing changed active batch composition
- Eval > train by a lot: might be format mismatch between train and eval sets

### 2.4 Evaluation: Beyond Loss

Loss is a training signal, not a shipping decision. Evaluate:

**1. Task-specific metric** — accuracy, F1, RAGAS, BLEU/ROUGE for summarization, or LLM-judge score. Whatever the fine-tune was for.

**2. General-capability sanity check** — MMLU sample, HellaSwag sample, GSM8k sample. Fine-tune should not tank these. Look for drops > 3 percentage points as a warning sign.

**3. Latency benchmark** — same prompt, base vs fine-tuned, at your production infra. Merged models should perform identically to base; unmerged adapters ~2-5% slower.

**4. Cost check** — token counts. Fine-tuned models sometimes produce longer responses; that's inference cost. Fine-tuning for verbosity reduction is a legitimate goal.

**5. LLM-judge A/B** — for each of 100+ eval prompts, sample from base and from fine-tune; ask a stronger model to judge pairwise. Win rate > 55% is meaningful; 60%+ is strong.

Simple TRL-style eval harness:

```python
# eval.py
from datasets import load_dataset
import anthropic, json

judge = anthropic.Client()
eval_set = load_dataset("json", data_files="data/eval.jsonl")["train"]

wins_ft, wins_base, ties = 0, 0, 0
for row in eval_set:
    ft_resp = generate_with_finetune(row["prompt"])
    base_resp = generate_with_base(row["prompt"])
    result = judge.messages.create(
        model="claude-opus-5",
        max_tokens=200,
        messages=[{"role": "user", "content": PAIRWISE_JUDGE_PROMPT.format(
            prompt=row["prompt"],
            a=ft_resp,
            b=base_resp,
        )}],
    )
    verdict = parse_verdict(result.content[0].text)
    if verdict == "A": wins_ft += 1
    elif verdict == "B": wins_base += 1
    else: ties += 1

print(f"Fine-tune wins: {wins_ft}/{len(eval_set)} = {wins_ft/len(eval_set):.2%}")
```

Randomize A/B order per row to control for position bias (Month 6 Module 08).

### 2.5 Deployment

Three paths:

**A) Serve the adapter with vLLM's `--enable-lora`** (Self-Hosting Module 01) — swap adapters at request time via `LoRAModuleName` header.

**B) Merge + save + serve as normal model** — merge with `.merge_and_unload()`, save, load in vLLM/TGI as any HuggingFace model.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, "./runs/legal-classifier/final")
merged = model.merge_and_unload()
merged.save_pretrained("./merged/legal-classifier")
```

**C) Convert to GGUF for llama.cpp / Ollama** — for local/desktop deployment, convert the merged model:

```bash
# From llama.cpp repo
python convert_hf_to_gguf.py ./merged/legal-classifier --outtype bf16
# Then quantize for size
./quantize ./merged/legal-classifier/ggml-model-bf16.gguf ./legal-classifier-Q4_K_M.gguf Q4_K_M
```

Then `ollama create legal-classifier -f Modelfile` and it's servable via `ollama run legal-classifier`.

### 2.6 Monitoring the Fine-Tuned Model in Production

Same monitoring as any LLM (Month 6 Module 09), plus specific eyes on:

- **Refusal rate** — did fine-tuning shift the refusal shape unintentionally?
- **Response length distribution** — did the model become terser or more verbose than base?
- **Task-specific metric** — regularly sample and grade
- **General-capability samples** — a small "canary" set of general questions to verify the model hasn't degraded

Deploy strategy: **shadow → canary → full**.

1. Shadow: fine-tune serves alongside base, receives copies of production traffic, outputs logged and evaluated but not returned to users
2. Canary: fine-tune serves 1% → 5% → 25% of traffic; automated rollback on metric drop
3. Full: fine-tune replaces base entirely; keep the base loaded on the side for emergency rollback

`[IMG-FT05-01]` — *Prompt: An operational flow diagram of a fine-tune training and deployment pipeline. Left: dataset icon labeled with size and format. Middle: training loop shown as a curved arrow going into a "GPU (H100 / A100 / L4)" server icon with configuration parameters (rank=16, alpha=32, LR=2e-4, epochs=2, packing=True) listed beside it. Below the GPU: a monitored loss curve and eval-metric curve, with checkpoints marked as dots. Right: a decision diamond "quality gate passed?" — if yes, model artifact flows down to "adapter" and "merged" branches; if no, arrow loops back to dataset for iteration. Bottom: deployment showing shadow → canary → full rollout with % traffic labels. Clean modern MLOps pipeline diagram.*

---

## 3. Mental Models & Analogies

### Model 1: The Manufacturing Line

Think of the training run as one pass through a manufacturing line. Raw material (dataset) goes in one end, product (checkpoint) comes out the other. The line has stations (each training step); quality-control stations (eval); reject bins (failed checkpoints); packaging (adapter save); and shipping (deploy).

You want to be the person who runs a *reliable* manufacturing line, not one that makes brilliant units by accident and defective ones the rest of the time. That means: repeatable configs, versioned artifacts, automated QC, and roll-back-ready packaging. Every "let me just try something" run should be a scripted, versioned experiment — not something you'll fail to reproduce next week.

### Model 2: Bake, Taste, Ship

Fine-tuning ops resembles a bakery:

- **Bake** = the training run. It takes time; you can't rush it. You can watch it (loss curves) but you can't intervene once started.
- **Taste** = evaluation. You don't ship without tasting. And you don't just taste one thing — you taste the specific dish, plus a "reference bread" to make sure your oven didn't miscalibrate everything else.
- **Ship** = deployment. Small batches first (shadow, canary). Watch how it lands. Scale if it's welcomed.

The failure mode of amateur fine-tuning is treating this as "bake and ship." The failure mode of professional fine-tuning is treating "taste" as a checkbox instead of a discipline.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Not versioning your data + config + code.**
You run a training. It works well. Two weeks later you want to run the same fine-tune on updated data. You can't remember which script version you used, whether you had packing on, or which subset of the dataset was in play. Result: irreproducible run; second attempt is worse; you can't diagnose why.

Fix: **every training run is a versioned artifact.** MLflow / W&B run with:
- Data hash (of the JSONL file)
- Code commit SHA
- Full config as parameters
- Loss + eval metric curves
- Final model artifact URL

If your training script isn't logging these, add them before the next run.

**Pitfall #2 — Loading the fine-tuned model with a different tokenizer than you trained on.**
Common bug: you save the adapter but not the tokenizer alongside it, or you assume "same base model → same tokenizer." Sometimes true, but if you added tokens (rare) or the tokenizer's `chat_template` was updated in the Hub, you get subtle inference mismatches.

Fix: **save the tokenizer alongside the adapter** every time (`tokenizer.save_pretrained(output_dir)`). Load them together. Never mix tokenizer versions across train and inference.

**Pitfall #3 — Trusting the loss curve as your shipping signal.**
Training loss going down is necessary but not sufficient. A model can have low training loss and:
- Memorized the training set (val loss climbing = overfit)
- Learned an off-policy behavior your evals don't catch
- Broken its general capabilities
- Started producing outputs of the wrong length distribution

**The loss says the model learned something; the evals say it learned the right thing.** Never skip the eval battery. Never ship a model that hasn't been compared to the base on task + general + latency + cost.

---

## 5. Self-Assessment Bank

**Q1 (MC):** For fine-tuning Llama 3.1 8B with QLoRA, a reasonable single-GPU choice is:
A) RTX 3060 12GB
B) RTX 4090 24GB / A10G 24GB
C) 4× H100
D) CPU only

**Q2 (short):** Explain why `paged_adamw_8bit` is used with QLoRA training.

**Q3 (MC):** In `SFTConfig`, setting `per_device_train_batch_size=4, gradient_accumulation_steps=8` gives an effective batch size of:
A) 4
B) 8
C) 12
D) 32

**Q4 (short):** Your training loss drops smoothly for 200 steps, then spikes to NaN. What is the most likely cause and how do you fix it?

**Q5 (MC):** Which of the following should you evaluate before shipping a fine-tuned model?
A) Task-specific metric only
B) Task metric + general-capability sanity check + latency + cost
C) Only loss curves
D) User feedback in production

**Q6 (short):** Compare the trade-offs of shipping (a) unmerged adapter + base loaded separately vs (b) merged model as the deployment artifact.

**Q7 (MC):** A merged Llama-3.1-8B + LoRA adapter file weighs approximately:
A) 200 MB (adapter only)
B) 16 GB (full fp16 model)
C) 4 GB (4-bit quantized)
D) 100 GB

**Q8 (short):** Your fine-tuned model matches base on your task metric but adds 200ms to p50 latency. What's the likely cause?

**Q9 (MC):** For rolling out a fine-tune to production, the safest sequence is:
A) Replace base with fine-tune immediately
B) Shadow → canary at 1-25% → full replacement
C) A/B split 50/50 from day one
D) Only serve fine-tune internally forever

**Q10 (short):** You realize your fine-tune degraded general capability (MMLU dropped 6 points). Name three concrete steps you would take before your next training run.

---

### Answer Key

**A1: B.** RTX 4090 (24GB) or A10G (24GB) is the sweet spot for 8B QLoRA fine-tuning. Fits the 4-bit base model (~5 GB), the bf16 adapters (~0.5 GB), activations, and optimizer state comfortably. RTX 3060 12GB is possible for very tight configs but batch sizes are cramped. 4×H100 is massive overkill for a single 8B QLoRA run. CPU-only training on 8B is theoretically possible but takes days to weeks per epoch.

**A2:** `paged_adamw_8bit` combines two optimizations: (1) **8-bit optimizer states** — Adam's first and second moment estimates ($m$ and $v$) are stored in 8-bit precision instead of fp32, ~4× reduction; (2) **paged memory** — uses NVIDIA unified memory to spill state to CPU RAM if GPU runs tight, preventing OOM during transient memory spikes. Together they let QLoRA runs survive on borderline hardware where standard Adam would OOM. Trade-off: slight throughput overhead (~5-10%) and tiny loss of numeric precision (rarely material in practice).

**A3: D — 32.** Effective batch size = `per_device_batch × gradient_accumulation_steps × num_gpus`. Here: 4 × 8 × 1 (single GPU) = 32. Gradient accumulation is how you achieve big effective batches on small GPUs: run 8 mini-batches of 4, accumulate their gradients, then step once as if it were a batch of 32.

**A4:** **Learning rate too high** for the current gradient magnitudes. Training was stable initially but at some point a batch with unusual gradients (e.g., a rare example, or activations getting close to fp16/bf16 limits) triggered a divergence — parameters updated so aggressively that subsequent activations produced NaN and everything collapsed. Fix: **restart with a lower learning rate** (halve it — e.g., 2e-4 → 1e-4), and/or increase warmup ratio (0.05 → 0.1). Adding gradient clipping (`max_grad_norm=1.0` — often already default) provides some safety. If the crash is reproducible after LR reduction, inspect the batch of examples right before the NaN — sometimes a single malformed row is the culprit.

**A5: B.** Multi-dimensional eval battery. Task metric alone can hide catastrophic forgetting; latency can degrade if serving path changed; cost can spike if verbosity increased; general capability drop is the classic silent bug. Loss curves are training signals, not shipping signals. Production user feedback is valuable but comes after ship — you evaluate *before* to protect users.

**A6:**
- **Unmerged (base + adapter separate):** small adapter file (200 MB), can swap or A/B-test multiple adapters against the same loaded base; multi-tenant model-per-customer. Downsides: ~2-5% inference overhead; not all serving stacks support LoRA loading (vLLM supports via `--enable-lora`; TGI similar); slightly more operational complexity.
- **Merged (full fp16/bf16 model):** standard model artifact (~16GB for 8B); works with any serving stack; max throughput. Downsides: no more swap flexibility — each variant is a full model artifact; storage per variant is full-size; harder to A/B without side-by-side deploys.

Rule: **merge for single-purpose production; keep unmerged for multi-variant serving or experimentation.**

**A7: B.** Merging a LoRA adapter into an 8B base model reconstructs a full fp16 (or bf16) model weighing ~16 GB. The adapter's low-rank update `BA` is absorbed into the base weight matrix; the result is a standard model with the same architecture and parameter count as the original base. Storing at bf16 vs fp16 is same size (both 2 bytes/param); storing at 4-bit int (Q4_K_M in GGUF world) drops to ~4-5 GB but that's a separate quantization step, not the merged file itself.

**A8:** Two most common causes:
1. **You didn't merge the adapter** — running unmerged base + adapter adds a forward-pass overhead (~2-5%, which on an 8B model at typical latency can easily be 100-200ms). Fix: merge and re-benchmark.
2. **Fine-tune produces longer responses** than the base did. Even if quality is identical, response tokens dominate LLM latency. Check the length distribution of fine-tune vs base outputs on your eval set. If the fine-tune is more verbose, fix by adding brevity examples to training data or by adjusting your dataset's response-length target.

**A9: B.** Shadow → canary at increasing traffic percentages → full replacement is the standard safe rollout. Shadow (0% user impact) catches quality regressions or crashes without user harm. Canary (small % real traffic) surfaces production-only issues (real prompt distribution, real user behavior) before broad exposure. Automated rollback on metric regression at each canary step. Full replacement only after canary looks clean. Never replace immediately (A) unless you have a very high-confidence eval battery and appetite for risk. 50/50 A/B (C) is fine for measuring but not the *first* step — you shadow first. Internal-only forever (D) means you can't validate real production behavior.

**A10:** Three concrete steps before the next run:
1. **Mix in general instruction data.** Add 10-20% of your dataset from a diverse public source (Alpaca, Dolly, WildChat mix). Keeps the model's general capability signal alive during training. Watch MMLU/HellaSwag delta drop.
2. **Reduce training aggressiveness.** Lower learning rate (2e-4 → 1e-4), lower epoch count (2 → 1), or lower rank (32 → 16). All three reduce how much the model shifts.
3. **Use LoRA/QLoRA if you weren't** — freezing base weights structurally prevents most catastrophic forgetting. If you were already on LoRA, verify target modules aren't too broad (`all-linear` is fine; adding embeddings can cause forgetting).

Bonus fourth step: add MMLU/HellaSwag/GSM8k mini-evals to your training-time eval callback so you see the degradation *during* training and can early-stop when it drops, not after.

---

**Related modules:**
- `learning/01-sft-fundamentals.md` through `04-dataset-curation.md` — the prerequisites
- `builds/01-fine-tune-legal-classifier.md` — hands-on integration
- `../self-hosting/learning/01-vllm-serving.md` — deploying with LoRA
- `../production-ai/learning/05-mlflow.md` — versioning training artifacts
- `../production-ai/learning/09-llm-observability.md` — monitoring after ship

**Practice prompts:**
1. Take a small public preference dataset and run 200 DPO steps on a small model (Phi-3-mini or similar). Log to MLflow. Compare base vs post-DPO generations on 20 prompts.
2. Merge one of your LoRA adapters. Convert to GGUF. Serve locally with Ollama. Time base vs fine-tuned on 100 prompts.
3. Build a reusable `train_qlora.sh` script that takes: base model ID, dataset path, output dir, epoch count. This will be your team's canonical entry point.

**References:**
- TRL docs — https://huggingface.co/docs/trl
- Axolotl — https://github.com/axolotl-ai-cloud/axolotl
- Unsloth — https://github.com/unslothai/unsloth (fastest single-GPU)
- torchtune — https://github.com/pytorch/torchtune
- Modal ML infra docs
- The Hugging Face fine-tuning cookbook
