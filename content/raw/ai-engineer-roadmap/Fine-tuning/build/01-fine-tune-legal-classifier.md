# Fine-Tuning Build 01 — Legal-Text Classifier on Llama 3.1 8B (QLoRA)

> **Deliverable:** A fine-tuned Llama 3.1 8B model that classifies short legal text passages by area of law (contract, torts, criminal, constitutional, evidence, property, and "other"). Trained via QLoRA on a curated dataset of ~2500 examples. Achieves ≥85% accuracy on a held-out test set, deployed as a FastAPI endpoint, evaluated against the base model with a full A/B report.
>
> **Time:** 12-16 hours end-to-end. Data curation is the long tail.
>
> **What you'll be able to say:** "I QLoRA fine-tuned Llama 3.1 8B for legal document classification. Dataset was 2500 curated examples across 7 classes, deduped semantically. Training ran 2 epochs on a single A10G GPU in about 40 minutes. Fine-tuned model hits 87% test accuracy vs 62% for the base with zero-shot prompting. Deployed as a merged model in vLLM behind a FastAPI service."

---

## 1. Project Overview

### Why this task

- **Well-defined output space** (7 fixed classes) — easy to measure success
- **Base model can already sort of do this** with prompting — so you can measure a *real* fine-tune uplift
- **Requires legal domain vocabulary** — a case where a smaller specialized model can rival a larger general one
- **Naturally connects to Brian's Montana Legislative Branch context** — same underlying skill applies to bill categorization, statute retrieval, or the internal search work at legmt.gov

### Architecture

`[IMG-FT-BUILD-01]` — *Prompt: End-to-end fine-tuning project architecture. Left: dataset construction pipeline showing raw sources (public case law corpus, Wikipedia legal snippets, synthetic prompts distilled from Claude) → filtering pipeline → curated 2500-row JSONL. Middle: training pipeline — QLoRA config on Llama-3.1-8B-Instruct, hyperparameters listed (rank=16, alpha=32, LR=2e-4, epochs=2), loss and accuracy curves. Right: deployment pipeline — merge adapter → save merged model → serve with vLLM behind FastAPI → evaluation A/B against base with 87% vs 62% accuracy bars. Include monitoring loop back with prod thumbs-down samples feeding into next dataset iteration. Clean MLOps pipeline diagram.*

### Repo layout

```
legal-classifier/
├── pyproject.toml
├── README.md
├── data/
│   ├── raw/                # source datasets
│   ├── processed/          # cleaned intermediate
│   ├── train.jsonl         # final training set
│   ├── val.jsonl
│   └── test.jsonl
├── src/
│   ├── build_dataset.py    # data curation pipeline
│   ├── train.py            # QLoRA training entrypoint
│   ├── eval.py             # comparison harness
│   ├── merge.py            # adapter → merged model
│   └── serve.py            # FastAPI endpoint
├── configs/
│   └── qlora.yaml          # training config
├── runs/
│   └── llama-3.1-8b-legal-v1/   # MLflow-tracked run
├── merged/
│   └── llama-3.1-8b-legal-v1/   # deployment artifact
└── notebooks/
    └── 01-dataset-explore.ipynb
```

### Prerequisites

- **GPU access:** one card with ≥16 GB VRAM. On Modal / Runpod: A10G ($0.40/hr) or L4 ($0.70/hr). Local: RTX 3090/4090.
- **Hugging Face account** (accept Llama 3.1 license at meta-llama/Llama-3.1-8B-Instruct)
- **Anthropic API key** (or OpenAI) for the LLM-judge in evaluation
- Python 3.11+, `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`, `mlflow`, `fastapi`, `uvicorn`, `vllm`
- ~$5-15 of GPU credits for the training run + judge eval

---

## 2. Step-by-Step Milestones

### Milestone 1 — Build the dataset (4-6 hours)

Goal: 2500 rows, 7 balanced classes, deduplicated, format-consistent.

**Sources:**
1. **Public case law snippets** from CaseLaw Access Project (Harvard) or Free Law Project's CourtListener API. Extract 3-5 sentence excerpts labeled by area from the case metadata.
2. **Distilled examples from Claude/GPT** — hand-write 20 seed prompts per class, ask Claude Opus to generate 50 variations per seed (with your seeds as few-shots). Filter aggressively.
3. **Wikipedia legal encyclopedia entries** — abstracts of major cases, categorized by legal area.

```python
# src/build_dataset.py
import json
from datasets import Dataset
from datasketch import MinHash, MinHashLSH

CLASSES = ["contract", "torts", "criminal", "constitutional",
           "evidence", "property", "other"]

def load_raw_sources():
    """Merge from all sources into a single list of (text, label) tuples."""
    rows = []
    rows.extend(load_caselaw_snippets())      # your implementation
    rows.extend(load_distilled_from_claude())  # your implementation
    rows.extend(load_wiki_legal())            # your implementation
    return rows

def dedupe(rows):
    """MinHash near-dedup on text field."""
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    minhashes = {}
    kept = []
    for i, row in enumerate(rows):
        m = MinHash(num_perm=128)
        for w in row["text"].split():
            m.update(w.lower().encode())
        # Query for existing near-dupes
        if lsh.query(m):
            continue
        lsh.insert(i, m)
        minhashes[i] = m
        kept.append(row)
    return kept

def filter_length(rows, min_tok=15, max_tok=400):
    return [r for r in rows if min_tok <= len(r["text"].split()) <= max_tok]

def balance_classes(rows, target_per_class=350):
    """Downsample to a balanced set."""
    from collections import defaultdict
    import random
    by_cls = defaultdict(list)
    for r in rows: by_cls[r["label"]].append(r)
    random.seed(42)
    balanced = []
    for c in CLASSES:
        pool = by_cls[c]
        if len(pool) < target_per_class:
            print(f"WARNING: class {c} has only {len(pool)}")
            balanced.extend(pool)
        else:
            balanced.extend(random.sample(pool, target_per_class))
    return balanced

def to_messages_format(row):
    return {
        "messages": [
            {"role": "system", "content":
             "You are a legal-text classifier. Classify the passage by area of law. "
             "Respond with exactly one word from: " + ", ".join(CLASSES) + "."},
            {"role": "user", "content": row["text"]},
            {"role": "assistant", "content": row["label"]},
        ]
    }

def split(rows, train_frac=0.85, val_frac=0.10):
    import random
    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return rows[:n_train], rows[n_train:n_train+n_val], rows[n_train+n_val:]

if __name__ == "__main__":
    raw = load_raw_sources()
    print(f"Raw: {len(raw)}")
    filtered = filter_length(raw)
    print(f"After length filter: {len(filtered)}")
    deduped = dedupe(filtered)
    print(f"After dedup: {len(deduped)}")
    balanced = balance_classes(deduped)
    print(f"After balance: {len(balanced)}")

    train, val, test = split(balanced)
    for split_name, subset in [("train", train), ("val", val), ("test", test)]:
        with open(f"data/{split_name}.jsonl", "w") as f:
            for r in subset:
                f.write(json.dumps(to_messages_format(r)) + "\n")
        print(f"{split_name}: {len(subset)}")
```

**Exit criterion:** three files (`train.jsonl`, `val.jsonl`, `test.jsonl`) totaling ~2500 rows, roughly balanced across the 7 classes, no near-duplicates between splits.

### Milestone 2 — Establish base-model baseline (1 hour)

Before training, measure what Llama 3.1 8B Instruct can do zero-shot. This is your baseline for the A/B.

```python
# src/eval.py (baseline part)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

CLASSES = ["contract", "torts", "criminal", "constitutional",
           "evidence", "property", "other"]

SYSTEM = ("You are a legal-text classifier. Classify the passage by area of law. "
          f"Respond with exactly one word from: {', '.join(CLASSES)}.")

def classify_zero_shot(text):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=10, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    resp = resp.strip().lower()
    for c in CLASSES:
        if c in resp: return c
    return "other"

correct = 0
total = 0
test = [json.loads(l) for l in open("data/test.jsonl")]
for row in test:
    text = row["messages"][1]["content"]
    true_label = row["messages"][2]["content"]
    pred = classify_zero_shot(text)
    total += 1
    if pred == true_label: correct += 1

print(f"Base zero-shot accuracy: {correct/total:.3f}")
```

**Expected result:** 55-65% accuracy. If it's much higher, your task isn't hard enough to be interesting; if it's much lower, your test set is noisy or the system prompt needs work.

### Milestone 3 — Configure and launch QLoRA training (1-2 hours setup, ~40 min run)

```yaml
# configs/qlora.yaml (Axolotl-style, or use as reference for TRL)
base_model: meta-llama/Llama-3.1-8B-Instruct
tokenizer_type: AutoTokenizer
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: bfloat16

adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: data/train.jsonl
    type: chat_template
val_set_size: 0
test_datasets:
  - path: data/val.jsonl
    type: chat_template
    split: train

sequence_len: 1024
sample_packing: true

num_epochs: 2
micro_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.05
optimizer: paged_adamw_8bit
weight_decay: 0.0
bf16: true
flash_attention: true

logging_steps: 5
eval_steps: 50
save_steps: 50
save_total_limit: 3

output_dir: ./runs/llama-3.1-8b-legal-v1
mlflow_tracking_uri: file:./mlruns
mlflow_experiment_name: legal-classifier
```

Launch:
```bash
accelerate launch -m axolotl.cli.train configs/qlora.yaml
```

Or if using TRL directly, follow the script in `fine-tuning/learning/05-fine-tuning-ops.md` §2.2.

**What to watch during training:**
- Train loss should start around 1.8-2.0, drop to 0.3-0.6 by end
- Val loss should track within 0.1-0.2 of train
- If val loss climbs while train drops: overfitting, stop early
- Total wall time: 30-60 min on A10G, 15-25 min on A100/H100

**Exit criterion:** trained adapter saved at `./runs/llama-3.1-8b-legal-v1/final/`; MLflow run has all metrics logged.

### Milestone 4 — Evaluate fine-tune vs base (1-2 hours)

Full A/B on the test set:

```python
# src/eval.py (comparison part)
from peft import PeftModel

# Load fine-tuned model
ft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
ft_model = PeftModel.from_pretrained(ft_model, "./runs/llama-3.1-8b-legal-v1/final")
ft_model.eval()

def classify_ft(text):
    # Same as classify_zero_shot but using ft_model
    ...

# Run both on test set
results = []
for row in test:
    text = row["messages"][1]["content"]
    true = row["messages"][2]["content"]
    results.append({
        "text": text[:100],
        "true": true,
        "base_pred": classify_zero_shot(text),
        "ft_pred": classify_ft(text),
    })

# Compute accuracies
import pandas as pd
df = pd.DataFrame(results)
base_acc = (df.base_pred == df.true).mean()
ft_acc = (df.ft_pred == df.true).mean()
print(f"Base:     {base_acc:.3f}")
print(f"Fine-tune: {ft_acc:.3f}")

# Per-class breakdown
for c in CLASSES:
    subset = df[df.true == c]
    if len(subset) == 0: continue
    b = (subset.base_pred == c).mean()
    f = (subset.ft_pred == c).mean()
    print(f"  {c:<15} base={b:.3f}  ft={f:.3f}  delta={f-b:+.3f}")

# General-capability sanity check
# Run a small MMLU or HellaSwag subsample to check for catastrophic forgetting
```

Also run a **general-capability sanity check** — a small MMLU-style multiple-choice set (say 50 questions from MMLU across domains). Fine-tune should be within 3 pts of base; a drop of 5+ points means forgetting is happening despite QLoRA.

**Exit criterion:** Fine-tune ≥85% test accuracy; base <75%; general-capability delta within 3 points.

### Milestone 5 — Merge and prepare deployment artifact (30 min)

```python
# src/merge.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16
)
peft = PeftModel.from_pretrained(base, "./runs/llama-3.1-8b-legal-v1/final")
peft.eval()
merged = peft.merge_and_unload()
merged.save_pretrained("./merged/llama-3.1-8b-legal-v1", safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.save_pretrained("./merged/llama-3.1-8b-legal-v1")
```

**Verify** the merged model produces identical outputs to the un-merged combo on 20 test prompts. If they differ, dropout wasn't disabled during merge or something else went wrong.

### Milestone 6 — Serve behind FastAPI + vLLM (1-2 hours)

Two options:

**A) Simple: use vLLM's OpenAI-compatible server**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./merged/llama-3.1-8b-legal-v1 \
    --port 8000 \
    --max-model-len 2048
```

Then your FastAPI service can call it as if it were OpenAI.

**B) Direct FastAPI + vLLM Python API**

```python
# src/serve.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from vllm import LLM, SamplingParams

STATE = {}
CLASSES = ["contract", "torts", "criminal", "constitutional",
           "evidence", "property", "other"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["llm"] = LLM(
        model="./merged/llama-3.1-8b-legal-v1",
        dtype="bfloat16",
        max_model_len=2048,
    )
    STATE["sampling_params"] = SamplingParams(
        max_tokens=10, temperature=0.0, stop=["\n"]
    )
    yield
    STATE.clear()

app = FastAPI(lifespan=lifespan)

class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str

class ClassifyResponse(BaseModel):
    predicted_class: str
    raw_output: str

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if not (10 <= len(req.text) <= 4000):
        raise HTTPException(400, "text length must be 10-4000 chars")
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a legal-text classifier. Classify the passage by area of law. Respond with exactly one word from: {', '.join(CLASSES)}.<|eot_id|><|start_header_id|>user<|end_header_id|>

{req.text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    outputs = STATE["llm"].generate([prompt], STATE["sampling_params"])
    raw = outputs[0].outputs[0].text.strip().lower()
    predicted = next((c for c in CLASSES if c in raw), "other")
    return ClassifyResponse(predicted_class=predicted, raw_output=raw)

@app.get("/healthz")
def health():
    return {"status": "ok", "model_loaded": "llm" in STATE}
```

Test:
```bash
curl -X POST http://localhost:8000/classify \
    -H "content-type: application/json" \
    -d '{"text":"The plaintiff alleges that the defendant breached the terms of the merger agreement by failing to close on the scheduled date."}'
# Expected: {"predicted_class": "contract", "raw_output": "contract"}
```

### Milestone 7 — Add production discipline (2-3 hours)

Wire the deployment following the Month 6 playbook:
- Dockerize (multi-stage, non-root)
- Deploy on Modal / Runpod / your GPU-enabled Fly.io or Cloud Run alternative
- Add Prometheus metrics (RED + tokens + latency)
- Add structured logging with request_id
- Add rate limiting per user
- Add golden-set eval to CI
- Write a runbook: "if classification accuracy drops on the sampled prod stream, likely fine-tune drift; roll back to previous adapter version"

### Milestone 8 — Optional: A/B in shadow traffic (2-4 hours)

If you have real production usage:
1. Deploy fine-tune alongside base model
2. Send copies of production traffic to both
3. Log both predictions + user thumbs
4. After a week, compare confusion matrices and thumbs rates
5. Full rollout if fine-tune wins by a meaningful margin

---

## 3. Expected Outcomes

| Metric | Base (zero-shot) | Fine-tuned | Delta |
|--------|------------------|------------|-------|
| Test accuracy | ~60% | ~87% | +27 pts |
| Per-class F1 (macro) | ~0.55 | ~0.85 | +0.30 |
| MMLU sanity | ~68% | ~66% | -2 pts (acceptable) |
| p50 latency | ~120ms | ~120ms | ~0 (merged) |
| Response length | ~5 tokens (varies) | ~2 tokens (single class name) | shorter |
| Adapter file size | — | ~180 MB | — |
| Merged model size | — | ~16 GB | same as base |

If your numbers vary a lot from these, the debugging is educational:
- Fine-tune barely beats base → dataset too small or too noisy; re-curate
- Fine-tune massively beats base → possible train/test leakage; verify splits
- General-cap drop > 5pts → too many epochs, LR too high, or dataset too narrow

---

## 4. Extensions

- **Multi-class → hierarchical classification.** Add sub-classes (contract → sales/employment/lease). Task complexity increases; dataset needs 500+ per leaf class.
- **DPO on top of SFT.** Generate paired responses from your fine-tune; ask Claude to judge them; DPO train. Should further improve accuracy and reduce edge-case failures.
- **Try smaller base models.** Fine-tune Phi-3-mini (3.8B) or Qwen2.5-3B with the same dataset. Compare quality, latency, cost. This teaches the size-vs-quality trade-off.
- **Try full-parameter fine-tuning as an experiment.** Not for production — for the direct comparison against QLoRA. Verify LoRA quality is within 1-2% of full and speak to it in interviews.
- **Cross-lingual transfer.** Add Spanish or French legal text (with translations for training). See how much cross-lingual transfer occurs.
- **Ship a production Streamlit demo.** Public URL where users can paste text and see the classification. Great portfolio piece.

---

## 5. Deliverables Checklist

By the end you should have committed to a public GitHub repo:

- [ ] Dataset build script (`src/build_dataset.py`) with reproducible seed
- [ ] Curated dataset files (train/val/test JSONL) — or the script + instructions to regenerate if data is not shareable
- [ ] Training config (`configs/qlora.yaml`)
- [ ] Training script (`src/train.py`)
- [ ] Evaluation script (`src/eval.py`) with the A/B report saved as `reports/base-vs-ft-v1.md`
- [ ] Merged model in an artifact registry (HF Hub, S3, or private repo) — with model card
- [ ] Serving script (`src/serve.py`)
- [ ] Dockerfile + deployment config
- [ ] MLflow run logged (screenshot in README)
- [ ] README explaining the whole project, screenshot of results, link to deployed URL if public
- [ ] Blog post (portfolio-optional but strongly recommended)

---

## 6. Interview Talking Points

- **"Walk me through a fine-tune you did."** — Follow this exact structure: dataset, method, evaluation, deployment, honest results. 4-6 minutes.
- **"Why QLoRA?"** — 8B model on a single 24 GB GPU, ~1% trainable params, quality within 1-2% of full SFT, dramatically less catastrophic forgetting.
- **"Why 2 epochs?"** — Empirical sweet spot for LLM SFT; 1 sometimes underfits; 3+ overfits and degrades general capability. Watched val loss to confirm.
- **"How did you evaluate?"** — Task accuracy on held-out test + per-class F1 + MMLU sanity check for forgetting + latency benchmark. Base baseline for comparison; ship gate was ≥85% task accuracy and ≤3 point general drop.
- **"Why not just use a bigger model with better prompts?"** — Cost per inference (an 8B fine-tune is ~1/20th the cost of a Claude Sonnet call for the same output), latency (locally-served 8B is ~5× faster than a hosted API), and control (I can update it, deprecate it, self-host it). For a bounded classification task these matter.
- **"What would you do differently?"** — Bigger dataset (2500 → 5000+); stratified data collection to better cover minority classes; add DPO on top for edge-case robustness.

---

## 7. References

- Llama 3.1 model card and license (meta-llama/Llama-3.1-8B-Instruct)
- Axolotl QLoRA recipes — `examples/llama-3/qlora.yaml` in the Axolotl repo
- Hugging Face TRL fine-tuning cookbook
- Cai et al., "Legal-BERT: The Muppets Straight Out of Law School" (2020) — earlier legal-domain fine-tuning
- CaseLaw Access Project (Harvard Law)
- CourtListener API — https://www.courtlistener.com/help/api/
- Unsloth's Llama 3.1 fine-tuning notebook (for a lower-VRAM version)
