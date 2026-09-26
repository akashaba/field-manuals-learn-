# Fine-Tuning — Goals

> **North star:** Know when to fine-tune (vs prompt-engineer, vs RAG, vs both), which technique to use (SFT / LoRA / QLoRA / DPO), how to build a dataset that doesn't ruin your model, and how to actually run the training on a single 24GB GPU without buying a supercomputer. Be able to ship a fine-tuned model behind an API and defend the decision to fine-tune in an interview.

Fine-tuning is the most feared and mystified topic in applied AI engineering. In practice most teams don't need it — good prompting and RAG cover 80% of applications. But for the 20% where fine-tuning wins, being the person on the team who can actually do it (not just talk about it) is a significant signal at Senior+ interview loops.

---

## What "fine-tuning" actually means

Fine-tuning = continuing to train an already-trained model on new data, updating (some of) its weights. Three flavors matter in 2026:

- **Supervised Fine-Tuning (SFT)** — teach the model to imitate a set of (input, desired output) pairs. This is what "fine-tuning" usually means.
- **Parameter-Efficient Fine-Tuning (PEFT), specifically LoRA/QLoRA** — SFT that only trains small adapter matrices, not the full model. ~1% of the parameters, ~1% of the VRAM, essentially the same quality on most tasks. QLoRA additionally 4-bit-quantizes the frozen base — this is why 8B and 70B models fit on consumer GPUs.
- **Preference / alignment tuning (DPO, ORPO, KTO)** — teach the model from *pairs of responses* ranked by preference. This is the modern replacement for RLHF for most practical purposes.

Modern LLMs (Claude, GPT-4/5, Llama 3.x, Qwen 2.5) go through: pretraining → SFT → preference tuning. When you fine-tune a base model, you're either doing the middle two steps yourself or continuing them.

---

## Learning outcomes

By the end of this track you should be able to:

**Decision-making**
- State the four questions that determine whether to fine-tune at all
- Compare fine-tuning vs. RAG vs. prompt-only economics and quality for a given task
- Choose SFT vs. LoRA vs. QLoRA vs. DPO for a specific problem

**Dataset craft**
- Build a 500-5000 example dataset with the right format, quality bar, and length distribution
- Detect and mitigate label leakage, prompt duplication, and toxic examples
- Use synthetic data generation responsibly

**Training operations**
- Run a LoRA fine-tune on a single 24 GB GPU (A10, RTX 4090, L4)
- Read and interpret a loss curve; know when to stop
- Merge adapters, quantize for serving, and export in ready formats

**Evaluation**
- Compare fine-tuned vs. base on a held-out set with an LLM-judge
- Detect catastrophic forgetting on general benchmarks
- Cost the fine-tuning run and the resulting inference

**Deployment**
- Serve the adapter via vLLM or Text Generation Inference
- A/B a fine-tuned model against the base in shadow traffic
- Know when to *stop* using the fine-tuned version

---

## The 5-module track + 1 capstone

| # | Module | Question it answers |
|---|--------|--------------------|
| 01 | SFT fundamentals | What supervised fine-tuning *does* to a model, mathematically |
| 02 | PEFT / LoRA / QLoRA | Why we don't full-parameter fine-tune anymore |
| 03 | DPO & preference tuning | How to teach the model *style* and *ranking*, not just imitation |
| 04 | Dataset curation | The real work — how the dataset determines the outcome |
| 05 | Fine-tuning ops | Training infra, evaluation, deployment, monitoring |
| Capstone | Fine-tune a legal-text classifier on Llama 3.1 8B | End-to-end: dataset → QLoRA → eval → serving |

**Time budget:** 3–4 weeks of evenings and weekends for the modules; the capstone build is a full weekend.

---

## Prerequisites (all covered in earlier months)

- Python + PyTorch fluency (Month 3)
- Neural network training basics — loss, gradient, optimizer (Month 3)
- Transformer architecture — attention, MLP, layer norm (Month 3 + Capstone 03)
- LLM APIs and tokenization (Month 4)
- Basic experiment tracking with MLflow (Month 6)

Plus one physical prerequisite: **GPU access**. Options in order of accessibility:
- Colab Pro / Pro+ (~$10-50/mo) — A100 access, easiest onramp
- Modal / Runpod / Vast.ai (~$0.30-1.00/hr on-demand H100 or A100)
- Local RTX 3090/4090 or Mac M-series (M2/M3/M4 max with unified memory)

You do not need to own a GPU to do this track.

---

## What this track is not

- Not a research-scientist track. We use techniques that are stable and shipped; we don't train new architectures.
- Not a "how to fine-tune GPT-4" guide. Managed models can be fine-tuned via provider APIs, and the deltas from what's below are minor. We focus on open-weight models because that's where the interesting engineering lives.
- Not RLHF from scratch. That's a research project. DPO gets 90% of the value at 1% of the complexity — that's what we teach.

Let's begin.
