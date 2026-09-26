# Month 3 — Deep Learning 🔥 — Goals

> **Track:** Deep Learning (Module 3 of the broader learning journey)
> **Prerequisites:** Foundations + Month 2 (ML). Linear algebra, calculus (chain rule!), probability, Python.
> **Meta-goal:** *Cross the bridge into modern AI engineering — from a first neuron to a working transformer to using LLMs in real systems.*

---

## 1. The North Star

The single sentence that governs every decision this month:

> **"When I finish this month, I understand every layer, every gradient, and every design choice inside a modern transformer well enough that when I use an LLM in a system, I know what it's doing — I'm not just calling `.generate()` on faith."**

Every module, every exercise, and every evaluation below is scored against that sentence.

---

## 2. Terminal Learning Objectives (mastery, not exposure)

### 2.1 Neural Network Fundamentals
- **LO-1.1** Derive the forward pass of a fully-connected network for a mini-batch.
- **LO-1.2** Derive and hand-compute **backpropagation** for a 2-layer network with any loss + any activation.
- **LO-1.3** Explain **vanishing / exploding gradients** and how modern activations, init, and normalization fix them.
- **LO-1.4** Choose an appropriate **loss function** given the task (regression, classification, ranking, density).
- **LO-1.5** Choose an appropriate **optimizer** and know why (SGD, Momentum, Adam, AdamW).

### 2.2 Regularization and Generalization
- **LO-2.1** Apply and reason about **L2 weight decay**, **dropout**, **early stopping**, **label smoothing**, **batch/layer normalization**, **data augmentation**.
- **LO-2.2** Interpret **training loss curves** — spot overfitting, underfitting, learning-rate issues, dead neurons.
- **LO-2.3** Use **mixed precision (fp16/bf16)** and **gradient accumulation** to fit larger models.

### 2.3 Architectures
- **LO-3.1** Design and train a **CNN** end-to-end — conv, pool, batch-norm, residual blocks — for image classification.
- **LO-3.2** Explain **RNN / LSTM / GRU** and why they fail on very long sequences.
- **LO-3.3** Derive **scaled dot-product attention** from first principles.
- **LO-3.4** Build a **transformer encoder** and decoder block from scratch in PyTorch.
- **LO-3.5** Understand **positional encoding** (sinusoidal, learned, RoPE, ALiBi).

### 2.4 LLMs and Modern AI Engineering
- **LO-4.1** Explain the **pretraining → fine-tuning → RLHF/DPO** pipeline.
- **LO-4.2** Use a pretrained **HuggingFace** model for classification or generation.
- **LO-4.3** Apply **parameter-efficient fine-tuning** (LoRA, QLoRA).
- **LO-4.4** Do **prompt engineering** effectively and know its limits.
- **LO-4.5** Understand **inference costs** — quantization, KV cache, batching, sampling parameters.

### 2.5 PyTorch Fluency
- **LO-5.1** Build a model as an `nn.Module` subclass with proper parameter registration.
- **LO-5.2** Write a training loop with `DataLoader`, optimizer, scheduler, autocast, checkpointing.
- **LO-5.3** Debug NaNs, exploding losses, stuck models, and OOM errors.
- **LO-5.4** Move a model to GPU efficiently and use `DataParallel` / `DistributedDataParallel`.
- **LO-5.5** Use `torch.compile` (PyTorch 2.x) for speedups.

---

## 3. Deliverable

You will produce **one** portfolio-grade artifact:

**Build #1 — Image & Text Classification System.** A CNN for image classification AND a transformer-based text classifier, unified under one training/serving framework. See `builds/01-image-text-classification.md`.

The build is where reading turns into muscle memory. Ship it or the track is incomplete.

---

## 4. Definition of Done (per topic)

A topic is **done** when you can:

- [ ] Derive the math on a whiteboard without notes.
- [ ] Implement a **from-scratch NumPy or minimal PyTorch** version to prove understanding.
- [ ] Reproduce the standard PyTorch result to within numerical tolerance.
- [ ] Name one **failure mode** you've personally caused or debugged.
- [ ] Score ≥ 8/10 on the module's Self-Assessment Bank.

---

## 5. Anti-goals (what you are *not* doing here)

- **You are not training a state-of-the-art model.** Understanding beats leaderboard.
- **You are not doing pure ML research.** Reading a paper end-to-end once or twice is welcome; producing novel research is out of scope.
- **You are not skipping the math to "just write the code".** The whole point is *why*, not *how* to `.fit`.
- **You are not pretending TensorFlow doesn't exist**, but you're focusing on PyTorch — the modern research/industry default.
- **You are not deploying a foundation model to production this month.** That's month 4 (MLOps).

---

## 6. Cadence (suggested)

| Week | Focus |
|------|-------|
| 1 | Neural nets, backpropagation, activations, losses, optimizers — plus PyTorch basics |
| 2 | Regularization, CNNs |
| 3 | Sequence models, attention |
| 4 | Transformers, LLMs — **Build #1** |

Adjust to reality, but keep the **build immovable**.

---

## 7. Success Signals

You'll know Month 3 is behind you when:

- You can sketch the forward and backward pass of a transformer block on a napkin.
- You feel physical discomfort when you see a training loop that doesn't zero-grad, doesn't use autocast, or doesn't checkpoint.
- You know why AdamW is the modern default over Adam.
- You can read a paper's architecture section and pattern-match to what you've built.
- When someone says "attention is all you need," you know exactly which four equations they mean.
- You use HuggingFace but you're not scared to `model.forward` a batch manually and inspect the tensors.

---

## 8. Related Files

- `learning/01-neural-networks.md`
- `learning/02-backpropagation.md`
- `learning/03-activation-functions.md`
- `learning/04-loss-functions.md`
- `learning/05-optimizers.md`
- `learning/06-regularization.md`
- `learning/07-cnns.md`
- `learning/08-sequence-models.md`
- `learning/09-attention.md`
- `learning/10-transformers.md`
- `learning/11-llms.md`
- `learning/12-pytorch.md`
- `builds/01-image-text-classification.md`
- `self-assessment.md`
