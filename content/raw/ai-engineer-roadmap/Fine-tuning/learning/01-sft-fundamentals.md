# 01 — Supervised Fine-Tuning Fundamentals

> **Module goal:** Understand exactly what SFT does to a pretrained language model — what changes in the weights, what the loss is over, why it works, and when it doesn't. Be able to articulate the difference between pretraining, SFT, and preference tuning at the level a Senior AI Engineer interview expects.

---

## 1. Executive Summary & Core Concepts

**Supervised Fine-Tuning (SFT)** is the process of taking an already-trained language model and continuing to train it on a curated set of **(prompt, desired response)** pairs. The training objective is the same as pretraining — next-token prediction (cross-entropy loss) — but the data changes character: from a large messy corpus of "internet text" to a small clean corpus of "how we want the model to behave."

The training loop:
1. Feed the model a prompt-response pair, tokenized and concatenated.
2. Compute cross-entropy loss on the response tokens (usually only — see prompt masking below).
3. Backprop; update weights.
4. Repeat over the dataset for 1–3 epochs.

That's it. Everything else in this track is either (a) making SFT more efficient (LoRA), (b) using preference pairs instead of imitation (DPO), or (c) getting the dataset right.

**Why SFT works:** the pretrained model already knows language, world facts, and syntax. SFT nudges it toward a specific *style* of response: format, persona, task shape. It's not teaching new facts (that's what RAG or continued pretraining is for) — it's teaching *behavior*.

**Three questions to decide whether to SFT:**

1. Can prompt engineering + few-shot examples get you there? If yes, don't fine-tune.
2. Is the task a *behavior* the model can express but doesn't reliably, or does it require *knowledge* the model lacks? Behavior → SFT good; knowledge → RAG better.
3. Do you have or can you build 500+ high-quality examples? Below that, the cost/benefit rarely works out.

Common SFT wins:
- Structured output (specific JSON schemas, XML tags)
- Domain-specific style (legal, medical, technical writing conventions)
- Reduced verbosity for latency/cost
- Refusal patterns aligned to your product's policy
- Function-calling accuracy on your specific tool set

Common SFT losses (where teams waste effort):
- Injecting factual knowledge (use RAG)
- Fixing arithmetic (use tool-calling)
- Matching a target model's quality on a general benchmark (dataset dominates model choice)

---

## 2. Deep-Dive Breakdown

### 2.1 The Loss Objective — Same Math as Pretraining

Given input sequence $x = (x_1, x_2, \ldots, x_T)$, the causal language modeling loss is:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

For fine-tuning, we split $x$ into a **prompt** portion and a **response** portion:

$$
x = \underbrace{(p_1, \ldots, p_n)}_{\text{prompt}} \circ \underbrace{(r_1, \ldots, r_m)}_{\text{response}}
$$

**Prompt masking (aka completion-only training):** we compute the loss only over the response tokens, not the prompt:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=n+1}^{n+m} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

Concretely: the prompt tokens contribute to *conditioning* (the model attends to them) but don't count in the loss. Why? Because we don't want the model to learn to *generate* prompts — we want it to learn to *respond* to them. Loss on prompt tokens teaches the model to reproduce user text, wasting capacity on useless targets.

Every serious SFT framework (Hugging Face TRL, Axolotl, torchtune) supports prompt masking; make sure it's on. It's the single biggest correctness detail.

### 2.2 The Chat Template

Modern LLMs use a **chat template** — a structured way to encode multi-turn conversations as a flat token sequence. Llama 3.1's template looks like:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What is 2+2?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

4
<|eot_id|>
```

**During fine-tuning you MUST use the exact chat template the model was trained with.** Using a different template produces "template drift" — the model's response quality collapses because the pretraining/SFT signals it was calibrated on are no longer present. Every Hugging Face tokenizer for a chat model has a `.apply_chat_template()` method that emits the correct format.

Also: you compute loss only on the tokens *after* `<|start_header_id|>assistant<|end_header_id|>\n\n` and up to (and including) `<|eot_id|>`. Any framework worth using has a "response template" or "instruction template" argument that handles this.

### 2.3 Hyperparameters That Actually Matter

Most fine-tuning results are decided by a handful of choices:

| Hyperparameter | Typical range | What it controls |
|----------------|---------------|------------------|
| **Learning rate** | 1e-5 to 5e-5 (full SFT), 1e-4 to 3e-4 (LoRA) | The biggest lever; too high = catastrophic forgetting, too low = no learning |
| **Epochs** | 1–3 | More = overfit fast; you almost never want > 3 |
| **Batch size (effective)** | 8–64 | Uses gradient accumulation to fit big effective batches on small GPUs |
| **Warmup ratio** | 0.03–0.1 | Fraction of steps ramping LR from 0 to peak; stabilizes early training |
| **LR scheduler** | cosine or linear | Decay to (near-)zero by end of training |
| **Weight decay** | 0.0 to 0.1 | Regularization; often 0 with LoRA |
| **Max seq length** | 2048 to 4096 | Longer = more VRAM; truncate or pack |

**The single most impactful debug lever is learning rate.** If your loss doesn't decrease at all, LR is too low. If it decreases then diverges (NaN or explodes), LR is too high. Sweep three or four values on a small subset before committing to a full run.

### 2.4 Catastrophic Forgetting

The dark side of SFT: you can *break* the base model's general capabilities while teaching it a narrow task. Symptoms:

- Fine-tuned model excels on your task but fails at basic reasoning ("what is 2+2" now returns hallucinated math)
- General benchmarks (MMLU, HellaSwag) drop 5-15 points
- Model becomes stilted or overly formal when asked casual questions

Causes:
- Too many epochs (3+ almost always damages)
- LR too high
- Dataset too narrow (all examples look similar)
- Full-parameter fine-tuning on a small dataset

Mitigations:
- **Use LoRA/QLoRA** (Module 02) — dramatically reduces forgetting because base weights don't move
- **Add "general" examples to your dataset** (10-20% of it) drawn from general instruction-tuning corpora (Alpaca, ShareGPT, Dolly)
- **Stop training early** — evaluate general capability every N steps, stop when it drops
- **Keep the LR small** — 1e-5 for full SFT, 2e-4 for LoRA — and use warmup

### 2.5 Packing vs. Padding

Fine-tuning batches are variable-length. Two strategies for the ragged edges:

**Padding:** every sequence padded to `max_seq_length`. Wastes compute (padded tokens are masked from loss but still computed by attention).

**Packing:** concatenate multiple examples into one `max_seq_length` sequence, separated by a special token, with attention masking so examples don't attend across boundaries. ~30-70% throughput gain, standard in modern trainers.

`packing=True` in TRL's `SFTTrainer` gives you it. Turn it on unless your examples are already near-max-length.

`[IMG-FT01-01]` — *Prompt: A textbook illustration of the SFT pipeline. Left: pretrained base model shown as a large stack of transformer blocks. Middle: a dataset icon (JSONL) flowing into a training loop diagram — small batch → forward pass → cross-entropy loss ONLY on response tokens (highlighted region in a token sequence with prompt tokens grayed out and response tokens colored bright) → backward pass → parameter update. Right: the resulting fine-tuned model, same architecture but with slightly modified weights indicated by a subtle color shift. Below: a loss curve showing typical descent from ~2.0 to ~0.7 over 1000 steps with occasional bumps. Clean textbook illustration style with soft blues and one accent color.*

---

## 3. Mental Models & Analogies

### Model 1: The Method Actor Coach

A pretrained model is a working actor — they can read a script, deliver lines, adopt accents. **SFT is a director drilling them for a specific role.** Repeated (prompt, response) examples are like table reads: "for this kind of scene, this is the tone we want."

The actor doesn't forget English while learning the role. But if the director drills them for weeks on nothing but Shakespearean tragedy, they may struggle to switch back to sitcom timing — that's catastrophic forgetting. Good directors mix rehearsal material precisely to avoid this: a few pages of the target role, occasionally interleaved with other scripts to keep the range alive.

The dataset is the director's rehearsal plan. Bad rehearsal = bad performance, no matter how talented the actor.

### Model 2: The Grooved Path

Imagine the base model's response space as a wide plain with faint tracks worn by pretraining. When you ask a base model a question, it wanders in the general direction indicated by those tracks — sometimes on-target, sometimes not, and it may be verbose or malformed.

SFT is walking the same path many times with a specific style. After 1000 traversals, the path becomes a groove. The model *doesn't* forget the surrounding plain — it can still leave the groove — but the groove is now the default. Ask for a JSON response after SFT on JSON: it produces JSON. Not because it "learned JSON" (it already knew) but because the groove of "responses to this shape of prompt look like this" is now deep enough to dominate.

Too much grooving on too-similar prompts erodes the surrounding plain — the model can't leave the groove even when it should. That's overfitting, and it's why 1-3 epochs is the norm.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — "Fine-tuning will make the model know new facts."**
It will *sometimes*, unreliably, and only for facts stated repeatedly in the training data. And it will confidently *invent* related facts it hasn't seen. Fine-tuning is a poor knowledge-injection tool because:
1. LLMs blur facts under training — a rare fact in 100 examples may or may not survive.
2. The model gets *more confident* about wrong answers to related questions.
3. Updating the "known facts" requires re-fine-tuning; RAG lets you update the retrieval corpus in seconds.

**Use RAG for knowledge. Use SFT for behavior.** This is the single most durable rule in applied fine-tuning.

**Pitfall #2 — "I'll train for 10 epochs to make sure it really learns."**
No. LLMs memorize fast. By epoch 3 on most datasets, your training loss is near zero and your held-out perplexity is climbing (you're overfitting). The model has memorized the training examples but generalizes worse. Symptoms:
- Model repeats training examples verbatim on similar prompts
- General benchmarks tank
- Refusal patterns break

Watch validation loss; stop when it starts climbing. Or just default to 1 epoch and only bump to 2-3 if evaluation says you're still improving.

**Pitfall #3 — Wrong chat template = silent quality collapse.**
Common bug: fine-tune Llama 3.1 with the Llama 2 chat template (`<s>[INST]...[/INST]`), or use no template at all. Training loss looks fine. Eval quality is terrible. The model was trained to expect very specific control tokens; without them, it never enters "assistant mode" properly.

Fix: always use `tokenizer.apply_chat_template()` from the exact tokenizer of your base model. Verify a sample training example by decoding it back — it should look like the format in the model card.

Related: if you fine-tune a base model (not a chat model — e.g., `Llama-3.1-8B` vs `Llama-3.1-8B-Instruct`), you're on the wrong start point unless you specifically want to define your own chat format. Fine-tune the Instruct version 90% of the time.

---

## 5. Self-Assessment Bank

**Q1 (MC):** Prompt masking in SFT means:
A) The prompt tokens are hidden from the model
B) The loss is computed only over response tokens, not prompt tokens
C) The prompt is anonymized before training
D) The prompt is dropped from the input

**Q2 (short):** Why does using the wrong chat template silently destroy fine-tuning quality?

**Q3 (MC):** You have a task where the model needs to produce a very specific JSON schema. Prompt engineering gets 80% accuracy. You have 2000 hand-labeled examples. Which is the right move?
A) Continue prompt engineering
B) RAG
C) SFT with LoRA
D) Full-parameter SFT

**Q4 (short):** Explain catastrophic forgetting and name three mitigations.

**Q5 (MC):** For SFT of Llama 3.1 8B on 3000 examples, a reasonable epoch count is:
A) 0.5
B) 1–3
C) 5–10
D) 20+

**Q6 (short):** Your fine-tuned model excels at the target task but fails simple arithmetic that the base model handled. What likely happened and how would you diagnose it?

**Q7 (MC):** "Packing" in fine-tuning refers to:
A) Compressing the model weights
B) Concatenating multiple training examples into one sequence with attention masks
C) Batching multiple GPUs
D) Quantizing the optimizer state

**Q8 (short):** You want the model to "know" that your company's HR policy allows 15 sick days. Fine-tune or RAG? Why?

**Q9 (MC):** Loss during your training run starts at 2.1, drops to 0.4 by step 200, and stays at 0.4 for the remaining 1800 steps. What does this suggest?
A) Training is going great
B) The model has memorized the dataset and is now overfitting
C) The learning rate is too high
D) The batch size is too small

**Q10 (short):** Name three "wins" and three "losses" for SFT — tasks where it works well vs where it wastes effort.

---

### Answer Key

**A1: B.** Prompt masking means the cross-entropy loss is computed only on response tokens. Prompt tokens still condition the model (attention flows over them), but they don't produce gradients. This prevents the model from learning to reproduce user prompts, which is wasted capacity and can actively hurt quality.

**A2:** LLMs are trained with specific control tokens (`<|start_header_id|>assistant<|end_header_id|>`, `[INST]`, etc.) that signal "you are now responding as the assistant." These tokens are baked into the model's pretraining conditioning. Fine-tuning with the wrong template means gradient updates target sequences that don't include these signals — the model unlearns to *enter assistant mode* while learning your task. At inference time when you present the correct template, the model doesn't recognize it as strongly. Symptoms: shorter responses, less coherent style, worse safety/refusal behavior, occasionally the model starting to hallucinate the "wrong" template's control tokens. Fix: always use `tokenizer.apply_chat_template()` for your specific base model.

**A3: C.** LoRA-based SFT. Prompt engineering at 80% suggests the task is learnable but not clean; 2000 hand-labeled examples is well above the SFT viability threshold (~500). LoRA gives you the quality gain at ~1% of the VRAM and compute of full SFT. Full SFT (D) works but wastes money and increases forgetting risk with no measurable quality benefit at this dataset size.

**A4:** Catastrophic forgetting = SFT damages the base model's general capabilities while teaching a narrow task. The model becomes worse at things it used to do (arithmetic, general reasoning, casual conversation) because gradient updates pushed weights away from patterns that supported those behaviors.

Three mitigations: (1) **LoRA/QLoRA** — freeze the base weights entirely, train only adapters, so general capabilities are structurally preserved; (2) **mix in general instruction data** (Alpaca, ShareGPT samples at 10-20% of the dataset) to keep general behavior in the training signal; (3) **stop training early / few epochs** — evaluate on general benchmarks periodically and stop when they drop meaningfully.

**A5: B.** 1-3 epochs is the standard range for LLM SFT. 0.5 is fine for very large datasets (>50k examples); 5+ almost always overfits and degrades general capability. On 3000 examples, start with 1 epoch, add 1 more only if held-out metrics are still climbing.

**A6:** Likely catastrophic forgetting. Diagnosis: (1) evaluate the fine-tuned model on a small general-capability battery (basic arithmetic, MMLU sample, HellaSwag sample) and compare to the base model's score before shipping; (2) if you didn't do that, run it now and compare; (3) look at your training config — high LR (>3e-4 for LoRA, >5e-5 for full), too many epochs, narrow dataset. Fix by re-training with LoRA (if you weren't), lower LR, fewer epochs, or add general instruction data to the mix.

**A7: B.** Packing = concatenating shorter examples into one long sequence with attention masks preventing cross-example attention. Standard in TRL, Axolotl, torchtune. Improves throughput significantly (30-70%) with no quality change because the attention mask ensures examples remain independent from the model's perspective.

**A8:** RAG, not fine-tuning. HR policy is *knowledge* — a discrete fact that (a) needs to be verifiable, (b) may change (policy updates), (c) is one of many similar facts you'll want the model to know. Fine-tuning would produce an unreliable memorization ("the policy is 15… or 12… or 20 days"); RAG retrieves the exact policy document, model cites it, and updating is as simple as replacing the doc. Fine-tuning here is expensive, error-prone, and wrong. Use SFT for how the model *responds* about HR (tone, format, disclaimer discipline); use RAG for *what* it says.

**A9: B.** Rapid drop to 0.4 then flat plateau means the model has memorized the dataset quickly. On a small dataset (< ~10k examples), 200 steps of LLM SFT is often enough to memorize; the remaining 1800 steps are overfitting. Stop early or reduce dataset size. If training loss is 0.4 but held-out loss is climbing, you're already past the sweet spot. (C, D, A are all defensible in different scenarios but the pattern described is textbook memorization + overfitting.)

**A10:** Wins: structured output (specific JSON schema), domain-specific style (legal/medical writing conventions), function-calling with your specific tool set, refusal pattern aligned to product policy, reduced verbosity for latency, tone consistency in a specific persona. Losses: injecting factual knowledge (use RAG), fixing arithmetic (use tool-calling with a calculator), improving general benchmarks (dataset quality dominates), teaching the model reasoning it doesn't have (SFT can't imbue new capability, only smooth existing capability into a target shape), keeping-current information (needs continuous update — RAG is superior).

---

**Related modules:**
- `learning/02-peft-lora.md` — why we don't do full SFT anymore
- `learning/03-dpo-preference-tuning.md` — the next step after SFT
- `learning/04-dataset-curation.md` — where the results actually come from
- `../deep-learning/learning/11-llms.md` — the pretraining background
- `../capstones/03-understand-transformers.md` — the architecture SFT operates on

**Practice prompts:**
1. Take an open Instruct model (Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct). Load its tokenizer. Verify what its chat template renders for a simple 2-turn conversation.
2. Write a "not-fine-tune" decision doc for one of your projects — argue whether prompt engineering + RAG could cover it before spending time on SFT.
3. Look at any three fine-tuning tutorials online. Note which ones handle prompt masking correctly and which silently compute loss on prompt tokens.

**References:**
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — foundational for Module 02
- Wolf et al., "TRL: Transformer Reinforcement Learning" — the go-to library
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT paper)
- Zhou et al., "LIMA: Less Is More for Alignment" (2023) — argues that small high-quality SFT datasets dominate
- Alignment Handbook (Hugging Face) — https://github.com/huggingface/alignment-handbook
