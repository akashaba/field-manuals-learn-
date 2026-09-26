# Large Language Models (LLMs) — Master Study Guide

> **Track:** Deep Learning · **Module:** 11 — *The bridge into modern AI engineering.*
> **Prerequisites:** Modules 01–10 (especially 09 attention and 10 transformers).
> **Time budget:** ~15–20 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** LLMs (Large Language Models) are the current center of gravity of AI. They power ChatGPT, Claude, Gemini, Copilot, and thousands of internal enterprise tools. Every application-layer AI engineer today needs fluency in:

- What an LLM actually is under the hood (a transformer, trained by next-token prediction).
- The pretraining → post-training → serving lifecycle.
- Prompt engineering — what it can and can't do.
- Fine-tuning — full, LoRA, QLoRA, RLHF, DPO.
- Serving — inference performance, context management, evaluation.
- Retrieval-Augmented Generation (RAG), tool use, agents.

This module is the **bridge** the goal document referenced. Everything you learned about transformers now maps onto GPT-4, Claude, LLaMA — the difference is scale, data, and post-training.

**Fundamental principles you must own:**

1. **An LLM is a decoder-only transformer** trained to predict the next token on massive text corpora.
2. **Pretraining is the expensive step**; fine-tuning and alignment are relatively cheap.
3. **Emergent abilities** — behaviors that appear only at scale (few-shot in-context learning, chain-of-thought reasoning).
4. **Post-training (SFT + RLHF/DPO)** makes a raw language model into a helpful assistant.
5. **Prompt engineering** is real but has limits — anything the base model can't do won't be unlocked by a magic prompt.
6. **Inference cost dominates** — production LLM systems live and die by their token-throughput economics.

If you retain nothing else: **an LLM is a next-token predictor good enough that predicting the next token requires modeling the world.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Tokenization

The LLM doesn't see characters or words — it sees **tokens**, integer IDs from a fixed vocabulary of 30k–200k subword pieces.

**Byte-Pair Encoding (BPE)** — the dominant tokenization method:

1. Start with individual bytes / characters.
2. Repeatedly find the most frequent adjacent pair and merge it into a new token.
3. Stop at a target vocabulary size.

**Example (illustrative):**

```
"unbelievable" → ["un", "believ", "able"]
"tokenization" → ["token", "ization"]
```

**Variants:**
- **BPE (GPT-2, GPT-3, GPT-4)** — the classic.
- **WordPiece (BERT)** — similar; slightly different merging criterion.
- **Unigram (SentencePiece — T5, LLaMA)** — probabilistic; treats vocabulary as a probability distribution.
- **Byte-level BPE (GPT-2 onward)** — works on bytes, so it handles any character in any language.

**Why subwords, not words?**
- **Words** — vocabulary explodes; OOV (out-of-vocabulary) is a problem.
- **Characters** — sequences become very long; each character has little semantic info.
- **Subwords** — best of both. Common words are one token; rare words split into pieces.

**Practical consequences:**
- **Token count ≠ word count.** English is ~4 characters per token on average. "Deep learning" might be 2 tokens; "photosynthesis" 3.
- **Non-Latin scripts** often tokenize into more tokens per character — Chinese, Japanese, Arabic can be 1 character = 1 token, while a word takes 5. Watch pricing.
- **Whitespace matters.** " Hello" and "Hello" are usually different tokens.
- **BPE artifacts** — tokenizers introduce quirks. "Doubling" a punctuation ends up as its own weird token. This is why LLMs sometimes fail at reversing strings or counting letters — they don't see letters.

**Tools:**
- `tiktoken` (OpenAI): `tiktoken.get_encoding("cl100k_base")` — GPT-3.5/4 tokenizer.
- `transformers.AutoTokenizer` — HuggingFace's universal wrapper.
- `sentencepiece` — the T5/LLaMA library.

---

### 2.2 Pretraining, Data, and Emergent Abilities

**Pretraining objective**: next-token prediction (causal language modeling):

$$\mathcal{L} = -\sum_{t=1}^T \log P(x_t \mid x_1, \ldots, x_{t-1}; \theta)$$

The model is trained on a **web-scale corpus** — internet text, code, books, Wikipedia, papers, forums. Cleaned, deduplicated, filtered.

**Corpus size** — modern frontier LLMs train on trillions of tokens. LLaMA-3 8B was trained on ~15T tokens.

**Compute** — measured in GPU-hours or FLOPs. GPT-4 reportedly took months on thousands of H100 GPUs; training compute in the $10^{25}$–$10^{26}$ FLOPs range.

**Data quality > data quantity** past a point. High-quality, deduplicated, filtered data yields substantially better models than raw internet dumps. The best open-source pretraining datasets — Pile, RedPajama, SlimPajama, FineWeb-Edu — are curated.

**Emergent abilities** (Wei et al., 2022) — capabilities that appear only above a scale threshold:

- **Few-shot in-context learning** — GPT-3 sized. Give a few examples in the prompt; the model performs the task without gradient updates.
- **Chain-of-thought reasoning** — appears around GPT-3.5.
- **Instruction following** — needs post-training but also a certain scale of pretraining.
- **Multilingual transfer** — smaller models generalize poorly across languages; larger ones do.

Whether these are truly *emergent* (phase transitions) or just gradual capabilities revealed by better metrics is contested (Schaeffer et al., 2023). Practically, bigger models can do things smaller ones can't.

**Scaling laws** (Chinchilla, Module 10) — for a fixed compute budget:

- Optimal ratio of tokens to parameters is ~20:1.
- Loss decreases as a power law of parameters, data, and compute.
- No sign of the curve flattening within current-scale experiments.

**Base model outputs** — a raw pretrained LLM (before fine-tuning) is a **text completer**. Given "The capital of France is", it says "Paris." Given "Q: What is X? A:", it continues the pattern. It's not yet an "assistant" — that's what post-training adds.

---

### 2.3 Post-Training: SFT, RLHF, DPO

Turning a raw LLM into a helpful assistant like Claude or GPT-4 requires multiple post-training stages.

**Supervised Fine-Tuning (SFT):**

- Train on curated (instruction, response) pairs written by humans.
- The model learns the "answer questions helpfully" pattern.
- Datasets: ShareGPT, OpenAssistant, Alpaca, Dolly, WizardLM, curated proprietary sets.
- Small-scale fine-tuning (thousands to hundreds of thousands of examples).

**Reinforcement Learning from Human Feedback (RLHF):**

Three sub-steps:

1. **Reward model** — collect pairs of responses (A, B); a human labels which is better. Train a **reward model** (typically another LLM with a scalar output) to predict the human's preference.
2. **PPO fine-tuning** — use the reward model as a reward signal; run Proximal Policy Optimization (a reinforcement-learning algorithm) to fine-tune the LLM to maximize expected reward.
3. **KL penalty** — regularize the RL to not drift too far from the SFT model, preserving general capabilities.

Result: the model is *aligned* with human preferences — helpful, honest, harmless.

**Direct Preference Optimization (DPO)** — introduced 2023, now widely used:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)$$

Where $y_w$ is the "winning" response, $y_l$ the "losing," and $\pi_{\text{ref}}$ is a frozen reference (typically the SFT model).

- **No separate reward model.** DPO directly optimizes for the same objective RLHF targets.
- **No RL loop.** Simpler, more stable, cheaper.
- Now standard for many open-weights models (Zephyr, Tulu, LLaMA-3 Instruct).

**Constitutional AI / RLAIF** (Anthropic, 2022) — replace some human preferences with AI-generated preferences guided by principles (a "constitution"). Scales alignment.

**Alignment is not fully solved.** Models still hallucinate, produce harmful outputs when jailbroken, and have subtle biases. Alignment research is a huge active area.

---

### 2.4 Parameter-Efficient Fine-Tuning: LoRA, QLoRA

Full fine-tuning of a 70B model requires hundreds of GB of GPU memory (optimizer state alone). Parameter-Efficient Fine-Tuning (PEFT) methods let you fine-tune with a small fraction of that.

**LoRA (Low-Rank Adaptation)** (Hu et al., 2021) — the dominant method.

**Idea:** freeze the pretrained weight matrix $W \in \mathbb{R}^{d \times k}$. Add a **low-rank update** $\Delta W = BA$ where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, and $r \ll \min(d, k)$. During fine-tuning, only $A$ and $B$ are trained; the original $W$ stays frozen.

$$h = Wx + \Delta W x = Wx + BAx$$

**Parameters trained:** $r \cdot (d + k)$ vs $d \cdot k$ for full fine-tuning. Reduction of ~$1000\times$ for $r = 8$ typical.

**Which layers to LoRA:** attention Q and V projections (original paper); modern practice adds all attention + FFN projections.

**Hyperparameters:**
- **rank $r$** — 8, 16, or 32 typical. Higher = more capacity.
- **$\alpha$** — a scaling factor; effective LR for the LoRA weights is $\alpha / r$.
- **dropout** — small dropout on LoRA activations, often 0.05.

**QLoRA** (Dettmers et al., 2023) — LoRA + 4-bit quantized base model:

- Store the base model in 4-bit precision (NF4 dtype).
- Dequantize on the fly for forward/backward pass.
- LoRA weights stay in fp16/bf16.
- Enables fine-tuning **70B models on a single 48 GB GPU.**

**Serving LoRA models:**

- Store base model once; distribute LoRA adapters (~50 MB each) for each fine-tuned variant.
- Can serve **many LoRAs on the same base** via batched adapter routing.
- Or **merge** the LoRA into the base weights ($W \leftarrow W + BA$) — a normal model afterward.

**Alternatives:**

- **Adapters** — small bottleneck layers inserted between transformer layers.
- **Prefix tuning / prompt tuning** — learn a small set of virtual tokens prepended to inputs.
- **BitFit** — fine-tune only biases.

For most practical use, **LoRA / QLoRA is what you should reach for.**

---

### 2.5 Inference: Serving, Sampling, Context, Cost

**Autoregressive generation loop:**

```
tokens = tokenize(prompt)
for _ in range(max_new_tokens):
    logits = model(tokens)               # (T, vocab_size)
    next_tok = sample(logits[-1])
    tokens.append(next_tok)
    if next_tok == EOS: break
return detokenize(tokens)
```

**Sampling strategies:**

- **Greedy** — argmax at each step. Deterministic, boring, often loops.
- **Beam search** — track top-K partial hypotheses. Better for structured output (translation); rarely used for open chat.
- **Temperature sampling** — softmax with temperature $T$: divide logits by $T$ before softmax.
  - $T \to 0$ = greedy.
  - $T = 1$ = "raw" softmax distribution.
  - $T > 1$ = more diverse, more random.
- **Top-$k$ sampling** — sample from the $k$ most likely tokens.
- **Top-$p$ (nucleus) sampling** — sample from the smallest set whose cumulative probability exceeds $p$ (typically 0.9). Modern default.
- **Min-$p$** — set threshold as fraction of the top probability. Recent addition, sometimes better than top-$p$.

Common defaults: temperature 0.7, top_p 0.95 for creative tasks; temperature 0 for deterministic tasks (code, math).

**KV cache** — see Module 10. Absolutely essential; without it, generation is quadratic per token.

**Speculative decoding** — a small "draft" model proposes several tokens; the large model verifies in one pass. Big speedup with no quality loss.

**Quantization for inference:**

- **INT8** — usually near-free quality loss.
- **INT4 (GPTQ, AWQ, GGUF)** — 4× memory savings, noticeable but often acceptable quality loss.
- **FP8** — modern trick, uses new GPU features.

**Batching:**

- **Static batching** — group requests; wait for all to finish before serving next batch.
- **Continuous batching** — as one sequence finishes, another slot fills. Much better GPU utilization; used by vLLM, TensorRT-LLM.

**Serving stacks:**

- **vLLM** — most popular open-source inference server; PagedAttention for KV cache.
- **TGI (Text Generation Inference)** — HuggingFace's server.
- **TensorRT-LLM** — NVIDIA's optimized inference kernels.
- **llama.cpp** — CPU/consumer-GPU inference in C++.
- **Ollama, LM Studio** — user-friendly wrappers around llama.cpp.

**Cost math.** For a 70B model on an H100 (~40 tokens/sec typical), a 1000-token response costs roughly a few seconds of GPU time. GPU-hour cost × generation time × request volume = your infra bill. LLM economics live and die on caching, KV reuse, batching, and quantization.

**Retrieval-Augmented Generation (RAG):**

Instead of pushing everything into the model's context, retrieve relevant chunks from a vector database:

1. **Embed** all documents; store in a vector index (FAISS, Pinecone, Qdrant, pgvector).
2. **At query time**, embed the query; retrieve top-$k$ most similar chunks.
3. **Inject retrieved context** into the LLM's prompt.

RAG is how modern LLM apps handle domain knowledge, freshness, and citations without fine-tuning.

**Agents and tool use:**

- **Function calling / tool use** — the model outputs structured requests to call external tools (search, calculator, code execution, DB queries).
- **ReAct pattern** — Reason + Act loop. Model reasons about what to do next, calls a tool, observes the result, iterates.
- **Multi-step agents** — plan → execute → reflect. Frameworks: LangChain, LangGraph, DSPy, or roll-your-own.

---

## 3. Mental Models & Analogies

### 3.1 The "Encyclopedic Autocomplete" Model

An LLM is **radically-scaled autocomplete**. Given a prefix of tokens, it predicts a distribution over the next token — and iteratively samples to continue.

The reason this is useful for far more than autocomplete: to predict "The capital of France is ___" you must know that Paris is the capital of France. To predict "def fibonacci(n):\n    if ___" you must know Python and the Fibonacci recurrence. To predict "Q: What's 12 × 15? A: ___" you must know multiplication.

Predicting the next token, at web scale, is a task that requires modeling **the whole world implicit in text**. Language modeling isn't a task; it's every task, filtered through the medium of language.

This mental model demystifies both LLM capabilities and LLM failures:
- **What it knows** — everything present in its training data.
- **What it can't reliably do** — anything the training data doesn't hint at, or anything requiring genuinely new inference at scales the training didn't cover.
- **Why prompts work** — you're steering the completion by writing a prefix that only makes sense as the beginning of the response you want.
- **Why it hallucinates** — "make it look like a plausible completion" doesn't distinguish "true" from "sounds true." The training objective doesn't reward calibration on facts.

### 3.2 The "Chef Trained on Every Cookbook" Model

Pretraining is like a chef reading every cookbook ever written, in every language. They emerge from this training with **broad general knowledge**: they know French cuisine, Japanese cuisine, molecular gastronomy, home cooking; they know what butter is; they know baking chemistry.

But they've read cookbooks, not worked in a kitchen. They can produce recipes that read beautifully but don't test — like the classic hallucinated shrimp scampi that includes ingredients that don't exist. They don't know which recipes their bosses will consider "good" or "bad."

**Instruction fine-tuning** = the chef does a residency where they cook thousands of dishes under a mentor's supervision, learning: "when I ask you to make dinner, here's the shape of a good response."

**RLHF/DPO** = the chef then works in a real kitchen where customers give feedback. Over time they learn what people actually like — not just what's technically correct.

**Base model → SFT → RLHF/DPO** = cookbook autodidact → residency → seasoned professional. Each stage adds a different capability. Skip any stage and you get a subtly worse assistant.

![IMG-LLM-01](/3%20—%20Deep%20Learning/images/IMG-LLM-01.jpg)
> **Caption:** LLM lifecycle — pretraining dominates compute; alignment stages are relatively cheap.
> **Placement:** Section 2.3.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Prompt Engineering Solves Everything"

Prompt engineering is real and useful, but it has a ceiling: **it can only surface capabilities the base model already has**. If the model was never trained on your domain, no clever prompt teaches it that domain — it will hallucinate confidently. When prompts stop unlocking behavior you need, the options are (a) provide the missing info via RAG, (b) fine-tune, (c) upgrade to a stronger base model.

### 4.2 "Hallucination Is a Bug That Can Be Fixed"

Hallucination is a **feature of the training objective**, not a bug. Next-token prediction rewards making the output *look plausible*, not being *true*. The model has no concept of "I don't know." Reducing hallucination requires:

- **Retrieval augmentation** — ground responses in cited sources.
- **Fact-checking layers** — post-generation verification.
- **Fine-tuning on refusal / calibration** — teach the model to say "I don't know."
- **Better base models** with stronger world models.

Even frontier models still hallucinate, sometimes with high confidence. Design your application to detect and mitigate this.

### 4.3 "Bigger Is Better, Full Stop"

Bigger models are usually better on average — but for a given task, cost, and latency budget, a well-tuned smaller model often beats a big general one. Consider:

- A 7B fine-tuned on your domain often beats a 70B general model on your task.
- Distilled versions of frontier models are 10× cheaper with 95% of the quality.
- On-device / edge inference needs small models (1B–8B).

The right model = the smallest one that clears your quality bar.

---

## 5. Self-Assessment Bank (LLMs)

### Questions

**Q1 (Short answer).** What is a token in an LLM, and what is BPE?

**Q2 (Multiple choice).** The pretraining objective of a decoder-only LLM is:
A. Reconstruct masked tokens (MLM).
B. Predict the next token given the previous tokens (causal LM).
C. Classify sentiment.
D. Contrast positive and negative pairs.

**Q3 (Short answer).** Describe the three-stage post-training pipeline (SFT → RLHF or DPO) in one paragraph.

**Q4 (Multiple choice).** LoRA fine-tuning:
A. Retrains all parameters at 4-bit precision.
B. Freezes the base model and trains low-rank updates $\Delta W = BA$ with $r \ll d$.
C. Doubles model size.
D. Only works for encoder-only models.

**Q5 (Short answer).** Explain the KV cache. Why is it essential for efficient LLM inference?

**Q6 (Multiple choice).** Temperature sampling with $T \to 0$ becomes:
A. Random sampling from the uniform distribution.
B. Greedy decoding (argmax).
C. Beam search.
D. Nucleus sampling.

**Q7 (Short answer).** What is "in-context learning" and how does it relate to model scale?

**Q8 (Multiple choice).** Retrieval-Augmented Generation (RAG) mostly helps with:
A. Making the model faster.
B. Grounding responses in external knowledge (domain data, latest info) without fine-tuning.
C. Reducing hallucination on tasks the model has never seen.
D. Both B and C.

**Q9 (Short answer).** Why can an LLM struggle to count letters or reverse strings, despite excellent general performance?

**Q10 (Multiple choice).** DPO (Direct Preference Optimization) replaces:
A. Supervised fine-tuning entirely.
B. The RL step of RLHF — training against preferences without an explicit reward model.
C. Tokenization.
D. Attention.

---

### Answer Key & Detailed Explanations

**A1.** A **token** is an integer ID from the model's vocabulary — usually a subword piece, not a full word. **BPE (Byte-Pair Encoding)** builds this vocabulary by starting with characters and repeatedly merging the most frequent adjacent pair into a new token, until reaching a target vocabulary size (30k–200k). Result: common words are one token; rare words split into meaningful pieces.

**A2. B.** Decoder-only LLMs are trained with **causal language modeling** — predict the next token given all previous tokens. MLM (A) is the BERT-style encoder objective. C and D are downstream tasks or different training paradigms.

**A3.** **Stage 1: SFT (Supervised Fine-Tuning)** — train the pretrained model on curated (instruction, response) pairs to learn the "helpful assistant" pattern. **Stage 2: reward modeling** (RLHF) or preference collection (DPO) — collect human preferences between response pairs. **Stage 3: alignment** — RLHF uses PPO to fine-tune against the reward model with a KL penalty to stay near SFT; DPO directly optimizes the same preference objective without RL or a separate reward model. The pipeline turns a "text completer" into an aligned assistant.

**A4. B.** LoRA freezes the pretrained $W$ and learns a low-rank update $\Delta W = BA$ where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, $r \ll \min(d, k)$. Only $A, B$ are trained, using ~1000× fewer trainable parameters than full fine-tuning. QLoRA adds 4-bit quantization of the base model on top.

**A5.** During autoregressive generation, each new token requires an attention computation over all previous tokens. Without caching, computing K and V for every previous token at every new step is $O(n^2)$ per token and $O(n^3)$ for a full generation. The **KV cache** stores K, V for all previous tokens; each new step only computes K, V for the current position and appends to the cache. Per-token cost drops from $O(n^2)$ to $O(n)$; full generation from $O(n^3)$ to $O(n^2)$. Essential — without it, real-time LLM inference is impractical.

**A6. B.** Temperature divides logits before softmax. As $T \to 0$, the softmax becomes a delta on the argmax — you always sample the most likely token. That's greedy decoding.

**A7.** **In-context learning** — the LLM can perform a task by seeing a few examples in its prompt, without any gradient updates or fine-tuning. Give it "1 → apple\n2 → banana\n3 →" and it learns the pattern from context. Discovered in GPT-3 (Brown et al., 2020). It's an **emergent ability** — smaller models can't do this; ability scales with model size. It relies on the transformer's ability to attend to and generalize from prompt examples.

**A8. D.** RAG grounds responses in external knowledge — retrieved chunks from a vector database — which both (a) lets the model use up-to-date or private information it wasn't trained on and (b) reduces hallucination by anchoring generations to verified sources. It doesn't make the model faster — actually adds retrieval latency.

**A9.** LLMs don't see characters — they see tokens. "Strawberry" might be a single token; asking "how many R's are in strawberry" requires the model to spell it out mentally. Since the training loss doesn't strongly reward character-level accuracy, and the model can't easily "see" characters, tasks that require character-level reasoning (counting, reversing) are surprisingly hard. This is one of the clearest examples of the tokenization-abstraction gap.

**A10. B.** DPO reformulates the RLHF preference-learning objective so it can be optimized directly on preference pairs without needing a separately-trained reward model or RL loop. It's simpler, more stable, and computationally cheaper than PPO-based RLHF, while achieving similar or better alignment quality. Now widely used in open-source instruction models (Zephyr, Tulu, LLaMA-3 Instruct).

---

## 6. Practice Prompts

1. **Tokenizer exploration.** Install `tiktoken`. Encode various strings and observe tokenization. Encode "Strawberry" — how many tokens? How many "R"s does it contain from the model's view?
2. **Prompting patterns.** Try zero-shot, few-shot, and chain-of-thought prompting on a math or reasoning task with a HuggingFace model. Compare quality.
3. **LoRA fine-tuning.** Use `peft` + `transformers` to LoRA-fine-tune a small model (e.g., TinyLlama or Mistral-7B) on a domain-specific dataset (e.g., legal text). Compare to base.
4. **RAG minimal.** Build a minimal RAG pipeline: embed a small corpus (via `sentence-transformers`), store in an in-memory FAISS index, retrieve top-5 for a query, feed to an LLM.
5. **Inference speedup.** Serve the same model via `vllm` vs `transformers.pipeline`. Compare tokens/sec for concurrent requests.

---

## 7. References

- Radford et al., ["Improving Language Understanding by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018) — GPT-1.
- Brown et al., ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (2020) — GPT-3.
- Ouyang et al., ["Training language models to follow instructions with human feedback"](https://arxiv.org/abs/2203.02155) (2022) — InstructGPT / RLHF.
- Rafailov et al., ["Direct Preference Optimization"](https://arxiv.org/abs/2305.18290) (2023) — DPO.
- Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (2021).
- Dettmers et al., ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314) (2023).
- Lewis et al., ["Retrieval-Augmented Generation"](https://arxiv.org/abs/2005.11401) (2020).
- Kwon et al., ["Efficient Memory Management for LLM Serving with PagedAttention"](https://arxiv.org/abs/2309.06180) (2023) — vLLM.
- Andrej Karpathy, ["Intro to Large Language Models"](https://www.youtube.com/watch?v=zjkBMFhNj_g) (2023).
