# Cost & Latency Optimization — Master Study Guide

> **Track:** LLM Engineering · **Module:** 10
> **Prerequisites:** Modules 01–09.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** In production, cost and latency define whether an LLM feature ships or dies. A brilliant assistant that costs $2 per query cannot become a chatbot for a million users. A great answer that takes 15 seconds to arrive cannot ship in a keystroke-fast UI. Every serious LLM team spends more time optimizing these than tuning prompts.

The good news: **most systems have 5×–20× cost/latency wiggle room** through techniques the model provider already offers. If you leave this on the table, you're paying for indulgent engineering.

**Fundamental principles you must own:**

1. **Measure before optimizing.** You cannot optimize what you don't measure. Instrument every call.
2. **The biggest wins are architectural** — cache, batch, right-size the model — not "shave 3ms off tokenization."
3. **Latency has two components**: time-to-first-token (TTFT) and inter-token latency (ITL). Different bottlenecks; different fixes.
4. **Cost has three components**: input tokens, output tokens, and infra overhead (retries, orchestration).
5. **The right model for the job is usually smaller than you think.** Model choice is a bigger lever than any micro-optimization.
6. **Prompt caching alone can 5–10× cost/latency wins.** It's the highest-leverage single technique.

If you retain nothing else: **measure → cache → right-size → batch → stream. In that order.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Cost/Latency Anatomy of a Request

**Cost breakdown of an LLM API call:**

$$\text{cost} = P_{\text{in}} \cdot n_{\text{in}} + P_{\text{out}} \cdot n_{\text{out}} + P_{\text{cached}} \cdot n_{\text{cached}}$$

Where:
- **$P_{\text{in}}$** = price per input token.
- **$P_{\text{out}}$** = price per output token (typically 2–5× input).
- **$P_{\text{cached}}$** = discounted price for cached tokens (0.10–0.50× input).
- **$n_x$** = token counts.

Output tokens are more expensive because each requires a full forward pass; input tokens are prefilled in parallel.

**Latency breakdown:**

$$\text{latency} = \text{TTFT} + n_{\text{out}} \cdot \text{ITL}$$

Where:
- **TTFT (time to first token)** — dominated by prefill (input length matters).
- **ITL (inter-token latency)** — dominated by decode step (varies by model size; typical 20–100 ms/token).

**Key numbers to know (order of magnitude):**

- Small model (GPT-4o mini, Claude Haiku, Gemini Flash): ~50 tokens/sec output; input ~$0.10–0.30/M.
- Mid model (GPT-4o, Claude Sonnet, Gemini Pro): ~40 tokens/sec; input ~$3/M.
- Frontier (GPT-4/5, Claude Opus, Gemini Pro Ultra): ~30 tokens/sec; input ~$10–15/M.
- Cached tokens: 0.10× normal price.
- Batch API tokens: 0.50× normal price with 24-hour latency.

Numbers move quarterly — always check current provider pricing pages.

**Where the money goes on a chatbot:**

For a conversational assistant with a long system prompt:
- Per turn: system prompt (5000 tokens) + history (3000 tokens) + user (100 tokens) + output (300 tokens).
- Without caching: 8100 tokens in + 300 tokens out.
- With caching: 100 tokens uncached in + 8000 cached in + 300 out.
- **Cost reduction: ~5–8×** just from caching.

---

### 2.2 Caching in Its Many Forms

**Prompt caching** (see Module 09.4) — provider caches the KV state of a prefix.

**Semantic caching** — cache **responses** keyed by embedding similarity. If a new query is semantically close to a cached one, return the cached response.

```python
import hashlib
from redis import Redis
from sentence_transformers import SentenceTransformer
import numpy as np

emb = SentenceTransformer("all-MiniLM-L6-v2")
r = Redis()

def get_or_compute(query, compute_fn, threshold=0.95):
    q_vec = emb.encode(query, normalize_embeddings=True)
    # look up nearest cached embedding
    hits = search_vector_index(q_vec, top_k=1)
    if hits and hits[0].score >= threshold:
        return hits[0].cached_response
    response = compute_fn(query)
    store(q_vec, response)
    return response
```

Best for FAQ-style workloads with heavy repetition.

**Exact caching** — for byte-identical queries, return the exact stored response. Fastest, but only helps for exact repeats.

**Feature/embedding caching** — embeddings are deterministic given a model. Cache them. Never re-embed the same string.

**Tool-result caching** — a tool call that hits an external API is a candidate for TTL-based caching. E.g., cache "weather in Helena" for 15 minutes.

**Warning: cache invalidation.** The old joke ("There are only two hard things in computer science: cache invalidation and naming things") is exactly right here. Design cache keys carefully; version them when models or prompts change.

---

### 2.3 Right-Sizing the Model

**Smaller models are 10×–100× cheaper and faster** than frontier models. For many tasks — classification, summarization of short text, routing, entity extraction, simple Q&A — a small model is enough.

**Model tier heuristic:**

| Task | Try first | Why |
|------|-----------|-----|
| Simple classification / routing | Small (Haiku, GPT-4o-mini, Gemini Flash) | 90%+ tasks solvable |
| Extraction to JSON | Small with structured output | Format compliance is the hard part |
| Chat with short answers | Small; upgrade if quality bar not met | Cheap enough to try both |
| Long-form generation, code, reasoning | Medium (Sonnet, GPT-4o, Gemini Pro) | Sweet spot |
| Deep reasoning, agentic multi-step | Frontier (Opus, GPT-5, Gemini Pro Ultra) | Worth the cost when it matters |

**Cascading / model routing** — a common pattern:

1. Try the small model.
2. If it fails a confidence check (or a validator), retry with a larger model.
3. Log both.

Empirically, this handles 70–90% of traffic on the cheap model and only escalates the hard cases.

**Speculative decoding** (provider-side) — a small "draft" model proposes several tokens; the large model verifies them in parallel. Same output distribution, 2–3× faster. Enabled by default on some inference stacks.

**Fine-tuning a small model** — for high-volume, narrow tasks, LoRA-fine-tuning a small open model on your data often beats calling a big frontier model, both cheaper and faster.

---

### 2.4 Batching, Streaming, and Parallelism

**Batching** — send multiple independent requests together:

- **Async concurrency** — use `asyncio.gather` to fire N requests in parallel. Great for embarrassingly parallel workloads (embeddings, classification of many rows).
- **Batch API** — OpenAI, Anthropic offer 50% discounts for asynchronous batch endpoints with 24-hour SLA. Perfect for offline processing.
- **Server-side dynamic batching** — vLLM, TensorRT-LLM stack incoming requests into a shared decode batch. Massively improves GPU utilization.

**Concurrency for latency-critical work.** If you're building an agent that calls three tools in parallel, don't serialize them:

```python
import asyncio

async def run_all(query):
    tasks = [call_tool_a(query), call_tool_b(query), call_tool_c(query)]
    results = await asyncio.gather(*tasks)
    return synthesize(results)
```

Latency = max(individual) instead of sum.

**Streaming.** See Module 07. Streaming doesn't reduce total time but dramatically improves perceived latency — the user sees output within ~500ms instead of waiting 5 seconds.

**Speculative execution.** Fire multiple candidate strategies in parallel; return the first that succeeds. Wasteful in compute; huge in latency. Use judiciously.

---

### 2.5 Prompt Engineering for Cost

**Trim the prompt.** Every unnecessary token costs money forever. Common redundancies:
- Verbose politeness ("Please, if you would be so kind...").
- Repetitive instructions.
- Overly detailed examples.
- Long chain-of-thought preambles when the task is simple.

**Concise instructions beat verbose ones.** Empirically, well-written short prompts often outperform bloated long ones — less noise for the model to filter.

**Move examples into fine-tuning.** If you have 5+ few-shot examples repeated in every prompt, that's several thousand tokens of tax. Fine-tune the model on those examples and drop them from the prompt.

**Structured extraction is cheaper than free-form.** A model asked for JSON with 5 keys produces 20–50 tokens; the same info as prose is 100–200. Design outputs to be terse.

**Stop sequences.** Use `stop` parameter to end generation early when the model reaches a known marker. Saves output tokens.

**Max tokens.** Always set `max_tokens` to your realistic worst case, not the model's default. Prevents runaway generations.

**Chain-of-thought only when it helps.** CoT increases output tokens 3×–10×. Only use it for tasks where CoT measurably improves quality (multi-step reasoning, math). For classification or extraction, skip.

**Cost accounting.** Log every request's input/output tokens and cost. Aggregate weekly by user, feature, and model. You cannot manage what you don't measure.

---

## 3. Mental Models & Analogies

### 3.1 The "Utility Bill" Model

LLM cost is your utility bill. Every extra token is a light left on:
- The system prompt is the base charge — you pay for it on every request.
- User queries are metered usage — variable but small.
- Long responses are peak-demand pricing — expensive per second.
- Prompt caching is the solar-panel install — high setup thinking, then massive month-over-month savings.
- Model selection is the appliance you chose — using a frontier model for a lightbulb task is running an industrial oven to make toast.

Read the bill (log tokens). Find the top consumers. Optimize the biggest ones first (Pareto: usually 10 endpoints eat 90% of the cost).

### 3.2 The "Response Time Budget" Model

Imagine a latency budget as a coin purse of 2000ms for a chatbot response to *feel* real-time:

- Network round-trip: ~50ms.
- Auth + preprocessing: ~50ms.
- Retrieval (if RAG): ~100–300ms.
- LLM TTFT: 500–1500ms — the biggest chunk.
- Post-processing + streaming to UI: ~50ms.

**Total realistic budget: ~2s to first token.** Everything after can stream.

Every technique in this module is essentially "spend fewer coins" or "get change back":
- Smaller model → TTFT drops.
- Prompt caching → TTFT drops.
- RAG (smaller context) → TTFT drops.
- Streaming → perceived latency drops even if total doesn't.

Design each request as: "here's my budget; how do I stay under it while meeting the quality bar?"

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "We'll Optimize Later — Ship First"

By the time "later" arrives, you have 100 endpoints and no telemetry. Instrument on day 1: log `model`, `input_tokens`, `output_tokens`, `latency_ms`, `cache_hit`, `endpoint`, `user_id`. It's ~10 lines of code and it prevents a $100k surprise.

### 4.2 "Frontier Model Always Wins on Quality"

Not on the task you're doing. A small model with prompt engineering, structured outputs, and good retrieval routinely beats a frontier model with a sloppy prompt. Always benchmark small models on your actual task before committing to a frontier one — you're often paying 10× for a rounding error in quality.

### 4.3 "Streaming Doesn't Help — Total Latency Is the Same"

True total latency, but perceived latency drops 5×–10×. Users abandon in ~3 seconds of dead time; they don't abandon a stream that starts in 300ms. This UX difference is worth prioritizing.

---

## 5. Self-Assessment Bank (Cost/Latency)

### Questions

**Q1 (Short answer).** Define TTFT and ITL. Which is affected more by input length?

**Q2 (Multiple choice).** Which technique typically gives the biggest cost reduction on a chatbot with a large system prompt?
- (a) Streaming.
- (b) Prompt caching.
- (c) Smaller max_tokens.
- (d) Larger batch size.

**Q3 (Short answer).** Why are output tokens typically more expensive than input tokens?

**Q4 (Multiple choice).** For a bulk offline embedding of a 1M-row dataset, the most cost-effective approach is usually:
- (a) Call the OpenAI real-time endpoint one by one.
- (b) Use the async Batch API (50% discount, 24hr SLA).
- (c) Fine-tune a model first.
- (d) Send them one per second to avoid rate limits.

**Q5 (Short answer).** Describe semantic caching and when it makes sense.

**Q6 (Multiple choice).** A cascading (model router) approach:
- (a) Always uses the frontier model.
- (b) Tries a small model first; escalates to a larger one only on failure/low confidence.
- (c) Runs all models in parallel; returns the fastest.
- (d) Randomly selects a model.

**Q7 (Short answer).** Why does streaming reduce *perceived* latency but not total latency?

**Q8 (Multiple choice).** Which of these is TRUE about prompt caching?
- (a) It works even if the cached prefix changes each request.
- (b) The prefix must be byte-identical across requests to hit the cache.
- (c) It's automatic on all providers.
- (d) It requires you to store the tokens locally.

**Q9 (Short answer).** Give three concrete techniques to reduce output tokens for a summarization task.

**Q10 (Multiple choice).** Chain-of-thought (CoT) prompting affects cost:
- (a) No effect.
- (b) Increases output tokens (and cost) 3–10× — use only when it improves quality.
- (c) Reduces total tokens because the model can be shorter.
- (d) Only reduces latency.

---

### Answer Key & Detailed Explanations

**A1.** **TTFT (Time To First Token)** — latency from sending the request until the first output token arrives. Dominated by **prefill** — processing the entire input in parallel. Grows with input length. **ITL (Inter-Token Latency)** — time between successive output tokens. Roughly constant per model regardless of input length; determined by the decode step's cost per token.

**A2. (b).** Prompt caching often gives 5–10× cost reduction on repeat-prefix workloads. Streaming doesn't reduce cost. Smaller `max_tokens` helps modestly. Batching is for offline workloads.

**A3.** Output tokens require a full autoregressive decode step each — the model runs one forward pass per output token, one at a time (autoregressively). Input tokens are processed in a single **parallel prefill**, then discarded. Provider pricing reflects the compute cost: output typically 2–5× input price.

**A4. (b).** OpenAI/Anthropic Batch APIs charge ~50% of normal pricing for jobs with a 24-hour SLA. Perfect for bulk offline processing. Real-time calls one by one would cost 2× and hit rate limits.

**A5.** Semantic caching stores response results keyed by the **embedding** of the query. When a new query arrives, embed it; if a stored embedding is very similar (cosine > 0.95 or so), return the cached response instead of calling the LLM. Best for FAQ-style workloads with high query redundancy (e.g., support chatbots where "how do I reset my password" is asked 500 times a day).

**A6. (b).** Cascade / router: run a small cheap model first; if it fails (e.g., low confidence, invalid output, or a validator rejects) escalate to a larger model. Handles the bulk of traffic cheaply and escalates only when needed. Common cost reduction: 5×–20×.

**A7.** Streaming outputs each token as it's generated instead of waiting for the whole response. Total generation time is unchanged, but the **user sees output starting within a second** rather than staring at a spinner for 5 seconds. Perceived latency crashes; total wall-clock time is the same.

**A8. (b).** The cached prefix must be byte-identical across requests — any difference (whitespace, timestamps, dynamic content) invalidates the cache. Most providers require explicit opt-in and mark the cacheable prefix with a flag. Local storage is provider-side.

**A9.** (1) Ask for **structured output** (JSON with specific keys) — usually 50–70% fewer tokens than prose. (2) Set an aggressive `max_tokens` to cap generation. (3) Use `stop` sequences at natural end markers. (4) Instruct explicit brevity ("Answer in 2 sentences"). (5) Skip chain-of-thought if not needed. Any three.

**A10. (b).** CoT reasoning writes out intermediate thinking as tokens, tripling to ten-times the output token count. It improves quality on tasks requiring reasoning; it's wasted on classification/extraction/simple tasks. Rule: measure before enabling.

---

## 6. Practice Prompts

1. **Cost dashboard.** Write a decorator that wraps every LLM call, logs `(model, input_tokens, output_tokens, cost, latency_ms, cache_hit)` to a database. Build a small dashboard aggregating by endpoint and model.
2. **Model cascade.** Implement a router that (a) tries `gpt-4o-mini`, (b) validates the output with a schema check, (c) falls back to `gpt-4o` on failure. Measure quality vs cost tradeoff.
3. **Prompt caching test.** Reproduce module 09 caching benchmark: make 3 identical requests, measure cost and TTFT with and without caching. Confirm ~5× reduction.
4. **Streaming vs blocking.** Build the same endpoint as blocking and streaming. Instrument both. Measure user-visible latency (time until first token displayed) vs total time.
5. **Batch discount.** Take a 10k-item classification dataset. Run it (a) via real-time API in parallel, (b) via batch API. Compare cost and wall-clock time.

---

## 7. References

- OpenAI docs: [Batch API](https://platform.openai.com/docs/guides/batch), [Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching).
- Anthropic docs: [Message Batches API](https://docs.claude.com/en/docs/build-with-claude/message-batches), [Prompt Caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching).
- vLLM: [Continuous batching](https://github.com/vllm-project/vllm).
- Chen et al., ["FrugalGPT"](https://arxiv.org/abs/2305.05176) (2023) — LLM cascading.
- Leviathan et al., ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) (2023).
