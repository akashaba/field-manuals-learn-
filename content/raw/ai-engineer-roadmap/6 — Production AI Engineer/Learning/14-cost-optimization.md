# 14 — Cost Optimization

> **Module goal:** Turn a hemorrhaging LLM bill into a controlled operating expense. Master the levers — model selection, prompt caching, semantic caching, output-length control, batching, off-peak scheduling, model routing, self-hosting economics — and the FinOps practices that put a per-request dollar cost on your Grafana dashboard so cost becomes a **first-class SLI**.

---

## 1. Executive Summary & Core Concepts

Every LLM request has a cost:

$$
\text{cost} = (T_{\text{in}} - T_{\text{cache\_read}}) \cdot p_{\text{in}} + T_{\text{cache\_read}} \cdot p_{\text{cache}} + T_{\text{out}} \cdot p_{\text{out}}
$$

where $T$ are token counts and $p$ are per-million-token prices. For a typical Anthropic model call, output tokens cost roughly 5× input tokens; cached input tokens cost roughly 0.1× input tokens. **The output side dominates unless you have a very long prompt.**

Cost is a function of four things you control:

1. **How often the LLM is called** — cache, retry hygiene, agent step limits
2. **Which model handles it** — model routing (Haiku for classification, Opus for reasoning)
3. **How much context is sent** — prompt engineering discipline, prompt caching, retrieval precision
4. **How much output is generated** — `max_tokens`, structured outputs, "no unnecessary preamble" instructions

Plus one structural lever: **when** the compute happens — batch API, off-peak, self-hosted with reserved GPUs.

**FinOps discipline** (Cloud FinOps Foundation):
- **Inform** — know your costs by dimension (model, route, user, tenant, feature)
- **Optimize** — apply the levers below
- **Operate** — make cost a monitored, alerted, budgeted metric like latency

**Order-of-magnitude example.** A chatbot serving 100k users:
- **Naïve**: 1 call per interaction, 8k input + 1k output, Opus everywhere → ~ $0.09/interaction × 5 interactions/day × 100k users × 30 days ≈ **$1.35M/month**
- **Optimized**: model routing (80% Haiku, 20% Sonnet, 0% Opus except escalation), prompt caching (85% hit rate on 6k-token system prompt), max_tokens=512 unless needed, semantic cache 30% hit rate → ~ $0.006/interaction ≈ **$90k/month**
- **A 15× reduction with no quality loss** is realistic when applied comprehensively.

---

## 2. Deep-Dive Breakdown

### 2.1 Model Routing (The Highest-Impact Lever)

Not every request needs the flagship model. Route by:

- **Intent classification** — a cheap classifier (Haiku, or a small BERT) decides "simple FAQ" vs "complex reasoning" vs "code gen" and dispatches.
- **Explicit user tier** — free users get Sonnet, pro users get Opus.
- **Confidence-based escalation** — Haiku answers; if it flags low confidence or user thumbs-down, retry on Sonnet.
- **Length heuristic** — short factual questions to a small model; multi-turn code review to a large one.

```python
async def route_and_answer(request: str) -> str:
    intent = await classify(request, model="claude-haiku-4-5")  # cheap
    if intent == "faq":
        return await run(request, "claude-haiku-4-5")
    if intent == "reasoning":
        return await run(request, "claude-sonnet-5")
    if intent == "hard":
        return await run(request, "claude-opus-5")
```

Instrument every route with per-model metrics; watch for accidental fallthrough where a "simple" intent starts eating Opus tokens.

### 2.2 Prompt Caching Revisited (Specific to Cost)

Module 11 covered how prompt caching works. Cost impact:

- Anthropic pricing: cache-read tokens cost ~10% of normal input tokens (specific ratio varies by model; check the pricing page).
- Break-even: **2–3 uses of a cached prefix**. Anything > 1024 tokens reused across requests should be cached.
- What to cache: the system prompt, few-shot examples, tool definitions, large reference documents, long user history summaries.
- What NOT to cache: per-request content that varies (the user's current message), transient state.

The pattern in prompt design: **put stable content first, dynamic content last**. The cache matches prefixes byte-for-byte from the start; changing anything early breaks the cache. Order:
1. Big system instructions (rarely change)
2. Tool definitions (change per-deploy)
3. Retrieved context (per-user, but may be reusable within a session)
4. Chat history (grows every turn)
5. Current user message (always changes)

Set `cache_control` breakpoints between stable sections to allow partial cache hits.

### 2.3 Output Cost Discipline

Output tokens are the expensive dimension. Techniques:

- **Set `max_tokens` explicitly.** Default may be very high; users writing "give me a 3-word answer" don't need 4096 tokens allocated.
- **Structured output caps generation.** JSON schema with fixed field lengths yields shorter, predictable outputs.
- **Instruct brevity.** "Answer in one sentence" or "no preamble, no closing pleasantries" in the system prompt reduces filler tokens substantially.
- **Stop sequences.** Terminate when the model outputs a boundary token; useful for repeated-generation loops.
- **Sampling temperature 0** for deterministic tasks (classification, extraction) — no need for repeated sampling.

Measuring where output goes: log `output_tokens` per route; identify the p99 routes and investigate. Often one route with a "generate detailed explanation" instruction is 80% of your output cost.

### 2.4 Batch API — Async at Half the Price

For non-realtime workloads, most providers offer a **batch API**: submit jobs, receive results within 24 hours, pay ~50%. Perfect for:
- Nightly evaluation runs
- Backfilling embeddings
- Bulk classification (labeling old records)
- Non-urgent report generation

Anthropic's Message Batches API accepts up to 100k messages per batch; returns a results URL when complete.

```python
batch = await client.messages.batches.create(
    requests=[
        {"custom_id": f"req-{i}", "params": {...}}
        for i in range(50_000)
    ]
)
# Poll batch.id until completed; download results
```

Rule of thumb: **if your workload doesn't need a response in the current HTTP request, batch it.**

### 2.5 Semantic Cache & Deduplication

Module 11 covered semantic caching. Cost impact per hit: ~$0.001 in embedding + Redis ops vs $0.01–$1 for the skipped LLM call. Achievable hit rates on well-defined workloads (FAQ, product search, customer support) are 30–60%; on open-ended chat, 5–15%.

**Deduplication for streaming input**: if the same user sends the same question twice in 10 seconds (accidental double-submit), respond from cache without re-invoking. Even a 1% dedup rate at scale saves noticeably.

### 2.6 Retrieval Precision

RAG systems retrieve top-K chunks; each chunk consumes prompt tokens. Cost levers:
- **Reduce K** — 5 relevant chunks beats 20 mediocre ones on both quality and cost.
- **Rerank** — a lightweight reranker (Cohere Rerank, cross-encoder) turns 50 candidates into the 5 best; smaller context, better answer.
- **Compress context** — LLMs like LangChain's `ContextualCompressor` extract only relevant sentences from retrieved chunks.
- **Chunk size tuning** — smaller chunks = more precise, less waste per chunk; but too small loses context. Empirically 300–800 tokens per chunk works.
- **Metadata filtering** — filter by tenant/category/date *before* semantic search; smaller candidate set to rerank.

### 2.7 Self-Hosting Economics

At sufficient scale, self-hosting open-weight models (Llama, Mistral, Qwen, DeepSeek) beats managed APIs. The math:
- **Managed API**: pay per token, scales linearly with usage, zero infra work.
- **Self-hosted**: pay per GPU-hour whether utilized or not. Break-even at ~40–70% sustained utilization of the largest model tier.

Levers to make self-hosted economics work:
- **Batch multiple requests per forward pass** (dynamic batching via vLLM, TGI, SGLang) — sub-linear cost per request as batch size grows.
- **Quantization** (INT8, INT4, GPTQ, AWQ) — 2–4× smaller model → runs on cheaper GPUs.
- **KV-cache reuse** across sessions — vLLM's prefix caching, similar to Anthropic's prompt cache.
- **Speculative decoding** — small "draft" model proposes tokens, big model verifies; 2–3× speedup.
- **Right-size the model** — a fine-tuned 7B for classification beats an untuned 70B on both quality and cost.
- **Spot/preemptible GPUs** — 60–90% cheaper if your workload can tolerate interruption (batch, off-peak).

**Reality check**: if you're processing < 500B tokens/month, managed API is almost always cheaper (once you count infra, ops, on-call, the wrong-gpu-purchased mistake). At > 1T tokens/month, self-hosting starts winning; requires an ML platform team.

### 2.8 The Cost Dashboard (Making It a First-Class SLI)

Build these panels (Module 7 introduced the metrics):

- **$ per hour / per day** — moving average, plus 30-day trend
- **$ per user (top 10 spenders)** — flag outliers, spam accounts, or one power user underpricing your tier
- **$ per route** — where the money goes
- **$ per model** — is model routing working?
- **Cache hit rate & $ saved by cache** — direct ROI on your cache work
- **Cost anomaly alert** — when 15-min cost rate is 2× the trailing 24-hour baseline, page

Bind budget alerts at the cloud level too (AWS Budgets, GCP Budgets) as a backstop against per-metric misconfig.

`[IMG-14-01]` — *Prompt: A cost-optimization stack for LLM applications. Vertical waterfall diagram, top labeled "Naïve LLM spend $1.35M/month", each descending step labeled with a lever and % savings: "Model routing (-70%)", "Prompt caching (-50%)", "Semantic cache (-20%)", "max_tokens + brevity prompt (-25%)", "Batch API for offline (-10%)", down to a final small bar labeled "$90k/month". Green descending steps against neutral background, clean financial dashboard style, clearly labeled percentages that don't need to compound arithmetically since each stacks on the remainder.*

---

## 3. Mental Models & Analogies

### Model 1: The Utility Bill

Your household electric bill has fixed charges (delivery, meter fee — like base infra), variable per-kWh cost (like tokens), and time-of-use rates (peak vs off-peak — like the batch API discount). You can:
- Switch to more efficient appliances (a smaller model that does the job)
- Insulate the house (prompt caching to avoid heat loss)
- Run the dishwasher at night (batch API for non-urgent work)
- Turn off lights (max_tokens = don't generate what you won't read)
- Solar panels if you use enough (self-host at scale)
- Check the bill monthly and investigate anomalies (cost dashboards + alerts)

Cost engineering is not one heroic optimization; it's a portfolio of small habits that compound. And ignoring the bill entirely until it's shocking is what "surprise $47k Anthropic bill" incidents look like.

### Model 2: The Rush-Hour Cab vs the Airport Shuttle

You need to go from A to B. A rush-hour taxi (managed synchronous API, biggest model) gets you there fastest, at highest cost. A ride-share (managed API, appropriate model) is cheaper if you accept some routing. The airport shuttle (batch API) is much cheaper if you can wait until it fills. Public transit (self-hosted) is cheapest per rider but requires you to run the transit system. Walking (cache hit — you didn't need to travel) is free but only works for the trip you already made.

Match the transport to the trip. Emergency room = flagship model + synchronous. Grocery run = mid-tier model. Vacation planning that can wait = batch. The bug is when every trip uses the emergency ride.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Using the flagship model everywhere "for quality."**
Opus-class models are extraordinary at long-form reasoning and code review; they're 5–10× more expensive than Haiku-class and provide *no measurable quality improvement* on tasks like classification, extraction, and short factual Q&A. Teams often anchor on the biggest model because it was easiest to prototype with, then leave it in production. **Fix:** build an offline eval that scores your task on Haiku, Sonnet, Opus. Often the smallest model that clears the quality bar is 2 tiers below what's in production. Model routing (2.1) formalizes this — a cheap classifier picks the model per request.

**Pitfall #2 — Cost invisible until end-of-month.**
Your finance team forwards you an invoice: "Anthropic $47,000 this month, was $8,000 last month." You have no telemetry for what caused the jump — a new feature? A retry loop bug? A bad prompt version? A single user? By the time you know, it's already spent. **Fix:** attach `$` as a first-class dimension on every LLM span (Module 9), roll up per-user, per-route, per-model in real time, and set alerts on 15-min cost-rate anomalies. Cost visibility should be as sharp as latency visibility.

**Pitfall #3 — Optimizing tokens while ignoring compute-hours (or vice versa).**
Two failure modes:
- **API-only team**: optimizes prompts, ignores that a self-hosted 8B model would replace 90% of their calls at 1/20 cost. They're penny-wise (fewer tokens) but pound-foolish (no self-host strategy).
- **Self-host-only team**: buys 8× H100s, uses them at 5% average utilization, spends more on idle GPUs than a Sonnet API bill would have been. They optimize $/token in-batch but ignore $/token-in-real-world.
**Fix:** track **total cost of ownership**: API $ + infra $ + ops $ + engineering $, divided by tokens actually served. Recompute quarterly. The right architecture at 1M tokens/day is different from 100M tokens/day is different from 10B tokens/day.

---

## 5. Self-Assessment Bank

**Q1 (MC):** Which cost lever typically has the highest impact on a chatbot serving a mix of simple and complex queries?
A) Prompt caching
B) Model routing (small model for simple, large for complex)
C) Reducing max_tokens
D) Self-hosting

**Q2 (short):** Walk through the math of prompt caching break-even. If cached tokens cost 10% of normal input tokens and cache-writing costs 25% more than normal, how many uses are needed before caching a prefix pays off?

**Q3 (MC):** Anthropic's Batch API typically offers:
A) Same price, faster response
B) ~50% price discount for asynchronous return within 24h
C) Free calls under a quota
D) Priority latency at higher cost

**Q4 (short):** You detect that one user has generated 12% of last week's API cost. Give three plausible causes and one action per cause.

**Q5 (MC):** Output token pricing is typically:
A) The same as input token pricing
B) Cheaper than input
C) Roughly 3–5× more expensive than input
D) Free once you pay for input

**Q6 (short):** Why does prompt caching require the cached content to be at the *start* of the prompt, byte-for-byte?

**Q7 (MC):** A "$5/day per user" cap is enforced most robustly by:
A) A UI warning at $4.50
B) A Redis-backed spend counter incremented at request time, refunded on cap-hit (Module 10 pattern)
C) A monthly budget alert in AWS
D) A prompt instruction telling the model to be efficient

**Q8 (MC):** Which is NOT a reasonable action when a single route is 80% of your LLM cost?
A) Set a lower max_tokens for that route
B) Route that intent to a smaller model
C) Add prompt caching to the system prompt
D) Delete the route

**Q9 (short):** At what scale does self-hosting an open model start beating managed API pricing? Give the rough token-volume threshold and three factors that shift it.

**Q10 (MC):** Cost should be treated as:
A) A finance-team concern, not engineering
B) A first-class SLI with dashboards and alerts, right alongside latency and error rate
C) Reviewed quarterly
D) Optimized only after all other work is done

---

### Answer Key

**A1: B.** Model routing typically dominates because the price gap between a small model (Haiku-class) and a large one (Opus-class) is often 5–20×. Even a modest routing rule — 60% simple requests to the small model — cuts cost by 40–60% instantly. Prompt caching (A) is a strong second (up to 90% off the cached portion, but only on repeated prefixes). max_tokens (C) helps but the ceiling is limited (typical output usage is well below max_tokens for most requests). Self-hosting (D) is the biggest lever *at scale* but only after ~1T tokens/month.

**A2:** Let `p_in` = normal input token price. Cache creation costs `1.25 × p_in`; cache reads cost `0.1 × p_in`. For a prefix of `T` tokens used `n` times:
- Without caching: `n × T × p_in`
- With caching: `1.25 × T × p_in + (n−1) × T × 0.1 × p_in = T × p_in × (1.25 + 0.1(n−1))`
Set them equal to find break-even:
`n = 1.25 + 0.1(n−1)`
`n − 0.1n = 1.15`
`0.9n = 1.15`
`n ≈ 1.28`
So **~1.3 uses pays off**. In practice, if a prefix is used ≥ 2 times, cache it. This is why Anthropic recommends caching aggressively — the break-even is well under 2 uses under typical pricing.

**A3: B.** Batch API trades latency (async completion within 24h) for ~50% off. Ideal for nightly evals, embedding backfills, large-scale classification, non-urgent report generation. If it can wait, batch it.

**A4:** Three possible causes + action each:
1. **Retry loop bug** — the user's client is retrying on 200-OK responses; each intended interaction generates 10× the calls. *Action:* pull traces for that user; identify pattern; contact the user's integration owner; add server-side dedup for identical consecutive requests within N seconds.
2. **Legitimate power user underpriced by your tier** — they're using the feature exactly as designed but your pricing didn't account for this consumption level. *Action:* introduce a higher tier or overage pricing; discuss with product/sales; grandfather this user until change effective.
3. **Abuse / automation** — the user is scraping or resell-arbitraging your service. *Action:* apply per-user rate limit (Module 10), require captcha / MFA on high-volume, review ToS, potentially suspend and investigate.
4. (Bonus) **Bug in prompt design** — a specific input pattern the user hits causes a 100k-token retrieval expansion. *Action:* trace, reproduce, fix the retrieval or the prompt template.

**A5: C.** Output tokens are typically 3–5× more expensive than input tokens across major providers (Anthropic, OpenAI, Google). This is because generating tokens autoregressively is more compute-intensive than processing input (which can be parallelized). Consequence: reducing output length or `max_tokens` has outsize cost impact vs reducing input length.

**A6:** The provider's cache key is a **content hash of the prompt prefix**. To reuse a cached KV-state, everything from token 0 up to the cache breakpoint must match byte-for-byte with the original cache entry — otherwise the cached internal state doesn't correspond to what the model is being asked to condition on. Change one character early → new hash → cache miss → full processing. Change something *after* the last cache breakpoint → still a hit for the cached portion. Design implication: put stable content (system prompt, tools, few-shot) at the top; put per-request content (user's current message, current chat turn) at the bottom.

**A7: B.** UI warnings (A) are user-facing hygiene, not enforcement. AWS budgets (C) are backstops (24h+ latency, coarse). Prompt instructions (D) are advisory to the model, not enforced. Only a server-side counter with hard-refuse-on-cap (B) actually prevents spend. Pattern in Module 10: `INCRBYFLOAT` estimate → check cap → refund on rejection → reconcile with actual cost post-response.

**A8: D.** Deleting the route removes cost but likely destroys the feature and user value. A, B, C are all reasonable *tuning* moves that keep the feature. The right response to a hot cost route is: instrument it (per-user cost, output length histogram, cache hit rate); then choose the smallest set of levers (routing to a cheaper model + max_tokens cap + prompt cache on system prompt + brevity instruction) that meets quality bar at target cost.

**A9:** Rough threshold: **1T (10^12) tokens per month** is where a well-run self-hosted stack tends to beat managed API pricing. Three factors that shift it:
1. **Utilization** — if your traffic is bursty (10% peak, 1% average), you pay for idle GPUs. Managed API wins until you can sustain 40–70% utilization.
2. **Model size / hardware fit** — small models on cheap GPUs (7B on L4/A10) shift break-even lower; huge models (70B+) needing H100s shift it higher.
3. **Team capability** — running vLLM at scale requires a platform team. If you don't have that skill in-house, you're eating engineering cost that offsets $ savings.
Others: quantization aggressiveness (INT4 halves GPU need), spot-instance eligibility (up to 90% discount), and requiring one specific model your open ecosystem doesn't cover (forces managed).

**A10: B.** Cost is a user-visible attribute (via tier limits, latency of degraded modes) and a business survival attribute. Treating it like latency — SLIs, SLOs, dashboards, alerts, error budgets — is what mature FinOps looks like. (A) treats it as post-hoc; (C) is too infrequent for LLM cost's ability to spike hourly; (D) means it never gets attention until crisis.

---

**Related modules:**
- `learning/09-llm-observability.md` — where $ per span comes from
- `learning/07-monitoring.md` — cost dashboards live here
- `learning/10-rate-limiting.md` — cost caps and tier budgets
- `learning/11-caching.md` — every layer in the cache stack is a cost lever
- `learning/08-model-evaluation.md` — quality-per-dollar comparisons across models
- `../llm-engineering/06-cost-latency-optimization.md` — Month 4's cost primer
- `builds/01-production-hardening.md` — where these levers get applied

**Practice prompts:**
1. Add `llm_cost_usd_total` counter to your last build (label by `model, route, tenant`). Build a Grafana panel "$/hour" and "$/user top 10."
2. Implement model routing: a Haiku-based intent classifier that dispatches to Haiku/Sonnet/Opus. Compare cost + quality on 200 test queries.
3. Add Anthropic prompt caching to your longest system prompt. Measure `cache_read_input_tokens / (cache_read_input_tokens + cache_creation_input_tokens + input_tokens)`; that's your cache hit rate.
4. Configure a 15-minute cost-anomaly alert (`rate(llm_cost_usd_total[15m])` > 2× trailing 24h baseline). Trigger it deliberately to test.
5. Estimate the break-even token volume for self-hosting Llama-3.1 8B on a single A10 GPU vs Anthropic Haiku; show your assumptions.

**References:**
- Cloud FinOps Foundation — https://www.finops.org/
- Anthropic pricing — https://www.anthropic.com/pricing
- Anthropic Message Batches API docs
- vLLM project — dynamic batching, prefix caching
- Chip Huyen — *AI Engineering* Ch. 10 (Production)
- Simon Willison's LLM pricing tracker
