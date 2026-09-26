# Model Selection — Master Study Guide

> **Track:** LLM Engineering · **Module:** 08
> **Prerequisites:** Modules 01–07.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** In 2026, you have hundreds of models to choose from — from GPT-5 down to a 3B open-source model you can self-host. Pick the wrong one and you'll overpay 10× for negligible quality gain, or underpay and fail tasks silently. Picking the right one is an engineering skill grounded in **task, budget, latency, privacy, and quality requirements** — not vibes or hype.

**Fundamental principles you must own:**

1. **There's no "best model" — there's the best model for your task, budget, and constraints.**
2. **Capability tiers roughly cluster.** Frontier (GPT-4o / Claude Sonnet 4.5), mid (GPT-4o-mini / Claude Haiku), small (LLaMA-8B), tiny (Phi-3, Gemma-2B).
3. **Use the cheapest model that meets your quality bar**, not the most expensive.
4. **Test empirically.** Public benchmarks (MMLU, HumanEval, MT-Bench) don't predict your task's outcome — build your own eval.
5. **Multi-model architectures** — cascades, routers, ensembles — often beat single-model on cost or quality.
6. **Open vs closed** trades control, privacy, and per-call cost against ease and best-in-class quality.

If you retain nothing else: **pick the smallest model that clears your quality bar; validate empirically; be ready to route among models.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Modern Model Landscape (Rough 2026 Categorization)

**Frontier / flagship** — bleeding edge, expensive, best quality:
- OpenAI: GPT-5, o3.
- Anthropic: Claude Opus 5, Claude Sonnet 5, Claude Fable 5.1.
- Google: Gemini 2 Pro / Ultra.
- Open: LLaMA 4 405B, Mistral Large, Qwen 3 Max.
- Prices in mid-2026: roughly $3–15 per million input tokens; $15–75 per million output tokens.

**Mid-tier** — 80% of the quality at 10–20% of the price:
- OpenAI: GPT-4o-mini, GPT-4.1-mini.
- Anthropic: Claude Sonnet 4.5, Claude Haiku 4.5.
- Google: Gemini 2 Flash.
- Open: LLaMA 4 70B, Mixtral 8x22B, Qwen 3 72B.
- Prices: $0.15–3 per M input; $0.60–15 per M output.

**Small / edge** — cheap, fast, self-hostable:
- Open: LLaMA 4 8B, Phi-4 mini, Gemma 3 4B, Mistral 7B v3.
- Great for: classification, extraction, on-device inference, RAG synthesis on constrained hardware.

**Reasoning-specialized** — long chain-of-thought at inference:
- OpenAI: o3, o3-mini.
- Anthropic: Claude Sonnet 5 with extended thinking.
- Google: Gemini 2 Deep Think.
- DeepSeek: R1, R2.
- Slower and pricier; huge gains on math, code, and multi-step reasoning.

**Multi-modal specialists:**
- Vision + text: GPT-4o, Claude Sonnet 4+, Gemini, LLaVA (open).
- Audio: Whisper (STT), GPT-4o audio, Gemini Live.
- Video understanding: Gemini 1.5+, GPT-4o.

**Specialty models:**
- Code: DeepSeek-Coder, StarCoder, Cursor's proprietary.
- Medical: Med-PaLM 2, MedLM, GPT-4o with domain fine-tuning.
- Legal: proprietary vertical (Harvey, Casetext).

**The landscape shifts every quarter.** The specific names here will age; the tiers won't.

---

### 2.2 The Decision Matrix — How to Pick

Given a task, ask:

**1. Quality requirements.** How wrong can we be? Cancer diagnosis → frontier. Autocomplete of a dropdown → tiny.

**2. Latency requirements.** Interactive chat → target < 2s TTFT. Batch overnight → doesn't matter.

**3. Cost budget.** Cost per request × request volume × margin. Every 10× cost matters at scale.

**4. Context window.** How much text needs to fit? Legal contracts → 100k+. Chat with a doc → 32k. Classification → 4k.

**5. Capabilities needed.**
- **Structured output** with strict schemas — most modern models support; verify.
- **Tool calling** — modern only.
- **Vision** — a smaller list.
- **Multi-lingual** — non-English quality varies; test.
- **Long context recall** — even 1M-context models forget middle content ("needle in a haystack" tests).

**6. Privacy / data residency.**
- Must not leave your VPC → self-host or dedicated deployment.
- HIPAA / GDPR → BAAs / data-processing agreements with the provider.

**7. Fine-tuning / adaptation.** Do you need to fine-tune? Open models let you own that; closed models offer paid fine-tuning APIs (OpenAI, Anthropic).

**8. Reliability / support.** Frontier providers have well-run infra and support; smaller providers may be less reliable.

**A rough decision tree:**

```
Is the task safety-critical or reasoning-heavy?
├─ Yes → Frontier reasoning model (o3, Claude Sonnet 5 with thinking)
└─ No →
    Does it need vision / audio?
    ├─ Yes → GPT-4o / Claude Sonnet 4.5 / Gemini
    └─ No →
        Is it high-volume / cost-sensitive?
        ├─ Yes → Mid-tier (GPT-4o-mini, Haiku 4.5) or open (LLaMA 70B via Groq)
        └─ No →
            Is it a simple task (classification / extraction)?
            ├─ Yes → Small model, possibly fine-tuned
            └─ No → Frontier
```

Then **validate empirically** on your task. A GPT-4o-mini often ties Claude Opus 5 on classification but loses badly on reasoning.

---

### 2.3 Building Your Own Eval

**Public benchmarks lie.** MMLU, HellaSwag, and GSM8K are proxies. Your production task is not those.

**Build a golden set:**

1. Collect 20–200 real user inputs (or realistic synthetic).
2. Annotate ideal outputs (or grading criteria).
3. Include variety: easy, hard, adversarial, edge cases.
4. Include failure-mode traps you've seen before.

**Grading methods:**

- **Exact match** — for classification, extraction where output is a specific value.
- **Regex / structural** — for output format compliance.
- **BLEU / ROUGE / METEOR** — for translation, summarization (rough).
- **Embedding similarity to reference** — soft match.
- **LLM-as-judge** — a stronger model grades under a rubric.
- **Human rating** — expensive but gold-standard.

**LLM-as-judge template:**

```
Grade the response on:
- Accuracy (1-5): does it correctly answer the question?
- Faithfulness (1-5): does it stick to the provided context?
- Format compliance (1-5): does it match the requested format?

Response: {response}
Reference: {reference}

Return JSON: {"accuracy": _, "faithfulness": _, "format": _, "notes": "..."}
```

**Comparing models on your eval:**

```python
def evaluate_model(model_name, test_cases):
    scores = []
    for tc in test_cases:
        response = call(model_name, tc.input)
        score = grade(response, tc.expected)
        scores.append(score)
    return {"mean": mean(scores), "std": std(scores), "p5": p5(scores)}

for m in ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]:
    print(m, evaluate_model(m, test_cases))
```

**Report cost + latency alongside quality.** A dashboard with (quality, cost, p95_latency) per model tells the full story.

---

### 2.4 Multi-Model Architectures — Router, Cascade, Ensemble

**Model router.** Classify the incoming request; route to the right model.

```python
def classify_intent(query: str) -> str:
    """Cheap model decides: simple / complex / reasoning-heavy."""
    return small_model.classify(query)

def route(query: str):
    intent = classify_intent(query)
    if intent == "simple":  return gpt_4o_mini(query)
    if intent == "complex": return gpt_4o(query)
    if intent == "reasoning": return o3(query)
```

Router cost is small (a cheap model classifying). Savings can be large.

**Cascade.** Try a cheap model first; escalate to a stronger model if the cheap model is unsure or fails.

```python
def cascade(query):
    result = gpt_4o_mini(query)
    if result.confidence < 0.7 or looks_wrong(result):
        result = gpt_4o(query)
    return result
```

Requires a confidence signal — either from logprobs, from the model's self-rated confidence, or from a downstream validator.

**Ensemble / voting.** Sample $N$ responses (from one model or multiple) and combine. Improves reliability at $N$× cost.

- **Majority vote** for classification.
- **Self-consistency** for reasoning (Wang et al., 2022).
- **Judge-and-select** — a judge model picks the best of $N$.

**Speculative decoding at the app layer.** A small "draft" model generates a candidate; a big model verifies. Frontier providers implement this internally; some open serving stacks (vLLM) expose it.

**Reasoning + non-reasoning composition.** Use a reasoning model to *plan*, a smaller model to *execute* each step.

**Specialization by task.** Different sub-tasks might route to different specialized models (vision → GPT-4o, code → DeepSeek-Coder, chat → Claude).

---

### 2.5 Open vs Closed — When to Self-Host

**Closed (OpenAI, Anthropic, Google) advantages:**
- Top-of-the-leaderboard quality on most tasks.
- Zero infrastructure burden.
- Instant scale, worldwide latency, redundancy.
- Frontier features (structured output, prompt caching, tool use) usually best-in-class here first.

**Closed disadvantages:**
- Per-token cost.
- Data-flow constraints (your prompts go to their servers; may or may not be logged/trained on).
- Vendor lock-in (limited migration options).
- Model deprecations force refactors.

**Open (LLaMA, Mistral, Qwen, DeepSeek, etc.) advantages:**
- Weights you can inspect, quantize, fine-tune, deploy anywhere.
- Predictable cost — you own the compute.
- Data stays where you want it.
- Not subject to sudden API changes.
- Fine-tuning is straightforward (LoRA, full-parameter).

**Open disadvantages:**
- Infrastructure to run and maintain.
- Below-frontier quality (though the gap has narrowed dramatically by 2026).
- You debug performance issues yourself.

**Hybrid patterns:**
- **Frontier + open specialist** — use frontier for hard cases; fine-tuned open model for common cases (much cheaper).
- **Provider-hosted open** — Groq, Together, Fireworks, DeepInfra host open models with fast inference. You get open-model economics without ops burden.
- **On-prem for sensitive** — HR docs, medical records, legal → self-hosted open.

**A word on Groq, Cerebras, and SambaNova.** These custom-hardware providers run open models with dramatic speedups (250–1000+ tokens/sec). Great when latency dominates and open-model quality is enough.

**Model deprecation planning.** OpenAI and Anthropic deprecate models. Design your app to be **model-agnostic**: pin to model version identifiers, log the version per call, and have a migration playbook.

---

## 3. Mental Models & Analogies

### 3.1 The "Vehicle Fleet" Model

Think of models as a fleet of vehicles for your business:

- **Sports car** — frontier model. Fast, powerful, expensive. Use for the CEO's trip; not for the mail run.
- **Sedan** — mid-tier model. Comfortable, moderate cost. Everyday commuting.
- **Delivery van** — mid-tier with tools/vision. Utility work.
- **Compact hatchback** — small model. Cheap trips.
- **Cargo truck** — reasoning model. Slow, hauls heavy loads.
- **Motorcycle courier** — Groq / Cerebras. Fast for small deliveries.

You don't drive the sports car to the grocery store. You match vehicle to trip. A well-run fleet has each vehicle covering the trips it's best at, with a dispatcher (router) deciding.

Same rule with models: **match capability to task; keep a fleet, not a single vehicle**. And keep track of your fuel economy (cost) and reliability (downtime).

### 3.2 The "Ordering the Right Tool from a Catalog" Model

You have a Sears catalog of models with columns: *quality score* (on your eval), *cost per 1M in/out tokens*, *max context*, *TTFT*, *features* (vision, tools, JSON mode). Your task lives in a specific spot in that spec space.

Buying strategy:
- **Never buy a $10k tool for a $10 job.** If GPT-4o-mini clears your eval bar, using Claude Opus is waste.
- **Never buy a $10 tool for a $10k job.** Save on the wrong task and you'll build a bad product.
- **Batch buy where you can.** Prompt caching, batch APIs, cheaper models for high-volume subtasks.

Every quarter, the catalog updates. Cheaper tools get better; new categories appear. **Rerun your eval quarterly** — the model you picked six months ago may be overpriced by today.

![IMG-SEL-01](/4%20—%20LLM%20Engineering/images/IMG-SEL-01.jpg)

> **Caption:** Plot every candidate model on (cost, quality) for your task. Pick the leftmost point above your quality bar.
> **Placement:** Section 2.3.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Just Use GPT-5 for Everything"

Overkill for 80% of tasks, and you'll pay 10× the cost. The other 20% deserve it. The right approach is **routing** — a cheap model dispatches, expensive models handle hard tasks. Blanket frontier-only is engineering laziness that shows up in the CFO's spreadsheet.

### 4.2 "Public Benchmarks Tell Me What's Best"

They don't — for your task. Public benchmarks (MMLU, HumanEval, MT-Bench) test general capabilities; they can be gamed (contamination, deliberate training on benchmarks); and even honest scores don't predict domain performance. **Your task's eval is the only benchmark that matters.** Build it, run it, report it.

### 4.3 "Once I Pick a Model, I Never Need to Revisit"

Model landscape shifts every quarter. New releases beat old ones; old ones get cheaper. What was optimal 6 months ago may be strictly dominated today. Schedule quarterly model reviews. Log which model each production request used, so you can revisit historical performance.

---

## 5. Self-Assessment Bank (Model Selection)

### Questions

**Q1 (Short answer).** Name at least five factors that go into choosing a model for a task.

**Q2 (Multiple choice).** For a high-volume classification task where 95% of inputs are simple:
- (a) Always use the strongest model.
- (b) Route: cheap model for most, escalate to strong model on low-confidence.
- (c) Use only a self-hosted 3B model.
- (d) Use random routing.

**Q3 (Short answer).** Why are public benchmarks (MMLU, MT-Bench) insufficient for choosing a model for your production task?

**Q4 (Multiple choice).** A cascade pattern is:
- (a) Multiple models called in parallel and averaged.
- (b) Try cheap model first; escalate to a stronger one if the cheap answer is unsure or fails.
- (c) A specific fine-tuning method.
- (d) Only usable in reasoning tasks.

**Q5 (Short answer).** When would you self-host an open model instead of using a closed-provider API?

**Q6 (Multiple choice).** Reasoning-specialized models (o3, Claude Sonnet 5 extended thinking) are:
- (a) The right choice for all tasks.
- (b) Slower and pricier, but strongly better on multi-step math/logic/coding tasks.
- (c) Only useful for vision.
- (d) A marketing gimmick.

**Q7 (Short answer).** Give an example of a task where a small (3–8B) model is a legitimate choice.

**Q8 (Multiple choice).** Groq and Cerebras primarily offer:
- (a) Proprietary frontier models.
- (b) Custom hardware that runs open models at extreme speed (100+ tokens/sec).
- (c) Vector databases.
- (d) Fine-tuning services.

**Q9 (Short answer).** Explain the "quality vs cost" plot as a model selection tool, and how you'd use it.

**Q10 (Multiple choice).** For a chat feature with tight latency SLOs (<500ms TTFT):
- (a) Use GPT-4o exclusively.
- (b) Consider Groq/Cerebras-hosted open models or the smallest capable API model.
- (c) Increase temperature.
- (d) Turn off streaming.

---

### Answer Key & Detailed Explanations

**A1.** Any five of: **quality on the task** (measured empirically), **cost per token**, **latency (TTFT, TPOT)**, **context window**, **required capabilities** (vision, tool use, structured output, multilingual), **privacy / data residency**, **fine-tuning support**, **rate limits / scale**, **provider reliability / SLA**.

**A2. (b).** Routing / cascade: run a cheap model for the 95% simple cases; escalate to a strong model only on the 5% hard ones. Delivers roughly 20× cost reduction while preserving quality on the tail.

**A3.** Public benchmarks test general capability — not your specific data, your specific format, your specific edge cases. They can also be **contaminated** (models trained on benchmark data) or **gamed** (specialized fine-tuning to score high without transferring). Your task's own eval, with your own real inputs and grading criteria, is the only reliable signal.

**A4. (b).** Cascade: cheap model first; escalate to strong model when confidence is low or the cheap answer fails a validation. Optimizes average cost while preserving worst-case quality.

**A5.** When (a) data privacy or compliance forbids sending prompts to third parties; (b) per-token cost dominates and self-hosting is cheaper at your volume; (c) you need to fine-tune with control over the weights; (d) you need offline / on-prem / edge inference; (e) you need latency lower than any hosted API can offer.

**A6. (b).** Reasoning models spend more inference time on internal chain-of-thought. They're 5–50× slower and often more expensive per response, but on tasks requiring multi-step reasoning (math, code, planning, complex extraction), they can outperform non-reasoning frontier models by wide margins.

**A7.** Examples: (a) high-volume email classification, (b) code autocomplete for a fixed autocomplete dropdown, (c) intent classification in a router, (d) RAG synthesis when you have strong retrieval and just need fluent restating of retrieved content, (e) on-device inference for privacy or offline use, (f) named-entity extraction with a fine-tuned Phi/Gemma. Small models with fine-tuning often outperform frontier models on narrow tasks at a fraction of the cost.

**A8. (b).** Groq (LPU) and Cerebras (WSE) build custom silicon optimized for LLM inference. They host open-weight models and deliver 5–20× the speed of GPUs, at competitive prices. Great choice when latency dominates.

**A9.** Plot models on axes: horizontal = cost per M tokens (input + output blended), vertical = quality on YOUR eval. Draw your quality bar. Pick the leftmost (cheapest) model above the bar. Re-run this plot quarterly as prices drop and new models release.

**A10. (b).** Sub-500ms TTFT is hard on any large hosted API — inherent network + model queue overhead. Options: Groq / Cerebras-hosted open models (typical TTFT ~200ms), the smallest capable API model (GPT-4o-mini can hit 400ms), local/edge inference with a small model. Turning off streaming (D) makes it worse. Increasing temperature (C) doesn't affect latency.

---

## 6. Practice Prompts

1. **Model shootout.** Run your production prompt through GPT-4o, GPT-4o-mini, Claude Sonnet 4.5, and Claude Haiku on a 50-case eval. Report quality, cost, and p95 latency per model.
2. **Build a router.** Write a classify-then-route function that dispatches easy cases to a small model and hard cases to a frontier model. Measure quality-preserving cost savings.
3. **Cascade.** Add a confidence signal (self-rated confidence via prompt, or logprobs where available). Cascade based on it.
4. **Groq comparison.** Deploy the same open model (LLaMA 4 70B) via Groq and Together. Compare TTFT and TPOT.
5. **Quality/cost plot.** Build the (cost, quality) plot from Section 2.5 for your own task. Identify the current best choice.

---

## 7. References

- LMSys Chatbot Arena leaderboard: [https://chat.lmsys.org/?leaderboard](https://chat.lmsys.org/?leaderboard) — human preference rankings.
- Artificial Analysis: [https://artificialanalysis.ai/](https://artificialanalysis.ai/) — cost & speed benchmarks.
- Hugging Face Open LLM Leaderboard: [https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).
- SEAL Leaderboards (Scale AI): task-specific held-out benchmarks.
- Simon Willison's posts on evaluating models on your own tasks.
