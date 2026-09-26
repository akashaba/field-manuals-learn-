# 09 — LLM Observability

> **Module goal:** Master the specialized observability stack for LLM applications — tracing every prompt, tracking every token, watching for hallucinations, guardrail hits and cost outliers, and turning production traces into datasets that feed the next model release. Distinguish generic observability (Month 6 modules 6–7) from what LLMs specifically demand.

---

## 1. Executive Summary & Core Concepts

An LLM call is **fundamentally different** from a traditional HTTP call in ways that break generic observability:

1. **The input matters.** A `POST /users` payload rarely tells you why the response was bad. A prompt *is* the code and the data, and you must retain it (with PII redaction) to diagnose anything.
2. **The output is unstructured.** No status code says "the answer was wrong." Success is a semantic judgment.
3. **Non-determinism is default.** Same prompt → different response. You can't reproduce a bug from logs alone unless you seed and freeze.
4. **Cost is per-token.** A pathological input can cost 100× a normal one. Metrics need $ dimensions.
5. **Latency has fat tails.** Streaming makes "first token latency" and "total generation time" distinct SLIs.
6. **Composition is deep.** One user request → 5 LLM calls (planner, tool-caller, retriever's rerank, generator, judge). Traces span services and models.

**LLM observability = tracing + evaluation + cost + quality signals**, unified around the trace as the primary object.

**Core objects:**

| Object | Description | Retention |
|--------|-------------|-----------|
| **Trace** | One user request end-to-end | Days–weeks; sampled |
| **Span** | One LLM call or tool call within a trace | Same as trace |
| **Dataset** | Curated traces used for eval or fine-tuning | Forever, versioned |
| **Score** | Numeric/categorical label on a trace/span (from LLM-judge, user feedback, human review) | Forever |
| **Session** | Multi-turn conversation grouping | Weeks |

**The tools:**

| Tool | Type | Strengths |
|------|------|-----------|
| **LangSmith** | Managed (LangChain Inc.) | Deep LangChain integration, dataset management, eval runners |
| **LangFuse** | Open-source, self-hostable | Vendor-neutral, cheap, prompt versioning |
| **Arize Phoenix** | Open-source | Great UX, embedding drift viz, OTel-native |
| **Weights & Biases Weave** | Managed | Ties to W&B ecosystem for experiments |
| **Braintrust** | Managed | Strong on eval iteration workflow |
| **Datadog LLM Observability** | Managed enterprise | Integrated with existing Datadog fleet |
| **Home-grown (OTel + Postgres)** | DIY | Full control, aligned with your existing stack |

The pattern is convergent: they all record traces, attach scores, expose dashboards. **Pick one, instrument once with OTel semantic conventions** (`gen_ai.*`), avoid lock-in.

**Signals to capture per LLM call:**
- **Inputs**: system prompt, user messages, model, temperature, max_tokens, tools available, retrieved context
- **Output**: raw response, parsed structured output, tool calls returned, finish_reason
- **Usage**: input_tokens, output_tokens, cached_tokens (prompt cache hits), reasoning_tokens
- **Timing**: request start, first token, last token, total duration
- **Metadata**: request_id, user_id (hashed), session_id, environment, git SHA, prompt version
- **Cost**: computed from usage × price table
- **Quality signals**: user thumbs, LLM-judge score, guardrail results

The user-facing win: when someone reports "the bot said something weird," you can pull the exact trace, see the retrieved context, the tool calls, the prompt version — and immediately reproduce or file it as an eval case.

---

## 2. Deep-Dive Breakdown

### 2.1 Tracing an LLM Call

The minimum viable trace with LangFuse:

```python
from langfuse import Langfuse, observe
from langfuse.openai import openai  # drop-in replacement

lf = Langfuse()

@observe()  # creates trace
def answer_question(user_id: str, question: str) -> str:
    # Retrieval as a span
    context = retrieve(question)
    # LLM call auto-instrumented via the openai wrapper
    resp = openai.chat.completions.create(
        model="claude-sonnet-5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}"},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    lf.update_current_trace(user_id=user_id, tags=["prod", "v3.2.1"])
    return answer

@observe()
def retrieve(question: str) -> str:
    # ... vector DB call ...
    return "..."
```

LangFuse (or Phoenix, or LangSmith) captures the nested spans automatically, records tokens/cost/latency, and shows a waterfall UI you can click through per trace.

**With OTel semantic conventions** (portable, backend-agnostic):

```python
from opentelemetry import trace

tracer = trace.get_tracer("my-ai-service")

def answer(question: str) -> str:
    with tracer.start_as_current_span("llm.chat") as span:
        span.set_attributes({
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-sonnet-5",
            "gen_ai.request.temperature": 0.2,
            "gen_ai.request.max_tokens": 1024,
            "user.id_hash": hash_user(user_id),
        })
        resp = client.messages.create(...)
        span.set_attributes({
            "gen_ai.usage.input_tokens": resp.usage.input_tokens,
            "gen_ai.usage.output_tokens": resp.usage.output_tokens,
            "gen_ai.usage.cache_read_input_tokens": resp.usage.cache_read_input_tokens,
            "gen_ai.response.finish_reasons": [resp.stop_reason],
            # PII-safe: hash or truncate input; skip in some tenants
            "gen_ai.request.messages_hash": sha256(str(messages)),
        })
        return resp.content[0].text
```

### 2.2 Prompt Versioning

A subtle production trap: **you change the prompt, quality drops, and you have no idea when or which change was the culprit**. Every LLM observability tool now offers **prompt management** — versioned strings with a stable ID.

```python
# Fetch prompt from LangFuse; if service is down, fall back to bundled default
prompt_template = lf.get_prompt("customer-support-classifier", version="production")
# `production` is an alias like MLflow's; you promote versions via UI or API
messages = prompt_template.compile(user_question=q, context=ctx)
# The prompt version ID gets attached to the trace automatically
```

Benefits:
- Every trace links to the exact prompt version that produced it
- Roll back a bad prompt without redeploying
- A/B test prompts by version-tagged cohort
- Compare eval scores across versions in one dashboard

Discipline: **prompts are code.** They live in a repo. The LangFuse/LangSmith UI is a *deploy convenience,* not the source of truth — sync prompt changes back to git via PR.

### 2.3 Real-Time User Feedback Loop

Attach user signals (thumbs up/down, edits, copy-events) to the trace by ID:

```python
# Return trace_id to the frontend
@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    with tracer.start_as_current_span("chat") as span:
        trace_id = span.get_span_context().trace_id
        answer = await answer_question(req.user_id, req.question)
        return ChatResponse(answer=answer, trace_id=format(trace_id, "032x"))

# Frontend calls this when user clicks 👍/👎
@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    lf.score(trace_id=req.trace_id, name="user_thumbs", value=req.value)
```

Now your dashboard can filter to `user_thumbs = -1` traces, cluster them, and generate new eval cases from real failures.

### 2.4 Sampling & Cost Control (Traces Are Expensive)

Full-fidelity tracing of every prompt at scale is costly (storage and observability-tool billing). Strategies:

- **Head sampling** — decide at trace start whether to keep (e.g., 10%). Cheap but blind to rare failures.
- **Tail sampling** — decide after seeing the outcome. Keep 100% of errors, 100% of slow requests, 5% baseline. Requires an OTel Collector with the `tail_sampling` processor.
- **Always sample**: errors, refusals, guardrail blocks, low LLM-judge scores, cost outliers, user thumbs-down.
- **Compress**: hash prompt/context if you can't retain full text; keep the full trace only for a sampled subset.

For LLM apps specifically, the trace *is* the debugging tool — err on the side of full retention for a shorter window (7–30 days), then drop or aggregate.

### 2.5 Dashboards & Alerts Specific to LLMs

Beyond RED/USE, an LLM-observability dashboard should show:

| Row | Panels |
|-----|--------|
| Cost | $/hour, $/user, $/route, cost by model |
| Volume | RPM by route, tokens/sec by model, cache hit rate |
| Latency | p50/p95/p99 TTFT (time-to-first-token), full generation, per model |
| Quality | LLM-judge score trend, refusal rate, guardrail block rate, thumbs ratio |
| Safety | PII detection hits, jailbreak attempts, blocked outputs |
| Errors | API error rate by model/provider, retry rate, fallback rate |

Alerts worth setting:
- **Cost spike:** `rate(llm_cost_usd_total[15m]) > 3 × baseline` → page
- **Model provider outage:** `rate(llm_errors_total{provider="anthropic",code="5xx"}[5m])` → alert + auto-fail-over
- **Quality drop:** rolling LLM-judge score falls > 5 points → alert
- **Refusal spike:** refusal rate 2× baseline → alert (could be prompt regression or attack)
- **Latency SLO burn** on TTFT (streaming users feel first-token latency, not full latency)

`[IMG-09-01]` — *Prompt: A four-quadrant LLM observability dashboard mockup. Top-left: a trace waterfall showing parent span "chat_request" with nested spans "retrieve_context" (200ms), "rerank" (150ms), "llm_generate" (1.8s, highlighted). Top-right: a cost-per-hour line chart with a spike labeled "New feature launch — investigate". Bottom-left: a heatmap of user thumbs-down clustered by intent category. Bottom-right: prompt version A/B comparison bar chart with LLM-judge scores. Modern dark UI, Grafana-like.*

---

## 3. Mental Models & Analogies

### Model 1: The Airline Black Box

Every commercial aircraft has a Flight Data Recorder (metrics-like: continuous parameters) and a Cockpit Voice Recorder (log-like: raw signals) that together let investigators reconstruct any incident post hoc. Crucially, the recorders capture what's needed *before* an incident happens — you don't know which flight will crash, so you record all of them.

LLM tracing is the black box for your AI service. When a user reports "the bot gave me wrong medication info," you cannot say "we didn't save that request." Every prompt-response pair with its retrieval context, tool calls, and versioned prompt IS your black box. Without it you cannot fix, cannot regulate, cannot improve. The rest of the module is *how* to record without going bankrupt (sampling), *what* to redact (PII), and *how* to search back to a single trace at 3 AM.

### Model 2: The Restaurant Kitchen Ticket

In a busy kitchen, every order flows through a chit — the ticket printed at the pass. It records what was ordered, when, by which server, which stations touched it, and any special instructions. Expo staff can pick up any ticket and reconstruct why a table waited 40 minutes. At end-of-service, tickets are sorted into "well-run," "80s" (issues), and comps — data that improves next service.

Your LLM trace is that ticket. Each span (retrieval, rerank, generate, judge) is a station stamp. The trace's `finish_reason` is "delivered" or "sent back." Thumbs-down is a comp. Sampling those into a dataset for the next model iteration is the head chef's Monday morning debrief — the mechanism by which service actually improves. A restaurant without tickets runs on the memory of the person who happened to be there; an LLM service without traces is the same.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Retaining raw prompts with PII forever.**
LLM traces contain user text. Emails, SSNs, medical questions, credit cards — anything users type. Storing this untreated in a third-party SaaS violates GDPR, HIPAA, likely your own privacy policy, and creates a breach surface. **Fixes:** (1) redact PII pre-storage (regex + entity-recognition pass, see Module 6); (2) hash-only sensitive fields; (3) short retention (7–30 days for raw text; longer for redacted/aggregated); (4) region-pinned storage (EU data in EU tenant); (5) for regulated workloads, self-host LangFuse/Phoenix on your VPC; (6) offer per-user opt-out of trace storage. Not planning this is the #1 way LLM observability becomes a liability.

**Pitfall #2 — Logging structured outputs as strings.**
Your model returns JSON. You log the string `'{"intent": "cancel"}'`. Later you want to filter traces by `intent = "cancel"`, but you're doing regex on strings across 10 million rows. **Fix:** parse the output into a Python object at the boundary, attach as structured span attributes (`output.intent = "cancel"`) *and* keep the raw string for full fidelity. Same for tool calls: log both the tool name/args as structured attributes AND the raw model output. Now you can dashboard by intent, tool name, or error class trivially.

**Pitfall #3 — Cost visibility only at end-of-month.**
Your finance team gets a $47,000 Anthropic bill. You have no idea which endpoint, model, or customer drove it. This has happened to real teams multiple times. **Fix:** compute cost per span using a price table at ingestion time, attach as an attribute, and roll up in real time — $/route, $/user, $/model, $/prompt-version. Alert on 15-minute cost-rate spikes. Cap per-user spend with a rate-limiter (see Module 10). The rule: **you should be able to see today's spend before lunch, not next month.** LangFuse, Phoenix, LangSmith, Braintrust, Datadog LLM all provide cost views out of the box; use them.

---

## 5. Self-Assessment Bank

**Q1 (MC):** A trace in LLM observability represents:
A) A single LLM API call
B) One user request end-to-end, potentially spanning multiple LLM/tool calls
C) A log line
D) A Prometheus metric

**Q2 (short):** Explain the difference between TTFT (time-to-first-token) and total generation latency. Why does TTFT matter for streaming UIs?

**Q3 (MC):** Your finance team asks why last month's LLM bill jumped 4×. Your observability stack is only recording aggregate token counts by day. What did you fail to instrument?
A) Nothing — that's enough
B) Per-request, per-user, per-endpoint, per-model cost dimensions
C) GPU utilization
D) Prompt versions

**Q4 (MC):** Which of the following is the correct approach to storing prompts with user PII in a hosted LLM-observability tool?
A) Store as-is; it's your data
B) Redact PII (regex + NER) before storage, hash sensitive fields, set short retention, and pin storage region
C) Never store prompts
D) Encrypt at rest and forget it

**Q5 (short):** Describe tail sampling and give three signals that should force a trace to be kept.

**Q6 (MC):** Prompt versioning in LLM observability tools lets you:
A) Roll back a bad prompt without a code deploy
B) Attach the prompt version ID to every trace it produced
C) A/B test prompt variants by cohort
D) All of the above

**Q7 (short):** You have LangFuse, LangSmith, Phoenix, and Datadog LLM Observability shortlisted. Name one differentiator each.

**Q8 (MC):** OpenTelemetry semantic conventions for GenAI use the `gen_ai.*` namespace. Which of these is a valid attribute?
A) `gen_ai.request.model`
B) `openai.model`
C) `llm.completion`
D) `ai.gen.model`

**Q9 (short):** You get a bug report: "Yesterday at 3 PM the bot answered incorrectly." How does end-to-end tracing let you diagnose this in minutes rather than hours?

**Q10 (MC):** Which of the following is NOT a signal you should attach as a score on a trace?
A) User thumbs up/down
B) LLM-judge faithfulness score
C) Guardrail pass/fail
D) The user's IP address

---

### Answer Key

**A1: B.** A trace is the whole request-scoped tree; individual LLM calls are spans within it. Distributed-tracing terminology carries over from generic APM: trace = root, spans = nodes. One user chat message may contain a planner call, several tool calls, a retriever call, and a generator call — all under one trace ID.

**A2:** **TTFT** = wall time from request start to the first token appearing in the response stream. **Total generation latency** = TTFT + time to produce all remaining tokens. For streaming UIs, TTFT is what the user *feels* as "the bot is responding" — a fast TTFT of 400ms feels snappy even if full generation takes 6 seconds. A slow TTFT (say 3s) with the same total feels broken. So streaming applications should SLO both: TTFT p95 < 1s *and* total p95 < 8s, say. Non-streaming services only need total latency.

**A3: B.** Aggregate tokens/day hides everything. You need cost attributed at request granularity so you can group by user (who's driving spend), route (which feature), model (are we over-using the expensive one), and prompt version (did a bad prompt double token usage). Instrument at the trace/span level with a price-table multiplication; visualize by all four dimensions.

**A4: B.** The pragmatic answer: (1) run a redaction pass (SSNs, emails, cards, names via NER) before writing to storage; (2) hash irreversibly what you can't fully clean; (3) 7–30 day retention on raw text, longer on redacted; (4) region-pinned tenant for cross-border compliance; (5) self-host for regulated data. (A) is a lawsuit; (C) makes debugging impossible; (D) doesn't help — you can still leak to authorized staff, subpoenas, or in a breach.

**A5:** **Tail sampling** decides whether to keep a trace *after* the request completes, based on outcome — as opposed to head sampling, which decides at start. This lets you retain interesting traces without the cost of retaining everything. Three signals that should force keeping: (1) error / non-2xx / model API failure; (2) latency > SLO threshold; (3) user thumbs-down or LLM-judge score below floor. Also strong candidates: guardrail hit, refusal, cost above per-request cap, novel intent, new prompt version's first hour of traffic (over-sample new versions).

**A6: D — all of the above.** Prompt versioning gives you (1) rollback without redeploy, since the prompt is fetched at runtime; (2) automatic version→trace linkage for regression debugging; (3) cohort-based A/B testing (route 10% of users to v3.1, 90% to v3.0, compare scores). The catch is you must still sync prompt changes back into your git repo — the SaaS UI is a *deploy pane*, not the source of truth.

**A7:**
- **LangSmith** — deep first-party LangChain/LangGraph integration; probably the best dataset+eval iteration UX if you're on LangChain.
- **LangFuse** — open-source, self-hostable; strong prompt management; vendor-agnostic (works with any LLM SDK); cheapest for high volume.
- **Arize Phoenix** — open-source, OTel-native; excellent embedding drift visualization; strong for research/prototyping.
- **Datadog LLM Observability** — integrated with your existing Datadog fleet metrics/logs/APM; enterprise-friendly if Datadog is already your platform. (Others valid: Braintrust — eval iteration; Weave — W&B ecosystem.)

**A8: A.** `gen_ai.request.model` is the OTel-conventional attribute. The `gen_ai.*` namespace covers `gen_ai.system`, `gen_ai.request.model`, `gen_ai.request.temperature`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, and others. The convention is drafted by the OTel community and adopted by Anthropic, OpenAI SDK auto-instrumentation, LangSmith, LangFuse, Datadog — using it means dashboards and alerts port across backends.

**A9:** With tracing: pull user's request from that time window by user_id or session_id → open the trace → see the retrieved context (was it wrong or missing?), the tool calls made (did the extraction fail?), the exact prompt version used (was it the one we rolled back Tuesday?), the raw model response (did it hallucinate or was it intercepted?), the LLM-judge score if attached, and any guardrail results. Total time: 2–5 minutes to root cause. Without tracing: you have only aggregate logs, cannot reproduce the response (non-determinism), don't know which prompt/model version was active, and rely on the user to remember exactly what they typed. Hours to give up.

**A10: D.** IP address is PII (in many jurisdictions), high-cardinality, and not a quality/behavior signal about the LLM interaction. Scores attach evaluative labels — thumbs, judge, guardrail, human review verdict. IP belongs in access logs (short retention, restricted access), not in trace scores. (A, B, C) are all standard score inputs used to filter for eval-case candidates and to trend quality over time.

---

**Related modules:**
- `learning/06-logging.md` — the base logging layer traces sit on
- `learning/07-monitoring.md` — the metrics layer alongside traces
- `learning/08-model-evaluation.md` — scores from evals attach to traces
- `learning/14-cost-optimization.md` — LLM cost is a first-class LLM-obs metric
- `../agents-production/12-agent-evaluation.md` — agent trace shapes and evaluation

**Practice prompts:**
1. Sign up for LangFuse Cloud (or self-host). Wire it into your last build. Send 20 requests. Explore the traces UI.
2. Add prompt versioning: extract your system prompt to LangFuse. Change it there without redeploy. Confirm the traces show the new version.
3. Add a `/feedback` endpoint that accepts `(trace_id, thumbs)` and posts a score to LangFuse. Build a dashboard filtered to thumbs-down traces.
4. Write a Python script that pulls traces with `user_thumbs = -1` from the last week and outputs a JSONL eval-set candidate file.

**References:**
- OpenTelemetry GenAI Semantic Conventions — https://opentelemetry.io/docs/specs/semconv/gen-ai/
- LangFuse docs — https://langfuse.com/docs
- Arize Phoenix — https://docs.arize.com/phoenix
- LangSmith docs — https://docs.langchain.com/langsmith
- Chip Huyen, "AI Engineering" (2024) — chapter on observability
- Hamel Husain, "Fuck You, Show Me the Prompt" — practical guide to inspecting LangChain internals
