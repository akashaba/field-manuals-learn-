# 06.04 — Tracing & Debugging

> **Module 6: Evaluation** · Subtopic 4 of 4
> **Prerequisites:** 05.01–05.03 (tool runtime, agent loops, event-sourced state), 02.02 §8 (context manifest), 04.02 §1 (failure attribution), 06.01–06.03, OpenTelemetry basics (traces, spans, context propagation).
> **Outcome:** you can instrument LLM applications and agents end to end with **OpenTelemetry GenAI semantic conventions**, capture the right data safely (redaction, sampling, retention), reproduce any production failure deterministically, and run a systematic **error analysis** practice that turns traces into prioritised fixes and new eval cases.

> **Version note (checked September 2026).** The OpenTelemetry GenAI semantic conventions are widely used but **still under active development** (not yet stable). Attribute names below match the `opentelemetry-semantic-conventions` Python package 0.65b0. Examples: `gen_ai.provider.name` (which replaced the older `gen_ai.system`), `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.reasoning.output_tokens`, and the operations `chat`, `embeddings`, `retrieval`, `execute_tool`, `invoke_agent`, `invoke_workflow`, and `create_agent`. Instrumentations typically let you opt in to the latest experimental conventions and to **message-content capture, which is off by default**. Check your instrumentation's documentation.

---

## 1. Why LLM Systems Need Tracing More Than Most

A single user request fans out into:
- query planning;
- several retrievals;
- reranking;
- one or more model calls with tool loops;
- sub-agents;
- remote MCP/A2A calls.

Failures are **non-deterministic**, **silent** (a wrong answer with a 200 status), and **compositional** (a bad chunk → a wrong citation → a wrong answer). Logs and metrics alone can't answer "why did the model say that?". The **trace** — a causally linked tree of spans carrying prompts, context manifests, tool arguments and results, tokens, and versions — is the primary debugging artifact.

```
trace 4bf92f35…  user: "Did HB 45 change county insurance premiums?"                         total 6.8 s  $0.031
├─ invoke_agent legislative-assistant                                                         6.8 s
│  ├─ chat planner-small            (plan: route=search, ids=[HB-45])        in 820 / out 96   0.41 s
│  ├─ retrieval hybrid              (bm25: 100, dense: 100 → fused 30; k-shortfall: no)          0.09 s
│  ├─ rerank cross-encoder          (30 → 6; top p̂=0.94; gate kept 4)                            0.05 s
│  ├─ chat answer-large             (ctx manifest: 4 items, 3,120 tok; cache_read 2,048)          3.9 s
│  │  └─ execute_tool get_bill_status  args#a91f  → {status: enacted}                            0.12 s
│  ├─ chat answer-large (turn 2)    finish_reasons=[end_turn]                in 3,410 / out 402  1.8 s
│  └─ verify citations              (4/4 cited ids in manifest ✔)                                0.02 s
└─ feedback: copy=true, thumbs=null
```

---

## 2. Trace Design with OpenTelemetry GenAI Conventions

### 2.1 Span taxonomy

| Operation (`gen_ai.operation.name`) | Span name | Key attributes |
|---|---|---|
| `chat` / `text_completion` / `generate_content` | `chat {gen_ai.request.model}` | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.request.temperature`, `gen_ai.request.max_tokens`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`, `gen_ai.usage.reasoning.output_tokens`, `gen_ai.response.finish_reasons`, `gen_ai.conversation.id` |
| `embeddings` | `embeddings {model}` | model, input tokens, dimensions |
| `retrieval` | `retrieval {data_source}` | Your own attributes: retriever legs, k per leg, fused count, filters hash, index version |
| `execute_tool` | `execute_tool {gen_ai.tool.name}` | `gen_ai.tool.name`, `gen_ai.tool.call.id`, (opt-in) `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` |
| `invoke_agent` / `invoke_workflow` | `invoke_agent {gen_ai.agent.name}` | `gen_ai.agent.name`, agent/workflow version, budgets |

**Opt-in content attributes** (`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, tool arguments and results) can contain sensitive data. Enable them deliberately, redact them (§3), and route them to a store with appropriate access controls.

### 2.2 Attributes you add (the ones that make debugging possible)

- **Versions:** `app.prompt.id` / `app.prompt.version` (02.01 §7.3), `app.index.version`, `app.embedding_model`, `app.reranker.version`, `app.dataset.version` for eval runs, and the deploy SHA.
- **Context manifest:** the item IDs, sources, scores, and token counts (02.02 §8), stored as an event or a linked blob.
- **Decisions:** router route, planner filters, gate thresholds, and budget remaining (05.02 §3).
- **Outcome signals** linked later: judge scores, user feedback, and task completion (06.03). Attach them with span links, or with events keyed by trace ID.

### 2.3 Context propagation across services

Use W3C `traceparent` everywhere: HTTP between services, **Kafka headers** for asynchronous steps, **MCP `_meta`** (`traceparent`, `tracestate`; 05.04 §2.3), and **A2A** request metadata or headers (05.05). With this, a trace can span your Spring Boot API → a Python agent service → an MCP server → a partner A2A agent.

**JVM:** the OpenTelemetry Java agent auto-instruments HTTP, JDBC, and Kafka. Spring AI emits Micrometer observations for chat, embedding, and tool calls, which can be bridged to OpenTelemetry. Add the custom attributes from §2.2 in a `ChatClient` advisor or an observation filter.

```python
import time
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

exporter = InMemorySpanExporter()                      # production: OTLP exporter → collector → Tempo/Jaeger/Langfuse
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("legis-assistant", "1.4.0")

CAPTURE_CONTENT = False                                # opt-in, after redaction (§3)


@contextmanager
def chat_span(provider_name: str, model: str, prompt_version: str, conversation_id: str, messages=None):
    with tracer.start_as_current_span(f"chat {model}", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", provider_name)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.conversation.id", conversation_id)
        span.set_attribute("app.prompt.version", prompt_version)
        if CAPTURE_CONTENT and messages is not None:
            span.set_attribute("gen_ai.input.messages", redact(str(messages)))
        t0 = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, type(e).__name__))
            raise
        finally:
            span.set_attribute("app.latency_ms", round((time.perf_counter() - t0) * 1e3, 1))


def record_usage(span, usage: dict, finish_reason: str, response_model: str):
    span.set_attribute("gen_ai.response.model", response_model)
    span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
    span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
    span.set_attribute("gen_ai.usage.cache_read.input_tokens", usage.get("cache_read_input_tokens", 0))
    span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])


@contextmanager
def tool_span(name: str, call_id: str, args_hash: str):
    with tracer.start_as_current_span(f"execute_tool {name}", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)
        span.set_attribute("gen_ai.tool.call.id", call_id)
        span.set_attribute("app.tool.args_sha256", args_hash)
        yield span
```

---

## 3. Capturing Data Safely

### 3.1 Redaction

Prompts and tool results contain personal data. Redact **before** export, in-process or in the collector (with an OpenTelemetry Collector processor):
- Deterministic placeholders keep references consistent (`<EMAIL_1>` appears everywhere that email address appeared).
- Keep domain identifiers that aren't personal (bill numbers, statute citations), because debugging needs them.
- Hash tool arguments by default; store the raw arguments only in a restricted store.

```python
import hashlib
import re

PII_PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def redact(text: str, salt: str = "trace-v1") -> str:
    """Deterministic, per-value placeholders; statute citations like 2-18-303 are left intact."""
    def sub(kind):
        def repl(m):
            tag = hashlib.sha256(f"{salt}:{m.group(0)}".encode()).hexdigest()[:6]
            return f"<{kind}_{tag}>"
        return repl
    for kind in ("EMAIL", "SSN", "CARD", "PHONE"):
        text = PII_PATTERNS[kind].sub(sub(kind), text)
    return text
```

### 3.2 Sampling and retention

Storing full prompt/response content for every request is expensive and risky. Use **tail-based sampling**: decide *after* the trace completes, so the interesting traces are kept:
- keep 100% of errors, timeouts, and guardrail violations;
- keep 100% of traces with negative feedback or low judge scores;
- keep slow traces (above P95);
- keep traces from new releases at a higher rate for the first days;
- keep a small uniform sample (e.g. 1–5%) for unbiased statistics.

Retention tiers: full content for 7–30 days (restricted access); metadata and metrics for longer; and **promoted** traces copied into eval datasets (06.02) under dataset governance.

```python
import random


def keep_trace(t: dict, p95_ms: float, uniform_rate: float = 0.02, new_release_rate: float = 0.25,
               rng: random.Random = random.Random(0)) -> tuple[bool, str]:
    if t.get("error") or t.get("guardrail_violation"):
        return True, "error_or_violation"
    if t.get("feedback") == "negative" or (t.get("judge_score") is not None and t["judge_score"] < 0.5):
        return True, "negative_signal"
    if t.get("duration_ms", 0) > p95_ms:
        return True, "slow"
    if t.get("release_age_days", 99) < 3 and rng.random() < new_release_rate:
        return True, "new_release"
    return (rng.random() < uniform_rate), "uniform"
```

---

## 4. Reproducing Failures Deterministically

A production failure you can't reproduce, you can't fix with confidence. Capture enough to **replay**:

1. **Pinned versions:** the model ID and snapshot, the prompt version, the index snapshot or version, and the tool versions.
2. **Recorded external interactions** ("cassettes"): model responses, tool results, and retrieval results, keyed by a hash of the request.
3. **Replay modes:**
   - **Full replay:** everything from the cassette. Verifies deterministic post-processing.
   - **Partial replay:** re-run one component live (e.g. the new prompt) with the rest from the cassette. This isolates the effect of a change.
   - **Fork:** resume from step $t$ of an event-sourced run (05.03 §2.2) with a modified decision.

```python
import hashlib
import json


class Cassette:
    """Record/replay for LLM, retrieval, and tool calls. mode: 'record' | 'replay' | 'passthrough'."""

    def __init__(self, mode: str = "record", store: dict | None = None):
        self.mode, self.store, self.misses = mode, store if store is not None else {}, []

    @staticmethod
    def key(kind: str, request: dict) -> str:
        return hashlib.sha256(f"{kind}:{json.dumps(request, sort_keys=True, default=str)}".encode()).hexdigest()

    def call(self, kind: str, request: dict, live_fn):
        k = self.key(kind, request)
        if self.mode == "replay":
            if k not in self.store:
                self.misses.append((kind, request))
                raise KeyError(f"cassette miss for {kind}")
            return self.store[k]
        result = live_fn(request)
        if self.mode == "record":
            self.store[k] = result
        return result
```

### 4.1 Trace diffing

When version B fails where A succeeded on the same input, diff the traces structurally:
- the span sequence (did B skip the verification step?);
- the tool calls and argument hashes;
- the retrieved IDs (manifest diff);
- the token counts;
- the finish reasons.

```python
def diff_traces(a: list[dict], b: list[dict]) -> dict:
    """a, b: ordered span summaries [{'name', 'attrs': {...}}]. Structural + attribute diff."""
    names_a, names_b = [s["name"] for s in a], [s["name"] for s in b]
    only_a = [n for n in names_a if n not in names_b]
    only_b = [n for n in names_b if n not in names_a]
    attr_changes = {}
    for sa in a:
        sb = next((s for s in b if s["name"] == sa["name"]), None)
        if sb:
            changed = {k: (sa["attrs"].get(k), sb["attrs"].get(k))
                       for k in set(sa["attrs"]) | set(sb["attrs"]) if sa["attrs"].get(k) != sb["attrs"].get(k)}
            if changed:
                attr_changes[sa["name"]] = changed
    return {"only_in_a": only_a, "only_in_b": only_b, "changed": attr_changes}
```

---

## 5. Error Analysis: The Highest-Leverage Practice

Metrics tell you *that* quality dropped. Error analysis tells you *why*, and *what to fix first*. The practice (adapted from qualitative research):

1. **Sample** 50–100 traces: failures (negative feedback, low judge scores) plus a random slice.
2. **Open coding:** read each trace end to end, and write a short free-text note on the *first* thing that went wrong (the upstream-most error).
3. **Axial coding:** group the notes into failure categories — e.g. *retrieval missed amended version*, *planner dropped the session year*, *model ignored the abstention rule*, *tool timeout not handled*.
4. **Quantify:** frequency × severity, and localise each category to a component (04.02 §1 attribution).
5. **Act:** fix the top categories; **add each category's examples to the eval set** (06.02), with a targeted grader where possible.
6. **Repeat** after each release. The category distribution itself is a metric.

| Failure category (example) | Count /100 | Severity (1–3) | Component | Fix | New eval cases |
|---|---|---|---|---|---|
| Cited superseded statute version | 14 | 3 | Retrieval (temporal) | As-of filtering (04.02 §5) | 40 time-sensitive questions |
| Planner dropped bill ID on follow-up | 9 | 2 | Query planner | `validate_plan` ID restoration (04.04 §5) | 25 multi-turn cases |
| Answered despite no evidence | 7 | 3 | Generation / gate | Calibrated abstention gate (04.03 §5) | 30 unanswerable questions |
| Tool timeout → hallucinated status | 4 | 3 | Tool runtime / prompt | Error-aware prompting; is_error handling | 10 fault-injected cases |

```python
from collections import Counter


def prioritise(codes: list[dict]) -> list[dict]:
    """codes: [{'trace_id', 'category', 'severity' (1-3), 'component'}] from axial coding."""
    freq = Counter(c["category"] for c in codes)
    sev = {}
    for c in codes:
        sev[c["category"]] = max(sev.get(c["category"], 0), c["severity"])
    comp = {c["category"]: c["component"] for c in codes}
    n = len({c["trace_id"] for c in codes}) or 1
    rows = [{"category": k, "share": round(v / n, 3), "severity": sev[k], "component": comp[k],
             "priority": round(v / n * sev[k], 3)} for k, v in freq.items()]
    return sorted(rows, key=lambda r: -r["priority"])
```

**Automation, carefully:** LLMs can *propose* category assignments for new failing traces once the taxonomy exists, and cluster free-text notes. Humans must own the taxonomy, and must review new or ambiguous cases. Otherwise you optimise for the model's idea of what's wrong.

---

## 6. Debugging Playbook by Symptom

| Symptom | First look in the trace | Typical root causes |
|---|---|---|
| Wrong but confident answer | Context manifest: was the evidence there? | Retrieval miss (chunking, filters, versioning) vs generation ignoring evidence |
| Hallucinated citation | Cited IDs vs manifest IDs | Missing citation validation; IDs not given to the model |
| Slow responses | Span durations; retries; cache_read tokens | Cold prompt cache; long contexts; tool latency; reranker depth |
| Cost spike | Tokens per span over time | Context bloat from tool results; loops; cache invalidation (02.02 §6) |
| Agent loops | Repeated `execute_tool` spans with identical argument hashes | Missing loop detection; unhelpful tool errors (05.01, 05.02) |
| Refusal spike | Model and prompt versions; `finish_reasons` | Provider model update; a policy prompt edit |
| Inconsistent answers | Same input, different traces | Temperature; retrieval ties; batch non-determinism (01.02 §10) |
| "Works in dev, fails in prod" | Version attributes; filters hash; ACL groups | Index or version skew; ACL filters (03.03) |

---

## 7. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Traces without context** | You can see latency, but not *why* the answer was wrong | Version attributes, context manifests, decision attributes (§2.2) |
| **PII in observability stores** | Compliance risk | Redaction before export; opt-in content capture; restricted stores; retention tiers |
| **Storage cost** | Tracing bills rival inference | Tail sampling; content in blob storage keyed by trace; tiered retention |
| **Broken traces across services** | Orphaned spans; gaps at queues | `traceparent` in HTTP, Kafka, MCP `_meta`, and A2A metadata; instrument the consumers |
| **Can't reproduce failures** | "Flaky" bugs never fixed | Cassettes; pinned versions; event-sourced forks |
| **Unstructured bug triage** | Fixing anecdotes | Error analysis with coded categories and priorities |
| **Convention churn** | Attribute names change between versions | Pin the semconv version; map it in the collector; keep your own `app.*` attributes stable |

---

## 8. Hands-On Projects

### Project 1 — End-to-End OpenTelemetry Instrumentation Across Java and Python

**User stories**
- *As an SRE* for the legislative assistant, I want one trace per user request spanning the Spring Boot API, the Python agent service, the MCP servers, and Kafka-driven ingestion, with GenAI attributes and no PII leaks.

**Acceptance criteria**
1. The Spring Boot API uses the OpenTelemetry Java agent plus Spring AI observations bridged to OTel. The Python agent uses the OTel SDK with GenAI conventions (`chat_span`, `tool_span`, retrieval and agent spans) and the custom `app.*` attributes (§2.2).
2. Propagation works across HTTP, Kafka headers, and MCP `_meta`. A trace-completeness test confirms no orphaned spans over 1,000 requests.
3. Redaction runs in-process (`redact`) and in the Collector (a transform processor). A PII test suite of seeded emails, phones, and IDs shows zero leaks in the exported data.
4. A Collector with tail sampling (errors, slow, negative feedback, a uniform 2%) → Tempo or Jaeger for traces, plus Langfuse or Phoenix for LLM-specific views. Cost per 1M requests is estimated.
5. Dashboards: tokens and cost per span type, cache-hit ratio, tool error rates, and latency breakdowns per route.

**Step-by-step**
1. Deploy the Collector (Kubernetes) with OTLP receivers, the transform/redaction processors, tail sampling, and the exporters.
2. Instrument the Java service (agent + Spring AI observation config + custom attributes).
3. Instrument the Python agent and MCP servers; propagate context through the MCP `_meta` and the Kafka headers.
4. Write the completeness and PII tests; run the load tests.
5. Build the dashboards; document the conventions (a semconv version pin and the `app.*` attribute registry).

---

### Project 2 — Trace → Replay → Regression Test Pipeline

**User stories**
- *As a developer*, when a user reports a bad answer, I want to turn its trace into a reproducible test in minutes, verify my fix against it, and keep it in the regression suite forever.

**Acceptance criteria**
1. All model, retrieval, and tool calls go through `Cassette` (record mode in production for sampled traces, with redaction). Cassettes are stored alongside the traces.
2. A CLI takes `trace_id` → exports an `EvalCase` (06.01) with the input, the context, and the expected-behaviour annotation, plus its cassette.
3. Replay modes: full, partial (swap the prompt or model version, the rest recorded), and fork-from-step (with the 05.03 event store). Cassette misses are reported clearly.
4. `diff_traces` output is shown in the PR comment when a regression case changes behaviour.
5. A demo on 10 real or realistic incidents: each is reproduced, fixed, and added to the suite, and the fixes are verified in CI.

**Step-by-step**
1. Wrap the LLM, retriever, and tool clients with cassette calls; add the sampling-aware recording.
2. Build the export CLI (pull the trace + cassette → write the case + fixtures into the dataset repository).
3. Implement the replay runners and the diffing.
4. Integrate them with the eval CI (06.01 §6).
5. Run the incident drill, and document the workflow.

---

### Project 3 — Error-Analysis Program with Measured Impact

**User stories**
- *As the AI product lead*, I want a repeatable process that tells us, every release, which failure modes matter most, and proves that fixing them improved the system.

**Acceptance criteria**
1. Each cycle: sample 100 traces (70% failure-enriched, 30% random), open-code and axial-code them with ≥ 2 reviewers, and measure agreement on the category assignment (κ).
2. Maintain a failure taxonomy (versioned), with definitions and examples; `prioritise` output per cycle.
3. For the top 3 categories: implement fixes, add ≥ 20 eval cases each (06.02), and create targeted graders (deterministic where possible).
4. Measure the impact: category frequency in the next cycle's sample, targeted-grader pass rates, and online metrics (06.03) — with CIs.
5. After ≥ 3 cycles: a report showing how the taxonomy evolved, the fixes, and the effect sizes; an LLM-assisted pre-coding pilot validated against human coding.

**Step-by-step**
1. Build a trace-review UI (a Langfuse/Phoenix annotation queue, or a small internal app) with note-taking and category tagging.
2. Run the first cycle; build the taxonomy from the notes.
3. Implement the fixes, the cases, and the graders; ship them.
4. Repeat for 2 more cycles; measure.
5. Pilot LLM pre-coding; measure agreement with humans; decide how to use it.

---

## 9. Standards, Papers & Reading

- OpenTelemetry — *Semantic Conventions for Generative AI* (spans, metrics, events; agent and tool spans) and the attribute registry (`gen_ai.*`)
- OpenTelemetry Blog, May 2026 — *Inside the LLM Call: GenAI Observability with OpenTelemetry*
- W3C — *Trace Context* (Recommendation)
- Sigelman et al., 2010 — *Dapper, a Large-Scale Distributed Systems Tracing Infrastructure*
- Shankar et al., 2024 — *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences* (criteria drift; grounding evals in observed failures)
- Strauss & Corbin — *Basics of Qualitative Research* (open and axial coding; the basis of error-analysis practice)
- Cemri et al., 2025 — *Why Do Multi-Agent LLM Systems Fail?* (a trace-derived failure taxonomy)
- Barnett et al., 2024 — *Seven Failure Points When Engineering a Retrieval Augmented Generation System*
- MCP Specification 2026-07-28 — trace-context propagation in `_meta`

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **OpenTelemetry SDKs + Java agent + Collector** (tail sampling, transform processors) | Instrumentation, redaction, sampling, routing |
| **OpenLLMetry / OpenInference instrumentations** | Auto-instrumentation for LLM SDKs and frameworks |
| **Spring AI observability (Micrometer)** | JVM LLM, embedding, and tool observations |
| **Grafana Tempo / Jaeger** | Trace storage and search |
| **Langfuse / Arize Phoenix / LangSmith / Braintrust** | LLM-specific trace views, annotation queues, datasets from traces |
| **Presidio** | PII detection for redaction pipelines |
| **vcrpy-style cassettes / your own recorder** | Deterministic replay |
