# Capstone Project 09 — Monitor Production Systems

> **Deliverable:** A full monitoring stack — Prometheus + Grafana + Alertmanager + LangFuse (LLM observability) + structured logging — bolted onto a production AI service. When something breaks, you find out before users; when it doesn't break, dashboards tell you why the last deploy improved (or hurt) latency, cost, quality. Runbooks turn a 3 AM page into a checklist.
>
> **Time:** 6–8 hours.
>
> **What you'll be able to say:** "I set up the observability stack: RED metrics, LLM-specific metrics ($ per user, tokens by model, cache hit rate), SLO burn-rate alerts, structured logs with correlation IDs, tail-sampled traces to LangFuse, and one-page runbooks per alert. My on-call rotation handles pages by following the runbook, not by paging me."

---

## 1. Project Overview

The three pillars of observability, adapted for AI:

| Pillar | For AI, specifically |
|--------|---------------------|
| **Metrics** | RED (Rate, Errors, Duration) + LLM ($ per user, tokens/sec, cache hit rate, refusal rate) |
| **Logs** | Structured JSON with request_id + user_id_hash; PII-redacted; tail-sampled |
| **Traces** | End-to-end LLM call traces via LangFuse or LangSmith; `gen_ai.*` semantic conventions |

Plus:
- **SLOs** with error-budget-based alerting (multi-window multi-burn-rate)
- **Runbooks** wired to alerts
- **Dashboards** that answer "how is the service?" in one glance

### Architecture
![IMG-CAP09-01](/7%20-%20Projects/images/IMG-CAP09-01.jpg)

### Prerequisites

- The AI service to instrument (any of Capstones 04-06)
- Docker Compose for local Prometheus/Grafana
- A LangFuse account (free tier at cloud.langfuse.com, or self-host)
- Optional: PagerDuty free tier, Slack webhook

---

## 2. Step-by-Step Implementation

### Step 1 — Prometheus metrics (`app/metrics.py`)

```python
"""metrics.py — the four categories every AI service needs."""
from prometheus_client import Counter, Histogram, Gauge
import time
from contextlib import contextmanager

# --- HTTP / RED ---
REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "Request duration",
    ["method", "route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
INFLIGHT = Gauge("http_requests_inflight", "In-flight requests", ["route"])

# --- LLM-specific ---
LLM_TOKENS_IN = Counter("llm_tokens_input_total", "Input tokens", ["model", "route"])
LLM_TOKENS_OUT = Counter("llm_tokens_output_total", "Output tokens", ["model", "route"])
LLM_TOKENS_CACHE_READ = Counter("llm_tokens_cache_read_total", "Cached-read tokens", ["model"])
LLM_COST = Counter("llm_cost_usd_total", "Cumulative LLM cost", ["model", "route"])
LLM_LATENCY = Histogram(
    "llm_request_duration_seconds", "LLM API call duration",
    ["model", "route"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
LLM_TTFT = Histogram(
    "llm_ttft_seconds", "Time to first token (streaming)",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)
LLM_REFUSALS = Counter("llm_refusals_total", "Model refusals", ["model", "route", "reason"])

# --- Cache ---
CACHE_HITS = Counter("cache_hits_total", "Cache hits", ["cache"])
CACHE_MISSES = Counter("cache_misses_total", "Cache misses", ["cache"])

# --- Rate limits ---
RATE_LIMITED = Counter("rate_limit_rejections_total", "429s emitted", ["dimension"])

# --- Errors ---
ERRORS = Counter("app_errors_total", "Application errors", ["kind"])

# --- Guardrails ---
GUARDRAIL_BLOCKS = Counter("guardrail_blocks_total", "Guardrail hits", ["direction", "rule"])

# --- Helpers ---
@contextmanager
def observe_http(method, route):
    INFLIGHT.labels(route=route).inc()
    start = time.perf_counter()
    status = 500
    try:
        yield lambda s: setattr_local(status_ref, s)
        # Note: real impl uses a middleware; see below
    finally:
        INFLIGHT.labels(route=route).dec()
        LATENCY.labels(method=method, route=route).observe(time.perf_counter() - start)

# Simpler: use as middleware (see Step 2)
```

### Step 2 — FastAPI middleware

```python
"""app.py"""
import time
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app
from .metrics import REQUESTS, LATENCY, INFLIGHT, ERRORS

app = FastAPI()

@app.middleware("http")
async def metrics_mw(request: Request, call_next):
    # Get route template (not raw path — avoids high cardinality)
    route = request.scope.get("route").path if request.scope.get("route") else request.url.path
    INFLIGHT.labels(route=route).inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        ERRORS.labels(kind="unhandled").inc()
        raise
    finally:
        elapsed = time.perf_counter() - start
        INFLIGHT.labels(route=route).dec()
        REQUESTS.labels(method=request.method, route=route, status=status).inc()
        LATENCY.labels(method=request.method, route=route).observe(elapsed)
    return response

app.mount("/metrics", make_asgi_app())
```

**Why route templates and not raw paths?** `/orders/12345` and `/orders/67890` would each be their own metric label — exploding cardinality. Route templates (`/orders/{id}`) keep it bounded.

**Why the try/finally?** So even on exceptions, we still record the request. Metrics that only work in the happy path lie during incidents.

### Step 3 — Record LLM-specific metrics at call site

```python
"""llm_client_instrumented.py"""
import time
from .metrics import LLM_TOKENS_IN, LLM_TOKENS_OUT, LLM_TOKENS_CACHE_READ, \
                     LLM_COST, LLM_LATENCY, LLM_TTFT, LLM_REFUSALS

PRICE = {
    "claude-sonnet-5": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "claude-haiku-4-5": {"input": 0.8, "output": 4.0, "cache_read": 0.08},
    "claude-opus-5": {"input": 15.0, "output": 75.0, "cache_read": 1.5},
}

async def instrumented_llm_call(client, route: str, **kwargs):
    model = kwargs.get("model", "claude-sonnet-5")
    start = time.perf_counter()
    resp = await client.messages.create(**kwargs)
    elapsed = time.perf_counter() - start

    # Metrics
    LLM_LATENCY.labels(model=model, route=route).observe(elapsed)
    LLM_TOKENS_IN.labels(model=model, route=route).inc(resp.usage.input_tokens)
    LLM_TOKENS_OUT.labels(model=model, route=route).inc(resp.usage.output_tokens)
    cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
    if cache_read:
        LLM_TOKENS_CACHE_READ.labels(model=model).inc(cache_read)

    p = PRICE.get(model, {"input": 0, "output": 0, "cache_read": 0})
    cost = (
        resp.usage.input_tokens * p["input"] +
        resp.usage.output_tokens * p["output"] +
        cache_read * p["cache_read"]
    ) / 1_000_000
    LLM_COST.labels(model=model, route=route).inc(cost)

    if resp.stop_reason == "refusal":
        LLM_REFUSALS.labels(model=model, route=route, reason="model").inc()

    return resp
```

### Step 4 — LangFuse tracing (`app/tracing.py`)

```python
"""tracing.py — attach trace attributes at every LLM call."""
from langfuse import Langfuse, observe
from langfuse.decorators import langfuse_context
import os

lf = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

@observe(name="chat_request")
async def handle_chat(user_id: str, question: str):
    langfuse_context.update_current_trace(
        user_id=user_id,
        session_id=...,
        tags=["prod", "v1.4.2"],
        metadata={"prompt_version": "v3.2"},
    )
    # ... nested calls ...
```

Langfuse's `@observe` creates the trace and nested spans; token counts and costs come from the LLM SDK wrapper. Traces appear in the LangFuse UI with waterfall views, replayable inputs, and attached scores.

### Step 5 — Structured logs with correlation IDs (`app/logging_setup.py`)

```python
"""logging_setup.py"""
import structlog, logging, re, uuid
from contextvars import ContextVar

log_ctx = ContextVar("log_ctx", default={})

PII_PATTERNS = {
    "ssn":   (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    "email": (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    "key":   (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[REDACTED_KEY]"),
    "card":  (re.compile(r"\b\d{13,19}\b"), "[REDACTED_CARD]"),
}

def redact_pii(_, __, event_dict):
    for k, v in list(event_dict.items()):
        if not isinstance(v, str):
            continue
        for _name, (pat, replace) in PII_PATTERNS.items():
            v = pat.sub(replace, v)
        event_dict[k] = v
    return event_dict

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_pii,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

log = structlog.get_logger()

# FastAPI middleware to attach request_id
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=rid)
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        structlog.contextvars.clear_contextvars()
        return response
```

**Why contextvars?** They flow through async boundaries automatically — no need to pass `request_id` explicitly through every function call. `log.info("...")` from deep in a call chain still emits with the correct `request_id`.

**Why PII redaction in the log pipeline itself?** Because any code path might log something with PII in it. Redaction at the sink is defense in depth beyond code-level care.

### Step 6 — SLO alert rules (`prometheus/alerts.yml`)

```yaml
groups:
- name: slo-chat-api
  interval: 30s
  rules:
    # Multi-window multi-burn-rate — Google SRE pattern
    - alert: ChatApiSLOBurnFast
      expr: |
        (
          sum(rate(http_requests_total{route="/chat",status=~"5.."}[2m]))
          /
          sum(rate(http_requests_total{route="/chat"}[2m]))
        ) > (14.4 * 0.001)
      for: 2m
      labels:
        severity: page
        service: chat-api
      annotations:
        summary: "Chat API burning error budget fast (2h window, 14.4x rate)"
        runbook: "https://runbooks.ourteam.dev/chat-api-errors"
        dashboard: "https://grafana.ourteam.dev/d/chat-api"

    - alert: ChatApiSLOBurnSlow
      expr: |
        (
          sum(rate(http_requests_total{route="/chat",status=~"5.."}[6h]))
          /
          sum(rate(http_requests_total{route="/chat"}[6h]))
        ) > (6 * 0.001)
      for: 15m
      labels:
        severity: ticket
        service: chat-api
      annotations:
        summary: "Chat API burning budget slowly (6h window)"

- name: llm-cost
  rules:
    - alert: LLMCostSpike15m
      expr: |
        (
          sum(rate(llm_cost_usd_total[15m])) * 3600
        ) > 3 * avg_over_time(
          (sum(rate(llm_cost_usd_total[15m])) * 3600)[24h:15m]
        )
      for: 5m
      labels:
        severity: page
      annotations:
        summary: "LLM cost/hour is 3x above 24h baseline"
        runbook: "https://runbooks.ourteam.dev/cost-spike"

- name: cache
  rules:
    - alert: CacheHitRateDropped
      expr: |
        (
          sum(rate(cache_hits_total[10m])) / (sum(rate(cache_hits_total[10m])) + sum(rate(cache_misses_total[10m])))
        ) < 0.3
      for: 15m
      labels:
        severity: ticket
      annotations:
        summary: "Cache hit rate below 30% for 15min"

- name: rate-limits
  rules:
    - alert: HighRateLimitRejections
      expr: |
        sum(rate(rate_limit_rejections_total[5m])) > 5
      for: 5m
      labels:
        severity: warn
      annotations:
        summary: "Rate limits rejecting >5 rps sustained"
```

**Why multi-window multi-burn-rate?** A single 5-minute threshold either fires on short blips (noise) or misses long slow burns (real). Multiple windows at different burn thresholds catch both.

**Why runbook links in every alert annotation?** The on-call's first click reveals what to do. No hunting through wikis at 3 AM.

### Step 7 — Alertmanager routing (`alertmanager.yml`)

```yaml
route:
  receiver: default
  routes:
    - matchers: [severity="page"]
      receiver: pagerduty
    - matchers: [severity="ticket"]
      receiver: slack-tickets
    - matchers: [severity="warn"]
      receiver: slack-warnings

receivers:
  - name: default
    slack_configs:
      - api_url: "$SLACK_WEBHOOK"
  - name: pagerduty
    pagerduty_configs:
      - service_key: "$PAGERDUTY_KEY"
  - name: slack-tickets
    slack_configs:
      - api_url: "$SLACK_WEBHOOK"
        channel: "#tickets"
  - name: slack-warnings
    slack_configs:
      - api_url: "$SLACK_WEBHOOK"
        channel: "#warnings"

inhibit_rules:
  - source_matchers: [severity="page"]
    target_matchers: [severity="ticket"]
    equal: [service]
```

**Why inhibit rules?** When a page fires, silence lower-severity tickets for the same service — the on-call is already engaged. Reduces alert fatigue during incidents.

### Step 8 — Grafana dashboards

Build one dashboard with four rows (Module 7's pattern):

**Row 1 — RED**
- Requests per second by route
- Error rate (5xx / total)
- p50/p95/p99 latency histograms

**Row 2 — LLM**
- Tokens/sec by model
- $/hour rolling
- Prompt cache hit rate (`cache_read / total_input_tokens`)
- Refusal rate

**Row 3 — Cache / Cost / Rate limits**
- Cache hit rate by layer
- Top 10 spender users
- Rate-limit rejections/min

**Row 4 — SLO**
- Error rate vs budget (line + threshold)
- Latency SLO burn (multi-window)
- 30-day error budget consumed

Export as JSON committed to `dashboards/main.json`; version-control it. On any new environment, import once.

`[IMG-CAP09-02]` — *Prompt: A Grafana dashboard mockup with four rows of panels. Row 1: requests/sec line (bright), error rate line (orange), latency histogram heatmap (dark blue gradient). Row 2: tokens/sec line by model color-coded, $/hour big-number tile trending green, cache hit rate donut (65%), refusal rate percentage. Row 3: cache hit rate multi-line (in-process, Redis, semantic layers), top-10 spending users bar chart, rate-limit rejections stacked bar. Row 4: SLO burn rate multi-window overlay chart, 30-day error budget consumed as horizontal bar showing 34% used. Header shows service name, environment, and time-range selector. Grafana native dark UI aesthetic.*

### Step 9 — Runbooks

For every alert, a one-page runbook committed to the repo at `docs/runbooks/<alert-name>.md`:

```markdown
# ChatApiSLOBurnFast Runbook

## What this means
Error rate on /chat exceeded 14.4× the SLO burn rate over 2 min.
Left unchecked, we'd exhaust our monthly error budget in ~2 days.

## First checks (30 seconds each)
1. Anthropic status page — https://status.anthropic.com
2. Grafana dashboard — error rate by model (is it one model's fault?)
3. Fly.io status — https://status.flyio.net
4. Recent deploys — `fly releases` (was there a deploy in the last hour?)

## Likely causes (in order of probability)
1. **Upstream provider outage** — Anthropic 5xx. Wait ~2 min, if not recovering, fail over to backup provider or return degraded mode.
2. **Bad deploy** — a recent release broke a code path. Roll back: `fly releases rollback`.
3. **Downstream dependency** — Redis/Postgres unreachable. Check readiness endpoint, restart if needed.
4. **Traffic spike beyond capacity** — check RPS panel; scale out with `fly scale count 4`.

## If you need to escalate
Page @secondary-oncall in Slack #incident. Include:
- Which cause you ruled in/out
- What action you've taken
- Whether users are still affected

## Post-incident
Fill out incident review template within 24h.
```

### Step 10 — Tail-sampled traces

For cost control on trace ingestion, sample smartly. OTel Collector config snippet:

```yaml
processors:
  tail_sampling:
    decision_wait: 30s
    num_traces: 100000
    expected_new_traces_per_sec: 1000
    policies:
      - name: errors-always
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow-always
        type: latency
        latency: {threshold_ms: 5000}
      - name: baseline-sample
        type: probabilistic
        probabilistic: {sampling_percentage: 5}
```

Keeps: 100% of errors, 100% of slow, 5% baseline. Enough to diagnose without paying for full retention.

---

## 3. What Each Piece Buys You

| Piece | Question it answers |
|-------|--------------------|
| RED metrics | "Is the service healthy right now?" |
| LLM cost metrics | "Where is money going, and is it trending up?" |
| Cache metrics | "Is our caching config working?" |
| SLO burn alerts | "Should I be worried, or can this wait until Monday?" |
| Structured logs | "Show me everything that request touched" |
| LangFuse traces | "What exactly did the LLM see and produce for that user?" |
| Runbooks | "What do I DO at 3 AM?" |
| Dashboards | "How is the service, generally?" |

---

## 4. Common Pitfalls

**Pitfall #1 — Alerting on causes, not symptoms.** "CPU > 90%" pages during legitimate batch work; users unaffected. Alert on user-visible signals (error rate, latency SLO burn); keep resource metrics for diagnosis.

**Pitfall #2 — High-cardinality labels.** `user_id` as a label = one series per user = Prometheus falls over at scale. Keep dimensions bounded; put user-specific info in logs/traces.

**Pitfall #3 — Averages hide the tail.** Reporting mean latency looks fine while p99 users churn. Alert on percentiles, always.

**Pitfall #4 — No runbook = the on-call becomes the expert.** New team members can't handle pages. Runbooks encode institutional knowledge; every alert must link one.

**Pitfall #5 — Metrics without cost tracking.** LLM services can hemorrhage money silently. Cost must be a first-class SLI.

---

## 5. Extensions

- **Synthetic canaries.** A cron that runs a known-good request every minute and alerts if it errors or drifts.
- **Distributed tracing** across your own services with W3C traceparent.
- **Anomaly detection** — statistical or ML-based alerting for metrics without hard thresholds.
- **Chaos drills.** Deliberately break something monthly; measure how fast the on-call finds and fixes it.
- **Incident review pipeline.** Every P1 gets a written blameless post-mortem; findings feed back into runbooks and alerts.

---

## 6. Interview Talking Points

- **"How do you monitor an AI service?"** Three pillars — metrics (RED + LLM-specific), logs (structured, correlated), traces (LangFuse for LLM specifically); plus SLO discipline, cost as SLI, and runbooks per alert.
- **"What's your SLO?"** Concrete: "p99 latency of /chat < 2s over 30 days, 99.9% success rate." Then error-budget-driven release policy.
- **"What's multi-window multi-burn-rate?"** Fire different alerts at different (window, threshold) pairs so short fast burns and long slow burns both trigger appropriately. Reduces false positives and false negatives simultaneously.
- **"How do you avoid PII in logs?"** Structured logging with a redaction processor in the pipeline; regex + entity-recognition patterns; log the *fact* something happened, not the sensitive value.
- **"What happens when an alert fires?"** On-call gets paged with runbook link; runbook has 30-second first checks and ordered likely causes; they follow, escalate if needed, and file a post-mortem for anything user-impacting.
- **"How do you handle upstream (Anthropic) outages?"** Retry with backoff, circuit-breaker on sustained failures, fail-open to a degraded mode or backup provider if one is set up, alert operators.

---

## 7. References

- Google SRE Book (free online) — especially the SLO / error-budget / burn-rate chapters
- *Distributed Systems Observability* by Cindy Sridharan — foundational text
- Prometheus docs — recording rules, alerting rules
- Grafana docs — dashboards as code
- LangFuse docs — https://langfuse.com/docs
- Charity Majors — "Observability is not three pillars" blog; wider view of what actually matters
