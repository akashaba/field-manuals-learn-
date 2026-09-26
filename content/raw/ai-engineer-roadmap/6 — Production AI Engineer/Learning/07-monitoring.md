# 07 — Monitoring & Metrics

> **Module goal:** Master the discipline of turning a running AI service into a signal-rich system — one where you know *before your users do* that latency spiked, error rate climbed, GPU utilization tanked, or token cost per request tripled. Learn the Prometheus/OpenTelemetry stack, the RED and USE methods, SLOs, alerting, and the metric taxonomy specific to LLM/ML workloads.

---

## 1. Executive Summary & Core Concepts

**Monitoring** is the practice of collecting, storing, and alerting on **numeric time-series measurements** (metrics) from your systems. Logs tell you *what happened*, traces tell you *why a request was slow*, metrics tell you *how the system is behaving right now, aggregated*.

Three signal types (the "three pillars of observability"):

| Signal | What it is | Cardinality | Cost | Best for |
|--------|-----------|-------------|------|----------|
| **Metrics** | Numeric time series (counter, gauge, histogram) | Low (~thousands of series) | Cheap | Dashboards, alerts, capacity planning |
| **Logs** | Discrete events with structured fields | High (per event) | Expensive | Debugging specific requests, audit |
| **Traces** | Causal chain of spans across services | Very high | Very expensive | Distributed root-cause analysis |

**RED method** (for request-driven services):
- **Rate** — requests per second
- **Errors** — error rate (or ratio)
- **Duration** — request latency distribution

**USE method** (for resources):
- **Utilization** — % of time resource is busy
- **Saturation** — queue depth / backlog
- **Errors** — resource error events

**SLI/SLO/SLA:**
- **SLI** (Service Level Indicator) — the metric (e.g., "p99 latency of `/chat` endpoint")
- **SLO** (Service Level Objective) — the target (e.g., "p99 < 2s over 30 days")
- **SLA** (Service Level Agreement) — the contract (e.g., "99.9% uptime or refund")
- **Error budget** — `1 - SLO` (e.g., 0.1% of requests can fail before you halt releases)

**Prometheus** is the de-facto open-source metrics stack: it **scrapes** an HTTP endpoint (`/metrics`) that your service exposes in text format. **Grafana** visualizes; **Alertmanager** pages.

**OpenTelemetry (OTel)** is the emerging standard for *all three* signals — a vendor-neutral SDK + wire protocol (OTLP). Modern practice: instrument once with OTel, export to whichever backend (Prometheus, Datadog, Honeycomb, Grafana Cloud).

For **AI services specifically**, RED/USE isn't enough. Add:
- **Tokens per second** (throughput), **tokens per request** (cost driver)
- **Model latency percentiles** by model/route (a Haiku call and an Opus call have very different distributions)
- **GPU utilization, VRAM, memory bandwidth** (for self-hosted)
- **Cache hit rate**, **retrieval hit rate** (for RAG)
- **Cost per request** (rolled up from token counters × price table)
- **Guardrail block rate**, **refusal rate** — safety and product signals

The rule: **if a metric is not tied to a decision** (alert threshold, capacity plan, product KPI), it's noise. Cardinality is expensive; each label combination is a new time series.

---

## 2. Deep-Dive Breakdown

### 2.1 Metric Types & Prometheus Fundamentals

**Four metric types:**

| Type | What it does | Example |
|------|-------------|---------|
| **Counter** | Monotonically increasing (only resets on process restart) | `http_requests_total` |
| **Gauge** | Value that can go up or down | `queue_depth`, `gpu_memory_bytes` |
| **Histogram** | Samples observations into pre-defined buckets, exposes `_bucket`, `_sum`, `_count` | `http_request_duration_seconds` |
| **Summary** | Client-side quantiles (φ 0.5, 0.95, 0.99) | Less common; can't aggregate across instances |

**Prefer histograms over summaries** in distributed systems — you can aggregate histograms across replicas server-side (`histogram_quantile()` in PromQL).

**Python instrumentation example:**

```python
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from fastapi import FastAPI, Request
import time

# Define metrics ONCE at module level
REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
INFLIGHT = Gauge(
    "http_requests_inflight",
    "In-flight requests",
    ["route"],
)

# LLM-specific
TOKENS_IN = Counter("llm_tokens_input_total", "Input tokens", ["model", "route"])
TOKENS_OUT = Counter("llm_tokens_output_total", "Output tokens", ["model", "route"])
LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM API call latency",
    ["model", "route"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

app = FastAPI()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    route = request.url.path  # NOTE: high cardinality risk; use route templates
    INFLIGHT.labels(route=route).inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(method=request.method, route=route, status=status).inc()
        LATENCY.labels(method=request.method, route=route).observe(elapsed)
        INFLIGHT.labels(route=route).dec()
    return response

# Expose /metrics
app.mount("/metrics", make_asgi_app())
```

**Cardinality caution**: never label by `user_id`, `session_id`, raw path with IDs (`/orders/12345`), or free-form input. Use **route templates** (`/orders/{id}`). Every unique label combination is a new time series stored forever.

Rule of thumb: **keep total series per service under ~10k**. A service with `route=100 × status=5 × method=4` = 2000 combinations for HTTP alone is already substantial.

### 2.2 PromQL — Query Language Essentials

Prometheus stores time series; PromQL queries them. The core operations:

**Rate (counter → per-second rate):**
```promql
rate(http_requests_total[5m])
```

**Error ratio:**
```promql
sum(rate(http_requests_total{status=~"5.."}[5m]))
  /
sum(rate(http_requests_total[5m]))
```

**p99 latency from histogram:**
```promql
histogram_quantile(
  0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, route)
)
```

**Tokens per second (cost-driver):**
```promql
sum(rate(llm_tokens_output_total[5m])) by (model)
```

**SLO burn rate** (multi-window multi-burn-rate is the Google SRE pattern):
```promql
(
  sum(rate(http_requests_total{status=~"5.."}[1h]))
    /
  sum(rate(http_requests_total[1h]))
) > 14.4 * (1 - 0.999)  # burning 30-day budget in ~2 days
```

`14.4` means burning the entire monthly error budget in 2 days; you page immediately. Slower burn rates page at lower urgency.

### 2.3 OpenTelemetry — The Unification

OTel decouples **instrumentation** (SDK in your code) from **collection** (Collector) from **backend** (Prometheus, Datadog, Jaeger, whatever). You write OTel calls; a config change swaps backends.

```python
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="http://otel-collector:4317"),
    export_interval_millis=10_000,
)
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))

meter = metrics.get_meter("my-ai-service")
llm_latency = meter.create_histogram(
    "llm.request.duration",
    unit="s",
    description="LLM API call latency",
)

# Later, in a request handler:
llm_latency.record(elapsed, attributes={"model": "claude-sonnet-5", "route": "/chat"})
```

**Auto-instrumentation** exists for FastAPI, httpx, SQLAlchemy, Redis, boto3 — you install one package and get request spans, DB spans, HTTP client spans automatically.

**Semantic conventions** (`gen_ai.*` namespace) are being standardized for LLM workloads:
- `gen_ai.system` (e.g., `"anthropic"`)
- `gen_ai.request.model` (e.g., `"claude-opus-5"`)
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- `gen_ai.response.finish_reasons`

Following conventions means dashboards, alerts, and cost tooling work across services without rewiring.

### 2.4 SLOs, Error Budgets, and Alerting

**SLO discipline** turns "we should try to be reliable" into an operational contract. Steps:

1. **Pick user-visible SLIs.** For a chat API: `latency < 5s p95` and `success rate > 99.5%`. Not "CPU < 80%" — that's an internal proxy.
2. **Set SLO from real user tolerance**, not arbitrary. 99.9% (three nines) allows 43 min/month downtime; 99.99% (four nines) allows 4 min/month and is *expensive*.
3. **Define error budget.** `(1 - SLO) × total_requests`.
4. **Alert on burn rate**, not on the raw SLI. Google's multi-window multi-burn-rate:
   - **Fast burn** (2h window, 14.4× rate) → page immediately
   - **Slow burn** (6h window, 6× rate) → ticket during business hours

**Alert design principles:**

- **Every alert must be actionable.** If the on-call can't do anything about it, it's an FYI, not an alert.
- **Alert on symptoms, not causes.** "p99 > 5s" (user-visible) beats "GC pause > 200ms" (internal, may not affect users).
- **Aggregate below the human threshold.** Don't page for every 5xx — page when error rate crosses a burn-rate threshold.
- **Include runbook link in alert payload.** The on-call's first click should reveal what to do.

Example Prometheus alerting rule:

```yaml
groups:
- name: slo-chat-api
  rules:
  - alert: ChatApiHighErrorRateFastBurn
    expr: |
      (
        sum(rate(http_requests_total{route="/chat",status=~"5.."}[2m]))
        /
        sum(rate(http_requests_total{route="/chat"}[2m]))
      ) > (14.4 * 0.001)
    for: 2m
    labels:
      severity: page
    annotations:
      summary: "Chat API burning error budget fast (2h window, 14.4x rate)"
      runbook: "https://runbooks.internal/chat-api-errors"
```

### 2.5 AI-Specific Metrics

Standard RED/USE misses the metrics that matter for LLM/ML services:

**Cost metrics (the CFO's favorite):**
```python
COST_USD = Counter(
    "llm_cost_usd_total",
    "Cumulative LLM cost in USD",
    ["model", "route"],
)

# After each call:
input_cost = input_tokens * PRICE_TABLE[model]["input"]  # per token
output_cost = output_tokens * PRICE_TABLE[model]["output"]
COST_USD.labels(model=model, route=route).inc(input_cost + output_cost)
```

Then in PromQL:
```promql
sum(rate(llm_cost_usd_total[1h])) * 3600  # cost per hour
sum(increase(llm_cost_usd_total[30d]))    # monthly cost by model
```

**Quality / product metrics:**
- `llm_refusals_total{model, route}` — model declined to answer
- `guardrail_blocks_total{rule, direction}` — input/output filter fired
- `retrieval_hits_total{k_hit}` — how often did retrieval return relevant chunks (needs offline eval to label)
- `hallucination_rate` — from LLM-as-judge or user feedback, sampled

**Model quality drift:**
- **Data drift** — input distribution shift (input length, language mix)
- **Concept drift** — the input-output relationship changed (customer intent evolved)
Emit as gauges from a scheduled evaluation job, alert if beyond baseline.

**GPU metrics (self-hosted):**
- `nvidia_gpu_utilization_percent`
- `nvidia_gpu_memory_used_bytes`
- `nvidia_gpu_temperature_celsius`
- Scraped via **DCGM Exporter** (NVIDIA's Prometheus exporter)

`[IMG-07-01]` — *Prompt: A dashboard mockup showing four rows of graphs for an AI service: top row RED metrics (request rate, error rate, p50/p95/p99 latency lines); second row LLM-specific (tokens/sec by model, cost/hour trend line, cache hit rate donut, refusal rate); third row USE (CPU%, memory%, queue depth); fourth row GPU (utilization, VRAM used, temperature). Dark theme, teal and orange accents, Grafana-style. Widgets labeled clearly.*

---

## 3. Mental Models & Analogies

### Model 1: The Aircraft Cockpit

An airline pilot doesn't watch every wire in the plane. They watch a fixed set of instruments — altitude, airspeed, heading, fuel, engine RPM, warning lights — chosen because they summarize the whole aircraft's state in a way the pilot can act on. Everything else is data the maintenance crew examines *after* landing (that's logs).

Your metrics dashboard is your cockpit. It should fit on one screen and answer: *"Is this thing flying?"* The temptation is to add more gauges until the panel is unreadable. Resist. Every metric should map to a decision — "if this crosses X, I do Y." If you can't finish that sentence, the metric doesn't belong on the cockpit; it belongs in the flight data recorder (logs/traces you look at post-hoc).

### Model 2: The Doctor's Vital Signs vs. the Lab Report

Vital signs — pulse, blood pressure, temperature, respiratory rate, oxygen — are **continuous, cheap to measure, and low-cardinality.** A nurse reads them every hour, plots them on a chart, and if pulse crosses 130 or oxygen drops below 90, alarms fire. That's your **metrics** system.

Lab reports — blood chemistry, biopsy, MRI — are **discrete, expensive, and high-cardinality.** You order one when the vitals suggest something is wrong, and you dig deep into the results. That's your **logs and traces** system.

Vitals never diagnose the disease. They tell you *something is wrong* and *where to look*. You then pull the labs (a specific trace, a specific request's log) to find the cause. Building an AI service without this two-tier structure — trying to alert on log lines directly, or diagnosing from Prometheus gauges alone — is like a hospital that either only takes vitals (misses the diagnosis) or does an MRI on every patient (bankrupt).

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Averages hide the users you're failing.**
Reporting mean latency looks great: "our p50 is 400ms!" But the users who *churn* are the ones in the tail — the 1% seeing 30-second waits. **Always alert and dashboard on p95 or p99**, never on mean. Averaging out latency is like averaging your income with Bill Gates — the summary statistic is meaningless.

Corollary: never take the **average of a percentile across replicas.** `avg(p99_by_instance)` is mathematically nonsense. Aggregate the raw histogram (`sum(rate(_bucket))`), *then* compute the quantile once at the end.

**Pitfall #2 — Label explosion (a.k.a. "high cardinality kills Prometheus").**
Adding `user_id` as a label because "it'd be nice to filter" turns a service with 100k users into a service with 100k time series *per metric per label combination*. Prometheus falls over; your cloud metrics bill spikes. **Labels are for dimensions with small, bounded cardinality** (route, method, status, model, region). Anything user-specific belongs in logs or traces, joined at query time. Same for `request_id`, `session_id`, raw URLs with IDs, error messages with stack traces.

**Pitfall #3 — Alerting on causes instead of symptoms.**
"CPU > 90%" pages the on-call at 3 AM. They log in. CPU is at 90% because a nightly batch job is running as designed. Users are unaffected. That's a false page — trust in the alerting system erodes, and real pages get ignored. **Alert on user-visible symptoms** (error rate, latency SLO burn), and let internal metrics like CPU/memory be diagnostic when investigating. The exception is *leading indicators of imminent failure* — disk 95% full — where you page early because there's a fixed lag between symptom and outage.

Corollary pitfall: **paging on the same event multiple times.** If a single incident triggers 12 alerts across dependent services, on-call is drowning while trying to fix the actual issue. Configure alert grouping/inhibition in Alertmanager.

---

## 5. Self-Assessment Bank

**Q1 (MC):** Which metric type would you use for the number of currently connected WebSocket clients?
A) Counter
B) Gauge
C) Histogram
D) Summary

**Q2 (MC):** Why is `histogram_quantile()` preferred over the Summary type in Prometheus?
A) Summaries use more memory
B) Histograms can be aggregated across replicas server-side; summary quantiles cannot
C) Summaries are deprecated
D) Histograms are faster to scrape

**Q3 (short):** You define a metric `http_requests_total{method, route, status, user_id}` and your service has 50k users. What's wrong, and what should you do instead?

**Q4 (MC):** In SRE parlance, what is the "error budget" for a 99.9% availability SLO over 30 days?
A) 0.1% of that period's requests may fail (~43 minutes of downtime)
B) 1% of requests
C) 0.01% of requests
D) There is no budget; you must be 100% available

**Q5 (short):** Write a PromQL query for the p95 latency of the `/chat` route, over the last 5 minutes, grouped by model.

**Q6 (MC):** You get paged because CPU on one replica is at 92%. Users report no issues. What does this most likely indicate?
A) A real incident — investigate immediately
B) A false page — alerting on a cause (CPU) rather than a symptom (user-visible errors/latency)
C) A memory leak
D) A DDoS attack

**Q7 (short):** Name three AI-specific metrics beyond RED/USE that a chat API should emit, and explain what decision each supports.

**Q8 (MC):** What is the difference between an SLI and an SLO?
A) There is none; they're synonyms
B) SLI is the measurement; SLO is the target value for that measurement
C) SLI is external; SLO is internal
D) SLI is monthly; SLO is daily

**Q9 (short):** Explain the "multi-window multi-burn-rate" alerting pattern and why it beats a single-threshold alert on error rate.

**Q10 (MC):** Your team wants to instrument a service once and be able to swap backends (Prometheus, Datadog, Grafana Cloud) without touching code. Which project is designed for this?
A) Prometheus client libraries
B) OpenTelemetry
C) StatsD
D) Datadog Agent

---

### Answer Key

**A1: B (Gauge).** Currently connected clients can go up (new connect) and down (disconnect); Gauge is the type for values that fluctuate. A Counter only increases.

**A2: B.** Summary computes quantiles client-side, per instance — you cannot mathematically combine per-instance quantiles into a fleet-wide quantile. Histograms record raw buckets; the server aggregates buckets across instances with `sum(rate(_bucket))`, then `histogram_quantile()` computes the quantile from the aggregated buckets. Also (A) is false — summaries are typically cheaper CPU-wise but cost you aggregation capability.

**A3:** `user_id` is a high-cardinality label. Every unique user creates a new time series *per method × route × status combination*. With 50k users and even 10 route/method/status combinations, that's 500k time series for a single metric — Prometheus memory blows up and queries slow to a crawl. **Fix:** remove `user_id` from the metric. If you need per-user visibility for debugging, use structured logs or traces with `user_id` as a field/attribute; aggregate metrics stay low-cardinality. If you need per-user aggregate metrics (top 10 heaviest users), compute them in a separate batch job that emits pre-aggregated series (`top_users_requests_total{rank="1"}` etc.).

**A4: A.** 99.9% availability = 0.1% error budget. Over 30 days (43,200 minutes), that's ~43.2 minutes. If you exceed the budget mid-month, SRE convention is to halt feature releases and focus on reliability until the budget resets.

**A5:**
```promql
histogram_quantile(
  0.95,
  sum(rate(http_request_duration_seconds_bucket{route="/chat"}[5m])) by (le, model)
)
```
Key point: aggregate the *buckets* by `le` (bucket boundary) and `model` before computing the quantile; grouping by `le` is mandatory for `histogram_quantile` to work.

**A6: B.** CPU is a resource-level metric, not user-facing. It can be high because of legitimate work (batch, warm-up, model loading) without harming users. The alert is symptom-blind. Rewrite the alert around user-visible SLIs (error rate burning budget, or latency SLO breach). Keep CPU as a *dashboard* metric for post-alert investigation, not as a pager trigger — unless CPU saturation *causally* leads to imminent user harm (in which case set the threshold and window such that firing means real trouble is minutes away).

**A7:** Sample answers (any three):
- **Tokens per request (histogram)** — supports capacity planning and per-user cost analysis; alert if p99 crosses context-window limit.
- **Cost USD per request (histogram or derived counter)** — supports budget alerts and per-route profitability.
- **Cache hit rate (ratio of hits/(hits+misses))** — supports caching-layer tuning and cost reduction; alert if hit rate drops (invalidation bug).
- **Retrieval hit rate / top-k relevance (from offline eval or user thumbs-down)** — supports RAG quality tracking; alert on drop.
- **Refusal / guardrail block rate** — supports safety monitoring and product-side investigation (are we refusing legitimate queries?).
- **Model latency by model** — supports per-model SLO tracking (Opus is slower than Haiku; can't compare on one metric).

**A8: B.** SLI = *Service Level Indicator*, the metric definition ("p99 latency of `/chat`"). SLO = *Service Level Objective*, the target ("SLI < 2s over 30 days measured minute-by-minute"). SLI is what you measure; SLO is the promise you make about that measurement.

**A9:** A single alert like "error rate > 1% for 5 min" has two failure modes: too sensitive (paging on brief blips), or too slow (missing a fast-burning outage that eats the whole monthly budget in an hour). The **multi-window multi-burn-rate** pattern fires *multiple* alerts at different (window, threshold) pairs:
- Short window + high threshold → page immediately for outages that would exhaust the monthly budget in hours
- Long window + low threshold → ticket for slow, chronic degradation
Each is compared to the SLO error budget: burning it >14.4× fast means you'll exhaust it in ~2 days at that rate. This pattern minimizes both false positives (short blips filtered by the long-window alert requiring sustained badness) and false negatives (real outages caught quickly by the short window). Google SRE workbook has the canonical multi-burn-rate table.

**A10: B.** OpenTelemetry is explicitly a vendor-neutral SDK + wire protocol (OTLP). You instrument once with OTel; the Collector or SDK exporter sends to whatever backend you configure. Prometheus client libraries only speak Prometheus format; Datadog Agent is Datadog-specific; StatsD is a legacy protocol Datadog and others accept but doesn't unify traces/logs.

---

**Related modules:**
- `learning/01-fastapi.md` — where you'll add the `/metrics` endpoint and middleware
- `learning/06-logging.md` — the sibling pillar; where correlation IDs bridge metrics ↔ logs
- `learning/09-llm-observability.md` — LLM-specific tracing (LangSmith, LangFuse, Phoenix)
- `builds/01-production-hardening.md` — hands-on: bolt Prometheus + Grafana onto a service

**Practice prompts:**
1. Take your last build. Add three metrics: request counter, latency histogram, in-flight gauge. Expose `/metrics`, scrape from Docker-Compose'd Prometheus + Grafana. Build a dashboard with RED signals.
2. Write two Prometheus alerting rules for your service: one fast-burn SLO alert, one slow-burn. Include a runbook link stub.
3. For an LLM API service, define the label sets for `llm_request_duration_seconds` and `llm_tokens_output_total`. Justify each label and estimate total cardinality.

**References:**
- Google SRE Book & Workbook — SLO, error budget, multi-window burn-rate chapters
- *Prometheus: Up & Running* — Julius Volz (Prometheus co-creator)
- OpenTelemetry docs — https://opentelemetry.io/docs/
- Brendan Gregg on USE method — https://www.brendangregg.com/usemethod.html
- Tom Wilkie's RED method talk (Weaveworks / Grafana Labs)
