# Logging — Master Study Guide

> **Track:** Production AI Engineer · **Module:** 06
> **Prerequisites:** Modules 01–05.
> **Time budget:** ~5–7 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Logs are how you learn what your service did. Bad logs = 3am incidents with no clues. Good logs = grep for a request_id and you have the whole story. For AI services specifically:

- LLM calls are opaque — logs are your only view into what the model saw and said.
- Costs need explaining — "who spent $500 yesterday?" starts with logs.
- Debugging distributed systems needs **correlation IDs** to reconnect a single user's journey across services.
- Compliance and audit often mandate specific log content and retention.

Getting logging right is unglamorous but pays back forever. The gap between prose print-statement logging and structured, searchable, sampled logs is enormous.

**Fundamental principles you must own:**

1. **Structured, not prose.** JSON logs with fields, not human-friendly sentences.
2. **Correlation IDs** flow through every log across services.
3. **Levels matter.** DEBUG, INFO, WARN, ERROR, CRITICAL. Use them consistently.
4. **Never log secrets, PII, or full prompts by default.** Redact aggressively.
5. **Sampling for high-volume logs.** Log all errors; sample happy paths.
6. **Log to stdout in containers.** Aggregator handles the rest.
7. **Metrics ≠ logs.** Logs are events; metrics are aggregates. Use both.

If you retain nothing else: **structured JSON logs, correlation IDs, PII-safe, stdout-first. Every log line is a searchable event with fields you can filter on.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Structured Logging (structlog)

**Structured logs** are JSON objects with named fields — searchable, filterable, aggregatable.

**Bad (prose):**
```
2026-09-22 15:32:14 INFO User Brian made request to /predict which succeeded in 340ms with model version 1.2.3
```

**Good (JSON):**
```json
{
  "timestamp": "2026-09-22T15:32:14.123Z",
  "level": "info",
  "event": "request_completed",
  "user_id": "usr_brian",
  "path": "/predict",
  "method": "POST",
  "status": 200,
  "latency_ms": 340,
  "model_version": "1.2.3",
  "request_id": "req_a1b2c3",
  "cost_usd": 0.012
}
```

Prose you `grep`; JSON you `jq | filter | aggregate`. At scale, structured wins by orders of magnitude.

**Setup with `structlog`:**

```python
import logging
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()
log.info("model_loaded", model_version="1.2.3", latency_ms=1200)
```

Alternative: `python-json-logger` on top of stdlib logging. Both work; structlog has better context binding.

**Context binding.** Attach fields once, propagate everywhere:

```python
# In your request middleware
structlog.contextvars.bind_contextvars(
    request_id=request_id,
    user_id=user_id,
    endpoint=request.url.path,
)

# All log calls in this request context automatically include those fields
log.info("prediction_start")
log.info("tool_called", tool="search_web", query="...")
log.info("prediction_complete", latency_ms=340)
```

Every log entry carries `request_id`, `user_id`, `endpoint` without repeating them.

**Clear when done:**
```python
structlog.contextvars.clear_contextvars()
```

Or use `structlog.contextvars.bound_contextvars(...)` as a context manager for auto-cleanup.

---

### 2.2 Correlation IDs and Distributed Tracing

A `request_id` (also called `trace_id`, `correlation_id`) is the string that ties together every log line, span, and metric for a single user request as it flows through your stack.

**Generation.** At the edge (LB, API gateway, or your first middleware):

```python
import uuid
from fastapi import Request

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    structlog.contextvars.clear_contextvars()
    return response
```

**Propagation.** When calling downstream services, pass the same `x-request-id`:

```python
async with httpx.AsyncClient() as client:
    r = await client.post(
        url,
        json=data,
        headers={"x-request-id": request.state.request_id},
    )
```

Downstream services log using the same `request_id`. You can grep across services and reconstruct the full journey.

**W3C Trace Context (industry standard).** Use `traceparent` header:

```
traceparent: 00-{trace_id}-{span_id}-{trace_flags}
```

OpenTelemetry propagates this automatically. Enables cross-vendor tracing.

**IDs in error responses:**

```json
{"error": {"code": "internal_error", "message": "...", "request_id": "req_a1b2c3"}}
```

Users report `req_a1b2c3`; you `grep request_id=req_a1b2c3` → the full story.

**Nested tracing (spans).** For per-tool-call tracing within a request, OpenTelemetry gives you spans:

```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("tool_search_web") as span:
    span.set_attribute("query", query)
    result = search_web(query)
    span.set_attribute("num_results", len(result))
```

Traces render as flame charts in Datadog / Honeycomb / Jaeger. Beyond logs; complementary.

---

### 2.3 Log Levels, Sampling, and Retention

**Levels:**
- **DEBUG** — detailed diagnostics; devs only. Off in prod by default.
- **INFO** — normal operation. Requests, cache hits, tool calls.
- **WARN** — something odd but not fatal. Retry succeeded. Deprecation warnings.
- **ERROR** — request failed. Alert-worthy.
- **CRITICAL** — service-wide failure. Page someone.

Use them consistently. `WARN` for every request is not helpful; `INFO` for every 500 error is a bug.

**Configuration by env:**
```python
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")   # DEBUG in dev, INFO in prod
```

Change without redeploy via env var + rolling restart.

**Sampling.** High-traffic services: logging every request is expensive (storage + processing costs). Sample smartly:

**Head-based (random sampling):**
```python
if random.random() < 0.10:   # 10% of INFO logs
    log.info("request", ...)
```

**Tail-based (sample errors, sample slow):**
```python
if latency_ms > 5000 or status >= 500:
    log.info("slow_or_error_request", latency_ms=latency_ms, status=status)
elif random.random() < 0.10:
    log.info("normal_request", ...)
```

Errors and slow requests are always logged; normal ones sampled. Best of both.

**Metric-based sampling.** If you have Prometheus counters for all requests, you can afford to log only a sample.

**Retention.** Typical:
- **Hot storage (searchable):** 7–30 days.
- **Warm storage (archived, searchable slowly):** 30–90 days.
- **Cold storage (S3, rarely queried):** 90 days – 7 years for compliance.

Costs scale with volume × retention. Sample aggressively; retain aggregate metrics forever, raw logs by tier.

**Compliance.** GDPR, HIPAA, SOC 2 may require:
- Specific log retention periods (min or max).
- PII redaction (never store SSNs in logs).
- Access controls on log queries.
- Audit trail of who queried what.

---

### 2.4 What to Log (and What NOT to)

**Log — every request:**
- `request_id`
- `method`, `path`
- `status_code`
- `latency_ms`
- `user_id` (hashed if privacy-sensitive)
- `model` and `model_version`
- `input_tokens`, `output_tokens`, `cost_usd` (for LLM calls)
- `error` (if any)
- `cache_hit` (for cache diagnostics)

**Log — significant events:**
- Startup / shutdown / config changes.
- Auth failures.
- Rate-limit rejections.
- Model reloads / updates.
- Cache invalidations.
- Circuit-breaker trips.
- Graceful degradation events.

**NEVER log:**
- Passwords, API keys, JWT tokens.
- Full credit card / SSN / payment info.
- Health data (HIPAA).
- Government IDs.
- Full user PII in identifiable form (use hashed IDs).
- **Full LLM prompts by default** — may contain user PII. Log a **hash** or **truncated + redacted** version.
- **Full model responses by default** — same reason.

**Redaction strategies:**

```python
import re

PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),           # SSN
    (re.compile(r"\b\d{16}\b"), "[CC]"),                        # credit card
    (re.compile(r"[\w.-]+@[\w.-]+"), "[EMAIL]"),                # email
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "[APIKEY]"),         # API keys
]

def scrub(text: str) -> str:
    for pat, replacement in PII_PATTERNS:
        text = pat.sub(replacement, text)
    return text
```

Use libraries for robust cases: `presidio` (Microsoft), `scrubadub`, `spacy`-based detectors.

**Sampling of prompt/response content.** Even redacted, storing full LLM I/O for every request is expensive. Sample:
- All error paths (help debugging).
- 1–5% of successful paths (sanity check quality).
- Any request flagged by a user (thumbs down).

Store elsewhere (LangSmith / Phoenix / Braintrust) rather than in your general log stream — better UI for LLM-specific inspection.

---

### 2.5 Log Aggregation, Search, and Alerts

**Container logs go to stdout.** Never log to files inside containers. The container runtime captures stdout/stderr; the log aggregator picks it up.

**Aggregation stack:**

- **Datadog** — hosted; full-service; expensive at scale.
- **Splunk** — enterprise; powerful; also expensive.
- **Elastic Stack (ELK)** — Elasticsearch + Logstash + Kibana. Self-host or Elastic Cloud.
- **Grafana Loki** — cheap; built for high-volume, structured logs; integrates with Grafana / Prometheus.
- **AWS CloudWatch Logs** — native AWS.
- **GCP Cloud Logging** — native GCP.
- **Better Stack, Axiom, Logtail, Papertrail** — smaller SaaS.

For solo / small team: Datadog trial, Grafana Loki, or CloudWatch (if on AWS). Consolidate later.

**Search patterns you'll use often:**
- All errors in the last hour: `level:error @timestamp>=now-1h`.
- All requests for a user: `user_id:"usr_brian" @timestamp>=now-24h`.
- All requests for a request_id: `request_id:"req_a1b2c3"`.
- Slow requests: `latency_ms:>3000 status:200`.
- Cost anomalies: `endpoint:"/predict" @timestamp>=now-1h | stats sum(cost_usd) by user_id | sort desc`.

**Alerting on logs.** Many aggregators support log-based alerts:
- Error rate > threshold → PagerDuty.
- Auth failure spike → security team.
- Cost per user > cap → email user + operator.

**Prefer metrics-based alerts.** Log-based alerts are slower and noisier. Aggregate structured logs into metrics (Prometheus) and alert on metrics. Logs are for investigation; metrics are for detection.

**Trace vs log.** A **log** is a discrete event (one line). A **trace** is a set of causally-linked spans (a distributed execution). Both matter:
- Logs: what happened, at what time, with what fields.
- Traces: how did this request flow through services, where did time go.

OpenTelemetry unifies both; export logs and traces to the same backend (Datadog, Honeycomb, Grafana Tempo).

---

## 3. Mental Models & Analogies

### 3.1 The "Airport Flight Log" Model

An airport keeps a log of every takeoff, landing, gate change, delay. Each entry has: flight number (correlation ID), timestamp, origin, destination, status, gate. When something goes wrong ("why was my flight 3 hours late?"), you look up the flight number and reconstruct the journey.

Your logs work the same way: `request_id` = flight number; every service / tool / DB call adds a line; when things go wrong, follow the ID.

Prose logs are like an airport writing "Yeah, Flight 447 had some issues, was late a bit, gate changed. Not sure what happened." Useless. Structured logs are "Flight 447: departed 09:15, delayed 30m at 09:45 (weather), gate changed A12→B7 at 10:00, boarded 10:30." Actionable.

### 3.2 The "Symptom vs Diagnosis" Model

Metrics show **symptoms** — error rate is spiking, latency is up, cost is climbing.

Logs show **diagnoses** — which requests failed, why, from which user, with what stack trace.

You need both. Symptoms alert; diagnoses explain. Building on-call around symptoms without logs = "the fever is up, no idea why." Investing in logs without alerts = "we have great diagnostic tools; nobody noticed the patient is sick."

Wire them together: metrics fire alerts; alerts include links to log queries scoped to the same time window; on-call gets a symptom + a starting point for diagnosis in one page.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "print() Is Fine for Logging"

`print()` goes to stdout (good in containers) but has no level, no timestamp, no fields, no structured aggregation. Every log tool wants JSON with metadata. `print` at scale is a wall of unsearchable text. Structured `structlog` is 5 lines of setup and pays back forever.

### 4.2 "Log Everything — Storage Is Cheap"

At scale, storage isn't cheap. 1M requests/day × 5 KB per log entry × 30 days = 150 GB. At Datadog pricing (~$1.70/GB/month), that's $250/month just for storage. Multiply by services. Sample aggressively; log errors and slow paths always, healthy paths at 5–10%.

### 4.3 "Logging User Prompts Is Fine for Debugging"

User prompts often contain PII, internal secrets, or things users don't want stored. Default: hash the prompt for correlation; log only a truncated/redacted preview. Store full prompts (redacted) only for sampled cases and in a separate LLM-observability tool with tighter access control. Compliance teams will thank you.

---

## 5. Self-Assessment Bank (Logging)

### Questions

**Q1 (Short answer).** In one paragraph, why structured (JSON) logs beat prose logs at scale.

**Q2 (Multiple choice).** For a request that flows across 3 services, the correlation ID:
- (a) Is unnecessary.
- (b) Is generated per-service.
- (c) Is generated at the edge and propagated as `x-request-id` (or W3C `traceparent`) through every downstream call.
- (d) Is stored in the DB.

**Q3 (Short answer).** Give three fields you'd log on every LLM API call.

**Q4 (Multiple choice).** In containerized environments, application logs should go to:
- (a) Files inside the container.
- (b) stdout/stderr — captured by the runtime and shipped to the aggregator.
- (c) A remote FTP server.
- (d) A local database.

**Q5 (Short answer).** Explain tail-based sampling for logs.

**Q6 (Multiple choice).** Full user prompts sent to an LLM should be:
- (a) Logged verbatim always.
- (b) Not logged in general request logs; sampled and redacted before storage; kept in a separate LLM-observability system with access controls.
- (c) Encrypted.
- (d) Discarded entirely.

**Q7 (Short answer).** Contrast metrics and logs. When would you alert on which?

**Q8 (Multiple choice).** structlog's `bind_contextvars` lets you:
- (a) Log to multiple files.
- (b) Attach fields once (per request) that automatically appear in all subsequent log calls in that context.
- (c) Speed up logging.
- (d) Rotate log files.

**Q9 (Short answer).** For a request that failed, the user reports `request_id=req_a1b2c3`. What's your diagnostic query, and what would you expect to find?

**Q10 (Multiple choice).** Log retention for a healthy production system typically follows:
- (a) Log everything forever.
- (b) Hot (searchable): 7–30 days; warm: 30–90 days; cold (S3): longer for compliance; aggregate metrics kept indefinitely.
- (c) Delete after 24 hours.
- (d) Only keep errors.

---

### Answer Key & Detailed Explanations

**A1.** Prose logs are searched by regex over free-form text — fragile, slow, hard to aggregate. Structured (JSON) logs have named fields; you filter by `user_id`, `status`, `latency_ms`, and aggregate (`sum(cost_usd) by user_id`) with SQL-like precision. At any real scale, structured is the only sane way to search and alert. Bonus: structured logs feed metrics and dashboards; prose can't.

**A2. (c).** Generate at the edge (LB, API gateway, or first middleware); accept an incoming `x-request-id` if provided; otherwise mint a UUID. Propagate to all downstream calls via header. Every log line across services carries the same ID; you can grep across services and reconstruct the whole journey. OpenTelemetry's W3C `traceparent` is the industry standard for this.

**A3.** Any three of: `request_id`, `user_id`, `model` (e.g., "claude-..."), `model_version`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`, `ttft_ms`, `cache_hit`, `finish_reason`, `endpoint`. Feeds cost dashboards, quality tracking, per-user analysis.

**A4. (b).** Containers should log to stdout/stderr; the container runtime (Docker, containerd) captures those and the log aggregator (Datadog agent, Fluent Bit, Loki Promtail) ships them. File-based logging in containers doesn't survive restarts and defeats the aggregation pattern.

**A5.** Tail-based sampling decides whether to log based on the *outcome* of the request, not randomly. Always log errors (5xx, exceptions) and slow requests (>5s). Sample happy paths at 5–10%. You catch every problem plus a representative slice of normal traffic — best signal-to-cost ratio. Head-based sampling (random uniformly) misses errors under low sampling rates.

**A6. (b).** User prompts often contain PII, confidential info, or content users didn't consent to have stored. General logs (in Datadog, Splunk) may have broad access. Sample redacted prompts for quality; store full (redacted) samples in an LLM-observability tool (LangSmith, Braintrust, Phoenix) with tight access controls. Compliance and privacy demand this discipline.

**A7.** **Metrics** are aggregate numeric time-series (error rate, p95 latency, cost per hour). Efficient to compute, alert on, and dashboard. **Logs** are discrete events with full context. Slow to aggregate but rich in detail. **Alert on metrics** (fast, cheap, precise). **Investigate with logs** (rich, per-request). A well-tuned alert links to a log query for follow-up.

**A8. (b).** `bind_contextvars` attaches fields to the current async / thread context; every subsequent log call within that context includes those fields automatically. Standard pattern: bind `request_id`, `user_id`, `endpoint` in middleware; all handler logs inherit them without repeating.

**A9.** Query: `request_id:"req_a1b2c3"` in your log aggregator, no time filter (or wide window). Expected: the full sequence — request received → middleware ran → tools called → downstream service called → error at line X with stack trace → response 500 sent. You'd see the user_id, the endpoint, the input (redacted), and the exception. Enough to reproduce, debug, or hand off to another team.

**A10. (b).** Tiered retention balances cost and utility: recent logs searchable fast (hot), older logs available slower (warm), oldest archived cheaply (cold, S3). Compliance often mandates minimum periods (7 years for financial, 6 years for HIPAA). Aggregate metrics (Prometheus/InfluxDB) are cheap enough to keep indefinitely and give long-term trends.

---

## 6. Practice Prompts

1. **Structured log setup.** In a FastAPI app, wire `structlog` with contextvars. Bind `request_id`, `user_id` in middleware. Verify all log calls in a handler auto-include them.
2. **Correlation across services.** Two microservices: service A calls service B. Propagate `x-request-id`. From a single request, grep both services' logs for the ID; assemble the journey.
3. **Sampling.** Add tail-based sampling: always log errors and slow requests; 10% of normal. Measure log volume reduction.
4. **PII redaction.** Write a scrubber for SSN, credit card, email, API key. Test with a synthetic input containing all four.
5. **Log-based dashboard.** Send your JSON logs to Grafana Loki (or Datadog free tier); build a dashboard showing p95 latency, error rate, top users by cost.

---

## 7. References

- structlog docs: [structlog.org](https://www.structlog.org/).
- OpenTelemetry Python: [opentelemetry.io/docs/languages/python/](https://opentelemetry.io/docs/languages/python/).
- W3C Trace Context spec.
- Presidio (Microsoft PII scrubber): [github.com/microsoft/presidio](https://github.com/microsoft/presidio).
- Grafana Loki: [grafana.com/oss/loki](https://grafana.com/oss/loki/).
- Google SRE Book — chapters on logging and monitoring.
