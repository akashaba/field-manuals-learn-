# Production Deployment — Master Study Guide

> **Track:** Agents + Production · **Module:** 13 (final learning module)
> **Prerequisites:** Modules 01–12.
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Deploying an agent to production has all the challenges of shipping any Python service — plus the LLM-specific concerns: cost per request that's 10-1000× a normal HTTP endpoint, latency that's counted in seconds not milliseconds, variable behavior across runs, and cascading dependency on upstream API providers. Add long-running workflows, streaming, checkpointing, and multi-tenancy — and you're building a system very different from a REST CRUD app.

Deploying agents well is what separates "impressive demo" from "product used by hundreds daily." This module covers the specific concerns you'll face.

**Fundamental principles you must own:**

1. **Agents need their own service architecture** — not just wrap them in an HTTP handler.
2. **Concurrency needs care** — LLM calls are slow; block on them naïvely and you fall over.
3. **Long-running agents need durable state** — checkpoints, resumability, backpressure.
4. **Rate limits and cost caps at every layer** — provider, per-user, per-tenant, global.
5. **Observability is not optional** — traces, metrics, alerts.
6. **Reliability = testing + rollout + monitoring + rollback**.

If you retain nothing else: **production agent infra is a real discipline. Design for the failure modes; measure the successes.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Service Architecture for Agents

Your agent typically sits inside an HTTP service (FastAPI, similar). The architecture:

```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
│ Client  │───│ API Gate │───│  Agent  │───│  LLM API │
│         │   │  (auth,  │   │ Runtime │   │(provider)│
│         │   │  quota)  │   │         │   └──────────┘
└─────────┘   └──────────┘   │         │
                             │         │───┌──────────┐
                             │         │   │  Tools   │
                             │         │   │(DBs,APIs)│
                             │         │   └──────────┘
                             └─────────┘
                                  │
                             ┌──────────┐
                             │ State DB │
                             │ (traces, │
                             │checkpoints)
                             └──────────┘
```

**Layers:**

- **API gateway** — auth, per-user rate limiting, request logging, TLS.
- **Agent runtime** — the loop, tools, state management.
- **LLM provider** — with retries + fallback.
- **Tools** — internal services, external APIs, MCP servers.
- **State store** — durable state, traces, checkpoints.
- **Observability** — traces to LangSmith/Braintrust; metrics to Prometheus; logs to Datadog/Loki.

**Async I/O.** LLM calls take 1–30 seconds. In a synchronous Python app (Flask/Django), each request blocks a worker for that long. Concurrent requests bottleneck immediately.

Solutions:
- **FastAPI + async** — handle thousands of concurrent long-running requests per worker.
- **Async LLM SDKs** — Anthropic's `AsyncAnthropic`, OpenAI's `AsyncOpenAI`.
- **Async tool clients** — httpx, aiohttp, asyncpg for DB.

**Sync/async isolation.** If any tool call is synchronous (blocking), wrap it in `run_in_executor` to avoid blocking the event loop.

**Worker sizing.** Async workers can juggle many concurrent requests. Rule of thumb: 1 worker per CPU core; each handling 100–1000 concurrent LLM calls. Test.

---

### 2.2 Long-Running Agents and Durable State

Some agents take minutes to hours. Users close their browsers; servers restart. You need **durable execution**.

**Patterns:**

**1. Short-lived agents (< 30s).** Keep in-process. Return the final answer synchronously (with streaming). Simple.

**2. Streaming long agents.** Stream trace events to the client via SSE. Client stays connected; sees progress. Server keeps state in memory. If disconnect, the agent finishes; result is discarded (or logged) — user must poll.

**3. Background jobs.** POST creates a job; returns a job_id. Agent runs in the background (Celery, RQ, or async task). Client polls or subscribes to updates via WebSocket.

**4. Durable execution.** Use a workflow engine (Temporal, Restate, Prefect). Agent state is persisted after each step; a crash resumes automatically. Essential for hours-long agents with flaky tools.

**Checkpointing.** Save state after each significant step. On restart, resume from the last checkpoint. LangGraph supports this natively.

**Idempotency.** Design agent steps so re-running them is safe (or detected as duplicates). Especially for tool calls with side effects — use idempotency keys.

**Cancellation.** Users get impatient. Long-running agents should:
- Accept a cancellation signal (via a shared queue, DB flag, or WebSocket).
- Check for cancellation between steps.
- Return partial results / clean up on cancel.

**Timeout policy.** Every agent has a max_wall_time. Beyond it, terminate the run gracefully — save partial state, notify user.

---

### 2.3 Rate Limiting, Cost Control, Quotas

Production agents blast through provider rate limits and consume budget fast.

**Rate limit layers:**

1. **Provider level** — OpenAI/Anthropic set per-org RPM / TPM.
2. **Application level** — enforce per-user and per-tenant rate limits.
3. **Per-agent-run** — max steps, max tokens, max cost.
4. **Global circuit breakers** — if provider is degraded, throttle all traffic.

**Handling provider rate limits (429):**

- Exponential backoff with jitter.
- Honor `Retry-After` header.
- If persistent, fail the request with a clean error; don't queue indefinitely.

**Cost budgets:**

- **Per-run cost cap** — abort if agent's cumulative token cost exceeds N dollars.
- **Per-user daily cap** — reject if user has spent > $X today.
- **Per-tenant monthly cap** — for multi-tenant SaaS.
- **Global daily cap** — cost circuit breaker.

Cost logs at every LLM call; aggregate; alert on anomalies (a 10× spike = a bug or attack).

**Concurrency limits.** Even with async, don't let a single user launch 10,000 concurrent agent runs. Limit concurrent runs per user/tenant.

**Multi-provider fallback.** If OpenAI is down, fall back to Anthropic. Requires:
- Provider-agnostic wrapper.
- Feature-parity between models you fall back to.
- Route logic (health-check, provider status).

Not always worth the complexity, but for critical apps it's the difference between "down" and "degraded."

---

### 2.4 Observability: Traces, Metrics, Logs

**Structured logs.** Every log line is JSON with request_id, user_id, endpoint, model, tokens, cost, latency, error.

**Traces.** For every agent run, record every step: LLM call, tool call, tool result. Store in LangSmith / Braintrust / Arize Phoenix / your own DB. Traces are your primary debugging tool.

**Metrics (Prometheus / OpenTelemetry):**

- **Request rate** per endpoint.
- **Success rate** per endpoint / task type.
- **Latency histograms** (p50, p95, p99) — total, TTFT, per-tool.
- **Cost per run** — histogram.
- **Step count per run** — histogram.
- **Tool error rate** per tool.
- **Refusal rate** — did the agent refuse?
- **Human intervention rate** — how often does it escalate?
- **Cache hit rate** — prompt cache, semantic cache.
- **Concurrent runs** — for capacity planning.

**Dashboards.** One per service; alerts on:
- Success rate drop > 5% week-over-week.
- Cost per run > 2× baseline.
- p95 latency > SLA.
- Refusal rate spike (> 2× normal) — signals retrieval / model regression.
- Provider errors (429/5xx) > threshold.

**Sentry / error tracking.** Auto-alert on exceptions, especially unhandled ones.

**Correlate.** A trace_id in every log; propagate through the whole stack. When something goes wrong, one query reconstructs the full picture.

---

### 2.5 Rollout, A/B Testing, and Reliability

**Deployment strategy:**

- **Blue-green** — two environments; switch traffic atomically.
- **Canary** — small % of traffic to new version; watch metrics; scale up.
- **Feature flags** — toggle changes per user / tenant.

**A/B testing:**

Run two agent versions in parallel; measure per-cohort metrics (task success, cost, latency, user satisfaction). Ship the winner.

Common experiments:
- Prompt A vs prompt B.
- Model A vs model B.
- Tool set A vs tool set B.
- Retrieval strategy A vs B.
- Framework A vs B.

**Metrics to gate on:**
- Task success rate.
- Faithfulness / correctness.
- Cost per successful run.
- Latency p95.
- User feedback (thumbs, retention).

**Reliability engineering:**

- **Chaos testing.** Inject failures — provider slowness, tool timeouts, DB delays. Verify agent degrades gracefully.
- **Load testing.** Simulate peak traffic (k6, Locust) to verify capacity.
- **Runbook.** Written procedures for common incidents (provider outage, cost spike, quality regression).
- **On-call.** Someone owns the pager.

**Rollback.** Every deploy must be rollbackable. Prompt changes, model swaps, tool updates — all reversible via config or feature flag.

**Model versioning.** Pin the model version in code (e.g., `claude-...`). Auto-upgrading to a new provider snapshot without testing is asking for regressions. Explicitly test-and-upgrade on your schedule.

**Regression testing (Module 12).** Every PR runs the eval suite. Merges blocked on regressions.

**Graceful degradation:**
- If a tool is unavailable, agent tells the user which capability is missing.
- If provider is degraded, warn the user and use fallback.
- If cost limit hit, tell the user, don't silently truncate.

Reliability engineering for agents is not fundamentally different from reliability engineering for any distributed system — it just has more failure surface. Apply the same discipline.

---

## 3. Mental Models & Analogies

### 3.1 The "Long-Running Serverless Function" Model

Think of an agent run as a **serverless function with a 30-second-to-hours execution time budget**.

- Traditional serverless (Lambda, Cloud Run) targets 10ms-30s.
- Agent runs are 5s-hours.
- You need: async I/O (many concurrent runs), durable state (survives restarts), streaming (send progress to client), budget caps (cost per invocation).

Your infra picks: async web framework + workflow engine (Temporal) for durability + LLM SDK with retries + observability. This is why "just an API endpoint" is often not enough.

### 3.2 The "Call Center vs Autonomous Vehicle" Model

Traditional web services are like a **call center**: each request is short, handled quickly, then done. Concurrent throughput is high; latency is low; each call is stateless.

Agent services are more like an **autonomous vehicle fleet**: each ride takes minutes to hours; state persists; navigation adapts to traffic; kids in the back need updates ("are we there yet?"); route can be canceled or replanned; the fleet must be monitored globally.

You wouldn't run an autonomous vehicle service with the same infra as a call center. Same for agent services: different traffic patterns, different observability needs, different reliability engineering.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Just Wrap the Agent Loop in a FastAPI Endpoint"

Works for a prototype. For production: you need async I/O (LLM calls are slow; sync = worker death), durable state for long-running work, streaming to keep users engaged, cost caps, retry logic on the LLM/tool layer, structured tracing, and often a background job runner. "Just wrap it" leaves out most of the production concerns.

### 4.2 "Sync Python Is Fine — We'll Scale Horizontally"

Sync Python + slow LLM calls means each worker handles ~1-10 concurrent requests. For 1000 concurrent users, you need 100-1000 workers — with the memory / cost / provisioning overhead. Async Python handles thousands per worker. The 1-hour cost of learning async pays off in 10× fewer workers.

### 4.3 "Traces Are Nice-to-Have; Log Files Are Enough"

Traces are the debugger for agents. Without them, a failing agent run is a mystery. Log files aren't structured for the multi-step, multi-tool nature of agent execution. Invest in tracing (LangSmith / Braintrust / OpenTelemetry / Phoenix) day one.

---

## 5. Self-Assessment Bank (Production Deployment)

### Questions

**Q1 (Short answer).** Why does an agent service almost always need async Python rather than sync?

**Q2 (Multiple choice).** For an agent that takes 30-60 minutes per run and must survive crashes:
- (a) Synchronous FastAPI endpoint.
- (b) Background job on Celery.
- (c) Workflow engine (Temporal / Restate) for durable execution.
- (d) In-memory queue.

**Q3 (Short answer).** Give three cost-control layers you'd implement in a production agent service.

**Q4 (Multiple choice).** For long-running agents, streaming trace events to the client:
- (a) Wastes bandwidth.
- (b) Improves UX (visible progress), enables cancellation, and gives users a reason to stay engaged.
- (c) Doesn't work in browsers.
- (d) Only for coding agents.

**Q5 (Short answer).** What metrics would you track on a production dashboard for an agent service?

**Q6 (Multiple choice).** When the LLM provider returns 429 (rate limited), the agent client should:
- (a) Fail immediately.
- (b) Exponential backoff with jitter; honor Retry-After header; cap total retries.
- (c) Retry as fast as possible.
- (d) Switch to a smaller model.

**Q7 (Short answer).** How would you A/B test two prompt versions of your agent in production?

**Q8 (Multiple choice).** Multi-provider fallback (Anthropic ↔ OpenAI):
- (a) Always necessary.
- (b) Useful for critical apps to survive provider outages, but adds complexity — evaluate per-service.
- (c) Impossible.
- (d) Slower than single provider.

**Q9 (Short answer).** Describe graceful degradation for an agent whose primary retrieval tool is down.

**Q10 (Multiple choice).** For traces (multi-step agent run logs), the recommended storage is:
- (a) Print to stdout.
- (b) Structured storage in LangSmith/Braintrust/Phoenix or your own DB — for querying, replay, and eval.
- (c) Local disk only.
- (d) Not needed.

---

### Answer Key & Detailed Explanations

**A1.** LLM calls take 1-30 seconds; tool calls similar. Sync Python blocks a worker for the whole call. For 100 concurrent long-running requests, you need 100 workers — memory-heavy, expensive. Async Python (asyncio + async LLM SDKs + async I/O) handles hundreds-to-thousands of concurrent long calls per worker. Order-of-magnitude infra savings.

**A2. (c).** Durable execution engines (Temporal, Restate) persist state after each step; a crash resumes automatically. Essential for hours-long agents on flaky infra. Celery (b) can work with careful state management but lacks native checkpointing. Sync FastAPI (a) has no durability. In-memory queue (d) loses everything on restart.

**A3.** (1) **Per-run cost cap** — abort a run if it exceeds $X. (2) **Per-user daily cap** — reject requests if user's daily spend > threshold. (3) **Per-tenant monthly cap** — for multi-tenant SaaS. (4) **Global circuit breaker** — throttle all traffic if daily cost approaches limit. (5) **Per-tool budget** — restrict expensive tool usage. Any three plus rationale.

**A4. (b).** Streaming trace events (SSE or WebSocket) shows users progress ("Searching web... Reading result 3..."), makes long-running agents feel responsive, allows the user to cancel if they see it going wrong, and produces the visible progress users need for multi-minute tasks. Standard for Deep Research-style agents.

**A5.** Any 5+ of: **success rate** per task type, **latency histograms** (p50/p95/p99 total, TTFT, per-tool), **cost per run** histogram, **step count per run** histogram, **tool error rate** per tool, **refusal rate**, **human intervention rate**, **cache hit rate** (prompt / semantic), **concurrent runs**, **provider errors** (429/5xx), **user feedback** (thumbs, retention).

**A6. (b).** Exponential backoff with jitter avoids stampeding the provider on recovery. Honor `Retry-After` header. Cap total retries (say 5) to fail cleanly rather than hang. Return a clean error to the user with retry suggestion.

**A7.** (1) Feature-flag two prompt versions; route users to A or B randomly (or by hash of user_id for consistency). (2) Log every run's version, task, outcome, cost, latency. (3) After enough traffic (statistical power), compare per-cohort metrics: task success rate, faithfulness, cost, latency. (4) If B wins on primary metric without regressing others, promote to 100%. Use per-user consistency to avoid within-user variance polluting the experiment.

**A8. (b).** Multi-provider fallback survives OpenAI outages (which happen) or Anthropic outages. Adds complexity: provider-agnostic wrapper, per-provider model mapping, health checks. Worth it for critical apps; overkill for many. Evaluate based on SLA and downside cost of an outage.

**A9.** (1) **Detect** the tool failure at the tool boundary; return structured error. (2) **Try fallback** — a secondary retrieval source if configured. (3) **If no fallback**, agent should tell the user: "I'm currently unable to search the web; here's what I can share from my general knowledge / cached info." (4) **Alert** operators; degrade the endpoint status to "partial." (5) **Optionally** offer to save the query and notify the user when service is restored.

**A10. (b).** Structured storage (LangSmith, Braintrust, Arize Phoenix, or a self-hosted DB) lets you query traces by user, run_id, task, error type; replay runs for debugging; feed into eval systems; correlate with metrics. Stdout / disk-only prevents any of that. Debugging agents without proper trace storage is nearly impossible at scale.

---

## 6. Practice Prompts

1. **Async FastAPI agent.** Wrap your Build agent in an async FastAPI endpoint; use async Anthropic SDK. Load-test with 100 concurrent requests; verify it stays responsive.
2. **Streaming trace to client.** Add SSE streaming: as your agent runs, emit `event: step, data: {...}` for each step. Client renders progress.
3. **Cost budget.** Enforce per-run ($1) and per-user daily ($10) cost caps. Simulate an abusive user; verify caps trigger.
4. **Multi-provider fallback.** Wrap Anthropic + OpenAI behind a common interface. Fall back on 5xx errors. Test with simulated provider outage.
5. **Observability stack.** Wire up: structured logs (JSON to stdout), Prometheus metrics endpoint (`/metrics`), traces to LangSmith or Phoenix. Build a small Grafana dashboard.

---

## 7. References

- FastAPI docs on async.
- Anthropic `AsyncAnthropic` docs.
- Temporal for LLMs — [temporal.io/blog](https://temporal.io/blog).
- LangSmith, Braintrust, Arize Phoenix docs on tracing.
- OpenTelemetry GenAI semantic conventions.
- Google SRE Book — reliability engineering principles.
- Anthropic ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents).
