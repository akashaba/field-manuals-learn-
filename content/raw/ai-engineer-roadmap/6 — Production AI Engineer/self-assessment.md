# Month 6 Exit Exam — Production AI Engineer 🚀

> **Format:** 40 rigorous questions spanning the 14 topics. Mix of multiple-choice, short-answer, and design questions. Take 60–90 minutes. Grade against the answer key at the bottom (aim for ≥ 80% before declaring the month complete).
>
> **What "passing" means:** you can walk into a Senior AI Engineer / MLOps interview and speak fluently about the whole production stack — not just theoretical, but *what breaks and how you'd fix it*.

---

## Section A — FastAPI & Async Python (Module 1)

**Q1 (MC):** In FastAPI, `async def` handlers are appropriate when:
A) All I/O in the handler is done with sync libraries
B) The handler awaits async I/O (httpx, asyncpg, redis.asyncio)
C) The handler is CPU-heavy
D) Always — FastAPI requires it

**Q2 (short):** Why is `Pydantic.ConfigDict(extra="forbid")` a security-relevant setting?

**Q3 (short):** Explain the FastAPI `lifespan` context manager and what belongs there vs. in per-request dependencies.

---

## Section B — Docker & Images (Module 2)

**Q4 (MC):** A multi-stage Docker build primarily helps by:
A) Speeding up runtime
B) Producing smaller, more secure final images by leaving build tools in earlier stages
C) Enabling parallel deploys
D) Reducing memory usage

**Q5 (MC):** Why prefer `python:3.11-slim` over `python:3.11` as a base for a production Python service?
A) Slim ships with more tools
B) Slim has a smaller attack surface and image size
C) Slim is faster at runtime
D) Slim is required by FastAPI

**Q6 (short):** What is a "distroless" image and when would you use one?

---

## Section C — Cloud Deployment (Module 3)

**Q7 (MC):** For an intermittently-called LLM service with unpredictable traffic, scale-to-zero is best supported by:
A) EC2 with an autoscaling group
B) GCP Cloud Run
C) A raw Kubernetes cluster
D) On-prem servers

**Q8 (short):** Explain concurrency-based autoscaling (Cloud Run's `--concurrency` flag) and why it fits I/O-bound AI services better than CPU-based scaling.

**Q9 (short):** Describe rolling vs blue-green vs canary deploy strategies and give one situation where each is preferred.

---

## Section D — CI/CD (Module 4)

**Q10 (MC):** GitHub Actions OIDC federation to AWS/GCP lets you:
A) Encrypt secrets at rest
B) Deploy without long-lived cloud credentials in the repo
C) Speed up job execution
D) Cache Docker layers

**Q11 (short):** In a CI pipeline for an LLM service, what belongs in the "must pass" jobs vs the "advisory" jobs, and why is eval-regression a first-class gate?

---

## Section E — MLflow (Module 5)

**Q12 (MC):** MLflow's Model Registry alias (e.g., `@champion`) vs the deprecated stage system offers:
A) The same thing with a new name
B) Multiple mutable pointers to specific versions, allowing atomic swaps for A/B and canary flows
C) Automatic performance tracking
D) Encrypted storage of models

**Q13 (short):** How would you wire MLflow into a FastAPI service so that promoting a new model requires no redeploy?

---

## Section F — Logging (Module 6)

**Q14 (MC):** The primary reason to prefer JSON structured logs over string-formatted logs in production:
A) They compress better
B) They're queryable — filter by field, group by field, join with traces/metrics
C) They're human-readable
D) They're required by SOC 2

**Q15 (short):** Why should you never log the raw `Authorization` header, and what pattern prevents it?

**Q16 (short):** Explain W3C traceparent propagation and how it enables cross-service correlation.

---

## Section G — Monitoring & Metrics (Module 7)

**Q17 (MC):** In PromQL, `histogram_quantile(0.99, sum(rate(x_bucket[5m])) by (le))` computes:
A) The average rate
B) The 99th percentile from the aggregated histogram buckets
C) The count of events
D) A random sample

**Q18 (short):** Explain multi-window multi-burn-rate SLO alerting and why it beats a single-threshold rule.

**Q19 (short):** For a chat API, name three AI-specific metrics beyond RED/USE and the decision each supports.

---

## Section H — Model Evaluation (Module 8)

**Q20 (MC):** A "golden set" in production eval is:
A) A random sample of the training set
B) A small, curated, hand-labeled set that never trains the model and encodes known edge cases and past bugs
C) The full held-out test set
D) A synthetic benchmark

**Q21 (short):** Describe the difference between shadow, canary, and A/B testing, and when to use each.

**Q22 (short):** LLM-as-judge is powerful but has known biases. Name three and one mitigation each.

---

## Section I — LLM Observability (Module 9)

**Q23 (MC):** In LLM observability, a "trace" represents:
A) A stack trace of an error
B) One user request end-to-end, potentially spanning multiple LLM/tool spans
C) A single log line
D) A Prometheus metric

**Q24 (short):** Explain the difference between TTFT and total generation latency, and why streaming UIs must SLO both.

**Q25 (short):** Give three signals that should force keeping a trace in tail-sampling.

---

## Section J — Rate Limiting (Module 10)

**Q26 (MC):** A well-behaved 429 response should include:
A) `Retry-After` header, `RateLimit-*` headers, JSON body describing which limit hit
B) Only a body
C) Only headers
D) 500 status code

**Q27 (short):** Why is `RPM` alone insufficient for LLM APIs, and what other dimensions should you rate-limit on?

**Q28 (short):** Sketch a Redis-backed sliding-window counter (in pseudocode or words) and explain why an atomic Lua script matters.

---

## Section K — Caching (Module 11)

**Q29 (MC):** Anthropic's prompt caching charges cached input tokens at approximately:
A) The same as normal
B) ~10% of normal input rate
C) Free
D) 2× normal

**Q30 (short):** You cache RAG answers with 24h TTL, but your corpus refreshes daily. Describe the bug and a fix.

**Q31 (short):** Why must cache keys include tenant_id in a multi-tenant AI service?

---

## Section L — Authentication (Module 12)

**Q32 (MC):** Which JWT attack does pinning `algorithms=["RS256"]` in your decode call mitigate?
A) Replay
B) `alg: none` / algorithm confusion
C) Weak secret
D) Token theft

**Q33 (short):** Design the auth flow for a multi-tenant SaaS where enterprises use their own OIDC IdPs. Where does `tenant_id` come from and where is it verified?

**Q34 (short):** Explain Postgres row-level security (RLS) and why it's valuable defense-in-depth for a multi-tenant AI service.

---

## Section M — Security (Module 13)

**Q35 (MC):** LLM01 (OWASP Top 10 for LLM Applications) refers to:
A) Insecure output handling
B) Prompt injection
C) Excessive agency
D) Model theft

**Q36 (short):** Explain the difference between direct and indirect prompt injection and one mitigation specific to indirect.

**Q37 (short):** Your model returns `"DROP TABLE users;"` and your service executes it. Which OWASP LLM vulnerability, and how do you prevent it?

---

## Section N — Cost Optimization (Module 14)

**Q38 (MC):** For a mixed-complexity chatbot workload, the highest-impact cost lever is typically:
A) Prompt caching
B) Model routing (Haiku/Sonnet/Opus per intent)
C) `max_tokens` cap
D) Batch API

**Q39 (short):** Walk through why prompt caching pays off after roughly 1.3 uses.

**Q40 (short — capstone):** You just discovered last week's Anthropic bill is 4× normal ($47k vs $12k). Walk through your investigation and remediation steps in order.

---

## Answer Key

### Section A

**A1: B.** `async def` is appropriate when the handler awaits async I/O. Using sync libraries inside async handlers blocks the event loop (worse than a sync handler). CPU-heavy work should either be a sync handler (FastAPI runs sync handlers in a thread pool) or offloaded via `asyncio.to_thread` or a worker queue.

**A2:** `extra="forbid"` causes Pydantic to reject any field not declared in the model schema. Without it, a client can send `{"role": "admin", "amount": 100}` to a handler that only expected `{"amount": int}`, and the extra `role` field is silently accepted (possibly passed to downstream code that trusts it). "Forbid" turns mass-assignment attacks into validation errors. Also helps in evolution: unknown fields from a new client version don't silently disappear.

**A3:** `lifespan` is an async context manager run once at app startup and once at shutdown. It owns **shared, long-lived resources**: HTTP clients (share connection pools), DB pools, Redis connections, ML model in memory, background task scheduler. Per-request dependencies (`Depends(...)`) build request-scoped values: authenticated user, tenant context, request ID. Rule: if construction/teardown is expensive and the resource is shared, it's in lifespan; if it's per-request, it's a dependency.

### Section B

**A4: B.** Multi-stage builds let you compile/install dependencies with a full toolchain image, then copy only the artifacts into a slim final image — no compilers, no test files, no cache. The final image has a smaller attack surface, downloads faster, and stores cheaper.

**A5: B.** `python:3.11-slim` is based on `debian:bookworm-slim` with only the Python runtime — no `git`, `curl`, `gcc`, or dozens of other tools present in the full `python:3.11` image. Smaller = smaller attack surface, faster pulls, less CVE noise from unused packages. Full `python:3.11` is `debian:bookworm` full (~900MB); slim is ~150MB.

**A6:** A **distroless** image (from `gcr.io/distroless/*`) contains only the language runtime and your app — no shell, no package manager, no OS utilities. Use it when you want the smallest possible attack surface. Trade-off: no `bash` means no `kubectl exec` for debugging; you must design for observability (logs + metrics + traces) that don't require shell-in.

### Section C

**A7: B.** Cloud Run natively supports scale-to-zero and per-request billing. Ideal for AI services with spiky, intermittent traffic — you pay nothing during quiet periods. EC2 ASGs can scale down but not to zero without additional plumbing; Kubernetes needs Knative or KEDA to reach zero; on-prem is by definition always running.

**A8:** `--concurrency=N` sets the max concurrent requests a single Cloud Run instance handles before another instance is spun up. AI services are I/O-bound: most of the request time is waiting on the LLM API (network I/O). A single Python process can handle 50–200 such concurrent awaits with tiny CPU. CPU-based scaling would either over-provision (creating instances that sit idle waiting on I/O) or under-scale (bottlenecking on requests that could have been handled by a single instance's `asyncio` loop). Concurrency-based scaling directly reflects "how many awaits are we juggling," which is the true bottleneck.

**A9:**
- **Rolling**: replace instances one-by-one with the new version; keeps capacity constant; simplest; small risk window per instance. Default; good baseline.
- **Blue-green**: deploy new version as a full parallel fleet, then flip the load balancer; near-instant rollback; wasteful (2× capacity briefly); good for high-risk changes needing atomic swap.
- **Canary**: deploy new version to a small percentage of traffic (1% → 5% → 25% → 100%); monitor SLIs at each step; automatic rollback on regression; best for changes with quality risk (new prompt, new model, tricky code path).

### Section D

**A10: B.** OIDC federation lets GitHub Actions authenticate to AWS/GCP as a workload identity — issuing short-lived tokens per job — instead of storing long-lived access keys in repo secrets. Eliminates the "leaked GitHub secret = permanent cloud access" risk. Setup: create an IAM role trusting `token.actions.githubusercontent.com` for your repo/branch, and use `permissions: id-token: write` in the workflow.

**A11:**
- **Must pass** (blocks merge): lint, type-check, unit tests, **eval-golden** (100% of golden cases pass), safety-required refusal tests, security scan (no CRITICAL CVEs), build succeeds.
- **Advisory** (reports to PR): coverage delta, full offline eval scores (report ± vs baseline), cost delta on eval set, response-length delta.
- **Post-merge, pre-prod**: shadow traffic hours, canary at 1% then 5%, A/B on primary KPI.
Eval-regression is first-class because *model output quality is the product* for an AI service; a code change that keeps tests passing but drops answer quality is a regression the traditional CI can't detect. Golden set + judge-scored full eval catches it.

### Section E

**A12: B.** Aliases are mutable named pointers to specific model versions. You can atomically flip `@champion` from v14 → v15 without redeploy; you can have `@champion`, `@challenger`, `@shadow` simultaneously for A/B/shadow flows. The old stage system (Staging/Production/Archived) was a single fixed enumeration and didn't support arbitrary pointers or multiple concurrent labels.

**A13:** Fetch the model with `mlflow.pyfunc.load_model("models:/my-model@champion")` at service startup (in `lifespan`). For refresh-without-redeploy: run a background task that periodically checks the current version bound to `@champion`; if changed, load the new version into a shadow slot and hot-swap the reference (a simple `app.state.model = new_model` with an atomic pointer swap). Alternative: hit a `/reload` admin endpoint after promoting in the registry.

### Section F

**A14: B.** JSON is queryable. `level=ERROR user_id=abc123 latency_ms>1000` in a log aggregator UI is a one-line search over structured fields, not a regex on strings. Enables SLO dashboards from logs, joins with traces (both carry `request_id`), and reliable alerting.

**A15:** The `Authorization` header contains the bearer token — anyone with logs access has valid credentials for that user until token expiry. Also: leaks to log aggregators, backups, screenshots, ticket systems. Pattern: **redaction middleware** that strips `Authorization`, `Cookie`, `x-api-key`, and any known-secret header names before the log line is written. Never `print(request.headers)`; use a serializer that knows the block-list.

**A16:** W3C traceparent is an HTTP header `traceparent: 00-<trace_id>-<span_id>-<flags>` propagated between services. When service A calls service B, A passes its current traceparent; B parses it, creates a child span under A's trace, and passes its own traceparent when calling service C. Result: all spans across A, B, C share one trace_id, and observability tools can reconstruct the full request flow. Same principle inside a single service across async boundaries via contextvars.

### Section G

**A17: B.** `histogram_quantile(0.99, sum(rate(x_bucket[5m])) by (le))` computes the 99th percentile from histogram buckets aggregated across replicas/labels. The `sum by (le)` aggregates the bucket counts (essential for correctness — you can't take an average of percentiles), then `histogram_quantile` computes the quantile from the aggregated buckets.

**A18:** Multi-window multi-burn-rate = fire alerts at different (window, burn-rate) combinations against the SLO error budget: e.g., a "2h window at 14.4× budget-burn rate" (page immediately — this pace exhausts the monthly budget in ~2 days) plus a "6h at 6×" (ticket) plus a "24h at 3×" (weekly review). Beats a single-threshold alert because a single threshold either fires too often (short blips) or too late (long slow burns eat the budget before firing). The multi-window approach catches both fast outages and chronic degradations while minimizing false positives.

**A19:**
- **$ per user per day (gauge)** — decides whether to enforce a spend cap or upgrade a user's tier.
- **Cache hit rate (per layer)** — decides whether caching config needs tuning (bad key design → hit rate drops after a deploy).
- **Refusal / guardrail block rate** — decides safety investigation (rate spike = attack; rate drop = safety regression).
- **TTFT p95** (for streaming) — decides whether users are perceiving degradation before full-latency SLO breaks.
- **Retrieval hit rate** (RAG) — decides whether corpus refresh or retrieval tuning is needed.

### Section H

**A20: B.** A golden set is curated: hand-labeled, includes past bug fixes and edge cases, small (100s to low 1000s), never in training data, versioned like code. It's institutional memory of quality — every regression that reached prod gets a case added. Random samples (A) and full test sets (C) are separate tools; synthetic benchmarks (D) are useful but no substitute for real edge cases.

**A21:**
- **Shadow**: new version receives *copies* of prod traffic; its output is logged and evaluated but not returned to users. Zero user risk; 2× inference cost. Use when the change is unproven.
- **Canary**: new version serves a real slice of traffic (1% → 5% → 25%); users get its output. Monitor SLIs; auto-rollback on regression. Use after shadow clears.
- **A/B test**: deterministic user split; measure a *business* metric (retention, task completion, thumbs-up) over days/weeks with statistical rigor. Use when the technical metrics look fine but you need to confirm product impact.

**A22:** Three biases + mitigation each:
1. **Position bias** — prefers first-shown option in pairwise. *Mitigation:* randomize order per call; run twice with swapped order.
2. **Verbosity bias** — prefers longer answers. *Mitigation:* explicit "ignore length" instruction; length-normalize scores.
3. **Self-preference / stylistic bias** — prefers answers from its own model family. *Mitigation:* use a different provider as judge; validate against human labels.
4. **Sycophancy** — aligns with perceived user preference. *Mitigation:* strip user preference cues from the judge prompt.

### Section I

**A23: B.** A trace = one request end-to-end as a tree of spans. Each LLM call, tool call, DB query is a span. In an agent, one user chat might be 5–10 spans under one trace.

**A24:** **TTFT** (time-to-first-token) = wall time from request start to first token in the streamed response. **Total generation latency** = time to the last token. For streaming UIs, TTFT determines *perceived responsiveness* — a 500ms TTFT feels snappy even if total is 8s; a 3s TTFT feels broken regardless of total. SLO both: TTFT p95 < 1s and total p95 < 8s (numbers depend on your product).

**A25:** Three signals to force-keep:
1. **Error / non-2xx / provider failure** — 100% retention for diagnosability.
2. **Latency > SLO threshold** — retain to root-cause tail latency.
3. **User thumbs-down or LLM-judge score below floor** — retain to feed the next eval iteration.
Also: guardrail hit, refusal, cost above per-request cap, first-hour traffic on a new prompt version.

### Section J

**A26: A.** All three — `Retry-After` (client knows when to retry), `RateLimit-*` headers (client knows their remaining budget), JSON body (client can distinguish which limit was hit: RPM vs TPM vs $). This is the modern well-behaved rate-limit response.

**A27:** RPM treats every request as equivalent, but an LLM request can vary in cost by 100×. A user sending 60 requests/min with 200k-token contexts and 8k-token outputs is doing ~$150/min in Sonnet costs; RPM alone would allow it. Add: **TPM** (tokens/min per user), **$/day per user**, optionally **concurrent requests per user**. Enforce the tightest active dimension.

**A28:** Sliding-window counter in Redis using a sorted set of request timestamps:
```
ZREMRANGEBYSCORE key 0 (now - window)    # drop expired
ZCARD key                                 # count current
if count < limit:
    ZADD key now <unique_id>              # accept and record
    EXPIRE key window
    return allowed
else:
    return rejected, retry_after=<time-until-oldest-expires>
```
Wrapping this in a Lua script makes the whole check-and-set atomic — otherwise two concurrent processes both read `count = limit-1` and both accept, letting `limit+1` requests through (a race). Lua ensures Redis executes the entire logic under a single-thread lock.

### Section K

**A29: B.** Anthropic prompt caching charges cached input tokens at approximately 10% of the normal input rate (specifics vary by model and cache TTL; check the pricing page). Cache-creation carries a small premium (~25% more than normal); cache reads are the discount. Break-even is under 2 uses, so caching any reused prefix ≥ 1024 tokens is net-positive.

**A30:** Bug: users get stale answers for up to 24h after corpus refresh because their cached response is bound to the old corpus. Fixes: (1) **version-in-key** — include `corpus_version` in the cache key; corpus refresh bumps the version and old keys become naturally unreferenced; (2) **explicit invalidation** — flush the RAG cache prefix on refresh; (3) **shorter TTL** aligned to refresh cadence; (4) **event-driven invalidation** — publish `corpus_refreshed`, listener flushes. #1 is the cleanest.

**A31:** Without `tenant_id` in the key, tenant A's cached response for a shared query can be returned to tenant B — a data leak between organizations. Cache scoping is not optional in multi-tenant systems; every layer (in-process, Redis, semantic, provider prompt cache) must be tenant-scoped. Also: `user_id` if answers are user-personalized; `locale`; `role` if authorization affects the answer.

### Section L

**A32: B.** Pinning the algorithm prevents both the `alg: none` attack (attacker sets header to `none`, submits unsigned token, naive verifier accepts) and the RS256/HS256 confusion (attacker signs an HS256 token using the RSA public key as the HMAC secret; verifier configured to accept HS256 validates it). Always pin: `jwt.decode(token, key, algorithms=["RS256"])`.

**A33:** Design:
1. Enterprise admin registers their OIDC issuer URL and their `tenant_id` (assigned by you).
2. User signs in through their enterprise IdP; your app receives an OIDC `id_token`.
3. Backend validates the token: signature verifies against the IdP's JWKS, `iss` matches the registered issuer, `aud` matches your app.
4. Backend resolves `tenant_id` from your registration table (mapping `iss` → `tenant_id`), NOT from the token's own claim (you don't trust the enterprise IdP to name its own tenant_id).
5. Backend mints its own short-lived JWT with `sub`, `tenant_id` (from your table), scopes.
6. Frontend uses your JWT for API calls; enterprise IdP tokens never reach your API endpoints — clean decoupling.
Verification of `tenant_id` happens *on your side of the trust boundary* — you set it based on which IdP the user authenticated through, not what the IdP claims.

**A34:** Row-level security = Postgres policies that filter rows returned/modified by every query based on session variables. Example:
```sql
CREATE POLICY tenant_isolation ON docs
  USING (tenant_id = current_setting('app.tenant_id')::uuid);
```
App sets `SET app.tenant_id = '<from JWT>'` at request start. Every subsequent query — no matter how it's written — automatically filters. **Defense in depth**: a buggy service missing a `WHERE tenant_id = ?` clause can no longer leak cross-tenant data; the DB refuses to return the wrong rows. Combined with app-layer checks, this reduces the AuthZ surface area dramatically for a multi-tenant AI service.

### Section M

**A35: B.** LLM01 = Prompt Injection — the flagship AI-specific vulnerability where user or third-party input manipulates model instructions. LLM02 is insecure output handling; LLM08 is excessive agency; LLM10 is model theft.

**A36:** **Direct** = user directly injects malicious instructions ("ignore prior instructions"). **Indirect** = malicious instructions are embedded in third-party content the model ingests — a webpage, a PDF, an email — and the user asked something innocuous ("summarize this page"). Mitigation specific to indirect: **isolate ingestion from action**. Run one model call that only extracts structured facts from untrusted content (no tools available); then a separate call that uses those extracted facts (with tools) but never sees the raw content. This prevents an injection embedded in a webpage from ever reaching a tool-authorized context. Also: strip HTML comments, hidden text, zero-width chars from ingested content.

**A37:** Vulnerability = **LLM02 Insecure Output Handling**. The service treated the model's string output as trusted SQL. Prevention: (1) never `exec()`/pass model output to a shell or SQL sink — always parameterize; (2) enforce structured output (`response_format=json_schema`) that constrains the shape; (3) if the model must produce SQL, validate it against a schema or a safe subset (a SQL AST parser rejecting DDL), and execute in a read-only role with row-level security; (4) monitor for output patterns that look like injection attempts.

### Section N

**A38: B.** Model routing typically dominates because the price gap between smallest (Haiku-class) and largest (Opus-class) is often 10–20×. Even a modest routing rule saves 40–60% on total spend at no measurable quality cost for the routed-cheap intents. Prompt caching (A) is a strong second at 80–90% off the cached portion, but the ceiling is set by prefix reuse. max_tokens (C) is a good hygiene lever but limited. Batch API (D) is 50% but only for offline work.

**A39:** With `p_in` as normal input token price, `1.25 × p_in` as cache creation, and `0.1 × p_in` as cache read:
- Without cache: `n × T × p_in`
- With cache: `T × p_in × (1.25 + 0.1(n−1))`
Break-even: `n = 1.25 + 0.1(n−1)` → `0.9n = 1.15` → `n ≈ 1.28`.
So after ~1.3 uses, caching is net-positive. This is why caching anything ≥ 1024 tokens that's reused *twice or more* is a default-yes decision. (Exact ratios depend on provider and model tier; check the current pricing.)

**A40 (capstone):** Investigation and remediation, in order:
1. **Stop the bleeding immediately if bleeding continues.** Check current hour's spend rate — is it still 4× or normalized? If still elevated, apply a temporary top-level rate limit (global RPM/TPM cap in your API gateway) while you diagnose. Set an emergency budget cap.
2. **Break the number down.** Use your cost dashboard (Module 9/14): `$` by day, by hour, by model, by route, by user, by prompt version. What changed?
3. **Identify the driver.** Common patterns: (a) one user driving huge share (abuse or underpriced power user); (b) one route driving huge share (a new feature launched, or an old route had a retry-loop bug regress); (c) one model driving share (router logic broke, everything falling through to Opus); (d) one prompt version (prompt regression bloated context or output).
4. **Trace-level dive.** Pull sampled traces from the peak hour; look for anomalies: retries, huge contexts, high output_tokens, prompt version, tool loops.
5. **Remediate the specific cause.**
   - Abuser: apply per-user rate limit; ban or upgrade-to-paid.
   - Retry bug: fix backoff logic; ship hotfix.
   - Router regression: fix the classifier or fallback default; roll back the responsible change.
   - Prompt regression: revert the prompt version via LangFuse alias flip (no redeploy needed).
   - Agent step-loop: cap max_steps; add cost per step check.
6. **Post-mortem.** How did this get to production undetected? What monitoring/alerting was missing? Add: (a) cost anomaly alert on 15-min rate > 3× baseline, if not present; (b) cost SLO — dashboarded, budgeted, alerted, treated as a first-class SLI; (c) daily automated cost report in your team's chat; (d) an eval-time cost delta gate in CI (a PR that inflates per-request cost by > 20% requires justification).
7. **Contain the loss.** Talk to finance/support; if a specific customer's tier priced this poorly, decide whether to grandfather-then-migrate or refund. Update pricing docs.
8. **Prevent recurrence.** The gate you didn't have is now the gate you have.
Total time-to-remediation for a mature stack: ~1 hour. Total time-to-permanent-fix: ~1 week.

---

## Scoring

- 36–40 correct: Ship it. You've earned the "Production AI Engineer" title. Update your resume; apply widely.
- 30–35 correct: Strong. Review the 5–10 you missed and re-take next week.
- 24–29 correct: Reasonable coverage; revisit weak sections and do the build (or an extension) to solidify.
- < 24: Don't panic — production engineering is broad. Prioritize the two lowest-scoring sections; each has a companion learning module and practice prompts to work through.

---

## What's Next

You've now completed all six months. Your durable outputs:

- **Foundations** (Month 1) — Python, NumPy, Pandas, SQL, Git, Linux, math foundations
- **Machine Learning** (Month 2) — the classical stack with statistical intuition
- **Deep Learning** (Month 3) — PyTorch, backprop, CNNs, RNNs, Transformers
- **LLM Engineering** (Month 4) — tokens, prompts, structured outputs, RAG
- **Agents + Production** (Month 5) — agent patterns, guardrails, agent evaluation
- **Production AI Engineer** (Month 6, this month) — deploy, monitor, scale, secure, cost-control

Suggested next steps:
1. Ship your Month 6 production-hardened build to a personal cloud account. Get the URL running for at least 30 days; watch the dashboards; endure a real (small) outage.
2. Interview loop: with the six months as portfolio, aim at Senior AI/ML/MLOps roles at remote-friendly employers.
3. Iterate: keep one live project, tune monthly, write about what surprised you. The blog is your credibility.
4. Consider certifications only strategically: AWS/GCP ML specialty if targeting cloud-native shops; Anthropic's or Databricks' offerings when their weight in a target market grows.
5. Stay curious about the next generation: model architectures, inference-time compute, agent orchestration, evaluation science are all moving.

Well done, Brian. This is the shape of six months of deliberate work.
