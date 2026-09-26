# Build 01 — Production Hardening of an AI Service

> **Purpose:** Take one of your previous builds (Month 4's "Chat with your documents" RAG, Month 5's autonomous agent, or a fresh minimal FastAPI + LLM service) and put it through a **complete production hardening pass**. When you finish, you have a service that survives outages, catches quality regressions before users do, controls costs, resists common attacks, and can be deployed to real cloud infrastructure by pressing "merge."
>
> This is the capstone that ties Month 6's fourteen topics into one deliverable. The goal is not to touch every checkbox academically — it's to build the muscle memory of thinking about a service through the production lens: observability, cost, safety, resilience, cost, cost.

---

## Choose Your Base Service

Pick one:

- **Option A (recommended for continuity):** Month 4's "Chat with your documents" RAG. Rich in observability and cost surface (multi-step: retrieval + generation).
- **Option B:** Month 5's autonomous agent. Rich in tool-use / security / rate-limiting surface.
- **Option C (fresh start):** A minimal `POST /chat` endpoint calling Anthropic (or any LLM API), no bells. Add features as you follow this build.

The rest of this doc assumes Option A; substitute mentally for B or C.

---

## Target Architecture

```
[Client] 
   │  (HTTPS + JWT bearer)
   ▼
[Cloudflare/WAF]  ← DDoS, TLS termination
   │
   ▼
[FastAPI service] (2+ replicas on Fly.io / Cloud Run / Fargate)
   │
   ├── /metrics       (Prometheus scrape)
   ├── /healthz       (liveness — always cheap)
   ├── /readyz        (readiness — dependency checks)
   ├── /chat          (main; auth + rate-limit)
   ├── /feedback      (attach thumbs to trace)
   └── /admin/*       (IP-restricted)
   │
   ├──▶ Redis          (rate-limit counters, response cache, cost counters)
   ├──▶ pgvector       (RAG index, per-tenant row-level security)
   ├──▶ Anthropic API  (with retry + circuit breaker)
   ├──▶ LangFuse       (trace + prompt versioning + eval store)
   └──▶ MLflow         (model + eval artifact registry)

[GitHub Actions]  → lint → type-check → test → eval-golden → build → scan → deploy
[Prometheus]      → scrape /metrics from replicas
[Grafana]         → RED + LLM cost + cache hit + SLO dashboards
[Alertmanager]    → PagerDuty on SLO burn / cost spike / error spike
```

`[IMG-BUILD-01]` — *Prompt: A production architecture diagram of the AI service described above. Show Cloudflare in front, two FastAPI replicas in the middle, Redis and pgvector sidecar, external Anthropic API, LangFuse and MLflow as observability side-stacks, GitHub Actions in a CI/CD lane on the left, Prometheus+Grafana+Alertmanager on a monitoring lane on the right. Use clean architecture-diagram styling with labeled arrows for auth, rate-limit-check, cache-check, LLM-call, trace-emit. Neutral colors.*

---

## Milestone Plan (10 milestones, ~2 weeks total effort)

You may knock any single milestone out in an evening; several fit in one commit.

### Milestone 1 — FastAPI Correctness Pass (Module 1)

- [ ] Convert all handlers to `async def` and switch to `httpx.AsyncClient` for the LLM call.
- [ ] Use `lifespan` for shared resources (LLM client, Redis, DB pool).
- [ ] All request/response bodies are Pydantic v2 models with `ConfigDict(extra="forbid")`.
- [ ] Add `x-request-id` middleware (generate if missing; bind to structlog contextvars).
- [ ] `/healthz` (liveness) and `/readyz` (readiness with Redis + pgvector pings).
- [ ] TestClient integration test for happy path + one error path.

**Exit criterion:** `uv run pytest` passes; `curl /readyz` returns 200 only when deps are up.

### Milestone 2 — Docker Hardening (Module 2)

- [ ] Multi-stage `Dockerfile` on `python:3.11-slim` base.
- [ ] `.dockerignore` excludes `.git`, `.venv`, `__pycache__`, `*.md` except needed, secrets.
- [ ] Non-root user (UID 1001), read-only rootfs if feasible.
- [ ] `uv sync --frozen --no-dev` for prod deps.
- [ ] Image scanned with Trivy in CI; no HIGH/CRITICAL CVEs before merge.
- [ ] Image size < 300 MB.
- [ ] `HEALTHCHECK` in Dockerfile calling `/healthz`.
- [ ] `docker-compose.yml` for local dev: app + Redis + Postgres/pgvector + Prometheus + Grafana + LangFuse (self-hosted).

**Exit criterion:** `docker compose up` gives you the full local stack; `docker images` shows a slim image; `trivy image` reports clean.

### Milestone 3 — Cloud Deploy + Rollback (Module 3)

Choose one platform:
- Fly.io: `fly.toml` with a rolling deploy strategy, min=2 replicas, health checks, secrets set via `fly secrets set`.
- Cloud Run: `gcloud run deploy --min-instances=1 --concurrency=80 --port=8080`.
- Fargate: task definition + ALB + service, deploy via CDK/Terraform.

Steps:
- [ ] Domain with HTTPS (Fly.io/Cloud Run gives it for free; on Fargate use ACM cert on ALB).
- [ ] Non-secret config via env vars; secrets via platform's secret manager (Fly.io secrets, GCP Secret Manager, AWS Secrets Manager).
- [ ] Rolling deploy strategy verified.
- [ ] Manual rollback tested — deploy a deliberately broken version; roll back with the platform's built-in mechanism.

**Exit criterion:** service is reachable at a public URL over HTTPS; a broken deploy can be reverted in < 60s.

### Milestone 4 — CI/CD Pipeline (Module 4)

Create `.github/workflows/ci.yml` with jobs:

- [ ] `lint`: ruff check + ruff format --check
- [ ] `typecheck`: mypy or pyright
- [ ] `test`: pytest with coverage; upload junit XML
- [ ] `eval-golden`: run `evals/run_golden.py` against a local mocked LLM (or real API in a scheduled job with a tight budget); 100% pass required
- [ ] `build-image`: build and push to GHCR / ECR / Artifact Registry
- [ ] `scan-image`: Trivy scan; fail on HIGH/CRITICAL
- [ ] `deploy-staging`: on merge to main, deploy to staging environment with OIDC-federated credentials (no long-lived keys)
- [ ] `deploy-prod`: manual approval gate, then deploy

Preview environments per PR (optional but recommended): use Fly.io's `fly launch --config fly.preview.toml` per PR ref, or Cloud Run's tag-based revisions.

**Exit criterion:** opening a PR triggers all checks; merging to main promotes to staging; a manual click promotes to prod.

### Milestone 5 — MLflow + Prompt Version Registry (Module 5, 9)

Two paths — pick one, or do both:

- [ ] **MLflow**: log experiments for your embedding model choice, retrieval reranker choice, and prompt version. Use aliases (`@champion`, `@challenger`). Load current champion at service start.
- [ ] **LangFuse prompts**: extract your system prompt into LangFuse; fetch at runtime with a `production` label; fall back to bundled default if LangFuse is unreachable.

Instrument:
- [ ] `prompt_version` on every trace (as attribute)
- [ ] `model_registry_alias` on every trace

**Exit criterion:** you can promote a new prompt (or model) via a config change or a single API call, and see the version reflected in traces without a redeploy.

### Milestone 6 — Structured Logging + Tracing (Modules 6, 9)

- [ ] `structlog` with contextvars binding, JSON renderer in prod, human-readable in dev.
- [ ] Middleware sets `request_id`, `user_id_hash`, `tenant_id` on the log context for every request.
- [ ] W3C `traceparent` header propagation.
- [ ] PII redaction pass on log lines (SSN, credit card, email, API key patterns).
- [ ] OTel tracing wired up; export to LangFuse (or Phoenix / LangSmith).
- [ ] Trace attributes follow `gen_ai.*` semantic conventions.
- [ ] Log levels correctly set: DEBUG dev-only, INFO for lifecycle, WARN for retries, ERROR for failures.
- [ ] Log sampling: 100% errors, 5% baseline for INFO paths.

**Exit criterion:** for any request in prod, you can pull its trace + logs by request_id in < 30s.

### Milestone 7 — Prometheus Metrics + Grafana Dashboard (Module 7)

- [ ] Instrument `http_requests_total`, `http_request_duration_seconds` (histogram), `http_requests_inflight`.
- [ ] LLM-specific: `llm_tokens_input_total`, `llm_tokens_output_total`, `llm_cost_usd_total`, `llm_request_duration_seconds` (histogram, labeled by `model`, `route`).
- [ ] Cache metrics: `cache_hits_total`, `cache_misses_total`, per cache layer.
- [ ] Rate-limit metrics: `rate_limit_rejections_total{reason}`.
- [ ] Exposition at `/metrics`.
- [ ] Prometheus (in docker-compose for local, managed in cloud) scrapes it.
- [ ] Grafana dashboard with 4 rows: RED, LLM cost/tokens, cache hit rates, GPU (if applicable) or resource USE.
- [ ] Alertmanager rules for: fast-burn SLO alert (2h window, 14.4× rate), slow-burn (6h, 6×), cost spike (15m rate > 3× baseline), cache-hit-rate collapse.

**Exit criterion:** loading the Grafana dashboard, you can answer "how is the service doing" in one glance.

### Milestone 8 — Rate Limiting + Cost Cap + Cache Layers (Modules 10, 11, 14)

- [ ] FastAPI middleware enforcing per-user RPM (60/min for free tier, 600 for pro), TPM (100k for free, 1M for pro), and $/day (5.00 for free, 50.00 for pro).
- [ ] Redis Lua script for atomic sliding-window counter.
- [ ] `Retry-After` and `RateLimit-*` headers on 429s.
- [ ] Response cache with version-in-key: `llm:{model}:{prompt_version}:{corpus_version}:{tenant_id}:{sha256(user_input)}`.
- [ ] Anthropic prompt caching on your system prompt + tool defs (Module 11 §2.3).
- [ ] Semantic cache on the RAG answer for high-hit-rate FAQ patterns (optional).
- [ ] Model routing: cheap intent classifier → Haiku for FAQ, Sonnet for reasoning, Opus for escalation.
- [ ] Metrics: cache hit rate per layer, $ per user per day (gauge).

**Exit criterion:** hitting the service with a load test at 10× normal rate shows: 429s emitted correctly, cache hit rates > 30%, no runaway cost.

### Milestone 9 — Authentication + Multi-Tenant Isolation (Module 12)

- [ ] OIDC integration (Auth0 / Cognito / Keycloak / your own). Backend validates JWT via JWKS.
- [ ] Every endpoint requires an authenticated user via `Depends(get_current_user)`.
- [ ] Scope check via `Depends(require_scope("write:chat"))`.
- [ ] `tenant_id` extracted from JWT claims, never from request body.
- [ ] Postgres row-level security enabled on all tenant-partitioned tables.
- [ ] Auth headers redacted in logs (Module 6 middleware).
- [ ] Negative auth tests in CI: no token → 401; wrong scope → 403; cross-tenant access attempt → 404 (never leak existence).

**Exit criterion:** you have two test users in different tenants; each can only see their own data even if a query is buggy; JWT with `alg:none` is rejected.

### Milestone 10 — Security + Evals + Runbook (Modules 8, 13)

- [ ] Golden eval set `evals/golden_set.jsonl` with ≥ 30 cases spanning: happy paths, edge cases, past bugs, refusal-required inputs (medical advice, PII request), prompt-injection attempts.
- [ ] `evals/run_golden.py` runs in CI; 100% pass required.
- [ ] Full offline eval with LLM-as-judge running nightly; results uploaded to MLflow or LangFuse.
- [ ] Security headers middleware (Module 13).
- [ ] Trivy image scan in CI; SBOM emitted (`syft` or `trivy sbom`).
- [ ] `gitleaks` pre-commit hook.
- [ ] Adversarial-prompt tests: 5 known-jailbreak prompts + 3 injection payloads; assert no unauthorized tool called and no system-prompt echo.
- [ ] URL-fetching / SSRF: allow-list only your intended sources; block private IPs (10/8, 172.16/12, 192.168/16, 169.254/16).
- [ ] Runbook document: for each alert (SLO burn, cost spike, error spike), a short "check X, then Y, then Z" checklist.

**Exit criterion:** you can rotate an API key, roll back a bad prompt, respond to a cost anomaly page, and answer an adversarial-prompt report — each in under 15 minutes, following the runbook.

---

## Load Testing Ritual

Before declaring "hardened," run one deliberate load test:

- [ ] Use `locust` or `k6` to generate 5× your normal peak RPS for 20 min.
- [ ] Assert: p99 latency stays within 20% of baseline; error rate < 1%; 429s emitted correctly for over-limit users; cost stays bounded (thanks to caps); autoscaling triggers as configured.
- [ ] Save the run's Grafana screenshots to `docs/load-tests/YYYY-MM-DD.png`.
- [ ] Repeat quarterly; regression indicates something broke.

---

## Chaos Drills

Once the system is live, deliberately break things to verify resilience:

- [ ] Kill one replica during a deploy → traffic reroutes; no error spike.
- [ ] Set Redis unreachable → app falls back to in-process rate-limit or fails closed (never fails open); no user-visible outage beyond expected degradation.
- [ ] Simulate Anthropic 429s at 50% rate → your retry/backoff handles it; slow tail visible but no 5xxs.
- [ ] Feed an adversarial injection payload → guardrail catches; refusal metric ticks; no exfiltration.
- [ ] Deliberately break a golden case → CI blocks the merge.
- [ ] Rotate a secret in the manager → app picks up the new value on next lease/restart without an outage.

---

## Cost Ledger (Track These Weekly)

Build a small script `scripts/cost_report.py` that pulls from your traces/metrics:

- $ per day, per model, per route, per top-10 users
- Cache hit rate per layer
- Prompt-cache read-token fraction (should climb to > 60% after Milestone 8)
- LLM-judge quality score trend
- SLO burn-rate 30-day trend

Commit weekly output to `reports/cost-YYYY-MM-DD.md`. This is your record — and it's exactly the artifact an interviewer will ask to see.

---

## Deliverables Checklist

By the end of this build, your repo has:

- [ ] `Dockerfile` (multi-stage, non-root, small)
- [ ] `docker-compose.yml` (full local stack)
- [ ] `.github/workflows/ci.yml` (lint, test, eval, build, scan, deploy)
- [ ] `fly.toml` / `service.yaml` / `task-def.json` (whichever platform)
- [ ] `pyproject.toml` (pinned deps, lockfile committed)
- [ ] `app/` (FastAPI code with all middlewares wired)
- [ ] `evals/golden_set.jsonl` + `evals/run_golden.py`
- [ ] `dashboards/grafana-slo.json` (exported dashboard config)
- [ ] `prometheus/alerts.yml` (SLO burn, cost, cache, error rules)
- [ ] `docs/runbook.md` (one page per alert)
- [ ] `docs/threat-model.md` (OWASP LLM Top 10 walkthrough for this service)
- [ ] `reports/` (weekly cost + quality snapshots)
- [ ] `README.md` (link the whole thing together; "how do I run this")

---

## Reflection Questions (Answer in `docs/reflection.md`)

Write 200–400 words each. This is what turns exercises into internalized skill.

1. Which of the fourteen production topics moved the needle most on real quality-of-service for your build, and why?
2. Which one surprised you with how much complexity it hid, and how would you explain that complexity to a junior developer?
3. Where is your service still weakest? If you had one more week, which two milestones would you invest in?
4. Cost your service today serves 100 requests/day. Cost it at 10,000 requests/day, and at 1,000,000/day. Which levers dominate at each scale?

---

## Extensions (Choose Any)

- **Observability**: swap LangFuse → LangSmith → Phoenix. Compare UX.
- **Model routing**: add a bandit-based router (Thompson sampling on quality/cost trade-off).
- **Self-hosted fallback**: run vLLM with an open model (Llama 3.1 8B) as your Haiku-tier fallback; measure cost delta.
- **Chaos day**: full drill — one team member introduces a broken deploy; the other diagnoses and rolls back using only the runbook + dashboards.
- **Multi-region**: deploy to two regions with active-active; add DNS-based failover.
- **Compliance profile**: draft a HIPAA or SOC 2 readiness memo for your service. Which controls are in place; which are missing?

---

**Related files:**
- All 14 learning modules — treat this build as their integration exam
- Prior builds — `../machine-learning/builds/`, `../deep-learning/builds/`, `../llm-engineering/builds/`, `../agents-production/builds/`
- `self-assessment.md` — the exit exam that mirrors this build's coverage
