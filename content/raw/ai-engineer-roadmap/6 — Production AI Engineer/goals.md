# Month 6 — Production AI Engineer 🚀 — Goals

> **Track:** Production AI Engineer (Module 6 of the broader learning journey)
> **Prerequisites:** Months 1–5. Especially the API and deployment sections of Month 4 (LLM Engineering) and Month 5 (Agents + Production).
> **Meta-goal:** *Take AI systems from "works on my machine" to "runs reliably, safely, and profitably for a team that depends on it."*

---

## 1. The North Star

The single sentence that governs every decision this month:

> **"When my AI system goes into production, I know what it costs per request, what its p95 latency is, who it's serving, whether it's being attacked, when it degrades, how to roll back safely, how to secure it, and how to make it cheaper next week — without asking anyone."**

This is the ops discipline. Prior months taught you to *build* AI. This month teaches you to *run* it.

---

## 2. Terminal Learning Objectives (mastery, not exposure)

### 2.1 Serving
- **LO-1.1** Build a production FastAPI service — async, typed, versioned, health-checked, streaming-capable.
- **LO-1.2** Package it as a small, secure, non-root Docker image.
- **LO-1.3** Deploy to a real cloud (AWS/GCP/Azure/Fly.io) with TLS, autoscaling, and a load balancer.
- **LO-1.4** Ship via CI/CD — tests, lint, build, push, deploy on merge.

### 2.2 Model Lifecycle
- **LO-2.1** Use MLflow for experiment tracking, model registry, model versions, and staged deployment.
- **LO-2.2** Evaluate models in production — offline eval in CI + online eval on sampled traffic.
- **LO-2.3** LLM observability — traces, prompts, tokens, costs — via LangSmith/Braintrust/Phoenix or self-hosted.

### 2.3 Runtime Concerns
- **LO-3.1** Structured logging with correlation IDs.
- **LO-3.2** Metrics for latency, cost, success rate, model version.
- **LO-3.3** Rate limiting per-user / per-tenant / global.
- **LO-3.4** Multi-tier caching — HTTP, prompt, semantic, embedding.
- **LO-3.5** Authentication (API keys, OAuth, JWT) and per-request authorization.

### 2.4 Safety
- **LO-4.1** Threat model an AI service: OWASP LLM Top 10.
- **LO-4.2** Handle secrets, PII, and audit logs correctly.
- **LO-4.3** Contain blast radius: least privilege, sandboxes, kill switches.

### 2.5 Economics
- **LO-5.1** Instrument cost per request; aggregate by user, endpoint, model.
- **LO-5.2** Apply the cost-reduction toolkit — caching, model routing, batching, quantization.
- **LO-5.3** Set budgets and alerts; enforce caps.

---

## 3. Deliverable

You will produce **one** portfolio-grade artifact:

**Build #1 — Production Hardening of an AI Service.** Take a prior build (from Month 4 or 5) and re-ship it as a **production-ready** system with the full ops stack. See `builds/01-production-hardening.md`.

Why this build (the user didn't specify one, so it's proposed): the previous months' builds already have real functionality. Month 6 is the operational discipline that turns those from working prototypes into services a team would trust in production.

---

## 4. Definition of Done (per topic)

A topic is **done** when you can:

- [ ] Configure or implement it in code that runs.
- [ ] Explain the failure mode it defends against.
- [ ] Measure or observe it in a running system.
- [ ] Score ≥ 8/10 on the module's Self-Assessment Bank.

---

## 5. Anti-goals (what you are *not* doing here)

- **You are not building a new AI capability.** This month is about running existing systems well.
- **You are not learning every cloud vendor deeply.** Pick one (Fly.io / Render for a solo project; AWS/GCP for an org context); apply the same patterns everywhere.
- **You are not writing Terraform / Pulumi from scratch** unless your build calls for it. Simple deployment tools first.
- **You are not becoming a full SRE.** You're becoming an AI engineer who can operate their own systems.
- **You are not adopting every observability vendor.** Pick a small stack; standardize; move on.

---

## 6. Cadence (suggested)

| Week | Focus |
|------|-------|
| 1 | FastAPI, Docker, Cloud, CI/CD (the runtime chassis) |
| 2 | MLflow, Logging, Monitoring, Model Eval (the ops disciplines) |
| 3 | LLM Observability, Rate Limits, Caching (the LLM-specific ops) |
| 4 | Auth, Security, Cost Optimization + **Build #1** |

---

## 7. Success Signals

You'll know Month 6 is behind you when:

- You never ship a service without health/readiness probes.
- Your Dockerfiles are multi-stage, non-root, and < 500 MB.
- Every request has a `trace_id` that flows through your entire stack.
- You have a Grafana or Datadog dashboard for every service you own.
- Your cost per request is a number you know for every endpoint.
- You have a per-user rate limiter and a per-tenant cost cap.
- You've been paged at least once — and had a runbook to solve it.
- When someone asks "how do we roll back", you answer with a command, not a shrug.
- You've thought about the OWASP LLM Top 10 for at least one of your services.
- You feel physical discomfort when someone commits `.env` or hardcodes an API key.

---

## 8. Related Files

- `learning/01-fastapi.md`
- `learning/02-docker.md`
- `learning/03-cloud-deployment.md`
- `learning/04-ci-cd.md`
- `learning/05-mlflow.md`
- `learning/06-logging.md`
- `learning/07-monitoring.md`
- `learning/08-model-evaluation.md`
- `learning/09-llm-observability.md`
- `learning/10-rate-limiting.md`
- `learning/11-caching.md`
- `learning/12-authentication.md`
- `learning/13-security.md`
- `learning/14-cost-optimization.md`
- `builds/01-production-hardening.md`
- `self-assessment.md`
