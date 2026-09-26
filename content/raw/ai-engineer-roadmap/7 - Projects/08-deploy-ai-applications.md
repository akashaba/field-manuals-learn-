# Capstone Project 08 — Deploy an AI Application

> **Deliverable:** Take an AI service (any of Capstones 04–06) from "runs on my laptop" to "runs on the public internet at a URL, with HTTPS, autoscaling, secrets in a manager, CI/CD, and one-click rollback." Choose one platform (recommended: Fly.io for cost & simplicity), but the pattern generalizes to Cloud Run and ECS/Fargate.
>
> **Time:** 6–8 hours.
>
> **What you'll be able to say:** "I deployed an AI service to production on Fly.io. Two replicas, HTTPS, secrets via `fly secrets`, GitHub Actions builds and deploys on merge, rolling strategy with health checks, roll back in under 60 seconds with `fly releases rollback`. All in a public GitHub repo I can walk you through."

---

## 1. Project Overview

Deployment is where beginners drop off — because it feels like "not the interesting part" until you realize *everything else you built is worthless until it's reachable and reliable*.

The path this project takes:

1. Dockerize (multi-stage, non-root, small)
2. Local `docker-compose` with sidecars (Redis, Postgres, Prometheus)
3. Pick a cloud target; deploy via CLI
4. Add HTTPS + custom domain
5. Set up secrets management
6. Add health checks and readiness probes
7. Configure autoscaling policy
8. Wire GitHub Actions for CI/CD
9. Test rollback under a deliberately broken deploy

### Architecture

![IMG-CAP08-01](/7%20-%20Projects/images/IMG-CAP08-01.jpg)

### Platforms Compared (Choose One)

| Platform | Pros | Cons | Cost floor |
|----------|------|------|-----------|
| **Fly.io** | Simple; free tier; global by default; great CLI | Smaller ecosystem than AWS | ~$0 to start; ~$5-10/mo real workloads |
| **GCP Cloud Run** | Scale-to-zero; per-request billing; strong observability | Cold starts on infrequent traffic | $0 idle; low usage cheap |
| **AWS Fargate + ALB** | Enterprise-grade; huge ecosystem | Complex; higher operational floor | ~$30/mo minimum for ALB alone |

We proceed with **Fly.io** for the walk-through — the others are covered comparatively at the end.

### Prerequisites

- Docker Desktop (or engine) installed locally
- `flyctl` installed (`brew install flyctl` or curl script)
- A GitHub account
- ~$10 credit if you exceed the free tier

---

## 2. Step-by-Step Implementation

### Step 1 — Dockerize the service

Assume the service is a FastAPI app in `app/main.py`.

```dockerfile
# Dockerfile
# --- Stage 1: build ---
FROM python:3.11-slim AS builder
WORKDIR /build
# Install uv (fast Python installer) or pip
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# --- Stage 2: runtime ---
FROM python:3.11-slim AS runtime
# Non-root user for security
RUN groupadd --system app && useradd --system --gid app --uid 1001 app
WORKDIR /app

# Copy only the built venv from stage 1
COPY --from=builder /build/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# App code
COPY --chown=app:app app/ /app/app/

USER app
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz', timeout=2).raise_for_status()" \
  || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**Why multi-stage?** Final image ships without compilers or build tooling — smaller (~180MB vs ~1GB) and less attack surface.

**Why non-root?** If an attacker gets code execution, they inherit only user `app`'s privileges — can't `apt install` a backdoor, can't touch system paths.

**Why 2 workers, not more?** For an async FastAPI service, 1 worker per CPU is a common baseline; oversubscribing gains little because the workload is I/O-bound (each worker's event loop is idle waiting on Anthropic). Tune later based on observed CPU/memory.

`.dockerignore`:
```
.git
.venv
__pycache__
.pytest_cache
*.pyc
.env
.env.*
node_modules
.DS_Store
tests/
```

Build and test locally:
```bash
docker build -t my-ai-service .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... my-ai-service
curl http://localhost:8000/healthz
```

### Step 2 — Local Compose stack

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgresql://postgres:dev@db:5432/app
    depends_on: [redis, db]
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/healthz')"]
      interval: 15s
      timeout: 3s
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: dev
      POSTGRES_DB: app
    ports: ["5432:5432"]
    volumes: ["./pgdata:/var/lib/postgresql/data"]
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports: ["9090:9090"]
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Admin
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
```

`docker compose up` spins the whole thing. Grafana at http://localhost:3000, Prometheus at :9090, your API at :8000. This is your local prod mirror.

### Step 3 — Deploy to Fly.io

Create `fly.toml`:

```toml
app = "my-ai-service"          # unique; change to yours
primary_region = "sea"          # Seattle; pick your closest

[build]
  # Fly reads your Dockerfile

[env]
  PORT = "8000"
  LOG_LEVEL = "info"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = false     # keep running (turn on for scale-to-zero)
  auto_start_machines = true
  min_machines_running = 2       # HA: 2 replicas minimum
  processes = ["app"]

  [http_service.concurrency]
    type = "requests"
    hard_limit = 200
    soft_limit = 100

  [[http_service.checks]]
    grace_period = "10s"
    interval = "15s"
    method = "GET"
    timeout = "3s"
    path = "/healthz"

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 512

[deploy]
  strategy = "rolling"           # options: rolling, bluegreen, canary, immediate
  wait_timeout = "5m"
```

Login and launch:
```bash
fly auth login
fly launch --no-deploy   # generates or edits fly.toml
fly secrets set ANTHROPIC_API_KEY=sk-...
fly secrets set REDIS_URL=redis://...  # or use Fly Redis addon
fly deploy
```

After a few minutes, your service is at `https://my-ai-service.fly.dev`, HTTPS-terminated, on 2 replicas in Seattle.

**Why `force_https = true`?** Free automatic TLS termination + redirect. No reason not to.

**Why `min_machines_running = 2`?** Any single-replica deploy has a rolling window where either the old or new is briefly down; with 2, you always have one serving. Also handles hardware failures.

**Why concurrency-based scaling?** For an I/O-bound AI service, each replica can hold 100+ concurrent LLM calls with tiny CPU. Scaling by CPU would falsely think idle-waiting replicas need more capacity.

**Why rolling strategy?** Simplest safe default. For riskier changes, switch to `bluegreen` (deploy full parallel fleet, atomic switch) or `canary` (percentage-based traffic split).

### Step 4 — Custom domain + HTTPS

```bash
fly certs create ai.brianlab.dev
# Follow the DNS instructions — add A/AAAA records at your registrar
fly certs check ai.brianlab.dev
```

Wait 1-5 minutes for propagation, then https://ai.brianlab.dev serves your API with a real cert.

### Step 5 — Secrets management

Never bake secrets into images or commit `.env`. Instead:

```bash
fly secrets set ANTHROPIC_API_KEY=sk-...
fly secrets set DATABASE_URL=postgresql://...
fly secrets list  # names only, no values
fly secrets unset FOO
```

Fly injects these as env vars at container start; rotating is a single command that triggers a rolling restart.

For AWS/GCP: use AWS Secrets Manager or GCP Secret Manager with IAM/workload identity. Same principle — the container reads secrets from a manager at start, never from disk.

### Step 6 — CI/CD with GitHub Actions

`.github/workflows/deploy.yml`:

```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test-and-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install uv && uv sync --frozen
      - run: uv run ruff check .
      - run: uv run pytest -x
      - run: uv run python evals/run_golden.py
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

  scan:
    runs-on: ubuntu-latest
    needs: test-and-eval
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build image
        uses: docker/build-push-action@v5
        with:
          context: .
          tags: my-ai-service:${{ github.sha }}
          load: true
      - name: Trivy scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: my-ai-service:${{ github.sha }}
          severity: HIGH,CRITICAL
          exit-code: '1'

  deploy:
    runs-on: ubuntu-latest
    needs: [test-and-eval, scan]
    steps:
      - uses: actions/checkout@v4
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - run: flyctl deploy --remote-only
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

Now every merge to `main` triggers: tests → eval regression → security scan → deploy. Failures block the deploy.

Store `FLY_API_TOKEN` as a repo secret. Create it with `fly tokens create deploy`.

### Step 7 — Health checks

`/healthz` (liveness) and `/readyz` (readiness) are distinct concepts:

- **Liveness**: is the process alive? Restart if not.
- **Readiness**: is it *ready to serve*? Route around it if not (e.g., during startup or when a dependency is down).

```python
@app.get("/healthz")
async def liveness():
    return {"status": "ok"}  # cheap, always returns

@app.get("/readyz")
async def readiness():
    try:
        # Check every critical dependency
        await redis.ping()
        await db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")
```

Fly (and Kubernetes, Cloud Run) route traffic only to replicas passing readiness; and restart replicas failing liveness. Wire your Fly.io config to check `/healthz` and let readiness gate rolling deploys.

### Step 8 — Deliberate rollback drill

```bash
# 1. Deploy a deliberately broken version
git checkout -b broken
sed -i 's/return/raise RuntimeError("broken")/' app/main.py
git commit -am 'broken'
git push origin broken

# Merge or manually deploy
fly deploy

# 2. Watch health checks fail; new machines don't reach ready
fly logs
fly status

# 3. Roll back
fly releases  # list versions
fly releases rollback  # or --to-version=N

# 4. Confirm service healthy again
curl https://ai.brianlab.dev/healthz
```

**Why do this?** Rollback should be a reflex, not a discovery. Practice it in a low-stakes moment so you don't fumble it at 3 AM.

### Step 9 — Observability wiring (metrics & logs)

Expose `/metrics` (Module 7 pattern):

```python
from prometheus_client import make_asgi_app
app.mount("/metrics", make_asgi_app())
```

Fly.io has managed Prometheus (`fly.toml [metrics]` section) or you can scrape from Grafana Cloud. For logs: `fly logs -f` for live tail; ship them to a log aggregator (Datadog, Better Stack, Grafana Cloud Loki, or `flyctl logs` piped to your own store).

Add structured logging (Module 6):

```python
import structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
```

### Step 10 — Cost sanity check

At the shape of this deployment (2 shared-CPU 512MB machines, ~1M requests/month):

- Fly compute: ~$4/mo per machine × 2 = **~$8/mo**
- Fly outbound bandwidth: often free within limits
- Anthropic API calls: **dominant cost**, depends on workload; a small service can be under $10/mo
- Registry storage: negligible
- **Total**: often < $30/mo for a low-traffic real service. Compare to AWS Fargate's ~$30/mo just for the ALB.

Set a Fly billing alert; set an Anthropic spend cap in the console; monitor cost as a first-class SLI (Module 14).

---

## 3. Other Platforms — Same Concepts, Different Commands

### GCP Cloud Run

```bash
gcloud run deploy my-ai-service \
  --source . \
  --region us-central1 \
  --min-instances 1 \
  --max-instances 20 \
  --concurrency 80 \
  --allow-unauthenticated \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

Scale-to-zero: `--min-instances 0` (cold starts ~1-3s first request after idle). Cloud Run's `--concurrency` mirrors Fly.io's `hard_limit`. Secrets via GCP Secret Manager.

### AWS Fargate + ALB (via CDK)

More involved: task definition, cluster, service, load balancer, target group, ACM cert. Use CDK or Terraform. Good when you're already in AWS; overkill otherwise.

### Railway / Render

Even simpler than Fly for prototypes: connect the repo, they detect the Dockerfile, deploy. Less control, comparable price for small workloads.

`[IMG-CAP08-02]` — *Prompt: A side-by-side comparison of three deployment topologies. Left panel "Fly.io": user → globe (edge) → two "Fly Machine" boxes with your Docker container → sidecar Redis + Postgres icons. Middle panel "GCP Cloud Run": user → HTTPS LB → Cloud Run auto-scaling instances (with a "scale to zero" indicator when idle) → Cloud SQL + Memorystore icons. Right panel "AWS Fargate": user → Route53 → ALB → 2 Fargate tasks → RDS + ElastiCache. Below each panel a small stats box showing typical cost floor and cold-start latency. Clean architecture-diagram styling with consistent icons.*

---

## 4. Why Each Piece Matters

| Piece | Failure mode without it |
|-------|-------------------------|
| Multi-stage Dockerfile | 1GB images = slow deploys + more CVEs |
| Non-root user | Container escape gets root |
| Health checks | Broken deploys serve 500s to users |
| Min 2 replicas | Any deploy has a downtime window |
| Concurrency-based scaling | Over- or under-provisioning I/O workload |
| Secrets manager | Credential leak in git or image |
| Rolling deploy | All-at-once flip = high-risk switch |
| CI gates | Ships broken code to prod |
| Rollback rehearsed | Fumbled recovery at 3 AM |
| HTTPS | Data in transit exposed; browsers refuse |
| Distinct healthz/readyz | Load balancer routes to warming-up instances |

---

## 5. Extensions

- **Blue-green deploy.** Switch strategy in `fly.toml`; better for high-risk changes.
- **Canary deploy.** 5% → 25% → 100% with automatic rollback on error rate spike.
- **Multi-region.** Fly.io: add regions in one CLI call — `fly regions add lhr fra sin`; anycast routing.
- **Autoscaling policies.** Scale on custom metric (queue depth, LLM cost rate) via Fly's autoscale API or Cloud Run's custom metrics.
- **DR: automated backups + cross-region replicas of the DB.**
- **Runbook.** For every alert, write the on-call action: "if p99 latency alert fires, check upstream Anthropic status page first, then look at cache hit rate."

---

## 6. Interview Talking Points

- **"Walk me through your deploy."** Start with the Dockerfile (multi-stage, non-root, small), then the CI pipeline (lint/test/eval-gate/scan/deploy), then the target platform (Fly/Cloud Run), then health checks and rollback.
- **"How do you handle secrets?"** Never in git, never in images. Secret manager injecting env vars at container start. Rotation is a CLI command triggering rolling restart.
- **"What's your deploy strategy?"** Rolling by default (min 2 replicas, health-checked); blue-green for high-risk; canary when we need statistical confidence.
- **"How do you roll back?"** One command (`fly releases rollback` or equivalent). Practiced regularly; every developer can do it.
- **"How do you know a deploy is healthy?"** Distinct liveness and readiness probes; readiness gates rolling deploys; post-deploy dashboard shows error rate/latency staying within SLO.
- **"Why Fly.io over AWS?"** Cost floor is 10x lower for small teams; global-by-default; simpler mental model. AWS is right when you're already in AWS or need specific AWS services.

---

## 7. References

- Fly.io docs — https://fly.io/docs
- GCP Cloud Run — https://cloud.google.com/run/docs
- Docker best practices — https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
- OWASP Docker Cheat Sheet
- Google SRE Book — Ch. on load balancing, rolling releases
- Charity Majors — everything on deploy safety and observability
