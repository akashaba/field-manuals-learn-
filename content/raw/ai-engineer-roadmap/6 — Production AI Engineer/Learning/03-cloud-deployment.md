# Cloud Deployment — Master Study Guide

> **Track:** Production AI Engineer · **Module:** 03
> **Prerequisites:** Modules 01–02.
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Your service has to run somewhere the world can reach. Cloud deployment is where infrastructure, cost, reliability, and operational maturity all meet. For AI services specifically:

- Autoscaling — traffic to LLM services is bursty.
- Cold starts — LLM containers can take 30+ s to warm.
- GPU access when running local models.
- Multi-region for latency-sensitive apps.
- Secure networking — private DBs, secrets managers, IAM.

You don't need to master every cloud. **Pick one, understand deeply, apply patterns everywhere.** This module covers the mental model + specific choices for common platforms (Fly.io, Render, AWS ECS/Fargate, GCP Cloud Run, Kubernetes at scale).

**Fundamental principles you must own:**

1. **Managed platforms first.** Only reach for raw Kubernetes when you have team + scale to justify it.
2. **Stateless containers, stateful stores.** Containers scale in/out; data lives in managed DBs.
3. **Environment via env vars + secrets manager.** Never in code, never in the image.
4. **Autoscale on the right signal.** CPU works for CPU-bound; concurrent-requests / queue-depth for I/O-bound (i.e., LLM services).
5. **TLS terminated at the LB.** App speaks HTTP internally.
6. **Zero-downtime deploys** — rolling updates, health probes, graceful shutdown.
7. **Cost = provisioned resources × time.** Autoscale down to zero when possible.

If you retain nothing else: **managed platform + stateless containers + env-var config + right autoscale signal. Move complexity out of your app.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Landscape (2026 Snapshot)

**PaaS (Platform-as-a-Service) — easiest.**

- **Fly.io** — deploys Docker images globally; autoscale; managed Postgres; GPU support. Solo devs and small teams love it.
- **Render** — similar; strong web UI; easy background workers.
- **Railway** — Heroku spiritual successor.
- **Vercel** — great for Next.js frontends; serverless functions with limitations for long-running.

**Serverless containers — mid.**

- **GCP Cloud Run** — deploys any container, scales to zero, per-request billing. Excellent for spiky AI workloads.
- **AWS Lambda + container images** — up to 15-minute execution; scales instantly; not great for long-running agents.
- **Azure Container Apps** — GCP Cloud Run analog.

**Managed orchestration.**

- **AWS ECS / Fargate** — Docker containers on AWS-managed infrastructure. No servers to manage.
- **AWS App Runner** — even simpler than ECS.
- **Google Cloud Run + Cloud Run Jobs** for scheduled work.
- **Azure Container Apps + Container Instances**.

**Kubernetes.**

- **EKS (AWS), GKE (GCP), AKS (Azure)** — managed Kubernetes.
- **Rancher, OpenShift** — enterprise K8s.

**Right-sized for team:**

| Team size / scale | Choice |
|-------------------|--------|
| Solo, hobby / MVP | Fly.io, Render, Railway |
| Small team, mid-scale, spiky traffic | Cloud Run, Fargate, App Runner |
| Growing team, complex workloads | ECS + Terraform, or GKE Autopilot |
| Big team, many services, strict SLAs | EKS/GKE + service mesh + platform team |

**Don't skip levels.** A 2-person team on raw Kubernetes drowns in Ops. A team of 20 on Fly.io outgrows it.

---

### 2.2 Deploying to a Managed Platform (Fly.io / Cloud Run walk-throughs)

**Fly.io** (as of 2026):

```toml
# fly.toml
app = "my-ai-service"
primary_region = "sea"  # Seattle

[build]
  image = "ghcr.io/org/service:v1.0.0"

[env]
  LOG_LEVEL = "info"
  MODEL_ID = "gpt-4o-mini"

[[services]]
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [services.http_checks.[]
    interval = "30s"
    grace_period = "10s"
    method = "get"
    path = "/healthz"

[[services.concurrency]]
  hard_limit = 100
  soft_limit = 50

[vm]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 512
```

Deploy:

```bash
fly secrets set OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

Autoscale:

```bash
fly scale count 2 --max-per-region 5
fly autoscale set min=1 max=10
```

Fly runs your container in a firecracker VM (fast start), replicates across regions, and gives you TLS + HTTP LB out of the box. Solid choice for teams up to a few engineers.

**GCP Cloud Run:**

```bash
gcloud run deploy my-ai-service \
  --image ghcr.io/org/service:v1.0.0 \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 100 \
  --concurrency 40 \
  --set-env-vars="LOG_LEVEL=info" \
  --set-secrets="OPENAI_API_KEY=openai-key:latest"
```

Cloud Run scales to zero when idle (saves cost) and up to hundreds/thousands of instances quickly. Great for bursty AI workloads.

**Cold-start caveat.** Scaling from zero means the first request waits for a container to boot. For LLM services with a 30 s cold start, use `min-instances 1` (or more) to keep a warm instance around. Trade cost for latency.

**AWS Fargate / ECS.** Higher setup complexity; more control. Skip unless you're already on AWS.

---

### 2.3 Networking, TLS, and DNS

**HTTPS is table stakes.** Never expose HTTP directly. Options:

- **PaaS auto-TLS** (Fly, Render, Cloud Run) — free certs via Let's Encrypt; auto-renewed.
- **AWS ALB + ACM** — provision cert via AWS Certificate Manager; attach to ALB.
- **Cloudflare** in front of your service — free TLS, DDoS protection, caching.
- **Caddy / Nginx** if self-managed.

**DNS.**
- **CNAME** to your platform's provided hostname (e.g., `my-service.fly.dev`).
- **Route 53 / Cloudflare** for authoritative DNS.
- **Health-checked failover** for multi-region.

**Reverse proxy inside the container?** Rarely needed for FastAPI. Uvicorn is production-grade; no Nginx sidecar necessary. The exception: complex TLS handling or WebSocket termination.

**Network isolation.**
- **Public LB → private compute.** Your service is only reachable via the LB.
- **Private DBs.** Databases in a private subnet, accessible only from your compute.
- **VPC / private networking** for internal services.

**Egress.**
- Your service calls out to LLM providers, GitHub, etc. Egress bandwidth is free on most platforms.
- **NAT gateways** on AWS: egress from private subnets costs money. Bulk-download workloads can hit real bills.

**IPv6.** Consider IPv6-only egress if your platform supports it (saves NAT costs). AWS EKS supports this well.

---

### 2.4 Autoscaling for AI Workloads

**CPU-based autoscale** works for CPU-bound services. Not for AI:

- LLM services are I/O-bound (waiting on the LLM API). CPU is idle even under heavy load. CPU-based autoscaling never kicks in even as latency degrades.

**Right signals for AI/LLM:**

1. **Concurrent requests per instance.** Cloud Run uses this natively — set `concurrency=40`, cloud auto-scales when instances hit their limit.
2. **Queue depth** — if you have a background job queue, scale on unprocessed job count.
3. **P95 latency** — scale up when latency breaches SLA.
4. **Request rate** — simple but effective as a heuristic.
5. **Custom metric** — active-in-flight count via Prometheus + KEDA (K8s).

**Autoscale bounds:**
- **min-instances** — keep enough warm to absorb sudden traffic. For frontier LLM services, 1–3 is typical.
- **max-instances** — cap runaway scale-out; prevents cost explosion.

**Cold-start mitigation:**
- **min-instances ≥ 1** — always have a warm instance.
- **Provisioned concurrency** (Lambda) — keeps N instances pre-warmed.
- **Lightweight images + fast startup** — Module 02.
- **Lazy model loading** — start responding to health checks quickly; load the model in background if traffic hasn't demanded it yet.

**Multi-region.** For latency-sensitive apps:
- Deploy to multiple regions.
- DNS-based geo-routing (Route 53, Cloudflare, Fly's built-in).
- Region-local DBs (or a global DB like Fly Postgres, Aurora Global, Cockroach).

Latency: US East to Europe = ~100ms one-way. That's a big chunk of any P95 budget. Multi-region matters when users are global and latency-sensitive.

**Cost of autoscaling.** More instances → more $. Balance:
- Under-scale → users see errors during spikes.
- Over-scale → paying for idle capacity.
- Sweet spot: min-instances covers baseline; autoscale handles bursts.

---

### 2.5 Deploy Strategies, Rollback, and Blast Radius

**Rolling deploy (default).** Replace instances one at a time. New version drains old traffic (respects graceful shutdown), then old shuts down. Zero-downtime if health probes and shutdown are correct.

**Blue-green.** Two environments (blue = current, green = new); swap traffic atomically. Instant rollback (swap back). Doubles infra cost during deploy but lowest risk.

**Canary.** Route small % (5%) of traffic to new version. Monitor metrics. If good, ramp up. If bad, roll back. Best for stateful or high-risk changes.

**Feature flags.** Deploy the code but keep the new behavior gated by a flag. Enable per-user, per-tenant, or globally. Rollback = flip the flag (no redeploy).

**Rollback.** Every deploy must be rollbackable in < 5 minutes. Practices:
- **Immutable images** — every deploy is a new image tag; rollback = re-deploy old tag.
- **Database migrations** — always backwards-compatible for one release. Never drop a column in the same deploy that stops writing to it.
- **Feature flags** — flip flags rather than redeploy for behavior rollback.
- **CDN cache purge** if you're serving static assets that changed.

**Blast radius controls:**

- **Per-environment isolation** — dev, staging, prod. Never share a DB across environments.
- **Per-tenant isolation** — for multi-tenant SaaS, one bad tenant shouldn't degrade others.
- **Circuit breakers** — if a downstream is failing, fail fast to prevent cascade.
- **Load shedding** — under overload, reject some requests cleanly rather than degrade all.

**Deploy discipline:**
- Every deploy has a `git sha` visible in `/version` endpoint.
- Every deploy is announced in Slack (or your channel) automatically.
- Every deploy runs the CI eval suite (Module 04).
- No manual "hot fix" merges without CI.
- Post-deploy: watch metrics for 30 minutes.

---

## 3. Mental Models & Analogies

### 3.1 The "Restaurant Franchise" Model

Think of your service as a **restaurant chain**:

- **Container image** = the franchise operations manual — same recipes everywhere.
- **PaaS (Fly.io, Cloud Run)** = a mall pod — franchise pays rent, mall handles infra (electricity, cleaning, security).
- **Managed Kubernetes** = renting a strip-mall unit — you have more control but more ops.
- **Raw servers** = building your own restaurant — everything on you.
- **Autoscaling** = adding/removing tables as customers arrive.
- **Multi-region** = franchises in different cities to be near customers.

A new franchise doesn't build a mall; they use one that exists. Your first AI service doesn't need raw K8s; use a managed platform. Graduate up only when you outgrow it.

### 3.2 The "Ship + Cargo" Model (Stateless Containers)

Containers are **cargo ships** — they move products, but the products themselves aren't stored on the ship. Cargo (data) lives in warehouses (DBs, object storage).

If a ship sinks, cargo is unaffected — reload onto a new ship. If the DB dies, data is lost.

Stateless containers: any container can serve any request. Autoscaling in/out just changes the number of ships in transit. Deploys replace ships mid-voyage without cargo damage.

State in containers = state on the ship = state lost when ship sinks. Never do it. Keep state in managed stores.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "We Need Kubernetes"

For most teams: no. K8s is a fantastic platform once you have 10+ engineers, dozens of services, and dedicated ops capacity. For a 3-person team with 2 services, managed platforms (Fly.io, Cloud Run, ECS Fargate) deliver 90% of the value with 10% of the ops overhead. Adopt K8s when the pain of not having it exceeds the pain of running it.

### 4.2 "CPU Autoscaling Handles AI Workloads"

LLM services spend most of their time waiting on the LLM API. CPU utilization stays low even under crushing load. CPU-based autoscaling never triggers → users see latency and timeouts. Use **concurrency-based** or **request-rate** autoscaling for AI. Cloud Run's built-in concurrency signal is designed for this.

### 4.3 "Deploying a New Version Is Deploying"

Deploying without health probes = brief window of broken traffic. Deploying without rollback plan = broken until manually reverted. Deploying without post-deploy metric review = surprise incident hours later. Deploy is a process: build → test → deploy → observe → confirm. Each step exists for a reason.

---

## 5. Self-Assessment Bank (Cloud Deployment)

### Questions

**Q1 (Short answer).** For a 3-person team shipping a new AI service, why is a managed platform (Fly.io / Cloud Run) usually better than raw Kubernetes?

**Q2 (Multiple choice).** For an LLM service that's mostly I/O-bound, the right autoscale signal is:
- (a) CPU utilization.
- (b) Concurrent requests per instance / request rate.
- (c) Memory usage.
- (d) Number of workers.

**Q3 (Short answer).** Give three reasons stateless containers + managed state stores > stateful containers.

**Q4 (Multiple choice).** For a service with a 30-second cold start on an LLM model:
- (a) Use `min-instances = 0` to save cost.
- (b) Use `min-instances ≥ 1` (or provisioned concurrency) to keep warm; trade a bit of cost for latency.
- (c) Just tolerate the cold start.
- (d) Cold starts don't happen.

**Q5 (Short answer).** Describe rolling deploy vs blue-green.

**Q6 (Multiple choice).** For fast rollback (< 5 min), the best practice is:
- (a) `git revert` and redeploy.
- (b) Immutable image tags — rollback = redeploy the previous tag; also feature flags for behavior toggle.
- (c) SSH in and edit.
- (d) Delete the service and recreate.

**Q7 (Short answer).** For a multi-tenant SaaS, why do you want per-tenant isolation at the deployment / DB / rate-limit layer?

**Q8 (Multiple choice).** TLS termination is best done:
- (a) In your FastAPI app.
- (b) At the load balancer / CDN (Cloudflare, ALB, Cloud Run's built-in); app speaks HTTP internally.
- (c) Not needed.
- (d) At every hop.

**Q9 (Short answer).** Give three risks of scaling to zero (`min-instances = 0`) for an LLM API.

**Q10 (Multiple choice).** Environment configuration should be:
- (a) Baked into the Docker image.
- (b) In env vars at runtime, with secrets from a secrets manager (Fly secrets, AWS Secrets Manager, GCP Secret Manager).
- (c) Committed to git.
- (d) In a global variable.

---

### Answer Key & Detailed Explanations

**A1.** Managed platforms provide autoscaling, TLS, health-checked LBs, secrets, and monitoring out of the box, with minimal ops. K8s adds power but requires dedicated ops effort (cluster upgrades, addon management, RBAC, networking). For a small team, K8s' fixed cost of complexity exceeds its variable-cost savings. Adopt K8s only when service count / team size grow past what managed platforms handle well.

**A2. (b).** LLM services are I/O-bound; CPU stays low even under load. Autoscaling on CPU never triggers → latency spikes go unaddressed. Concurrency (in-flight requests) or request rate directly reflect load and scale correctly. Cloud Run's built-in concurrency signal is designed for exactly this pattern.

**A3.** (1) **Elasticity** — containers scale in/out freely; state persists. (2) **Zero-downtime deploys** — no state migration on redeploy. (3) **Resilience** — a container crash doesn't lose data. (4) **Multi-instance** — any instance serves any request; no sticky sessions or state coordination. (5) **Simpler backups** — one thing to back up (the DB), not N instances. Any three plus rationale.

**A4. (b).** Cold start of 30 s on scale-from-zero = the first user waits 30 s. That's often unacceptable. `min-instances ≥ 1` keeps a warm instance ready; you pay a small idle cost, but latency stays low. For truly latency-critical services, `min-instances = 3+` for redundancy.

**A5.** **Rolling deploy** replaces instances one at a time; old drains gracefully; total capacity is briefly reduced but never zero. Simple; default; good for stateless services. **Blue-green** deploys the new version alongside the old (both fully provisioned); swaps traffic atomically. Instant rollback (swap back); doubles infra cost during deploy; lowest risk for high-stakes changes.

**A6. (b).** Immutable image tags (`v1.2.3`, or `git-sha-abc123`): every deploy is a new tag; rollback = redeploy the previous tag. Fast (~1 min for image pull + start). Feature flags provide even faster rollback for behavior changes (no redeploy needed — flip the flag).

**A7.** Isolation prevents one tenant's issues from affecting others: (1) **DB isolation** — noisy queries on tenant A don't impact tenant B; (2) **Rate-limit isolation** — an abusive user in tenant A doesn't consume tenant B's quota; (3) **Blast radius** — a config bug affecting tenant A doesn't degrade the whole service; (4) **Auditing** — per-tenant logs simplify compliance.

**A8. (b).** Terminate TLS at the load balancer / CDN; internal traffic is HTTP. Simpler cert management (one place). CDN can offer caching, WAF, DDoS. Modern PaaS platforms (Fly, Cloud Run) do TLS transparently; you get it "for free."

**A9.** (1) **Cold start latency** — 30+ s to load model on first request; users see timeouts. (2) **DDoS amplification** — a burst of requests triggers many parallel cold starts, each of which loads the model, inflating cost. (3) **Session state** — if any user context lived in-memory (avoid this, but common), it's lost between scale-downs. Also: (4) health-check thrash at scale-0-to-1, (5) unpredictable behavior on cold-cached embeddings, etc.

**A10. (b).** Env vars at runtime (via platform config or `docker run -e`); secrets from a dedicated manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault, or Fly's built-in). Never bake secrets into the image (registry compromise = leak forever). Never commit secrets to git. Config that varies by environment (DB URL, feature flags) is env-var; secrets get manager treatment.

---

## 6. Practice Prompts

1. **Fly.io deploy.** Deploy a FastAPI service on Fly.io with health probes, TLS, and one custom env var. Test scaling with `fly scale count`.
2. **Cloud Run deploy.** Deploy the same service on Cloud Run with concurrency-based autoscaling. Load-test with a burst; observe scale-out.
3. **Rollback drill.** Deploy v2. Introduce a bug on purpose. Roll back to v1 in < 5 minutes. Document the runbook.
4. **Multi-region.** Deploy to two regions on your chosen platform. Configure DNS geo-routing. Verify latency drop from far-away users.
5. **Cost model.** For a service with 100 req/min average and 500 req/min peak, model the cost on: (a) always-on 3 instances, (b) autoscale 1–10 instances. Compare monthly bill.

---

## 7. References

- Fly.io docs: [fly.io/docs](https://fly.io/docs).
- GCP Cloud Run docs: [cloud.google.com/run/docs](https://cloud.google.com/run/docs).
- AWS Fargate + ECS docs.
- The Twelve-Factor App: [12factor.net](https://12factor.net/).
- Kelsey Hightower's talks on managed vs raw K8s.
- Google SRE Book — reliability principles.
