# Docker for AI Services — Master Study Guide

> **Track:** Production AI Engineer · **Module:** 02
> **Prerequisites:** Foundations Module 06 (Linux/CLI/Docker basics).
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Docker is how AI services travel. Same image runs on your laptop, in CI, and in production. No more "it worked on my machine." For AI workloads specifically, containers matter *more* than for typical microservices because:

- Python dependency hell is real. Docker pins it.
- Models are big; you need layered caching to keep images sane.
- GPU access requires specific runtimes (nvidia-container-toolkit).
- Cold starts matter — every second of startup time is money.

Getting Docker "right" for AI means: small images, secure images, fast cold starts, deterministic builds, and CI/CD that plays nicely with all of it.

**Fundamental principles you must own:**

1. **Multi-stage builds** cut image sizes 5–10×.
2. **Layer order matters** — put slowest-changing steps first for cache efficiency.
3. **Non-root runtime** — always. `USER 1000:1000`.
4. **`.dockerignore` matters as much as `.gitignore`**.
5. **Health checks in the Dockerfile** — orchestrators use them.
6. **Model weights are the elephant** — big; separate layer; consider mounts or object storage.
7. **GPU containers** need the CUDA runtime, NVIDIA container toolkit, and matching driver versions.

If you retain nothing else: **small, secure, fast-to-start, deterministic images. Multi-stage always.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Anatomy of a Good Dockerfile

**A production Dockerfile for a Python AI service:**

```dockerfile
# ---------- 1. Base ----------
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# ---------- 2. Builder ----------
FROM base AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ---------- 3. Runtime ----------
FROM base AS runtime
WORKDIR /app
# Copy only what runtime needs
COPY --from=builder /app/.venv /app/.venv
COPY src /app/src
COPY models /app/models

ENV PATH="/app/.venv/bin:$PATH"

# Non-root user
RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -M -s /sbin/nologin app
USER 1000:1000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why each choice:**

- **`python:3.11-slim`** — Debian-slim base; smaller than the default Python image; still has `apt`. `-alpine` is smaller but has C-extension quirks with Python packages; **don't use Alpine for Python AI images** (glibc-vs-musl issues).
- **`ENV` group at top** — Python config: unbuffered output (real-time logs), no `.pyc` (smaller image), no pip cache.
- **Multi-stage** — `builder` has dev tools (`uv`, compilers); runtime doesn't. Final image is much smaller.
- **`uv sync --frozen`** — deterministic dependency install from a lockfile. `--no-dev` skips test/dev deps.
- **Layer ordering** — deps first, source last. Source changes on every commit; deps rarely. Cache hits on `deps` layer save minutes per build.
- **Non-root user** — least privilege. Attackers escaping the container land as user 1000, not root.
- **`HEALTHCHECK`** — Docker's built-in health probe; orchestrators use it.
- **`CMD` in exec form** — proper signal handling for graceful shutdown.

**`.dockerignore`:**

```
.git/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
.env
.env.*
tests/
docs/
notebooks/
*.md
!README.md
data/raw/
data/interim/
```

**Why:** these files inflate the build context (slower builds), possibly leak secrets (`.env`), or ship files you don't need (tests, docs). A 2 GB context sent to the daemon vs a 20 MB context is minutes of waiting per build.

---

### 2.2 Layer Caching Strategy

Docker caches layers by their input hash. Same input → cached; different input → rebuild that layer + everything after.

**Rule: order layers slowest-changing to fastest-changing.**

**Bad ordering:**

```dockerfile
COPY . /app         # Any file change invalidates this
RUN pip install -r requirements.txt   # Deps re-installed on every build 😭
```

**Good ordering:**

```dockerfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev   # Cached unless deps changed
COPY src /app/src               # Only source changes invalidate this
```

**Split model weights into a separate layer:**

```dockerfile
COPY models/checkpoint.pt /app/models/checkpoint.pt   # 2 GB — cache aggressively
COPY src /app/src                                     # Small; changes often
```

If model doesn't change between deploys, the layer is cached. Otherwise, downloading a 2 GB layer on every deploy wastes bandwidth and time.

**BuildKit and `--mount=type=cache`.** For heavy build steps (compiling native extensions), mount a cache dir:

```dockerfile
# syntax=docker/dockerfile:1.4
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

Cached across builds; not baked into the image. Fast rebuilds without image size cost.

**Cache-mount for `uv`/`poetry`** — same pattern; look up the tool's cache directory.

---

### 2.3 Image Size Discipline

Big images = slow deploys, expensive registry storage, slow cold starts.

**Size hierarchy** (indicative):
- `python:3.11` — ~1 GB (full Debian).
- `python:3.11-slim` — ~150 MB.
- `python:3.11-alpine` — ~50 MB (but glibc issues — avoid for AI).
- **Distroless** (`gcr.io/distroless/python3`) — ~50 MB, no shell, minimal attack surface. Great for prod.

**Target for AI services:** < 500 MB for a service with a small model; < 2 GB with a bigger model.

**Techniques to shrink:**

1. **Multi-stage.** Move build tools out of the runtime image.
2. **`--no-cache-dir`** on pip installs.
3. **`.dockerignore`** to skip files that don't need to be copied.
4. **Delete unnecessary files** after install:
   ```dockerfile
   RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
       && rm -rf /var/lib/apt/lists/*
   ```
5. **Separate compiled deps in the builder stage; copy artifacts only.**
6. **Model weights as a separate mount** rather than baked in — see 2.5.
7. **Squash layers** with `docker build --squash` (BuildKit).
8. **Analyze with `dive`** — visualize layers, spot bloat.

**Bad practices that inflate size:**
- Running `apt-get update` without `rm -rf /var/lib/apt/lists/*`.
- Installing `build-essential` in the runtime image.
- Copying entire repo (including `.git`, `__pycache__`, `tests`).
- Including Jupyter notebooks and their `.ipynb_checkpoints`.
- Baking large logs/data files.

**Trade-off — image size vs cold start.** A 5 GB image with the model baked in has slow deploys but fast cold starts (model already there). A 200 MB image that downloads the model on startup has fast deploys but slow cold starts (potentially 30+ s to download and load). Choose based on: deploy frequency, autoscale rate, model size, network cost.

---

### 2.4 Security Hardening

Containers are process-level isolation, not security boundaries. Harden them:

**Non-root user.**

```dockerfile
RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -M -s /sbin/nologin app
USER 1000:1000
```

Runtime processes have no more permissions than user 1000. Escape → not root on the host.

**Read-only filesystem.** Combined with runtime option `--read-only`. Mount `/tmp` as tmpfs if you need writable space.

**Drop capabilities.** In Docker `run`:
```bash
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE ...
```

Kernel capabilities most containers don't need — drop them all, add back what you need.

**No `--privileged`.** Ever.

**Minimal base image.** Distroless when possible; `-slim` otherwise. Each package is an attack surface.

**Signed images.** Sign your images (Cosign / Notary); require signature verification at deploy.

**Scan for vulnerabilities.**
- **Trivy** — CLI, fast, comprehensive.
- **Grype** — similar.
- **Docker Scout** — integrated with Docker Hub.
- **Snyk** — commercial.

Run in CI on every image build:

```yaml
- name: Trivy scan
  uses: aquasecurity/trivy-action@0.24.0
  with:
    image-ref: ghcr.io/${{ github.repository }}:${{ github.sha }}
    severity: HIGH,CRITICAL
    exit-code: 1   # fail CI on findings
```

**No secrets baked in.** Secrets go via environment variables at runtime, not `ENV` or `ARG` in the image. Once baked in, they're forever in image history.

**Pin base image digests** for reproducibility:
```dockerfile
FROM python:3.11-slim@sha256:abc123...
```

If the tag `slim` gets updated, your build changes silently. Pin digests for security-critical services.

---

### 2.5 GPUs, Model Weights, and AI-Specific Concerns

**GPU access.** Requires:
1. NVIDIA driver on the host.
2. NVIDIA Container Toolkit installed.
3. `--gpus all` (or specific) at `docker run`.
4. Base image with CUDA (e.g., `nvidia/cuda:12.1.0-runtime-ubuntu22.04`).

**Base images:**
- `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` — CUDA + cuDNN, ~2 GB.
- `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` — PyTorch pre-installed.
- `nvcr.io/nvidia/pytorch` — NVIDIA's official PyTorch image.

**Version matching.** Host CUDA driver ≥ image CUDA runtime. Mismatched = fails to start.

**Model weights — three strategies:**

**1. Bake into the image.**
- Pros: fast cold start (no download).
- Cons: large image; every deploy pushes the model; hard to version model independently.
- Best when: models are stable; deploys infrequent.

**2. Download on startup.**
```dockerfile
CMD ["sh", "-c", "python -m src.download_models && exec uvicorn ..."]
```
- Pros: image stays small; model versioning independent of code.
- Cons: slow cold start (30+ s to download and load); requires network at start; download failures break startup.
- Best when: models change more often than code.

**3. Mount from persistent volume / object storage.**
- Save the model in an EBS/EFS volume or object store; mount into container.
- Pros: fast cold start; small image; model independent.
- Cons: infra complexity; local dev needs a mount too.
- Best when: production with orchestration; frequent model updates.

**Combining these:** many teams bake a *default* model into the image (fast fallback) and let a config point to a specific version to load from storage.

**Multi-arch builds.** For ARM (Apple Silicon dev) + x86 (production):

```bash
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t ghcr.io/org/service:v1 \
    --push .
```

Runs on any host without hassle. Common for widely distributed images.

**docker-compose for local dev:**

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgres://user:pass@db:5432/app
    depends_on:
      db: {condition: service_healthy}
      redis: {condition: service_started}
  db:
    image: pgvector/pgvector:pg16
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
    volumes: [pgdata:/var/lib/postgresql/data]
  redis:
    image: redis:7-alpine
volumes:
  pgdata:
```

`docker compose up` runs the whole stack — API + DB + Redis. One command for local dev; matches prod topology.

---

## 3. Mental Models & Analogies

### 3.1 The "Sandwich Layers" Model

A Docker image is like a **sandwich built in layers**:

- **Bread** = base image.
- **Butter** = system packages, once, mostly static.
- **Meat** = deps (Python packages) — big but change together.
- **Lettuce** = your code — thin, changes often.

Order matters. If you spread the butter on top of everything, then add the meat, you rebuild the whole sandwich when you swap butter. Put the butter down first; layer meat; then lettuce. Now you swap lettuce (source) fast, meat (deps) sometimes, butter (system) rarely.

Docker's cache uses layer inputs. Order stable-first, volatile-last, and rebuilds are fast.

### 3.2 The "Onion Skin" Model (Multi-Stage Builds)

A multi-stage Dockerfile is an onion:

- **Builder layer** — has all the compilers, dev deps, test data, source, temporary files.
- **Runtime layer** — has just the compiled artifacts, minimal deps, the app, nothing else.

You peel the onion: keep only the runtime skin. All the messy building bits (build-essential, node_modules for tests, dev packages) get discarded. Result: a much smaller, more secure image.

The alternative (single-stage) is a snowball: everything you did during the build ends up in the final image. Fine for a demo; bad for production.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Alpine Is Always Smaller — Use It"

Alpine uses musl libc; Python packages often assume glibc. Result: many packages either don't work on Alpine or require lengthy compilations (looking at you, cryptography, numpy). Debugging Alpine issues has cost more human hours than the image savings.

**Rule for Python AI services:** stick with `-slim` (Debian) or distroless. Only switch to Alpine if you're doing a lot of statically compiled Go/Rust and know what you're doing.

### 4.2 "Running as Root Is Fine — It's a Container"

Container root is not host root, but it's still bad practice:
- Escape (rare but happens) = root on the host.
- Some cluster policies reject root containers.
- Read-only filesystems + non-root = defense in depth.

Add a user; use it. Cost is 3 lines; benefit is meaningful.

### 4.3 "Baking Secrets into the Image Is Convenient"

Never. Image history preserves every `ENV KEY=value` and `ARG` you set. `docker history <image>` shows them. Push once → your API key is on the registry forever, even if you overwrite it in a later layer. Use runtime env vars (`docker run -e API_KEY=...`) or secret managers (AWS Secrets Manager, Vault).

---

## 5. Self-Assessment Bank (Docker)

### Questions

**Q1 (Short answer).** Explain the layer-caching benefit of ordering `COPY requirements.txt / pip install ...` before `COPY src /app/src`.

**Q2 (Multiple choice).** For a Python AI service, the recommended base image class is:
- (a) `python:3.11` (full).
- (b) `python:3.11-slim` (Debian slim).
- (c) `python:3.11-alpine`.
- (d) `ubuntu:22.04` (raw).

**Q3 (Short answer).** Give three benefits of a multi-stage build.

**Q4 (Multiple choice).** Non-root runtime users:
- (a) Are optional.
- (b) Are non-negotiable — least privilege at the container level.
- (c) Are for Windows only.
- (d) Slow down the container.

**Q5 (Short answer).** Contrast three strategies for shipping model weights with a Docker image.

**Q6 (Multiple choice).** For GPU-enabled containers, you need:
- (a) Just `--gpus all`.
- (b) NVIDIA driver on host + NVIDIA Container Toolkit + CUDA runtime in image + `--gpus`.
- (c) A special CPU flag.
- (d) Only PyTorch.

**Q7 (Short answer).** Why should you never `ENV API_KEY=...` in a Dockerfile?

**Q8 (Multiple choice).** For image vulnerability scanning in CI:
- (a) Not needed.
- (b) Run Trivy/Grype/Docker Scout on every build; fail CI on HIGH/CRITICAL findings.
- (c) Manual review is enough.
- (d) Only at deploy time.

**Q9 (Short answer).** How can you dramatically reduce your Docker build context size?

**Q10 (Multiple choice).** For a 2 GB image with the model baked in vs a 200 MB image that downloads the model on startup:
- (a) Baked = faster cold start; small = slower cold start.
- (b) Small always wins.
- (c) They're identical.
- (d) Baked is deprecated.

---

### Answer Key & Detailed Explanations

**A1.** Docker caches each layer by the hash of its inputs. Deps rarely change; source changes on nearly every commit. Ordering deps-first means the `pip install` layer is cached across most builds — you only rebuild the (fast) source-copy step. Reversed ordering means every commit invalidates the deps layer and reinstalls all packages, adding minutes to every build.

**A2. (b).** `python:3.11-slim` — Debian-slim base, small enough (~150 MB), fully compatible with Python packages (glibc). Full `python:3.11` (a) is unnecessarily large. Alpine (c) uses musl and often breaks Python C extensions. `ubuntu:22.04` (d) works but you have to install Python.

**A3.** (1) **Smaller runtime images** — build tools stay in the builder stage. (2) **Better security** — no compilers or dev tools in the final image (smaller attack surface). (3) **Faster deploys** — smaller images pull faster. (4) **Cleaner separation** — build concerns vs runtime concerns. Any three plus rationale.

**A4. (b).** Non-root runtime users are standard practice. Cost: 2-3 lines in a Dockerfile. Benefit: escape from container ≠ root on host; cluster policies happy; defense in depth. Never `USER root` in prod images.

**A5.** (1) **Bake into image** — fast cold start, big image, tight coupling of model + code versions. (2) **Download on startup** — small image, slow cold start (30+ s), needs network + working credential fetch. (3) **Mount from persistent volume / object storage** — small image, fast cold start, requires infra but decouples model from code lifecycle. Choose based on model size, deploy frequency, infra maturity.

**A6. (b).** All four are required. Driver on host (obviously). NVIDIA Container Toolkit gives the Docker daemon a way to expose the GPU to containers. CUDA runtime in the image (base like `nvidia/cuda:...`). Then `docker run --gpus all` at launch tells Docker to expose the GPU. Miss any one = "CUDA not available" errors.

**A7.** Image history preserves every `ENV`, `ARG`, `RUN`. Anyone with `docker pull` and `docker history` can see the value even if a later layer "overwrites" it — the layer is still there. Push a secret to a registry once and it's compromised forever. Use runtime env vars (`docker run -e KEY=value`) or secret managers (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault) that inject at container start.

**A8. (b).** Trivy / Grype / Docker Scout scan images for known CVEs; run in CI on every image build; fail the CI job on HIGH/CRITICAL findings. Cheap defense; catches known issues before they reach prod. Should be mandatory for any customer-facing service.

**A9.** Use `.dockerignore` aggressively — exclude `.git/`, `__pycache__/`, `.venv/`, `tests/`, `docs/`, `notebooks/`, `data/raw/`. A 2 GB build context sent to the daemon can drop to 20 MB after `.dockerignore`. Speeds every build; reduces the chance of accidentally baking dev files into the image.

**A10. (a).** Baked-in models = big image (slow deploy, expensive to push) but fast cold start (no download needed). Small image with startup download = fast deploy but 30+ s cold-start latency downloading the model. Choose based on: deploy frequency (bake if rare, download if frequent), autoscale characteristics (bake if scale-out matters), and cost of network vs registry.

---

## 6. Practice Prompts

1. **Multi-stage refactor.** Take an existing Dockerfile (yours or an open-source project); rewrite as multi-stage. Compare final image sizes.
2. **Layer ordering.** Deliberately mis-order layers; measure build time for a source-only change. Reorder correctly; measure again.
3. **Distroless.** Convert a `python:3.11-slim` runtime to `gcr.io/distroless/python3`. Handle any missing tools; verify the app still runs.
4. **Trivy scan.** Add a Trivy step to your CI. Trigger it on a known-vulnerable base image (e.g., an old `python:3.9` tag). Confirm CI fails.
5. **Compose stack.** Write a `docker-compose.yml` for an AI service + pgvector + redis. Verify one-command startup and inter-service communication.

---

## 7. References

- Docker docs: [docs.docker.com](https://docs.docker.com/).
- Docker best practices: [docs.docker.com/develop/dev-best-practices/](https://docs.docker.com/develop/dev-best-practices/).
- Distroless: [github.com/GoogleContainerTools/distroless](https://github.com/GoogleContainerTools/distroless).
- Trivy: [aquasecurity.github.io/trivy](https://aquasecurity.github.io/trivy).
- NVIDIA Container Toolkit docs.
- `dive` (image explorer): [github.com/wagoodman/dive](https://github.com/wagoodman/dive).
