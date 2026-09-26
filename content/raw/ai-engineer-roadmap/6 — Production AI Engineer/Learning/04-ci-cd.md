# CI/CD — Master Study Guide

> **Track:** Production AI Engineer · **Module:** 04
> **Prerequisites:** Modules 01–03 + Foundations Module 04 (Git/GitHub).
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Without CI/CD, "shipping" is: build locally, push, hope. With CI/CD, every commit is validated, tested, evaluated, containerized, and deployed automatically. The difference between "1 deploy a week (fragile)" and "10 deploys a day (safe)."

For AI services, CI/CD is doubly important:
- **Eval regression tests** — a prompt tweak must not silently regress model quality.
- **Model version pinning** — the CI pipeline knows exactly which model version each build uses.
- **Cost / performance regression** — CI can flag when latency or cost exceeds budget.
- **Secret handling** — CI is where secrets meet code; get it right or leak them.

**Fundamental principles you must own:**

1. **Every commit runs the pipeline.** No manual runs; no "I'll test later."
2. **Fast feedback.** CI should complete in < 10 min; ideally < 5.
3. **Test what matters.** Unit + integration + eval + security scans + smoke tests post-deploy.
4. **Deploy is a pipeline step**, not a manual action.
5. **Rollback is automated.** One command, or an auto-rollback on health-check failure.
6. **Every deploy is traceable** — commit SHA → CI run → deployed image → running service.
7. **Secrets never in code, never in logs.**

If you retain nothing else: **automated, fast, and trustworthy. Every merge to main should be a candidate for production.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The CI Pipeline (GitHub Actions Example)

A production-quality CI for a FastAPI service:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint-type:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install uv
        run: pip install uv
      - name: Sync deps
        run: uv sync --frozen
      - name: Lint
        run: uv run ruff check .
      - name: Format check
        run: uv run ruff format --check .
      - name: Type check
        run: uv run mypy src

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env: {POSTGRES_PASSWORD: test}
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
      redis:
        image: redis:7
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install uv && uv sync --frozen
      - name: Run tests
        env:
          DATABASE_URL: postgres://postgres:test@localhost:5432/postgres
          REDIS_URL: redis://localhost:6379
        run: uv run pytest -q --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v5

  eval-regression:
    runs-on: ubuntu-latest
    if: contains(github.event.head_commit.modified, 'prompts/') || contains(github.event.head_commit.modified, 'src/agent/')
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install uv && uv sync --frozen
      - name: Run eval suite
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_CI }}
        run: uv run python -m eval.run --dataset eval/dataset.jsonl
      - name: Regression check
        run: uv run python -m eval.assert_baseline --tolerance 0.02

  build:
    needs: [lint-type, test]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Trivy scan
        uses: aquasecurity/trivy-action@0.24.0
        with:
          image-ref: ghcr.io/${{ github.repository }}:${{ github.sha }}
          severity: HIGH,CRITICAL
          exit-code: 1

  deploy:
    needs: [build]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - name: Deploy
        run: flyctl deploy --image ghcr.io/${{ github.repository }}:${{ github.sha }}
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
      - name: Smoke test
        run: curl --fail https://my-service.fly.dev/healthz
```

**What this pipeline gives you:**

- **Fast fail on lint/type errors** — cheap parallel jobs.
- **Test with real DB/Redis** via GitHub Actions services.
- **Eval regression as a gate** — only runs when prompts/agent code changes (cost control).
- **Multi-stage Docker build with cache** — fast rebuilds.
- **Trivy scan** — fails on HIGH/CRITICAL CVEs.
- **Deploy only from main** — protects prod from feature branches.
- **Post-deploy smoke test** — confirms the deploy actually works.

---

### 2.2 Pipeline Structure and Dependencies

**Jobs vs Steps.**
- **Steps** run sequentially within a job; share the same runner (VM).
- **Jobs** can run in parallel on separate runners.
- **`needs:`** creates job dependencies (job B runs after job A succeeds).

**Design principles:**

**1. Parallel where possible.** Lint, type-check, security scans run independently → parallel jobs. Faster feedback.

**2. Fail fast.** Cheap checks (linting) before expensive ones (integration tests). If lint fails, no point testing.

**3. Only run what's needed.** Path filters (`paths:`) skip unchanged areas.

**4. Cache aggressively.**
- Docker layer cache (GHA cache backend).
- Package cache (`actions/setup-python@v5` with `cache: pip`).
- Custom caches (`actions/cache`) for anything expensive.

**5. Reusable workflows.** Extract common patterns:

```yaml
# .github/workflows/eval.yml (reusable)
on:
  workflow_call:
    inputs:
      dataset: {type: string, required: true}
    secrets:
      openai_key: {required: true}
jobs:
  ...
```

Call from other workflows:

```yaml
jobs:
  eval:
    uses: ./.github/workflows/eval.yml
    with:
      dataset: eval/prod.jsonl
    secrets:
      openai_key: ${{ secrets.OPENAI_API_KEY }}
```

**6. Concurrency guards.** Prevent multiple deploys stepping on each other:

```yaml
concurrency:
  group: deploy-prod
  cancel-in-progress: false   # queue, don't cancel
```

---

### 2.3 Secrets and Environment Management

**Never in code. Never in logs.**

**GitHub secrets:**
- **Repo-level** — global to the repo.
- **Environment-level** — scoped to `environment: production` (requires approval, protection rules).
- **Organization-level** — shared across repos.

Access in workflows:
```yaml
env:
  API_KEY: ${{ secrets.API_KEY }}
```

**Environment protection.**

```yaml
jobs:
  deploy:
    environment: production   # requires manual approval, restricted branches
```

Configure in GitHub UI: required reviewers, wait timers, allowed branches. Prod deploys gated behind human approval if you want.

**Redacting logs.** GitHub auto-redacts secret values from log output. But careful: if you `echo $API_KEY | base64`, the redaction may miss the base64-encoded form. Never print secrets, even indirectly.

**OIDC for cloud auth.** Avoid long-lived cloud credentials in GitHub secrets. Use OIDC:

```yaml
permissions:
  id-token: write

steps:
  - uses: aws-actions/configure-aws-credentials@v4
    with:
      role-to-assume: arn:aws:iam::123:role/gh-actions
      aws-region: us-east-1
```

Short-lived STS tokens; no static credentials to leak.

**Secrets manager integration.** For runtime secrets, pull from a secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault) rather than storing in GitHub. Rotation without redeploy.

**Dependency tokens.** For private registries, use GitHub's `${{ secrets.GITHUB_TOKEN }}` where possible; auto-generated per-run; scoped to the repo.

---

### 2.4 Deployment Strategies in CI/CD

**Deploy to a real environment on merge.**

**Simple pattern (auto-deploy on main):**
```yaml
deploy:
  if: github.ref == 'refs/heads/main'
  needs: [test, build]
  ...
```

Every merge to main → deploy. Fast; requires confidence in tests.

**Staged deploys.**

```yaml
deploy-staging:
  needs: [build]
  environment: staging
  ...

deploy-production:
  needs: [deploy-staging, integration-tests-staging]
  environment: production
  ...
```

Merge → staging automatically → manual approval → production. Common for stricter shops.

**Preview environments.** For PRs, spin up an ephemeral env:
```yaml
deploy-preview:
  if: github.event_name == 'pull_request'
  ...
```

Every PR gets its own URL. Product / QA can review. Tear down on merge or close.

**Canary from CI:**
1. Deploy new image to 5% of prod traffic.
2. Watch metrics for 10 minutes.
3. If green, ramp to 100%.
4. If bad, auto-rollback.

Implemented via a deploy tool (Argo Rollouts, Flagger) or platform features (Cloud Run traffic split, AWS CodeDeploy).

**Auto-rollback.** Deploy script watches health checks post-deploy. If liveness fails or error rate spikes, revert to previous image. Cheap insurance:

```yaml
- name: Deploy with rollback
  run: |
    flyctl deploy ...
    sleep 60
    if ! curl --fail https://my-service.fly.dev/healthz; then
      flyctl deploy --image ${{ env.PREVIOUS_IMAGE }}
      exit 1
    fi
```

---

### 2.5 Eval Regression as a First-Class Gate

For AI services, standard CI gates aren't enough. A prompt tweak passes linting and unit tests — but silently degrades model quality. **Eval regression tests catch this.**

**Pipeline pattern:**

```yaml
eval:
  runs-on: ubuntu-latest
  # Only when prompts or model code changes
  if: github.event.pull_request.paths contains 'prompts/' || contains 'src/agent/'
  steps:
    - uses: actions/checkout@v4
    - name: Run eval suite
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_EVAL }}
      run: python -m eval.run --dataset eval/tasks.jsonl --output results.json
    - name: Assert vs baseline
      run: |
        python -m eval.assert --results results.json --baseline eval/baseline.json --tolerance 0.02
    - name: Post results as PR comment
      uses: some-action/comment-on-pr@v1
      with:
        body-from-file: results.md
```

**What "regression" means for AI:**
- Task-level pass rate — drop > 2% fails the merge.
- Faithfulness score — drop > 2% fails.
- Average cost per task — increase > 20% flagged (may still pass with warning).
- Latency p95 — increase > 20% flagged.

**Cost of eval in CI.** Every run costs LLM API calls. Ways to control:
- **Path filters** — only run eval when relevant files change.
- **Sample vs full eval** — PRs run a small sample (10 tasks); main runs the full suite (100+).
- **Cached golden outputs** — for regression cases where inputs are stable, cache expected outputs.
- **Cheaper eval model** — use a smaller LLM for judgment where possible.

**Baseline management.** Store `baseline.json` in the repo. Update it deliberately (a specific PR labeled `baseline-update`) when you accept the current state as the new standard. Never let CI silently ratchet the baseline.

**Nightly extended eval.** For expensive full-suite runs, schedule nightly on a cron:

```yaml
on:
  schedule:
    - cron: '0 6 * * *'   # every day at 6 AM UTC
```

Catches slow drift; not gate on it — surface in a dashboard.

---

## 3. Mental Models & Analogies

### 3.1 The "Automated Assembly Line" Model

Traditional deploy is **hand-assembling each car**: put on the wheels, add the engine, install the seats. Slow, error-prone, no repeatability.

CI/CD is an **assembly line**: raw materials (commits) enter one end; a finished, tested, deployed product exits the other. Each station does one thing (lint, test, build, scan, deploy). Failures at any station halt the line and light up a warning.

Because it's automated, the line can run many times per day. Because each station is instrumented, when a car comes out defective, you know exactly which station failed. Because each build is identical, rollback = redeploy an earlier build.

Manual deploys are workshop cars. CI/CD is Toyota. Different scale, different reliability.

### 3.2 The "Immune System" Model

CI/CD is your codebase's **immune system**:

- **Lint / type-check** = first line of defense — cheap checks catch typos and type mistakes.
- **Unit tests** = specific antibodies for known behaviors.
- **Integration tests** = broader immune response — how components interact.
- **Eval regression** = surveillance for AI-specific issues.
- **Security scans** = detection of known threats (CVEs).
- **Post-deploy smoke tests** = confirming the vaccine actually worked.

A weak immune system → infections (bugs in prod). Overactive immune system → autoimmune (false-positive flakes that stall shipping). The tuning: catch real issues fast; don't tolerate false alarms; keep the whole thing quick enough to not stall the body (dev velocity).

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "CI Is Testing" (Only)

CI is: testing + linting + type-checking + eval + security scanning + building + pushing + deploying + smoke-testing. Most teams stop at testing. The rest are cheap and prevent whole classes of bugs.

### 4.2 "Slow CI Is Fine — We'll Wait"

Slow CI kills dev velocity. Every commit takes 30 min to feedback → devs multitask → context switches → productivity drops. Aim for < 10 min end-to-end. Parallelize; cache; skip unnecessary steps. Fast CI = more deploys = smaller changes = fewer bugs.

### 4.3 "Manual Deploys Are Safer"

Feels safer; empirically the opposite. Manual = irregular = accumulating drift = big-bang deploys = big-bang bugs. Automated CD = frequent small deploys = each change contains ~1 possible bug = easy diagnosis. Google's DORA metrics consistently show high-performing teams deploy more frequently, not less.

---

## 5. Self-Assessment Bank (CI/CD)

### Questions

**Q1 (Short answer).** Name six things a production CI pipeline should do (beyond testing).

**Q2 (Multiple choice).** For a fast CI, the right structure is:
- (a) One sequential job doing everything.
- (b) Multiple parallel jobs (lint, test, build) with `needs:` for dependencies; cache aggressively.
- (c) Skip CI to save time.
- (d) Manual runs only.

**Q3 (Short answer).** Why store secrets in GitHub Environment-level secrets vs Repo-level?

**Q4 (Multiple choice).** OIDC for cloud authentication in GitHub Actions:
- (a) Requires long-lived cloud credentials.
- (b) Issues short-lived STS tokens per run; no static credentials stored in GitHub.
- (c) Only works for AWS.
- (d) Is deprecated.

**Q5 (Short answer).** Explain how an eval regression test in CI protects AI services.

**Q6 (Multiple choice).** Preview environments per PR:
- (a) Are wasteful.
- (b) Let PR reviewers / QA test the change against a real URL before merging; auto-torn-down on merge.
- (c) Only work with Kubernetes.
- (d) Break CI.

**Q7 (Short answer).** How would you configure an auto-rollback in CD?

**Q8 (Multiple choice).** For deploy pipelines, `concurrency: group: deploy-prod, cancel-in-progress: false`:
- (a) Aborts old deploys.
- (b) Queues deploys so two never run simultaneously — prevents deploy races.
- (c) Increases parallelism.
- (d) Only for staging.

**Q9 (Short answer).** How can you keep CI eval costs manageable when eval requires paid LLM calls?

**Q10 (Multiple choice).** For a change that ONLY touches `docs/`, ideally CI:
- (a) Runs the full pipeline.
- (b) Uses path filters to skip test / build / deploy — only doc build runs.
- (c) Blocks the merge until manual approval.
- (d) Runs but takes forever.

---

### Answer Key & Detailed Explanations

**A1.** Beyond testing, CI should: (1) **lint** (ruff/black), (2) **type-check** (mypy), (3) **security scan** dependencies (`pip-audit`, Dependabot), (4) **security scan** the container image (Trivy/Grype), (5) **eval regression** for AI, (6) **build** the container, (7) **push** to registry, (8) **deploy** to environment, (9) **post-deploy smoke test**, (10) **notify** on failure. Any six from this list.

**A2. (b).** Parallel jobs (lint, type, test, build, scan) with explicit dependencies (`needs:`) run as fast as the slowest single job. Cache Docker layers (BuildKit / GHA cache), Python deps (`actions/setup-python` with cache), any expensive artifacts. Sequential CI is slow and blocks devs.

**A3.** Environment-level secrets are scoped to a specific environment (`production`), respect its protection rules (required reviewers, allowed branches, wait timers), and aren't accidentally available in non-prod workflows. Repo-level secrets are available to any workflow in the repo. Use environment-level for prod credentials; repo-level for shared non-prod secrets.

**A4. (b).** OIDC (OpenID Connect) issues short-lived STS tokens per workflow run, scoped to the specific job. No static credentials stored in GitHub secrets. Rotates automatically; can't be leaked long-term. Standard for AWS, GCP, Azure integrations now.

**A5.** A prompt tweak or model change may pass linting and unit tests but silently regress model quality. The eval regression test runs your labeled eval suite (Module 08 later) and compares scores against a stored baseline. If task-level pass rate drops > threshold (typically 2%), CI fails the merge. Catches silent quality regressions before they hit prod. Also catches unintended improvements — the baseline gets updated deliberately.

**A6. (b).** Preview envs let reviewers interact with the actual change (not just read code). Great for UI changes, API contract changes, or LLM behavior tweaks where "does this actually work?" beats reading a PR. Ephemeral resources tear down on merge/close.

**A7.** In the deploy step (or a follow-up step), after deploy: wait N seconds → hit health endpoint / check error rate / check p95 latency. If bad, redeploy the previous image tag automatically. Advanced: canary deploy, watching metrics dashboard for a longer window before promoting to 100%. Frameworks: Flagger, Argo Rollouts, or platform features (Cloud Run traffic split).

**A8. (b).** `concurrency` groups jobs by name; setting `cancel-in-progress: false` means new deploys queue instead of canceling running ones. Prevents two deploys stepping on each other (one pushing image A, another simultaneously pushing image B — result: race). For deploy pipelines specifically, `cancel-in-progress: false` (queue) is safer than `true` (kill in-flight).

**A9.** (1) **Path filters** — only run eval when prompts / agent code changes. (2) **Sample vs full** — PRs run 10-task sample; main runs full 100+. (3) **Nightly full eval** — schedule expensive runs off the critical path; alert but don't block PRs. (4) **Cheaper judge models** — use small models for LLM-as-judge where possible. (5) **Cache golden outputs** for deterministic parts of the eval.

**A10. (b).** Path filters (`paths: ['src/**', 'tests/**']` at the workflow or job level) skip work when unchanged. A docs-only change doesn't need the test suite. Saves runtime; keeps CI fast; enables high-frequency doc PRs without CI overload.

---

## 6. Practice Prompts

1. **Full pipeline.** Build a GitHub Actions workflow for your Month 4/5 build: lint + type + test + eval + build + scan + push + deploy. Aim for < 10 min end-to-end.
2. **Preview envs.** Add per-PR preview deploys on Fly.io. Each PR gets its own subdomain. Tear down on close.
3. **OIDC + AWS.** Configure OIDC in GitHub Actions to assume an AWS role. Deploy to ECR + ECS without a static AWS access key.
4. **Eval regression.** Add an eval-gate job that runs your RAG eval; asserts baseline; posts a PR comment with results table.
5. **Auto-rollback.** Add a post-deploy monitor: watch `/healthz` for 60s; if unhealthy, redeploy previous tag.

---

## 7. References

- GitHub Actions docs: [docs.github.com/en/actions](https://docs.github.com/en/actions).
- Google DORA metrics — deploy frequency and stability.
- Docker BuildKit cache: [docs.docker.com/build/cache/](https://docs.docker.com/build/cache/).
- Trivy: [aquasecurity.github.io/trivy](https://aquasecurity.github.io/trivy).
- Fly.io deploy action: [github.com/superfly/flyctl-actions](https://github.com/superfly/flyctl-actions).
- HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager — for runtime secrets.
