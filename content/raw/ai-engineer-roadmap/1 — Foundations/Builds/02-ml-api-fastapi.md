# Build #2 — ML API with FastAPI

> **Track:** Foundations · **Build:** 02
> **Prerequisites:** All Foundations modules, and Build #1 completed.
> **Time budget:** ~25–35 hours over ~2 weeks.
> **Deliverable:** A public GitHub repo containing a trained model exposed as a versioned HTTP service, dockerized, tested, and CI-deployed.

---

## 1. Executive Summary

The single most common "why can't I hand this off?" bottleneck in ML is the transition from *"model.pkl on my laptop"* to *"a service my team can call".* This build closes that gap.

You will:

1. Train a **small, honest supervised model** on a public dataset (any classification or regression task).
2. Wrap it in a **FastAPI** service that exposes prediction endpoints with request/response validation.
3. Add **authentication**, **rate limiting**, **structured logging**, and **health checks**.
4. Write **tests** — unit tests for the service, contract tests for the API, and a smoke test that hits a running container.
5. **Dockerize** it and set up a **GitHub Actions** pipeline that builds, tests, and pushes the image to a registry (GHCR or Docker Hub).
6. Optionally deploy to a free-tier host (Fly.io, Render, Railway) and share a live URL.

The model itself is not the point. **The point is production-shaped Python around it.**

---

## 2. Learning Outcomes

By the end you will have practiced:

1. Turning a Jupyter workflow into a production Python service.
2. Using **Pydantic** for request/response validation and error handling.
3. Serving models with **lifespan management** (load once, serve many).
4. Applying HTTP idioms — status codes, headers, versioning, idempotency.
5. Structured logging with request IDs, per-request timing, and PII hygiene.
6. Writing **contract tests** and **integration tests** with `httpx.TestClient`.
7. Building a **multi-stage Docker image** that stays under 300 MB.
8. Writing a CI workflow with `pytest`, `ruff`, `mypy`, container-build, and push.
9. Deploying to a free/cheap host and observing your service in the wild.

---

## 3. Choose a Model & Dataset

Pick a small, well-behaved task. Do **not** pick a state-of-the-art model — you're proving out the *serving* discipline, not the ML.

Suggested options:

- **Iris / Wine / Breast-Cancer** classifiers — trivial datasets, ideal for scaffold-focus.
- **Tabular** (California Housing, Boston Housing alt, Ames Housing) regression.
- **Text sentiment** (IMDB or SST-2 with a small sklearn or fasttext model). Bigger request payloads.
- **Time-series** (electricity load, retail demand) with a simple ARIMA/XGBoost.
- **Reuse the analysis dataset from Build #1** — a natural upgrade path.

Constraints:

- Total training time under 5 minutes.
- Trained artifact < 50 MB (serialize with `joblib`, `pickle`, or ONNX).
- Model has a **deterministic** inference path — no randomness at predict-time.

---

## 4. Repository Structure

```
ml-api/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── data/
│   └── README.md         # how to fetch training data
├── notebooks/
│   └── 01_train.ipynb    # exploratory training
├── models/
│   ├── model_v1.joblib
│   └── metadata.json     # {version, trained_at, metrics, features}
├── src/
│   └── ml_api/
│       ├── __init__.py
│       ├── main.py       # FastAPI app + routes
│       ├── config.py     # env + settings via pydantic-settings
│       ├── schemas.py    # Pydantic models: request / response / error
│       ├── model.py      # load + predict wrapper
│       ├── auth.py       # api-key check
│       ├── logging.py    # structlog / stdlib logging config
│       ├── middleware.py # request-id, timing, error-envelope
│       └── training/
│           ├── train.py
│           └── evaluate.py
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_predict.py
│   ├── test_auth.py
│   └── test_contract.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 5. API Design

### 5.1 Endpoints

- `GET  /healthz` — liveness (returns 200 always if the process is up).
- `GET  /readyz`  — readiness (200 only if the model loaded).
- `GET  /v1/model` — model metadata (version, features, metrics).
- `POST /v1/predict` — single prediction.
- `POST /v1/predict/batch` — batch predictions (max N items, 413 if larger).
- `GET  /metrics` — Prometheus-compatible metrics (optional stretch).

### 5.2 Request/Response Shape

**Request** (`POST /v1/predict`):
```json
{
  "features": {
    "sepal_length": 5.1,
    "sepal_width":  3.5,
    "petal_length": 1.4,
    "petal_width":  0.2
  },
  "request_id": "optional-client-supplied-uuid"
}
```

**Response** (200):
```json
{
  "request_id": "6a1e...",
  "model_version": "1.0.3",
  "prediction": "setosa",
  "probabilities": {"setosa": 0.98, "versicolor": 0.02, "virginica": 0.00},
  "latency_ms": 3
}
```

**Error envelope** (400/422/500):
```json
{
  "error": {
    "code": "invalid_features",
    "message": "sepal_length must be non-negative",
    "request_id": "6a1e..."
  }
}
```

### 5.3 Versioning

- Path-versioned: `/v1/*`. When you break the schema, ship `/v2/*` alongside.
- Include `model_version` in every response — so a client that pins to a model behavior can detect a drift.

### 5.4 Auth

- API key via `Authorization: Bearer <token>`.
- Keys read from an env var or a small secrets store.
- 401 if missing, 403 if valid key with wrong scope.

### 5.5 Rate Limiting

- Simple token-bucket per API key (in-memory for MVP; Redis for stretch).
- 429 with `Retry-After` header when exceeded.

---

## 6. Skeleton Code (starter, not final)

### 6.1 `src/ml_api/schemas.py`

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class IrisFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sepal_length: float = Field(ge=0, le=20)
    sepal_width:  float = Field(ge=0, le=20)
    petal_length: float = Field(ge=0, le=20)
    petal_width:  float = Field(ge=0, le=20)

class PredictRequest(BaseModel):
    features: IrisFeatures
    request_id: str | None = None

class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    prediction: Literal["setosa", "versicolor", "virginica"]
    probabilities: dict[str, float]
    latency_ms: int

class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str

class ErrorResponse(BaseModel):
    error: ErrorBody
```

### 6.2 `src/ml_api/model.py`

```python
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np

@dataclass
class ModelWrapper:
    estimator: object
    labels: list[str]
    version: str
    features: list[str]

    def predict(self, x: dict) -> tuple[str, dict[str, float]]:
        row = np.array([[x[f] for f in self.features]])
        probs = self.estimator.predict_proba(row)[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], {l: float(p) for l, p in zip(self.labels, probs)}


def load_model(path: Path) -> ModelWrapper:
    bundle = joblib.load(path)
    return ModelWrapper(
        estimator=bundle["estimator"],
        labels=bundle["labels"],
        version=bundle["version"],
        features=bundle["features"],
    )
```

### 6.3 `src/ml_api/main.py`

```python
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import settings
from .schemas import PredictRequest, PredictResponse, ErrorResponse, ErrorBody
from .model import load_model
from .auth import require_api_key
from .logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(Path(settings.MODEL_PATH))
    logger.info("model_loaded", version=app.state.model.version)
    yield
    logger.info("shutdown")


app = FastAPI(title="Iris API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    dur = (time.perf_counter() - t0) * 1000
    response.headers["x-request-id"] = rid
    response.headers["x-latency-ms"] = f"{dur:.1f}"
    logger.info("request", path=request.url.path, method=request.method,
                status=response.status_code, latency_ms=dur, request_id=rid)
    return response


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz(request: Request):
    return {"status": "ready", "model_version": request.app.state.model.version}


@app.post("/v1/predict", response_model=PredictResponse,
          responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def predict(body: PredictRequest, request: Request, _auth=require_api_key):
    t0 = time.perf_counter()
    model = request.app.state.model
    label, probs = model.predict(body.features.model_dump())
    return PredictResponse(
        request_id=body.request_id or request.state.request_id,
        model_version=model.version,
        prediction=label,
        probabilities=probs,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
```

---

## 7. Testing Strategy

### 7.1 Unit Tests
- `model.predict` deterministic and returns the right shape.
- Schema validation rejects malformed input.

### 7.2 Contract Tests
- Every endpoint's response matches the OpenAPI spec.
- Error envelope has the documented shape for 400/401/422/500.

### 7.3 Integration Tests
- `httpx.AsyncClient` against the app in-process (`app` fixture).
- API-key middleware behaves for missing / wrong / valid keys.

### 7.4 Smoke / End-to-End
- `docker compose up` + `curl` against `localhost:8000/healthz`.
- Included in CI via a lightweight script.

### 7.5 Load Test (stretch)
- `locust` or `k6` script that hits `POST /v1/predict` at increasing RPS.
- Measure p50/p95/p99 latency and error rate under load.

---

## 8. Observability Basics

- **Structured logs** (`structlog` or stdlib JSON formatter). Include `request_id`, `path`, `method`, `status`, `latency_ms`, `model_version`.
- **Never log** request bodies containing PII or authorization headers.
- **Optional metrics**: `/metrics` with Prometheus histograms for latency and counter for predictions per label.

---

## 9. Docker

**`Dockerfile` (multi-stage):**

```dockerfile
# ---------- builder ----------
FROM python:3.11-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 UV_SYSTEM_PYTHON=1
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ ./src/
COPY models/ ./models/

# ---------- runtime ----------
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/healthz || exit 1
USER 1000:1000
CMD ["uvicorn", "ml_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Rules:**

- No secrets baked in.
- Non-root user for runtime.
- Image size target: < 300 MB.
- Multi-arch build (`linux/amd64`, `linux/arm64`) is a nice stretch goal.

---

## 10. CI/CD (GitHub Actions)

**`.github/workflows/ci.yml`:**

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install uv && uv sync --frozen
      - run: uv run ruff check .
      - run: uv run mypy src
      - run: uv run pytest -q

  docker:
    needs: test
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
```

---

## 11. Milestones & Acceptance Criteria

### M1 — Train & Serialize (2h)
- [ ] `training/train.py` produces `models/model_v1.joblib` + `metadata.json`.
- [ ] Training is reproducible via a fixed random seed.

### M2 — Minimal API (3h)
- [ ] `POST /v1/predict` returns a valid response given valid input.
- [ ] `GET /healthz` and `/readyz` work.
- [ ] Pydantic schemas reject bad input with 422.

### M3 — Cross-cutting Concerns (4h)
- [ ] API-key auth middleware.
- [ ] Request-id + timing middleware.
- [ ] Structured logs.
- [ ] Rate limiting (in-memory).
- [ ] Error envelope on 4xx/5xx.

### M4 — Tests (4h)
- [ ] Unit tests on `model.predict` and schema validation.
- [ ] Integration tests via `TestClient` for every route.
- [ ] Contract tests verifying the OpenAPI response shapes.
- [ ] Coverage ≥ 80% on `src/ml_api/`.

### M5 — Docker & Compose (3h)
- [ ] `docker build` and `docker run` work locally.
- [ ] `docker compose up` runs the API on `:8000`.
- [ ] Image < 300 MB.
- [ ] Non-root user.

### M6 — CI/CD (2h)
- [ ] `ci.yml` runs lint, type-check, tests.
- [ ] Docker build + push to GHCR on `main`.
- [ ] Badges in README (ci, image size).

### M7 — Deploy & Document (3h)
- [ ] Deployed on Fly.io / Render / Railway free tier.
- [ ] Live URL in README with a `curl` example.
- [ ] `openapi.json` auto-generated and linked in README.

### M8 — Stretch (optional)
- [ ] Load test with `k6` or `locust`, publish results.
- [ ] Prometheus `/metrics` and a Grafana screenshot in README.
- [ ] Model-drift detection endpoint (`GET /v1/drift`) that compares feature distributions.
- [ ] A/B two model versions behind a feature flag.

---

## 12. Non-negotiables

1. **The model is loaded once**, at startup, via the FastAPI **lifespan** — never on every request.
2. **All endpoints are typed and validated** with Pydantic.
3. **No `print` for logs** — structured logs only.
4. **No secrets in code or git.** Env vars + `.env.example` only.
5. **Every response includes a `request_id`.**
6. **The service refuses to start** if the model is missing or has the wrong version.

---

## 13. Evaluation Rubric

| Area | 1 | 3 | 5 |
|------|---|---|---|
| Schema hygiene | dicts everywhere | Pydantic on inputs | Pydantic on I/O + strict `extra=forbid` |
| Errors | 500 for everything | 4xx / 5xx separated | typed error envelope + status codes |
| Auth | none | api-key in header | scoped keys + rate limiting |
| Logging | print statements | stdlib logging | structured JSON with request_id |
| Tests | none | unit + integration | + contract + smoke in CI |
| Docker | works locally | < 500 MB image | multi-stage, non-root, < 300 MB |
| CI | manual | lint + test | + build + push + release notes |
| Deploy | not deployed | working demo URL | + health + monitoring + rollback |

Aim for 4–5 across the board before you call it done.

---

## 14. Related Files

- `../goals.md`
- `../learning/01-python.md`
- `../learning/05-apis-json.md`
- `../learning/06-linux-cli.md` (Docker section)
- `../builds/01-data-analysis-project.md`
