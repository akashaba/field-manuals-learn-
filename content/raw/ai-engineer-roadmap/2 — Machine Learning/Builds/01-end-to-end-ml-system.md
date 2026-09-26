# Build #1 — End-to-End ML Prediction System

> **Track:** Machine Learning · **Build:** 01
> **Prerequisites:** All Machine-Learning learning modules (01–13), plus Foundations Builds #1 and #2.
> **Time budget:** ~35–50 hours over ~2 weeks.
> **Deliverable:** A public GitHub repo with a full ML lifecycle — problem framing, data ingestion, feature engineering, model training, evaluation, tuning, serialization, API service, monitoring, model card, and CI/CD.

---

## 1. Executive Summary

This is the *capstone* build of Month 2. It integrates everything you've learned:

- Data ingestion + cleaning + feature engineering (Foundations Build #1 muscle).
- Model training with proper validation, tuning, and metric selection (this month).
- Serving via a versioned FastAPI service (Foundations Build #2 muscle).
- Monitoring, drift detection, and a model card.
- Reproducibility and CI/CD.

The **point** is not to build the world's most accurate model. The point is to build a **well-engineered ML system** that a teammate could pick up, understand in an hour, and modify safely. If you can hand this repo to a hiring manager and they can spin it up in 10 minutes and understand every decision, the goal is met.

---

## 2. Learning Outcomes

By the end you will have practiced:

1. **Problem framing** — turning a vague ask into a modeling problem with a metric.
2. **Baseline discipline** — never trusting a fancy model without a dumb baseline for comparison.
3. **Leakage-free pipelines** — `Pipeline` + `ColumnTransformer` + CV that would survive a code review.
4. **Hyperparameter tuning** — Optuna (or `RandomizedSearchCV`) on a real search space.
5. **Model card / documentation** — writing what the model does, its data, its metrics, its limits, its intended and unintended uses.
6. **Serving** — a FastAPI endpoint that returns predictions with calibrated probabilities, request IDs, latency headers.
7. **Monitoring hooks** — logging predictions, tracking distribution drift.
8. **CI/CD** — automated tests, container build, image push, optional deploy.

---

## 3. Choose a Problem

Pick a problem with:

- **Clear labels** (supervised).
- **Non-trivial imbalance or nuance** — not Iris; something where metric choice matters.
- **Domain interest** — because you'll be staring at it for two weeks.
- **Manageable data** — 10k–1M rows, so training is fast but not trivial.

**Suggestions:**

### 3.1 Option A — Legislative Bill Passage Prediction (aligned with Brian's day job)
> Given features of a legislative bill (sponsor, chamber, session, topic, length, co-sponsors), predict whether it will pass. Rich features, real imbalance, interpretable output.

### 3.2 Option B — Credit Card Fraud Detection
> Public Kaggle dataset. Very imbalanced (0.17% positives). Classic testbed for imbalanced-classification techniques.

### 3.3 Option C — Customer Churn
> Telco churn or a similar dataset. Moderate imbalance, mixed feature types, interpretable business use case.

### 3.4 Option D — Football / Soccer Match Outcome (aligned with soccer analytics interest)
> Predict win / draw / lose from pre-match features (team form, home/away, injuries). Multiclass; good excuse for calibration work.

### 3.5 Option E — Hospital Readmission
> Public MIMIC or Kaggle dataset. Multi-source features, meaningful fairness concerns for the model card.

Whichever you pick, **write down the "business decision"** the model will inform. That decision drives your metric choice.

---

## 4. Repository Structure

```
ml-system/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── configs/
│   ├── default.yaml
│   └── tuning.yaml
├── data/
│   ├── raw/               # gitignored
│   ├── interim/           # gitignored
│   └── processed/         # small samples committed
├── docs/
│   ├── model_card.md
│   ├── data_dictionary.md
│   └── decisions.md       # ADRs (Architecture Decision Records)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_error_analysis.ipynb
├── src/
│   └── ml_system/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── ingest.py
│       │   ├── clean.py
│       │   └── schema.py
│       ├── features/
│       │   ├── build.py
│       │   └── transformers.py
│       ├── models/
│       │   ├── baseline.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── tune.py
│       │   └── registry.py
│       ├── api/
│       │   ├── main.py
│       │   ├── schemas.py
│       │   ├── auth.py
│       │   └── logging.py
│       └── monitoring/
│           ├── drift.py
│           └── metrics.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_api.py
│   └── fixtures/
├── artifacts/
│   ├── model_v1.joblib    # small — can commit; larger stored in cloud
│   └── metadata.json
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 5. Milestones & Acceptance Criteria

### M1 — Problem Framing (2h)
- [ ] Written `docs/decisions.md` ADR-001: what problem, why this metric, what data, what class of models.
- [ ] Explicit statement of positive vs negative cost. What does a false positive cost? A false negative?
- [ ] Chosen primary evaluation metric with justification.

### M2 — Data Ingestion & Cleaning (4–6h)
- [ ] `src/ml_system/data/ingest.py` idempotently pulls data.
- [ ] `clean.py` exposes pure functions for cleaning steps.
- [ ] `schema.py` uses Pandera or Pydantic to validate schema post-cleaning.
- [ ] Data dictionary in `docs/data_dictionary.md` documents every column.
- [ ] Small sample data committed for tests.

### M3 — EDA & Baseline (3–5h)
- [ ] `notebooks/01_eda.ipynb` — target distribution, missingness, correlations, class balance.
- [ ] `notebooks/02_baseline.ipynb` — logistic regression / dummy classifier as baseline; CV score reported with variance.
- [ ] Any real model must **beat** the baseline by a meaningful margin.

### M4 — Feature Engineering (4–6h)
- [ ] `features/transformers.py` — custom transformers (subclassing `TransformerMixin`), with unit tests.
- [ ] `features/build.py` — assembles a `ColumnTransformer` for the dataset.
- [ ] Handles: imputation, encoding, scaling, interactions, time features (if applicable).
- [ ] Leakage-safe — all transformers `.fit` on training only.

### M5 — Model Training with CV (4–6h)
- [ ] `models/train.py` — trains multiple candidates (logistic, RF, XGBoost) wrapped in the pipeline.
- [ ] Uses `StratifiedKFold` (or `TimeSeriesSplit` / `GroupKFold` as appropriate).
- [ ] Reports mean and std of CV scores per model.
- [ ] Log-loss AND target metric reported.
- [ ] Learning curves for the chosen model.

### M6 — Hyperparameter Tuning (4–6h)
- [ ] `models/tune.py` — Optuna study over a defined search space.
- [ ] `configs/tuning.yaml` — search space, trial budget, CV strategy.
- [ ] Report: best hyperparameters, best CV score, variance.
- [ ] Nested CV or hold-out test-set evaluation for the *final* honest estimate.

### M7 — Evaluation & Calibration (3–4h)
- [ ] Full confusion-matrix breakdown at chosen threshold.
- [ ] ROC and PR curves plotted; PR-AUC and ROC-AUC reported.
- [ ] Reliability diagram; Brier score.
- [ ] If needed: `CalibratedClassifierCV` and re-evaluate.
- [ ] Threshold selection with business justification.
- [ ] **Error analysis**: which subsets does the model do worst on? Any fairness slices?

### M8 — Model Card (2–3h)

`docs/model_card.md` covering:

- **Model details:** name, version, date, framework.
- **Intended use** and **out-of-scope use**.
- **Training data** — source, size, feature descriptions.
- **Metrics** — CV score with variance; test-set score.
- **Ethical considerations / fairness slices** — how does the model perform by subgroup?
- **Known limitations** — what does it get wrong, when should it not be trusted?
- **Contact.**

### M9 — Serialization & Model Registry (2h)
- [ ] `models/registry.py` — save trained pipeline as `joblib` artifact.
- [ ] `metadata.json` beside it: version, trained_at, train_data_hash, metrics, feature list, sklearn version.
- [ ] `load_model(version)` returns a ready-to-predict object.

### M10 — API Serving (4–5h)
- [ ] FastAPI service with `/healthz`, `/readyz`, `/v1/model`, `/v1/predict`, `/v1/predict/batch`.
- [ ] Pydantic schemas for request/response.
- [ ] Model loaded once at startup (lifespan).
- [ ] API key auth.
- [ ] Structured logs with `request_id`, `model_version`, `latency_ms`.
- [ ] Predictions returned with **probability** and **class**, plus threshold used.

### M11 — Monitoring Hooks (2–3h)
- [ ] Every prediction logged to a JSONL file (or Kafka, or stdout for downstream capture).
- [ ] `monitoring/drift.py`: given a recent window of inputs, compute basic drift stats vs training distribution (PSI, KS test).
- [ ] `/metrics` endpoint exposing prediction counters and latency histogram (Prometheus format optional).

### M12 — Tests (3–4h)
- [ ] Data-layer tests: schema validation, cleaning idempotence.
- [ ] Feature tests: transformer fits/transforms produce expected shape and dtypes.
- [ ] Model tests: trained pipeline predicts on a synthetic sample.
- [ ] API tests: `TestClient` for every endpoint, including auth failures and validation errors.
- [ ] Coverage ≥ 75% on `src/ml_system/`.

### M13 — Docker & CI/CD (3–4h)
- [ ] Multi-stage `Dockerfile` producing a < 500 MB image.
- [ ] `docker compose up` runs the API.
- [ ] `ci.yml` — lint (ruff), type-check (mypy), tests, build image, push on `main`.
- [ ] Optional: deploy to Fly.io / Render.

### M14 — README & Demo (2–3h)
- [ ] README with: quickstart, architecture diagram, link to model card, `curl` example, screenshot / gif of the API.
- [ ] `make demo` — trains a small model, spins up the API, sends a sample request.
- [ ] "Reproduce from scratch" section works from a fresh clone in < 15 minutes.

---

## 6. Non-negotiables

1. **A baseline exists.** Every real model is compared to a `DummyClassifier` / logistic regression.
2. **All preprocessing lives in a `Pipeline`.** No pandas transformations outside.
3. **CV variance is reported.** Never just a point score.
4. **The chosen threshold is justified** — either by max-F1 or by a business-driven rule ("recall ≥ 0.9").
5. **The test set is touched once.**
6. **The model card exists** and is kept up to date.
7. **API responses include** `request_id`, `model_version`, and calibrated probability.
8. **The service refuses to start** if the model artifact is missing or has the wrong version.
9. **No secrets in the repo.**

---

## 7. Nice-to-Haves (stretch)

- **SHAP-based explanations** in `/v1/predict/explain`.
- **MLflow** or **Weights & Biases** for experiment tracking.
- **DVC** or a small dbt project for the data pipeline.
- **Feature Store** integration (Feast, tiny local variant).
- **A/B two model versions** with a feature flag.
- **Drift alerting** via PagerDuty / Slack webhook.
- **Fairness metrics** across a protected attribute with mitigation experiments.

---

## 8. Evaluation Rubric

| Area | 1 (poor) | 3 (ok) | 5 (excellent) |
|------|----------|--------|---------------|
| Problem framing | vague | metric chosen | metric + cost analysis + baseline expectation |
| Data hygiene | notebooks only | scripts | package + schema + dictionary + tests |
| Baseline | none | dummy classifier | dummy + logistic + reported CV variance |
| Feature engineering | pandas ops | ColumnTransformer | custom transformers + tested + leakage-safe |
| Model selection | best of 1 | tries 2-3 models | full CV comparison with variance |
| Tuning | defaults | grid search | Optuna + honest CV + nested when needed |
| Evaluation | accuracy | AUC | AUC + PR-AUC + calibration + threshold rationale + error analysis |
| Model card | none | brief | full: use, data, metrics, limits, fairness |
| Serving | script | FastAPI endpoint | FastAPI + auth + logs + metadata + calibrated probs |
| Monitoring | none | logs | logs + drift metrics + Prometheus |
| CI/CD | manual | tests | tests + lint + type + build + push |
| Docs | short README | README + model card | README + model card + ADRs + reproducible from scratch |

Aim for 4 or 5 in every row before publishing.

---

## 9. Related Files

- `../goals.md`
- `../learning/09-feature-engineering.md` (pipelines)
- `../learning/10-train-val-test-cv.md` (validation)
- `../learning/11-classification-metrics.md` (metric choice)
- `../learning/12-hyperparameter-tuning.md` (Optuna)
- `../learning/13-scikit-learn.md` (Pipeline, ColumnTransformer)
- `../../foundations/builds/02-ml-api-fastapi.md` (FastAPI patterns)
