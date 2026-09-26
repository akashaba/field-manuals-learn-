# MLflow — Master Study Guide

> **Track:** Production AI Engineer · **Module:** 05
> **Prerequisites:** Modules 01–04; Month 2/3 ML/DL builds.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** ML in production is really about **model lifecycle management**: track experiments, register versions, promote through staging → prod, roll back safely, and know exactly which model served which prediction. **MLflow** is the open-source standard for this. It's not the only tool (Weights & Biases, DVC, Comet, Neptune all overlap), but it's the reference many enterprises adopt and it's free.

For AI engineers, MLflow is more relevant than most Ops tools because:
- It captures the **experiment → model → deployment** chain in one place.
- It works for both classical ML and LLM workflows.
- It integrates with every serving framework.
- It's self-hostable and vendor-neutral.

**Fundamental principles you must own:**

1. **Experiments** — runs grouped by task. Each **run** logs params, metrics, artifacts.
2. **Model Registry** — versioned, stage-managed model artifacts (Staging, Production, Archived).
3. **Serving** — MLflow can spawn a REST endpoint for any registered model (`mlflow models serve`).
4. **Reproducibility** — every run's code, data hash, environment captured for later reruns.
5. **Bridge to non-ML** — logging RAG evals, agent runs, LLM prompt versions.

If you retain nothing else: **MLflow gives you experiments + a versioned model registry + a way to serve. That's the model-lifecycle stack.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 MLflow Tracking — Experiments and Runs

Log training runs so you can compare, reproduce, and audit:

```python
import mlflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("intent-classifier")

with mlflow.start_run(run_name="lightgbm-v3") as run:
    # Params (hyperparameters, config)
    mlflow.log_params({"n_estimators": 500, "max_depth": 8, "lr": 0.05})

    # Train
    model = train(X_train, y_train, **params)

    # Metrics
    mlflow.log_metrics({"train_auc": 0.92, "val_auc": 0.87, "test_auc": 0.86})

    # Artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_dict(feature_importance, "feature_importance.json")

    # Model — logs and registers in one shot
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="intent-classifier",
        signature=mlflow.models.infer_signature(X_val, model.predict(X_val)),
    )

    print(f"Run: {run.info.run_id}")
```

**What gets stored:**
- Params (typed key-value).
- Metrics (numeric, timestamped — can log over training epochs).
- Artifacts (any file — plots, JSON, model weights).
- Source code snapshot (git commit / uncommitted diff).
- Environment (`requirements.txt`, `conda.yaml`).
- Model itself, with a signature (input/output schema).

**UI.** Access `http://mlflow-server:5000`. Browse experiments, compare runs, filter by params/metrics, view artifacts.

**Autologging.** For common frameworks, one line auto-logs everything:

```python
mlflow.sklearn.autolog()      # or .xgboost / .lightgbm / .tensorflow / .pytorch
model.fit(X, y)               # everything logged automatically
```

**Nested runs** for hyperparameter search:

```python
with mlflow.start_run(run_name="grid-search") as parent:
    for params in param_grid:
        with mlflow.start_run(nested=True) as child:
            model = train(**params)
            mlflow.log_params(params)
            mlflow.log_metric("val_auc", auc)
```

**Cost of tracking.** Metrics and params are cheap. Model artifacts are big — bytes at scale. Use MLflow's artifact backend (S3, GCS, Azure Blob) for large model storage; SQL DB for metadata.

---

### 2.2 Model Registry — Versioning and Stages

Once trained, register the model for lifecycle management:

```python
# Register (from a run)
result = mlflow.register_model(
    "runs:/{run_id}/model".format(run_id=run.info.run_id),
    "intent-classifier",
)
print(result.version)  # "3"

# Or promote existing version
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="intent-classifier",
    version=3,
    stage="Staging",
)
```

**Stages:** `None`, `Staging`, `Production`, `Archived`. Convention:
- **None** — freshly registered; not yet promoted.
- **Staging** — under evaluation.
- **Production** — currently serving traffic.
- **Archived** — retired.

Promote after evaluation:

```python
# In your eval pipeline
if val_auc > baseline_auc:
    client.transition_model_version_stage(name="intent-classifier", version=3, stage="Staging")

# Manual review / A/B → promote to prod
client.transition_model_version_stage(
    name="intent-classifier", version=3, stage="Production",
    archive_existing_versions=True,  # auto-archive the old prod
)
```

**Aliases (newer, preferred).** Instead of stages, MLflow 2.x recommends **aliases** (e.g., `champion`, `challenger`). Any string; you attach an alias to a version:

```python
client.set_registered_model_alias("intent-classifier", "champion", version=3)
client.set_registered_model_alias("intent-classifier", "challenger", version=4)
```

Aliases are more flexible; they support A/B testing patterns cleanly.

**Loading a model in serving code:**

```python
# By alias (recommended)
model = mlflow.pyfunc.load_model("models:/intent-classifier@champion")

# By version
model = mlflow.pyfunc.load_model("models:/intent-classifier/3")

# By stage (legacy)
model = mlflow.pyfunc.load_model("models:/intent-classifier/Production")
```

Then call `model.predict(X)` — works identically regardless of the underlying framework (sklearn, XGBoost, PyTorch, custom).

**Signatures** define input/output schema for validation:

```python
from mlflow.models.signature import infer_signature
signature = infer_signature(X_val, model.predict(X_val))
mlflow.sklearn.log_model(model, "model", signature=signature)
```

At serving time, MLflow validates input against the signature — rejects malformed requests early.

---

### 2.3 Model Serving

MLflow can spawn a REST server for any registered model:

```bash
mlflow models serve -m "models:/intent-classifier@champion" -p 8001 --no-conda
```

`POST /invocations` accepts JSON:

```json
{
  "dataframe_records": [
    {"feature_a": 1.0, "feature_b": "cat"}
  ]
}
```

Returns predictions. Simple; works for prototypes.

**For production**, `mlflow models serve` is often too limited (single-model, no batching, no caching). Options:

- **Wrap MLflow model in FastAPI.** Load with `mlflow.pyfunc.load_model` at startup; serve as your usual FastAPI endpoint. Full control.
- **KServe** — Kubernetes-native serving with MLflow integration.
- **BentoML** — model-serving framework with MLflow adapter.
- **SageMaker MLflow integration** — AWS.
- **Databricks** — first-class MLflow serving.

**Typical pattern for FastAPI:**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = mlflow.pyfunc.load_model("models:/intent-classifier@champion")
    app.state.model_version = mlflow.MlflowClient().get_model_version_by_alias(
        "intent-classifier", "champion"
    ).version
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(body: PredictRequest):
    result = app.state.model.predict([body.dict()])
    return {"prediction": result[0], "model_version": app.state.model_version}
```

Every response includes `model_version` — traceability from prediction to specific model.

---

### 2.4 MLflow for LLMs and Evals

MLflow evolved to cover LLM workflows:

**Log prompts as artifacts:**

```python
with mlflow.start_run():
    mlflow.log_param("model", "claude-...")
    mlflow.log_param("temperature", 0.7)
    mlflow.log_text(system_prompt, "system_prompt.txt")
    mlflow.log_text(user_prompt_template, "user_template.txt")

    # Run eval
    eval_result = evaluate(dataset, prompt_fn)
    mlflow.log_metrics({
        "faithfulness": eval_result.faithfulness,
        "answer_relevancy": eval_result.answer_relevancy,
        "cost_per_task": eval_result.cost,
    })
    mlflow.log_dict(eval_result.per_task, "per_task.json")
```

**MLflow LLM Evaluate** — built-in eval framework:

```python
import mlflow

with mlflow.start_run():
    results = mlflow.evaluate(
        model=lambda df: [my_llm(row["question"]) for _, row in df.iterrows()],
        data=eval_df,
        targets="answer",
        model_type="question-answering",
        evaluators="default",
        extra_metrics=[mlflow.metrics.faithfulness(model=judge_llm)],
    )
    print(results.metrics)
```

Integrates with the tracking/registry lifecycle.

**Traces (MLflow 2.14+).** Log full LLM traces (prompts, tools, responses):

```python
@mlflow.trace
def my_agent(query):
    with mlflow.trace_span("planning"):
        plan = plan(query)
    for step in plan:
        with mlflow.trace_span(f"step_{step}"):
            execute(step)
```

Trace UI in MLflow lets you inspect step-by-step. Alternative to LangSmith / Phoenix for teams that want it under MLflow.

**Prompt versioning.** Track prompt templates as artifacts with the run; roll them back like models. Some teams keep prompts in a git-managed folder and log their content hash per run.

---

### 2.5 Deployment, Scale, and Alternatives

**Self-hosted MLflow.** For teams, run MLflow server backed by a Postgres DB and S3/GCS/Azure Blob for artifacts:

```yaml
# docker-compose.yml
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  command: >
    mlflow server
    --backend-store-uri postgresql://user:pass@db:5432/mlflow
    --default-artifact-root s3://my-bucket/mlflow-artifacts
    --host 0.0.0.0
  ports: ["5000:5000"]
```

**Databricks-hosted MLflow.** Fully managed; deeper integrations (Delta Lake, Unity Catalog). Enterprise-friendly.

**Auth and RBAC.** MLflow OSS has limited auth by default; use a reverse proxy for auth (Nginx + oauth2-proxy), or Databricks / AWS SageMaker for production auth/permissions.

**Alternatives — when NOT MLflow:**

- **Weights & Biases (W&B)** — better UX for deep learning; hosted-first; strong visualizations. Popular in research and DL.
- **Neptune.ai** — similar to W&B; more customizable dashboards.
- **Comet** — competitor with strong experiment tracking.
- **DVC + CML** — git-based data + experiment tracking; more code-centric.
- **ClearML** — open-source ML platform; MLflow-ish.
- **LangSmith / Braintrust / Phoenix** — LLM-specific; better for prompt/agent tracing (see Module 09).

**Choose MLflow if:** you want an OSS, self-hostable, model-lifecycle-focused tool that works for both classical ML and LLMs, with strong tracking + registry.

**Choose W&B if:** deep-learning heavy, want a hosted service, prioritize visualizations.

**Choose LangSmith/Braintrust for LLM-specific work** and MLflow for model registry + classical ML. Some teams run both.

---

## 3. Mental Models & Analogies

### 3.1 The "Lab Notebook + Warehouse" Model

MLflow Tracking is your **lab notebook**: every experiment logged with what you tried, what happened, what fell out. Comparable across experiments; searchable; you can always answer "which config gave the best val AUC last month?"

MLflow Model Registry is your **warehouse**: shelves labeled by model name; versions on each shelf; each shelf tagged with `staging`, `production`, `archived`. Serving code walks to the shelf, grabs the `production` version, ships it out.

Without a lab notebook: experiments blur together; you re-run things you already tried. Without a warehouse: models get emailed around as `.pkl` files; nobody knows which is deployed; rollback = archaeology.

### 3.2 The "Git for Models" Model

MLflow is roughly to models what git is to code:

- **Commits (runs)** — each experiment.
- **Branches (experiments)** — logically grouped work.
- **Tags (versions)** — stable checkpoints.
- **Deploy pointers (aliases)** — which version is `champion` right now.
- **Merge conflicts** — none; models are immutable.

Git for code, MLflow for models. The parallel isn't perfect — models have weights, sizes, and framework specifics that git doesn't — but the mental model transfers.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Ad-Hoc `.pkl` Files Are Fine"

For a solo prototype, yes. Once anyone else needs your model, or you have > 1 model version live, you need a registry. Otherwise: model files emailed around; nobody knows which is in prod; rollback = ping the person who trained it; audit = impossible. A model registry is 10 lines of setup and saves months of confusion.

### 4.2 "MLflow Is Only for Classical ML"

MLflow now supports LLM prompt tracking, LLM evals, and tracing. Registry works for LLM adapters, LoRA weights, or prompt-template versions. Some LLM-specific tools (LangSmith, Braintrust) have better UX for prompt/agent iteration, but MLflow covers most needs.

### 4.3 "Autologging Captures Everything I Need"

Autologging is a great start but incomplete. It logs framework-native metrics (loss, accuracy) but not your business metrics (revenue impact, per-slice accuracy). Always add explicit `mlflow.log_metric` for the numbers your team actually cares about.

---

## 5. Self-Assessment Bank (MLflow)

### Questions

**Q1 (Short answer).** Describe MLflow's three main components.

**Q2 (Multiple choice).** For a serving app that always uses the current best model, the recommended MLflow reference is:
- (a) `models:/my-model/3` (hard-coded version).
- (b) `models:/my-model@champion` (alias) or `models:/my-model/Production` (stage).
- (c) `runs:/{run_id}/model` (from a run).
- (d) `path/to/local/model.pkl`.

**Q3 (Short answer).** Why should every response from a served model include the `model_version`?

**Q4 (Multiple choice).** MLflow experiments and runs relate:
- (a) 1:1 — each experiment has exactly one run.
- (b) 1:N — an experiment groups multiple runs (e.g., different hyperparameter attempts).
- (c) N:1 — many experiments per run.
- (d) N:M — arbitrary.

**Q5 (Short answer).** Contrast MLflow stages and aliases; which is preferred in modern usage?

**Q6 (Multiple choice).** `mlflow models serve -m "models:/my-model@champion"`:
- (a) Downloads the model, spins up a REST endpoint, exposes `POST /invocations`.
- (b) Trains the model.
- (c) Only works for scikit-learn models.
- (d) Requires GPUs.

**Q7 (Short answer).** For LLM prompt tracking in MLflow, what would you log per run?

**Q8 (Multiple choice).** Nested runs are useful for:
- (a) Real-time inference.
- (b) Hyperparameter search — parent run summarizes, child runs are the individual attempts.
- (c) Deploying multiple models at once.
- (d) Nothing useful.

**Q9 (Short answer).** Why is the MLflow model signature useful at serving time?

**Q10 (Multiple choice).** For a team using MLflow, artifacts (model files, plots) should be stored:
- (a) On the MLflow server's local disk.
- (b) In a shared object store (S3 / GCS / Azure Blob) via MLflow's artifact backend.
- (c) In git.
- (d) Emailed to teammates.

---

### Answer Key & Detailed Explanations

**A1.** (1) **MLflow Tracking** — logs experiments (params, metrics, artifacts) per run; provides a UI to browse and compare. (2) **Model Registry** — versioned, stage/alias-managed model artifacts; source of truth for what's deployed. (3) **MLflow Models + Serving** — standardized model format; `mlflow models serve` spins up a REST endpoint for any registered model. Also: MLflow Projects (reproducible run definitions) and MLflow Evaluate (built-in evaluation for classical + LLM models).

**A2. (b).** Use `@alias` (modern, MLflow 2.x preferred) or `/Stage` (legacy). Both let serving code always fetch "the current best" without knowing the specific version number — enables rollback / promotion without redeploying the serving code. Hard-coding a version (a) means every promotion requires a redeploy.

**A3.** So every prediction is traceable to the exact model that made it. When a bug or drift is reported, you know which model was serving; you can reproduce; you can compare against a different version. Also: enables A/B analysis (compare metrics per model version). Non-negotiable in regulated domains (medical, financial).

**A4. (b).** An experiment is a folder / grouping; runs are the individual attempts (varying params, models, algorithms) within it. Search / compare / promote within an experiment. Standard pattern: one experiment per task ("intent-classifier"), many runs per experiment (each training attempt).

**A5.** **Stages** (`None`, `Staging`, `Production`, `Archived`) are a fixed enum; a version can be in one stage at a time. Legacy; still supported. **Aliases** (arbitrary strings like `champion`, `challenger`) are more flexible — you can have multiple, name them for your workflow (e.g., `us-east-champion` vs `eu-champion` for regional rollouts). Modern MLflow docs recommend aliases.

**A6. (a).** `mlflow models serve` downloads the specified model, spins up a REST endpoint on the given port, exposes `POST /invocations`. Works for any MLflow-format model (sklearn, XGBoost, PyTorch, custom pyfunc). Fine for prototypes; for production, wrap the loaded model in FastAPI or use KServe / BentoML.

**A7.** For LLM runs: (1) **Model ID** (e.g., `gpt-4o-2024-08-06`). (2) **Temperature, max_tokens, top_p**. (3) **System prompt** and any few-shot examples (as artifacts). (4) **User-prompt template**. (5) **Eval results** — faithfulness, answer relevance, per-task accuracy. (6) **Cost per task**. (7) **Per-task outputs** (as artifact JSON). (8) **Dataset hash / version** for reproducibility.

**A8. (b).** Nested runs let you group related runs: parent = overall grid search; children = each param combination. Parent has aggregate metrics; children have per-attempt details. Clean UI browsing; easy comparison.

**A9.** The signature declares input/output schema (types, shapes, column names). At serving time, MLflow validates incoming requests against the signature — rejects malformed input before it reaches your model with a clear error. Also documents the API for downstream consumers; auto-generated from a sample during training.

**A10. (b).** MLflow's `--default-artifact-root` points to shared object storage (S3/GCS/Azure Blob). This way: (1) any teammate loading a model via `mlflow.load_model` can fetch it; (2) artifacts persist independent of the MLflow server; (3) scales to large models; (4) integrates with existing data infra. Local disk (a) doesn't share; git (c) can't handle large binaries; email (d) is chaos.

---

## 6. Practice Prompts

1. **Local MLflow.** Run MLflow locally (`mlflow ui`). Train a small model; log params, metrics, and the model. Browse the UI.
2. **Model registration + alias.** Register a model; promote it to `@champion`. In a serving script, load `models:/name@champion` and predict.
3. **Compare runs.** Train 3 versions with different hyperparams. Compare in the UI. Register the best.
4. **LLM eval log.** For a RAG task, log the prompt template, eval dataset hash, and RAGAS metrics per run. Compare 2 prompt versions.
5. **Deploy MLflow server.** Docker-compose MLflow + Postgres + MinIO (S3-compatible). Log a run; verify artifact ends up in S3.

---

## 7. References

- MLflow docs: [mlflow.org/docs](https://mlflow.org/docs/latest/index.html).
- MLflow Models: [mlflow.org/docs/latest/models.html](https://mlflow.org/docs/latest/models.html).
- MLflow Model Registry: [mlflow.org/docs/latest/model-registry.html](https://mlflow.org/docs/latest/model-registry.html).
- MLflow LLM evaluate.
- Weights & Biases: [wandb.ai](https://wandb.ai/).
- BentoML — model serving framework.
- KServe — Kubernetes-native model serving.
