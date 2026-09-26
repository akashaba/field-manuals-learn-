# Build #1 — Image & Text Classification System

> **Track:** Deep Learning · **Build:** 01
> **Prerequisites:** All Deep-Learning learning modules (01–12), Foundations & Machine Learning builds.
> **Time budget:** ~40–60 hours over ~2–3 weeks.
> **Deliverable:** A public GitHub repo containing two working classifiers — a CNN-based image classifier AND a transformer-based text classifier — unified under one training/serving framework, dockerized and CI'd.

---

## 1. Executive Summary

This capstone forces you to work with **both** vision (CNN) and language (transformer) — the two pillars of modern deep learning. Beyond a single-task ML system, you'll build a **framework** that can host multiple task-specific models under a common training/inference/serving substrate.

Why two models?
- **Vision** and **language** are the two dominant DL problem shapes; understanding both is core.
- Doing them **within one framework** forces good software engineering — you can't hack per-task shortcuts.
- The end product is portable — you can add new tasks (audio, tabular, time series) later with minimal ceremony.

Concretely, you will:

1. Train a **CNN image classifier** on a real dataset (CIFAR-10, Fashion-MNIST, or a custom set).
2. Train a **transformer text classifier** on a real dataset (IMDB, AG News, or a domain-specific set).
3. Wrap both under a common **training framework** (data loader → model → optimizer → scheduler → mixed precision → logging → checkpointing).
4. Expose both via a **FastAPI service** with per-model endpoints.
5. **Fine-tune a pretrained model** (ResNet-50 for images; a small BERT/DistilBERT for text) — the modern practical approach.
6. Add **inference optimizations** — batching, mixed precision, caching.
7. **Dockerize + CI/CD.**

---

## 2. Learning Outcomes

By the end you will have practiced:

1. Training CNNs and transformers end-to-end in PyTorch.
2. Using **pretrained models** with fine-tuning (transfer learning).
3. Building a **reusable training framework** — not per-task scripts, but composable modules.
4. Writing production-quality PyTorch training loops (mixed precision, gradient clipping, checkpointing).
5. Handling **data augmentation** appropriately for images vs. text.
6. Serving multiple model types under a common API.
7. **Model card + evaluation** for each task with the right metrics (accuracy for balanced tasks; PR-AUC for imbalanced; per-class breakdowns for fairness).
8. Docker packaging for a GPU-optional service.

---

## 3. Scope: Pick Your Datasets

**For the image classifier**, pick one:
- **CIFAR-10 / CIFAR-100** — 32×32 color images, 10 / 100 classes. Small enough to train from scratch quickly.
- **Fashion-MNIST** — 28×28 grayscale, 10 classes. Even faster iteration.
- **Custom small dataset** — e.g., 500 photos across 5 categories you care about (with data augmentation).
- **Stanford Dogs / Food-101 / Caltech-101** — pretrained backbone + fine-tuning story.

**For the text classifier**, pick one:
- **IMDB reviews** — binary sentiment. Classic baseline.
- **AG News** — 4-class topic classification. Balanced, moderate size.
- **HuggingFace `emotion` dataset** — 6-class emotion classification. Small.
- **Domain-specific** — e.g., customer support ticket categorization, or legislative bill topic tagging (nod to the day job).

**Rules:**
- Datasets must have public, redistributable licenses OR be your own data.
- Each dataset gives clear train / val / test splits.
- Class distributions are known — check imbalance before choosing metrics.

---

## 4. Repository Structure

```
dl-classifiers/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── docker.yml
├── configs/
│   ├── image_cifar10.yaml
│   ├── text_imdb.yaml
│   └── defaults.yaml
├── data/
│   ├── raw/               # gitignored
│   └── processed/         # small samples for tests
├── notebooks/
│   ├── 01_image_eda.ipynb
│   ├── 02_text_eda.ipynb
│   └── 03_error_analysis.ipynb
├── src/
│   └── dl_classifiers/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── image_dataset.py
│       │   └── text_dataset.py
│       ├── models/
│       │   ├── cnn.py           # your CNN (from scratch + ResNet-based)
│       │   ├── transformer.py   # small text transformer
│       │   ├── pretrained.py    # HuggingFace/torchvision loaders
│       │   └── registry.py      # look up model by name
│       ├── training/
│       │   ├── loop.py           # generic train/val loop
│       │   ├── mixed_precision.py
│       │   ├── callbacks.py      # checkpoint, early-stop, LR log
│       │   └── metrics.py
│       ├── inference/
│       │   ├── predictor.py      # image + text
│       │   └── batch.py          # dynamic batching
│       └── api/
│           ├── main.py           # FastAPI
│           ├── schemas.py
│           ├── auth.py
│           └── routes/
│               ├── image.py
│               └── text.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_api.py
├── artifacts/
│   ├── image_v1.pt              # small, may commit
│   ├── text_v1.pt               # or reference to cloud storage
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

### M1 — Kickoff & Scaffolding (2–3h)
- [ ] Repo on GitHub; MIT/Apache license; skeleton README.
- [ ] `pyproject.toml` pinning: torch, torchvision, transformers, datasets, pydantic, fastapi, uvicorn, ruff, mypy, pytest.
- [ ] Pre-commit hooks: ruff, black, isort.
- [ ] CI on push: lint + type-check + tests.
- [ ] `Makefile`: `setup`, `train-image`, `train-text`, `serve`, `test`, `docker`, `clean`.

### M2 — Data Pipeline for Images (3–4h)
- [ ] `image_dataset.py` — wraps `torchvision.datasets` or custom `ImageFolder`.
- [ ] Standard augmentations (RandomCrop, Flip, ColorJitter, Normalize) at train time; center-crop + normalize at eval.
- [ ] `DataLoader` factory with pin_memory + num_workers configured via YAML.
- [ ] Unit tests: verify shapes, dtypes, augmentation determinism (with seed).

### M3 — CNN From Scratch (4–5h)
- [ ] A ResNet-style CNN implemented from scratch (no `torchvision.models`): stem + 3–4 stages of residual blocks + GAP + classifier.
- [ ] Trained on CIFAR-10 to at least **90% test accuracy** (or comparable target for your dataset).
- [ ] Training curve plotted (train/val loss + accuracy).
- [ ] Saved as `artifacts/image_v1.pt`.

### M4 — Pretrained Fine-Tune (3–4h)
- [ ] Load a pretrained model (`torchvision.models.resnet50(weights='DEFAULT')`).
- [ ] Two strategies: (a) freeze backbone, train head only; (b) full fine-tune with lower LR.
- [ ] Comparison of test accuracy vs the from-scratch CNN.
- [ ] Save as `artifacts/image_pretrained_v1.pt`.

### M5 — Data Pipeline for Text (2–3h)
- [ ] `text_dataset.py` uses HuggingFace `datasets` + a tokenizer (BERT / DistilBERT / RoBERTa).
- [ ] Tokenized to fixed max length; padded via `pad_sequence` in `collate_fn` or the tokenizer's dynamic padding.
- [ ] Handles class labels; can filter/subset by class.

### M6 — Transformer From Scratch (5–7h)
- [ ] Build a small transformer encoder (2–4 layers, $d_\text{model}$ = 128–256).
- [ ] Includes: token embedding, positional encoding, N transformer blocks, mean-pool, classifier.
- [ ] Trained on IMDB (or your text dataset) — **at least 85% accuracy** on IMDB.
- [ ] Saved as `artifacts/text_v1.pt`.

### M7 — Pretrained Fine-Tune for Text (3–4h)
- [ ] Fine-tune `distilbert-base-uncased` (or similar) on your text task.
- [ ] Use HuggingFace `AutoModelForSequenceClassification`.
- [ ] Compare to the from-scratch transformer.
- [ ] Save as `artifacts/text_pretrained_v1.pt`.

### M8 — Unified Training Framework (4–5h)
- [ ] `training/loop.py` — a **generic** train/val loop that takes model + loaders + optimizer + scheduler + config.
- [ ] Callbacks: checkpoint on best val, early stopping, LR schedule step.
- [ ] Mixed precision via `autocast` + `GradScaler`.
- [ ] Gradient clipping.
- [ ] Same loop trains both the image and text tasks.
- [ ] YAML configs drive experiments.

### M9 — Metrics & Model Card (2–3h)
- [ ] `training/metrics.py` — accuracy, precision/recall/F1 per class, confusion matrix, top-k accuracy for images.
- [ ] Reliability diagram for calibration.
- [ ] Per-class breakdown to spot fairness / minority-class issues.
- [ ] `docs/model_card_image.md` and `docs/model_card_text.md` covering: purpose, dataset, metrics, limitations.

### M10 — Inference Layer (3–4h)
- [ ] `inference/predictor.py` — `ImagePredictor` and `TextPredictor` classes; load model at init, `.predict(x)` returns probabilities + label.
- [ ] Batch prediction — accept a list of inputs.
- [ ] Preprocessing consistent with training (same normalization, tokenizer).
- [ ] Half-precision inference option.

### M11 — API Serving (4–5h)
- [ ] FastAPI service with:
  - `GET /healthz` and `GET /readyz`.
  - `POST /v1/image/predict` — accepts a base64 image or a URL.
  - `POST /v1/text/predict` — accepts a text string.
  - `POST /v1/text/predict/batch` — accepts a list.
- [ ] Pydantic schemas for inputs/outputs.
- [ ] API-key auth.
- [ ] Structured logs with request_id + latency.
- [ ] Model version returned in every response.

### M12 — Docker + Compose (2–3h)
- [ ] Multi-stage `Dockerfile` producing a < 4 GB image (torch is heavy; be reasonable).
- [ ] `docker compose up` runs the API.
- [ ] Environment variables for model paths.
- [ ] Non-root runtime user.

### M13 — Tests (3–4h)
- [ ] Unit tests for data pipelines (shape, dtype, augmentation determinism).
- [ ] Unit tests for models (forward pass produces expected shapes).
- [ ] Integration tests for the training loop (train for 1 step on a small batch).
- [ ] API tests via `TestClient`.
- [ ] Coverage ≥ 70% on `src/dl_classifiers/`.

### M14 — CI/CD (2h)
- [ ] `ci.yml`: lint, type-check, tests.
- [ ] `docker.yml`: build + push to GHCR on `main`.
- [ ] Badges in README.

### M15 — Docs & Demo (2–3h)
- [ ] README quickstart, architecture diagram, model cards linked.
- [ ] Example `curl` commands for both endpoints.
- [ ] Live demo URL if deployed (Fly.io / Render — noting GPU is optional; CPU inference works but slower).

---

## 6. Non-negotiables

1. **A from-scratch implementation of each model type exists**, alongside the pretrained versions. You need both to prove understanding.
2. **The training loop is generic** — no task-specific hacks.
3. **All preprocessing is inside the model / pipeline**, so inference is a single call from raw input.
4. **Mixed precision + gradient clipping** are enabled by default.
5. **`.eval()` + `torch.no_grad()`** wrap every inference path.
6. **Every response includes model_version.**
7. **No secrets in the repo.**
8. **Reproducible training** — fixed seeds and pinned library versions.

---

## 7. Nice-to-Haves (stretch)

- **ONNX export** for the trained models; verify inference via ONNX Runtime.
- **LoRA fine-tuning** on a larger text model (e.g., LLaMA-7B or Mistral-7B).
- **Serving optimization** with `torch.compile` or dynamic batching (Ray Serve, vLLM if text-generation).
- **Grad-CAM** or attention visualizations for interpretability.
- **A/B two models per task** with a feature flag.
- **Multi-modal experiment** — classify an image + caption pair (concatenate CNN and transformer features).
- **Model quantization** — int8 with `torch.quantization`; benchmark accuracy vs speed.

---

## 8. Evaluation Rubric

| Area | 1 (poor) | 3 (ok) | 5 (excellent) |
|------|----------|--------|---------------|
| From-scratch models | none | one built | both built, meet accuracy targets |
| Pretrained fine-tune | none | one done | both with side-by-side comparison |
| Training framework | per-task scripts | shared functions | fully generic Trainer with callbacks |
| Data pipeline | in-notebook | Dataset + DataLoader | + augmentation, seed, tests |
| Metrics | accuracy only | accuracy + F1 | + per-class, calibration, confusion matrix |
| Mixed precision | not used | enabled | + gradient clipping + scaler checkpointing |
| API design | script | FastAPI endpoint | + auth, versioning, logs, batch, model_version |
| Docker | not done | works locally | multi-stage, small, non-root |
| CI/CD | manual | tests | tests + build + push |
| Docs | README | + model cards | + ADRs + reproducibility instructions |

Aim for 4 or 5 across the board before shipping.

---

## 9. Related Files

- `../goals.md`
- `../learning/07-cnns.md`
- `../learning/10-transformers.md`
- `../learning/11-llms.md` (for pretrained fine-tuning details)
- `../learning/12-pytorch.md` (for the training-loop patterns)
- `../../foundations/builds/02-ml-api-fastapi.md`
- `../../machine-learning/builds/01-end-to-end-ml-system.md`
