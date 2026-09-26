# Month 2 — Machine Learning 🧠 — Goals

> **Track:** Machine Learning (Module 2 of the broader learning journey)
> **Prerequisites:** Foundations complete (Python, NumPy/Pandas, SQL, Git, Linear Algebra, Probability/Stats).
> **Meta-goal:** *Understand WHY a model works, not just `.fit()` it.*

---

## 1. The North Star

The single sentence that governs every decision this month:

> **"When someone hands me a supervised or unsupervised problem, I can pick a model with a defensible reason, engineer features that respect the data-generating process, evaluate honestly with the right metric, tune without leaking, and explain the resulting decisions to a non-ML stakeholder."**

If a topic doesn't move you toward that sentence, deprioritize it.

---

## 2. Terminal Learning Objectives (mastery, not exposure)

By the end of this track you will be able to do **all** of the following without a reference open:

### 2.1 Model Understanding (the "why")
- **LO-1.1** State the **loss function**, **optimization procedure**, and **inductive bias** for linear regression, logistic regression, decision trees, random forests, gradient boosting, SVM, KNN, and k-means.
- **LO-1.2** Derive the closed-form OLS solution and the logistic-regression gradient from first principles.
- **LO-1.3** Explain **overfitting** and **underfitting** in terms of the **bias–variance decomposition**.
- **LO-1.4** Explain how each ensemble method (bagging, boosting, stacking) attacks bias or variance.
- **LO-1.5** Reason about a model's assumptions and how each is likely to be violated in real data.

### 2.2 Practice with scikit-learn
- **LO-2.1** Fluently use `Pipeline`, `ColumnTransformer`, `FunctionTransformer`, and custom `TransformerMixin` classes.
- **LO-2.2** Correctly split data with `train_test_split`, `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `GroupKFold`.
- **LO-2.3** Tune hyperparameters with `GridSearchCV`, `RandomizedSearchCV`, and (bonus) Optuna/`BayesSearchCV` — all **without leakage**.
- **LO-2.4** Compute and interpret precision, recall, F1, ROC-AUC, PR-AUC, log-loss, MAE, RMSE, R², MAPE.
- **LO-2.5** Serialize a full pipeline with `joblib` and load it in a service.

### 2.3 Feature Engineering
- **LO-3.1** Handle missing data — MCAR/MAR/MNAR distinctions and appropriate imputation.
- **LO-3.2** Encode categoricals — one-hot, ordinal, target/frequency encoding, hashing — with leakage-safe fitting.
- **LO-3.3** Transform numerics — scaling, binning, log/Box-Cox/Yeo-Johnson, polynomial features.
- **LO-3.4** Engineer time features (lags, rolling, calendar, holidays) without look-ahead leakage.
- **LO-3.5** Detect and treat data leakage in real pipelines.

### 2.4 Evaluation Discipline
- **LO-4.1** Choose the right validation scheme: holdout, k-fold, stratified, time-series, group.
- **LO-4.2** Choose the right metric for the *decision* the model informs, not the default.
- **LO-4.3** Report models with **confidence intervals** (bootstrap or CV variance), not point scores.
- **LO-4.4** Compare models with a **learning curve** and a **validation curve**.

### 2.5 Communication
- **LO-5.1** Explain any of the models you learned to a non-technical stakeholder in under 5 minutes.
- **LO-5.2** Produce a one-page model card documenting purpose, data, metric, fairness slice, and known limits.

---

## 3. Deliverable

You will produce **one** portfolio-grade artifact:

**Build #1 — End-to-End ML Prediction System.** A trained model with full pipeline, evaluation, model card, and API service. See `builds/01-end-to-end-ml-system.md`.

The build is where you convert reading into muscle memory. If you finish the reading without shipping the build, you have *not* completed this track.

---

## 4. Definition of Done (per topic)

A topic is **done** when you can:

- [ ] Write the **loss function** and **optimization method** from memory.
- [ ] Implement a minimal version **from scratch** in NumPy (except boosting and SVM, where the math is intense — use scikit-learn but explain the internals).
- [ ] Reproduce the scikit-learn result to within numerical tolerance.
- [ ] Name one **assumption** the model makes and one **failure mode** when violated.
- [ ] Score ≥ 8/10 on the module's Self-Assessment Bank.

---

## 5. Anti-goals (what you are *not* doing here)

- **You are not doing deep learning.** Neural nets come next month.
- **You are not chasing Kaggle leaderboards.** You are learning the craft of modeling, not gaming a metric.
- **You are not writing a new library.** scikit-learn is the target; your custom implementations exist only to prove understanding.
- **You are not memorizing formulas without deriving them once.** Derivation once beats memorization forever.
- **You are not skipping the "why does this work" step.** The whole point of this month.

---

## 6. Cadence (suggested)

| Week | Focus |
|------|-------|
| 1 | Linear + Logistic Regression, Decision Trees |
| 2 | Random Forest, XGBoost, SVM, KNN |
| 3 | Clustering, Feature Engineering, Validation/Metrics |
| 4 | Hyperparameter tuning, scikit-learn deep-dive, **Build #1** |

Adjust to reality — but keep the **build immovable**.

---

## 7. Success Signals

You'll know Month 2 is behind you when:

- You reach for a **baseline** (dummy or logistic) before any fancy model.
- You feel physical discomfort when you see `train_test_split` without `stratify` on an imbalanced dataset.
- You spot data leakage in someone else's notebook within 60 seconds.
- You *always* wrap preprocessing + model in a `Pipeline` so CV can't leak.
- You write "the model achieves 0.83 AUC (95% CI [0.79, 0.87])" instead of "0.83 AUC".
- You can defend your metric choice with a business story, not a default.

---

## 8. Related Files

- `learning/01-linear-regression.md`
- `learning/02-logistic-regression.md`
- `learning/03-decision-trees.md`
- `learning/04-random-forest.md`
- `learning/05-xgboost.md`
- `learning/06-svm.md`
- `learning/07-knn.md`
- `learning/08-clustering.md`
- `learning/09-feature-engineering.md`
- `learning/10-train-val-test-cv.md`
- `learning/11-classification-metrics.md`
- `learning/12-hyperparameter-tuning.md`
- `learning/13-scikit-learn.md`
- `builds/01-end-to-end-ml-system.md`
- `self-assessment.md`
