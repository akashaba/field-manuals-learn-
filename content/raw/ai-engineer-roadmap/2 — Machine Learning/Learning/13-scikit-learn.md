# scikit-learn — Master Study Guide

> **Track:** Machine Learning · **Module:** 13
> **Prerequisites:** Modules 01–12 (theory) + Python (Foundations Module 01).
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** scikit-learn is the single most widely-used machine-learning library in the world for tabular data. It's the *lingua franca* — nearly every ML tutorial, blog post, book, and job interview references it. Learning it well means:

- You can fluently express any tabular ML idea in a single, consistent API.
- You inherit thousands of hours of collective bug-fixing and correctness testing.
- You get free interoperability with the entire Python ML ecosystem (Pandas, NumPy, XGBoost, LightGBM, matplotlib, Optuna).

More than a library, scikit-learn is a **design philosophy**: every model exposes the same tiny interface (`fit`, `predict`, `transform`, `score`); everything composes; everything obeys the same conventions. Once you internalize the API, learning a new model is minutes, not hours.

**Fundamental principles you must own:**

1. **Estimator API** — every model has `.fit(X, y)`, `.predict(X)`, and (usually) `.predict_proba(X)` and `.score(X, y)`.
2. **Transformer API** — preprocessing steps have `.fit(X)`, `.transform(X)`, and `.fit_transform(X)`.
3. **Pipeline** chains transformers + a final estimator into a single object with the same API.
4. **ColumnTransformer** applies different transformers to different columns.
5. **All CV utilities** work with any estimator via the standard API — because everything speaks the same protocol.
6. **`random_state`** governs randomness; **fix it for reproducibility**.

If you retain nothing else: **fit, transform, predict — that's the API. Everything else is combinators.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Estimator API

Every scikit-learn model is an **estimator** — a class with a consistent interface.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)  # __init__ sets hyperparams
model.fit(X_train, y_train)                                        # learns from data
y_pred  = model.predict(X_test)                                    # class predictions
y_prob  = model.predict_proba(X_test)                              # probabilities (if supported)
score   = model.score(X_test, y_test)                              # default metric (accuracy for classifiers, R² for regressors)
coef    = model.coef_                                              # learned parameters (trailing _ means "learned")
```

**Conventions:**

- **Hyperparameters** (set via `__init__`) never have trailing underscores.
- **Learned attributes** always end with `_` (e.g., `coef_`, `feature_importances_`, `classes_`).
- **`fit`** returns `self` to allow chaining (`model.fit(X, y).predict(X_test)`).
- **`X`** is 2D (`n_samples × n_features`); **`y`** is 1D for classification/regression, 2D for multi-output.
- **Common types:** NumPy arrays, Pandas DataFrames, and (for many estimators) SciPy sparse matrices are all accepted.

**Estimator variants:**

- **Classifiers** — `predict`, `predict_proba`, `decision_function`. `classes_` attribute.
- **Regressors** — `predict`.
- **Clusterers** — `fit`, `predict` (or `labels_` for models fit-only like DBSCAN).
- **Transformers** — `fit`, `transform`, `fit_transform`.
- **Meta-estimators** wrap other estimators: `Pipeline`, `GridSearchCV`, `BaggingClassifier`.

**Building blocks in `sklearn.base`:**

- `BaseEstimator` — the base class.
- `ClassifierMixin`, `RegressorMixin`, `ClusterMixin`, `TransformerMixin` — mixins providing `score` methods and common behavior.

**Custom estimators:** subclass `BaseEstimator` + the appropriate mixin, implement `fit` and `predict` (or `transform`). Then your estimator plugs into `Pipeline`, `GridSearchCV`, `cross_val_score`, etc. — for free.

---

### 2.2 Pipelines and ColumnTransformer

The **`Pipeline`** chains transformers and a final estimator:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf",   LogisticRegression(C=1.0)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

- The **last step** is an estimator (has `fit` + `predict`); all earlier steps are transformers (have `fit` + `transform`).
- Calling `pipe.fit(X_train, y_train)` sequentially fits and transforms through every step, then fits the final estimator on the transformed features.
- Calling `pipe.predict(X_test)` transforms `X_test` through every step (using fitted transformers) before calling the final estimator's `predict`.

**Why this matters:**

- **Leakage prevention.** `pipe.fit(X_train)` only sees training data. When you pass `pipe` to `cross_val_score`, each fold refits the entire pipeline — no test-fold data ever touches the transformers' fits.
- **Serialization.** `joblib.dump(pipe, "model.pkl")` saves the entire preprocessing + model. Deploying the pipeline means one call — no custom preprocessing in production.
- **Hyperparameter tuning across steps.** `GridSearchCV(pipe, {"scale__with_mean": [True, False], "clf__C": [0.1, 1, 10]})`. The double-underscore syntax targets nested steps.

**`ColumnTransformer`** applies different transformers to different columns:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numeric_cols = ["age", "salary", "hours_per_week"]
categorical_cols = ["country", "occupation"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc",  StandardScaler())]),                       numeric_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh",  OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
], remainder="drop")

full_pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression())])
```

**Options:**
- **`remainder`**: `"drop"` (default), `"passthrough"`, or another transformer.
- **`transformers`**: list of `(name, transformer, columns)` tuples.
- Columns can be names, indices, or a `make_column_selector(dtype_include=np.number)` selector.

**`make_pipeline` and `make_column_transformer`** — shorter forms that auto-name steps.

---

### 2.3 Model Selection: CV, Search, Metrics

The `sklearn.model_selection` and `sklearn.metrics` modules are your evaluation toolbox.

**Splitters:**

- `train_test_split(X, y, test_size, stratify, random_state)` — one-shot split.
- `KFold(n_splits=5, shuffle=True, random_state=42)` — random k-fold.
- `StratifiedKFold` — preserves class ratios.
- `TimeSeriesSplit` — forward-only splits for time-series.
- `GroupKFold`, `LeaveOneGroupOut`, `StratifiedGroupKFold` — group-aware splits.

**Scorers:**

Metrics live in `sklearn.metrics`. Every function has a `_score` or `_error` variant. To pass a metric to CV utilities:

```python
from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, average="macro")
scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
```

Or use predefined string names: `"accuracy"`, `"roc_auc"`, `"f1"`, `"neg_mean_squared_error"`, etc. Note that error metrics are **negated** so higher is always better.

**Search:**

- `GridSearchCV(estimator, param_grid, cv, scoring)` — exhaustive.
- `RandomizedSearchCV(estimator, param_distributions, n_iter, cv, scoring)` — sampled.
- `HalvingGridSearchCV`, `HalvingRandomSearchCV` — successive halving; much faster on large param spaces.

**Cross-validation helpers:**

- `cross_val_score(model, X, y, cv, scoring)` — scores per fold.
- `cross_val_predict(model, X, y, cv)` — out-of-fold predictions.
- `cross_validate(model, X, y, cv, scoring, return_train_score)` — richer output (multiple metrics, times).

**Learning + validation curves:**

- `learning_curve(estimator, X, y, train_sizes, cv, scoring)`.
- `validation_curve(estimator, X, y, param_name, param_range, cv, scoring)`.

---

### 2.4 Preprocessing & Feature Engineering Utilities

**Numeric preprocessing** — in `sklearn.preprocessing`:

- `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`.
- `PowerTransformer(method="yeo-johnson"|"box-cox")` — normality-inducing.
- `QuantileTransformer` — rank-based; useful for heavily skewed data.
- `Normalizer` — row-wise L1/L2 normalization (rare; for cosine similarity setups).
- `PolynomialFeatures(degree=2, interaction_only=False)`.
- `SplineTransformer`, `KBinsDiscretizer` — for nonlinear transformations.

**Categorical preprocessing:**

- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.
- `TargetEncoder` (sklearn ≥ 1.3).
- Others via `category_encoders` package.

**Imputation** — `sklearn.impute`:

- `SimpleImputer(strategy="mean"|"median"|"most_frequent"|"constant", fill_value=...)`.
- `KNNImputer(n_neighbors=5)`.
- `IterativeImputer` — MICE.
- Pass `add_indicator=True` to add a missing-indicator column.

**Feature selection** — `sklearn.feature_selection`:

- `VarianceThreshold` — drop near-constant features.
- `SelectKBest(score_func=f_classif, k=10)` — filter methods.
- `RFECV(estimator, cv, step)` — recursive feature elimination with CV.
- `SelectFromModel(estimator, threshold="mean")` — embedded (Lasso, tree-based).

**Custom transformers via `FunctionTransformer`:**

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np
log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                      validate=True, feature_names_out="one-to-one")
```

Or subclass `BaseEstimator + TransformerMixin` for arbitrary logic.

---

### 2.5 Serialization, Deployment, and Production Gotchas

**Serialization.** Use `joblib` for scikit-learn models:

```python
import joblib
joblib.dump(pipeline, "model.joblib")
loaded = joblib.load("model.joblib")
loaded.predict(new_X)
```

**Reasons to use joblib over pickle:**
- Faster for large NumPy arrays (uses efficient array serialization).
- Standard scikit-learn recommendation.
- Compressible (`joblib.dump(..., compress=3)`).

**Reasons to *not* use pickle-based serialization (either) for interop:**
- Not language-agnostic.
- Version-fragile — models saved with sklearn 1.3 may not load in 1.4.
- Not human-inspectable.

**Alternatives for portability:**
- **ONNX** — cross-framework, language-agnostic. `sklearn-onnx` converter.
- **PMML** — older XML-based format.
- **Custom JSON schema** for simple models (linear, trees).

**Production gotchas:**

1. **Feature schema drift.** Ensure inference-time inputs match training-time schema: same columns, same order, same dtypes, same missing-value handling. Wrap in a `Pipeline` that starts with a `ColumnTransformer` referencing column names.
2. **Category set drift.** New categorical values at inference time. Use `handle_unknown="ignore"` in encoders.
3. **NumPy dtype quirks.** `float32` vs `float64` can silently change results. Standardize dtypes throughout.
4. **Version pinning.** Pin scikit-learn (and NumPy, SciPy, pandas) versions across training and inference environments. A model saved with a slightly different pandas version may fail to deserialize.
5. **Batch vs single-row inference.** `.predict(row.reshape(1, -1))` — always 2D.
6. **Reproducibility.** Fix `random_state` at every random step. Also fix `np.random.seed(...)`, `random.seed(...)`. In parallel jobs, seeds don't fully control non-determinism — do a `n_jobs=1` sanity check.

**Common patterns:**

- Wrap the entire pipeline (preprocessing + model) in one `joblib`-serialized artifact.
- Version the model file (e.g., `model_v1.2.3.joblib`).
- Store training metadata alongside: dataset hash, sklearn version, git SHA, training date, evaluation metrics.
- Enable model card / `metadata.json` — see Build #1.

---

## 3. Mental Models & Analogies

### 3.1 The "LEGO Blocks" Model

scikit-learn is a set of LEGO blocks with a **shared connector shape**. Every estimator has the same "studs" (`fit` / `predict` / `transform` / `score`), which means:

- Any transformer snaps onto any other transformer.
- The last block is a special one with a `predict` connector.
- Meta-estimators (Pipeline, GridSearchCV, VotingClassifier) are giant blocks that accept other blocks as parts.

This is why you can pick up a new algorithm in 5 minutes: the API is identical to what you already know. **The API is the abstraction, not the models.**

### 3.2 The "Assembly Line" Model (Pipelines)

A `Pipeline` is an assembly line. Raw material (data) enters one end; a finished product (predictions) exits the other. Each station transforms the material into a shape the next station can consume.

- **Fitting** = commissioning the line: each station learns its parameters from the material passing through, in order.
- **Predicting** = running the line at speed: each station applies what it learned, without re-learning.
- **Serialization** = shipping the whole factory in a crate — set it up on any machine with the crate.
- **`ColumnTransformer`** = branching lines: numeric widgets go left, categorical widgets go right, then rejoin at a merge station.

Once the pipeline is designed, you never touch individual stations again — you `fit` and `predict` the entire line.

![IMG-SKL-01](/2%20—%20Machine%20Learning/images/IMG-SKL-01.jpg)
> **Caption:** A Pipeline chains transformers ending in an estimator; the whole line is a single object with the standard API.
> **Placement:** Section 2.2.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "I'll Preprocess in Pandas, Then Feed to sklearn"

Common but leaky. If you scale in pandas using the whole DataFrame's mean and std, then split, test-set values contributed to the mean. Every CV fold's preprocessing is fixed rather than re-fit. Use `Pipeline` — it's the correct discipline.

### 4.2 "`.predict_proba` Gives Calibrated Probabilities"

`.predict_proba` gives whatever the model considers a probability — which for trees, forests, SVMs, and Naive Bayes is often miscalibrated. If you use the values downstream (thresholding on a specific probability level, cost-sensitive decisions), wrap the model in `CalibratedClassifierCV(method="isotonic")` first.

### 4.3 "`n_jobs=-1` Is Always Faster"

Not on tiny datasets — parallelization has overhead. Not with algorithms that hit BLAS-level parallelism internally (NumPy on multi-threaded MKL) — you can oversubscribe and slow down. Not in Jupyter with certain macOS setups (fork issues). Rule: start with `n_jobs=1` for small work, `n_jobs=-1` for large work; benchmark.

---

## 5. Self-Assessment Bank (scikit-learn)

### Questions

**Q1 (Short answer).** Describe the estimator API: what methods does every estimator implement?

**Q2 (Multiple choice).** After training, which attribute of a fitted `RandomForestClassifier` gives feature importances?
A. `feature_importances`
B. `feature_importances_`
C. `importances_`
D. `coef_`

**Q3 (Short answer).** Why is wrapping preprocessing and a model in a `Pipeline` essential for leakage-free CV?

**Q4 (Multiple choice).** In a `Pipeline`, `pipe.get_params()["clf__C"]` refers to:
A. A hyperparameter of a step named `clf`.
B. The score of the classifier.
C. A method of the pipeline.
D. An error — invalid syntax.

**Q5 (Short answer).** What is a `ColumnTransformer` and when do you use it?

**Q6 (Multiple choice).** `RandomizedSearchCV` differs from `GridSearchCV` in that it:
A. Randomizes the training data.
B. Samples hyperparameter combinations from distributions instead of a fixed grid.
C. Randomly picks the model class.
D. Doesn't require cross-validation.

**Q7 (Short answer).** In scikit-learn, why do error metrics like MSE appear as `neg_mean_squared_error` in `scoring=` arguments?

**Q8 (Multiple choice).** Which is the recommended way to serialize a scikit-learn model?
A. `pickle.dump`
B. `joblib.dump`
C. `json.dump`
D. `numpy.savez`

**Q9 (Short answer).** What does `handle_unknown="ignore"` do in `OneHotEncoder`, and why does it matter in production?

**Q10 (Multiple choice).** To get consistent, reproducible results across scikit-learn runs, you should:
A. Set `random_state` everywhere and pin library versions.
B. Restart your Jupyter kernel.
C. Use `n_jobs=1` only.
D. Nothing — sklearn is deterministic.

---

### Answer Key & Detailed Explanations

**A1.** Every estimator has:
- `__init__(**hyperparams)` — set hyperparameters (no data touched).
- `fit(X, y)` — learn from data; returns `self`.
- `predict(X)` — make predictions (classifiers/regressors).
- Optionally: `predict_proba(X)`, `decision_function(X)`, `transform(X)`, `fit_transform(X)`, `score(X, y)`.

Learned attributes end with `_` (e.g., `coef_`, `classes_`, `n_features_in_`).

**A2. B.** Trailing underscore = learned during `fit`. All scikit-learn learned attributes follow this convention.

**A3.** A `Pipeline`'s `fit(X_train)` sequentially fits and transforms each step using **only** the training data. When you pass a `Pipeline` to `cross_val_score` or `GridSearchCV`, each CV fold refits the entire pipeline from scratch — the fold's held-out data is never seen by the transformers. Without a pipeline, doing preprocessing before the CV split means the preprocessing has already seen "test" data, leaking information.

**A4. A.** Double underscore accesses nested step parameters: `<step_name>__<param_name>`. `clf__C` = the `C` hyperparameter of the step named `clf`. This lets `GridSearchCV` tune parameters of any step in the pipeline.

**A5.** `ColumnTransformer` applies **different transformers to different columns** in parallel. Numeric columns get scaled; categorical columns get one-hot encoded; text columns get vectorized. It's the correct tool whenever your dataset has mixed types, so you don't have to hand-slice and re-concatenate.

**A6. B.** `RandomizedSearchCV` samples `n_iter` combinations from user-specified distributions (`scipy.stats` distributions or lists). It typically finds good hyperparameters faster than a large grid, especially in high-dimensional hyperparameter spaces.

**A7.** scikit-learn's convention is that **higher is always better** for scorers. Since MSE (or any error) is lower-is-better, its scorer is negated: minimizing MSE = maximizing `neg_mean_squared_error`. This convention lets `GridSearchCV.best_score_` etc. always mean "highest" regardless of metric direction.

**A8. B.** `joblib.dump` is optimized for large NumPy arrays, is the scikit-learn team's recommendation, supports compression, and is faster than `pickle` for typical models. `pickle` works but is slower on arrays.

**A9.** When `OneHotEncoder(handle_unknown="ignore")` encounters a category at `.transform` time that wasn't seen during `.fit`, it outputs a row of all zeros for that categorical position instead of raising an error. In production, new categorical values are inevitable (new countries, new product IDs, new users) — this setting prevents crashes. Trade-off: the unknown category has zero signal to the downstream model. `handle_unknown="infrequent_if_exist"` is a newer option that groups rare categories.

**A10. A.** Fix `random_state` on every source of randomness (train_test_split, KFold, model constructor, etc.). Also fix `np.random.seed()` and `random.seed()` if any code uses them. Pin library versions (`scikit-learn`, `numpy`, `scipy`) — internal changes can affect exact results. Even then, `n_jobs > 1` and different BLAS implementations can introduce minor numerical differences.

---

## 6. Practice Prompts

1. **Full pipeline.** Build a `Pipeline` combining `SimpleImputer` + `OneHotEncoder` for categoricals, `SimpleImputer` + `StandardScaler` for numerics, and `LogisticRegression` as the final estimator. Fit on a real dataset; evaluate with `cross_val_score`.
2. **Custom transformer.** Write a `WeekdayExtractor` transformer (subclassing `BaseEstimator + TransformerMixin`) that extracts weekday + is_weekend features from a `datetime` column. Plug it into a `Pipeline`.
3. **GridSearchCV on a pipeline.** Tune the `C` of the logistic regression AND the `strategy` of the imputer in a single `GridSearchCV`. Use the `step__param` syntax.
4. **Save & load.** Fit a pipeline, `joblib.dump` it, restart Python, `joblib.load`, and predict on a new sample. Confirm output identical.
5. **ONNX export.** Convert your fitted pipeline to ONNX with `sklearn-onnx`. Load it in `onnxruntime` and verify predictions match the original.

---

## 7. References

- scikit-learn user guide: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html) — pretty much every module and concept, with examples.
- Pedregosa et al., ["Scikit-learn: Machine Learning in Python"](https://www.jmlr.org/papers/v12/pedregosa11a.html) (2011) — the original paper.
- API design paper: Buitinck et al., ["API design for machine learning software: experiences from the scikit-learn project"](https://arxiv.org/abs/1309.0238) (2013).
- Andreas Müller & Sarah Guido, *Introduction to Machine Learning with Python* (uses scikit-learn extensively).
