# Train / Validation / Test & Cross-Validation — Master Study Guide

> **Track:** Machine Learning · **Module:** 10
> **Prerequisites:** Modules 01–09 (context on models and features).
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every ML model's final claim rests on **how you evaluated it**. Get the split wrong and every subsequent number is a lie. Real teams have shipped models with 0.95 CV AUC that dropped to 0.60 in production — always because of a leaky or unrepresentative validation scheme.

Getting this right is not glamorous. It's engineering discipline. But it's **the single most impactful thing you can do to make your models actually work**.

**Fundamental principles you must own:**

1. **The test set is sacred.** You look at it *once*, at the very end, and never touch it during modeling. If you look before, it's no longer a test set.
2. **The validation set is where you make modeling decisions.** Hyperparameter search, feature selection, model choice — all happens on validation.
3. **Cross-validation gives more stable estimates** than a single split, at the cost of $k$× compute.
4. **The split must reflect production reality.** Random splits are only right when data is i.i.d.; time-series, grouped, or imbalanced data need special splits.
5. **Every transformation must be fit only on training data** — inside each fold, in CV.
6. **Report variance, not just point estimates.** "Accuracy 0.83 ± 0.02" > "Accuracy 0.83."

If you retain nothing else: **your validation strategy must simulate deployment as closely as possible.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Three-Way Split (Train / Validation / Test)

The classic split serves three distinct purposes:

- **Training set** (~60–80%) — fit model parameters.
- **Validation set** (~10–20%) — tune hyperparameters, compare models, decide features.
- **Test set** (~10–20%) — a single, final estimate of production performance.

**Why three, not two?** If you tune hyperparameters on your test set, you *leak* — you've selected the model that best fits that particular test set. Your test estimate is now optimistically biased. Keep a strictly untouched **test set** so the final number is trustworthy.

**Split proportions.**

- **Very small data (< 1k rows)**: 60/20/20, or use k-fold CV in place of a separate validation set.
- **Medium data (1k–100k)**: 70/15/15 or 80/10/10.
- **Big data (millions+)**: 98/1/1 might be plenty — 10k rows is often enough for a stable estimate.

**Stratification.** For classification (especially imbalanced), stratify the split by class label so each part has the same class proportions:

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

Same for k-fold: use `StratifiedKFold`.

**Time-based splits.** For time-series or any data where the future is what you're predicting, **the split must respect time**:

- Train on the past, validate/test on the future.
- Never shuffle time-series data.
- Use `TimeSeriesSplit` for CV.

**Group-based splits.** When rows aren't independent (multiple rows per patient, per user, per session), keep all rows for one group together in a single split:

- Never mix a user's data across train and test.
- Use `GroupKFold` or `GroupShuffleSplit`.

**Common failure**: doing a random split on user-level records with duplicated users → test set contains users your model has seen → inflated performance.

---

### 2.2 K-Fold Cross-Validation

**K-fold CV** partitions the training set into $k$ folds. For $k$ iterations, use $k-1$ folds to train and 1 to validate. Average the $k$ validation scores.

**Advantages over a single validation set:**
- Uses every point for validation exactly once.
- More stable estimate — lower variance than a single split.
- Reveals variance across folds — if fold scores range 0.60 to 0.85, your model is unstable.

**Choosing $k$:**
- $k = 5$ or $k = 10$ are standard.
- $k = n$ = **Leave-One-Out CV (LOOCV)** — extreme; low bias but very high variance and $n$× compute.
- Larger $k$ = less bias, higher variance, more compute.

**StratifiedKFold** for classification (stratify by class):

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

**Report both mean and standard deviation.** A model with mean 0.82 and std 0.01 is more trustworthy than one with mean 0.83 and std 0.05.

**Repeated k-fold** (`RepeatedKFold`) does k-fold multiple times with different shuffles for even more stable estimates.

**Time-Series CV.** Special variant where each fold trains on all data up to a cutoff and validates on the next window:

```
Fold 1:  [------ train ------][val]
Fold 2:  [-------- train --------][val]
Fold 3:  [---------- train ----------][val]
...
```

scikit-learn's `TimeSeriesSplit` does this automatically. **Never** use random KFold on time-series — the model would train on future data and predict the past.

**GroupKFold** ensures no group appears in both training and validation of the same fold. Essential for repeated-measures data.

---

### 2.3 Nested Cross-Validation

The problem: if you use CV to tune hyperparameters *and* to estimate performance, the tuned model has (implicitly) seen the CV score during selection. The reported CV score is optimistically biased.

**Nested CV** solves this with two loops:

- **Outer loop** — estimates performance. Split data into outer folds.
- **Inner loop** — within each outer training fold, run CV to tune hyperparameters. Use the tuned model to predict on the outer validation fold.

```
Outer fold 1: train_outer_1 [CV inside → best params → refit → test on val_outer_1]
Outer fold 2: train_outer_2 [CV inside → best params → refit → test on val_outer_2]
...
```

**When to use nested CV:**
- **Small datasets** where every point matters.
- **Reporting** an unbiased estimate to a paper reviewer or a strict audit.

**When to skip:**
- You have a separate untouched test set — the test set gives you the honest estimate.
- Big data: nested CV is $O(k_{outer} \cdot k_{inner})$ folds — 25 CV runs for 5×5 nested — often not worth it if you have a proper test set.

The rule of thumb: **use a dedicated test set when you can; use nested CV when you can't**.

---

### 2.4 Learning Curves & Validation Curves

Once splits are set, diagnostic plots reveal *why* a model is failing.

**Learning curve** — plot train and validation score vs training-set size:

```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring="accuracy"
)
```

**Interpretation:**

- **Both curves converge low** → high bias (underfitting). Model can't capture the pattern. More data won't help; use a more expressive model.
- **Train high, val low, gap wide** → high variance (overfitting). More data or regularization would help.
- **Both high, small gap** → well-fit model. Nice.
- **Val curve still rising** at max training-set size → you'd benefit from more data.

**Validation curve** — plot train and validation score vs a hyperparameter:

```python
from sklearn.model_selection import validation_curve
train_scores, val_scores = validation_curve(
    estimator, X, y, param_name="max_depth", param_range=[1, 3, 5, 10, 20], cv=5
)
```

**Interpretation:**

- **Left of curve** (weak model / heavy regularization) — high bias.
- **Right of curve** (complex model / no regularization) — high variance.
- **Peak of validation** — sweet spot.

Together, learning curves + validation curves tell you the bias-variance profile of your model on this problem.

---

### 2.5 Bias–Variance and the Bigger Picture

The **bias-variance decomposition** of expected prediction error for a model $\hat f$ at a point $\mathbf{x}$:

$$E[(y - \hat f(\mathbf{x}))^2] = \underbrace{(E[\hat f(\mathbf{x})] - f(\mathbf{x}))^2}_{\text{bias}^2} + \underbrace{\text{Var}(\hat f(\mathbf{x}))}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

Where:
- **$f(\mathbf{x})$** = the true (unknowable) function.
- **$\hat f$** = your model, treated as a random variable over different training sets.
- **$\sigma^2$** = noise inherent to the data-generating process.

You can drive bias and variance to zero *individually* only in extreme (impractical) cases. Real modeling is trading them off:

| Regime | Bias | Variance | Symptom |
|--------|------|----------|---------|
| Underfit (weak model) | high | low | poor train + val |
| Sweet spot | low | low | good train + val |
| Overfit (strong model) | low | high | great train, poor val |

**Regularization moves you left** on the curve (higher bias, lower variance).
**More training data moves the whole curve down**, mostly by reducing variance.
**More features / more expressive model moves you right** (lower bias, higher variance).

**Double descent** — modern deep networks show that as model capacity grows well beyond the training-set size, error can *decrease* again after peaking. Not typical for scikit-learn models; specific to overparameterized deep nets and near-interpolation.

**Rule of thumb for tabular ML:** start with a moderate-capacity model + strong regularization, and let the data pull you off that starting point.

---

## 3. Mental Models & Analogies

### 3.1 The "Practice, Simulation, Real Exam" Model

Studying for a professional exam:

- **Training** — your practice problems. You do them, get feedback, adjust.
- **Validation** — mock exams. You take them under conditions similar to the real thing, then adjust your study strategy. You do many mocks.
- **Test** — the real exam. You take it once. If you did mock exams that resembled the real exam and adjusted your studying accordingly, you should score similarly. If you didn't, you'll be surprised.

**Cross-validation** is doing many mocks and averaging your scores — you get a more stable sense of your prep.

**Nested CV** is *two levels* of mocks: an inner series of mocks to decide your study plan, then an outer series to estimate how well your final plan will do on unseen material. Twice the work, but the "unseen material" estimate isn't contaminated by having chosen the study plan on the same mocks.

**Test-set leakage** is peeking at the answer key before the real exam. You'll ace the mock exam version of the real exam, then bomb the real exam.

### 3.2 The "Weather Forecast" Model (Time Series)

You are a meteorologist trying to predict tomorrow's temperature. You have historical data. Your training scheme must reflect how you'll actually operate:

- **You cannot use tomorrow's data to predict today's temperature** — even in your model evaluation. That's the definition of look-ahead bias.
- Your validation must simulate: at time $t$, using only data up to $t$, predict $t + 1$.
- **`TimeSeriesSplit`** captures this: expanding training window, always predicting the next chunk.

If you shuffle time-series data randomly and do k-fold, your model will "know" some future values while predicting past ones. Its evaluation will look great — and it will fail on the day you deploy it.

Same logic applies to **user-level** or **session-level** data (`GroupKFold`) or **hierarchical** data (predicting patient outcomes by drug, where patients cluster by clinic).


![IMG-RF-01](/2%20—%20Machine%20Learning/images/IMG-CV-01.jpg)
> **Caption:** k-fold rotates the validation slice; TimeSeriesSplit only lets it move forward in time.
> **Placement:** Section 2.2.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "I'll Just Look at Test-Set Performance to Guide Modeling"

Once you look at test-set numbers to decide what to do next — try a new hyperparameter, drop a feature, switch models — you've contaminated the test set. Every subsequent choice is subtly optimized for that particular test set. You've turned it into a second validation set. Do modeling on train + validation (or CV); touch the test set **once**, at the end, to report.

### 4.2 "Random Splits Are Always Fine"

Only when data is truly i.i.d. Otherwise:
- **Time-series** → random split leaks the future.
- **Grouped data** (multiple rows per patient/user/session) → random split leaks group membership.
- **Sequential experiments** (an A/B test rollout) → random split ignores rollout dynamics.
- **Imbalanced classes** → stratify or you may end up with a fold that has no positives.

Ask: **what does deployment look like?** Your split should imitate that.

### 4.3 "Fitting Preprocessing on the Whole Dataset Then Splitting Doesn't Really Leak"

It does. Even small leaks show up as a systematic gap between CV and test performance. `StandardScaler` fit on train + test uses test-set values to compute mean and std; encoders fit on the union see rare test-set categories; feature-selection performed before splitting picks features that happen to work on the test set. **Rule:** every preprocessing step goes inside a `Pipeline` that is fit only on training data. Then CV re-fits per fold and the test set stays untouched.

---

## 5. Self-Assessment Bank (Splits & CV)

### Questions

**Q1 (Short answer).** Why do we need three sets (train / validation / test) instead of just two?

**Q2 (Multiple choice).** For imbalanced binary classification with 5% positives, the right CV strategy is:
A. `KFold` with `n_splits=5`.
B. `StratifiedKFold` with `n_splits=5`.
C. Random split with no stratification.
D. `LeaveOneOut`.

**Q3 (Short answer).** Explain what `TimeSeriesSplit` does and why it's necessary for time-series data.

**Q4 (Multiple choice).** In nested cross-validation:
A. There is only one loop, run more efficiently.
B. There are two nested loops — an outer for performance estimation, an inner for hyperparameter tuning.
C. It's the same as regular CV but with more folds.
D. It's used only for deep learning.

**Q5 (Short answer).** Describe the shape of a learning curve when a model is underfitting.

**Q6 (Multiple choice).** You have a dataset of medical records with multiple visits per patient. Which split respects the data structure?
A. `KFold`
B. `StratifiedKFold`
C. `GroupKFold` with `groups=patient_id`
D. `TimeSeriesSplit`

**Q7 (Short answer).** Give the bias-variance decomposition of expected squared error and label each term.

**Q8 (Multiple choice).** Reporting "accuracy 0.82" from cross-validation is:
A. Fine — it's the mean, that's what matters.
B. Incomplete — you should also report variance across folds.
C. Wrong — always report AUC.
D. Wrong — always report a single test-set number.

**Q9 (Short answer).** Why must feature scaling, encoding, and other transformations happen **inside** the CV loop rather than before?

**Q10 (Multiple choice).** After tuning hyperparameters on the validation set and picking a model, the correct final step before deployment is:
A. Report validation performance as the production estimate.
B. Refit on train+val, evaluate once on the untouched test set.
C. Do another round of tuning on the test set.
D. Use the model as-is without further evaluation.

---

### Answer Key & Detailed Explanations

**A1.** If you tune hyperparameters, select features, or compare models using the test set, you've *implicitly optimized for that specific test set*. Test performance is no longer an unbiased estimate of generalization. The **validation** set exists as the "expendable" set for modeling decisions; the **test** set stays clean for one final measurement.

**A2. B.** With 5% positives, a random split can produce folds with 0 positives, breaking evaluation. Stratification ensures each fold has ~5% positives.

**A3.** `TimeSeriesSplit` produces folds where the training set is always chronologically earlier than the validation set — the training window grows across folds, and the validation window rolls forward. It's necessary because time-series data has temporal structure; a random split would let the model train on future data and predict past data, which doesn't reflect deployment reality.

**A4. B.** Nested CV has an outer CV loop for performance estimation and an inner CV loop within each outer training fold for hyperparameter selection. The inner loop's chosen hyperparameters are then used to refit on the outer training fold and predict on the outer validation fold. Result: an unbiased performance estimate that accounts for the tuning process.

**A5.** Both training and validation curves are **low** (poor scores). They converge to each other — small gap — but at an unsatisfying level. The model isn't complex enough to capture the pattern. Solution: use a more expressive model or richer features; more data won't help.

**A6. C.** `GroupKFold` ensures a given patient's rows never appear in both training and validation of the same fold. Random splits (`KFold`, `StratifiedKFold`) would leak patient identity across sets, inflating validation performance.

**A7.** $E[(y - \hat f(\mathbf{x}))^2] = \underbrace{(E[\hat f(\mathbf{x})] - f(\mathbf{x}))^2}_{\text{bias}^2} + \underbrace{\text{Var}(\hat f(\mathbf{x}))}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$. Bias measures systematic under-fitting; variance measures instability across different training sets; irreducible noise is the noise inherent to $y$ that no model can predict.

**A8. B.** Cross-validation gives you $k$ numbers, not one. Reporting only the mean hides how stable the estimate is. A mean of 0.82 with std 0.01 tells a very different story than 0.82 with std 0.08. Always report both — e.g., "0.82 ± 0.01."

**A9.** Fitting transformations outside the CV loop means the transformer sees the entire dataset — including the held-out fold — during fit. Feature statistics (means, category counts, target means for target encoding) leak into training. The CV score is optimistically biased. Wrapping in a `Pipeline` and calling `cross_val_score` refits per fold, keeping the held-out data unseen at every step.

**A10. B.** Refit on train+val (using the hyperparameters you selected) so the final production model uses as much data as possible; then run the test set through it exactly once to get the deployment estimate. This gives you (a) the strongest possible model to deploy and (b) an honest generalization estimate.

---

## 6. Practice Prompts

1. **Leaky vs proper.** Simulate leakage by fitting `StandardScaler` on the whole dataset before splitting. Compare "CV score" vs test score. Note the gap. Wrap in `Pipeline`; watch it close.
2. **StratifiedKFold vs KFold.** On a dataset with 5% positives, run both. Confirm regular KFold sometimes produces folds with 0 positives.
3. **TimeSeriesSplit.** On a time-series dataset (retail sales, temperature), fit an XGBoost with random KFold and with TimeSeriesSplit. Notice how the random-split CV score dramatically over-predicts real deployment performance.
4. **Learning curve.** Fit a model at 10 fractions of the training data. Plot the learning curve. Diagnose bias vs variance.
5. **Nested CV.** On a small dataset (~200 rows), compare nested CV to non-nested (single CV for both tuning and estimation). Confirm non-nested is optimistically biased.

---

## 7. References

- ISL chapter 5.
- ESL chapter 7.
- Cawley & Talbot, ["On Over-fitting in Model Selection and Subsequent Selection Bias"](https://www.jmlr.org/papers/v11/cawley10a.html) (2010).
- Belkin et al., ["Reconciling modern machine-learning practice and the classical bias-variance trade-off"](https://arxiv.org/abs/1812.11118) (2019) — for double descent context.
- scikit-learn docs: [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
