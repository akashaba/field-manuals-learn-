# Hyperparameter Tuning — Master Study Guide

> **Track:** Machine Learning · **Module:** 12
> **Prerequisites:** Modules 10 (Splits & CV) and 11 (Metrics).
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every ML model has knobs — regularization strength, tree depth, learning rate, number of neighbors — that aren't learned from data. These **hyperparameters** dramatically affect performance, sometimes turning a mediocre model into a state-of-the-art one. Finding good ones is a search problem, and doing it well involves:

1. Choosing the **right search strategy** (grid, random, Bayesian).
2. Choosing the **right validation strategy** so results aren't leaky.
3. Choosing the **right metric** and letting it drive the search.
4. Not tuning yourself into the ground — knowing when to **stop**.

Getting this right can be the difference between a 0.75 and a 0.85 AUC. Getting it wrong (tuning on the test set, over-searching, chasing noise) makes your headline number a fiction.

**Fundamental principles you must own:**

1. **Random search usually beats grid search** for the same compute budget.
2. **Bayesian optimization beats random search** for expensive-to-train models.
3. **Tuning must happen inside cross-validation** — otherwise you optimize for a specific validation split and overfit to it.
4. **Nested CV** gives unbiased estimates when the tuning is part of the process.
5. **Early stopping** is a form of hyperparameter tuning: it selects `n_estimators` from the data.
6. **Overtuning** exists — searching too aggressively finds hyperparameters that fit the validation set's noise, not the underlying pattern.

If you retain nothing else: **start with a strong default, tune with random search, cross-validate honestly, and stop when marginal gains flatten.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Grid Search and Random Search

**Grid search.** Specify a discrete grid of hyperparameter values; evaluate every combination.

```python
from sklearn.model_selection import GridSearchCV
param_grid = {"n_estimators": [100, 200, 500],
              "max_depth":    [3, 5, 7, None],
              "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="roc_auc")
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_
```

- Pros: exhaustive within the grid; deterministic; easy to reason about.
- Cons: **curse of dimensionality** — 3 hyperparams × 5 values each = 125 combos × 5-fold CV = 625 fits. Adding a 4th param blows this up.

**Random search.** Sample hyperparameter combinations from distributions.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
param_dist = {"n_estimators":     randint(50, 1000),
              "max_depth":        randint(3, 20),
              "min_samples_leaf": randint(1, 20),
              "max_features":     loguniform(0.1, 1.0)}
rs = RandomizedSearchCV(RandomForestClassifier(), param_dist,
                        n_iter=50, cv=5, scoring="roc_auc", random_state=42)
rs.fit(X_train, y_train)
```

- Bergstra & Bengio (2012) showed that for the same budget, random search usually **finds better hyperparameters than grid search**, because it explores each dimension independently rather than wasting effort on grid intersections.
- Especially better when a few hyperparameters are much more important than others (which is usually the case).

**Rule of thumb.** For up to 3 discrete hyperparameters, grid is fine. For anything more, use random.

**Log-scale sampling** for parameters that vary over orders of magnitude:

- Regularization strength `C`, `alpha`, `lambda` → `loguniform(1e-4, 1e2)`.
- Learning rate → `loguniform(1e-3, 1)`.
- Number of estimators → `randint(50, 5000)`.
- Batch size in DL → `[16, 32, 64, 128, 256]` (log2 scale).

Uniform sampling of `C ∈ [0.001, 100]` puts ~99% of samples above 1 — you barely explore the small-`C` regime. Log-uniform balances the exploration.

---

### 2.2 Bayesian Optimization

**Bayesian optimization** models the objective (validation score as a function of hyperparameters) with a **surrogate** — usually a Gaussian process or a tree-based ensemble. It picks the next hyperparameter to try by balancing:

- **Exploitation** — try near current best.
- **Exploration** — try in uncertain regions.

An **acquisition function** (Expected Improvement, Upper Confidence Bound, TPE-based) formalizes this trade-off. Each new evaluation updates the surrogate.

**Libraries:**

- **Optuna** — the modern favorite. Simple API, prunes bad trials early, supports distributed.
- **Hyperopt** — older; TPE algorithm.
- **scikit-optimize (skopt)** — Gaussian-process-based; drop-in `BayesSearchCV`.
- **Ax / BoTorch** — Meta's production-grade Bayesian optimization.

**Optuna example:**

```python
import optuna

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth":     trial.suggest_int("max_depth", 3, 12),
        "reg_lambda":    trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
    }
    model = xgb.XGBClassifier(**params, n_estimators=1000, early_stopping_rounds=50)
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=3600)
print(study.best_params, study.best_value)
```

**When Bayesian pays off:**
- Each training run is expensive (minutes+).
- You have >3 hyperparameters.
- You have 50+ trials of budget.

**When it's overkill:**
- Very cheap models (< a few seconds per fit) — random search is simpler and often just as good.
- Continuous vs discrete mismatch — categorical hyperparameters weaken the GP surrogate.

**Pruning** (early termination of bad trials) is a huge win: Optuna and Ray Tune both support median stopping, Hyperband, ASHA. If a trial's mid-run performance is clearly worse than the running median, kill it and try a new hyperparameter.

---

### 2.3 Early Stopping as Hyperparameter Selection

For iterative models (gradient boosting, neural networks), **early stopping** treats the "number of iterations" as a hyperparameter selected from the data:

1. Split off a validation subset from the training data.
2. Train for many iterations, evaluating on validation each round.
3. Stop when validation score stops improving for a "patience" number of rounds.

Return the best-observed round's model.

```python
model = XGBClassifier(n_estimators=5000, learning_rate=0.05,
                     early_stopping_rounds=50, eval_metric="auc")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
# model.best_iteration_ is the chosen n_estimators
```

**Advantages:**
- No search needed for `n_estimators`.
- Practically free — you'd have trained the model anyway.
- Empirically, one of the most effective regularizers.

**Caveats:**
- Requires a **separate validation split** or the internal split of scikit-learn CV. Not the same as your final test set.
- Interacts with learning rate: lower learning rate + more rounds + early stopping ≥ high learning rate + few rounds. The former usually wins.
- The chosen `n_estimators` is specific to the validation set — different splits will pick different numbers. In CV, average or refit on full data with the median best iteration.

---

### 2.4 What to Tune (and What Not To)

Not all hyperparameters are equal. For each model type, a small subset dominates.

**Logistic Regression:**
- `C` (inverse regularization strength) — most important.
- `penalty` (`l1`, `l2`, `elasticnet`) — pick based on feature selection needs.
- `class_weight` — for imbalance.
- Solvers rarely matter unless you have huge or specific data.

**Random Forest:**
- `n_estimators` — set high (500+), diminishing returns after that.
- `max_features` — most important for controlling correlation between trees.
- `max_depth`, `min_samples_leaf` — for overfitting control.

**XGBoost / LightGBM:**
- `learning_rate` + `n_estimators` (with early stopping) — most important.
- `max_depth` / `num_leaves` — tree complexity.
- `min_child_weight` / `min_data_in_leaf` — leaf-size regularization.
- `subsample`, `colsample_bytree` — row/column randomness.
- `reg_lambda`, `reg_alpha` — L2/L1 on leaf weights.

**SVM:**
- `C` — margin/error trade-off.
- `gamma` (RBF) — kernel width. Grid over log-space.
- Kernel choice — often decided by prior knowledge or exploration.

**KNN:**
- `n_neighbors` — the main knob.
- `weights` (uniform/distance).
- `metric` — cosine vs Euclidean depending on data.

**Neural nets (out of scope for this module, but):**
- Learning rate.
- Batch size.
- Architecture size (layers, hidden units).
- Regularization (dropout, weight decay).
- Optimizer parameters.

**Don't tune:**
- Random seeds. If you're chasing which seed gives the best score, you're over-searching noise.
- Preprocessing choices in the same loop as the model (they should be in the pipeline, but stability across CV is enough — no need to formally tune).

---

### 2.5 Avoiding Overtuning & Reporting Honestly

**Overtuning symptoms:**

- Best CV score keeps improving as trials grow, but held-out test score doesn't move (or moves down).
- The gap between best and 2nd/3rd best CV scores is tiny — you're chasing noise.
- Different random seeds pick very different "best" hyperparameters.

**Guards against overtuning:**

- **Fix random seeds** in your search wherever possible so results are reproducible.
- **Stop when marginal gains flatten** — set a patience on tuning trials.
- **Use nested CV** for reporting when the dataset is small; the outer loop gives an unbiased estimate.
- **Keep the test set truly untouched** — the ultimate check.
- **Compare against a strong baseline** with default hyperparameters. If tuning only bought +0.005 AUC, was it worth it?

**Time budget as a hyperparameter.** Give your search a fixed wall-clock time or trial count, and don't extend it because "one more trial might help." That instinct is how you overfit to the validation set.

**Reporting a tuned model.**

- **Report CV score with variance** across folds ("0.83 ± 0.02 5-fold AUC").
- **Report the tuning strategy** (Optuna, 100 trials, over these ranges).
- **Report the test-set score** (single number from an untouched test set).
- **Never report the best of many test-set scores** — that's tuning on test.

---

## 3. Mental Models & Analogies

### 3.1 The "Mixing Board" Model

An audio engineer sits at a mixing board with dozens of sliders. The song is a hyperparameter set; the sound is the model. To get the best mix, they:

- Start with reasonable defaults (a strong baseline).
- Isolate one or two sliders at a time (learning rate, max depth) that matter most.
- Sweep those over a range and listen (evaluate).
- Once those are dialed in, adjust the less-important ones.
- Know when to stop — chasing 0.5 dB over an hour is diminishing return.

**Random search** is like sending a slightly-drunk assistant to try 50 random slider combos. Sometimes better than sweeping systematically, because they don't get stuck spending 20 minutes on the master volume when the reverb slider is what mattered.

**Bayesian optimization** is like the assistant keeping notes: after every combo, they update their belief about which sliders matter and where the sweet spot is. Their next trial is *informed* by what they've heard so far.

**Overtuning** is the engineer who spends 12 hours on one song. At some point, they're not making the song better — they're just responding to their own fatigue and drift. Same happens to models: past a point, "improvements" are chasing noise in the validation split.

### 3.2 The "Landscape Explorer" Model

Imagine hyperparameter space as a mountainous landscape. Each point is a hyperparameter set; elevation is (negated) validation loss. You want the highest peak.

- **Grid search** is walking every intersection of a coarse street grid. Thorough near the grid, blind between grid lines.
- **Random search** is throwing darts. Doesn't cluster near street corners, so it often finds better peaks with the same number of stops.
- **Bayesian optimization** is a smart hiker: after each stop, they update their mental map, weigh where they've been high vs where they're uncertain, and pick the next stop accordingly.

**Local minima** are traps — an area that looks like a peak but sits below the real one. Bayesian methods with proper acquisition functions (that value exploration) escape these; greedy hill-climbing methods don't.

**Random restarts** — repeating the search from different starting points — is the poor person's escape from local minima.

![IMG-HP-01](/2%20—%20Machine%20Learning/images/IMG-HP-01.jpg)

> **Caption:** Three tuning strategies visualized: grid samples uniformly, random samples scattered, Bayesian concentrates near promising regions.
> **Placement:** Mental Models section.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "More Trials Is Always Better"

Past a point, additional trials find hyperparameters that fit the validation split's *noise* rather than signal. Symptoms: CV score keeps rising while a held-out test score doesn't. Fix: set a trial budget in advance, monitor a held-out test score periodically (without letting it drive decisions), and stop when marginal gains flatten.

### 4.2 "Grid Search Is Rigorous; Random Is Sloppy"

The opposite, for high-dimensional searches. Grid search wastes evaluations on non-informative dimensions. Random search explores each dimension independently, and with a modest budget usually finds better peaks. The Bergstra & Bengio (2012) paper is worth reading.

### 4.3 "The Tuned Model's Best CV Score Is a Valid Estimate of Production Performance"

No. The best CV score after tuning is optimistically biased — you searched *for* the best score, so you found it partly by luck. The unbiased estimate comes from either:
- **A separate untouched test set** used *once*.
- **Nested cross-validation** where the tuning happens inside the inner loop.

Reporting the best CV score of the search as "expected production performance" is a common but serious mistake.

---

## 5. Self-Assessment Bank (Hyperparameter Tuning)

### Questions

**Q1 (Short answer).** Explain, in one paragraph, why random search often outperforms grid search on the same compute budget.

**Q2 (Multiple choice).** For a hyperparameter that ranges over 4 orders of magnitude (e.g., `C` in `[1e-4, 1e2]`), the right sampling distribution is:
A. Uniform.
B. Log-uniform.
C. Normal.
D. Categorical.

**Q3 (Short answer).** Describe how Bayesian optimization decides which hyperparameters to try next.

**Q4 (Multiple choice).** Early stopping in gradient boosting effectively:
A. Reduces the learning rate.
B. Selects `n_estimators` from a validation-derived signal, avoiding a separate search over it.
C. Replaces cross-validation entirely.
D. Removes overfitting completely.

**Q5 (Short answer).** What is "nested cross-validation" and when should you use it?

**Q6 (Multiple choice).** You tuned a model over 500 trials. The best CV AUC is 0.85, and the top 20 configurations are all within 0.001 AUC of each other. What's the risk?
A. You've found the true optimum.
B. You may be chasing noise; the best-configured model may not generalize any better than the 20th.
C. You should run 500 more trials.
D. You should widen the search space.

**Q7 (Short answer).** Which hyperparameters of XGBoost tend to matter most for practical performance? Name three.

**Q8 (Multiple choice).** For an SVM with an RBF kernel, the two hyperparameters most critical to tune are:
A. `kernel` and `degree`.
B. `C` and `gamma`.
C. `max_iter` and `tol`.
D. `probability` and `shrinking`.

**Q9 (Short answer).** Why must hyperparameter tuning happen **inside** a CV loop rather than after a single train/val split?

**Q10 (Multiple choice).** To honestly report the performance of a tuned model, you should:
A. Report the best CV score from the search.
B. Report the score of the best configuration on the untouched test set (single evaluation).
C. Repeat the search on the test set for verification.
D. Report the average of the top-10 CV scores.

---

### Answer Key & Detailed Explanations

**A1.** In high-dimensional hyperparameter spaces, only a few dimensions typically dominate performance. Grid search evaluates on a fixed lattice — it spends many "trials" varying unimportant dimensions while exploring only a coarse grid on important ones. Random search decouples dimensions: each trial samples every hyperparameter independently, so with $N$ trials you get $N$ distinct values along each dimension. For the same budget, you cover the important dimensions more finely. Bergstra & Bengio (2012) formalized this.

**A2. B.** Log-uniform (`scipy.stats.loguniform` or Optuna's `log=True`) samples uniformly in log space. Uniform sampling of `[1e-4, 1e2]` puts 99% of samples above 1 and never explores the interesting small-`C` regime.

**A3.** Bayesian optimization fits a **surrogate model** (Gaussian process or tree-based ensemble) that predicts objective value as a function of hyperparameters, plus an uncertainty estimate. An **acquisition function** (Expected Improvement, UCB, TPE) combines predicted value with uncertainty — balancing exploitation (try near current best) with exploration (try in high-uncertainty regions). The next trial is the argmax of the acquisition function. After the trial, the surrogate is updated.

**A4. B.** Early stopping picks `n_estimators` at the point where validation performance plateaus. It removes the need to sweep `n_estimators` in your grid — the data tells you when to stop.

**A5.** Nested CV has two loops: an **outer** CV for unbiased performance estimation and an **inner** CV within each outer training fold for hyperparameter tuning. The inner loop picks the best hyperparameters using only the outer training data; the outer loop measures performance on truly-held-out outer validation folds. Use it when: (a) you don't have a separate untouched test set, (b) you need to report an unbiased estimate (papers, audits), and (c) the dataset is small enough that CV overhead is acceptable.

**A6. B.** When many configurations perform nearly identically, you may be at the noise floor of your validation set. The "best" is best partly by luck. The 20th configuration is likely just as good. Guarantees: cross-validate variance across folds, sanity-check the top-3 on a held-out test set to see if the ordering holds.

**A7.** Three strong choices: `learning_rate` (with a properly high `n_estimators` and early stopping), `max_depth` (tree complexity), and `min_child_weight` (leaf regularization). Also strong: `subsample`, `colsample_bytree`, and `reg_lambda`. Tune `learning_rate` and tree structure first.

**A8. B.** `C` controls the margin/error trade-off; `gamma` controls the RBF kernel width. Both need to be tuned together, typically over log-spaced grids. Other parameters are usually less impactful.

**A9.** A single train/val split means your best hyperparameters are optimal *for that specific validation set*, which includes its noise. Different splits would pick different hyperparameters. CV averages over multiple splits, giving a more stable and less optimistically biased performance signal — which is what you want when comparing configurations.

**A10. B.** The correct pattern: tune using CV on train+val, pick the best hyperparameters, refit on train+val with those params, then evaluate ONCE on the untouched test set. Reporting the CV best score is optimistically biased. Repeating on test contaminates test. Averaging top-10 CV scores doesn't help — they're all from the same tuning process.

---

## 6. Practice Prompts

1. **Grid vs random.** On XGBoost with 5 hyperparameters, run grid search (small grid) and random search (same total fits). Compare best test-set score.
2. **Optuna sweep.** Use Optuna to tune LightGBM with 100 trials. Plot the study's `plot_optimization_history` and `plot_param_importances`.
3. **Log vs linear.** For `C` in logistic regression, tune via uniform `[0.001, 100]` and via loguniform `[0.001, 100]`. Compare best `C`.
4. **Overtuning demo.** Do 1000 trials of random search on a small dataset with a ~200-row test set. Plot cumulative best CV score vs cumulative best test score. Watch them diverge.
5. **Nested CV.** On a 500-row dataset, compare nested CV score (5-outer, 3-inner) vs regular CV. Confirm nested is closer to a truly held-out test score.

---

## 7. References

- Bergstra & Bengio, ["Random Search for Hyperparameter Optimization"](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) (2012) — the paper.
- Snoek, Larochelle & Adams, ["Practical Bayesian Optimization of Machine Learning Algorithms"](https://arxiv.org/abs/1206.2944) (2012).
- Akiba et al., ["Optuna: A Next-generation Hyperparameter Optimization Framework"](https://arxiv.org/abs/1907.10902) (2019).
- Li et al., ["Hyperband"](https://arxiv.org/abs/1603.06560) (2018) — bandit-style pruning.
- scikit-learn docs: [Tuning hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html); Optuna docs.
