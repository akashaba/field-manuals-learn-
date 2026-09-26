# Feature Engineering — Master Study Guide

> **Track:** Machine Learning · **Module:** 09
> **Prerequisites:** Modules 01–08 for what different models want from features.
> **Time budget:** ~12–15 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** The often-quoted line "**garbage in, garbage out**" is even truer in ML: no model, however sophisticated, can extract signal that isn't in the features. **Feature engineering is the single highest-leverage activity in tabular ML** — a well-crafted feature routinely beats a fancier model on identical raw data.

More importantly, feature engineering is *where domain knowledge lives*. It's how you inject expert intuition ("weekends matter for retail," "a transaction 5000 miles from home is suspicious," "log-transform prices") into a model that would otherwise treat features as anonymous numbers.

The three biggest sins in feature engineering:

1. **Doing it wrong** — bad encodings, unbalanced scaling, useless transformations.
2. **Not doing it** — leaving `datetime` fields as raw strings, categoricals as random integers, no interactions.
3. **Doing it with leakage** — computing feature statistics on the whole dataset, then splitting → your test-set stats have already peeked at the training set.

**Fundamental principles you must own:**

1. **Always fit transformations on training data only, then apply to validation/test.** No exceptions. Wrap in a scikit-learn `Pipeline`.
2. **The right encoding depends on the model.** One-hot for linear/NN; ordinal for trees; target encoding for high-cardinality with care.
3. **The right transformation depends on the feature.** Log for skewed positives; Box-Cox/Yeo-Johnson for signed data; robust scaling for outliers.
4. **Feature interactions matter** — either engineer them explicitly or use a model that captures them (trees, kernels, neural nets).
5. **Missingness is signal** — sometimes more signal than the value would have been.
6. **Time features require particular care** — extraction, lag/rolling features, and *no look-ahead*.

If you retain nothing else: **every transformation is a decision, and every decision should be reproducible from a saved Pipeline.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Handling Missing Data

**MCAR / MAR / MNAR** — a critical distinction:

- **MCAR** (Missing Completely At Random): missingness is independent of everything, observed or unobserved. Rare in practice.
- **MAR** (Missing At Random): missingness depends on other observed variables. E.g., income is missing more often for young respondents.
- **MNAR** (Missing Not At Random): missingness depends on the *unobserved* value itself. E.g., high earners refuse to report income. Hardest to handle.

Under MCAR, simple imputation gives unbiased estimates. Under MAR, more careful imputation (e.g., using other features) works. Under MNAR, no imputation is fully safe — you're guessing about a systematically biased sample.

**Imputation techniques:**

- **Simple:**
  - **Mean/median/mode** — fast; median is more robust to outliers.
  - **Constant** — useful when missingness is meaningful (e.g., "no delivery address given").
- **Model-based:**
  - **KNN imputation** — replace missing with the value from similar rows.
  - **Iterative imputation** (MICE / `IterativeImputer`) — model each missing feature as a function of the others, cycle until convergence.
- **Domain-aware:**
  - Fill with previous value in time series.
  - Fill with 0 when missing means "did not happen."

**Missingness indicators.** Whenever you impute, add a boolean column `feature_is_missing`. If the missingness itself is informative, the indicator lets the model use that.

**Model-specific handling.** Some models (XGBoost, LightGBM, CatBoost, HistGradientBoosting in scikit-learn) handle missing values natively — often better than any imputation. Use these when possible.

**Pipeline safety:** always fit imputer on training only. `SimpleImputer().fit(X_train)`, then `.transform(X_val)`.

---

### 2.2 Encoding Categorical Features

**Which encoding? It depends on:**
- Model type (linear/NN vs trees).
- Cardinality (how many unique values).
- Whether there's an ordinal relationship.

**Common options:**

**One-hot encoding.** Each level becomes a binary column.
- ✅ Interpretable, no false ordering.
- ❌ Explodes dimensionality with high cardinality.
- ❌ Sparse.
- Best for: linear/NN models, low cardinality (< ~20).

**Ordinal (label) encoding.** Assign integers 0, 1, 2, ...
- ✅ Compact.
- ❌ Imposes false order (Chicago = 0, Denver = 1, doesn't mean Chicago < Denver in any meaningful way).
- Best for: trees, or where an ordering is real (education level).

**Frequency (count) encoding.** Replace each level with its count in the training set.
- ✅ Simple, works well with trees.
- ❌ Ties across levels with the same count.

**Target (mean) encoding.** Replace each level with the mean of the target for that level in training data.
- ✅ Compact, captures signal directly.
- ❌ **High leakage risk** — you're peeking at $y$ during encoding.
- Solutions: out-of-fold encoding, smoothing toward the global mean, adding noise. Use `category_encoders` or `sklearn.preprocessing.TargetEncoder` (added recently); wrap in a Pipeline.

**Hashing (feature hashing / hashing trick).** Hash each level to one of $k$ bins.
- ✅ Fixed dimensionality regardless of cardinality.
- ❌ Collisions (different levels hash to same bin).
- Best for: streaming, extremely high-cardinality text/URL features.

**Entity embeddings.** Learn a dense representation of each level as part of a neural network.
- ✅ Very expressive.
- ❌ Requires NN infrastructure and lots of data.

**CatBoost's ordered target encoding** — a specific target-encoding scheme designed to avoid leakage. Best-in-class for tabular data with many categoricals.

**Rule of thumb:**
- Low cardinality (< 20) → one-hot.
- Trees → ordinal encoding is fine.
- High cardinality → target encoding (with care) or hashing.
- Very high cardinality → hashing or embeddings.

---

### 2.3 Numeric Transformations & Scaling

**Scaling** puts features on comparable ranges. Essential for:
- Distance-based models (KNN, K-means, SVM).
- Regularized linear models (Ridge, Lasso).
- Neural networks.

Not essential for:
- Tree-based models (scale-invariant).

**Options:**

- **StandardScaler** — $(x - \mu) / \sigma$. Zero mean, unit variance. Default choice for most cases.
- **MinMaxScaler** — $(x - x_\min) / (x_\max - x_\min)$. Scales to $[0, 1]$. Useful when a bounded range matters (e.g., neural nets with bounded activations).
- **RobustScaler** — $(x - \text{median}) / \text{IQR}$. Uses percentiles instead of mean/std → robust to outliers.
- **MaxAbsScaler** — $x / |x|_\max$. Preserves sparsity; scales to $[-1, 1]$. For sparse matrices.

**Nonlinear transformations** to fix skewness or improve linearity:

- **Log transform** — $\log(x)$ or $\log(1 + x)$. For strictly positive, right-skewed data (prices, incomes, counts).
- **Box-Cox** — parametric family: $y = (x^\lambda - 1) / \lambda$ for $\lambda \neq 0$, $\log x$ for $\lambda = 0$. Requires $x > 0$. Finds optimal $\lambda$ to maximize normality.
- **Yeo-Johnson** — extension of Box-Cox that handles zero and negative values. `PowerTransformer` in scikit-learn.
- **Rank transform / quantile transform** — replace values with their empirical quantiles (uniform or normal). Robust; destroys magnitude information.
- **Reciprocal** — $1 / x$. Useful for fractions and rates.

**Binning / discretization:**
- Convert continuous to categorical (equal-width or equal-frequency bins).
- Can help capture nonlinear thresholds for linear models.
- **Losing information** — trees can already find splits; binning for trees is usually pointless.

**Polynomial and interaction features** — `PolynomialFeatures(degree=2)` in scikit-learn expands to include $x_i^2$ and $x_i x_j$. Great for linear models on nonlinear data. Explodes with high $p$.

---

### 2.4 Time Features & Time-Aware Feature Engineering

Time features carry rich signal — and enormous leakage risk. Fundamentals:

**Extraction from `datetime`:**

- Day of week (0–6).
- Hour of day.
- Month, quarter, year.
- Is-weekend, is-holiday (join a holiday calendar).
- Day of year (for seasonality).
- Cyclic encoding for periodicity:

  $$\text{sin\_hour} = \sin(2\pi \cdot h/24), \quad \text{cos\_hour} = \cos(2\pi \cdot h/24)$$

  This preserves the "23:00 is close to 00:00" property that a raw integer 23 vs 0 doesn't.

**Time-since features:**

- Time since last event (session start, purchase, login).
- Time until next expected event.
- Age of account.

**Lag features:**

- **Lags** — value at previous time step(s): $x_{t-1}, x_{t-7}$, etc. Standard for time-series prediction.
- **Rolling statistics** — moving average / std / max over a window: 7-day rolling mean of clicks.
- **Expanding statistics** — cumulative statistics from beginning to $t$.
- **Exponential moving averages** — smoother than rolling means, recent-heavy.

**Critical rule: NO LOOK-AHEAD.**

Every feature at time $t$ must be **computable from data available strictly before $t$**. Violations:

- Using future timestamps in rolling averages (e.g., a 7-day window centered on $t$ instead of ending at $t$).
- Aggregating over the whole dataset to fit an encoder, then applying to individual rows.
- Filling missing values with a global mean that includes test-period data.
- Using next-day price to predict today's price.

If a look-ahead sneaks in, your validation metrics will look great and your production predictions will fail.

**Time-aware splitting** — never split time-series data randomly. Use `TimeSeriesSplit`, or a hand-crafted "train on Jan-Sept, validate on Oct, test on Nov-Dec" schedule.

---

### 2.5 Feature Interactions, Feature Selection, and the Pipeline

**Interactions.** Some models capture them natively:
- Trees / RF / GBM — through their branching structure.
- Kernel SVMs — through the kernel.
- Deep neural networks — through hidden layers.

Linear models don't. If a linear model is your target, engineer interactions explicitly:
- `PolynomialFeatures(interaction_only=True)` for pairwise products.
- Hand-crafted domain interactions (e.g., `price × in_promotion`).
- Ratios and differences (`revenue_per_user`, `days_since_signup`).

**Feature selection:**

- **Filter methods** — score features independently by mutual information, chi-square, F-test; keep top-$k$. Fast; may miss interactions.
- **Wrapper methods** — RFE (Recursive Feature Elimination), forward/backward selection. Slow but principled.
- **Embedded methods** — L1 penalty (Lasso), tree feature importance. Built into model fitting.
- **Permutation importance** on a held-out set — the most trustworthy.

**Bias to avoid.** Feature selection **must** happen inside your CV loop, not before. Otherwise you're peeking at test data.

**The `Pipeline` and `ColumnTransformer` discipline:**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

numeric_features = ["age", "salary"]
categorical_features = ["country", "job"]

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(C=1.0)),
])

model.fit(X_train, y_train)
model.predict(X_test)
```

This is the **leakage-proof** way. Every transformation is fit only on the training fold; `.predict` on new data applies fitted transformations without re-fitting.

**Model card / feature dictionary.** Document every feature: name, source, type, transformation, missingness handling, range, distribution. This is the paper trail that lets you debug in production 6 months from now.

---

## 3. Mental Models & Analogies

### 3.1 The "Chef Prep Station" Model

A chef doesn't hand raw ingredients to the customer. They wash, chop, portion, season, and arrange — all *before* the plate hits the table. Feature engineering is that prep station. The "model" is the customer; the "raw data" is what came off the truck.

- **Scaling** = making sure the ingredients are cut into comparable sizes.
- **Encoding** = translating exotic ingredient names into something the customer can consume.
- **Transformation** = cooking (log, Box-Cox) to unlock flavor a raw version wouldn't have.
- **Interactions** = combining ingredients into dishes (`sauce = butter × flour × milk`) rather than dropping each ingredient alone on the plate.
- **Missing values** = deciding whether a missing ingredient means "the dish is different" or "the ingredient is optional."

And critically — **the prep is done in the kitchen**, not at the table. If you prep with knowledge of what the customer will *later* say, you're cheating. That's what leakage is: prep done with information from the future.

### 3.2 The "Museum Curation" Model

You have a warehouse of ancient artifacts. Feature engineering is **curation**: deciding what to display, how to label it, in what order, next to what other artifacts, and what context to provide.

- **Bad curation**: dump everything unsorted into one room.
- **Good curation**: group by era, cross-reference with maps, provide translations, and highlight the most illuminating pieces.

The model is a museum visitor. It doesn't have your domain knowledge — you have to encode that knowledge into the layout. And *good curation is often more impactful than adding more artifacts* — more features rarely helps a bad set; better features often do.

![IMG-RF-01](/2%20—%20Machine%20Learning/images/IMG-FE-01.jpg)
> **Caption:** Feature engineering is a pipeline; every step is fit on training data only and applied consistently to unseen data.
> **Placement:** Section 2.5.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Fitting Transformers on the Whole Dataset Then Splitting Is Fine"

No — this is the classic **data leakage**. When you fit `StandardScaler` on all rows, the mean and std incorporate test-set values; your training set now knows something about the test set. Test performance is inflated; production performance disappoints. **Always: split first, then fit only on train.** Use `Pipeline` to make this automatic.

### 4.2 "Target Encoding Is Just Replacing Categories with Their Mean Target"

Naively, yes — and that's dangerously leaky. Doing this on the training set means the training row for a rare category = its own target (perfect fit). Fixes:
- **Out-of-fold** encoding: for each row, compute its category's target mean using only rows in *other* CV folds.
- **Smoothing** toward the global mean: $\bar y_c^\text{smooth} = \frac{n_c \bar y_c + \alpha \bar y}{n_c + \alpha}$.
- **Adding noise** to the encoded values.
- Use `sklearn.preprocessing.TargetEncoder` or `category_encoders`, which handle these safely.

### 4.3 "One-Hot Encoding Trees Is Best Practice"

Trees can natively handle ordinal-encoded categoricals. One-hot encoding a 100-level categorical creates 100 mostly-zero binary features; each tree's split can only inspect one at a time, so trees have to build 100 shallow splits to reconstruct what a single ordinal split gives. Worse for high-cardinality features. Use ordinal encoding, or ideally native categorical support (LightGBM, CatBoost, HistGradientBoosting).

---

## 5. Self-Assessment Bank (Feature Engineering)

### Questions

**Q1 (Short answer).** Why must transformers (scalers, encoders, imputers) be fit only on the training set?

**Q2 (Multiple choice).** For a linear regression model with a categorical feature that has 4 levels (no ordinal relationship), the best encoding is:
A. Ordinal (0, 1, 2, 3).
B. One-hot with drop-first for identifiability.
C. Target encoding.
D. Feature hashing.

**Q3 (Short answer).** Explain what "MAR" means, and how it differs from MCAR and MNAR.

**Q4 (Multiple choice).** Which transformation is appropriate for a right-skewed feature that contains only positive values?
A. StandardScaler.
B. MinMaxScaler.
C. Log transform.
D. Ordinal encoding.

**Q5 (Short answer).** Give the formula for the cyclic sine/cosine encoding of the hour of day, and explain the property it preserves that a raw integer doesn't.

**Q6 (Multiple choice).** In time-series feature engineering, using a rolling window that is centered on time $t$ (i.e., extends both before and after) is:
A. Best practice; captures more information.
B. Data leakage — uses future information.
C. Only allowed for classification, not regression.
D. Faster to compute.

**Q7 (Short answer).** Why is naive target encoding on the training set dangerous, and what techniques mitigate the risk?

**Q8 (Multiple choice).** For a tree-based model, feature scaling:
A. Is essential for convergence.
B. Improves accuracy significantly.
C. Is unnecessary — trees are scale-invariant.
D. Is required by scikit-learn.

**Q9 (Short answer).** When would you prefer `RobustScaler` over `StandardScaler`?

**Q10 (Multiple choice).** The most reliable way to avoid data leakage during preprocessing is to:
A. Split then transform manually every time.
B. Wrap preprocessing and model in a `Pipeline` and use CV.
C. Trust that scikit-learn handles it.
D. Only apply transformations after training.

---

### Answer Key & Detailed Explanations

**A1.** Transformers compute statistics (means, quantiles, category counts, target means) from the data they're fit on. If those statistics include validation/test data, information about the test set leaks into training — your test error is optimistically biased, and production performance won't match. Fitting only on training keeps the test set "unseen."

**A2. B.** One-hot encoding gives each level its own binary column, so the linear model can assign a coefficient to each level. Dropping the first (or last) avoids the "dummy variable trap" (perfect collinearity with the intercept). Ordinal (A) would impose a false ordering. Target encoding (C) is a leakage risk and unnecessary for low cardinality. Hashing (D) is overkill.

**A3.** **MAR (Missing At Random)** means the probability of missingness depends on *observed* variables — e.g., income is missing more often for young respondents, but conditional on age, income missingness is random. **MCAR (Missing Completely At Random)** means missingness is independent of everything. **MNAR (Missing Not At Random)** means missingness depends on the *unobserved* value itself — e.g., high earners refuse to report income. MAR is generally tractable with model-based imputation using the observed variables; MNAR is fundamentally difficult without extra information.

**A4. C.** Log transform pulls in the long right tail, making the distribution more symmetric. StandardScaler and MinMaxScaler don't fix skewness — they just re-center and rescale a still-skewed distribution.

**A5.** $\text{sin\_hour} = \sin(2\pi h / 24)$, $\text{cos\_hour} = \cos(2\pi h / 24)$. This encoding preserves the fact that hour 23 and hour 0 are close (both map near the same point on the unit circle), whereas the raw integers 23 and 0 look maximally apart.

**A6. B.** A window centered on $t$ includes data from times $> t$, which won't be available at prediction time. Use windows that end at $t$ (i.e., look strictly backward). This is one of the most common look-ahead bugs.

**A7.** Naive target encoding assigns each level the mean target computed from all training rows including the row itself. This means the training data for a rare-category row is essentially its own target — a perfect fit that vanishes in production. Mitigations: **out-of-fold** encoding (compute each row's encoded value from other folds' rows), **smoothing** toward the global target mean, **adding noise**, or using **CatBoost's ordered target encoding**.

**A8. C.** Trees split on thresholds, and rescaling doesn't change which splits are optimal. Feature scaling neither helps nor hurts tree accuracy (it might minutely affect implementation-specific tiebreaking but no real difference).

**A9.** When the data contains outliers. `StandardScaler` uses mean and std, both sensitive to outliers. `RobustScaler` uses median and IQR, which are robust — a few extreme values won't distort the scale for the rest of the data.

**A10. B.** A `Pipeline` ensures that fit is only called on training data during each CV fold. It's the difference between "leakage as a possibility every code change" and "leakage as a code smell you'd have to work at to introduce."

---

## 6. Practice Prompts

1. **Leakage lab.** Write two versions of a training script: one that fits `StandardScaler` on the whole dataset before splitting, one that uses a `Pipeline`. Compare CV vs test scores. Notice the inflated CV in the leaky version.
2. **Encoding shootout.** On a high-cardinality categorical dataset, compare one-hot, ordinal, frequency, and target encoding for a logistic regression baseline and for XGBoost. Which combos work best?
3. **Time-feature crafting.** Given a `datetime` column and a target, extract at least 8 useful features (extracted parts, cyclic encodings, holiday flags, days-since features). Measure feature importance.
4. **Missingness signal.** Take a dataset with real missing values. Fit a model with and without a `feature_is_missing` indicator column. See when the indicator itself is important.
5. **Custom transformer.** Write a `sklearn`-compatible `TransformerMixin` that computes a rolling z-score. Verify it fits/transforms correctly inside a Pipeline.

---

## 7. References

- Alice Zheng & Amanda Casari, *Feature Engineering for Machine Learning* (2018).
- Kuhn & Johnson, *Feature Engineering and Selection* (free web version).
- Rubin, *Multiple Imputation for Nonresponse in Surveys* (canonical on missing data).
- Prokhorenkova et al., ["CatBoost: unbiased boosting with categorical features"](https://arxiv.org/abs/1706.09516) (2017).
- scikit-learn docs: [Compose](https://scikit-learn.org/stable/modules/compose.html), [Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html).
