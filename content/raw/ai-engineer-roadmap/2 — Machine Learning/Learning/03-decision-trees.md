# Decision Trees — Master Study Guide

> **Track:** Machine Learning · **Module:** 03
> **Prerequisites:** Basic probability, entropy intuition helps.
> **Time budget:** ~8–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Decision trees are the **building block** of every modern tabular-data ML method. Random forests are ensembles of trees. Gradient boosting is a sequence of trees. Even neural networks on tabular data underperform tree-based methods often enough that XGBoost / LightGBM / CatBoost remain the state of the art for structured data.

A single decision tree is also the **most interpretable ML model** you can build. It's a flowchart. A stakeholder who has never heard of statistics can read one and follow its logic. That interpretability comes at a cost — single trees are high-variance, and they overfit spectacularly — but understanding *why* is the whole reason ensembles work.

**Fundamental principles you must own:**

1. **A tree partitions feature space into axis-aligned rectangles.** At each split, it picks the (feature, threshold) that best separates the target.
2. **The "best" split maximizes an impurity reduction** — Gini or entropy for classification, MSE for regression.
3. **Trees are grown greedily** — locally optimal splits, no backtracking, no global optimum.
4. **Unregularized trees always overfit** to any training data — they can memorize. You must control complexity via depth, min-samples, or pruning.
5. **Trees are scale-invariant** — no need to standardize features.
6. **Trees handle nonlinearities and interactions natively** — every path is a conjunction of conditions.

If you retain nothing else: **a tree is a chain of `if/else` questions chosen to reduce impurity.** Everything else is engineering that principle.

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Splitting Criteria: Gini, Entropy, MSE

**For classification**, at a node $N$ with class proportions $p_1, p_2, \ldots, p_K$:

**Gini impurity:**

$$G(N) = 1 - \sum_{k=1}^K p_k^2$$

- Range: $[0, 1 - 1/K]$; 0 = pure node (all one class).
- Intuition: probability that two random draws from the node have different labels.

**Entropy (Shannon):**

$$H(N) = -\sum_{k=1}^K p_k \log_2 p_k$$

- Range: $[0, \log_2 K]$; 0 = pure.
- Intuition: information needed to specify a random draw's class.

**Information gain** from a candidate split partitioning $N$ into $N_L, N_R$:

$$\text{IG} = I(N) - \frac{|N_L|}{|N|} I(N_L) - \frac{|N_R|}{|N|} I(N_R)$$

Where $I$ is Gini or entropy. The tree picks the split maximizing IG.

**For regression**, use variance / MSE at each node:

$$\text{MSE}(N) = \frac{1}{|N|}\sum_{i \in N} (y_i - \bar y_N)^2$$

Split reduces MSE:

$$\Delta = \text{MSE}(N) - \frac{|N_L|}{|N|}\text{MSE}(N_L) - \frac{|N_R|}{|N|}\text{MSE}(N_R)$$

**Gini vs entropy — does it matter?** In practice, almost never. Gini is slightly faster (no log); trees built with either usually agree on ~95% of splits. **Use Gini by default** and don't overthink it.

---

### 2.2 The Greedy Splitting Algorithm

At each node:

1. For each feature $j = 1 \ldots p$:
   - Sort samples by $x_j$.
   - For each candidate threshold $t$ (usually the midpoints between consecutive distinct values):
     - Compute the impurity reduction from splitting at $x_j \leq t$.
2. Choose the (feature, threshold) with the maximum reduction.
3. Recurse on the two child nodes until a stopping criterion is met.

**Stopping criteria:**

- Node is pure (all one class / low variance).
- `max_depth` reached.
- `min_samples_split` — node too small to split.
- `min_samples_leaf` — split would create a leaf smaller than this.
- Minimum impurity decrease.

**Complexity.** At each node, sorting each feature is $O(n \log n)$; across $p$ features it's $O(np \log n)$. Trees of depth $d$ do at most $O(2^d)$ node splits. Total: $O(np \log n \cdot 2^d)$ worst case, though real datasets don't approach this.

**Categorical features.** Trees natively split on categorical variables, but scikit-learn's implementation doesn't — it requires numeric input, so you must one-hot or ordinally encode. LightGBM and CatBoost do handle categoricals natively.

---

### 2.3 Overfitting & Controls (Pre-pruning, Post-pruning)

An unrestricted decision tree can grow until every leaf contains one sample — perfect training accuracy, garbage test accuracy. Controls come in two forms.

**Pre-pruning (stop early):**

- `max_depth` — hard cap on tree depth.
- `min_samples_split` — minimum samples to consider a node for splitting.
- `min_samples_leaf` — minimum samples in a leaf. Direct control on overfitting.
- `max_leaf_nodes` — cap on total leaves. More flexible than `max_depth`.
- `min_impurity_decrease` — split only if the impurity drop is above this.

**Post-pruning (grow, then prune):**

- **Cost-complexity pruning** (Breiman, 1984), scikit-learn's `ccp_alpha`.
- The idea: define the **cost-complexity criterion**:

  $$C_\alpha(T) = \sum_{\text{leaves}} \text{impurity}(\text{leaf}) + \alpha |T|$$

  Where $|T|$ = number of leaves. Larger $\alpha$ → smaller tree. Use CV to select $\alpha$.

**Bias–variance for trees.**

- **Deep trees**: low bias, high variance. Individual predictions swing wildly with small changes in training data.
- **Shallow trees**: higher bias, lower variance. Miss patterns but don't overfit.

**The key insight — trees are hard to regularize well as single models.** Their variance is *inherent* to the greedy split process. That's why **ensembles** (RF, GBM) exist: they reduce that variance without needing to hobble each tree.

---

### 2.4 Interpretation & Feature Importance

**Tree structure interpretation.** Read a tree top-down as nested `if/else`:

```
if petal_length <= 2.45:
    return "setosa"
else:
    if petal_width <= 1.75:
        if petal_length <= 4.95: return "versicolor"
        else:                    return "virginica"
    else:
        return "virginica"
```

Every leaf is a conjunction of conditions. This is why trees are **transparent** in a way linear models aren't for feature interactions — a tree "sees" `petal_length > 2.45 AND petal_width > 1.75` as a single rule.

**Feature importance.** In scikit-learn, the default `feature_importances_` for trees is **mean decrease in impurity (MDI)** — for each feature, sum the impurity reduction across all splits using it, weighted by the number of samples reaching those nodes.

$$\text{MDI}(j) = \frac{1}{|T|}\sum_{\substack{\text{nodes } v \\ \text{splitting on } j}} \frac{|N_v|}{n} \cdot \Delta I_v$$

**Warning about MDI:** it's biased toward high-cardinality features and continuous features. For serious analysis, use **permutation importance** — shuffle one feature's values and measure how much accuracy drops. It's slower but honest.

**Partial dependence plots (PDP)** and **individual conditional expectation (ICE) plots** show how predictions change as a feature varies, marginalizing over others. Great for interpreting trees and ensembles.

---

### 2.5 Regression Trees & Extensions

Regression trees replace class-proportion splits with variance-reduction splits, and each leaf outputs the **mean** of the training targets in that leaf. Predictions are piecewise-constant — a staircase, not a curve.

**Consequences:**

- Excellent for capturing non-monotone, interaction-heavy relationships.
- Bad at extrapolating — a tree cannot predict outside the range of the training targets. If your minimum training $y$ is 10, no leaf will output 5. Ensembles inherit this.
- Discontinuous predictions — small feature changes can push a sample into a different leaf and jump the prediction.

**Beyond CART (Classification and Regression Trees):**

- **ID3, C4.5, C5.0** — classic entropy-based, with support for categorical splits.
- **CHAID** — chi-square-based splits, common in marketing.
- **Model trees** (M5, cubist) — linear regressions at leaves instead of constants; smoother predictions.
- **Oblique / multivariate trees** — splits are linear combinations, not axis-aligned. More powerful but slower and less interpretable.
- **Isolation Forest** — trees used for anomaly detection (isolate outliers with few splits).

---

## 3. Mental Models & Analogies

### 3.1 The "20 Questions" Model

Decision trees are the game of 20 Questions where you always ask the **most informative yes/no question**, given what you know so far. "Is it bigger than a breadbox?" cuts the space of possibilities in half. Then, based on the answer, the next question narrows further.

At each node, the tree looks over every possible question (feature/threshold combination) and picks the one that most reduces uncertainty (measured by Gini or entropy). It never revises earlier questions — that's the greedy part. If an early bad question was chosen, subsequent questions can only paper over the mistake.

This is why **a single tree is fragile**: get the first split wrong (which happens easily with small samples or noisy features) and everything downstream compounds the error. **Ensembles** average over many trees with intentional randomness — they're playing many games of 20 Questions in parallel and averaging the answers.

### 3.2 The "Cutting the Cake with Straight Cuts" Model

Imagine your feature space as a 2D map, with each data point colored by class. Splitting is like slicing a rectangular cake with straight, axis-aligned cuts (horizontal or vertical). Each slice partitions the cake into two smaller rectangles. Keep slicing recursively — each new slice can only be horizontal or vertical, and must respect existing slices.

- **Trees can carve arbitrarily complex axis-aligned regions** by combining many slices — but they cannot make diagonal cuts.
- **Diagonal boundaries** require many small stair-step slices; the tree approximates them poorly.
- **Deep trees** = fine-grained slicing, potentially cutting one cell per training point (overfit).
- **Shallow trees** = broad regions, missing detail.

This model shows why data preprocessing matters less for trees: rescaling a feature just relabels the tick marks on that axis — the same cuts remain optimal.

![IMG-DT-01](/2%20—%20Machine%20Learning/images/IMG-DT-01.jpg)
> **Caption:** A tree is a set of rules; each leaf is an axis-aligned region of feature space.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Trees Are Robust to Noise, So I Don't Need to Clean Data"

Trees are *scale-invariant* and *tolerant of monotone transformations*, but they are **not** robust to:

- **Noisy labels** — trees will happily create leaves matching noise.
- **Irrelevant features** — trees waste splits on them, especially with many features and few samples.
- **Missing values** — scikit-learn's tree doesn't handle them natively; you must impute.

Clean data still matters. Trees are just less sensitive to feature *scaling* than distance-based models.

### 4.2 "Feature Importance from scikit-learn Tells Me What Really Matters"

The default MDI importance is **biased** toward high-cardinality features (many possible split points → many chances to be the best split) and continuous features (finer thresholds). A `user_id` column encoded as integers will often top the importance list — even though it's noise.

**Use permutation importance** for anything you'll act on: it's model-agnostic and directly measures predictive value.

### 4.3 "Trees Can Extrapolate Because They Handle Nonlinearity"

Trees handle *nonlinearity within the training data*. They cannot extrapolate. If your training targets range from 10 to 100 and a test sample has features suggesting the true answer is 150, a tree will output at most 100 (the largest leaf mean). This bites hardest in time-series regression where the target has a trend — a tree literally cannot predict a new maximum. Solutions: model residuals with a linear model, add a linear extrapolation layer, or use gradient boosting on differenced targets.

---

## 5. Self-Assessment Bank (Decision Trees)

### Questions

**Q1 (Short answer).** Write the Gini impurity formula and the information gain of a split.

**Q2 (Multiple choice).** A decision tree is **greedy** because it:
A. Splits on the feature with the highest correlation to target.
B. Chooses the locally best split at each node without backtracking.
C. Always minimizes global error.
D. Splits until every leaf is pure.

**Q3 (Short answer).** In practice, do Gini and entropy usually produce different trees? Which does scikit-learn use by default?

**Q4 (Multiple choice).** Which of the following will most reliably prevent overfitting in a single decision tree?
A. Increasing `max_depth`.
B. Setting `min_samples_leaf` to a larger value.
C. Adding more features.
D. Removing feature scaling.

**Q5 (Short answer).** Why does scikit-learn's `feature_importances_` give misleading results in the presence of high-cardinality features, and what is a better alternative?

**Q6 (Multiple choice).** A regression tree's prediction for any input is:
A. A linear function of the input features.
B. A weighted sum of neighbor targets.
C. The mean target of the training samples that fall in the same leaf.
D. Bounded only by the tree's max depth.

**Q7 (Short answer).** Why can't a decision tree extrapolate beyond the range of the training targets in regression?

**Q8 (Multiple choice).** Decision trees are:
A. Scale-variant — you must standardize features.
B. Scale-invariant — feature scaling doesn't change the splits.
C. Only handle continuous features.
D. Always require one-hot encoding of continuous features.

**Q9 (Short answer).** What is cost-complexity pruning and how do you select its $\alpha$?

**Q10 (Multiple choice).** Which of the following is true about the *bias–variance* profile of an unrestricted decision tree?
A. Low bias, low variance.
B. High bias, low variance.
C. Low bias, high variance.
D. High bias, high variance.

---

### Answer Key & Detailed Explanations

**A1.** Gini: $G(N) = 1 - \sum_k p_k^2$. Information gain from a split producing children $N_L, N_R$: $\text{IG} = I(N) - (|N_L|/|N|)I(N_L) - (|N_R|/|N|)I(N_R)$, where $I$ is Gini or entropy.

**A2. B.** Greedy = locally optimal at each step, no reconsideration. This is why trees can miss globally better structures (an early split that seems suboptimal but enables great downstream splits).

**A3.** No — Gini and entropy usually agree on splits. scikit-learn defaults to Gini for `DecisionTreeClassifier` because it's slightly cheaper (no `log`). Don't tune between them expecting big gains.

**A4. B.** `min_samples_leaf` directly forces leaves to be large enough to represent averages of multiple samples, which reduces variance. Increasing `max_depth` does the opposite. Adding features gives more overfitting opportunities. Feature scaling is irrelevant.

**A5.** MDI accumulates impurity reduction across splits using each feature. High-cardinality features (many possible thresholds) have *more chances* to be picked and to produce small impurity drops that add up. So a `user_id` column can look important even when it's essentially random. **Permutation importance** avoids this: it measures the *actual* drop in test performance when the feature's values are shuffled.

**A6. C.** Regression trees output leaf means. Predictions are piecewise-constant. This is why they can't extrapolate and why they can't produce smooth predictions.

**A7.** A leaf's prediction is the **mean of the training targets** in that leaf. There is no leaf whose mean is larger than the training maximum (or smaller than the training minimum) because a mean of values in some range is itself in that range. Ensembles of trees inherit this.

**A8. B.** Trees split on thresholds ($x_j \leq t$); the ordering of feature values is what matters, not the units. Rescaling $x_j$ by a constant just shifts $t$ correspondingly. Categorical features must be encoded numerically for scikit-learn but the tree itself doesn't require one-hot.

**A9.** Cost-complexity pruning grows a full tree, then walks upward pruning subtrees to minimize $C_\alpha(T) = \sum_{\text{leaves}} \text{impurity} + \alpha |T|$. Higher $\alpha$ → smaller tree. Select $\alpha$ via cross-validation across a range of values; scikit-learn's `cost_complexity_pruning_path` yields the candidate $\alpha$s.

**A10. C.** Deep unrestricted trees have low training bias — they can memorize the training set. Their variance is high — small changes in the data can radically alter the top splits and thus every downstream branch. This is the exact reason bagging (Random Forest) works: averaging many high-variance, low-bias learners reduces variance.

---

## 6. Practice Prompts

1. **From scratch (small).** Implement a decision-tree classifier that supports Gini and `max_depth`. Verify on Iris that it matches scikit-learn's tree structure.
2. **Depth vs test error.** Fit trees of `max_depth` 1..20 on a real dataset. Plot training and validation error. Identify the sweet spot.
3. **Feature importance comparison.** On the same dataset, compare `feature_importances_` (MDI) against `permutation_importance` from scikit-learn. Note the differences.
4. **Extrapolation demo.** Train a regression tree on $y = x + \text{noise}$ for $x \in [0, 10]$. Predict on $x \in [10, 20]$. Confirm the prediction saturates at the max training leaf mean.
5. **Cost-complexity pruning.** Use `cost_complexity_pruning_path` to get candidate $\alpha$s, cross-validate, and pick the best. Compare final test accuracy to a hand-tuned `max_depth`.

---

## 7. References

- Breiman, Friedman, Olshen & Stone, *Classification and Regression Trees* (CART, 1984) — the canonical text.
- ISL chapter 8 (free PDF).
- ESL chapter 9.
- scikit-learn docs: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html).
- Terence Parr & Jeremy Howard, ["How to explain gradient boosting"](https://explained.ai/gradient-boosting/) (starts with regression trees, essential for Module 05).
