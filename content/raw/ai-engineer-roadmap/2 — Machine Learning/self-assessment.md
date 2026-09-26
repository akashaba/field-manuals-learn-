# Machine Learning — Consolidated Self-Assessment

> **Scope:** All 13 learning modules **and** Build #1.
> **Format:** Timed (2.5 hours), no reference material, notebook off.
> **Passing bar:** ≥ 80% overall **and** ≥ 60% in every section.

This is the exit exam for Month 2 — Machine Learning. Take it when you *think* you're ready. Score honestly. Retake two weeks later after remediating weaknesses.

---

## Section A — Linear & Logistic Regression (12 points)

**A1 (1 pt, MC).** The closed-form solution to OLS is $(X^\top X)^{-1} X^\top \mathbf{y}$ and it exists iff:
- (a) $n > p$
- (b) $X^\top X$ is invertible (i.e., $X$ has full column rank)
- (c) $y$ is Normal
- (d) always

**A2 (2 pt).** Explain why Lasso can drive coefficients exactly to zero but Ridge cannot.

**A3 (1 pt).** Write the sigmoid function and its derivative in terms of $\sigma$ itself.

**A4 (2 pt).** Derive binary cross-entropy from the assumption that each label is Bernoulli with $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol\beta)$.

**A5 (2 pt).** Interpret a logistic-regression coefficient of $\hat\beta_j = 0.7$.

**A6 (2 pt).** Why must you standardize features before Ridge regression, but not (strictly) before OLS?

**A7 (2 pt).** In scikit-learn's `LogisticRegression`, what does a *smaller* `C` mean, and what is the practical effect?

---

## Section B — Trees, Forests & Boosting (14 points)

**B1 (1 pt).** Write the Gini impurity formula for a node with class proportions $p_k$.

**B2 (2 pt, MC).** A decision tree grown to full depth typically has:
- (a) high bias, low variance
- (b) low bias, high variance
- (c) low bias, low variance
- (d) high bias, high variance

**B3 (2 pt).** Explain what bootstrap sampling and feature subsampling separately contribute in Random Forest.

**B4 (2 pt).** Under what conditions does a Random Forest fail to predict beyond the range of the training targets? Give a real-world scenario.

**B5 (2 pt).** In gradient boosting, at round $m$, what is the tree $h_m$ trained to predict?

**B6 (2 pt).** Explain XGBoost's second-order optimization: what does it use that plain GBM doesn't, and what advantage does it give?

**B7 (2 pt).** Contrast the bias-variance strategies of RF vs XGBoost — which primarily attacks variance, which primarily attacks bias?

**B8 (1 pt, MC).** In XGBoost, the most reliable single technique to prevent overfitting is:
- (a) `n_estimators = 10000`
- (b) early stopping on a validation set with small `learning_rate`
- (c) `max_depth = None`
- (d) `learning_rate = 1.0`

---

## Section C — SVM & KNN (10 points)

**C1 (1 pt).** What does "max-margin" mean in an SVM, and why do we want it?

**C2 (2 pt).** In one paragraph, explain the "kernel trick".

**C3 (2 pt).** Which training samples influence an SVM's decision function, and what property of $\alpha_i$ identifies them?

**C4 (1 pt).** In KNN, describe the bias-variance behavior of small $k$ vs large $k$.

**C5 (2 pt).** Describe the "curse of dimensionality" and its practical effect on KNN.

**C6 (1 pt, MC).** For text with 100k features and 500k samples, the practical SVM is:
- (a) `SVC(kernel='rbf')`
- (b) `LinearSVC` or `SGDClassifier(loss='hinge')`
- (c) `SVC(kernel='poly', degree=5)`
- (d) don't use SVM

**C7 (1 pt).** Why is feature scaling essential for both RBF-SVM and KNN?

---

## Section D — Clustering (8 points)

**D1 (1 pt).** What quantity does K-means minimize?

**D2 (2 pt).** Give one strength and one weakness of DBSCAN vs K-means.

**D3 (2 pt).** Define the silhouette score of a point $i$ and interpret values close to +1, 0, and -1.

**D4 (1 pt, MC).** For non-convex "moon-shaped" clusters, the better choice is:
- (a) K-means
- (b) Agglomerative with single linkage or DBSCAN
- (c) Bisecting K-means
- (d) None of the above

**D5 (2 pt).** Name two methods of evaluating a clustering when you have no ground-truth labels.

---

## Section E — Feature Engineering (10 points)

**E1 (2 pt).** Why must transformers (scalers, encoders, imputers) be fit only on training data?

**E2 (2 pt).** For a linear model with a 100-level categorical feature (no ordinal relationship), name the trade-offs of one-hot vs target encoding.

**E3 (2 pt).** Give the cyclic sine/cosine encoding formula for hour of day and explain the property it preserves.

**E4 (2 pt).** Explain naive target encoding's leakage risk and one technique to mitigate it.

**E5 (2 pt).** In time-series feature engineering, why is a "rolling window centered on time $t$" a bug?

---

## Section F — Validation & Metrics (14 points)

**F1 (2 pt).** Why do we need train / validation / test — three splits, not two?

**F2 (2 pt).** When should you use `StratifiedKFold` vs `KFold` vs `TimeSeriesSplit` vs `GroupKFold`? Give one scenario each.

**F3 (2 pt).** Describe how a learning curve reveals underfitting vs overfitting.

**F4 (2 pt).** Define precision, recall, and specificity in terms of TP/FP/FN/TN.

**F5 (2 pt).** For a dataset with 0.5% positives, why is PR-AUC generally more informative than ROC-AUC?

**F6 (2 pt).** State the probabilistic interpretation of ROC-AUC in one sentence.

**F7 (2 pt).** Explain calibration and give one classifier known to be well-calibrated by default and one known to be poorly calibrated.

---

## Section G — Hyperparameter Tuning (8 points)

**G1 (2 pt).** Explain why random search often beats grid search on the same compute budget.

**G2 (2 pt).** What does Bayesian optimization use to decide the next hyperparameter to try?

**G3 (2 pt).** Why must tuning happen inside CV rather than on a single train/val split?

**G4 (1 pt, MC).** To honestly report tuned-model performance:
- (a) report the best CV score from the search
- (b) report the score on an untouched test set (single evaluation)
- (c) repeat search on the test set
- (d) average the top-10 CV scores

**G5 (1 pt).** What is early stopping in gradient boosting, and which hyperparameter does it effectively replace?

---

## Section H — scikit-learn & Build Discipline (14 points)

**H1 (2 pt).** Describe the estimator API: what methods does every scikit-learn estimator implement?

**H2 (2 pt).** Why is wrapping preprocessing and model in a `Pipeline` essential for leakage-free CV?

**H3 (2 pt).** What is `ColumnTransformer` and when do you use it?

**H4 (1 pt, MC).** In scikit-learn, error metrics like MSE appear as `neg_mean_squared_error` because:
- (a) higher is always better for scorers
- (b) it's a typo
- (c) MSE is negative
- (d) legacy convention

**H5 (2 pt).** In a Pipeline, `pipe.get_params()["clf__C"]` refers to what?

**H6 (2 pt).** What is a "model card" and name three sections it should have.

**H7 (2 pt).** In an ML API service, why must the model be loaded once at startup (via lifespan) rather than on each request?

**H8 (1 pt).** Which two library versions must be pinned across training and inference environments to avoid deserialization surprises?

---

# Answer Key & Detailed Explanations

## Section A

**A1. (b).** $X^\top X$ must be invertible — equivalently, $X$ must have full column rank (no perfect multicollinearity), which requires $n \geq p+1$ but that alone doesn't suffice.

**A2.** Ridge's L2 penalty $\|\boldsymbol\beta\|_2^2$ has smooth spherical level sets. The OLS ellipse contacts the sphere on its surface, generally at points with nonzero coordinates. Lasso's L1 penalty $\|\boldsymbol\beta\|_1$ has a diamond level set with corners on the coordinate axes. The OLS ellipse hits a corner with high probability — a corner has one or more coordinates equal to zero. So Lasso produces sparse solutions; Ridge shrinks but doesn't zero out.

**A3.** $\sigma(z) = 1/(1 + e^{-z})$; $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**A4.** $P(y_i | \mathbf{x}_i) = p_i^{y_i}(1 - p_i)^{1 - y_i}$. Likelihood $\mathcal{L} = \prod P(y_i | \mathbf{x}_i)$. Negative log-likelihood $-\log \mathcal{L} = -\sum [y_i \log p_i + (1 - y_i)\log(1 - p_i)]$. Divide by $n$ for the mean cross-entropy. Minimizing this is MLE.

**A5.** Holding other features constant, a one-unit increase in $x_j$ multiplies the odds of the positive class by $e^{0.7} \approx 2.01$ — the odds roughly double.

**A6.** OLS coefficients scale-equivariantly with feature units, so scaling doesn't change predictions. Ridge's L2 penalty $\lambda \|\boldsymbol\beta\|^2$ penalizes all coefficients uniformly regardless of feature scale — an unscaled feature with a large range is under-penalized. Standardization puts all features on equal footing so regularization behaves symmetrically.

**A7.** scikit-learn parameterizes as $C = 1/\lambda$; smaller `C` means larger $\lambda$, meaning **stronger regularization**. Practically: coefficients shrink toward zero, bias rises, variance drops. Useful when overfitting is the risk.

---

## Section B

**B1.** $G(N) = 1 - \sum_k p_k^2$.

**B2. (b).** Deep unrestricted trees can memorize training data (low bias) but small data changes radically alter top splits (high variance). Why bagging works.

**B3.** **Bootstrap** decorrelates trees at the row level — each tree sees a slightly different training set. **Feature subsampling** decorrelates at the split level — each tree considers a different random subset of features at every node. Both drive down the pairwise correlation $\rho$ between tree predictions, which is what reduces ensemble variance.

**B4.** Trees output the mean target of a leaf; RF averages trees. Both are convex combinations of training targets, so no output can exceed the training range. Real-world bite: time-series forecasting with a trend — an RF trained on data through Sept can't predict a new max in Oct even if the trend extends beyond training.

**B5.** The negative gradient of the loss with respect to the current ensemble's predictions, evaluated at $F_{m-1}$. For squared error, this is the residual $y_i - F_{m-1}(\mathbf{x}_i)$. For other losses, it's the "pseudo-residual" from the gradient.

**B6.** XGBoost uses the **second derivative** (Hessian) in addition to the gradient — Newton-style optimization at each split. This gives closed-form optimal leaf weights, better split gains, and faster convergence than plain gradient-only boosting.

**B7.** **RF** primarily attacks **variance** — averaging many low-bias, high-variance trees. **XGBoost** primarily attacks **bias** — each shallow, high-bias tree corrects the ensemble's residual, so bias shrinks with each round. RF is embarrassingly parallel; boosting is sequential.

**B8. (b).** Early stopping with a small learning rate is the most reliable regularizer. A blind large `n_estimators` (a) can overfit; `learning_rate = 1.0` (d) makes each tree contribute too much; unrestricted `max_depth` (c) is fine only with strong regularization elsewhere.

---

## Section C

**C1.** Max-margin means finding the separating hyperplane with the largest perpendicular distance to the nearest training points. Wider margin → more robust classifier → better generalization guarantees (statistical learning theory).

**C2.** SVM's optimization only ever uses dot products between feature vectors. A kernel $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top \phi(\mathbf{z})$ computes the dot product in a (potentially infinite-dim) feature space $\phi$ *without ever computing $\phi$ explicitly*. This lets you fit nonlinear boundaries in the original space at the cost of a linear one in the transformed space — with $O(n^2)$ Gram matrix rather than the actual dimensionality.

**C3.** Only **support vectors** — samples with $\alpha_i > 0$. Non-SVs have $\alpha_i = 0$ and could be removed from training without changing the model.

**C4.** Small $k$ = **high variance, low bias** — a single noisy neighbor dominates. Large $k$ = **low variance, high bias** — averages over many points, oversmoothing local structure.

**C5.** As $p \to \infty$, the ratio $d_\max / d_\min \to 1$ for pairwise distances. "Nearest neighbor" loses meaning because points concentrate at equidistant. Additionally, feature-space volume grows exponentially — no realistic dataset can densely sample high-$p$ space. KNN accuracy degrades sharply.

**C6. (b).** Kernel SVMs need an $O(n^2)$ Gram matrix — $500k^2$ entries is infeasible. Linear formulations scale linearly and are the practical choice.

**C7.** Both use Euclidean distance. Unscaled features let the largest-range feature dominate every distance calculation; smaller-range features are effectively ignored. Standardization puts features on equal footing.

---

## Section D

**D1.** Within-cluster sum of squares (inertia): $\sum_j \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol\mu_j\|^2$.

**D2.** **DBSCAN strengths:** discovers arbitrarily shaped clusters, doesn't require $k$, robust to outliers (they become noise). **DBSCAN weaknesses:** sensitive to `eps` and `min_samples`; struggles with variable-density clusters. K-means opposite: assumes convex spherical clusters and requires $k$, but is fast and simple.

**D3.** $s(i) = (b(i) - a(i)) / \max(a(i), b(i))$ where $a(i)$ = mean distance within its cluster and $b(i)$ = mean distance to the nearest other cluster. **+1** = well clustered (much closer to own cluster). **0** = boundary between clusters. **-1** = possibly in the wrong cluster.

**D4. (b).** K-means enforces convex boundaries and cuts moon shapes wrongly. Single-linkage agglomerative or DBSCAN can trace along density chains.

**D5.** Internal metrics: silhouette, Davies-Bouldin, Calinski-Harabasz, gap statistic. Stability: refit on bootstrap samples and measure agreement. Downstream utility: cluster IDs as features. Visual inspection via PCA/UMAP. Domain interpretation: can you name each cluster?

---

## Section E

**E1.** Transformers compute statistics (means, quantiles, category counts) from the data they're fit on. If those include validation/test data, information leaks — test error is optimistically biased, production performance disappoints. Fitting only on training keeps the test set truly unseen.

**E2.** **One-hot**: interpretable, no false ordering, but 100 sparse binary columns. **Target encoding**: one compact column carrying signal, but high leakage risk (must use out-of-fold + smoothing) and gives away target information. For linear models, one-hot is usually safer; target encoding needs careful implementation.

**E3.** $\text{sin\_hour} = \sin(2\pi h / 24)$; $\text{cos\_hour} = \cos(2\pi h / 24)$. Preserves the fact that hour 23 and hour 0 are close on a circle — a raw integer 23 vs 0 looks maximally apart, breaking distance-based models.

**E4.** Naive target encoding assigns each level the mean target on training data, including the row itself — the training row for a rare category is essentially its own target, causing perfect fit that vanishes in production. Mitigations: out-of-fold encoding, smoothing toward global mean, adding noise, or CatBoost's ordered target encoding.

**E5.** A window centered on time $t$ includes data from times $> t$, which won't be available at prediction time — classic look-ahead leakage. Use windows that end at $t$ (strictly backward-looking).

---

## Section F

**F1.** If you tune hyperparameters on the test set, you *implicitly optimize for that specific test set* — its performance is no longer an unbiased estimate. The **validation** set is for modeling decisions (hyperparameters, features, model class); the **test** set is touched once at the end for the final production estimate.

**F2.** **StratifiedKFold** — imbalanced classification (preserve class ratios per fold). **KFold** — balanced, i.i.d. data. **TimeSeriesSplit** — any time-series (past → future). **GroupKFold** — repeated measures (multiple rows per patient/user/session) so groups never span fold boundaries.

**F3.** **Underfitting**: both curves low, close to each other — model can't capture pattern; more data won't help. **Overfitting**: training high, validation lower, wide gap — model memorizes; more data or regularization would help. **Well-fit**: both high with small gap.

**F4.** Precision = $TP / (TP + FP)$. Recall = $TP / (TP + FN)$. Specificity = $TN / (TN + FP)$.

**F5.** ROC-AUC's x-axis is FPR = $FP / (FP + TN)$. When $TN$ dominates (0.5% positives → 99.5% negatives), doubling $FP$ barely moves FPR — ROC-AUC looks great even with terrible precision. PR-AUC uses precision, which directly compares TP to FP, so it reflects the operating cost.

**F6.** ROC-AUC = $P(f(\mathbf{x}^+) > f(\mathbf{x}^-))$ — probability that the classifier scores a random positive higher than a random negative.

**F7.** Calibration = predicted probabilities match empirical frequencies (predictions of 0.7 should correspond to ~70% true positive rate). Well-calibrated by default: **logistic regression**. Poorly calibrated: **Random Forest** (biased toward 0.5), **SVM decision scores**, **Naive Bayes** (often over-confident).

---

## Section G

**G1.** In high-dimensional hyperparameter spaces, only a few dimensions dominate. Grid search wastes evaluations on unimportant dimensions and gives coarse coverage of important ones. Random search decouples: with $N$ trials, each dimension gets $N$ distinct values sampled — better coverage of the important dimensions with the same budget.

**G2.** A **surrogate model** (Gaussian process or tree-based ensemble) predicting the objective as a function of hyperparameters, plus an **acquisition function** (Expected Improvement, UCB, TPE) that balances exploitation (near current best) and exploration (high uncertainty). The next trial is the argmax of the acquisition function.

**G3.** A single train/val split means best hyperparameters are optimal *for that specific val split's noise*. CV averages across splits, giving a more stable, less optimistically biased signal for model comparison.

**G4. (b).** Refit on train+val with best hyperparameters; evaluate ONCE on the untouched test set. Reporting the CV best (a) is optimistically biased. Repeating on test (c) contaminates it.

**G5.** Early stopping monitors validation loss during boosting and stops when it stops improving for a "patience" number of rounds — effectively selecting `n_estimators` from data rather than sweeping it.

---

## Section H

**H1.** `__init__(**hyperparams)`, `fit(X, y)` (returns self), `predict(X)`, and optionally `predict_proba(X)`, `decision_function(X)`, `transform(X)`, `fit_transform(X)`, `score(X, y)`. Learned attributes end with trailing underscore (`coef_`, `feature_importances_`, `classes_`).

**H2.** A `Pipeline`'s `fit(X_train)` sequentially fits and transforms each step using only training data. Passed to `cross_val_score` or `GridSearchCV`, each CV fold refits the entire pipeline — the held-out fold is never seen by any transformer during fit. Without a pipeline, preprocessing done before CV already peeked at held-out data → leakage.

**H3.** `ColumnTransformer` applies **different transformers to different columns** in parallel — numeric columns get scaled, categoricals get one-hot encoded, text gets vectorized, all in one composite transformer with the standard API.

**H4. (a).** scikit-learn's convention: higher is always better for scorers. MSE is lower-is-better, so its scorer is negated. `GridSearchCV.best_score_` etc. always mean "highest" regardless of metric direction.

**H5.** The `C` hyperparameter of the step named `clf` inside the pipeline. Double underscore syntax accesses nested-step parameters — lets `GridSearchCV` tune parameters of any step.

**H6.** A **model card** is a document describing a model for users and reviewers. Sections: (1) Model details (name, version, framework, date); (2) Intended use and out-of-scope use; (3) Training data; (4) Metrics and evaluation datasets; (5) Ethical considerations / fairness across subgroups; (6) Known limitations; (7) Contact.

**H7.** Loading a model is expensive (deserialize, warm-up, potentially load into GPU). Per-request loading kills latency and wastes memory. FastAPI's `lifespan` context manager loads the model once at startup, keeps it in `app.state`, and reuses it across requests. It also lets the service refuse to start if the model is missing or the wrong version.

**H8.** `scikit-learn` and `numpy` (and often `scipy`, `pandas`). Internal changes across minor versions can break `joblib` deserialization or silently change results.

---

## Scoring

| Section | Points | Yours |
|---------|--------|-------|
| A. Linear & Logistic | 12 | |
| B. Trees, Forests, Boosting | 14 | |
| C. SVM & KNN | 10 | |
| D. Clustering | 8 | |
| E. Feature Engineering | 10 | |
| F. Validation & Metrics | 14 | |
| G. Hyperparameter Tuning | 8 | |
| H. sklearn & Build Discipline | 14 | |
| **Total** | **90** | |

**Retake conditions.** Anything under 60% in a section, or under 80 overall → spend a week on the weak spots and retake a shuffled version.

**Passing.** Once you clear the bar — and Build #1 is shipped — you're ready to move to Month 3 (deep learning, most likely). Do not skip the build; the build is where the learning consolidates.
