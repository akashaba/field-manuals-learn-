# Linear Regression — Master Study Guide

> **Track:** Machine Learning · **Module:** 01
> **Prerequisites:** Foundations (linear algebra, probability/stats, NumPy).
> **Time budget:** ~10–15 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Linear regression is the "hello world" of statistical learning, but it also earns its keep in production. It's the model you *should* try first — because it's fast, interpretable, gives closed-form solutions, and often ties or beats fancier models on tabular data with modest signal-to-noise. Even when you don't use it as the final model, it's the **baseline** every future model is compared against.

More importantly, linear regression is the **conceptual foundation** for almost every other model. Logistic regression is its classification sibling. Neural network last layers are usually linear. Gradient boosting fits residuals (which is what linear models minimize). Regularization is best understood here first. **Master this and half of ML falls out for free.**

**Fundamental principles you must own:**

1. **Linear regression fits a linear function** $\hat{y} = X\boldsymbol\beta$ **to minimize squared error.** Everything else is a decoration.
2. **It has a closed-form solution** — you can solve $\boldsymbol\beta$ exactly, no iteration needed. This is *unusual* in ML.
3. **It makes strong assumptions** — linearity, independence, homoscedasticity, no multicollinearity, Gaussian errors — and the quality of your inference (CIs, p-values) hinges on those.
4. **Regularization (Ridge, Lasso, ElasticNet)** trades bias for variance; understand *why* before choosing hyperparameters.
5. **Coefficients are the units of interpretation.** A coefficient of 0.3 on "years of education" means "one extra year of education is associated with +0.3 on the target, holding all else constant".

If you retain nothing else: **fit the linear model first. Always.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Model & the Least-Squares Loss

**The model.** For $n$ observations and $p$ features, with $\mathbf{x}_i \in \mathbb{R}^p$ and target $y_i \in \mathbb{R}$:

$$\hat{y}_i = \beta_0 + \sum_{j=1}^p \beta_j x_{ij} = \mathbf{x}_i^\top \boldsymbol\beta$$

Where **$\boldsymbol\beta \in \mathbb{R}^{p+1}$** absorbs the intercept if you prepend a column of ones to $X$.

Stacked:

$$\hat{\mathbf{y}} = X \boldsymbol\beta, \quad X \in \mathbb{R}^{n \times (p+1)}$$

**The loss function** — Ordinary Least Squares (OLS):

$$\mathcal{L}(\boldsymbol\beta) = \frac{1}{2n} \|\mathbf{y} - X\boldsymbol\beta\|_2^2 = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \boldsymbol\beta)^2$$

**Closed-form solution (the "Normal Equations").** Setting $\nabla_\beta \mathcal{L} = 0$:

$$X^\top (X\boldsymbol\beta - \mathbf{y}) = 0 \;\Longrightarrow\; \boxed{\boldsymbol\beta^* = (X^\top X)^{-1} X^\top \mathbf{y}}$$

Provided $X^\top X$ is invertible (columns linearly independent).

**Numerical practice:** Never actually invert $X^\top X$. Use the SVD-based pseudo-inverse (`numpy.linalg.lstsq(X, y)`), which is numerically stable and handles rank-deficient $X$ gracefully.

**Why squared error?** Three good reasons:

1. **Under Gaussian noise** $y_i = \mathbf{x}_i^\top\boldsymbol\beta + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, OLS is the **maximum-likelihood estimator**.
2. It's **differentiable everywhere**, so gradients are easy.
3. It punishes big errors more than small ones, matching many practical loss functions.

The downside: it's **sensitive to outliers**. One point with residual 100 costs the model as much as 10,000 points with residual 1.

---

### 2.2 Assumptions (and What Happens When They Break)

The five OLS assumptions (the "Gauss-Markov" conditions + normality for inference):

1. **Linearity** — $E[y \mid X] = X\boldsymbol\beta$. The **relationship** between features and target is linear.
2. **Independence** — observations don't influence each other. Fails badly in time-series and clustered data.
3. **Homoscedasticity** — variance of the residuals is constant across $X$. If variance grows with $\hat{y}$, your CIs are wrong.
4. **No perfect multicollinearity** — no feature is a linear combination of others. If violated, $X^\top X$ is singular and $\boldsymbol\beta$ is undefined.
5. **Normality of residuals** — for inference (CIs, p-values), not for prediction. In large $n$ the CLT makes this less critical.

**Diagnostics you should run:**

- **Residuals vs fitted** plot. Should be a random blob around zero. A funnel → heteroscedasticity. A curve → nonlinearity.
- **QQ plot** of residuals for normality.
- **Variance Inflation Factor (VIF)** for each feature; VIF > 5–10 suggests multicollinearity.
- **Leverage and Cook's distance** for influential points.
- **Durbin-Watson** for autocorrelation.

**Fixes** when assumptions break:

- Nonlinearity → polynomial features, splines, log-transform the target.
- Heteroscedasticity → transform the target (log/sqrt), or use **weighted least squares** with weights $\propto 1/\hat\sigma_i^2$.
- Correlated errors → GLS or use time-series methods.
- Multicollinearity → drop one of the pair, or use **Ridge regression**.
- Outliers → check them (are they errors or real signal?), or use **Huber regression** / **RANSAC**.

---

### 2.3 Regularization: Ridge, Lasso, ElasticNet

When you have many features (especially when $p$ approaches or exceeds $n$), OLS overfits — the coefficients grow large and unstable. **Regularization** adds a penalty on coefficient magnitude to shrink them toward zero.

**Ridge (L2) regression:**

$$\mathcal{L}_{\text{ridge}}(\boldsymbol\beta) = \frac{1}{2n}\|\mathbf{y} - X\boldsymbol\beta\|_2^2 + \frac{\lambda}{2}\|\boldsymbol\beta\|_2^2$$

Closed-form:

$$\boldsymbol\beta^*_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$$

**Lasso (L1) regression:**

$$\mathcal{L}_{\text{lasso}}(\boldsymbol\beta) = \frac{1}{2n}\|\mathbf{y} - X\boldsymbol\beta\|_2^2 + \lambda\|\boldsymbol\beta\|_1$$

No closed form; solved by coordinate descent or LARS. **Key property:** Lasso drives some coefficients exactly to zero → **automatic feature selection**.

**ElasticNet:** combines both, with a mixing ratio $\alpha \in [0, 1]$:

$$\mathcal{L}_{\text{en}} = \frac{1}{2n}\|\mathbf{y} - X\boldsymbol\beta\|_2^2 + \lambda\!\left(\alpha \|\boldsymbol\beta\|_1 + \frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2\right)$$

**Bias–variance intuition:**

- Larger $\lambda$ → more bias, less variance (coefficients shrink).
- $\lambda = 0$ → OLS.
- $\lambda \to \infty$ → all coefficients zero (intercept-only model).

**Critical practical note:** always **standardize features** (subtract mean, divide by std) before regularization. Otherwise, a feature measured in "millions" is penalized much less than one in "unit fraction" simply due to scale.

**Choosing $\lambda$:** cross-validation. `RidgeCV`, `LassoCV`, `ElasticNetCV` in scikit-learn do this automatically over a log-spaced grid.

---

### 2.4 Gradient Descent (When the Closed Form Isn't Practical)

For very large $n$ or $p$, forming and inverting $X^\top X$ ($O(p^3)$ time, $O(p^2)$ memory) is impractical. **Gradient descent** iteratively updates $\boldsymbol\beta$ in the direction of steepest descent:

$$\boldsymbol\beta^{(t+1)} = \boldsymbol\beta^{(t)} - \eta \, \nabla_\beta \mathcal{L}$$

Where **$\eta$** is the learning rate.

Gradient of MSE:

$$\nabla_\beta \mathcal{L} = -\frac{1}{n} X^\top(\mathbf{y} - X\boldsymbol\beta)$$

**Stochastic Gradient Descent (SGD)** uses one sample (or a mini-batch) per update:

$$\boldsymbol\beta^{(t+1)} = \boldsymbol\beta^{(t)} + \eta \, \mathbf{x}_i (y_i - \mathbf{x}_i^\top \boldsymbol\beta^{(t)})$$

Cheaper per step, noisier convergence — but for huge datasets, it dominates.

**Learning rate matters:**

- Too small → slow convergence.
- Too large → oscillation, divergence.
- **Adaptive** methods (AdaGrad, RMSprop, Adam) adjust per-parameter learning rates.

**Feature scaling** is critical for gradient descent — otherwise features with large ranges dominate the update direction.

**Convergence.** For convex problems (like OLS), gradient descent is guaranteed to reach the global minimum with a small enough step. For non-convex problems (neural nets, decision trees under greedy splits, etc.), only a local minimum.

---

### 2.5 Interpreting Coefficients & Inference

**Coefficient interpretation:** with a fitted $\hat\beta_j$,

> "Holding all other features constant, a one-unit increase in feature $j$ is associated with a $\hat\beta_j$-unit change in the predicted target."

Key words: **associated** (not "causes") and **holding others constant** (which only holds if features are truly independent — otherwise it's a mathematical construct).

**Standard errors and CIs.** Under the classical assumptions, the sampling distribution of $\hat{\boldsymbol\beta}$ is:

$$\hat{\boldsymbol\beta} \sim \mathcal{N}\!\left(\boldsymbol\beta,\; \sigma^2 (X^\top X)^{-1}\right)$$

Where $\sigma^2$ is estimated by the residual variance:

$$\hat\sigma^2 = \frac{1}{n - p - 1} \sum_{i=1}^n (y_i - \hat y_i)^2$$

The 95% CI for $\beta_j$ is $\hat\beta_j \pm t_{n-p-1, 0.975} \cdot \text{SE}(\hat\beta_j)$.

**p-value** for $H_0: \beta_j = 0$: `statsmodels`' `summary()` gives you these; scikit-learn does not.

**When to trust these numbers:**

- Linear model is approximately correct.
- Residuals are approximately normal (or $n$ is large).
- Errors are independent and homoscedastic.
- **Not** when you selected features by looking at the data — that inflates significance ("data snooping").

**Standardized coefficients.** If features have different scales, the raw $\hat\beta_j$ can't be compared across features. Fit on **standardized features** (each with mean 0, std 1), and $\hat\beta_j$ becomes the effect of a one-standard-deviation change in $x_j$, comparable across features.

**Goodness of fit.**

- **R² (coefficient of determination):**

  $$R^2 = 1 - \frac{\sum(y_i - \hat y_i)^2}{\sum(y_i - \bar y)^2}$$

  Proportion of variance explained. $\in [0, 1]$ for fitted models; can be negative for models worse than the mean.

- **Adjusted R²** penalizes for the number of features:

  $$R^2_{\text{adj}} = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}$$

- **RMSE / MAE** for prediction quality on new data.

---

## 3. Mental Models & Analogies

### 3.1 The "Best-Fit Line Through a Cloud" Model

Imagine a scatter of points in 2D. Linear regression finds the line that minimizes the **sum of squared vertical distances** from each point to the line. Not perpendicular distances (that's total least squares) — vertical, because $y$ is the "response" you're predicting from $x$.

Extend that to $p$ dimensions: instead of a line, it's a **hyperplane** in $\mathbb{R}^{p+1}$ minimizing squared vertical distances in the $y$ direction.

Now: **regularization** is the constraint that says "keep the plane tilted only as much as necessary". Ridge says "don't tilt any single dimension too hard"; Lasso says "and prefer axes with zero tilt." Both are ways of hedging against the temptation to fit noise.

### 3.2 The "Ingredient Weights in a Recipe" Model

Think of the target $y$ as the **taste rating** of a dish, and features as **ingredient amounts** ($x_1$ salt, $x_2$ butter, $x_3$ sugar, ...). The coefficients $\beta_j$ are the **contribution per gram** of each ingredient to the rating:

$$\text{rating} = \beta_0 + \beta_{\text{salt}} \cdot g_{\text{salt}} + \beta_{\text{butter}} \cdot g_{\text{butter}} + \cdots$$

- **$\beta_0$** — baseline rating for a dish with zero ingredients (usually not physically meaningful; often just the intercept).
- **$\beta_{\text{salt}} > 0$** — salt helps.
- **$\beta_{\text{salt}} < 0$** — salt hurts.
- **Multicollinearity** — if butter and cream are almost always used together in your dataset, the model can't tell whether butter or cream matters more; the coefficients become unstable.
- **Regularization** — a cook worried about overfitting to a few dishes says "don't attribute all the taste to one ingredient — spread the credit."

This is why linear coefficients are *powerful*: they translate directly into human-readable "how much each ingredient matters, holding the rest constant."

![IMG-NP-01](/2%20—%20Machine%20Learning/images/IMG-LR-01.png)

> **Caption:** OLS minimizes the sum of squared vertical residuals — in 2D it's a line, in higher dims a hyperplane.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Correlation = Causation via Coefficients"

A coefficient tells you **association**, not causation. If your model estimates $\hat\beta_{\text{ice-cream}} = 0.4$ on "drownings/day", ice cream doesn't cause drowning — summer confounds both. Even worse, when you *add* confounders to the model, the coefficient often changes direction ("Simpson's paradox in a linear regression"). Causal inference requires either **randomization** (experiments), **causal DAGs** with proper adjustment, or **instrumental variables** — not just running OLS.

### 4.2 "R² Is How Good the Model Is"

R² measures how much variance in $y$ is explained *on the training data*. Two failure modes:

1. **You can trivially inflate R² by adding features** — even useless ones. Use adjusted R² or, better, out-of-sample RMSE.
2. **A high R² on training + terrible R² on test = overfit.** Regularize or reduce features.

Also, a **low R² isn't necessarily bad**. If the underlying process is genuinely noisy (predicting stock returns, say), even a well-specified model may have R² = 0.02 — and still be useful.

### 4.3 "I'll Just Skip Feature Scaling for Linear Regression"

For **OLS**, scaling doesn't change predictions — the closed form is scale-equivariant. So people conclude scaling is optional. But:

- The **coefficients** aren't comparable across features unless standardized.
- **Regularization** is scale-dependent — an unscaled feature with a huge range is penalized less. Always scale before Ridge/Lasso.
- **Gradient descent** convergence is much better with scaled features.

Rule: use `StandardScaler` unless you have a specific reason not to.

---

## 5. Self-Assessment Bank (Linear Regression)

### Questions

**Q1 (Short answer).** Write the OLS loss function and its closed-form solution. Under what condition does the closed form exist?

**Q2 (Multiple choice).** Under which assumption does OLS give the maximum-likelihood estimator?
A. Uniform noise.
B. Laplace noise.
C. Gaussian noise.
D. Any noise, always.

**Q3 (Short answer).** Explain, in one paragraph, the difference between Ridge and Lasso and why Lasso can do automatic feature selection while Ridge cannot.

**Q4 (Multiple choice).** You add a new predictor to your regression and R² increases from 0.60 to 0.61 while adjusted R² decreases. What does this suggest?
A. The new predictor is highly significant.
B. The new predictor adds noise, not signal.
C. R² is broken.
D. You should always add it.

**Q5 (Short answer).** In what situation is `np.linalg.solve(X.T @ X, X.T @ y)` a bad idea, and what should you use instead?

**Q6 (Multiple choice).** If you fit `LinearRegression` on unscaled features, then compare coefficient magnitudes, what have you actually measured?
A. Feature importance.
B. A mixture of feature importance and feature scale.
C. Nothing meaningful.
D. R².

**Q7 (Short answer).** Why must you **standardize** features before Ridge regression, but not (strictly) before OLS?

**Q8 (Multiple choice).** In OLS, an observation with high **leverage** but low **residual** is:
A. Definitely an outlier that should be removed.
B. A point far from the mean of $X$ that happens to lie on the fitted line — usually not a problem.
C. A high-influence outlier.
D. Guaranteed to bias the model.

**Q9 (Short answer).** What is heteroscedasticity, why does it matter for inference (not prediction), and name one remedy.

**Q10 (Multiple choice).** Which of the following can be exactly zero in a Lasso solution but almost never in a Ridge solution?
A. R² on training data.
B. The intercept $\beta_0$.
C. A feature's coefficient $\beta_j$.
D. The residual sum of squares.

---

### Answer Key & Detailed Explanations

**A1.** Loss: $\mathcal{L}(\boldsymbol\beta) = \frac{1}{2n}\|\mathbf{y} - X\boldsymbol\beta\|_2^2$. Closed form: $\boldsymbol\beta^* = (X^\top X)^{-1} X^\top \mathbf{y}$. It exists when $X^\top X$ is invertible — i.e., $X$ has full column rank (no perfect multicollinearity, and $n \geq p+1$).

**A2. C.** With $y_i = \mathbf{x}_i^\top\boldsymbol\beta + \varepsilon_i,\ \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d., the log-likelihood is (up to constants) $-\frac{1}{2\sigma^2}\|\mathbf{y}-X\boldsymbol\beta\|^2$. Maximizing it is equivalent to minimizing squared error.

**A3.** Ridge penalizes $\|\boldsymbol\beta\|_2^2$; Lasso penalizes $\|\boldsymbol\beta\|_1$. Geometrically, the Lasso constraint region is a **diamond** with corners on the axes — the OLS ellipse hits the diamond at a corner (a coefficient exactly zero) with much higher probability than it would a Ridge sphere (no corners → coefficients shrunk but never zero). Practically: Lasso drives some $\hat\beta_j$ to exactly zero, giving sparse solutions; Ridge shrinks them all smoothly.

**A4. B.** Adjusted R² penalizes for adding features. If it decreases, the new predictor adds less signal than the penalty for the extra degree of freedom — most likely noise.

**A5.** When $X^\top X$ is ill-conditioned (near-singular). Small numerical errors get amplified massively. Use `np.linalg.lstsq(X, y, rcond=None)` (SVD-based, stable) or `scipy.linalg.lstsq`.

**A6. B.** Unscaled coefficients depend on the units. A "years of education" coefficient of 500 (target in dollars) and a "salary" coefficient of 0.01 (target in dollars) do NOT mean education matters 50,000× more — the units are just wildly different.

**A7.** OLS is scale-equivariant in the sense that scaling $x_j$ by a constant $c$ divides $\hat\beta_j$ by $c$ and leaves predictions unchanged. Ridge is not — the L2 penalty $\lambda \sum \beta_j^2$ penalizes coefficients uniformly regardless of feature scale, so a feature with a large range gets under-penalized. Standardizing puts all features on equal footing.

**A8. B.** **Leverage** measures distance in $X$-space; a high-leverage point *could* strongly influence the fit if its residual is also large. But if it lies on the fitted line (small residual), it's just an unusual $x$-value that happens to be consistent with the model. Cook's distance combines leverage and residual to flag actually-influential points.

**A9.** Heteroscedasticity means the variance of the residuals varies with $X$ or $\hat y$ (e.g., wider spread at higher predictions). Prediction is unbiased, but **standard errors** for $\hat{\boldsymbol\beta}$ are wrong under classical OLS assumptions — CIs and p-values are unreliable. Remedies: transform the target (log, sqrt), use **weighted least squares**, or use **heteroscedasticity-consistent standard errors** ("Huber-White" / "sandwich" estimator).

**A10. C.** Lasso's L1 penalty has non-differentiable corners at axes, and the KKT conditions produce exact-zero coefficients. Ridge's L2 penalty is smooth; coefficients shrink toward but rarely reach zero.

---

## 6. Practice Prompts

1. **From scratch.** Implement OLS via the normal equations *and* via gradient descent on a synthetic dataset. Verify both match `sklearn.linear_model.LinearRegression` to numerical tolerance.
2. **Coefficient diagnosis.** Simulate two correlated features ($\rho = 0.95$) and fit OLS. Repeat 100 times; plot the histogram of $\hat\beta_1$. Compare to Ridge with $\lambda = 1$. Watch variance drop.
3. **Regularization path.** Fit Ridge and Lasso over a log-spaced grid of $\lambda \in [10^{-3}, 10^3]$; plot each coefficient's path. Identify the $\lambda$ at which each Lasso coefficient hits zero.
4. **Diagnostics.** Fit OLS on a real dataset (Ames housing works well). Produce residuals-vs-fitted, QQ plot, VIF, and Cook's distance. Diagnose which assumptions look violated.
5. **Feature scaling matters.** Fit unscaled and scaled Lasso on the same data. Compare which features survive.

---

## 7. References

- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free PDF) — chapters 3 and 6.
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free PDF).
- Gelman, Hill & Vehtari, *Regression and Other Stories* — for the practical/inferential angle.
- scikit-learn docs: [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html).
