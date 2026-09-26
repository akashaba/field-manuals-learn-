# Logistic Regression — Master Study Guide

> **Track:** Machine Learning · **Module:** 02
> **Prerequisites:** Module 01 (Linear Regression), calculus (gradients), probability.
> **Time budget:** ~10–15 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Logistic regression is the workhorse of **binary classification**. It's fast, calibrated, interpretable, and — like linear regression — often good enough to beat fancy models on tabular data. If you've ever built a spam filter, a credit-scoring model, an ad-click predictor, or a churn model, chances are logistic regression was involved somewhere.

Beyond its practical value, logistic regression is the **bridge** between linear models and neural networks. It introduces:

- The **sigmoid** (aka logistic) function.
- The **cross-entropy** (aka log loss) as a loss function.
- **Maximum likelihood estimation** in a non-Gaussian setting.
- **Optimization by iterative methods** (no closed form here).

Every single one of those concepts is reused in deep learning. Mastering logistic regression means you already understand the last layer of most classification neural nets.

**Fundamental principles you must own:**

1. **The output is a probability, not a class.** Logistic regression predicts $P(y = 1 \mid \mathbf{x})$; the classification threshold is a separate decision.
2. **It models log-odds as a linear function of features.** That's what "linear" means here — not the output.
3. **The loss is cross-entropy**, which comes from maximum likelihood under a Bernoulli response.
4. **There is no closed form.** You optimize iteratively — usually via Newton's method (IRLS) or gradient descent.
5. **Regularization is essential** for feature-heavy problems.
6. **It's a linear classifier** — the decision boundary is a hyperplane. If your data isn't linearly separable in feature space, you either need engineered features or a nonlinear model.

If you retain nothing else: **logistic regression is linear regression through a sigmoid, trained to maximize likelihood.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Sigmoid, Log-Odds, and the Model

**The sigmoid (logistic) function:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma: \mathbb{R} \to (0, 1)$$

**Properties:**

- $\sigma(0) = 0.5$, $\sigma(z) \to 1$ as $z \to +\infty$, $\sigma(z) \to 0$ as $z \to -\infty$.
- Symmetric: $\sigma(-z) = 1 - \sigma(z)$.
- Its derivative has a beautiful form:

  $$\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$$

  This makes gradient computation cheap.

**The model.** For binary $y \in \{0, 1\}$:

$$\hat{p} = P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \boldsymbol\beta)$$

Rearranging:

$$\log\frac{\hat p}{1 - \hat p} = \mathbf{x}^\top \boldsymbol\beta$$

That left-hand side is the **log-odds** (or **logit**). So **logistic regression models the log-odds as a linear function of the features.**

**Interpretation of a coefficient $\beta_j$:**

- A one-unit increase in $x_j$ multiplies the **odds** by $e^{\beta_j}$.
- If $\beta_j = 0.7$, then $e^{0.7} \approx 2$ — a one-unit change in $x_j$ **doubles** the odds.
- If $\beta_j = -0.7$, odds are **halved**.

**The decision rule:**

- If $\hat p > 0.5$ → predict class 1.
- If $\hat p \leq 0.5$ → predict class 0.

But 0.5 is just the default. You'll often tune the threshold to balance precision/recall (see Module 11).

---

### 2.2 The Loss Function: Cross-Entropy from Maximum Likelihood

We model each label as a Bernoulli variable with parameter $p_i = \sigma(\mathbf{x}_i^\top\boldsymbol\beta)$:

$$P(y_i \mid \mathbf{x}_i, \boldsymbol\beta) = p_i^{y_i} (1 - p_i)^{1 - y_i}$$

**Likelihood** over all $n$ samples (assuming independence):

$$\mathcal{L}(\boldsymbol\beta) = \prod_{i=1}^n p_i^{y_i} (1 - p_i)^{1 - y_i}$$

Take the log (converts product to sum) and negate (turn maximization into minimization):

$$\boxed{\mathcal{J}(\boldsymbol\beta) = -\frac{1}{n} \sum_{i=1}^n \left[y_i \log p_i + (1 - y_i) \log(1 - p_i)\right]}$$

This is **binary cross-entropy**, also known as **log loss**.

**Why this loss and not MSE?** Two reasons:

1. **MSE with sigmoid outputs has non-convex regions** — gradient descent gets stuck.
2. **Cross-entropy is convex** in $\boldsymbol\beta$ (for logistic regression) and has a unique global minimum.

**Regularized log-loss.**

- **L2 (Ridge-style):**

  $$\mathcal{J}_{\text{L2}}(\boldsymbol\beta) = \mathcal{J}(\boldsymbol\beta) + \frac{\lambda}{2}\|\boldsymbol\beta\|_2^2$$

- **L1 (Lasso-style):**

  $$\mathcal{J}_{\text{L1}}(\boldsymbol\beta) = \mathcal{J}(\boldsymbol\beta) + \lambda\|\boldsymbol\beta\|_1$$

scikit-learn parameterizes with **$C = 1/\lambda$**, so smaller $C$ = stronger regularization. Note the direction is inverted.

---

### 2.3 The Gradient and Optimization

**Gradient of the cross-entropy loss:**

$$\nabla_\beta \mathcal{J} = \frac{1}{n} X^\top (\mathbf{p} - \mathbf{y})$$

Where **$\mathbf{p} = \sigma(X\boldsymbol\beta)$** applied element-wise. Note the elegant form: it's the same as linear regression's gradient with $\hat{\mathbf{y}}$ replaced by $\mathbf{p}$.

**Hessian** (second derivatives):

$$H = \frac{1}{n} X^\top W X, \quad W = \text{diag}(p_i(1-p_i))$$

$H$ is positive semi-definite → the loss is **convex** → any local minimum is global.

**Optimization methods:**

- **Gradient descent** (batch, mini-batch, stochastic) — the simplest.
- **Newton-Raphson** / **IRLS (Iteratively Reweighted Least Squares)** — uses the Hessian:

  $$\boldsymbol\beta^{(t+1)} = \boldsymbol\beta^{(t)} - H^{-1} \nabla_\beta \mathcal{J}$$

  Converges quadratically near the minimum. Standard method in `statsmodels`.
- **L-BFGS** — quasi-Newton, uses gradient info + a low-rank Hessian approximation. Default in scikit-learn.
- **SAG / SAGA** — stochastic methods that store previous gradients; good for very large datasets.

**Convergence in practice.** For well-conditioned problems, L-BFGS converges in tens of iterations. If it doesn't, either:

- Features aren't scaled.
- Data is (nearly) linearly separable — coefficients want to blow up to infinity; **regularization** fixes this.
- Too few iterations allowed (`max_iter` in scikit-learn).

---

### 2.4 Multiclass Extensions: One-vs-Rest & Softmax

**One-vs-Rest (OvR)** — fit $K$ binary classifiers, one per class ("is this class 1 vs everything else?"). Prediction: highest-scoring classifier wins. Simple, embarrassingly parallelizable, but the K models are trained independently — probabilities don't sum to 1 without renormalization.

**Multinomial (Softmax) logistic regression** — a single joint model. Each class gets its own coefficient vector $\boldsymbol\beta_k$. The probability of class $k$:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{x}^\top \boldsymbol\beta_k}}{\sum_{j=1}^K e^{\mathbf{x}^\top \boldsymbol\beta_j}}$$

This is the **softmax** function; it generalizes sigmoid to $K$ classes and guarantees probabilities sum to 1.

**Multiclass cross-entropy** (aka categorical cross-entropy):

$$\mathcal{J} = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K \mathbb{1}[y_i = k] \log P(y_i = k \mid \mathbf{x}_i)$$

**When to choose which:**

- Small number of classes (2–10) → softmax.
- Large number (100+) → OvR is often faster to train.
- Softmax is more theoretically clean when classes are mutually exclusive.

scikit-learn's `LogisticRegression(multi_class="multinomial")` uses softmax; `"ovr"` uses one-vs-rest.

---

### 2.5 Calibration and Threshold Selection

**Calibration** = do the predicted probabilities match the empirical frequencies? If your model outputs 0.8 for a bunch of samples, is the true positive rate among those samples actually ~80%?

Logistic regression is usually **well-calibrated out of the box** because its loss is proper. This is one of its underrated advantages over trees and SVMs, which produce miscalibrated scores.

**Calibration diagnostics:**

- **Reliability diagram** — bin predictions by confidence, plot predicted vs empirical rate.
- **Brier score** — mean squared error between predicted probabilities and true 0/1 labels.

**Threshold selection.** The default 0.5 threshold is *rarely* optimal. Considerations:

- **Class imbalance** — with 99% negatives, a naïve threshold predicts everything negative.
- **Different costs** — a false positive on cancer screening (unneeded biopsy) is not the same as a false negative (missed cancer).
- **Business objectives** — you want to catch 90% of fraud (fix recall at 0.9), then set threshold accordingly.

**Techniques:**

1. Plot **precision-recall curve** (or ROC curve) and pick the operating point.
2. Optimize a **utility function**: $U = \text{TP} \cdot v_{\text{tp}} - \text{FP} \cdot c_{\text{fp}} - \text{FN} \cdot c_{\text{fn}}$.
3. **Isotonic** or **Platt scaling** if the model is miscalibrated (rare for logistic regression; common for RF/XGBoost).

**Class-weighting.** For imbalanced data, scikit-learn's `class_weight="balanced"` reweights the loss:

$$\text{weight}_k = \frac{n}{K \cdot n_k}$$

where $n_k$ is the count of class $k$. This helps the model learn from minority-class examples.

---

## 3. Mental Models & Analogies

### 3.1 The "Squished Linear Regression" Model

Logistic regression is exactly linear regression whose output is passed through a **squishing function** (the sigmoid) that turns any real number into a probability. Everything you know about linear regression applies:

- Coefficients weight features.
- Regularization shrinks coefficients.
- Multicollinearity destabilizes them.

The only two differences:

1. The output is squished into $(0, 1)$.
2. The loss is cross-entropy, not MSE, because MSE + sigmoid is non-convex and unstable.

If a stakeholder asks "why is it *called* regression when it does classification?" — the answer is: because it *regresses* the log-odds on the features. It's a regression under the hood.

### 3.2 The "Betting Odds" Model

At a horse race, oddsmakers post **odds** (say, 3:1 in favor). This is the ratio $p / (1 - p)$. Logistic regression's central claim is:

> **Every additional unit of feature $x_j$ multiplies the odds of "class = 1" by $e^{\beta_j}$.**

If $\beta_{\text{is-fraud}}$ has $\beta_{\text{unusual-location}} = 2.3$, then a transaction from an unusual location has odds of fraud multiplied by $e^{2.3} \approx 10$. If the base odds are 1:1000, they become 1:100.

This turns interpretation into language humans natively speak: "this factor makes it 10× more likely." That's why regulators and business folks love logistic regression: it's a **probabilistic story** told in odds.

![IMG-NP-01](/2%20—%20Machine%20Learning/images/IMG-LG-01.jpg)
> **Caption:** Left — the sigmoid maps the linear score into a probability. Right — the decision boundary is a hyperplane in feature space.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Model Failed to Converge — Try Fancier Features"

If your dataset has **near-perfect separation** (e.g., a feature that on its own separates the classes almost perfectly), the maximum-likelihood solution has $\beta \to \infty$ (log-odds should be $\pm\infty$). Optimizers oscillate or fail to converge, and scikit-learn warns. The fix is not more features — it's **regularization**. Add L2 (`penalty="l2"`, `C` small) and the coefficients stay finite.

### 4.2 "0.5 Is the Threshold, End of Story"

On imbalanced data, 0.5 gives you either meaningless predictions ("everything is negative") or the wrong precision/recall trade-off for your use case. The threshold is **an operating decision, not a modeling decision**. Choose it by looking at the PR curve and picking the point that matches your business cost structure.

### 4.3 "Coefficients Are Feature Importance"

Not without care. Two traps:

1. **Feature scale matters** — a coefficient of 0.001 on "annual salary" is a bigger effect than 0.5 on "years", because a 1-unit change in salary is $1 versus 1 year. Standardize before comparing.
2. **Correlated features share credit** — if two features carry the same information, the coefficient gets split, and neither looks important. Multicollinearity destabilizes attribution but *not* prediction.

For proper importance, use standardized coefficients + confidence intervals, or model-agnostic methods (permutation importance, SHAP).

---

## 5. Self-Assessment Bank (Logistic Regression)

### Questions

**Q1 (Short answer).** Write the sigmoid function and its derivative in a form using $\sigma$ itself.

**Q2 (Multiple choice).** What quantity does logistic regression model as a linear function of features?
A. $P(y = 1 \mid \mathbf{x})$
B. $y$
C. $\log P(y = 1 \mid \mathbf{x})$
D. $\log \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})}$ (log-odds)

**Q3 (Short answer).** Derive the binary cross-entropy loss from the assumption that each label is Bernoulli with $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol\beta)$.

**Q4 (Multiple choice).** In scikit-learn's `LogisticRegression`, smaller `C` means:
A. Weaker regularization.
B. Stronger regularization.
C. More features.
D. Bigger dataset.

**Q5 (Short answer).** Explain, in one paragraph, why using MSE as the loss for logistic regression is a bad idea.

**Q6 (Multiple choice).** If your logistic regression fails to converge with a warning about the max iteration limit, the most likely fix is:
A. Increase `max_iter` blindly.
B. Standardize features and add L2 regularization.
C. Switch to a different model type.
D. Reduce the number of training samples.

**Q7 (Short answer).** Given $\hat\beta_{\text{unusual-location}} = 2.3$, interpret this coefficient in terms of odds.

**Q8 (Multiple choice).** For a 3-class problem with softmax logistic regression, what does the sum of predicted class probabilities equal, for any single sample?
A. 0
B. 0.5
C. 1
D. It depends on the sample.

**Q9 (Short answer).** What is a **reliability diagram**, and what would a well-calibrated model's reliability diagram look like?

**Q10 (Multiple choice).** Which of the following is **true** for logistic regression?
A. The loss is non-convex; you can get stuck in local minima.
B. The decision boundary is a hyperplane in the feature space.
C. It cannot output probabilities — only class labels.
D. It handles nonlinear class boundaries out of the box.

---

### Answer Key & Detailed Explanations

**A1.** $\sigma(z) = 1 / (1 + e^{-z})$; $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**A2. D.** Logistic regression models the **log-odds** (logit) as a linear function: $\log(p/(1-p)) = \mathbf{x}^\top\boldsymbol\beta$. The probability itself is a nonlinear function of the linear score.

**A3.** Under Bernoulli: $P(y_i \mid \mathbf{x}_i) = p_i^{y_i}(1-p_i)^{1-y_i}$. Likelihood: $\mathcal{L} = \prod P(y_i \mid \mathbf{x}_i)$. Negative log-likelihood: $-\log\mathcal{L} = -\sum[y_i \log p_i + (1-y_i)\log(1-p_i)]$. Dividing by $n$ gives the mean cross-entropy. Minimizing it = maximum likelihood estimation.

**A4. B.** scikit-learn parameterizes regularization strength as $C = 1/\lambda$; smaller `C` means larger $\lambda$, meaning stronger regularization.

**A5.** MSE on sigmoid outputs is non-convex, so gradient descent can get stuck. Its gradient with respect to $\boldsymbol\beta$ is proportional to $\sigma'(z)$, which is close to zero for large $|z|$ — meaning when the model is very confidently wrong, gradients vanish and learning stalls. Cross-entropy avoids both problems: it's convex, and its gradient stays linear in the error even when confident.

**A6. B.** Non-convergence is usually a numerical conditioning issue: unscaled features or near-perfect separation. Standardizing (via `StandardScaler`) and adding regularization keeps coefficients bounded and finite. Blindly upping `max_iter` may hide the real problem.

**A7.** Holding other features constant, being at an "unusual location" multiplies the odds of fraud by $e^{2.3} \approx 9.97$ — roughly 10×. If base odds were 1:100, they become ~1:10.

**A8. C.** Softmax always outputs a probability vector that sums to 1 by construction: $\sum_k P(y=k \mid \mathbf{x}) = 1$.

**A9.** A reliability diagram bins predictions by confidence (e.g., 10 bins of [0.0-0.1], [0.1-0.2], ...) and plots the mean predicted probability against the observed positive rate in each bin. A perfectly calibrated model produces the diagonal $y = x$. Systematic deviation above means underconfidence; below means overconfidence.

**A10. B.** The decision boundary is $\mathbf{x}^\top\boldsymbol\beta = 0$, a hyperplane. Logistic regression's loss is convex (A wrong); it *does* output probabilities via sigmoid/softmax (C wrong); and its boundary is linear, so it can't natively handle nonlinear separations (D wrong) unless you engineer nonlinear features.

---

## 6. Practice Prompts

1. **From scratch.** Implement logistic regression via gradient descent on a synthetic 2D binary dataset. Plot the decision boundary. Verify accuracy matches scikit-learn.
2. **Non-convergence lab.** Create a linearly separable dataset. Fit unregularized logistic regression; watch coefficients blow up. Add L2 regularization; watch them settle.
3. **Threshold tuning.** On an imbalanced dataset (10:1 negative:positive), fit LR and plot precision, recall, and F1 as functions of threshold. Pick the threshold that maximizes F1.
4. **Calibration.** Fit LR *and* a Random Forest on the same data. Compute Brier score for each; plot reliability diagrams. Confirm LR is better calibrated.
5. **Multiclass.** Fit softmax LR on the MNIST-tiny dataset. Confirm probabilities sum to 1 per sample. Compare to one-vs-rest.

---

## 7. References

- Bishop, *Pattern Recognition and Machine Learning*, chapter 4.
- Andrew Ng, CS229 lecture notes on logistic regression.
- Hastie, Tibshirani & Friedman, ESL chapter 4.
- scikit-learn docs: [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
- Gareth James et al., ISL chapter 4.
