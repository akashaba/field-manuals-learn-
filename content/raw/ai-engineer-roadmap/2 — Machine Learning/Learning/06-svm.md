# Support Vector Machines (SVM) — Master Study Guide

> **Track:** Machine Learning · **Module:** 06
> **Prerequisites:** Modules 01–02 (regression basics), linear algebra (dot products, norms).
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Support Vector Machines were the dominant ML method from the mid-1990s until deep learning eclipsed them for perception tasks. Even now, SVMs excel in:

- **Small-to-medium tabular datasets** where you can afford the $O(n^2)$–$O(n^3)$ training cost.
- **Text classification** with a linear kernel (competitive with modern methods on many benchmarks).
- **High-dimensional-few-sample** problems (bioinformatics, genomics).
- **When you need a rigorous max-margin classifier** with a specific theoretical guarantee.

More importantly, SVMs teach concepts that appear everywhere: **margin**, **kernels**, **support vectors**, **duality**, **convex optimization**. Understanding SVMs sharpens your intuition for regularization, non-linear feature spaces, and generalization theory.

**Fundamental principles you must own:**

1. **SVM finds the max-margin separating hyperplane.** Not just any separator — the one that leaves the biggest gap between classes.
2. **The margin is determined by a few "support vectors"** — the points closest to the boundary. Everything else is irrelevant to the model.
3. **Kernels enable nonlinear boundaries** by implicitly mapping features into higher-dimensional spaces.
4. **The problem is convex** — a unique optimal solution exists.
5. **The `C` parameter trades margin width for classification error** on training data. Smaller `C` = wider margin, more errors tolerated.
6. **SVMs don't output calibrated probabilities natively** — they output signed distances to the hyperplane. Use Platt scaling if you need probabilities.

If you retain nothing else: **SVM = maximize the margin, ignore everything except support vectors, and use kernels for nonlinearity.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Max-Margin Hyperplane

**Setup.** Binary classification with labels $y_i \in \{-1, +1\}$ and features $\mathbf{x}_i \in \mathbb{R}^p$. A hyperplane is $\mathbf{w}^\top \mathbf{x} + b = 0$; the decision function is:

$$\hat y = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$$

**The margin** is the perpendicular distance from the hyperplane to the closest point:

$$\text{margin} = \frac{1}{\|\mathbf{w}\|_2}$$

We want the widest margin subject to correctly classifying every point. **Hard-margin SVM** (assumes data is linearly separable):

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2 \quad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \; \forall i$$

Where:
- **$\mathbf{w}$** = normal vector to the hyperplane.
- **$b$** = bias.
- **$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$** means every point is at least distance $1/\|\mathbf{w}\|$ from the hyperplane on the correct side.

Minimizing $\|\mathbf{w}\|^2/2$ is equivalent to maximizing the margin $1/\|\mathbf{w}\|$ — same optimizer, easier arithmetic.

**Soft-margin SVM** (relaxes for non-separable data) — introduces slack variables $\xi_i \geq 0$:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^n \xi_i$$

subject to $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$.

Where:
- **$C > 0$** = penalty on violations. Large $C$ → few violations, narrow margin. Small $C$ → wide margin, more violations.
- **$\xi_i$** = how much sample $i$ violates the margin. $\xi_i = 0$ means it's outside the margin; $0 < \xi_i < 1$ means it's inside the margin but on the right side; $\xi_i > 1$ means it's misclassified.

This is exactly the **hinge loss + L2 regularization** formulation:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^n \max(0,\; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$$

The hinge loss $\max(0, 1 - y \cdot f(\mathbf{x}))$ is zero for correctly classified points beyond the margin, and linear otherwise. This is what makes SVMs *sparse* in the dual — only points near or inside the margin have nonzero weight.

---

### 2.2 The Kernel Trick

**The insight.** Everywhere in the SVM formulation, feature vectors appear only inside **dot products** $\mathbf{x}_i^\top \mathbf{x}_j$. If we could replace those dot products with a function $K(\mathbf{x}_i, \mathbf{x}_j)$ that computes the dot product *in a higher-dimensional space* $\phi(\mathbf{x})$ — without ever computing $\phi$ explicitly — we get nonlinear boundaries for free.

**A kernel** $K(\mathbf{x}, \mathbf{z})$ is a function such that $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top \phi(\mathbf{z})$ for some feature map $\phi$. Mercer's theorem tells us when such a $\phi$ exists (roughly, when $K$ is positive semi-definite).

**Common kernels:**

- **Linear:** $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^\top \mathbf{z}$. No implicit mapping — equivalent to plain SVM in original space. Fast, good for high-dim sparse data.
- **Polynomial:** $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^\top \mathbf{z} + r)^d$. Includes cross-terms up to degree $d$.
- **RBF (Gaussian):** $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$. Infinite-dimensional implicit feature space. **Most common default.**
- **Sigmoid:** $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^\top \mathbf{z} + r)$. Historically used to mimic neural networks; not always positive definite; rarely used now.

**The RBF kernel deserves special attention.** Its $\gamma$ parameter controls the "reach" of each support vector:

- **Small $\gamma$** — smooth decision boundary; each SV influences many points.
- **Large $\gamma$** — jagged boundary; each SV influences only nearby points; risk of overfitting.

Rule of thumb: $\gamma = 1 / (p \cdot \text{Var}(X))$ is a reasonable default (scikit-learn's `gamma="scale"`).

**Computational cost.** With a kernel, the SVM's dual formulation requires the $n \times n$ **Gram matrix** $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ — $O(n^2)$ memory, $O(n^2)$ to $O(n^3)$ training time. This is why SVMs don't scale to millions of rows.

---

### 2.3 The Dual Problem and Support Vectors

The Lagrangian dual of soft-margin SVM is:

$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

subject to $\sum_i \alpha_i y_i = 0$ and $0 \leq \alpha_i \leq C$.

**Where:**
- **$\alpha_i \geq 0$** = Lagrange multipliers, one per training point.

**Key insight — KKT conditions** classify each training point into three categories:

- **$\alpha_i = 0$** — the point is outside the margin (correctly classified, well away). **Not a support vector.** These points can be removed without changing the model.
- **$0 < \alpha_i < C$** — the point is exactly on the margin. **A "free" support vector.**
- **$\alpha_i = C$** — the point is inside the margin or misclassified. **A "bounded" support vector.**

**Prediction** uses only the support vectors (points with $\alpha_i > 0$):

$$f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$

**This sparsity is a major SVM feature.** Only a fraction of training points end up as SVs; prediction cost is proportional to the number of SVs, not the training-set size.

**Estimating** $b$ can be done from any free support vector (KKT: $y_j(\sum_i \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}_j) + b) = 1$).

---

### 2.4 Multiclass, Regression, and One-Class SVMs

**Multiclass.** SVMs are inherently binary. Extensions:

- **One-vs-Rest (OvR)** — $K$ classifiers, "is this class $k$ or not?" Fast; slightly biased on imbalanced classes.
- **One-vs-One (OvO)** — $K(K-1)/2$ classifiers, one per pair. Each classifier trained on only the two classes' data. Voted. scikit-learn's `SVC` uses this by default.
- **Crammer-Singer** — a joint multiclass formulation. Rare in practice.

**Support Vector Regression (SVR).** Uses an **$\epsilon$-insensitive** loss: no penalty if the prediction is within $\epsilon$ of the target; linear penalty beyond. This produces a "tube" of width $2\epsilon$ around the regression function, with SVs on or outside the tube.

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \max(0, |y_i - f(\mathbf{x}_i)| - \epsilon)$$

**One-Class SVM.** Learns the "support" of the data — trained on positives only, predicts positive for points that look like training data and negative for outliers. Useful for **novelty/anomaly detection**.

---

### 2.5 Practical Considerations

**Scale your features.** SVMs (especially RBF) are distance-based. Unscaled features destroy the kernel's meaning. Always use `StandardScaler` (or `MinMaxScaler`) before SVM.

**Choosing kernel:**
- Text or very-high-dim data → **linear**. Use `LinearSVC` or `SGDClassifier` — orders of magnitude faster than `SVC(kernel='linear')`.
- Small dataset, unknown structure → **RBF**. Tune `C` and `gamma`.
- Cannot beat a linear boundary → RBF likely won't help either; the data may not have exploitable nonlinearity.

**Tuning `C` and `gamma`.** Grid search over log-spaced values, e.g., $C \in \{10^{-2}, 10^{-1}, 1, 10, 100\}$, $\gamma \in \{10^{-3}, ..., 10^{1}\}$. Use CV. Nested CV if you need honest estimates on very small data.

**Probabilities.** `SVC(probability=True)` fits **Platt scaling** on top: a logistic regression from decision-function scores to $[0, 1]$. Slow (does 5-fold CV internally); results are approximate. Prefer logistic regression if you need calibrated probabilities.

**Class imbalance.** Use `class_weight="balanced"` — reweights $C$ per class inversely to class frequency.

**Interpretability.** SVMs are less interpretable than linear/logistic regression (no coefficient story) and much less interpretable than trees. Focus on cross-validated performance rather than interpretation.

**Deprecation.** For large-scale datasets, SVMs have been largely replaced by gradient boosting or neural networks. Keep them in your toolbox for smaller problems and for fundamental understanding.

---

## 3. Mental Models & Analogies

### 3.1 The "Road Between Cities" Model

Imagine two cities on a map (two classes of points). You want to draw a **road** between them that:

- Doesn't touch either city.
- Is as **wide** as possible (largest safety margin).

The middle line of the road is the SVM's decision boundary. The cities' *closest buildings* on each side — the ones just barely off the edge of the road — are the **support vectors**. Every other building is irrelevant: you could remove it from the map and still draw the same road.

**Soft margin** = allowing a few buildings inside the road (with a fine per foot inside). The fine is $C$: high $C$ means the government hates any road encroachment (narrow road but no encroachments); low $C$ means it tolerates encroachment (wider road, some buildings in it).

**Kernel** = a topographic mapmaker who says "flat maps can't fit a road here — I'll re-project the map into 3D, and then a plane can cleanly separate the cities." You never see the 3D map, only the resulting 2D road that looks curved.

### 3.2 The "Feature Space Warp" Model (Kernels)

Imagine points on a 2D plane where blue and red are arranged in concentric rings — no straight line separates them. Now imagine **lifting** the plane into 3D so that blue points rise higher than red points (perhaps proportional to their distance from the center). Suddenly a *horizontal* plane in 3D cleanly separates them.

That vertical lift is $\phi(\mathbf{x}) = [x_1, x_2, x_1^2 + x_2^2]$. In the lifted space, a linear boundary works. When you project back to 2D, the boundary looks like a circle.

The **kernel trick** is that you never have to actually build the 3D lift. Kernel-based algorithms only ever compute dot products, and there's a shortcut ($K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top \phi(\mathbf{z})$) that computes the dot product in the lifted space directly from the original coordinates. So you get all the expressive power of the lift with none of the compute of building it.

![IMG-RF-01](/2%20—%20Machine%20Learning/images/IMG-SVM-01.jpg)

> **Caption:** SVM finds the max-margin hyperplane. Support vectors are the boundary points; kernels lift data into a higher dimension where linear separation becomes possible.
> **Placement:** Section 2.1–2.2.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "SVMs Are Magic for Nonlinear Data"

Only with the right kernel and tuning. An RBF SVM with default hyperparameters on unscaled features often performs *worse* than logistic regression. The kernel doesn't automatically pick the right nonlinearity — you have to specify `gamma` (and the kernel choice). Tuning is essential; grids over log-spaced $C$ and $\gamma$ are non-negotiable.

### 4.2 "SVM Outputs Are Probabilities"

The decision function is a signed distance to the hyperplane — not a probability. `SVC(probability=True)` fits a Platt scaling layer on top with internal 5-fold CV — slow, and the resulting probabilities are approximate. If you need probabilities, use logistic regression; if you need calibrated SVM probabilities specifically, use `CalibratedClassifierCV` with `method="isotonic"` or `"sigmoid"`.

### 4.3 "SVMs Scale to Any Dataset"

Nope. Kernel SVMs require an $n \times n$ Gram matrix in memory and take $O(n^2)$–$O(n^3)$ to train. Beyond ~$10^5$ samples, they're infeasible. Alternatives:

- **`LinearSVC`** — linear kernel, uses primal formulation, scales to $10^7$+ samples.
- **`SGDClassifier(loss='hinge')`** — SVM via stochastic gradient descent. Massive scale.
- **Kernel approximations** (`Nystroem`, `RBFSampler`) — approximate nonlinear kernels with random features, then fit a linear model on the transformed data.

---

## 5. Self-Assessment Bank (SVM)

### Questions

**Q1 (Short answer).** What does "max margin" mean, and why do we want it?

**Q2 (Multiple choice).** In soft-margin SVM, the parameter $C$ controls:
A. The width of the RBF kernel.
B. The tradeoff between margin width and classification error on training data.
C. The number of support vectors.
D. The polynomial degree.

**Q3 (Short answer).** Explain the "kernel trick" in one paragraph. Why is it powerful?

**Q4 (Multiple choice).** After training an SVM, which training samples influence the decision function?
A. All of them equally.
B. Only the support vectors (samples with $\alpha_i > 0$).
C. Only correctly classified samples.
D. Only misclassified samples.

**Q5 (Short answer).** Why is feature scaling essential for RBF SVM?

**Q6 (Multiple choice).** The RBF kernel's $\gamma$ parameter, when set very large, tends to produce:
A. Underfit models with wide, smooth boundaries.
B. Overfit models with jagged boundaries that fit training noise.
C. Linear boundaries.
D. Sparser support vector sets.

**Q7 (Short answer).** What is the hinge loss, and how does it relate to the soft-margin SVM formulation?

**Q8 (Multiple choice).** For a text classification problem with 100k features and 500k samples, the practical SVM choice is:
A. `SVC(kernel='rbf')` — best performance.
B. `LinearSVC` or `SGDClassifier(loss='hinge')` — linear + scalable.
C. `SVC(kernel='poly', degree=5)`.
D. Don't use SVM.

**Q9 (Short answer).** Why doesn't `SVC.decision_function()` return probabilities, and how can you get them?

**Q10 (Multiple choice).** Compared to logistic regression, SVMs typically:
A. Have calibrated probabilities.
B. Are faster to train on large data.
C. Focus on samples near the decision boundary; distant samples don't affect the model.
D. Are more interpretable via coefficients.

---

### Answer Key & Detailed Explanations

**A1.** The margin is the perpendicular distance from the hyperplane to the closest training points. Max margin means finding the hyperplane that maximizes this distance. Intuition: a wider margin means the classifier is more "confident" and more robust to small perturbations of the data — statistical learning theory shows that larger margins give tighter generalization bounds.

**A2. B.** $C$ trades off the two terms in the soft-margin objective: $\frac{1}{2}\|\mathbf{w}\|^2$ (margin regularization) vs $C \sum_i \xi_i$ (violation penalty). Large $C$ = strong penalty on violations = narrow margin, few errors. Small $C$ = weak penalty = wide margin, more errors tolerated.

**A3.** SVM's algorithm only ever computes dot products between feature vectors. A kernel $K(\mathbf{x}, \mathbf{z})$ is a function that returns the dot product $\phi(\mathbf{x})^\top \phi(\mathbf{z})$ in some (potentially infinite-dimensional) feature space $\phi$, without ever computing $\phi$ explicitly. This lets us fit nonlinear decision boundaries in the original space by fitting a linear one in the transformed space — all while paying only $O(n^2)$ compute for the Gram matrix, not the cost of the (possibly infinite) $\phi$ space.

**A4. B.** Support vectors (samples with $\alpha_i > 0$) are the only ones whose weights appear in the prediction formula. All other samples have $\alpha_i = 0$ and could be removed from training without changing the model.

**A5.** RBF (Gaussian) kernel is $\exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$. The norm $\|\mathbf{x} - \mathbf{z}\|$ is dominated by features with the largest scale. An unscaled feature with range 0–10000 completely swamps a feature with range 0–1. The result is that only the large-scale feature matters, and the kernel behaves as if the small ones are noise. Always standardize.

**A6. B.** Large $\gamma$ means each support vector's influence decays quickly with distance ($\exp(-\gamma d^2)$ is a narrow bell curve). The decision boundary becomes a collection of small bumps around each SV — overfitting the training data. Small $\gamma$ = wide bells = smooth, potentially underfit boundaries.

**A7.** Hinge loss: $L(y, f(\mathbf{x})) = \max(0, 1 - y \cdot f(\mathbf{x}))$ with $y \in \{-1, +1\}$. It's zero when the sample is correctly classified with margin ≥ 1, and grows linearly otherwise. Soft-margin SVM is exactly $\min \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i L(y_i, \mathbf{w}^\top \mathbf{x}_i + b)$ — L2 regularization + hinge loss.

**A8. B.** Kernel SVMs (`SVC(kernel='rbf')`) require $O(n^2)$ memory for the Gram matrix — $500k^2 \approx 2.5 \cdot 10^{11}$ entries, infeasible. Linear formulations (`LinearSVC`, `SGDClassifier`) use the primal and scale linearly in $n$.

**A9.** `decision_function` returns the signed distance to the hyperplane: negative for one class, positive for the other, with magnitude reflecting confidence but not a probability. To get probabilities, set `probability=True` (fits Platt scaling internally with 5-fold CV — slow) or wrap with `CalibratedClassifierCV(method="isotonic")`.

**A10. C.** SVMs are inherently *sparse* — only support vectors matter. Logistic regression uses all samples. On calibration (A), logistic wins. On speed (B), logistic often wins too. On interpretability (D), linear logistic regression's coefficients tell a cleaner story than an SVM's dual coefficients (though `LinearSVC` also has coefficients).

---

## 6. Practice Prompts

1. **From scratch (linear).** Implement soft-margin linear SVM via gradient descent on the hinge loss. Verify it matches `LinearSVC` on a small dataset.
2. **Kernel exploration.** On the "two moons" dataset, fit SVM with linear, polynomial (degree=3), and RBF kernels. Plot decision boundaries.
3. **`C` and `gamma` grid.** Grid search over $C \in \{0.01, 0.1, 1, 10, 100\}$ and $\gamma \in \{0.01, 0.1, 1, 10\}$. Plot validation accuracy as a heatmap. Note the "sweet spot" region.
4. **Support vector count.** For RBF SVMs at various $C$ and $\gamma$, count the number of support vectors. When does the SVM start memorizing (SVs ≈ n)?
5. **Kernel approximation.** Replace `SVC(kernel='rbf')` with `Pipeline([('rbf', RBFSampler()), ('linear', LinearSVC())])`. Compare accuracy and training time on a moderately-sized dataset.

---

## 7. References

- Cortes & Vapnik, ["Support-Vector Networks"](https://link.springer.com/article/10.1007/BF00994018) (1995) — the foundational paper.
- Bernhard Schölkopf & Alexander Smola, *Learning with Kernels* (comprehensive).
- ISL chapter 9.
- ESL chapter 12.
- scikit-learn docs: [SVMs](https://scikit-learn.org/stable/modules/svm.html).
