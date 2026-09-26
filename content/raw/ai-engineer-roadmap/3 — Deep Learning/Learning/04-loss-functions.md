# Loss Functions — Master Study Guide

> **Track:** Deep Learning · **Module:** 04
> **Prerequisites:** Modules 01–03.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** The loss function defines **what the model is optimizing for**. Every training run answers a very specific question: "which parameters minimize this expected loss on this data?" If the loss doesn't reflect your business objective, you'll train a technically excellent model that solves the wrong problem.

Choosing the loss is often a bigger decision than choosing the model. MSE trains for the mean; MAE trains for the median; cross-entropy trains for the posterior class distribution; contrastive losses train for embedding neighborhoods; focal loss trains for imbalanced problems. Each shapes the model's behavior profoundly.

**Fundamental principles you must own:**

1. **The loss is the training signal.** Every gradient traces back to $\partial \mathcal{L} / \partial \hat y$.
2. **The loss must match your output activation and task type.** MSE for unbounded regression; cross-entropy for classification; margin losses for retrieval.
3. **Many losses arise from maximum-likelihood estimation** under a probabilistic model. Understanding the assumed noise distribution reveals the loss.
4. **Weighting the loss can express business preferences** — cost-sensitive classification, focal loss, class weights.
5. **The scale of the loss doesn't matter directly** — but affects learning rate. Different loss scales need different LR tuning.
6. **Never train against a metric with zero gradient almost everywhere** (e.g., accuracy). Use a proxy loss (cross-entropy) whose optimization leads to good metric values.

If you retain nothing else: **the loss is the objective; the metric is the report. Sometimes they coincide, often they don't — match your training signal to the *decision* the model informs.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Regression Losses: MSE, MAE, Huber, Quantile

**MSE (Mean Squared Error / L2):**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat y_i)^2$$

- Derivable from **Gaussian noise MLE** — minimizing MSE = MLE assuming $y \sim \mathcal{N}(\hat y, \sigma^2)$.
- Predicts the **conditional mean** $E[y \mid \mathbf{x}]$.
- **Sensitive to outliers** — one point with residual 100 contributes as much as 10,000 points with residual 1.
- Convex, smooth — gradients are large where errors are large.
- Gradient with respect to prediction: $\partial\mathcal{L}/\partial \hat y_i = -2(y_i - \hat y_i)/n$.

**MAE (Mean Absolute Error / L1):**

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat y_i|$$

- Derivable from **Laplace noise MLE**.
- Predicts the **conditional median** $\text{median}(y \mid \mathbf{x})$.
- **Robust to outliers** — gradient magnitude is $\pm 1$ regardless of residual size.
- Non-differentiable at zero (use subgradients).

**Huber loss** — best of both:

$$\mathcal{L}_{\text{Huber}}(y, \hat y) = \begin{cases}\frac{1}{2}(y - \hat y)^2 & |y - \hat y| \leq \delta \\ \delta |y - \hat y| - \frac{1}{2}\delta^2 & |y - \hat y| > \delta\end{cases}$$

- Quadratic near zero (smooth), linear far from zero (robust to outliers).
- Hyperparameter $\delta$ — the crossover point (~1.0 default).
- Also known as **smooth L1 loss** in PyTorch.

**Quantile loss** (for quantile regression):

$$\mathcal{L}_\tau = \begin{cases}\tau(y - \hat y) & y > \hat y \\ (1 - \tau)(\hat y - y) & y \leq \hat y\end{cases}$$

- Predicts the $\tau$-quantile (e.g., $\tau = 0.9$ for the 90th percentile).
- Useful for uncertainty quantification — train models to predict multiple quantiles simultaneously.

**Log-cosh** — smooth approximation to Huber, differentiable everywhere.

**Which to use:**

- **No outliers, want fast convergence** → MSE.
- **Outliers to handle** → Huber or MAE.
- **Uncertainty intervals** → Quantile.
- **Heavy-tailed targets** → transform target with log/Box-Cox first, then MSE.

---

### 2.2 Classification Losses: Cross-Entropy Family

**Binary Cross-Entropy (BCE):**

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\right]$$

- Derivable from **Bernoulli MLE**.
- Requires $\hat p_i \in (0, 1)$ — apply sigmoid to raw logits first.
- Numerically stable version: `BCEWithLogitsLoss` = sigmoid + BCE fused with log-sum-exp.

**Categorical Cross-Entropy** (multiclass):

$$\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat p_{ik}$$

With one-hot $y_i$, this simplifies to $-\frac{1}{n}\sum_i \log \hat p_{i, y_i}$ — the negative log-probability of the correct class.

- Derivable from **categorical MLE**.
- Requires $\hat{\mathbf{p}}_i$ to sum to 1 — apply softmax to logits first.
- PyTorch: `CrossEntropyLoss` = softmax + NLL, takes raw logits.
- Gradient with respect to logits is elegantly $\hat p_k - y_k$.

**Sparse Categorical Cross-Entropy** — same as CE but with integer class labels instead of one-hot. In PyTorch, `CrossEntropyLoss` takes integer targets — no need to one-hot.

**Focal Loss** (Lin et al., 2017) — for **class imbalance** and **hard-example mining**:

$$\mathcal{L}_{\text{focal}} = -\alpha (1 - \hat p_t)^\gamma \log \hat p_t$$

Where $\hat p_t$ is the predicted probability of the true class, $\gamma > 0$ is the focusing parameter, $\alpha$ balances classes. The $(1 - \hat p_t)^\gamma$ term down-weights **easy examples** (where $\hat p_t$ is close to 1) and focuses training on hard examples.

- $\gamma = 0$ recovers ordinary CE.
- $\gamma = 2$ is a common choice.
- Widely used in object detection (RetinaNet).

**Label smoothing** — a regularization technique. Replace the one-hot target with:

$$y'_k = (1 - \varepsilon) \cdot y_k + \varepsilon / K$$

That is, assign $1 - \varepsilon$ to the true class and $\varepsilon / K$ to every class (including the true one). Smooths the target distribution, preventing overconfidence, and often improves generalization. Common $\varepsilon = 0.1$.

---

### 2.3 Metric Losses: Contrastive, Triplet, InfoNCE

**When to use:** when you want to learn an **embedding space** where similar items are close and dissimilar are far — the foundation of retrieval, face recognition, and self-supervised learning.

**Contrastive loss** (Hadsell et al., 2006):

$$\mathcal{L}_{\text{contrastive}} = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2$$

- **$y = 1$** if pair is similar, $0$ otherwise.
- **$d = \|\mathbf{h}_1 - \mathbf{h}_2\|_2$** — Euclidean distance between embeddings.
- **$m$** — margin: dissimilar pairs need distance > $m$ or they contribute loss.

**Triplet loss** (Schroff et al., 2015, FaceNet):

$$\mathcal{L}_{\text{triplet}} = \max(0, \|\mathbf{h}_a - \mathbf{h}_p\|^2 - \|\mathbf{h}_a - \mathbf{h}_n\|^2 + m)$$

Where $(\mathbf{a}, \mathbf{p}, \mathbf{n})$ is an **anchor, positive, negative** triplet. Anchor should be closer to positive than to negative, by at least margin $m$.

**InfoNCE / NT-Xent** (Oord et al., 2018; SimCLR, 2020) — the workhorse of modern self-supervised representation learning:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau)}$$

Where $\mathbf{h}_i^+$ is the positive pair and the denominator sums over one positive and many negatives. Effectively a **softmax cross-entropy over similarities to negatives**. Uses cosine similarity typically, with temperature $\tau$ (0.07 common).

Modern retrieval systems (dense retrievers for RAG, embedding models like `sentence-transformers`, CLIP) train with InfoNCE-family losses.

---

### 2.4 KL Divergence, Cross-Entropy, and Their Relationship

**KL divergence** between two distributions $p$ and $q$ over the same support:

$$\text{KL}(p \| q) = \sum_k p(k) \log \frac{p(k)}{q(k)} = \sum_k p(k) \log p(k) - \sum_k p(k) \log q(k)$$

$$= -H(p) + H(p, q)$$

Where:
- $H(p) = -\sum p(k) \log p(k)$ is the entropy of $p$.
- $H(p, q) = -\sum p(k) \log q(k)$ is the **cross-entropy** between $p$ and $q$.

**Key insight:** if $p$ is fixed (like a one-hot label or a target distribution), $H(p)$ is a constant. Minimizing $\text{KL}(p \| q)$ is equivalent to minimizing $H(p, q)$ — cross-entropy is the useful part.

**Asymmetric.** $\text{KL}(p \| q) \neq \text{KL}(q \| p)$ in general.

**Uses in DL:**

- **Model distillation** — student $q$ trained to match teacher $p$'s output distribution via $\text{KL}(p \| q)$.
- **Variational autoencoders** — KL term regularizes the encoder's posterior to match a prior.
- **RL / policy gradients** — KL constrains how far a new policy can drift from the previous one (PPO, TRPO).

**Jensen-Shannon divergence** — symmetric variant, bounded in $[0, \log 2]$.

**Wasserstein / Earth Mover's** distance — geometry-aware distance between distributions. Central to GAN training (Wasserstein GANs).

---

### 2.5 Loss Design in Practice: Scaling, Weighting, Curriculum

**Class weighting** for imbalanced classification:

```python
weights = torch.tensor([n_total / (K * count_k) for count_k in class_counts])
criterion = nn.CrossEntropyLoss(weight=weights)
```

The rarer a class, the higher its weight. Works well for moderate imbalance (10:1). For extreme imbalance (100:1+), consider focal loss or resampling.

**Sample weighting** — arbitrary per-sample weights (e.g., emphasize recent data):

```python
criterion = nn.CrossEntropyLoss(reduction="none")
losses = criterion(logits, y)  # shape (B,)
weighted_loss = (losses * sample_weights).mean()
```

**Multi-task losses** — sum of task-specific losses, often weighted:

$$\mathcal{L}_{\text{total}} = \sum_t w_t \mathcal{L}_t$$

Choosing $w_t$ is a modeling problem. Techniques:
- **Uncertainty weighting** (Kendall et al., 2018) — treat weights as learnable log-variances.
- **GradNorm** — dynamically balance gradient magnitudes across tasks.

**Curriculum learning** — start with easy examples, progressively add harder. Encoded as a schedule on the loss or the training set.

**Contrastive + reconstruction hybrids** — e.g., BERT's masked LM loss + next-sentence prediction; CLIP's contrastive + softmax over batch; VAE's reconstruction + KL.

**Practical tips:**

- Always **log the loss curve**. Training loss should decrease smoothly; validation loss should follow with some lag.
- **NaN loss** in the first few steps → learning rate too high, bad init, or a bug in loss computation.
- **Sudden loss spike** mid-training → data corruption in a specific batch, bad label, or numerical instability.
- **Loss doesn't move** → learning rate too small, dead neurons, disconnected graph, or wrong loss (accuracy has zero gradient almost everywhere).

---

## 3. Mental Models & Analogies

### 3.1 The "Compass Needle" Model

The loss is a **compass**. Every training step, the loss tells the model which direction to move to reduce error. But the compass points to whatever it's calibrated for — if you calibrate it to point to Chicago, it doesn't matter how good the compass is; you'll end up in Chicago.

- **MSE calibration:** the average outcome.
- **MAE calibration:** the median outcome.
- **Cross-entropy calibration:** the correct probability distribution.
- **Focal calibration:** hard-to-classify examples get extra pull.

**Your metric is the destination you actually care about.** Your loss is the compass. Sometimes they coincide (MSE = MSE). Often they don't (you care about F1; you train with cross-entropy because F1 has no useful gradient). When they diverge, be conscious of it — a compass pointing 5° off leads you 100 miles wide of the target.

### 3.2 The "Force Field" Model

Each training example exerts a **force** on the model parameters — pushing them to reduce that example's error. The gradient direction is the direction of the force; the gradient magnitude is the force's strength.

- **MSE:** force is proportional to the error (residual). Small errors → small force. Large errors → huge force. Outliers pull violently.
- **MAE:** force is constant regardless of error size. Every point pulls with the same strength.
- **Cross-entropy:** force is proportional to $\hat p - y$. Confidently wrong predictions ($\hat p \to 0$ for true class 1) pull very hard, via the log's steepness.
- **Focal loss:** force is downweighted for easy examples. Confident-correct examples exert almost no force; hard examples dominate.

This makes it clear why loss choice matters: **loss = force**. Whatever forces you compose determine the model's equilibrium behavior. Careful about "outliers" — under MSE, one bad label is a small army pulling the model into their story.

![IMG-LOSS-01](/3%20—%20Deep%20Learning/images/IMG-LOSS-01.jpg)
> **Caption:** MSE punishes large errors quadratically; MAE punishes linearly; Huber is a smooth combination.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Optimize Directly for the Metric"

You often can't. Accuracy has gradient 0 almost everywhere — a small parameter change either doesn't cross the decision threshold (no change in accuracy) or crosses it (a jump). No usable gradient signal.

The standard practice: **train on a smooth proxy loss** (cross-entropy) that has good gradients, and **evaluate on the metric you care about** (accuracy, F1, AUC). The training loss is a means; the metric is the end.

Exceptions where direct-metric optimization exists: differentiable AUC surrogates (rank losses), F1 relaxations, etc. These are research-level tools, rarely the right first move.

### 4.2 "Double-Applying Softmax to the Loss"

Same as Module 03's pitfall — `nn.CrossEntropyLoss` includes softmax internally with log-sum-exp for stability. Applying softmax before passing to the loss doubles it, mangling gradients. Always feed raw logits to `CrossEntropyLoss` and `BCEWithLogitsLoss`.

### 4.3 "MSE Works Fine, Even with Outliers"

MSE's gradient scales with the error magnitude. A single outlier with residual 100 contributes 10,000 to the loss and pulls the model toward itself with force proportional to 100. The model bends to satisfy the outlier at the cost of accuracy on everyone else. Symptoms: predictions shifted toward outlier values; validation loss dominated by tail behavior. Fix: outlier detection, target transform (log), or a robust loss (Huber, MAE).

---

## 5. Self-Assessment Bank (Loss Functions)

### Questions

**Q1 (Short answer).** Under what noise assumption is MSE the maximum-likelihood estimator? MAE?

**Q2 (Multiple choice).** In PyTorch, `nn.CrossEntropyLoss`:
A. Expects softmax'd probabilities as input.
B. Expects raw logits and applies softmax internally.
C. Only works for binary classification.
D. Requires one-hot targets.

**Q3 (Short answer).** Explain the relationship between KL divergence and cross-entropy, and why they're interchangeable for training when the target distribution is fixed.

**Q4 (Multiple choice).** Focal loss is designed to:
A. Speed up training.
B. Down-weight easy examples and focus on hard ones — useful for class imbalance.
C. Replace softmax with a differentiable step.
D. Enforce sparsity in weights.

**Q5 (Short answer).** Why can't you train directly on classification accuracy?

**Q6 (Multiple choice).** For a regression problem with heavy-tailed targets (occasional huge values), the best-behaved default loss is:
A. MSE.
B. MAE or Huber.
C. Cross-entropy.
D. InfoNCE.

**Q7 (Short answer).** Describe label smoothing and why it can improve generalization.

**Q8 (Multiple choice).** InfoNCE (contrastive loss) is essentially:
A. A margin loss.
B. Softmax cross-entropy over similarity scores of one positive vs many negatives.
C. A KL divergence.
D. MSE between embeddings.

**Q9 (Short answer).** What is label imbalance, and name two ways to handle it via the loss function (not resampling).

**Q10 (Multiple choice).** For a multi-label classification task (each label independent), the correct loss is:
A. Cross-entropy with softmax.
B. Binary cross-entropy per label (`BCEWithLogitsLoss`).
C. MSE.
D. Contrastive loss.

---

### Answer Key & Detailed Explanations

**A1.** **MSE** = MLE under Gaussian noise ($y_i = \hat y_i + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$). **MAE** = MLE under Laplace noise ($\varepsilon_i \sim \text{Laplace}(0, b)$). Different assumed noise distributions lead to different maximum-likelihood losses.

**A2. B.** `nn.CrossEntropyLoss` in PyTorch applies log-softmax internally in a numerically stable manner (log-sum-exp) and combines with NLL loss. You pass **raw logits** and **integer class targets** (not one-hot).

**A3.** KL divergence $\text{KL}(p \| q) = H(p, q) - H(p)$, where $H(p, q) = -\sum p(k) \log q(k)$ is cross-entropy and $H(p) = -\sum p(k) \log p(k)$ is entropy of the target. When $p$ is fixed (e.g., a one-hot label), $H(p) = 0$ is constant with respect to model parameters. Minimizing KL is thus equivalent to minimizing cross-entropy. They're interchangeable for training in this case.

**A4. B.** Focal loss reshapes cross-entropy: $-\alpha(1-\hat p_t)^\gamma \log \hat p_t$. The $(1 - \hat p_t)^\gamma$ term downweights well-classified examples (large $\hat p_t$) and lets the loss focus on hard, misclassified examples. It's particularly effective for object detection and severely imbalanced classification.

**A5.** Accuracy is a step function. A small parameter change either doesn't cross a decision boundary (no change in accuracy) or crosses one (a discrete jump). The gradient of accuracy with respect to parameters is zero almost everywhere and undefined at the crossings — useless for gradient descent. Cross-entropy is the smooth, differentiable proxy whose minima correspond to accurate models.

**A6. B.** MSE amplifies errors quadratically; extreme values dominate training. MAE (linear penalty) and Huber (quadratic near zero, linear beyond) are robust and give more stable training on heavy-tailed data. Alternatively, transform the target with $\log$ or $\text{yeo-johnson}$ and use MSE.

**A7.** Label smoothing replaces the one-hot target with a soft distribution: $y'_k = (1-\varepsilon)y_k + \varepsilon/K$. This prevents the model from becoming overconfident (predicting probability 1.0), makes gradients bounded, and improves calibration. Common $\varepsilon = 0.1$. Empirically improves test accuracy and calibration on classification tasks.

**A8. B.** InfoNCE is a softmax cross-entropy: numerator is $\exp(\text{sim}(h_i, h_i^+)/\tau)$, denominator sums over one positive and $N-1$ negatives. Maximizing this pushes the positive similarity high and negative similarities low — equivalently, it's a classification problem where the classes are "which sample was the positive." This turns unsupervised learning into softmax classification.

**A9.** Label imbalance: some classes are far more frequent than others. Loss-function-based approaches: (1) **Class weighting** — reweight each class's contribution inversely to its frequency (`nn.CrossEntropyLoss(weight=...)`). (2) **Focal loss** — down-weight easy examples so the loss focuses on the minority class implicitly. (3) **Label-distribution-aware margin (LDAM)** — enforce larger margins for minority classes.

**A10. B.** Multi-label = each label is independent (a sample can be both "cat" and "outdoors"). Softmax forces labels to compete (sum to 1) which is wrong. Sigmoid per output + BCE per output handles each label independently. In PyTorch: `nn.BCEWithLogitsLoss` on raw logits.

---

## 6. Practice Prompts

1. **Loss shapes.** Plot MSE, MAE, Huber (δ=1), and log-cosh as functions of residual for $r \in [-5, 5]$.
2. **Gradient inspection.** Compute $\partial \mathcal{L} / \partial \hat y$ analytically for MSE, MAE, cross-entropy (with sigmoid output). Compare to PyTorch's `.grad` on a small example.
3. **Class weighting.** On CIFAR-10 with an artificially imbalanced subset (100x more of class 0 than others), train a small CNN with (a) plain CE, (b) weighted CE, (c) focal loss. Compare per-class recall.
4. **Label smoothing effect.** Train a model on MNIST with `label_smoothing=0.0` and `label_smoothing=0.1`. Compare test accuracy and prediction entropy on test set.
5. **Triplet loss.** Implement triplet mining (hard/semi-hard) on a face-recognition-like toy problem. Plot the embedding space with t-SNE before and after training.

---

## 7. References

- Bishop, *Pattern Recognition and Machine Learning*, chapter 5.
- Lin et al., ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002) (2017).
- Chen et al., ["A Simple Framework for Contrastive Learning of Visual Representations"](https://arxiv.org/abs/2002.05709) (2020) — SimCLR, InfoNCE.
- Szegedy et al., ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567) (2015) — introduced label smoothing.
- PyTorch docs: [Loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions).
