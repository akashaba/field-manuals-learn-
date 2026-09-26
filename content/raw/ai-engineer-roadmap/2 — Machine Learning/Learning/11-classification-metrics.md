# Classification Metrics: Precision, Recall, F1, AUC — Master Study Guide

> **Track:** Machine Learning · **Module:** 11
> **Prerequisites:** Modules 02 (Logistic Regression), 10 (Splits & CV).
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** A model's "accuracy" is almost never the number you want. The metric you choose defines **what kind of mistake matters** and therefore **what the model learns to prioritize**. Picking the wrong metric leads to models that look great on paper and fail in the real world.

Consider:

- A fraud model that catches 3% of fraud but wakes up your entire ops team with false alarms — high recall, terrible precision.
- A cancer screener that has 99% accuracy — because 99% of tested patients don't have cancer, and it always predicts "no cancer."
- An ad-click predictor with AUC 0.85 — but the AUC is dominated by ordering the boring 95% of the traffic, and the model does terribly on the top 1% where actual money is made.

**Fundamental principles you must own:**

1. **Accuracy is misleading on imbalanced data.**
2. **Confusion matrix is the ground truth** — every other metric is a summary of it.
3. **Precision and recall trade off**, and where you sit is a **business decision**, not a modeling one.
4. **F1 is one summary; there are others (F-beta, MCC, balanced accuracy)**. Match the metric to the cost structure.
5. **ROC-AUC** measures ranking; **PR-AUC** measures precision-recall behavior. On imbalanced data, PR-AUC is usually more informative.
6. **Log-loss** rewards calibrated probabilities; use it when downstream decisions depend on probability magnitude, not just class labels.

If you retain nothing else: **choose the metric that matches your decision cost — never trust the default "accuracy".**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Confusion Matrix

For binary classification, the confusion matrix has four entries:

|                   | Predicted Positive | Predicted Negative |
|-------------------|-------------------|-------------------|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

Everything else is a function of these four numbers.

**Basic rates:**

- **Accuracy:**

  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

  Fraction of correct predictions. Misleading with class imbalance: if 99% of samples are negative, a model that always predicts negative scores 0.99 accuracy.

- **Precision** (aka Positive Predictive Value):

  $$\text{Precision} = \frac{TP}{TP + FP}$$

  Of the samples I predicted positive, what fraction actually are? Answers: "when I ring the alarm, how often am I right?"

- **Recall** (aka Sensitivity, True Positive Rate):

  $$\text{Recall} = \frac{TP}{TP + FN}$$

  Of the actually-positive samples, what fraction did I catch? Answers: "how many bad things did I catch?"

- **Specificity** (True Negative Rate):

  $$\text{Specificity} = \frac{TN}{TN + FP}$$

  Of the actually-negative samples, what fraction did I correctly ignore?

- **False Positive Rate:** $\text{FPR} = 1 - \text{Specificity} = FP / (FP + TN)$.

- **False Negative Rate:** $\text{FNR} = 1 - \text{Recall} = FN / (TP + FN)$.

**Concrete example.** A test set of 1000 samples, 50 truly positive. Model predicts 100 positive, of which 45 are TPs and 55 FPs.

- $TP = 45, FP = 55, FN = 5, TN = 895$.
- Accuracy = $(45 + 895)/1000 = 0.94$.
- Precision = $45/100 = 0.45$.
- Recall = $45/50 = 0.90$.
- Specificity = $895/950 = 0.942$.

A 94% "accuracy" hides a model that's wrong more than half the time when it rings an alarm.

---

### 2.2 F1 and F-Beta: Balancing Precision and Recall

**F1** is the harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

Harmonic mean pulls toward the smaller of the two numbers — you can't ace F1 unless *both* precision and recall are decent.

**F-beta** — general form that weights recall $\beta$ times more than precision:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}$$

- $\beta = 1$: F1 (balanced).
- $\beta = 2$: F2 — recall matters more (e.g., cancer screening: don't miss cases).
- $\beta = 0.5$: F0.5 — precision matters more (e.g., spam filter: don't flag legitimate mail).

**When to use F1 (and not accuracy):**

- Class imbalance where you care about the minority class.
- Both precision and recall are business-critical.

**Where F1 fails:**

- If **specificity** matters (true negatives are informative), F1 ignores TN entirely.
- With extreme imbalance and misaligned costs, F-beta with the right $\beta$ is better.

**Other summary metrics:**

- **Balanced accuracy** = $(\text{recall} + \text{specificity}) / 2$. Symmetric in classes; robust to imbalance.
- **Matthews Correlation Coefficient (MCC):**

  $$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

  In $[-1, 1]$; robust to imbalance; incorporates all four entries. Often the single best summary metric for imbalanced problems.

- **Cohen's kappa** — agreement rate corrected for chance.

---

### 2.3 ROC and AUC-ROC

**ROC (Receiver Operating Characteristic)** curve plots **True Positive Rate (recall)** on the y-axis against **False Positive Rate** on the x-axis as the classification threshold sweeps from high to low.

- **Threshold high** → few predictions positive → low TPR and low FPR → lower-left of curve.
- **Threshold low** → most predictions positive → high TPR and high FPR → upper-right.
- **Perfect model** → passes through (0, 1) — 100% TPR at 0% FPR.
- **Random model** → the diagonal, y = x.

**AUC-ROC** (Area Under the ROC Curve): a single number summary in $[0, 1]$.

- **AUC = 0.5** → random.
- **AUC = 1.0** → perfect ranking.
- **AUC = 0.85** → typical "good" model.

**Interpretation.** AUC-ROC equals the probability that the model ranks a randomly chosen positive sample above a randomly chosen negative sample:

$$\text{AUC-ROC} = P(f(\mathbf{x}^+) > f(\mathbf{x}^-))$$

This is a **ranking** metric — it's insensitive to the actual threshold and just cares about ordering.

**Advantages:**
- Threshold-independent.
- Compares different classifiers on ranking ability.
- Widely reported and understood.

**Limitations:**
- **Overly optimistic on imbalanced data.** With 99% negatives, the FPR denominator ($FP + TN$) is huge; even lots of FPs barely move FPR. AUC can be 0.95 while precision is 0.10.
- Doesn't tell you the operating point.
- Meaningless if the model is not intended to be used as a ranker (e.g., a fixed-threshold classifier).

---

### 2.4 PR Curve and PR-AUC

**PR (Precision-Recall) curve** plots **precision** (y) against **recall** (x) as the threshold sweeps.

- **Threshold high** → high precision, low recall (right end).
- **Threshold low** → low precision, high recall (left end).
- **Perfect** → passes through (1, 1).
- **Random baseline** → horizontal line at $y = \text{prevalence}$ (the positive class rate).

**PR-AUC** (or **Average Precision**, AP): the area under this curve. `sklearn.metrics.average_precision_score`.

**Why PR curves matter on imbalanced data.** Consider 99% negatives:

- ROC treats TN and FP symmetrically. FPR = 1% means 1% of the 99% → still a lot of FPs.
- PR curve: precision explicitly compares TP to FP. A model with 100 FPs vs 100 TPs has precision 0.5, regardless of how large the negative class is.

**Rule of thumb:**

- **Balanced data** → ROC-AUC is fine.
- **Imbalanced data** → PR-AUC is more informative.

**Baseline for PR-AUC:** $\text{prevalence}$. If your positive rate is 5%, a random model has PR-AUC ≈ 0.05. This is very different from ROC-AUC's baseline of 0.5.

---

### 2.5 Log-Loss, Brier Score, and Calibration

**Log-loss (Binary Cross-Entropy):**

$$\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log \hat p_i + (1 - y_i) \log(1 - \hat p_i)\right]$$

- Rewards **calibrated** probabilities: predicting 0.99 for a positive scores much better than predicting 0.51.
- Punishes overconfident wrong predictions severely (a prediction of 0.99 for a true 0 costs about $\log(0.01) \approx -4.6$).
- Use log-loss when downstream decisions use the **probability magnitude**, not just the label.

**Brier score:**

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^n (\hat p_i - y_i)^2$$

- MSE between predicted probabilities and true labels.
- Also rewards calibration; less sensitive to extremes than log-loss.

**Calibration.** A classifier is **well-calibrated** if among samples for which it predicts probability $p$, the observed positive rate is indeed $p$.

**Reliability diagram:**
- Bin predictions by $\hat p$ (e.g., 10 bins: [0.0-0.1], [0.1-0.2], ...).
- Plot mean predicted $\hat p$ vs empirical positive rate for each bin.
- Perfect calibration: diagonal $y = x$.
- Above diagonal: model is under-confident.
- Below: over-confident.

**Calibration fixes:**
- **Platt scaling** — fit a logistic regression from decision-function scores to $[0, 1]$. Assumes S-shaped miscalibration.
- **Isotonic regression** — a nonparametric monotone fit. More flexible; needs more data.
- Use `sklearn.calibration.CalibratedClassifierCV` — wraps any classifier and calibrates via internal CV.

**Which models need calibration?**

- **Logistic regression** — usually well-calibrated by default (its loss is proper).
- **Naive Bayes, RF, XGBoost, SVM** — often miscalibrated (biased toward extremes or 0.5 depending on model). Calibrate if you use probabilities downstream.

---

## 3. Mental Models & Analogies

### 3.1 The "Alarm Bell" Model (Precision vs Recall)

Imagine you're the security guard in a museum. Your job: press the alarm when a thief is in the gallery.

- **Precision** = "When I press the alarm, how often is there actually a thief?" If it's low, guards get called to false alarms all night. Guards get burnt out; when a real alarm rings, they ignore it.
- **Recall** = "Of all the thieves who came through tonight, how many did I catch?" If it's low, thieves walk away with paintings.

Every choice of alarm sensitivity is a trade-off:

- **Trigger-happy** → high recall (catch every thief) but low precision (constant false alarms).
- **Cautious** → high precision (every alarm is real) but low recall (some thieves get away).

**The right operating point depends on the museum**: a bank vault errs toward high recall; a supermarket theft detector errs toward high precision. You (the modeler) don't decide alone — the business does.

**F1** = "how good is my guard, weighing both sins equally?"
**F2** = "how good is my guard when missing a thief is twice as bad as a false alarm?"

### 3.2 The "Ranking Game" Model (AUC)

Suppose I hand you two envelopes: one contains a customer who will churn, one a customer who won't. Your model gives each a score. **AUC-ROC is the probability that your model gave a higher score to the churner.**

- AUC = 0.5 → your model is guessing. You get it right half the time.
- AUC = 0.85 → in 85% of trials, your model's ranking is correct.
- AUC = 1.0 → your model always ranks the churner higher.

This is the pure ranking signal. It doesn't ask about a threshold, doesn't ask about probability magnitudes. Just ordering.

**Why PR-AUC differs:** if churners are rare (say 3%), most envelopes contain non-churners, and your model has many chances to be almost-right by mistake. AUC-ROC forgives; PR-AUC doesn't.

![IMG-METRIC-01](/2%20—%20Machine%20Learning/images/IMG-METRIC-01.jpg)

> **Caption:** ROC vs PR — both threshold-sweep curves, but PR is more informative when the positive class is rare.
> **Placement:** Section 2.3–2.4.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Accuracy Is a Reasonable Default Metric"

Not on imbalanced data. A dataset with 99% negatives has a trivial baseline: always predict negative → 99% accuracy. Any real classifier's accuracy must beat this bar — usually by a much smaller margin than the raw number suggests. Consider balanced accuracy, F1, PR-AUC, or MCC whenever imbalance exists.

### 4.2 "ROC-AUC Is the Gold Standard"

On imbalanced data, it isn't. The x-axis of the ROC is FPR = $FP / (FP + TN)$. When $TN$ dwarfs $FP$, doubling $FP$ from 100 to 200 barely moves the ROC. The model looks great by ROC while its precision at operating point is terrible. **On imbalanced data, use PR-AUC.**

### 4.3 "0.5 Is the Right Threshold"

0.5 is the default in scikit-learn's `predict()`. It's the right threshold *only* when:
- The classifier is well-calibrated.
- False positives and false negatives are equally costly.
- Classes are balanced.

Almost never all three. Choose the threshold by optimizing your business metric (F-beta, custom utility, or the operating point where you hit a required recall). Then serve the model with that threshold, not 0.5.

---

## 5. Self-Assessment Bank (Classification Metrics)

### Questions

**Q1 (Short answer).** Define precision, recall, and specificity in terms of TP/FP/FN/TN.

**Q2 (Multiple choice).** A model achieves 99% accuracy on a dataset with 99% negatives. What's the most likely explanation?
A. The model is excellent.
B. The model always predicts the majority class.
C. The evaluation is broken.
D. The features are perfect.

**Q3 (Short answer).** When would you optimize F2 rather than F1?

**Q4 (Multiple choice).** For extreme class imbalance (0.5% positives), the more informative summary is:
A. Accuracy.
B. ROC-AUC.
C. PR-AUC (Average Precision).
D. Sensitivity alone.

**Q5 (Short answer).** State the probabilistic interpretation of ROC-AUC in one sentence.

**Q6 (Multiple choice).** Log-loss is minimized by:
A. Any classifier that gets the labels right.
B. Well-calibrated predicted probabilities.
C. High-confidence predictions regardless of correctness.
D. A random baseline.

**Q7 (Short answer).** What is calibration? Which classifiers are typically well-calibrated out of the box, and which are not?

**Q8 (Multiple choice).** Recall = 0.95 and Precision = 0.10. F1 =
A. 0.525
B. ~0.18
C. ~0.50
D. Cannot be computed.

**Q9 (Short answer).** Why is the "default threshold of 0.5" often wrong in imbalanced classification?

**Q10 (Multiple choice).** Balanced accuracy is:
A. `(precision + recall) / 2`
B. `(sensitivity + specificity) / 2`
C. `2 * precision * recall / (precision + recall)`
D. `TP / (TP + FP + FN + TN)`

---

### Answer Key & Detailed Explanations

**A1.**
- Precision = $TP / (TP + FP)$ — of predicted positives, fraction actually positive.
- Recall = $TP / (TP + FN)$ — of actual positives, fraction caught.
- Specificity = $TN / (TN + FP)$ — of actual negatives, fraction correctly rejected.

**A2. B.** With 99% negatives, always predicting negative yields 99% accuracy — often the "smart" baseline. Confusion matrix will show 0 TP. Always compare to a majority-class baseline before claiming a model works.

**A3.** When recall is more valuable than precision — for example, cancer screening (missing a positive is much more costly than a false positive that leads to a follow-up test) or fraud detection where every missed fraud costs money and false alarms are cheap to review. $\beta > 1$ emphasizes recall.

**A4. C.** PR-AUC (average precision) is sensitive to the trade-off between precision and recall. ROC-AUC is inflated on extreme imbalance because false positives don't move the FPR much when $TN$ is huge.

**A5.** ROC-AUC equals the probability that the classifier ranks a randomly chosen positive sample higher than a randomly chosen negative sample: $P(f(\mathbf{x}^+) > f(\mathbf{x}^-))$.

**A6. B.** Log-loss is a **proper scoring rule** — it's uniquely minimized by predicting the true probabilities. A classifier that predicts 0.99 for positives and 0.01 for negatives will have very low log-loss; a poorly calibrated classifier that predicts 0.51 vs 0.49 will have much higher log-loss even if labels are correct.

**A7.** Calibration = whether predicted probabilities match empirical frequencies (among samples predicted 0.7, is the true positive rate ~0.7?). Typically well-calibrated: **logistic regression**, well-trained neural nets on classification tasks. Typically miscalibrated: **RF** (biased toward 0.5), **XGBoost** (depends on loss), **SVM decision scores** (need Platt scaling), **Naive Bayes** (often over-confident). Fix with `CalibratedClassifierCV` or Platt/isotonic scaling.

**A8. B.** $F_1 = 2 \cdot \frac{0.10 \cdot 0.95}{0.10 + 0.95} = 2 \cdot \frac{0.095}{1.05} \approx 0.181$. Harmonic mean pulls the score toward the smaller of the two.

**A9.** With imbalance and typical model probability distributions, 0.5 may leave you with either near-zero positive predictions or too many. Also, false positives and false negatives usually have different costs. Choose the threshold by plotting the precision-recall curve and picking the operating point that matches business objectives (e.g., "we want recall ≥ 0.9, maximize precision").

**A10. B.** Balanced accuracy = $\frac{1}{2}(\text{sensitivity} + \text{specificity})$. It gives equal weight to each class regardless of size, so it's robust to imbalance. Option C is F1 (recall/precision harmonic mean).

---

## 6. Practice Prompts

1. **Confusion matrix arithmetic.** Given TP=100, FP=200, FN=50, TN=9650, compute accuracy, precision, recall, specificity, F1, MCC, and balanced accuracy by hand.
2. **ROC vs PR under imbalance.** Simulate a dataset with 1% positives. Fit a logistic regression. Plot ROC and PR curves. Note how good the ROC-AUC looks and how modest the PR-AUC is.
3. **Threshold sweep.** For a fitted classifier, plot precision, recall, and F1 as functions of threshold. Pick the threshold that maximizes F1; then the threshold that guarantees recall ≥ 0.9. Compare implications.
4. **Calibration.** Train RF and XGBoost on a binary task. Plot reliability diagrams. Wrap each in `CalibratedClassifierCV(method="isotonic")`; re-plot.
5. **Choose the right metric.** For a fraud dataset with 0.3% positives and asymmetric costs (missed fraud costs $500; false alarm costs $5 to review), compute the expected cost at multiple thresholds. Pick the threshold that minimizes cost. Compare to what F1 would have picked.

---

## 7. References

- Provost & Fawcett, ["Robust Classification for Imprecise Environments"](https://link.springer.com/article/10.1023/A:1007601015854) (2001) — ROC curves and cost-sensitive classification.
- Davis & Goadrich, ["The Relationship Between Precision-Recall and ROC Curves"](https://www.biostat.wisc.edu/~page/rocpr.pdf) (2006).
- Guo et al., ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599) (2017).
- Saito & Rehmsmeier, ["The Precision-Recall Plot Is More Informative than the ROC Plot"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) (2015).
- scikit-learn docs: [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
