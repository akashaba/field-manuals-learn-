# Regularization — Master Study Guide

> **Track:** Deep Learning · **Module:** 06
> **Prerequisites:** Modules 01–05.
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Regularization is **the collection of techniques that shrink the generalization gap** — the difference between training and validation performance. Modern neural networks are massively overparameterized; without regularization, they memorize training data and fail on new inputs. Every state-of-the-art model uses several regularizers stacked.

The rich diversity of regularization techniques reflects a deep fact: **overfitting is not one thing**. It can be small weights growing large, dead neurons, brittle feature detectors, unhelpful correlations. Each regularizer attacks a different mechanism.

**Fundamental principles you must own:**

1. **Regularization trades bias for variance.** You accept some fit-to-training-data quality to gain generalization.
2. **Multiple regularizers stack.** Weight decay + dropout + augmentation + early stopping are commonly combined.
3. **Data augmentation is often the highest-leverage regularizer** — it's free training data at zero labeling cost.
4. **Batch/layer normalization** are secretly regularizers on top of being training accelerators.
5. **Dropout was revolutionary in 2012 and remains important** for MLPs, RNNs, and smaller transformers.
6. **Modern large models (transformers) rely more on scale-appropriate architecture and data** than on classical regularization.

If you retain nothing else: **use weight decay + data augmentation + normalization + dropout by default. Add more only as diagnostics call for them.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Weight Decay (L2) and L1

**L2 weight decay** — penalize the squared norm of weights:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$

Or equivalently (in vanilla SGD) shrink the weight at each step:

$$\theta \leftarrow (1 - \eta \lambda) \theta - \eta \mathbf{g}$$

**Why it helps:**
- Small weights make the model **smoother** — small input changes → small output changes.
- Discourages **spurious correlations** with high weight magnitude.
- Interpretable as a Gaussian prior on weights: $p(\mathbf{w}) \propto \exp(-\lambda \|\mathbf{w}\|^2 / 2)$.

**L1 weight decay** — penalize absolute weights:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \|\mathbf{w}\|_1$$

- Drives some weights to **exactly zero** — sparsity → implicit feature selection.
- Less common in deep learning than L2, but useful when you want an interpretable, sparse model.

**Combination (elastic net):** $\lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$.

**Practical values:**
- CNNs: $\lambda \sim 10^{-4}$ (5e-4 is a canonical value for ImageNet).
- Transformers: $\lambda \sim 10^{-2}$ (0.01, common in BERT / GPT).
- The right value depends on your model size, dataset size, and learning rate. Tune it.

**Don't decay:** biases, LayerNorm/BatchNorm parameters, embeddings (usually). See Module 05 for the parameter grouping pattern.

**Weight decay ≠ L2 in Adam.** See Module 05.4. Use **AdamW** for correct weight-decay behavior with adaptive optimizers.

---

### 2.2 Dropout

**Dropout** (Srivastava et al., 2014) — during training, randomly zero each activation with probability $p$; scale the survivors by $1/(1-p)$ to preserve expected magnitude. At inference, no dropout — just use all activations.

$$\mathbf{h}_{\text{dropout}} = \mathbf{h} \odot \mathbf{m}, \quad \mathbf{m}_j \sim \text{Bernoulli}(1 - p)/(1-p)$$

**Why it works** (multiple interpretations):
- **Ensemble** — each iteration trains a different "subnetwork"; inference averages over an exponential number of them.
- **Feature co-adaptation prevention** — no single neuron can rely on any other being alive, so features must be individually robust.
- **Noise injection** — dropout is stochastic regularization; the model must be robust to random silencing.

**Where to apply dropout:**
- **After activations in hidden layers.**
- **Not on the input layer** typically (unless the input is very high-dim like text embeddings).
- **Before the final classifier** especially useful.
- **Inside transformer FFN layers** (residual dropout) — standard.
- **On attention weights** in some transformers.

**Common values:**
- $p = 0.2$–$0.5$ for MLPs and small CNNs.
- $p = 0.1$ for transformers (BERT / GPT-2 defaults).
- $p = 0.0$ increasingly common in very large models — data volume regularizes enough.

**PyTorch:** `nn.Dropout(p)`. Set `.train()` for training, `.eval()` for inference — this switches dropout on/off automatically.

**Variants:**

- **Dropout2d / SpatialDropout** — drops entire feature maps in CNNs (channels), not individual pixels. Better for convolutional inputs.
- **DropConnect** — drops individual weights instead of activations.
- **DropPath / Stochastic Depth** — drops entire residual blocks in deep networks. Used in modern vision transformers.

---

### 2.3 Batch Normalization and Layer Normalization

**Batch Normalization (BN)** (Ioffe & Szegedy, 2015) — for each feature (channel), normalize using the mini-batch's statistics:

$$\mu_B = \frac{1}{B}\sum_i x_i, \quad \sigma_B^2 = \frac{1}{B}\sum_i (x_i - \mu_B)^2$$
$$\hat x_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$
$$y_i = \gamma \hat x_i + \beta$$

Where $\gamma, \beta$ are learnable per-channel scale and shift.

**Benefits:**
- **Stabilizes training** — activations don't drift into pathological ranges.
- **Enables higher learning rates.**
- **Acts as a mild regularizer** — the noise from mini-batch statistics prevents overfitting.

**Gotchas:**
- **Training vs inference behavior differs.** During inference, uses running (exponential moving) averages of $\mu_B, \sigma_B$ from training. Buggy behavior if you forget `.eval()`.
- **Small batches hurt** — batch statistics become noisy, unstable.
- **Distributed training** — batch statistics are per-GPU by default; SyncBN synchronizes across devices for large-batch training.

**Layer Normalization (LN)** (Ba et al., 2016) — normalize over features within each sample:

$$\mu_i = \frac{1}{H}\sum_j x_{ij}, \quad \sigma_i^2 = \frac{1}{H}\sum_j (x_{ij} - \mu_i)^2$$

Same shift-and-scale formula, but the statistics are computed **per-example, over features**, not per-feature over batch.

**Advantages of LN:**
- No batch-size dependency.
- Works for RNNs (variable-length sequences).
- **Standard in transformers.**

**Group Norm** — a middle ground; normalize over groups of channels. Popular for small-batch CNN training.

**Placement in transformers:**
- **Post-norm** (original, "Attention Is All You Need"): $x + \text{LN}(\text{Attention}(x))$.
- **Pre-norm** (modern, more stable): $x + \text{Attention}(\text{LN}(x))$. Used in GPT-2/3/4, LLaMA. Better gradient flow, less need for warmup.

**Practical rule:**
- CNNs → BatchNorm.
- Transformers → LayerNorm (pre-norm).
- Small-batch training → GroupNorm.

---

### 2.4 Data Augmentation

**The most underrated regularizer.** Augment training data with label-preserving transformations, exposing the model to variation it will see in production.

**Image augmentation:**
- **Geometric:** random crop, flip, rotation, scale, translation.
- **Color:** brightness, contrast, saturation, hue shifts.
- **Cutout / random erasing** — zero out a random rectangle.
- **Mixup** — train on convex combinations $(\alpha \mathbf{x}_i + (1 - \alpha)\mathbf{x}_j, \alpha y_i + (1 - \alpha)y_j)$.
- **CutMix** — paste a patch from one image into another; mix labels proportionally.
- **AutoAugment / RandAugment** — learned or randomized augmentation policies. State-of-the-art for image classification.

**Text augmentation:**
- **Synonym replacement, back-translation** — semantic-preserving text edits.
- **Token dropout, token masking** — MLM-style noise.
- **Sentence shuffling, paraphrase generation.**
- Modern LLMs and instruction tuning rely heavily on augmentation via paraphrased data.

**Audio augmentation:**
- **SpecAugment** — mask time and frequency bands in the spectrogram.
- **Time stretch, pitch shift, noise injection.**

**General principles:**
- Augmentations must be **label-preserving** for the task. (Rotating a "6" 180° is a "9" — bad augmentation for digit classification.)
- Apply **only at training time**, not at inference.
- Test-time augmentation (TTA) — apply augmentations at inference and average predictions — can boost final accuracy.

**Effect on training curve.** Aggressive augmentation flattens the training-loss curve (harder problem) but improves validation — that's the goal.

---

### 2.5 Early Stopping, Label Smoothing, and Other Regularizers

**Early stopping** — halt training when validation loss stops improving. Use a "patience" of $N$ epochs.

```python
best_val = float("inf")
patience_counter = 0
for epoch in range(max_epochs):
    train_one_epoch()
    val_loss = evaluate()
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

- Effectively regularizes by preventing overfitting past the sweet spot.
- Roughly equivalent (in some analyses) to L2 with an appropriate $\lambda$.
- Free — you'd have monitored validation anyway.

**Label smoothing** — soften one-hot targets (see Module 04.2). Replace $y_k = 1$ with $1 - \varepsilon$ and 0s with $\varepsilon / K$. Prevents overconfidence, improves calibration.

**Weight tying** — share weights across parts of the network. Classic example: an autoencoder's decoder uses the encoder's transposed weights; language models tie the input embedding and output projection.

**Stochastic weight averaging (SWA)** — after most training, keep a running average of weight snapshots from later epochs. Often better generalization than the final checkpoint alone. `torch.optim.swa_utils` in PyTorch.

**Model ensembling** — train several models with different seeds/architectures and average predictions. The reliable-but-expensive path to +1-2% accuracy.

**Adversarial training** — augment training with adversarial examples (small perturbations designed to fool the model). Improves robustness (and, controversially, sometimes clean accuracy too).

**Dropout variants:** DropConnect, DropPath, Zoneout (for RNNs).

**Reducing capacity as regularization** — a smaller model can be a more effective regularizer than a heavier penalty. Match your model to your data.

**Practical stacking:** on a modest-sized problem, expect to use **weight decay + dropout + data augmentation + early stopping + LR schedule**. Each layer of regularization has a modest effect; combined they close large generalization gaps.

---

## 3. Mental Models & Analogies

### 3.1 The "Fitness Coach" Model

Overfitting is like a memorization-based approach to a fitness test — the model memorizes every exact rep in the practice videos and fails on any novel exercise.

Regularization is the coach's tricks to force actual fitness:

- **Weight decay** = don't let any muscle group get abnormally over-developed.
- **Dropout** = force randomly-selected muscles to sit out at each workout — the rest must pick up the slack. Every muscle must learn to work independently.
- **Data augmentation** = varied workouts, different angles, different equipment — instead of the same rep over and over.
- **Batch/layer norm** = normalize the intensity of every exercise so no single move dominates.
- **Early stopping** = go home before you overtrain.
- **Ensembling** = have five athletes each train separately and combine their strengths.

The moral: any single trick helps a little; combined, they build robust, general capability. In production, you never know exactly which exercises you'll be asked to do — better be broadly fit.

### 3.2 The "Chalk on Blackboard" Model (Bias-Variance)

A blackboard represents the space of functions your model class can express. Training data are the constraints — you must chalk up a function consistent with them.

- **No regularization:** the chalk can go anywhere the constraints allow, and there's a lot of "anywhere." Different training runs (or different training subsets) chalk very different pictures — high variance. But each fits the training data perfectly — low bias.
- **Weight decay:** you draw with a chalk that likes to stay near zero. Fewer wild pictures possible; more stable across training runs — lower variance. Slightly worse fit — mild bias.
- **Dropout:** every 5 seconds, half your chalks disappear and you have to reproduce your picture. You end up drawing pictures that survive random-erase — robust, not delicate. Lower variance.
- **Data augmentation:** the constraints multiply — instead of 100 constraint points, 10,000. Fewer wildly-different pictures satisfy all of them.
- **Early stopping:** stop drawing before you fill in every constraint's fine detail. Slightly rougher fit, more general.

The result of stacking regularizers: **you paint a robust picture across many random-erasure attempts.** That robustness is what generalization looks like.

![IMG-REG-01](/3%20—%20Deep%20Learning/images/IMG-REG-01.jpg)
> **Caption:** Regularization narrows the generalization gap; early stopping selects the sweet spot.
> **Placement:** Section 2.5 / Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Weight Decay Alone Is Enough"

For deep networks, no. Weight decay smooths individual weights but doesn't address feature co-adaptation, dead neurons, brittle features, or shape-specific overfitting. Modern practice stacks weight decay + dropout + batch/layer norm + data augmentation + LR scheduling.

### 4.2 "Dropout at Test Time"

Do not. Dropout is a training-time only technique. At inference:
- PyTorch's `nn.Dropout` module respects `.eval()` mode — sets $p$ effectively to 0.
- Forgetting `model.eval()` = using dropout at inference = unstable predictions across identical inputs.

Same for BatchNorm — `.eval()` uses running averages instead of batch stats. Every inference pipeline **must** start with `model.eval()`.

Exception: **Monte Carlo Dropout** — intentionally keep dropout on at inference and average many stochastic predictions for uncertainty estimation. Advanced use.

### 4.3 "More Regularization = Better Generalization"

Only up to a point. Over-regularized models underfit — training loss stays high, validation loss also high. The sweet spot depends on model capacity, data size, and task complexity. Signs of over-regularization:
- Training accuracy stagnates below expected.
- Validation accuracy plateaus with train accuracy — no gap, but at low levels.
- Loss curves flatten prematurely.

Fix: reduce dropout probability, reduce weight decay, less augmentation. Regularization is not a free lunch — it's a bias-variance trade.

---

## 5. Self-Assessment Bank (Regularization)

### Questions

**Q1 (Short answer).** State the bias-variance trade in one sentence and explain how weight decay fits in.

**Q2 (Multiple choice).** During training, `nn.Dropout(p=0.5)`:
A. Scales all activations by 0.5.
B. Randomly sets ~50% of activations to zero and scales survivors by 2× to keep expected magnitude.
C. Removes 50% of parameters.
D. Halves the loss.

**Q3 (Short answer).** Explain the difference between BatchNorm and LayerNorm — what's normalized over what, and when do you use each?

**Q4 (Multiple choice).** Data augmentation for image classification:
A. Is done at inference to improve robustness.
B. Applies label-preserving transformations to expand the effective training set at zero labeling cost.
C. Replaces the need for weight decay.
D. Doesn't affect final accuracy.

**Q5 (Short answer).** What is label smoothing, and what problem does it prevent?

**Q6 (Multiple choice).** Which is TRUE about BatchNorm at inference?
A. It uses the current batch's statistics.
B. It uses running (exponential moving average) statistics collected during training.
C. It behaves the same as during training.
D. It disables normalization entirely.

**Q7 (Short answer).** Why is Layer Normalization the standard choice for transformers rather than BatchNorm?

**Q8 (Multiple choice).** Early stopping is roughly equivalent to which regularizer, in effect?
A. Data augmentation.
B. Dropout.
C. L2 weight decay (both prevent excessive parameter movement).
D. Ensembling.

**Q9 (Short answer).** In a training run, you see training accuracy at 99% and validation accuracy at 70%. Name three regularization techniques that might narrow the gap.

**Q10 (Multiple choice).** Failing to call `model.eval()` before inference will:
A. Have no effect.
B. Still use dropout and use current-batch statistics in BatchNorm, producing unstable predictions.
C. Cause a NaN.
D. Only affect logging.

---

### Answer Key & Detailed Explanations

**A1.** The bias-variance trade says any model's expected error is bias² + variance + irreducible noise; reducing one usually increases the other. Weight decay increases bias (fits the training data slightly worse) but reduces variance (smaller weights → less sensitive to specific training samples) — often lowering the total error on unseen data.

**A2. B.** Inverted dropout (PyTorch's default): with $p = 0.5$, each element of the activation is zeroed with probability 0.5, and survivors are multiplied by $1/(1 - 0.5) = 2$ so that the expected activation is unchanged. At eval, no scaling is applied — but nothing is dropped either.

**A3.** **BatchNorm** normalizes each feature (channel) using statistics over the **minibatch dimension**. It requires batches; unstable for small batches; standard in CNNs. **LayerNorm** normalizes over the feature dimension **within each sample**. No batch dependency; works for variable-length sequences; standard in transformers and RNNs.

**A4. B.** Data augmentation applies transformations (crops, flips, color jitter, mixup, etc.) that don't change the label, effectively multiplying training data. Only applied at training. It's one of the highest-value regularizers because it directly injects the invariances the model should learn.

**A5.** Label smoothing replaces one-hot targets with a soft distribution: $y'_k = (1 - \varepsilon) \cdot y_k + \varepsilon/K$ for smoothing $\varepsilon$. It prevents the model from producing extreme (near-1.0) probabilities on any single class, discouraging overconfidence. Improves both generalization and probability calibration.

**A6. B.** During training, `BatchNorm` uses the current batch's mean and variance and updates running (EMA) statistics. At inference (`.eval()` mode), it uses those accumulated running statistics — the model behavior is now deterministic across identical inputs, and doesn't depend on the current batch.

**A7.** Transformers process variable-length sequences and are often trained with small effective per-device batch sizes. BatchNorm's mini-batch statistics would be noisy and inconsistent across positions/samples. LayerNorm normalizes each token's feature vector independently, so it's insensitive to batch size and sequence length. It also plays well with recurrent-like autoregressive generation.

**A8. C.** Early stopping halts training before parameters drift far from their initialization; L2 weight decay penalizes parameter magnitude, keeping them near zero. Both restrict how far parameters can move from a benign starting point, giving similar generalization benefits in many analyses.

**A9.** (1) **Data augmentation** — expand the effective training set. (2) **Dropout** — force robust features. (3) **Weight decay (AdamW)** — smooth the model. (4) **Early stopping** — halt at the validation optimum. (5) **Label smoothing** — reduce overconfidence. (6) **More training data** or a **smaller model**. Any three from this list would be a correct answer.

**A10. B.** Without `model.eval()`, dropout stays active (predictions randomly vary) and BatchNorm uses the current batch's statistics (which change with batch composition — even single-example predictions differ per batch). Every inference path must start with `model.eval()` and, for pure inference, wrap in `torch.no_grad()` for speed.

---

## 6. Practice Prompts

1. **Overfit-then-regularize.** Train a small CNN on MNIST *without* any regularization and *with* data augmentation + dropout + weight decay. Compare training/validation curves and final test accuracy.
2. **BatchNorm vs LayerNorm on transformers.** Take a small transformer training script; swap LayerNorm with BatchNorm1d. Observe the training instability.
3. **Mixup demo.** Implement Mixup ($\alpha = 0.2$) on a CIFAR-10 CNN. Compare final accuracy to no-Mixup.
4. **Early-stopping loop.** Write a training loop with early stopping (patience=5 on validation loss). Save the best model checkpoint. Confirm it selects a checkpoint before terminal training loss.
5. **Model eval mode audit.** Take any training loop; find every call to `model.eval()` and `model.train()`. Add a `print()` that warns if you predict without `eval()` being active.

---

## 7. References

- Srivastava et al., ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) (2014).
- Ioffe & Szegedy, ["Batch Normalization"](https://arxiv.org/abs/1502.03167) (2015).
- Ba et al., ["Layer Normalization"](https://arxiv.org/abs/1607.06450) (2016).
- Zhang et al., ["mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/abs/1710.09412) (2017).
- Loshchilov & Hutter, ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) (2019).
- Cubuk et al., ["AutoAugment"](https://arxiv.org/abs/1805.09501) (2019); RandAugment (2019).
- Goodfellow et al., *Deep Learning*, chapter 7.
