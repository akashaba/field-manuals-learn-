# Optimizers — Master Study Guide

> **Track:** Deep Learning · **Module:** 05
> **Prerequisites:** Modules 01–04.
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** The optimizer converts **gradients into weight updates**. Every deep-learning framework hides most of this from you; every state-of-the-art model depends on the right choice. The wrong optimizer + hyperparameters = a model that either diverges or trains at 1/10th the achievable speed.

**Fundamental principles you must own:**

1. **All optimizers are variations on gradient descent** — take the gradient, take a step opposite to it.
2. **The vanilla gradient descent update is too crude for deep networks.** Adaptive learning rates, momentum, and per-parameter scaling all matter.
3. **Adam / AdamW is the default modern optimizer** for most deep-learning work.
4. **SGD with Nesterov momentum + a well-tuned schedule** can still beat Adam on some tasks (large CNNs).
5. **Learning-rate scheduling matters more than most people realize** — warmup, cosine decay, step decay.
6. **Weight decay ≠ L2 regularization** for adaptive optimizers — hence AdamW's existence.

If you retain nothing else: **use AdamW with a small warmup and cosine decay unless you have a specific reason not to.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 SGD, Mini-Batch SGD, Momentum

**Batch gradient descent** — one step per pass through the full training set:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

For large datasets, computing the full gradient is impossibly slow. So we use...

**Stochastic Gradient Descent (SGD)** — one step per training example:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_i(\theta)$$

Where $\mathcal{L}_i$ is the loss on a single example $i$. Noisy but fast; the noise itself sometimes helps escape saddle points.

**Mini-batch SGD** — one step per minibatch of $B$ examples (typical $B = 32, 64, 128, 256$):

$$\theta \leftarrow \theta - \eta \nabla_\theta \frac{1}{B}\sum_{i \in \text{batch}} \mathcal{L}_i(\theta)$$

This is what "SGD" means in modern practice. Batch size $B$ trades gradient noise (small $B$) for hardware efficiency (large $B$).

**Momentum** (Polyak, 1964) — accumulate a running gradient:

$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \mathbf{v}_t$$

Where $\mu \in [0, 1)$ is the momentum coefficient (typically 0.9). Interpretation:

- Weighted sum of past gradients, decayed by $\mu$.
- Smooths out noise in stochastic gradients.
- Accelerates convergence along consistent gradient directions.
- Helps push through flat regions and past small local barriers.

**Nesterov momentum** — a slight variant that computes the gradient at a "look-ahead" point:

$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_t - \eta \mu \mathbf{v}_{t-1})$$

Often converges slightly faster than classical momentum.

**When SGD wins:** large CNNs (ResNet), when you have compute budget to tune LR carefully. Community wisdom: SGD-momentum with a well-chosen schedule can beat Adam on ImageNet-scale CNNs, producing better generalization.

---

### 2.2 Adaptive Learning Rates: AdaGrad, RMSProp, Adam

The problem with SGD: **one learning rate for all parameters**. If some parameters need small updates (they're already near optimal) and others need large ones (still far off), a single LR is a compromise. Adaptive optimizers give each parameter its own effective LR based on its gradient history.

**AdaGrad** (Duchi et al., 2011):

$$G_t = G_{t-1} + \mathbf{g}_t^2 \quad \text{(elementwise)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \varepsilon} \mathbf{g}_t$$

- Accumulates squared gradients; effective LR shrinks over time.
- Great for sparse features (rare features get bigger updates).
- Problem: LR monotonically shrinks to zero — training stalls.

**RMSProp** (Hinton, lecture notes 2012) — fixes AdaGrad's shrinkage with a moving average:

$$G_t = \rho G_{t-1} + (1 - \rho) \mathbf{g}_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \varepsilon} \mathbf{g}_t$$

With $\rho \approx 0.9$. LR stays bounded — training doesn't stall.

**Adam** (Kingma & Ba, 2015) — RMSProp + momentum + bias correction:

Maintain two moving averages:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \quad \text{(first moment)}$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \quad \text{(second moment)}$$

Bias-corrected:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

Update:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon}$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, LR = $10^{-3}$.

**Why Adam works so well:**

- **Per-parameter effective LR** — parameters with volatile gradients get smaller updates.
- **Momentum** — smooths noisy updates.
- **Bias correction** — prevents the tiny early-training updates that would otherwise happen because $\mathbf{m}_0 = \mathbf{v}_0 = 0$.

Adam is the default for transformers, most modern architectures, and anytime you don't want to think.

**Variants:**

- **AdamW** — decouples weight decay from the adaptive gradient step (see 2.4). **This is the actual modern default.**
- **AMSGrad** — takes the max of past $\hat v_t$ to guarantee monotone descent; rarely used in practice.
- **RAdam** — rectified Adam, addresses the "warmup" issue by adjusting variance early on.
- **AdaBelief** — uses variance of gradient rather than magnitude; interesting research variant.
- **Lion** (2023) — memory-efficient alternative discovered via search; simpler than Adam.

---

### 2.3 Learning-Rate Schedules

The learning rate is the single most important hyperparameter. And it's rarely optimal to keep it constant.

**Step decay** — divide LR by 10 at fixed epochs (e.g., epochs 30, 60, 90 for a 100-epoch ImageNet run).

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}$$

Common in classical CNN training.

**Cosine annealing** (Loshchilov & Hutter, 2016) — smooth decrease from $\eta_0$ to a minimum:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\!\left(\pi \frac{t}{T}\right)\right)$$

The de-facto default for transformer training.

**Cosine with warm restarts (SGDR)** — periodically reset LR to $\eta_0$ and start a new cosine cycle. Escapes flat local minima.

**Linear warmup** — start LR at 0, ramp up over $T_{\text{warmup}}$ steps, then apply your main schedule:

$$\eta_t = \begin{cases}\eta_{\max} \cdot t / T_{\text{warmup}} & t < T_{\text{warmup}} \\ \text{schedule}(t) & t \geq T_{\text{warmup}}\end{cases}$$

Essential for transformer training — Adam's early updates are wild without warmup because $\hat v_t$ is unreliable at $t = 1$.

**Reduce-on-plateau** — halve LR whenever validation loss doesn't improve for $N$ epochs. Simple, effective, no schedule design needed.

**One-cycle policy** (Smith, 2018) — LR rises linearly for the first half, falls for the second; momentum inversely. Popular in `fastai` community. Often beats classical schedules with fewer epochs.

**Practical starting point** for a modern setup:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(step / warmup_steps, 1.0) * cosine_decay(step, total_steps)
)
```

Or use the `transformers.get_cosine_schedule_with_warmup` helper.

---

### 2.4 Weight Decay: AdamW vs Adam

**Weight decay** shrinks weights toward zero at each step:

$$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \mathbf{g}_t$$

In vanilla SGD, weight decay is equivalent to adding an L2 penalty $\frac{\lambda}{2} \|\theta\|^2$ to the loss. The gradient of that penalty is $\lambda \theta$, which enters the update:

$$\theta_{t+1} = \theta_t - \eta (\mathbf{g}_t + \lambda \theta_t) = (1 - \eta \lambda) \theta_t - \eta \mathbf{g}_t \quad \checkmark$$

But **in adaptive optimizers like Adam**, this equivalence breaks. The L2 gradient $\lambda \theta$ gets divided by $\sqrt{\hat v_t}$, so the effective decay is not $\eta \lambda \theta$ but $\eta \lambda \theta / \sqrt{\hat v_t}$ — a much messier expression.

**AdamW** (Loshchilov & Hutter, 2017/2019) fixes this by applying weight decay **outside** the gradient step:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon} - \eta \lambda \theta_t$$

This makes weight decay behave as intended (uniform shrinkage) and generally improves generalization. **AdamW is now the default in HuggingFace, PyTorch's `torch.optim`, and modern transformer training.**

**Common weight-decay values:**

- Vision: 5e-4 with SGD; 1e-2 to 5e-2 with AdamW.
- Transformers: 0.01 is standard.

**Don't apply weight decay to:** biases, LayerNorm/BatchNorm parameters, embeddings. These are usually excluded from decay by grouping parameters:

```python
decay_params = [p for n, p in model.named_parameters() if p.dim() > 1]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() <= 1]
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4)
```

---

### 2.5 Gradient Clipping, Mixed Precision, Gradient Accumulation

**Gradient clipping** — cap the gradient norm to prevent explosions:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Applied between `loss.backward()` and `optimizer.step()`. Standard for:
- **RNNs / LSTMs** — where BPTT can produce huge gradients.
- **Transformers** — grad clip 1.0 is standard.
- Any training with occasional NaN losses.

**Two variants:**
- **Norm clipping** — scale all gradients so the total norm ≤ threshold. Preferred.
- **Value clipping** — clip each gradient element to $[-c, +c]$. Cruder.

**Mixed precision** (fp16 or bf16) — store weights in fp32, do most compute in fp16/bf16:

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(x)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- 2× memory savings → bigger batches, bigger models.
- 2-3× speedup on modern GPUs with TensorCores.

**Gotchas with fp16:**
- Loss scaling needed to prevent underflow (GradScaler does this automatically).
- Overflow to `inf` possible with poorly-conditioned models.
- `bf16` (bfloat16) is often more stable than `fp16` because it has the same dynamic range as fp32 (fewer mantissa bits, more exponent bits).

**Gradient accumulation** — simulate a large batch on limited memory:

```python
accum_steps = 4  # effective batch = batch_size * accum_steps

for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / accum_steps  # scale down
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Perfect equivalent to a larger batch (up to slight numerical differences). Essential for training large models on smaller GPUs.

**Practical training-loop skeleton with everything:**

```python
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

for epoch in range(epochs):
    for i, (x, y) in enumerate(loader):
        with autocast(dtype=torch.bfloat16):
            loss = criterion(model(x), y) / accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
```

---

## 3. Mental Models & Analogies

### 3.1 The "Ball on a Landscape" Model

The loss surface is a landscape. Parameters are the position of a ball on it. Gradient descent rolls the ball downhill.

- **Vanilla SGD** = a small, weightless ball. Follows the local slope exactly; zigzags in narrow valleys; gets stuck on flat plateaus.
- **Momentum** = a heavier ball with inertia. Zigzags less. Rolls over small bumps. Overshoots minima if too heavy.
- **Nesterov** = a smart ball that looks a step ahead to decide the current push — brakes when about to overshoot.
- **Adam** = a ball wearing shoes with **variable grip per leg**: legs where the ground has been rough recently get careful, small steps; legs where the ground has been smooth take big confident strides.

The landscape isn't fixed — stochastic gradients mean the terrain shakes a little each step. That noise sometimes helps escape saddle points. But too much shaking (batch too small, LR too high) means the ball never settles anywhere.

### 3.2 The "Learning Rate as Zoom Level" Model

Think of training as **mapping a coastline**. Your LR is your zoom level.

- **High LR** = high altitude. You see the overall shape and move quickly across it. But you can't see cliff details; you might miss a fjord entirely.
- **Low LR** = low altitude, drone-level. You see every detail but move one meter at a time; charting the whole coast takes forever.

**Warmup** = start high (fast crude exploration), zoom in gradually.

**Cosine decay** = smoothly drop to low altitude toward the end (fine detail-mapping of the region you're in).

**Restart / warm restarts** = periodically fly back up to check you're on the right coast, then zoom in again.

The best schedules combine broad exploration early with fine refinement late. A constant LR either misses the big picture (too low) or the details (too high).

![IMG-OPT-01](/3%20—%20Deep%20Learning/images/IMG-OPT-01.jpg)
> **Caption:** Different optimizers navigate the loss surface differently; scheduled learning rates blend exploration with refinement.
> **Placement:** Section 2.3 / Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Adam Beats SGD Always"

Adam is the safer default and beats SGD in **early convergence** on almost anything. But on well-tuned large CNN training (ImageNet, style transfer, some GANs), SGD with momentum and a good schedule often produces **better final test accuracy**. Adam finds sharper minima; SGD tends to find flatter ones that generalize better. The gap has narrowed with AdamW and modern schedules, but it's not gone. Rule: start with AdamW; switch to SGD-momentum for the last leg on vision tasks if you're chasing SOTA.

### 4.2 "L2 Regularization in the Loss Is the Same as Weight Decay with Adam"

Not with adaptive optimizers. See section 2.4. In Adam, adding $\lambda \theta$ to the gradient means the "regularization" gets divided by $\sqrt{\hat v_t}$, distorting its effect. **Use AdamW** for correct decoupled weight decay. This bug is subtle, silent, and widespread in older code.

### 4.3 "Any LR Works if You Wait Long Enough"

Not really. LR too high = divergence, no matter the patience. LR too low = the model may plateau at a poor minimum because momentum can't build enough force to escape. A well-chosen LR (and schedule) is often the difference between a good model and a bad one — worth tuning first. Andrej Karpathy's rule: "3e-4 is the default learning rate for Adam." Not because it's optimal, but because it's a serviceable starting point that saves you from wasting a whole day scanning LR.

---

## 5. Self-Assessment Bank (Optimizers)

### Questions

**Q1 (Short answer).** Write the update rule for SGD with momentum.

**Q2 (Multiple choice).** Adam maintains two running quantities per parameter. What are they?
A. Gradient and Hessian.
B. First moment (mean of gradients) and second moment (mean of squared gradients).
C. Weight and bias.
D. Learning rate and momentum coefficient.

**Q3 (Short answer).** Why does Adam apply "bias correction" via $\hat m_t = m_t / (1 - \beta_1^t)$?

**Q4 (Multiple choice).** AdamW differs from Adam in that:
A. It uses a different learning rate.
B. It decouples weight decay from the adaptive gradient step.
C. It has no momentum.
D. It's slower.

**Q5 (Short answer).** Why is a linear warmup usually necessary when training a transformer with Adam/AdamW?

**Q6 (Multiple choice).** Gradient clipping is most critical for:
A. CNNs on ImageNet.
B. RNNs, LSTMs, and transformers — where gradients can spike due to depth or long sequences.
C. Linear regression.
D. Batch normalization.

**Q7 (Short answer).** What is gradient accumulation, and when do you use it?

**Q8 (Multiple choice).** Cosine annealing decays the learning rate:
A. In discrete steps at fixed epochs.
B. Smoothly following a cosine curve from $\eta_0$ to $\eta_\min$ over training.
C. By halving at every 100 iterations.
D. Only when validation loss plateaus.

**Q9 (Short answer).** Which parameters should typically be excluded from weight decay, and why?

**Q10 (Multiple choice).** On modern GPUs with TensorCores, training in `bf16` mixed precision:
A. Is slower than fp32.
B. Provides ~2× memory savings and 2–3× speedup with minimal stability loss.
C. Requires manual loss scaling.
D. Is only for inference.

---

### Answer Key & Detailed Explanations

**A1.** $\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \mathbf{g}_t$; $\theta_{t+1} = \theta_t - \eta \mathbf{v}_t$. (Some formulations move the LR into $\mathbf{v}$; equivalent.)

**A2. B.** Adam maintains $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$ (first moment estimate of the gradient, giving momentum) and $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$ (second moment estimate, giving per-parameter scaling).

**A3.** The moving averages are initialized to zero. Without correction, at $t = 1$ we'd have $\mathbf{m}_1 = (1-\beta_1)\mathbf{g}_1 \approx 0.1 \mathbf{g}_1$ — a very underestimated gradient. Bias correction $\mathbf{m}_1 / (1 - \beta_1^1) = \mathbf{g}_1$ restores the correct initial magnitude. Over time, $1 - \beta_1^t \to 1$ and correction vanishes.

**A4. B.** AdamW applies weight decay directly to parameters: $\theta \leftarrow \theta - \eta \lambda \theta$, separately from the Adam gradient step. This decouples decay from the adaptive scaling, restoring the "shrink weights toward zero" behavior that vanilla weight decay is supposed to have.

**A5.** Adam's per-parameter scaling relies on $\hat v_t$, which is an unreliable estimate at $t = 1$ (only one gradient sample seen). If you start at full LR, the effective update can be huge or tiny depending on the noisy first gradient — often destabilizing training. Linear warmup starts at LR 0 and ramps up over hundreds/thousands of steps, giving $\hat v_t$ time to stabilize before full updates.

**A6. B.** Recurrent architectures and transformers on long sequences can produce very large gradients (BPTT compounds; attention distributions can concentrate). Clipping to norm 1.0 is standard. CNNs usually don't need clipping.

**A7.** Gradient accumulation splits an effective large batch into several smaller minibatches, backpropagating each, summing the gradients, and stepping once every $N$ minibatches. It lets you simulate large-batch training on limited GPU memory. Essential for training big models on consumer GPUs.

**A8. B.** $\eta_t = \eta_\min + \frac{1}{2}(\eta_0 - \eta_\min)(1 + \cos(\pi t/T))$ — a smooth decrease from $\eta_0$ (at $t = 0$) to $\eta_\min$ (at $t = T$). Widely used in modern training because it consistently outperforms step decays.

**A9.** **Biases**, **LayerNorm / BatchNorm scale-shift parameters**, and often **embeddings**. Reason: these parameters aren't "weights" in the classical sense — they don't have the "keep them small" prior that weight decay imposes. Biases just shift outputs; norm parameters recalibrate features; embeddings are already regularized by the model's structure. Applying WD to them typically hurts performance.

**A10. B.** `bf16` (bfloat16) preserves fp32's dynamic range (same 8-bit exponent) but reduces mantissa precision — ideal for gradients. Modern GPUs with TensorCores accelerate bf16 matmuls significantly. `bf16` doesn't need loss scaling (unlike fp16), which makes it more robust in practice.

---

## 6. Practice Prompts

1. **Implement SGD+momentum.** Write a from-scratch momentum optimizer as a subclass of `torch.optim.Optimizer`. Verify parity with `torch.optim.SGD(momentum=0.9)` on a small task.
2. **Implement Adam.** Write from-scratch Adam. Check parity with `torch.optim.Adam`.
3. **AdamW vs Adam.** Train the same model with `Adam(weight_decay=0.01)` and `AdamW(weight_decay=0.01)`. Compare test accuracy on CIFAR-10 or a similar benchmark.
4. **LR range test.** Sweep LR from 1e-6 to 10 in a linear-ish schedule over one epoch. Plot loss vs LR. Identify the "sweet zone."
5. **Warmup + cosine.** On a small transformer training on a toy dataset, compare (a) no warmup, (b) 100-step linear warmup + cosine decay. Observe stability at start.

---

## 7. References

- Kingma & Ba, ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) (2015).
- Loshchilov & Hutter, ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) (2017/19) — AdamW.
- Loshchilov & Hutter, ["SGDR: Stochastic Gradient Descent with Warm Restarts"](https://arxiv.org/abs/1608.03983) (2016).
- Sebastian Ruder, ["An Overview of Gradient Descent Optimization Algorithms"](https://arxiv.org/abs/1609.04747) (2016).
- Leslie Smith, ["Cyclical Learning Rates"](https://arxiv.org/abs/1506.01186) (2015) — one-cycle.
- Chen et al., ["Symbolic Discovery of Optimization Algorithms"](https://arxiv.org/abs/2302.06675) (2023) — Lion.
