# Activation Functions — Master Study Guide

> **Track:** Deep Learning · **Module:** 03
> **Prerequisites:** Modules 01–02.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** The activation function is the **nonlinearity** that gives a neural network its expressive power. Without it, stacked linear layers collapse into a single linear layer (Module 01, section 2.2). *With* it, a network can approximate arbitrarily complex functions.

Choosing the right activation matters more than it seems. The rise of ReLU in 2011 was a **critical enabler** of the deep-learning revolution — it made training networks with dozens of layers practical for the first time. And every modern activation (GELU, Swish, SiLU) is fine-tuning on that theme.

**Fundamental principles you must own:**

1. **The activation must be nonlinear** for depth to buy expressiveness.
2. **The activation must have a non-vanishing derivative** in the range of typical inputs — otherwise gradients die.
3. **Different activations suit different contexts** — hidden layers, output layers, gates, attention.
4. **ReLU is the default** for hidden layers in feedforward and convolutional nets.
5. **GELU (or SiLU/Swish)** is the modern default for transformers.
6. **Softmax and sigmoid** live in output layers and gates — not in hidden feedforward stacks.

If you retain nothing else: **for hidden layers, use ReLU (or GELU/SiLU). For output layers, use the activation your loss expects (sigmoid for BCE, softmax for CE, none for MSE).**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Sigmoid and Tanh — The Historical Defaults

**Sigmoid:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

- Output range $(0, 1)$; interpretable as probability.
- Peak derivative $0.25$ at $z = 0$.
- **Saturates** for $|z| \gg 0$ — derivative $\to 0$.

**Tanh:**

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$

- Output range $(-1, 1)$; zero-centered (unlike sigmoid).
- Peak derivative $1.0$ at $z = 0$.
- Also saturates for $|z| \gg 0$.

**Why they fell out of favor for hidden layers:**

1. **Vanishing gradients.** Both saturate → derivative goes to 0 → backprop through many layers loses signal exponentially.
2. **Slow convergence.** For sigmoid, outputs are always $> 0$ — activations don't have zero mean, which slows down gradient descent (a subtle effect).
3. **`exp` is expensive.** Tanh and sigmoid both require exponentials, which are slower than the simple `max(0, x)` of ReLU.

**Where they still shine:**

- **Sigmoid** as an output activation for **binary classification** paired with BCE loss.
- **Sigmoid** as a **gate** in LSTM/GRU (its output is a "how much of this to keep" scalar in $[0, 1]$).
- **Tanh** as an activation in specific recurrent architectures (RNN, LSTM).

For a fresh network, **do not use sigmoid or tanh in hidden feedforward layers.** ReLU-family activations are strictly better.

---

### 2.2 ReLU and Its Variants

**ReLU** (Rectified Linear Unit):

$$\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases}1 & z > 0 \\ 0 & z < 0 \\ \text{undefined} & z = 0\end{cases}$$

(By convention, we use 0 or 1 at $z = 0$ — the discontinuity is measure-zero.)

**Advantages:**
- **No saturation** for $z > 0$ — gradient is 1, so signal passes cleanly through deep stacks.
- **Sparse activation** — half the units are zero on average.
- **Cheap to compute** — a max, no exponentials.
- Empirically massive speedups over sigmoid/tanh, and better final accuracy.

**The dying-ReLU problem.** A neuron whose input is always $\leq 0$ has output always 0, gradient always 0, and never learns. Once dead, always dead. Common with large learning rates.

**Variants that fix dying ReLU:**

**LeakyReLU** — nonzero slope for $z < 0$:

$$\text{LeakyReLU}(z) = \begin{cases}z & z > 0 \\ \alpha z & z \leq 0\end{cases}, \quad \alpha \in (0, 1) \text{ typically } 0.01$$

**PReLU** — LeakyReLU with $\alpha$ as a learnable parameter (per channel or per unit).

**ELU** (Exponential Linear Unit):

$$\text{ELU}(z) = \begin{cases}z & z > 0 \\ \alpha(e^z - 1) & z \leq 0\end{cases}$$

- Smooth, saturates on the negative side (converges to $-\alpha$).
- Zero-centered on average → better convergence in some tasks.

**SELU** (Scaled ELU) — a specific scaled ELU that, combined with LeCun-normal initialization and no batch normalization, gives self-normalizing properties. Rarely used today because BatchNorm/LayerNorm dominate.

**In practice for feedforward/CNN hidden layers:** ReLU is the default. Switch to LeakyReLU (with $\alpha = 0.01$) if you suspect dying ReLU issues. Rarely need PReLU/ELU unless you're doing research.

---

### 2.3 Modern Smooth Activations: GELU, Swish/SiLU, Mish

**GELU** (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016):

$$\text{GELU}(z) = z \cdot \Phi(z) = z \cdot \frac{1}{2}\!\left[1 + \text{erf}\!\left(\frac{z}{\sqrt{2}}\right)\right]$$

Where $\Phi$ is the standard-normal CDF. GELU is smoothly interpolated between "keep the input" (large positive) and "zero it out" (large negative), with the transition governed by a Gaussian.

Common tanh-based approximation for speed:

$$\text{GELU}(z) \approx 0.5 z \left(1 + \tanh\!\left[\sqrt{2/\pi}\,(z + 0.044715 z^3)\right]\right)$$

**Swish / SiLU** (Sigmoid-Weighted Linear Unit):

$$\text{Swish}(z) = z \cdot \sigma(\beta z)$$

With $\beta = 1$ (fixed), this is called **SiLU**. With $\beta$ learnable, "Swish".

**Mish**:

$$\text{Mish}(z) = z \cdot \tanh(\text{softplus}(z)) = z \tanh(\ln(1 + e^z))$$

**Properties of this family:**

- **Smooth** (differentiable everywhere) — some kernels/optimizers prefer this.
- **Nonzero gradient** for $z < 0$ → no dying neurons.
- **Non-monotonic** — small negative values return small positive outputs, then dip.
- Empirically produce small but consistent accuracy gains over ReLU on many benchmarks.

**Which to use in modern nets:**

- **Transformers**: GELU is the default (BERT, GPT-2/3/4, T5).
- **Vision transformers**: GELU.
- **CNNs**: still often ReLU, but SiLU/Swish are increasingly common (EfficientNet uses Swish).
- **Diffusion / audio models**: SiLU common.

Empirical gap is usually small (< 1% accuracy) — don't obsess over the choice for early modeling.

---

### 2.4 Output-Layer Activations: Softmax, Sigmoid, None

The output activation must **match your loss function** and the interpretation of your outputs.

**Regression** — no activation on the output layer. Predictions are unbounded reals. Loss: MSE, MAE, Huber.

**Binary classification** — **sigmoid** output; loss: BCE (binary cross-entropy).

$$\hat p = \sigma(z), \quad \mathcal{L}_{\text{BCE}} = -[y \log \hat p + (1-y)\log(1 - \hat p)]$$

**Multi-class classification** ($K$ mutually exclusive classes) — **softmax** output; loss: CE (categorical cross-entropy).

$$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad \mathcal{L}_{\text{CE}} = -\sum_k y_k \log \hat p_k$$

**Multi-label classification** ($K$ non-mutually-exclusive labels) — **sigmoid per output** (not softmax), loss: BCE per label, summed.

**Numerical stability.** In practice, PyTorch's `nn.CrossEntropyLoss` and `nn.BCEWithLogitsLoss` **include the softmax/sigmoid inside the loss** — you pass raw logits to them. This is more numerically stable (via log-sum-exp) than applying sigmoid/softmax explicitly and then computing loss. **Don't double-apply.**

```python
# CORRECT
logits = model(x)                       # raw scores, no softmax
loss = nn.CrossEntropyLoss()(logits, y) # combines softmax + NLL

# WRONG — double softmax
probs = nn.functional.softmax(model(x), dim=-1)
loss = nn.CrossEntropyLoss()(probs, y)
```

**Softmax gradient** is elegant. For loss $\mathcal{L}$ and softmax output $\hat p_k = e^{z_k}/\sum_j e^{z_j}$ with one-hot target $y$:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \hat p_k - y_k$$

A single subtraction — this is why softmax+CE is the training-time sweetheart of classification.

---

### 2.5 Gating and Special-Purpose Activations

Modern architectures use activations for **more than just adding nonlinearity** — they're used to *gate* information flow.

**In LSTMs / GRUs** — sigmoid gates control how much of a stream to keep:

$$\text{forget gate: } f_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f), \quad \text{new memory: } c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$$

The sigmoid's output in $[0, 1]$ acts as a soft on/off switch.

**In Highway networks** — gates control whether to use a transformed input or pass through unchanged:

$$y = t \cdot H(x, W_H) + (1 - t) \cdot x, \quad t = \sigma(W_T x + b_T)$$

**In GLU (Gated Linear Unit)** and its cousins — used in modern language models (PaLM, LLaMA, T5v1.1):

$$\text{GLU}(\mathbf{x}) = (W \mathbf{x} + \mathbf{b}) \odot \sigma(V \mathbf{x} + \mathbf{c})$$

**SwiGLU** — the LLaMA/PaLM variant with SiLU instead of sigmoid:

$$\text{SwiGLU}(\mathbf{x}) = (W \mathbf{x}) \odot \text{SiLU}(V \mathbf{x})$$

This appears in LLaMA's feedforward layers.

**Softmax outside classification.** Softmax appears inside **attention** (Module 09) — turning raw similarity scores into a probability distribution over tokens to attend to.

**Takeaway:** activations aren't just "make nonlinear." They're expressive components that can gate, normalize, and route information. Modern architectures use several kinds *within a single block*.

---

## 3. Mental Models & Analogies

### 3.1 The "Signal Through a Diode" Model (ReLU)

Imagine each ReLU neuron as an **electrical diode**: current flows freely in the positive direction, but is blocked in the negative direction. If the sum of inputs is positive, the neuron passes it through unchanged. If negative, it stops entirely.

- **Sparse activation** = most diodes are off at any moment; only a subset fire per input.
- **Dead ReLU** = a burned-out diode: it will never fire again, no matter what.
- **LeakyReLU** = a diode that leaks a tiny reverse current — never fully off.
- **GELU / Swish** = a smooth diode with fuzzy cutoff — passes some signal near zero, smoothly transitioning to fully off far below.

This model gives you an immediate intuition for why sparse activation is powerful (each pattern of "on" diodes represents a distinct feature detector) and why dying is a concern.

### 3.2 The "Volume Knobs and On/Off Switches" Model (Gates)

Think of a modern architecture (LSTM cell, transformer FFN with GLU) as a **mixing console** with two kinds of controls:

- **Signal channels** — carry information forward (the linear layer's output).
- **Sigmoid / SiLU gates** — volume knobs that scale each channel from 0 to 1.

$\mathbf{h} \odot \sigma(\mathbf{g})$ is literally: multiply the signal by the sigmoid'd gate, elementwise. Each channel gets its own knob. If a knob is at 0, that channel is muted; at 1, it's full-strength; anywhere between is a fade.

This lets the network *learn what to attend to* rather than always passing everything through. LSTM's forget gate: "how much of yesterday's memory should I keep?" Transformer's GLU: "which features from this projection are relevant here?" Attention softmax: "how much of each token's info should this position mix in?"

The trick: gates are just activations with a functional interpretation. Sigmoid → 0-to-1 volume. Softmax → normalized distribution. This is a big part of *why* modern architectures work: they're combinations of standard layers wired to gate each other.

![IMG-ACT-01](/3%20—%20Deep%20Learning/images/IMG-ACT-01.jpg)
> **Caption:** Common activation functions — S-shaped saturators (sigmoid, tanh) vs rectifiers (ReLU family) vs smooth rectifiers (GELU, SiLU).
> **Placement:** Section 2 opening.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Using Sigmoid in Hidden Layers Because It's Interpretable"

Sigmoid in hidden feedforward layers → saturation → vanishing gradients → no training. There is no "interpretability" to salvage inside a deep network — the intermediate activations aren't probabilities anyway. Reserve sigmoid for output layers of binary classifiers and for gates. Everywhere else in the hidden stack, use ReLU or GELU.

### 4.2 "Double-Applying Softmax"

`nn.CrossEntropyLoss` in PyTorch (and equivalents elsewhere) expect **raw logits** — they apply softmax internally in a numerically stable way. If you also apply `softmax` before passing to the loss, you softmax twice, which mangles the gradients and the loss values. Same for `nn.BCEWithLogitsLoss` — it takes logits, not sigmoid'd probs. When in doubt, feed raw logits to the loss.

### 4.3 "GELU vs ReLU Makes a Huge Difference"

Empirically, modern activations (GELU, SiLU) give <1% accuracy improvement over ReLU on most tasks. They're a modest polish, not a game-changer. Use ReLU for CNNs and traditional MLPs; use GELU for transformers because that's the convention. Don't spend weeks debating activation choice — spend it on data quality, architecture, and regularization.

---

## 5. Self-Assessment Bank (Activation Functions)

### Questions

**Q1 (Short answer).** Why is a nonlinear activation necessary between neural-network layers?

**Q2 (Multiple choice).** The primary reason ReLU is preferred over sigmoid for hidden layers is:
A. It's more biologically realistic.
B. It doesn't saturate for positive inputs; its gradient is 1 there, preventing vanishing gradients.
C. It has a smoother derivative.
D. It outputs probabilities.

**Q3 (Short answer).** Explain the "dying ReLU" problem and one fix.

**Q4 (Multiple choice).** For a 10-class classification model, the output activation should be:
A. ReLU.
B. Sigmoid on each of 10 outputs.
C. Softmax over 10 outputs.
D. Tanh.

**Q5 (Short answer).** Give the derivative of the sigmoid function in terms of itself, and use it to explain the vanishing-gradient problem.

**Q6 (Multiple choice).** In PyTorch, `nn.CrossEntropyLoss` expects:
A. Softmax probabilities.
B. Raw logits (no softmax applied).
C. One-hot encoded probabilities.
D. Log-probabilities.

**Q7 (Short answer).** What activation is used in modern transformer FFN layers, and why not just ReLU?

**Q8 (Multiple choice).** GELU can be understood as:
A. A strict rectifier at 0.
B. $z$ multiplied by the standard-normal CDF of $z$ — a smooth interpolation between "keep" and "zero out".
C. A saturating sigmoid.
D. The same as tanh.

**Q9 (Short answer).** Why is sigmoid still used inside LSTM/GRU cells despite its saturation issue?

**Q10 (Multiple choice).** For multi-label classification (each label independently 0 or 1), the correct output setup is:
A. Softmax + cross-entropy.
B. Sigmoid per output + BCE per output.
C. Tanh + MSE.
D. ReLU + softmax.

---

### Answer Key & Detailed Explanations

**A1.** Without nonlinearity, composing linear layers yields another linear function ($W_2(W_1 \mathbf{x}) = (W_2 W_1)\mathbf{x}$). Depth buys nothing. Nonlinear activations are what make deep networks strictly more expressive than shallow linear models.

**A2. B.** ReLU's derivative is 1 for positive $z$ — it doesn't compress signals as they pass through. Sigmoid's derivative peaks at 0.25 and saturates to 0 at extremes, so signals lose 4× or more per layer at best and much more when saturated.

**A3.** A neuron whose preactivation is always $\leq 0$ has output 0, gradient 0, and never updates — permanently dead. Common causes: large learning rate pushes weights into a bad region; poor initialization creates a large initial negative bias. Fixes: LeakyReLU / GELU (nonzero gradient below zero), Kaiming init, smaller learning rate.

**A4. C.** Softmax normalizes 10 raw logits into a probability distribution summing to 1 — appropriate for mutually-exclusive multiclass. Sigmoid-per-output (B) is for multi-label. ReLU/tanh don't produce probabilities.

**A5.** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, peaks at $0.25$ (when $\sigma(z) = 0.5$). Backpropagating through $L$ sigmoid layers multiplies $L$ derivatives, each at most $0.25$ — so gradients decay as $\leq (0.25)^L$, which for $L = 20$ is $\sim 10^{-13}$. Vanishing gradients kill training.

**A6. B.** `nn.CrossEntropyLoss` applies log-softmax internally (numerically stable) and combines with NLL loss. Passing raw logits is correct; passing softmax'd values applies softmax twice → wrong gradients.

**A7.** Modern transformers use **GELU** (or SiLU/SwiGLU). GELU is smooth (differentiable everywhere), non-zero-gradient for slightly-negative inputs (avoiding dead ReLU), and empirically improves training stability and final accuracy modestly on language tasks. ReLU works fine too — the gain is small but real.

**A8. B.** GELU: $z \cdot \Phi(z)$ where $\Phi$ is the standard-normal CDF. For large positive $z$, $\Phi(z) \to 1$, so GELU → $z$ (like ReLU passes it). For large negative $z$, $\Phi(z) \to 0$, GELU → 0. Between, smooth transition.

**A9.** Sigmoid inside gates outputs values in $[0, 1]$ interpretable as "how much of this signal to keep." A ReLU or GELU gate wouldn't work — you need bounded, near-zero-to-near-one values. Sigmoid saturation isn't fatal here because the gate is a small computation not stacked deeply, and the gate's output is multiplied elementwise (not chained through many layers).

**A10. B.** Multi-label = each label independent, so no cross-label normalization. Use sigmoid per output and BCE loss per output (summed or averaged). Softmax would incorrectly force the labels to compete.

---

## 6. Practice Prompts

1. **Plot them.** Plot sigmoid, tanh, ReLU, LeakyReLU, ELU, GELU, SiLU on the same axis for $z \in [-6, 6]$. Plot their derivatives on a second axis.
2. **Dying ReLU demo.** Build a 3-layer MLP with ReLU. Use a large learning rate to intentionally kill neurons. Count how many hidden units have zero activation across all training samples after 1 epoch.
3. **Sigmoid depth failure.** Train a 20-layer network with sigmoid activations on MNIST. Compare to the same network with ReLU. Note the training-loss failure.
4. **Softmax stability.** Compute softmax for a batch with large logits (e.g., `[1000., 1001., 1002.]`). Without the log-sum-exp trick, this overflows. Implement it correctly.
5. **GELU approx vs exact.** Compare the exact GELU (using `torch.erf`) to the tanh approximation. Plot the difference for $z \in [-4, 4]$.

---

## 7. References

- Nair & Hinton, ["Rectified Linear Units Improve Restricted Boltzmann Machines"](https://icml.cc/Conferences/2010/papers/432.pdf) (2010).
- Hendrycks & Gimpel, ["Gaussian Error Linear Units (GELUs)"](https://arxiv.org/abs/1606.08415) (2016).
- Ramachandran et al., ["Searching for Activation Functions"](https://arxiv.org/abs/1710.05941) (2017) — Swish paper.
- Shazeer, ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) (2020) — SwiGLU.
- Goodfellow et al., *Deep Learning*, chapter 6.3.
