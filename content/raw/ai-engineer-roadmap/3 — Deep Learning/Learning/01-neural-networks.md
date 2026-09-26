# Neural Networks (Fundamentals) — Master Study Guide

> **Track:** Deep Learning · **Module:** 01
> **Prerequisites:** Linear algebra (matrix multiplication), calculus (chain rule), logistic regression.
> **Time budget:** ~12–15 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** A neural network is a **stack of linear-then-nonlinear transformations trained by gradient descent.** That single sentence is enough to eventually explain every model you'll see this month — from a two-layer MLP to GPT-4. Getting fluent with the fundamentals now means every subsequent architecture (CNN, RNN, Transformer) is a *pattern*, not a new subject.

Historical note: the "neural" metaphor is more marketing than mechanism — biological neurons don't fire like $\sigma(\mathbf{w}^\top\mathbf{x})$, and the "brain-like" framing has misled generations of learners. Treat neural nets as **compositional differentiable function approximators** trained to minimize a loss.

**Fundamental principles you must own:**

1. **A neuron is a scalar function** $y = \phi(\mathbf{w}^\top\mathbf{x} + b)$ — linear combination + nonlinearity.
2. **A layer is a matrix multiply plus a nonlinearity** applied to a whole vector at once: $\mathbf{h} = \phi(W\mathbf{x} + \mathbf{b})$.
3. **A network is a composition of layers.** Stack them, and you get $f(\mathbf{x}) = \phi_L(W_L \ldots \phi_1(W_1 \mathbf{x} + \mathbf{b}_1) \ldots + \mathbf{b}_L)$.
4. **The universal approximation theorem** says that a sufficiently wide 2-layer network can approximate any continuous function on a bounded domain — arbitrarily well.
5. **The nonlinearity is critical.** Without $\phi$, stacking linear layers collapses to a single linear layer. Depth buys nothing without nonlinearity.
6. **Weights and biases are the parameters** — learned by gradient descent on a loss.

If you retain nothing else: **a neural network is a differentiable function with millions of knobs, tuned by gradient descent to minimize an error signal.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 A Single Neuron and a Layer

**A single neuron:**

$$y = \phi(\mathbf{w}^\top\mathbf{x} + b) = \phi\!\left(\sum_{j=1}^p w_j x_j + b\right)$$

Where:
- **$\mathbf{x} \in \mathbb{R}^p$** = the input vector.
- **$\mathbf{w} \in \mathbb{R}^p$** = the neuron's weights.
- **$b \in \mathbb{R}$** = the neuron's bias.
- **$\phi$** = the activation function (ReLU, sigmoid, tanh, …).
- **$y \in \mathbb{R}$** = the neuron's scalar output.

**A layer** of $H$ neurons operating on the same input $\mathbf{x}$:

$$\mathbf{h} = \phi(W\mathbf{x} + \mathbf{b})$$

Where:
- **$W \in \mathbb{R}^{H \times p}$** — stack of $H$ weight vectors as rows.
- **$\mathbf{b} \in \mathbb{R}^H$** — one bias per neuron.
- **$\phi$** is applied elementwise.
- **$\mathbf{h} \in \mathbb{R}^H$** — the layer's output ("hidden representation").

**A minibatch** of $B$ examples uses matrix inputs:

$$H = \phi(X W^\top + \mathbf{1}\mathbf{b}^\top), \quad X \in \mathbb{R}^{B \times p}, \quad H \in \mathbb{R}^{B \times H}$$

The broadcast $\mathbf{1}\mathbf{b}^\top$ adds the same bias vector to each row. In practice PyTorch handles this automatically — but understanding the shapes is table stakes.

**Terminology.**

- **Fully-connected / dense / linear layer** — the classic $W\mathbf{x} + \mathbf{b}$ operation.
- **Multi-Layer Perceptron (MLP)** — a stack of dense + activation layers.
- **Feedforward network** — same idea; information flows one direction, no cycles.

---

### 2.2 Composing Layers into a Network

Stack layers:

$$\mathbf{h}_1 = \phi(W_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \phi(W_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\vdots$$
$$\hat{\mathbf{y}} = W_L \mathbf{h}_{L-1} + \mathbf{b}_L$$

The final layer usually has **no activation** (for regression) or a task-appropriate one (softmax for multiclass classification, sigmoid for binary).

**Why nonlinearity matters.** Without $\phi$:

$$\mathbf{h}_2 = W_2 \mathbf{h}_1 + \mathbf{b}_2 = W_2 W_1 \mathbf{x} + W_2 \mathbf{b}_1 + \mathbf{b}_2 = W' \mathbf{x} + \mathbf{b}'$$

That is, two linear layers compose into a single linear layer. No matter how many you stack, you have a linear model. **Nonlinear activations are the only thing that gives depth its power.**

**Universal approximation theorem** (Cybenko 1989, Hornik 1991). A feedforward network with a single hidden layer of enough neurons (and a nonpolynomial activation) can approximate any continuous function on a compact domain to arbitrary accuracy.

**Caveat.** "Enough neurons" can mean unbounded. In practice **depth is exponentially more efficient than width** — a deep 5-layer network can represent functions that would need a preposterously wide 1-layer network. This is why modern nets are deep.

**Counting parameters** for an $L$-layer MLP with input size $p$ and hidden sizes $H_1, \ldots, H_L$:

$$\text{params} = \sum_{\ell=1}^L (H_{\ell-1} \cdot H_\ell + H_\ell) \quad \text{(with } H_0 = p\text{)}$$

A 3-layer MLP with $p = 784$ (MNIST pixels), $H_1 = H_2 = 256$, $H_3 = 10$: $(784 \cdot 256 + 256) + (256 \cdot 256 + 256) + (256 \cdot 10 + 10) \approx 269{,}000$ parameters.

---

### 2.3 Forward Pass, Loss, and Backpropagation (Overview)

**The training pipeline** for one minibatch:

1. **Forward pass** — feed batch $X$ through the network to get predictions $\hat{Y}$.
2. **Loss** — compute a scalar $\mathcal{L}(\hat{Y}, Y)$ measuring how wrong we are.
3. **Backward pass** — compute $\partial\mathcal{L} / \partial \theta$ for every parameter $\theta$ via the chain rule.
4. **Update** — adjust parameters: $\theta \leftarrow \theta - \eta \, \partial\mathcal{L}/\partial\theta$.
5. Repeat over many batches (epochs).

**PyTorch code sketch:**

```python
for epoch in range(epochs):
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)                          # forward
        loss  = criterion(y_hat, y)               # scalar
        optimizer.zero_grad()                     # clear old grads
        loss.backward()                           # backprop (autograd)
        optimizer.step()                          # update params
```

The magic step is `loss.backward()` — PyTorch's autograd traces the computation graph from `loss` back to every parameter and computes gradients automatically. Module 02 unpacks this.

**Bias term convention.** Some notation absorbs the bias into $W$ by prepending a 1 to $\mathbf{x}$ ("bias trick"). Modern PyTorch keeps them separate for readability. Both are equivalent.

---

### 2.4 Initialization Matters

Say you initialize all weights to zero. Then every neuron in a layer computes the same output and receives the same gradient — they update identically, so the network has no more expressive power than a single neuron per layer. This is the **symmetry problem**.

**Random initialization** breaks symmetry. But not any random initialization:

- **Too small** → activations shrink through layers → gradients vanish → the network doesn't learn.
- **Too large** → activations explode → gradients explode → NaN losses.

**Modern initializers** target unit variance of activations across layers:

**Xavier / Glorot (Glorot & Bengio, 2010)** — for tanh, sigmoid, softmax:

$$W_{ij} \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad \text{or} \quad U\!\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

**He / Kaiming (He et al., 2015)** — for ReLU / LeakyReLU (which zeros half the inputs on average):

$$W_{ij} \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}}}\right)$$

**PyTorch defaults:** `nn.Linear` uses **Kaiming uniform** for weights and small uniform for bias — safe for most cases. For anything unusual (Xavier for tanh/GRU, LeCun for SELU), initialize explicitly:

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
nn.init.zeros_(layer.bias)
```

**Biases** are usually initialized to zero. Exceptions: forget-gate bias of LSTM (often 1.0 to make the gate open by default).

---

### 2.5 Vanishing and Exploding Gradients

**The problem.** In deep networks, gradients propagate through many layers via the chain rule. Each layer's contribution multiplies the previous. If the multiplier is on average less than 1, gradients vanish exponentially with depth; more than 1, they explode.

**Mathematically**, for a scalar-input scalar-output composition of $L$ identical layers:

$$\frac{\partial y}{\partial x} = \prod_{\ell=1}^L \phi'(z_\ell) \cdot w_\ell$$

If $\phi'(z_\ell) \cdot w_\ell = 0.5$ on average, at $L = 30$ layers the gradient is $\sim 10^{-9}$ — the first layers get essentially zero signal and stop learning.

**Symptoms:**

- **Vanishing gradients**: loss stalls; deeper layers' weights don't move; sigmoid/tanh saturate.
- **Exploding gradients**: NaN losses; extremely large weights; unstable training.

**Fixes we now have:**

1. **ReLU activation** (Module 03) has $\phi'(z) \in \{0, 1\}$ — no exponential decay through positive activations.
2. **Careful initialization** (Xavier, Kaiming) keeps signal magnitudes stable at start.
3. **Batch/layer normalization** (Module 06) re-centers and re-scales at each layer.
4. **Residual connections** (introduced in ResNets, Module 07) provide a "gradient highway" bypassing each block.
5. **Gradient clipping** — cap the gradient norm at $\|\nabla\| \leq c$; standard for RNNs/transformers.

**In practice today**, well-designed architectures (transformers with pre-norm, LayerNorm, and residuals) rarely have these issues. But you'll still hit them — when you build something novel, when you skip initialization, or when you scale to hundreds of layers. Recognize the pattern.

---

## 3. Mental Models & Analogies

### 3.1 The "Feature Refinery" Model

Think of raw inputs (pixels, tokens, words) as **crude ore**. Each layer of a neural network is a **refinement stage**:

- **Layer 1** extracts elementary features — edges, corners in an image; token embeddings in text.
- **Layer 2** combines those into mid-level features — shapes, phrases.
- **Layer $L$** produces task-specific representations — object identity, sentiment.

Each layer's job is small (a linear projection + a nonlinear "pinch"). Their **composition** produces a hierarchy of representations, each layer building on the previous.

The **nonlinearity** is what makes each stage genuinely new. Without it, you're just linearly re-mixing the same ore over and over. With it, you can distill iron from raw dirt.

Modern insight: as networks grow deeper, they learn genuinely hierarchical, reusable features — which is why **transfer learning** works. A ResNet trained on ImageNet has already learned "generic vision" in its early layers.

### 3.2 The "Origami" Model (Nonlinear Warping)

Imagine your input space as a flat sheet of paper with data points on it. A linear layer is a rigid transformation — rotate, stretch, shift the paper. It cannot bend the paper.

An **activation function** is a fold. It bends the paper along some line. Do enough folds in a row and any target shape becomes achievable — you can crumple, twist, and reconfigure the paper into complex origami.

- **A 2-layer network** = a single crease. Simple curves.
- **A 5-layer network** = many creases. Complex shapes.
- **A 50-layer network** = arbitrarily fine origami. Any continuous decision boundary.

The **universal approximation theorem** is the mathematical version of this claim: given enough creases (neurons + nonlinearity), you can fold flat space into any target shape.

Chris Olah's classic blog post on this analogy — where he shows neural nets pulling apart interlocking manifolds — is worth an hour of your time.

![IMG-NN-01](/3%20—%20Deep%20Learning/images/IMG-NN-01.jpg)
> **Caption:** A feedforward network is a stack of matrix multiplications and pointwise nonlinearities.
> **Placement:** Section 2.2.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Add More Layers to Get Better Results"

Without residual connections, normalization, and modern initialization, depth **hurts** — gradients vanish, training destabilizes, and the deeper model underperforms the shallow one. ResNet's central discovery (2015) was that with residual connections, depth *does* help — you can train 152-layer networks. But naïvely stacking dense layers still fails past ~10 layers. Modern architectures (transformers) succeed because of the packaged design — attention + residuals + LayerNorm — not because layers are magic.

### 4.2 "Neural Networks Learn Like Human Brains"

They don't. Backpropagation is not biologically realistic. Real neurons are stochastic, sparse, asynchronous, and use spike-timing dynamics; ANN "neurons" are deterministic pointwise nonlinearities. This misconception leads to bad intuitions ("but the brain would ..."). Treat neural networks as **differentiable function approximators optimized by gradient descent**, not as a biology model. The name is historical.

### 4.3 "More Parameters = Always Better Model"

Not without corresponding **data**, **regularization**, or **inductive bias**. A network with more parameters than samples can memorize training data. Two decades of ML tried to prevent this. But modern **overparameterized** networks (billions of params on billions of tokens) somehow generalize anyway — the "double descent" and "implicit regularization" phenomena. This is an active research area. The safe rule for you: match model capacity to your task's data and problem complexity, and rely on regularization to close any gap.

---

## 5. Self-Assessment Bank (Neural Network Fundamentals)

### Questions

**Q1 (Short answer).** Write the forward pass of a single fully-connected layer, and of a stack of two such layers with an activation between them.

**Q2 (Multiple choice).** Without a nonlinear activation between layers, a 10-layer MLP is equivalent to:
A. A universal function approximator.
B. A single linear layer.
C. A wider network.
D. A convolutional network.

**Q3 (Short answer).** State the universal approximation theorem in one sentence. What's the key caveat?

**Q4 (Multiple choice).** Which initialization is appropriate for a ReLU activation?
A. Zeros.
B. Xavier / Glorot.
C. Kaiming / He.
D. Constant 0.5.

**Q5 (Short answer).** Explain the symmetry problem when initializing weights to zero.

**Q6 (Multiple choice).** In a network with sigmoid activations at every layer, the vanishing gradient problem occurs because:
A. Sigmoid outputs are always positive.
B. Sigmoid's derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ has maximum $0.25$; multiplied over many layers, gradients shrink exponentially.
C. Sigmoid inputs are bounded.
D. The chain rule doesn't apply to sigmoid.

**Q7 (Short answer).** In a network with layer sizes $(p, H_1, H_2, K)$, how many total learnable parameters are there? (Include biases.)

**Q8 (Multiple choice).** The bias term in a neural network layer is:
A. Necessary — allows the model to shift the activation output away from zero.
B. Optional but usually helpful.
C. Rarely used in modern architectures.
D. Only used in the output layer.

**Q9 (Short answer).** Why is depth (many layers) often more parameter-efficient than width (many neurons per layer)?

**Q10 (Multiple choice).** If your training loss goes to NaN after a few iterations, the most likely explanations include:
A. Learning rate too high, poor initialization, no gradient clipping.
B. Batch size too large.
C. Not enough epochs.
D. Using ReLU instead of tanh.

---

### Answer Key & Detailed Explanations

**A1.** Single layer: $\mathbf{h} = \phi(W\mathbf{x} + \mathbf{b})$ with $W \in \mathbb{R}^{H \times p}$, $\mathbf{b} \in \mathbb{R}^H$. Two layers with activation between: $\mathbf{h}_1 = \phi(W_1 \mathbf{x} + \mathbf{b}_1)$; $\mathbf{h}_2 = \phi(W_2 \mathbf{h}_1 + \mathbf{b}_2)$.

**A2. B.** Composition of linear functions is linear: $W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$. No matter how many, you collapse to a single effective linear layer.

**A3.** A feedforward network with a single hidden layer and enough neurons (using a nonpolynomial activation) can approximate any continuous function on a compact domain to arbitrary accuracy. **Caveat:** "enough neurons" may be unbounded; in practice, depth is exponentially more efficient than width.

**A4. C.** Kaiming / He initialization ($\text{Var}(W) = 2/n_{\text{in}}$) accounts for ReLU zeroing half the inputs on average, keeping activation variance stable through depth. Xavier is designed for symmetric activations like tanh; zeros/constants break symmetry.

**A5.** With all weights equal to zero (or any constant), every neuron in a layer computes the same output and receives the same gradient. They update identically forever — the layer has no more expressive power than a single neuron per row. Random initialization breaks this symmetry so neurons can specialize.

**A6. B.** $\sigma'(z) = \sigma(z)(1-\sigma(z))$ peaks at $0.25$ (when $\sigma(z) = 0.5$). Backpropagating through $L$ sigmoid layers multiplies $L$ such factors — an exponentially decaying signal. First layers' gradients become numerically indistinguishable from zero, and they stop learning.

**A7.** $(p \cdot H_1 + H_1) + (H_1 \cdot H_2 + H_2) + (H_2 \cdot K + K)$. Each layer contributes $n_{\text{in}} \cdot n_{\text{out}}$ weights + $n_{\text{out}}$ biases.

**A8. A.** Without a bias, the activation is centered at $\phi(0)$; the model cannot shift the decision boundary or activation region away from the origin. Modern architectures do use biases in dense layers (sometimes omitted after normalization layers because it's redundant).

**A9.** A deep network can represent hierarchical, compositional functions — the layers reuse each other's intermediate features. Approximating the same function with a shallow (2-layer) network can require exponentially many hidden units. Deep architectures exploit compositional structure that many real-world tasks (vision, language) actually have.

**A10. A.** All classic sources of NaN. Learning rate too high → weights blow up. Poor initialization (or none) → activations explode. No gradient clipping in RNNs / transformers → gradient norm spikes. Fixes: reduce learning rate, use Kaiming/Xavier init, add gradient clipping.

---

## 6. Practice Prompts

1. **From scratch in NumPy.** Implement a 2-layer MLP with ReLU + softmax that fits XOR (using $H=4$ hidden units). Verify it learns.
2. **Parameter counting.** For a network with input dim 784, hidden layers of size 512 and 256, and output size 10, count the exact number of parameters.
3. **Symmetry demo.** Initialize a 2-layer network with all weights = 0.01 (same value). Train and observe that all hidden neurons develop identical weights.
4. **Vanishing gradients.** Build a 20-layer feedforward net with sigmoid activations. Plot the mean gradient norm per layer after one forward+backward pass. Observe the exponential decay.
5. **Fix it.** Replace sigmoid with ReLU + Kaiming init + BatchNorm. Repeat step 4. Watch gradients survive.

---

## 7. References

- Goodfellow, Bengio & Courville, *Deep Learning* (free online) — chapter 6.
- Michael Nielsen, *Neural Networks and Deep Learning* (free online) — chapters 1–2.
- Andrej Karpathy, ["A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/) (2019).
- Chris Olah, ["Neural Networks, Manifolds, and Topology"](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/).
- Cybenko (1989), Hornik (1991) — universal approximation.
