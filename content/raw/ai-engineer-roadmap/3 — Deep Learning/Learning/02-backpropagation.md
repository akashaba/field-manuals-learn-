# Backpropagation — Master Study Guide

> **Track:** Deep Learning · **Module:** 02
> **Prerequisites:** Module 01, comfortable with the chain rule and matrix calculus.
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Backpropagation is **the single algorithm that made deep learning practical.** Before backprop was popularized (Rumelhart, Hinton, Williams 1986), training multi-layer networks was possible in principle but computationally impossible in practice. Backprop is what makes gradient descent tractable on models with millions or billions of parameters — it computes all gradients in a single sweep of the network, at the same cost as one forward pass.

You will rarely implement backprop by hand — PyTorch's autograd does it for you. But you must understand it, because:

1. When training breaks, you need to reason about what gradients are actually flowing.
2. Architecture design (residuals, LayerNorm, attention) is fundamentally about making gradients well-behaved.
3. New architectures are *invented* by playing with computational graphs.

**Fundamental principles you must own:**

1. **Backpropagation is the chain rule applied to computational graphs.** Nothing more mysterious.
2. **Every operation has a local gradient**; backprop composes them by multiplying local gradients along the reverse path.
3. **The cost of the backward pass ≈ the cost of the forward pass** — that's the miracle.
4. **Autograd tracks the graph automatically** in PyTorch; you write forward code, and gradients materialize.
5. **Gradient checking** — comparing analytical gradients to numerical ones — is the debugging technique to prove your derivation is correct.
6. **Modern issues (vanishing/exploding grads, dead ReLUs, saturation)** all trace back to which gradients backprop is producing.

If you retain nothing else: **backprop = reverse-mode automatic differentiation = the chain rule on a graph.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Chain Rule and Computational Graphs

Recall the univariate chain rule: if $y = f(g(x))$ then $\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$.

Extended to compositions: $y = f_3(f_2(f_1(x)))$:

$$\frac{dy}{dx} = f_3'(f_2(f_1(x))) \cdot f_2'(f_1(x)) \cdot f_1'(x)$$

A **computational graph** is a DAG where nodes are operations (or intermediate variables) and edges show data flow. Every neural network is a computational graph.

**Example** — a simple graph for $z = (a + b) \cdot c$:

```
    a ──┐
        ├── + ── s ──┐
    b ──┘             ├── * ── z
                  c ──┘
```

Given a scalar loss at the end, backprop computes $\partial L / \partial a$, $\partial L / \partial b$, $\partial L / \partial c$ by propagating from the output backward, multiplying local gradients at each node.

**Forward pass** (per operation):
- Compute $s = a + b$.
- Compute $z = s \cdot c$.

**Backward pass** (per operation, in reverse order):
- Given $\partial L / \partial z$ (from downstream).
- Local gradient of $z = s \cdot c$: $\partial z / \partial s = c$, $\partial z / \partial c = s$.
- So $\partial L / \partial s = \partial L / \partial z \cdot c$, $\partial L / \partial c = \partial L / \partial z \cdot s$.
- Local gradient of $s = a + b$: $\partial s / \partial a = 1$, $\partial s / \partial b = 1$.
- So $\partial L / \partial a = \partial L / \partial s \cdot 1$, $\partial L / \partial b = \partial L / \partial s \cdot 1$.

The **gradient flows through every node**, being multiplied by the local Jacobian. This generalizes to any graph.

---

### 2.2 Backprop Through a Fully-Connected Layer

Consider one dense layer:

$$\mathbf{z} = W\mathbf{x} + \mathbf{b}, \quad \mathbf{h} = \phi(\mathbf{z})$$

Given a scalar loss $L$ and the "upstream" gradient $\delta = \partial L / \partial \mathbf{h}$, we need $\partial L / \partial W$, $\partial L / \partial \mathbf{b}$, and $\partial L / \partial \mathbf{x}$ (to pass upstream).

**Gradient with respect to the pre-activation:**

$$\frac{\partial L}{\partial \mathbf{z}} = \delta \odot \phi'(\mathbf{z})$$

Where $\odot$ is elementwise product and $\phi'$ is the elementwise derivative of the activation.

**Gradient with respect to weights:**

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{z}} \, \mathbf{x}^\top$$

(An outer product: for each element $W_{ij}$, $\partial L/\partial W_{ij} = (\partial L/\partial z_i) \cdot x_j$.)

**Gradient with respect to bias:**

$$\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$$

**Gradient with respect to input** (for passing further upstream):

$$\frac{\partial L}{\partial \mathbf{x}} = W^\top \frac{\partial L}{\partial \mathbf{z}}$$

For a **minibatch** of $B$ examples with $X \in \mathbb{R}^{B \times p}$, $H = \phi(XW^\top + \mathbf{b})$, the gradients become:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Z}^\top X, \quad \frac{\partial L}{\partial \mathbf{b}} = \sum_i \frac{\partial L}{\partial \mathbf{z}_i}, \quad \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \, W$$

These formulas are the guts of every deep-learning library. You will run into them when implementing custom layers or debugging.

---

### 2.3 Autograd — How PyTorch Does It

PyTorch's **autograd** is reverse-mode automatic differentiation:

1. Every tensor with `requires_grad=True` becomes a node in the computational graph.
2. Every operation on such tensors is recorded — it creates a **`grad_fn`** node representing how to differentiate.
3. When you call `loss.backward()`, autograd traverses the graph in reverse topological order, calling each `grad_fn` to compute local gradients and accumulate them into the leaf tensors' `.grad` attributes.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = (x * 2).sum()
y.backward()
print(x.grad)   # tensor([2., 2., 2.]) — dy/dx = 2 for each element
```

**Key behaviors:**

- **Gradients accumulate** — `.backward()` adds to `.grad`, doesn't overwrite. That's why you must call `optimizer.zero_grad()` (or `param.grad.zero_()`) each iteration.
- **Only scalars** can call `.backward()` directly. For vector outputs, pass `gradient=torch.ones_like(y)` or reduce to a scalar first.
- **`requires_grad=False`** freezes a tensor (or subgraph) — no gradient computed, no memory spent tracking it.
- **`torch.no_grad()` context** disables graph tracking altogether — use during inference for speed and memory.
- **`.detach()`** returns a tensor sharing data but not participating in the graph — the classic way to break gradient flow deliberately.

**Under the hood:** autograd is a dynamic computational graph — it's built fresh every forward pass. This is different from static-graph frameworks (early TensorFlow) and is why PyTorch feels "Pythonic". The cost is a small runtime overhead per operation.

**`torch.compile`** (PyTorch 2.x) traces and optimizes the graph after the first forward pass, closing much of the gap with static-graph performance.

---

### 2.4 Gradient Checking

When you write custom layers (e.g., a novel activation), it's easy to derive the wrong analytical gradient. **Gradient checking** compares your analytical gradient to a numerical approximation.

**Central-difference approximation** of $f'(x)$:

$$f'(x) \approx \frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon}$$

with $\varepsilon \sim 10^{-5}$ for `float64` (larger for `float32` due to precision).

**For a parameter tensor $\theta$:** perturb each entry, recompute the loss, and finite-difference. Compare to your analytical gradient. Use relative error:

$$\text{rel\_err} = \frac{\|\text{grad}_{\text{analytic}} - \text{grad}_{\text{numeric}}\|}{\|\text{grad}_{\text{analytic}}\| + \|\text{grad}_{\text{numeric}}\|}$$

Rules of thumb:
- $< 10^{-7}$: excellent, gradient is correct.
- $10^{-4}$ to $10^{-7}$: possibly correct, worth another look.
- $> 10^{-4}$: bug.

**PyTorch tool**: `torch.autograd.gradcheck(fn, inputs)` does this rigorously for double-precision tensors. Use it whenever you implement a custom `Function`.

**Practical note:** ReLU has a discontinuous derivative at 0; gradient check at strictly positive/negative points. Losses like L1 have subgradients that require special handling.

---

### 2.5 Gradient Pathologies and Their Fixes

Modern deep learning is largely a story of taming gradient behavior in deep, wide networks.

**Vanishing gradients** — signal becomes exponentially small deep in the network.
- Cause: activations with derivatives < 1 chained through many layers.
- Fix: ReLU-family activations; skip connections (ResNet); LayerNorm; careful init.

**Exploding gradients** — signal becomes exponentially large.
- Cause: activations or weights with amplification factor > 1 chained through many layers, or a single layer with huge preactivations.
- Fix: **gradient clipping** — cap the total gradient norm:

  $$\text{if } \|\nabla\| > c: \quad \nabla \leftarrow \frac{c}{\|\nabla\|} \nabla$$

  Standard in RNNs and transformers.

**Dead ReLUs** — a neuron whose input is always $\leq 0$ has $\phi'(z) = 0$ everywhere → gradient of $0$ → never updates → stays dead forever.
- Fix: LeakyReLU, PReLU, ELU, GELU (all have non-zero derivatives for negative inputs); careful initialization; lower learning rate.

**Saturating activations** — sigmoid, tanh have derivative $\to 0$ at extremes ($|z| \gg 0$). Gradient dies.
- Fix: replace sigmoid/tanh in hidden layers with ReLU-family. Keep sigmoid for output gates (LSTM) or binary output.

**Numerical instability in softmax + log**:
- Cause: $\log \sum e^{z_i}$ can overflow if $z_i$ is large.
- Fix: **log-sum-exp trick** — subtract $\max_i z_i$ before exponentiating:

  $$\log \sum_i e^{z_i} = \max_j z_j + \log \sum_i e^{z_i - \max_j z_j}$$

  Every real deep-learning library does this.

**Recognition patterns for training bugs:**

| Symptom | Likely cause |
|---------|--------------|
| Loss → NaN | Exploding gradients; learning rate too high; unbounded activations; missing log-sum-exp |
| Loss flat, doesn't move | Vanishing gradients; dead ReLUs; learning rate too low; zero-init |
| Loss oscillates wildly | Learning rate too high; batch size too small; incompatible losses |
| Only some parameters update | Missed `.parameters()` in optimizer; frozen `requires_grad=False`; disconnected graph |

---

## 3. Mental Models & Analogies

### 3.1 The "Assembly Line with a Blame Chain" Model

Imagine a car assembly line. At the end of the line, an inspector rejects a car and hands back a **blame score**: how bad is this car? Backpropagation is how that blame flows back through the line to identify which station contributed how much to the defect.

- Each station transforms parts (**forward pass**).
- Each station has a **local rule** for how much a small change in its input would change its output (the local derivative).
- Blame flows backward: the last station's blame is multiplied by its local rule to become the previous station's incoming blame, and so on.
- By the time the blame reaches the raw material (or the settings on each station), it tells each station operator: "your setting is contributing this much to the defect. Adjust it."

**Why it's cheap:** the blame chain visits each station **exactly once**, in reverse order. The total work of blame propagation is proportional to the work of forward assembly. This is the miracle of reverse-mode differentiation.

### 3.2 The "River of Gradients" Model

Picture the loss as a river's source at the top of a valley. The parameters are individual pebbles along the valley walls. The gradient at each pebble tells you: which direction would this pebble move to make the river flow farther down?

Backpropagation is the river carving its way from the source (loss) back to every pebble (parameter). As the river flows, it splits at every fork in the graph — some water goes to the left child, some to the right — and it **combines** at joins where multiple upstream paths merge (gradient contributions from different downstream users add).

- **Dead neurons** = dams. The river can't flow past them, so nothing downstream (in the backward direction) gets water.
- **Skip connections** = bypass channels. Even if a section is dammed, water can still flow around it.
- **Gradient clipping** = a dam preventing flash floods from washing pebbles away.

Once every pebble has water on it, the optimizer nudges each pebble slightly in the water's direction. Then the river changes shape, and the cycle repeats.

![IMG-BP-01](/3%20—%20Deep%20Learning/images/IMG-BP-01.jpg)

> **Caption:** Backpropagation flows the loss gradient backward through the graph, multiplying local Jacobians at each step.
> **Placement:** Section 2.1 or Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "I Don't Need to Understand Backprop Because PyTorch Handles It"

You need to understand it *because* PyTorch handles it. When training silently fails — NaN losses, dead layers, gradients that clip to zero — the debug path leads through the computational graph. You need to:
- Recognize that a `.detach()` call breaks the gradient chain.
- Recognize that `torch.no_grad()` prevents `.backward()` altogether.
- Recognize that certain operations (`.tolist()`, `.numpy()`, indexing with integer tensors) can silently break autograd.
- Interpret warnings about "one of the variables needed for gradient computation has been modified by an inplace operation."

You can't debug what you don't understand.

### 4.2 "Forgetting to Call `optimizer.zero_grad()`"

Autograd **accumulates** gradients — each `.backward()` call adds to existing `.grad`. If you don't zero them, gradients from previous minibatches persist and your effective batch size grows unboundedly, ruining learning.

The correct pattern:

```python
optimizer.zero_grad()   # or model.zero_grad()
loss.backward()
optimizer.step()
```

This bug is invisible — losses may still decrease, just wrongly. Always include the zero_grad call.

### 4.3 "Detaching a Tensor Is the Same as `no_grad`"

Similar effects, different semantics:
- **`x.detach()`** returns a new tensor sharing data but *not* connected to the graph. Downstream ops on it don't build a graph.
- **`with torch.no_grad(): ...`** disables graph construction *for all ops in the block*.

Use `detach()` when you want to isolate a specific tensor from the graph (e.g., using a model's output as a fixed target for a loss). Use `no_grad()` for entire inference loops or evaluation blocks. Never assume they're interchangeable — the semantic differences matter for correctness.

---

## 5. Self-Assessment Bank (Backpropagation)

### Questions

**Q1 (Short answer).** State backpropagation in one sentence.

**Q2 (Multiple choice).** In a computational graph, the backward pass:
A. Traverses the graph in the same order as the forward pass.
B. Traverses the graph in reverse topological order.
C. Traverses random paths.
D. Only visits the loss node.

**Q3 (Short answer).** Given a dense layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, $\mathbf{h} = \phi(\mathbf{z})$, and upstream gradient $\delta = \partial L / \partial \mathbf{h}$, write the expressions for $\partial L / \partial W$, $\partial L / \partial \mathbf{b}$, $\partial L / \partial \mathbf{x}$.

**Q4 (Multiple choice).** In PyTorch, `x.detach()`:
A. Sets `requires_grad=True`.
B. Creates a tensor sharing the same data but with no gradient computation.
C. Removes the tensor from GPU.
D. Copies the tensor.

**Q5 (Short answer).** Why must you call `optimizer.zero_grad()` at each training iteration in PyTorch?

**Q6 (Multiple choice).** Which of these is a symptom of exploding gradients?
A. Loss stuck at initial value.
B. Loss becoming NaN.
C. Very slow decrease in loss.
D. Model predicts one class only.

**Q7 (Short answer).** What is "gradient checking" and how would you do it for a custom layer?

**Q8 (Multiple choice).** The computational cost of the backward pass is:
A. Proportional to the size of the smallest layer.
B. Roughly the same as the forward pass — the miracle of reverse-mode autodiff.
C. Exponential in depth.
D. Independent of the network's architecture.

**Q9 (Short answer).** Explain what a "dead ReLU" is and what fixes it.

**Q10 (Multiple choice).** During training, you notice your loss oscillates violently up and down. The most likely culprit is:
A. Batch size too small.
B. Learning rate too high.
C. Model too shallow.
D. Not enough epochs.

---

### Answer Key & Detailed Explanations

**A1.** Backpropagation is the algorithm that computes gradients of a scalar loss with respect to every parameter by applying the chain rule in reverse topological order over the network's computational graph.

**A2. B.** Reverse topological order — start at the loss node, visit each node only after all its downstream users have been visited, computing gradients as you go.

**A3.**
- $\partial L / \partial \mathbf{z} = \delta \odot \phi'(\mathbf{z})$ (elementwise activation derivative).
- $\partial L / \partial W = (\partial L / \partial \mathbf{z}) \, \mathbf{x}^\top$ (outer product).
- $\partial L / \partial \mathbf{b} = \partial L / \partial \mathbf{z}$.
- $\partial L / \partial \mathbf{x} = W^\top (\partial L / \partial \mathbf{z})$.

**A4. B.** `x.detach()` returns a new tensor sharing storage with `x` but disconnected from the computational graph. Operations on the detached tensor don't propagate gradients back to `x`.

**A5.** Because `.backward()` **accumulates** gradients into `.grad` — it adds rather than replaces. Without zeroing, gradients from previous minibatches persist and the effective gradient is a sum of many minibatches' gradients — the model's effective batch size grows unboundedly and training goes off the rails.

**A6. B.** Exploding gradients cause weight updates so large that activations become $\pm\infty$ or NaN. Loss becomes NaN once any preactivation overflows to `inf` and downstream ops yield `nan`.

**A7.** Gradient checking compares your analytical gradient against a numerical estimate via central differences: $f'(x) \approx (f(x+\varepsilon) - f(x-\varepsilon)) / (2\varepsilon)$ with $\varepsilon \sim 10^{-5}$. For a custom layer: perturb each parameter by $\varepsilon$, compute the loss change, and compare to the analytical gradient. Relative error < $10^{-7}$ is the pass criterion. PyTorch's `torch.autograd.gradcheck` does this rigorously.

**A8. B.** Reverse-mode automatic differentiation computes all gradients in a single reverse traversal of the graph, at cost proportional to the forward pass. This is the reason backprop is practical — otherwise, computing $M$ gradients would cost $M$× forward passes.

**A9.** A dead ReLU is a neuron whose pre-activation $z = \mathbf{w}^\top\mathbf{x} + b$ is always $\leq 0$ (across the training data). Its output is always 0, its derivative is always 0, so its gradient is always 0 — it never updates. Fixes: (1) LeakyReLU / PReLU / ELU / GELU (nonzero gradient for negative inputs), (2) Kaiming initialization keeps preactivations well-scaled at start, (3) lower learning rate reduces the chance of pushing all weights into a bad region.

**A10. B.** Oscillating loss = the optimizer is overshooting minima. Learning rate too high is the classic cause. Try `lr / 10`, add a learning-rate scheduler, or use gradient clipping. Small batch size (A) can add noise but doesn't cause pure oscillation.

---

## 6. Practice Prompts

1. **From scratch.** Implement forward and backward passes for a 2-layer MLP with ReLU + softmax + cross-entropy in NumPy. No autograd. Verify against PyTorch to numerical tolerance.
2. **Gradient checking.** Write a `gradcheck` function that uses central differences to verify your NumPy gradients from prompt 1. Run it.
3. **Autograd exploration.** In PyTorch, build a 3-layer network with `nn.Sequential`. Run one forward, one `backward()`. Print `layer.weight.grad` for each layer. Confirm the shapes match the weights.
4. **Zero-grad bug demo.** Write a training loop that *omits* `optimizer.zero_grad()`. Train for a few iterations. Compare the loss curve to a correct version. Notice the divergence.
5. **Detach vs no_grad.** Build a small example (e.g., teacher-student where the teacher's output is a target) using each. Observe the graph and gradient flow differences.

---

## 7. References

- Rumelhart, Hinton & Williams, ["Learning representations by back-propagating errors"](https://www.nature.com/articles/323533a0) (Nature, 1986).
- Andrej Karpathy, ["Yes, you should understand backprop"](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) (2016).
- Chris Olah, ["Calculus on Computational Graphs: Backpropagation"](https://colah.github.io/posts/2015-08-Backprop/).
- Goodfellow et al., *Deep Learning*, chapter 6.5.
- PyTorch autograd docs: [https://pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html).
