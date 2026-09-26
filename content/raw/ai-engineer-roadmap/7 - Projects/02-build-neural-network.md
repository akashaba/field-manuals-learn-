# Capstone Project 02 — Build a Neural Network (NumPy then PyTorch)

> **Deliverable:** A working image classifier for MNIST (handwritten digits, 10 classes) built **twice**: once from scratch in NumPy so you understand forward/backward propagation at the level of matrix operations, and once in PyTorch so you produce something modern and trainable on real hardware. You end with a small CNN reaching ≥98% test accuracy and a Colab notebook you can screen-share in interviews.
>
> **Time:** 6–8 hours; NumPy version 3–4h, PyTorch version 2–3h, evaluation & write-up 1h.
>
> **What you'll be able to say:** "I implemented backprop by hand — I know why the gradient of softmax + cross-entropy simplifies to `p - y`, and I know why ReLU deals with the vanishing-gradient problem better than sigmoid. Then I ported to PyTorch and added convolutions to reach 98.5% test accuracy."

---

## 1. Project Overview

We build the same classifier twice:

- **Part A** — 2-layer MLP in raw NumPy, trained on MNIST. Forward pass, backprop, mini-batch SGD, all hand-derived.
- **Part B** — Same MLP in PyTorch. Then upgrade to a small CNN. Add data augmentation, learning-rate scheduling, and TensorBoard logging.

The parallel structure is deliberate: doing it in NumPy first forces you to internalize the math; PyTorch after gives you the fluent tool.

### Architecture
![IMG-CAP02-01](/7%20-%20Projects/images/IMG-CAP02-01.jpg)


### Prerequisites

- Python 3.11+, NumPy, matplotlib
- PyTorch (`pip install torch torchvision`)
- Optional: TensorBoard (`pip install tensorboard`)
- 4 GB RAM (CPU works fine for MNIST; GPU is a nice-to-have)

---

## 2. Part A — Neural Network in Pure NumPy

### Step 1 — Load MNIST

```python
"""numpy_mlp.py"""
import numpy as np
from torchvision.datasets import MNIST

# Load once and cache as .npy
train = MNIST("./data", train=True, download=True)
test = MNIST("./data", train=False, download=True)

X_train = train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_train = train.targets.numpy()
X_test = test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test = test.targets.numpy()

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)
```

**Why one-hot?** Cross-entropy loss expects a probability distribution over classes. `[0,0,0,1,0,0,0,0,0,0]` is a distribution with all mass on class 3.

### Step 2 — Initialize weights (He initialization)

```python
def init_params(input_dim=784, hidden=128, output=10, seed=42):
    rng = np.random.default_rng(seed)
    # He init: stddev = sqrt(2/fan_in) — scaled for ReLU
    W1 = rng.standard_normal((input_dim, hidden)) * np.sqrt(2/input_dim)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, output)) * np.sqrt(2/hidden)
    b2 = np.zeros(output)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
```

**Why He init?** Zero-init breaks symmetry (all neurons compute the same thing and learn identically). Uniform-random too-large blows up activations; too-small kills them. He init keeps variance ~constant through ReLU-activated layers — variance in ≈ variance out.

### Step 3 — Forward pass

```python
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # Subtract max for numerical stability — softmax is shift-invariant
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]    # (N, 128)
    A1 = relu(Z1)                            # (N, 128)
    Z2 = A1 @ params["W2"] + params["b2"]    # (N, 10)
    P = softmax(Z2)                          # (N, 10)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "P": P}
    return P, cache
```

**Why subtract max before exp?** `exp(1000)` overflows to inf. `exp(1000-1000)=1` is fine. Softmax is invariant under adding a constant, so this is mathematically identical but numerically safe.

### Step 4 — Loss

```python
def cross_entropy(P, Y):
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))
```

### Step 5 — Backpropagation (the interesting part)

Given softmax followed by cross-entropy loss, the gradient at the pre-softmax layer simplifies **beautifully** to `P - Y`. This is the identity every deep-learning engineer knows.

```python
def backward(cache, Y, params):
    """Return gradients of loss w.r.t. each parameter."""
    N = Y.shape[0]
    P = cache["P"]

    # dL/dZ2 = (P - Y) / N   ← the softmax+CE simplification
    dZ2 = (P - Y) / N        # (N, 10)

    # dL/dW2 = A1^T @ dZ2
    dW2 = cache["A1"].T @ dZ2                 # (128, 10)
    db2 = dZ2.sum(axis=0)                     # (10,)

    # Backprop through W2 to A1
    dA1 = dZ2 @ params["W2"].T                # (N, 128)

    # Backprop through ReLU: derivative is 1 where Z1 > 0, else 0
    dZ1 = dA1 * (cache["Z1"] > 0)             # (N, 128)

    dW1 = cache["X"].T @ dZ1                  # (784, 128)
    db1 = dZ1.sum(axis=0)                     # (128,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

**Why `P - Y`?** Full derivation: L = -Σ y_i log(p_i); p_i = exp(z_i) / Σ exp(z_j); take ∂L/∂z_k; through calculus, the exp() and log() cancel and you're left with p_k - y_k. This is why softmax and cross-entropy are almost always paired — the combined gradient is simple and cheap.

**Why ReLU's derivative is `Z > 0`?** ReLU is `max(0, x)`. Its derivative is 1 for positive inputs, 0 for negative. This gates the gradient: "did this neuron contribute?"

### Step 6 — Training loop

```python
def sgd_step(params, grads, lr=0.1):
    for k in params:
        params[k] -= lr * grads[k]

def accuracy(P, y):
    return (P.argmax(axis=1) == y).mean()

def train(epochs=20, batch_size=64, lr=0.1):
    params = init_params()
    n = len(X_train)
    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            Xb, Yb = X_train[b], Y_train[b]
            P, cache = forward(Xb, params)
            loss = cross_entropy(P, Yb)
            grads = backward(cache, Yb, params)
            sgd_step(params, grads, lr)
            losses.append(loss)

        # Eval
        P_test, _ = forward(X_test, params)
        test_acc = accuracy(P_test, y_test)
        print(f"Epoch {epoch+1}: train_loss={np.mean(losses):.4f}, test_acc={test_acc:.4f}")

    return params

if __name__ == "__main__":
    params = train()
```

Expected outcome: ~97.5% test accuracy after 20 epochs, a few minutes on CPU. First epoch will look like ~92%; you're watching backprop actually work.

### Step 7 — Gradient check (the sanity test that catches bugs)

If your gradients are wrong, training still "works" — just slowly and to a worse minimum. Numeric gradient check is how you verify:

```python
def numerical_gradient(params, key, i, j, eps=1e-5):
    """Compute ∂L/∂params[key][i,j] by finite differences."""
    orig = params[key][i, j]
    params[key][i, j] = orig + eps
    P, _ = forward(X_train[:128], params)
    loss_plus = cross_entropy(P, Y_train[:128])
    params[key][i, j] = orig - eps
    P, _ = forward(X_train[:128], params)
    loss_minus = cross_entropy(P, Y_train[:128])
    params[key][i, j] = orig
    return (loss_plus - loss_minus) / (2 * eps)

# After a forward+backward, compare a few analytic gradients to numeric
params = init_params()
P, cache = forward(X_train[:128], params)
grads = backward(cache, Y_train[:128], params)

for _ in range(5):
    i, j = np.random.randint(128), np.random.randint(10)
    numeric = numerical_gradient(params, "W2", i, j)
    analytic = grads["W2"][i, j]
    rel_err = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-12)
    print(f"W2[{i},{j}]: numeric={numeric:.6f}, analytic={analytic:.6f}, rel_err={rel_err:.2e}")
```

You want `rel_err < 1e-5`. Higher means the analytic gradient is wrong.

---

## 3. Part B — Same Task in PyTorch

### Step 8 — Simple MLP in PyTorch

```python
"""torch_mlp.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
])
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=1000, num_workers=2)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # No softmax here — CrossEntropyLoss applies it internally

def train_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)

model = MLP().to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    tl = train_epoch(model, train_loader, opt, criterion)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}: loss={tl:.4f}, test_acc={acc:.4f}")
```

**Why is this the "same" as Part A?** Because PyTorch's autograd builds the identical computational graph. `CrossEntropyLoss` = log_softmax + NLL. The backprop is the same derivation you did by hand, autograd just tracks it automatically.

**Why no explicit softmax at the end?** `nn.CrossEntropyLoss` expects raw logits, applies log-softmax internally, and uses a numerically stable implementation. Applying softmax then feeding to log() would be redundant and less stable.

### Step 9 — Upgrade to a CNN

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 → 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 → 7
        x = x.view(x.size(0), -1)               # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

**Why convolutions?** MLPs treat every pixel independently — they must learn from scratch that pixel (0,0) is near (0,1). Convs bake in **translation equivariance** and **local connectivity**: a filter that detects an edge in one region detects it anywhere. Result: dramatically better sample efficiency for image tasks.

**Why max pooling?** Reduces spatial dimensions (compute cheaper), and adds mild translation invariance — a digit shifted by 1 pixel gives the same pooled output.

**Why dropout?** Regularization: randomly zeros half the neurons during training. Forces the network not to rely on any specific one — like ensembling many sub-networks.

### Step 10 — Data augmentation

```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

Small rotations and shifts create more effective training data — the model must learn features invariant to those perturbations.

### Step 11 — Learning-rate scheduling + AdamW

```python
model = SmallCNN().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    tl = train_epoch(model, train_loader, opt, criterion)
    scheduler.step()
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}: lr={opt.param_groups[0]['lr']:.5f}, loss={tl:.4f}, test_acc={acc:.4f}")
```

**Why AdamW instead of SGD?** Adam adapts per-parameter learning rates from running estimates of gradient moments — usually converges faster and is less sensitive to LR choice. The "W" fixes a bug in vanilla Adam where weight decay interacts badly with the moment estimates.

**Why cosine annealing?** Starts at full LR, gradually decreases to near-zero. Empirically better final accuracy than step decay — the model does coarse learning early, fine polishing late.

Expected result: **98.5%+** test accuracy in 20 epochs.

### Step 12 — TensorBoard logging

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist-cnn")

for epoch in range(20):
    tl = train_epoch(model, train_loader, opt, criterion)
    acc = evaluate(model, test_loader)
    writer.add_scalar("train/loss", tl, epoch)
    writer.add_scalar("test/accuracy", acc, epoch)
    writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
    # Log first-layer filters
    filters = model.conv1.weight.detach().cpu()
    writer.add_images("conv1_filters", filters, epoch)

writer.close()
```

Run `tensorboard --logdir=runs`. You can literally see the filters change from noise into edge and stroke detectors as training progresses.

`[IMG-CAP02-02]` — *Prompt: A visualization of the first conv layer's 32 filters after training an MNIST CNN. Show a 4×8 grid of small 3×3 filter images, each rendered as a heatmap. Some show diagonal edge patterns, some show center-surround activation, some show horizontal or vertical stripes. Below the grid, a training loss curve dropping from 0.5 to 0.02, and beside it a test accuracy curve rising from 92% to 98.7%. Textbook-illustration style with monochrome filters and green accuracy line.*

---

## 4. Comparison Table

| Aspect | NumPy MLP | PyTorch MLP | PyTorch CNN |
|--------|-----------|-------------|-------------|
| Test accuracy | ~97.5% | ~97.8% | 98.5%+ |
| Params | ~101k | ~101k | ~421k |
| Training time (CPU) | ~5 min | ~3 min | ~12 min |
| Lines of code | ~80 | ~40 | ~50 |
| GPU support | No | Yes | Yes |
| Autograd | Manual | Automatic | Automatic |

---

## 5. Why Each Thing Matters (Interview Ready)

- **Weight init:** wrong init = no training or exploding gradients. He init keeps ReLU activations well-scaled.
- **ReLU vs sigmoid:** sigmoid saturates (gradient → 0 for large |x|), causing vanishing gradients in deep nets. ReLU's gradient is 1 for positive inputs — trains deep networks.
- **Softmax + CE:** the loss gradient collapses to `p - y`, making backprop cheap and numerically stable.
- **Momentum:** SGD alone bounces around; momentum accumulates a velocity vector, damping oscillation.
- **Adam:** per-parameter adaptive learning rates — a "reasonable default" that works without careful LR tuning.
- **Weight decay:** L2 penalty on weights; discourages large weights = better generalization.
- **Dropout:** approximates ensembling by zeroing random neurons during training.
- **Batch normalization** (not in this project but should know): normalizes activations per mini-batch; hugely stabilizes deep-net training. LayerNorm is its cousin used in Transformers.

---

## 6. Common Bugs and How to Diagnose

- **Loss doesn't decrease** → learning rate too high (loss NaNs or oscillates) or too low (stagnates). Try LR range 10× lower and 10× higher.
- **Train acc high, test acc low** → overfitting. Add dropout, weight decay, data augmentation, or reduce model size.
- **Train acc stuck at chance (10%)** → check target encoding, loss function selection, model output dimensionality, and that gradients are flowing (any layer with all-zero grads?).
- **Test acc suspiciously high (99%+)** → check for data leakage: were test samples in your training set? Did you normalize test data using train stats or independently?
- **Loss NaN after some steps** → almost always exp/log overflow. Use log-softmax + NLL instead of softmax + log().

---

## 7. Extensions

- Try Fashion-MNIST (harder; same code)
- Add a residual connection (skip conv1 → block2 output); note the training stability gain
- Move to CIFAR-10 (10-class 32×32 color); MLP won't cut it, CNN needs more depth
- Add mixed-precision training (`torch.cuda.amp`); 2× faster on GPU with almost no accuracy loss

---

## 8. Interview Talking Points

If asked "explain a neural network you built":

1. **What backprop actually is.** "For each parameter, compute the gradient of loss with respect to it using the chain rule; update in the negative-gradient direction. I did it by hand in NumPy."
2. **Why softmax+CE.** "The gradient at the pre-softmax activations is just `predicted probs minus one-hot target`. Simple, cheap, numerically stable."
3. **Why He init.** "Preserves variance through ReLU layers; without it, deep MLPs saturate or explode."
4. **Convs vs dense.** "Convs bake in translation equivariance and local connectivity — a filter learned in one location applies everywhere. Sample-efficient for images."
5. **The transition to PyTorch.** "PyTorch's autograd builds the same graph my NumPy code did manually. The API lets me focus on architecture."
6. **What I'd do next.** "Modern MNIST work uses residual blocks and heavy augmentation. I'd try a small ResNet next; also CIFAR-10 to force real depth."

---

## 9. References

- Michael Nielsen, *Neural Networks and Deep Learning* (free, http://neuralnetworksanddeeplearning.com/)
- Stanford CS231n lecture notes — convolutional networks, backprop
- Kaiming He et al., "Delving Deep into Rectifiers" — He init paper
- Ioffe & Szegedy, "Batch Normalization" (2015)
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" — AdamW
