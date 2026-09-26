# PyTorch — Master Study Guide

> **Track:** Deep Learning · **Module:** 12
> **Prerequisites:** Modules 01–11 (theory) + Foundations Module 01 (Python).
> **Time budget:** ~12–15 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** PyTorch is **the** dominant deep-learning framework in 2026. It powers:

- Almost every recent research paper.
- HuggingFace transformers, Diffusers, Accelerate.
- Meta's, Anthropic's, and much of Google DeepMind's internal stacks.
- Every LLM and multimodal model you'll use.

Its Pythonic API, dynamic computational graph, and vast ecosystem make it the tool to master. TensorFlow is still around, JAX is hot in research, but for practical DL work in 2026, PyTorch fluency is table stakes.

**Fundamental principles you must own:**

1. **Tensors are NumPy arrays with autograd, GPU support, and a modeling API.**
2. **`nn.Module` is the base class for all neural networks** — subclass it, register parameters via child modules or `nn.Parameter`.
3. **Autograd** — automatic differentiation via a dynamic computational graph.
4. **DataLoader** — the standard data-feeding pipeline: `Dataset` → `DataLoader` → mini-batches.
5. **Training loop = zero_grad → forward → loss → backward → step.**
6. **Move everything (model, tensors) to the same device (GPU); mixed precision + gradient scaling for speed.**

If you retain nothing else: **PyTorch = tensors + autograd + nn.Module + DataLoader + training loop.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Tensors, Devices, and Dtypes

**Tensor creation:**

```python
import torch

a = torch.tensor([1., 2., 3.])              # 1D float32 tensor
b = torch.zeros(3, 4)                       # 3x4 zeros
c = torch.randn(3, 4)                       # standard normal
d = torch.arange(10)                        # [0,1,...,9] int64
e = torch.linspace(0, 1, 100)               # 100 points evenly spaced

# from numpy — shares memory with the array
np_arr = np.array([1., 2., 3.])
t = torch.from_numpy(np_arr)
```

**Operations** — every NumPy op has a PyTorch equivalent, plus more:

```python
c.matmul(d)         # or c @ d
c.reshape(4, 3)
c.transpose(0, 1)   # or c.T for 2D
c.sum(dim=1)
c.mean(dim=-1, keepdim=True)
c[:, 0]             # slicing works
c.unsqueeze(0)      # add a size-1 axis at position 0
c.squeeze()         # remove all size-1 axes
c.expand(3, 5, 4)   # broadcast to size, no memory copy
```

**Broadcasting** — identical to NumPy (see Foundations Module 02).

**Dtypes:**

- `torch.float32` (default), `torch.float64`, `torch.float16`, `torch.bfloat16`.
- `torch.int8`, `torch.int32`, `torch.int64`, `torch.bool`.
- Modern GPU training: **bf16** for compute (mixed precision), fp32 for weights.

**Device management:**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)                            # move to GPU
model = model.to(device)                    # move model
```

**Rules:**
- All operands to an operation must be on the same device.
- Moving between devices is expensive (goes over PCIe or NVLink).
- Prefer creating tensors on device directly: `torch.zeros(..., device="cuda")`.

**`requires_grad`** — enable gradient tracking:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)   # tensor([2., 4.])
```

**Detach & no_grad:**

```python
x.detach()            # new tensor, same data, no graph
with torch.no_grad():
    y = model(x)      # forward without building the graph — for inference
```

---

### 2.2 `nn.Module` and Building Networks

Everything you write in PyTorch is (or should be) an `nn.Module`.

**Anatomy:**

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

**What `nn.Module` gives you:**

- **`.parameters()`** — iterator over all trainable tensors (needed by the optimizer).
- **`.state_dict()`** — dict of all parameters + buffers for saving/loading.
- **`.train()`** and **`.eval()`** — set mode for dropout / batchnorm.
- **`.to(device)`** — recursively move everything to a device.
- **`.zero_grad()`** — zero gradients of all parameters.
- **Module tree traversal** — child modules registered automatically when assigned as attributes.

**Registration rules:**

- Assigning an `nn.Module` (like `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`) as an attribute automatically registers it as a submodule.
- Assigning an `nn.Parameter` registers it as a learnable parameter.
- Assigning a `torch.Tensor` (not `nn.Parameter`) does **NOT** register it — it won't be moved to GPU or included in `state_dict`. Use `self.register_buffer("name", tensor)` for non-learnable state.

**Common containers:**

- **`nn.Sequential`** — chain of modules applied in order. Handy for simple stacks.
- **`nn.ModuleList`** — Python list of modules that's properly registered. Use when iterating in `forward`.
- **`nn.ModuleDict`** — dict version.

**Standard layers:**

- `nn.Linear`, `nn.Conv1d/2d/3d`, `nn.ConvTranspose2d`.
- `nn.BatchNorm1d/2d/3d`, `nn.LayerNorm`, `nn.GroupNorm`, `nn.RMSNorm` (2.4+).
- `nn.Embedding`, `nn.EmbeddingBag`.
- `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Tanh`, `nn.Sigmoid`.
- `nn.Dropout`, `nn.Dropout2d`.
- `nn.LSTM`, `nn.GRU`, `nn.RNN`.
- `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`.

**Initialization** — done via `torch.nn.init`:

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
```

---

### 2.3 Data Pipeline: Dataset, DataLoader, Transforms

**`torch.utils.data.Dataset`** — subclass with `__len__` and `__getitem__`:

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
```

**`DataLoader`** — batching, shuffling, parallel loading:

```python
train_loader = DataLoader(
    MyDataset(x, y),
    batch_size=128,
    shuffle=True,
    num_workers=4,          # parallel data-loading workers
    pin_memory=True,         # faster GPU transfer
    persistent_workers=True, # reuse workers across epochs
    drop_last=True,          # drop the last incomplete batch
)

for x, y in train_loader:
    # x shape: (128, ...); y shape: (128, ...)
    ...
```

**Custom `collate_fn`** — combine samples into a batch. Default stacks along dim 0. For variable-length sequences, write your own:

```python
def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.stack(ys)
    return xs, ys
```

**Torchvision datasets and transforms:**

```python
from torchvision import datasets, transforms

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
```

**HuggingFace `datasets`** — for text, complements Torchvision:

```python
from datasets import load_dataset
ds = load_dataset("imdb", split="train")
ds = ds.map(lambda ex: tokenizer(ex["text"], truncation=True), batched=True)
ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

---

### 2.4 Training Loop (with modern touches)

A production-ready loop:

```python
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

model     = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
criterion = nn.CrossEntropyLoss()
scaler    = GradScaler()

best_val_loss = float("inf")

for epoch in range(epochs):
    # ---------- train ----------
    model.train()
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.bfloat16):
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # ---------- validate ----------
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast(dtype=torch.bfloat16):
                logits = model(x)
                loss   = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            correct  += (logits.argmax(-1) == y).sum().item()
            total    += x.size(0)
    val_loss /= total
    val_acc  = correct / total

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }, "best.pt")

    print(f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
```

**Key points:**

- **`zero_grad(set_to_none=True)`** — slightly faster than the default (sets grads to None instead of zero tensors).
- **`non_blocking=True`** on `.to(device)` — async transfer if `pin_memory=True` in the DataLoader.
- **`autocast`** — mixed precision. `bfloat16` is more stable than `float16`; on Ampere/Hopper GPUs it's practically free.
- **`clip_grad_norm_`** — gradient clipping. Cheap insurance against NaN losses in RNN/transformer training.
- **`no_grad` + `.eval()`** — during evaluation. Skips graph tracking, saves memory, faster.
- **Checkpoint everything** needed to resume: model, optimizer, scheduler, scaler, epoch.

**Reproducibility:**

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import numpy as np, random
np.random.seed(42); random.seed(42)
# For strict determinism (slower):
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

---

### 2.5 Debugging, Profiling, and Production Concerns

**Debugging shape / dtype issues:**

- Add `print(x.shape, x.dtype, x.device)` between suspected layers.
- Check that inputs match model's expected shape.
- Watch for accidental broadcasting bugs (a `(B, C)` × `(C,)` you thought was a matmul).

**Debugging NaN losses:**

- Check learning rate — too high causes explosion.
- Add gradient clipping.
- Check for `log(0)` — softmax followed by manual `log` can produce NaN. Use `F.log_softmax` or the fused loss (`nn.CrossEntropyLoss`).
- Print `torch.isnan(x).any()` at each layer's output to isolate.
- Use `torch.autograd.set_detect_anomaly(True)` during debugging (slow, but finds the offending op).

**Debugging OOM (out of memory):**

- Reduce batch size, or use **gradient accumulation** (Module 05).
- Enable **mixed precision** (bf16).
- Use **gradient checkpointing** — `torch.utils.checkpoint.checkpoint(fn, x)` recomputes intermediate activations during backward instead of storing them.
- Watch for memory leaks — keeping references to loss tensors across iterations.
- `torch.cuda.empty_cache()` — releases cached but unused memory (may help occasionally).

**Profiling:**

```python
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
) as prof:
    for i, batch in enumerate(loader):
        train_step(batch)
        prof.step()
        if i >= 5: break
```

Then view in TensorBoard.

**`torch.compile`** (PyTorch 2.0+) — Just-In-Time compilation of your model for speed:

```python
model = MyModel()
model = torch.compile(model)   # or torch.compile(model, mode="reduce-overhead")
```

Common gains: 1.5×–3× for transformer training. First forward pass is slow (compilation); subsequent are much faster.

**Multi-GPU:**

- **`DataParallel`** (deprecated) — single-process, multi-GPU. Simple but slow.
- **`DistributedDataParallel` (DDP)** — one process per GPU; the standard choice.
- **`FSDP` (Fully Sharded Data Parallel)** — shards parameters, gradients, optimizer states across GPUs. Enables training models larger than any single GPU's memory.

**Model serialization:**

- **Save state_dict** (recommended): `torch.save(model.state_dict(), "model.pt")`. Load: `model.load_state_dict(torch.load("model.pt"))`.
- **Full model pickle** (fragile): `torch.save(model, "model.pt")` — depends on class definitions.
- **TorchScript** — traceable / scripted model for deployment.
- **ONNX** — cross-framework export via `torch.onnx.export`.

**Production inference:**

- `model.eval()` + `torch.no_grad()`.
- Quantize with `torch.quantization` or export to a serving framework (TorchServe, Triton, ONNX Runtime).
- Batch requests; use `torch.compile` for speedup.

---

## 3. Mental Models & Analogies

### 3.1 The "Everything Is a Module" Model

Think of every piece of a neural network as a **LEGO brick that speaks a common language** (the `nn.Module` API). A brick has:

- Slots for inputs (`forward(x)`).
- Learnable parameters registered as its properties.
- Sub-bricks nested inside it.

Big models are just towers of bricks — attention blocks contain projections; a transformer contains blocks; a full LLM contains a transformer plus embeddings. Because they all speak `nn.Module`, moving to GPU, saving weights, computing gradients — everything works uniformly regardless of which bricks are stacked.

This is why PyTorch scales cleanly from a 100-line MLP to a 100k-line LLM stack: the abstraction stays the same. New models are new arrangements of the same bricks.

### 3.2 The "Assembly Line + Blueprint" Model (Training Loop)

Training is a factory assembly line:

- **DataLoader** is the raw-materials conveyor — batches roll off in sequence, in parallel workers.
- **Model** is the machine tooling that transforms materials into products (predictions).
- **Loss + optimizer** is the QA station — every batch's error tells the machine how to adjust itself.
- **Scheduler** is the shift manager — tweaks the pace of adjustments over time.
- **Checkpoints** are periodic snapshots of the assembly line's configuration, so if the factory burns down, you can rebuild it exactly.

**Autograd** is the amazing bit: as you install any new machine, you write the forward pass; the framework automatically figures out how to compute adjustments — no manual calculus, no bookkeeping. You write "transform x this way"; PyTorch figures out "if you want less error, adjust these parameters that way."

The training loop's five stages — `zero_grad → forward → loss → backward → step` — are the assembly line's rhythm. Miss any of them and the whole factory jams. Skip `zero_grad` and errors accumulate. Skip `backward` and no adjustment happens. Skip `step` and adjustments are computed but never applied.

![IMG-PT-01](/3%20—%20Deep%20Learning/images/IMG-PT-01.jpg)

> **Caption:** The canonical PyTorch training step: forward → loss → backward → step → zero_grad.
> **Placement:** Section 2.4.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Forgetting `model.eval()` / `.train()`"

Dropout and BatchNorm behave differently in training vs eval mode. Without `.eval()` at inference, dropout randomly zeros activations and BatchNorm uses noisy batch statistics — predictions become nondeterministic and lower quality. Without `.train()` before a fresh training run, dropout is off and the model doesn't regularize. **Rule: `model.train()` at the top of every training epoch; `model.eval()` before every evaluation or inference block.**

### 4.2 "Assigning a Tensor as an Attribute Registers It"

Only `nn.Module` (submodules) and `nn.Parameter` are auto-registered. Assigning a plain `torch.Tensor` (like `self.positional_encoding = torch.arange(...)`) does **not** register it — it won't move to GPU with `.to(device)`, won't appear in `state_dict()`, won't be saved. Use `self.register_buffer("positional_encoding", torch.arange(...))` for constant state that should follow the model.

### 4.3 "`torch.tensor(np_array)` and `torch.from_numpy(np_array)` Are the Same"

They aren't. `torch.tensor(x)` copies data; `torch.from_numpy(x)` shares memory. If you modify the NumPy array afterward, `torch.from_numpy`'s tensor sees the change; `torch.tensor`'s does not. For read-only interop, prefer `from_numpy` to avoid unnecessary copies.

---

## 5. Self-Assessment Bank (PyTorch)

### Questions

**Q1 (Short answer).** In one sentence, what does `nn.Module` provide that plain Python classes don't?

**Q2 (Multiple choice).** After `x = torch.tensor([1., 2., 3.], requires_grad=True); y = (x*x).sum(); y.backward()`, what does `x.grad` contain?
A. tensor([1., 2., 3.])
B. tensor([2., 4., 6.])
C. tensor([1., 1., 1.])
D. It raises an error.

**Q3 (Short answer).** What does `optimizer.zero_grad()` do, and why is it necessary?

**Q4 (Multiple choice).** Which is the recommended way to save a PyTorch model?
A. `torch.save(model, "path")`
B. `torch.save(model.state_dict(), "path")`
C. `pickle.dump(model, file)`
D. `json.dump(model, file)`

**Q5 (Short answer).** Explain what `torch.no_grad()` does and when you'd use it.

**Q6 (Multiple choice).** During evaluation, forgetting to call `model.eval()` typically causes:
A. A NaN loss.
B. Silent nondeterministic predictions (dropout stays active, BN uses batch stats).
C. An immediate error.
D. Slower inference.

**Q7 (Short answer).** What is the difference between `set_to_none=True` and `set_to_none=False` in `optimizer.zero_grad()`?

**Q8 (Multiple choice).** In a training loop with mixed precision (`autocast` + `GradScaler`), the correct order is:
A. `loss.backward()` → `scaler.step()` → `scaler.update()`.
B. `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()`.
C. `loss.backward()` → `optimizer.step()`.
D. `scaler.step()` → `loss.backward()`.

**Q9 (Short answer).** Why register a constant tensor via `self.register_buffer(...)` rather than as a regular attribute?

**Q10 (Multiple choice).** `torch.compile(model)` in PyTorch 2.x:
A. Converts the model to ONNX.
B. Just-in-time compiles the forward pass for speed; first call is slow (compilation), subsequent calls faster.
C. Freezes all parameters.
D. Reduces model size.

---

### Answer Key & Detailed Explanations

**A1.** `nn.Module` provides automatic parameter registration, recursive `.to(device)`/`.parameters()`/`.state_dict()`/`.train()`/`.eval()` traversal over submodules, and integration with the autograd/optimizer/loss ecosystem.

**A2. B.** $y = \sum x_i^2$; $\partial y / \partial x_i = 2 x_i$; for $x = [1, 2, 3]$, gradient is $[2, 4, 6]$.

**A3.** `optimizer.zero_grad()` zeros the `.grad` attribute of every parameter the optimizer manages. Necessary because PyTorch's autograd **accumulates** gradients — each `backward()` adds to `.grad` rather than replacing. Without zeroing, gradients from previous iterations pile up and the effective update is a sum of many minibatches' gradients, breaking training.

**A4. B.** `torch.save(model.state_dict(), ...)` saves only the parameters/buffers. Loading requires you to re-instantiate the model class and call `load_state_dict`. This is portable across code changes as long as the model class exists. Saving the full model (A) pickles class definitions — brittle across code changes. Options C/D don't handle tensors properly.

**A5.** `torch.no_grad()` is a context manager that disables autograd graph construction. Operations inside don't track gradients — saves memory (no graph to store) and speeds up computation. Use during inference/evaluation, and inside any code that doesn't need to differentiate.

**A6. B.** Without `.eval()`, dropout stays active (predictions vary randomly across calls) and BatchNorm uses the current batch's statistics (predictions depend on batch composition). No error is raised; failures are silent and easy to miss.

**A7.** `set_to_none=True` sets grads to `None` instead of a zero tensor. Slightly faster (skips allocation) and more memory-efficient. `set_to_none=False` (older default) writes zeros into the existing tensor. Both are correct; modern PyTorch defaults to `True`.

**A8. B.** With mixed precision + `GradScaler`: scale loss before backward, scaler-step (which unscales then steps), then update the scaler. `scaler.scale(loss).backward()` is critical — plain `loss.backward()` would compute unscaled gradients that overflow in fp16.

**A9.** `register_buffer` makes the tensor part of the module: it moves with `.to(device)`, appears in `state_dict()` (saved/loaded), and is included in `.eval()`/`.train()` operations. Plain attribute (`self.x = tensor`) does none of these — the tensor stays on CPU even after `.to("cuda")`, isn't saved, and can silently break your model. Use `register_buffer` for constants like positional encoding tables, running statistics, precomputed masks.

**A10. B.** `torch.compile` uses TorchDynamo + TorchInductor to JIT-compile the model's forward pass. First call incurs compilation overhead; subsequent calls are typically 1.5×–3× faster. It doesn't change the model's weights or shape; it optimizes the execution.

---

## 6. Practice Prompts

1. **Reimplement `nn.Linear`.** Write a `MyLinear(nn.Module)` from scratch, using `nn.Parameter` for weight and bias. Verify parity with `nn.Linear`.
2. **Custom Dataset + collate_fn.** Build a `Dataset` that produces variable-length text sequences (from JSONL). Write a `collate_fn` that pads and returns attention masks. Feed via DataLoader.
3. **Training loop template.** Write a reusable training-loop function `train(model, train_loader, val_loader, optimizer, scheduler, epochs)` with checkpointing, early stopping, and mixed precision.
4. **Profile a bottleneck.** Use `torch.profiler` on a small training loop. Identify the top-3 time-consuming ops. Try `torch.compile`; observe the speedup.
5. **DDP training.** Adapt a training script to use `DistributedDataParallel` on 2 GPUs (or 1 GPU with 2 processes on separate MIGs / just simulate). Confirm you get roughly linear speedup on a big batch.

---

## 7. References

- PyTorch documentation: [https://pytorch.org/docs](https://pytorch.org/docs) — everything is in there.
- Sasank Chilamkurthy et al., ["PyTorch tutorials"](https://pytorch.org/tutorials/) — the official learning path.
- Sebastian Raschka, ["Understanding PyTorch"](https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html) blog series.
- The nanoGPT repo (Karpathy) — clean end-to-end PyTorch example.
- HuggingFace Transformers source — a large-scale PyTorch codebase to learn from.
- Marcin Zablocki, ["Nine PyTorch tips"](https://towardsdatascience.com/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565) — practical performance tips.
