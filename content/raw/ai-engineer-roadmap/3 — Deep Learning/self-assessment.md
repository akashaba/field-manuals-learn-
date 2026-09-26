# Deep Learning — Consolidated Self-Assessment

> **Scope:** All 12 learning modules **and** Build #1.
> **Format:** Timed (2.5 hours), no reference, notebook off.
> **Passing bar:** ≥ 80% overall **and** ≥ 60% in every section.

This is the exit exam for Month 3 — Deep Learning. Take it when you *think* you're ready. Score honestly. Retake two weeks later after remediating.

---

## Section A — Neural Networks & Backpropagation (14 points)

**A1 (2 pt).** Explain why a nonlinear activation is necessary between layers. What happens if you omit it?

**A2 (1 pt, MC).** For Kaiming init on a ReLU layer with $n_{\text{in}} = 512$, weights are drawn from:
- (a) $\mathcal{N}(0, 1/512)$
- (b) $\mathcal{N}(0, 2/512)$
- (c) $\mathcal{U}(-1, 1)$
- (d) $\mathcal{N}(0, 1)$

**A3 (2 pt).** State backpropagation in one sentence. Why is the backward pass approximately as cheap as the forward pass?

**A4 (2 pt).** Given a dense layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, $\mathbf{h} = \phi(\mathbf{z})$, and upstream gradient $\delta = \partial L / \partial \mathbf{h}$, write $\partial L / \partial W$, $\partial L / \partial \mathbf{b}$, $\partial L / \partial \mathbf{x}$.

**A5 (2 pt).** Explain vanishing gradients in a deep sigmoid network mathematically.

**A6 (1 pt, MC).** In PyTorch, forgetting `optimizer.zero_grad()` causes:
- (a) NaN.
- (b) Gradients accumulate across iterations, growing unboundedly.
- (c) Learning rate doubles.
- (d) An error is raised.

**A7 (2 pt).** What is a "dead ReLU" and what causes it? Name one fix.

**A8 (2 pt).** What does `torch.no_grad()` do, and when should you use it?

---

## Section B — Activations & Losses (10 points)

**B1 (1 pt, MC).** For a 10-class classification model, the output activation should be:
- (a) ReLU
- (b) Sigmoid per output
- (c) Softmax
- (d) Tanh

**B2 (2 pt).** Why is `nn.CrossEntropyLoss` expected to be applied to raw logits (not softmax'd probabilities)?

**B3 (2 pt).** Explain, in one paragraph, why using MSE as the loss for logistic regression is a bad idea.

**B4 (2 pt).** Give the formula for GELU and explain intuitively how it differs from ReLU.

**B5 (1 pt, MC).** For multi-label classification (labels are independent), the correct loss is:
- (a) `CrossEntropyLoss` with softmax
- (b) `BCEWithLogitsLoss` (sigmoid + BCE per label)
- (c) MSE
- (d) Contrastive loss

**B6 (2 pt).** What is label smoothing? Why does it help generalization?

---

## Section C — Optimizers & Regularization (14 points)

**C1 (2 pt).** State Adam's update: what quantities does it track, and what's the final update?

**C2 (2 pt).** Why does Adam use bias correction $\hat m_t = m_t / (1 - \beta_1^t)$?

**C3 (2 pt).** Explain AdamW's key difference from Adam. Why does it matter?

**C4 (1 pt, MC).** Cosine annealing decays the LR:
- (a) At fixed epochs.
- (b) Smoothly on a cosine curve from initial to minimum.
- (c) By halving every 10 iterations.
- (d) Only on plateaus.

**C5 (2 pt).** Explain the difference between BatchNorm and LayerNorm — what's normalized over what, and when do you use each?

**C6 (1 pt, MC).** `nn.Dropout(0.5)` at training:
- (a) Halves the loss.
- (b) Zeros ~50% of activations and scales survivors by 2× so expected magnitude is preserved.
- (c) Removes half the parameters.
- (d) Delays training by 50%.

**C7 (2 pt).** Why is Layer Normalization the standard for transformers rather than BatchNorm?

**C8 (2 pt).** Name three regularization techniques you would stack for a small dataset training a mid-sized CNN.

---

## Section D — CNNs (10 points)

**D1 (2 pt).** In one sentence, what makes a convolutional layer more parameter-efficient than a fully-connected layer for images?

**D2 (2 pt).** Given a 64-channel feature map and a 3×3 conv layer that produces 128 output channels, compute the number of parameters (include biases).

**D3 (2 pt).** Explain what a residual (skip) connection does and why it's essential for training very deep networks.

**D4 (1 pt, MC).** For transfer learning on a small custom dataset with a pretrained ResNet backbone, the standard first move is:
- (a) Fine-tune all parameters at full LR.
- (b) Freeze backbone; retrain only the classifier head.
- (c) Retrain from scratch.
- (d) Delete half the filters.

**D5 (2 pt).** Explain what a 1×1 convolution does and one situation where it's useful.

**D6 (1 pt).** Why do we use `bias=False` on a Conv2d immediately followed by BatchNorm?

---

## Section E — Sequence Models & Attention (14 points)

**E1 (2 pt).** Write the recurrence of a vanilla RNN and describe why it suffers from vanishing gradients on long sequences.

**E2 (2 pt).** In an LSTM, what does the forget gate do, and why does it enable long-range gradient flow?

**E3 (1 pt, MC).** Compared to LSTM, a GRU:
- (a) Has more gates.
- (b) Has fewer parameters, roughly comparable performance.
- (c) Has no gates.
- (d) Doesn't have a hidden state.

**E4 (2 pt).** Write scaled dot-product attention as one formula. Explain each element.

**E5 (2 pt).** Why divide by $\sqrt{d_k}$ before softmax in attention?

**E6 (2 pt).** Explain the causal mask in a decoder-only transformer — what it prevents and how it's implemented on the score matrix.

**E7 (2 pt).** In multi-head attention with $h = 8$ heads and $d_{\text{model}} = 768$, what's the per-head key dimension, and why not use one big head?

**E8 (1 pt, MC).** Attention is:
- (a) $O(n)$ in sequence length.
- (b) $O(n \log n)$.
- (c) $O(n^2)$ in both time and memory.
- (d) $O(n^3)$.

---

## Section F — Transformers & LLMs (16 points)

**F1 (2 pt).** Write the two-line update of a pre-norm transformer block.

**F2 (1 pt, MC).** Approximately how many parameters does a transformer block have in terms of $d_{\text{model}}$?
- (a) $d_{\text{model}}$
- (b) $d_{\text{model}}^2$
- (c) $12 d_{\text{model}}^2$
- (d) $d_{\text{model}}^3$

**F3 (2 pt).** Explain the difference between encoder-only, decoder-only, and encoder-decoder transformers, and give one example model of each.

**F4 (2 pt).** What is the KV cache, and why is it essential for autoregressive inference?

**F5 (2 pt).** Describe the LLM post-training pipeline (SFT → RLHF or DPO) in one paragraph.

**F6 (2 pt).** Explain what LoRA does. Why does it enable fine-tuning huge models on modest hardware?

**F7 (2 pt).** What is "in-context learning" and why does it depend on scale?

**F8 (2 pt).** Why can an LLM struggle to count letters in a word or reverse a string, despite being excellent at complex reasoning?

**F9 (1 pt, MC).** Chinchilla scaling laws suggest that for a fixed compute budget:
- (a) The bigger model always wins.
- (b) There's an optimal balance of model size and training tokens (~20 tokens per param).
- (c) Data doesn't matter.
- (d) Depth matters more than width.

---

## Section G — PyTorch (12 points)

**G1 (2 pt).** In one sentence, what does `nn.Module` provide beyond a plain Python class?

**G2 (2 pt).** Given `x = torch.tensor([1., 2., 3.], requires_grad=True); y = (x*x).sum(); y.backward()`, what does `x.grad` contain, and why?

**G3 (1 pt, MC).** The recommended way to save a PyTorch model is:
- (a) `torch.save(model, path)`
- (b) `torch.save(model.state_dict(), path)`
- (c) `pickle.dump(model, file)`
- (d) `json.dump(model, file)`

**G4 (2 pt).** Explain why registering a constant tensor with `self.register_buffer(...)` matters vs assigning it as a plain attribute.

**G5 (2 pt).** In mixed-precision training with `autocast` + `GradScaler`, describe the correct order of operations in a step.

**G6 (2 pt).** Why is `set_to_none=True` slightly better than the default in `optimizer.zero_grad()`?

**G7 (1 pt, MC).** `torch.compile(model)` in PyTorch 2.x:
- (a) Converts to ONNX.
- (b) JIT compiles the forward pass — first call slow, subsequent faster.
- (c) Freezes parameters.
- (d) Reduces model size.

---

# Answer Key & Detailed Explanations

## Section A

**A1.** Without a nonlinear activation, stacked linear layers compose to a single effective linear layer ($W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + \text{const}$). No matter how many layers, the model can only represent linear functions. Nonlinear activations (ReLU, GELU) are what give depth expressive power.

**A2. (b).** Kaiming for ReLU: $\text{Var}(W) = 2 / n_{\text{in}}$, so $\mathcal{N}(0, 2/512)$.

**A3.** Backpropagation is the algorithm that computes gradients of a scalar loss with respect to every parameter by applying the chain rule in reverse topological order over the computational graph. It's cheap because reverse-mode automatic differentiation visits each node in the graph **once**, multiplying local Jacobians — total work is proportional to the forward pass.

**A4.**
- $\partial L / \partial \mathbf{z} = \delta \odot \phi'(\mathbf{z})$
- $\partial L / \partial W = (\partial L / \partial \mathbf{z}) \, \mathbf{x}^\top$
- $\partial L / \partial \mathbf{b} = \partial L / \partial \mathbf{z}$
- $\partial L / \partial \mathbf{x} = W^\top (\partial L / \partial \mathbf{z})$

**A5.** In a deep sigmoid network, the gradient at layer 1 involves a product of $L$ terms, each of the form $\sigma'(z) \cdot W$. Since $\sigma'(z) \leq 0.25$, and typical $W$ initialization gives a similar-order-of-magnitude factor, the product decays as $\leq 0.25^L$. For $L = 20$, that's $\leq 10^{-12}$ — the earliest layers see essentially zero gradient and stop learning.

**A6. (b).** Gradients accumulate — `backward()` adds to `.grad` rather than replacing. Without `zero_grad`, gradients from previous minibatches persist, and effective step size grows over iterations, destabilizing training.

**A7.** A dead ReLU is a neuron whose pre-activation is always ≤ 0. Its output is always 0, its derivative is always 0, so it never receives gradient and never updates. Caused by: too-large learning rate driving weights into a bad region, or poor initialization. Fixes: **LeakyReLU / GELU** (nonzero gradient below zero), Kaiming init, smaller LR.

**A8.** `torch.no_grad()` is a context manager that disables autograd graph construction. Operations inside don't record for backprop; saves memory (no graph stored) and speeds up computation. Use during inference, validation, and any code you don't need to differentiate.

---

## Section B

**B1. (c).** Softmax normalizes 10 raw logits to a probability distribution summing to 1 — appropriate for mutually-exclusive classes.

**B2.** `nn.CrossEntropyLoss` applies log-softmax internally (numerically stable via log-sum-exp) and combines with NLL. Passing raw logits is correct; passing softmax'd values applies softmax twice, mangling the loss and gradients.

**B3.** MSE with sigmoid outputs produces a non-convex loss landscape — gradient descent can get stuck. Additionally, the gradient of MSE + sigmoid is proportional to $\sigma'(z)$, which vanishes for large $|z|$, so a confidently wrong prediction produces near-zero gradient (no correction). Cross-entropy is convex and its gradient stays proportional to the error even at high confidence.

**B4.** $\text{GELU}(z) = z \cdot \Phi(z)$ where $\Phi$ is the standard-normal CDF. Intuitively, GELU multiplies $z$ by a smooth probability that scales from 0 (very negative) to 1 (very positive) — a smooth, differentiable "keep-or-drop" gate. Unlike ReLU's hard zero for negatives, GELU passes some signal near zero, avoiding dead neurons and being smoother for gradients.

**B5. (b).** Multi-label = each label independent. `BCEWithLogitsLoss` treats each output as an independent binary classification (sigmoid + BCE). Softmax would force labels to compete (sum to 1) — wrong for multi-label.

**B6.** Label smoothing replaces one-hot targets with soft distributions: $y'_k = (1 - \varepsilon) y_k + \varepsilon / K$. Instead of pushing $p_k$ toward 1.0 (impossible with a real dataset without overfitting), the loss encourages a more moderate confidence. This prevents overconfidence, keeps gradients bounded, and improves calibration and often test accuracy.

---

## Section C

**C1.** Adam tracks:
- **First moment** $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$ (momentum).
- **Second moment** $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$ (per-parameter scaling).

Bias-corrected: $\hat m_t = m_t / (1 - \beta_1^t)$, $\hat v_t = v_t / (1 - \beta_2^t)$.

Update: $\theta \leftarrow \theta - \eta \, \hat m_t / (\sqrt{\hat v_t} + \varepsilon)$.

**C2.** The moving averages start at $\mathbf{m}_0 = \mathbf{v}_0 = 0$. At $t = 1$, $\mathbf{m}_1 = (1 - \beta_1) \mathbf{g}_1 \approx 0.1 \mathbf{g}_1$ — a heavily underestimated gradient. Bias correction $\mathbf{m}_1 / (1 - \beta_1^1) = \mathbf{g}_1$ restores the correct scale. Over time, $(1 - \beta_1^t) \to 1$ and correction vanishes.

**C3.** AdamW **decouples** weight decay from the adaptive gradient step, applying $\theta \leftarrow \theta - \eta \lambda \theta$ separately from the Adam update. In plain Adam, adding $\lambda \theta$ to the gradient means the "decay" is scaled by $\sqrt{\hat v_t}$ — distorting the intended shrinkage. AdamW restores the correct decoupled decay behavior and generally improves generalization. It's the modern default.

**C4. (b).** Cosine annealing: $\eta_t = \eta_\min + \frac{1}{2}(\eta_0 - \eta_\min)(1 + \cos(\pi t / T))$ — a smooth decrease. Widely used in modern training.

**C5.** **BatchNorm** normalizes each feature (channel) using statistics over the **batch dimension**. Requires reasonable batch sizes; standard in CNNs. **LayerNorm** normalizes over the **feature dimension** within each sample. No batch dependency; works for variable-length sequences; standard in transformers and RNNs.

**C6. (b).** Inverted dropout: with $p = 0.5$, each element is zeroed with probability 0.5, and survivors are scaled by $1/(1-0.5) = 2$ to preserve expected magnitude.

**C7.** Transformers process variable-length sequences and often train with small effective per-device batches. BatchNorm's batch statistics would be noisy and inconsistent across positions. LayerNorm operates per-token on features, so it's batch-size-independent and handles the sequential/parallel nature of transformer training gracefully.

**C8.** Any three of: **weight decay** (AdamW), **dropout**, **data augmentation** (mixup, cutout, RandAugment), **early stopping**, **label smoothing**, **batch/layer norm**, or **reducing model size**.

---

## Section D

**D1.** **Weight sharing across spatial positions** — a single small filter (e.g., 9 weights for 3×3) is applied at every position of the image. A fully-connected layer would need a distinct weight for every (input position, output position) pair — billions of params for a high-res image.

**D2.** $(3 \cdot 3 \cdot 64) \cdot 128 + 128 = 576 \cdot 128 + 128 = 73{,}728 + 128 = 73{,}856$ params.

**D3.** A residual connection passes the input around a block: $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$. This gives gradients a **direct path** back to earlier layers, avoiding the exponential decay through many intermediate layers. It also makes "identity" the default behavior of a block, so training a deep network becomes learning small corrections rather than the entire transform. Enables training 100+ layer networks that vanilla stacks can't.

**D4. (b).** Freeze the pretrained backbone (large, well-trained features you shouldn't perturb without lots of new data); train only the final classifier on your small dataset. Optionally, once the classifier converges, unfreeze upper blocks with a small LR.

**D5.** A 1×1 conv is a **pointwise linear projection across channels** at each spatial location. Uses: (1) **Channel reduction** ("bottleneck") — reduce a 512-channel map to 128 before an expensive 3×3 conv; (2) **Channel expansion** — restore channels after a bottleneck; (3) **Combining parallel branch outputs** (Inception); (4) **Adding non-linearity cheaply**.

**D6.** BatchNorm has its own learnable shift ($\beta$) that subsumes any bias from the preceding Conv2d — the conv's bias becomes redundant and just wastes parameters and slightly slows compute. `bias=False` saves a small amount.

---

## Section E

**E1.** $\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h)$. The gradient $\partial \mathbf{h}_T / \partial \mathbf{h}_1$ is a product of $T$ Jacobians of the form $\text{diag}(\tanh'(\cdot)) \cdot W_{hh}$. If the max singular value of $W_{hh}$ (times $\tanh' \leq 1$) is $< 1$, the product decays exponentially with $T$ — early-timestep gradients vanish for long sequences.

**E2.** The forget gate $\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$ produces a vector in $[0, 1]$ that elementwise multiplies the previous cell state. Gate near 1 preserves the cell state; near 0 forgets. Long-range flow: with $\mathbf{f}_t \approx 1$, the cell state update $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \dots$ is nearly additive — gradients flow through the cell state almost unimpeded, giving LSTMs their ability to remember across many timesteps.

**E3. (b).** GRU merges LSTM's forget and input gates into one update gate; no separate cell state. Fewer parameters and simpler; comparable or slightly better on many small-to-medium tasks.

**E4.** $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$. $Q$ = queries; $K$ = keys; $V$ = values (each a linear projection of the input in self-attention). $d_k$ = key dimension; scaling by $\sqrt{d_k}$ keeps softmax gradients healthy.

**E5.** For high-dim $Q, K$ with random unit-variance entries, dot products $\mathbf{q} \cdot \mathbf{k}$ have variance $d_k$; standard deviation $\sqrt{d_k}$. Without scaling, softmax becomes very peaky (saturating) and produces near-zero gradient for all but the top few keys. Dividing by $\sqrt{d_k}$ normalizes the score variance and keeps softmax in a well-behaved region.

**E6.** During training, we compute predictions for all positions in parallel, but each position's prediction must depend **only on previous positions** — otherwise, the model would trivially copy future tokens as targets. The causal mask sets $S_{ij} = -\infty$ for $j > i$ before softmax. After softmax, $-\infty$ becomes 0, so future positions contribute no weight.

**E7.** Per-head $d_k = 768 / 8 = 96$. Multi-head lets **different heads attend to different patterns** — syntactic relations, semantic similarity, long-range references — in parallel subspaces. A single big head with the same total dimension would learn one mixed pattern; empirical results consistently show multi-head is better.

**E8. (c).** Attention computes $QK^\top \in \mathbb{R}^{n \times n}$ — $n^2$ entries in memory, $O(n^2)$ compute. Flash Attention reduces memory footprint but the total work is still $O(n^2)$.

---

## Section F

**F1.** $\mathbf{x}' = \mathbf{x} + \text{MHSA}(\text{LN}(\mathbf{x}))$; $\mathbf{y} = \mathbf{x}' + \text{FFN}(\text{LN}(\mathbf{x}'))$. LayerNorm before each sub-layer; residual from before-LN input.

**F2. (c).** Attention has $4 d_{\text{model}}^2$ (Q/K/V/output projections); FFN has $8 d_{\text{model}}^2$ ($d_\text{ff} = 4 d_\text{model}$, two matrices). Total ≈ $12 d_{\text{model}}^2$ per block.

**F3.** **Encoder-only** (BERT): bidirectional self-attention; good for classification and embeddings. **Decoder-only** (GPT, LLaMA): causal self-attention; good for autoregressive generation and few-shot learning. **Encoder-decoder** (T5, original transformer, Whisper): encoder is bidirectional; decoder is causal and cross-attends to encoder outputs. Good for seq2seq (translation, summarization).

**F4.** During autoregressive generation, each new token's attention would need to recompute K and V for every previous token — $O(n^2)$ per token, $O(n^3)$ total. The **KV cache** stores K, V for all previous tokens; each new step only computes K, V for the current position and appends. Reduces per-token cost to $O(n)$ and total generation to $O(n^2)$. Without it, real-time inference is impractical.

**F5.** (1) **Pretraining** on trillions of tokens with next-token prediction. (2) **Supervised Fine-Tuning (SFT)** on (instruction, response) pairs teaches the base model the "helpful assistant" pattern. (3) **RLHF or DPO** aligns the model with human preferences: RLHF trains a reward model on preference pairs then fine-tunes via PPO; DPO directly optimizes for the preferences without RL. The stages turn a raw text completer into a useful assistant.

**F6.** LoRA freezes the pretrained weight matrix $W$ and learns a low-rank update $\Delta W = BA$ where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, $r \ll \min(d, k)$. Only $A, B$ are trained — typically ~1000× fewer parameters than full fine-tuning. Combined with 4-bit quantization of the base (QLoRA), it enables fine-tuning 70B models on a single 48GB GPU. Multiple LoRAs share the same frozen base and are cheap to serve.

**F7.** In-context learning is the ability of an LLM to perform a task by seeing a few examples in its prompt, without any weight updates. Give it "cat → mammal, salmon → fish, eagle → ___" and it says "bird." Discovered in GPT-3; the ability strengthens with model scale. Smaller models can't do this well.

**F8.** LLMs process **tokens**, not characters. "Strawberry" might be a single token; the model doesn't see individual R's — it sees an integer ID. Tasks requiring character-level reasoning (counting, spelling, reversal) require the model to reconstruct the character sequence from its internal representation, which it wasn't strongly rewarded for during training. This is why LLMs can produce brilliant essays but botch "how many R's in strawberry."

**F9. (b).** Chinchilla: for a fixed compute budget, an optimal balance of model size and training tokens (~20 tokens per parameter) yields the best model. Undertrained huge models are wasteful; overtrained small models eventually plateau. GPT-3 (175B, 300B tokens) was severely undertrained.

---

## Section G

**G1.** `nn.Module` provides automatic parameter registration, recursive `.to(device)`/`.parameters()`/`.state_dict()`/`.train()`/`.eval()` traversal over submodules, and integration with autograd/optimizer/loss.

**G2.** `x.grad = tensor([2., 4., 6.])`. $y = \sum x_i^2$; $\partial y / \partial x_i = 2 x_i$; for $x = [1, 2, 3]$, gradients are $[2, 4, 6]$.

**G3. (b).** `torch.save(model.state_dict(), path)` — saves parameters + buffers as a dict. Portable across code changes as long as the model class exists. Full-model pickle depends on class-definition compatibility.

**G4.** `register_buffer` makes the tensor part of the module: it moves with `.to(device)`, appears in `state_dict()` (saved / loaded), and is properly tracked. Plain attribute assignment (`self.x = tensor`) does none of these — the tensor stays on CPU even after `.to("cuda")`, isn't saved, and silently breaks the model. Use for constants like positional encoding tables, precomputed masks, buffered statistics.

**G5.**
```python
with autocast(dtype=torch.bfloat16):
    loss = criterion(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                       # for gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

**G6.** `set_to_none=True` sets each `.grad` to `None` instead of writing zeros into an existing tensor. Faster (skips memory writes), lower memory (no persistent gradient tensor between steps). Modern PyTorch defaults to this.

**G7. (b).** `torch.compile` uses TorchDynamo + TorchInductor to JIT-compile the model's forward pass. First call is slow (compilation overhead); subsequent calls are typically 1.5×–3× faster. Same model interface, no code changes needed.

---

## Scoring

| Section | Points | Yours |
|---------|--------|-------|
| A. NN & Backprop | 14 | |
| B. Activations & Losses | 10 | |
| C. Optimizers & Regularization | 14 | |
| D. CNNs | 10 | |
| E. Sequence Models & Attention | 14 | |
| F. Transformers & LLMs | 16 | |
| G. PyTorch | 12 | |
| **Total** | **90** | |

**Retake conditions.** Anything under 60% in a section, or under 80 overall → spend a week on the weak spots and retake with shuffled questions.

**Passing.** Once you clear the bar — and Build #1 is shipped — you're ready to move into modern AI engineering: MLOps, agent frameworks, and multimodal systems. You've crossed the bridge.
