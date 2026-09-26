# Sequence Models — RNN, LSTM, GRU — Master Study Guide

> **Track:** Deep Learning · **Module:** 08
> **Prerequisites:** Modules 01–07.
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** From roughly 2013 to 2018, RNN-family models (LSTMs and GRUs in particular) were the standard for **any sequential problem**: language modeling, machine translation, speech recognition, time series. Transformers have largely displaced them, but:

- **RNNs are still competitive** for small-scale sequence problems, time series, edge deployment.
- **Understanding RNNs is prerequisite to understanding attention** — the whole point of attention was to fix RNNs' fundamental weakness at long dependencies.
- **The vocabulary of sequence modeling** (hidden state, teacher forcing, seq2seq, BPTT) originated with RNNs.
- **Some modern architectures blend the two** — Linear RNNs, state-space models (Mamba/S4/S5), RWKV — are increasingly interesting as transformer alternatives at long context.

**Fundamental principles you must own:**

1. **A recurrent network processes sequences one step at a time, maintaining a hidden state.**
2. **Backpropagation Through Time (BPTT)** unrolls the network across the sequence and backprops through it — same rules, longer graph.
3. **Vanishing / exploding gradients** are severe for RNNs on long sequences.
4. **LSTM and GRU** use gating mechanisms to selectively carry information across many timesteps.
5. **RNNs process sequentially** — they can't parallelize across the sequence dimension. Transformers can.
6. **Encoder-decoder / seq2seq architectures** were the state of the art for translation before transformers.

If you retain nothing else: **an RNN is a neural network with a for-loop and a persistent hidden state.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Vanilla RNN

At each timestep $t$, the RNN takes an input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, and computes:

$$\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

$$\mathbf{y}_t = W_{hy} \mathbf{h}_t + \mathbf{b}_y$$

Where:
- **$\mathbf{h}_t \in \mathbb{R}^H$** — hidden state at time $t$.
- **$W_{hh} \in \mathbb{R}^{H \times H}$** — hidden-to-hidden transition matrix.
- **$W_{xh} \in \mathbb{R}^{H \times p}$** — input-to-hidden.
- **$W_{hy} \in \mathbb{R}^{K \times H}$** — hidden-to-output.
- $\tanh$ is the activation.

**Same weights at every timestep** — like a CNN filter, one small parameter set applied repeatedly. This gives the RNN its parameter efficiency and its ability to process arbitrary-length sequences.

**Initial hidden state** $\mathbf{h}_0$ is usually zero, occasionally learnable.

**Output vs hidden.** The hidden state is the "memory"; the output can be anything computed from it (a classifier, a language model logit, a regression target).

**Sequence usage patterns:**

- **One-to-one** — regular feedforward.
- **Many-to-one** — sentence → sentiment (only the final $\mathbf{h}_T$ is used).
- **One-to-many** — image → caption (encoder is a CNN; RNN unrolls the caption).
- **Many-to-many (aligned)** — POS tagging, per-token labels ($\mathbf{y}_t$ at each $t$).
- **Many-to-many (unaligned, seq2seq)** — translation. Encoder consumes the source; decoder generates the target from the encoder's final state.

**Bidirectional RNNs** — run one RNN forward and one backward, concatenate the hidden states. Gives each position both past and future context. Standard for sequence labeling. Not usable for streaming or autoregressive generation.

**PyTorch:**

```python
rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
# input shape: (batch, seq_len, input_size)
# output: (batch, seq_len, hidden_size); hidden: (num_layers, batch, hidden_size)
```

---

### 2.2 Backpropagation Through Time (BPTT)

To train an RNN, we unroll the recurrence over $T$ timesteps and backprop through the resulting deep graph:

**Forward:** $\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T$ and $\mathbf{y}_1, \ldots, \mathbf{y}_T$.

**Loss:** $\mathcal{L} = \sum_t \mathcal{L}_t(\mathbf{y}_t, y_t^*)$ (sum over positions).

**Backward:** chain rule from $\mathcal{L}_t$ back through $\mathbf{h}_t, \mathbf{h}_{t-1}, \ldots, \mathbf{h}_1$.

The trouble: the gradient with respect to $\mathbf{h}_1$ involves the product:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_1} = \prod_{t=2}^T \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \prod_{t=2}^T \text{diag}(\tanh'(\cdot)) \cdot W_{hh}$$

If the largest singular value of $W_{hh}$ is $< 1$ (times the max derivative of tanh, which is 1), the product decays exponentially — **vanishing gradients**. Signal from timestep 1 dies before reaching timestep 100.

If it's $> 1$, gradients **explode**.

**Fixes:**

- **Truncated BPTT** — backprop only through the last $k$ steps (a sliding window). Loses very-long-range signal but manageable.
- **Gradient clipping** — cap gradient norm. Essential for RNN training.
- **Better architectures** — LSTM/GRU, next.
- **Careful initialization** — $W_{hh}$ initialized orthogonally.

**Practical rule:** for any RNN, always enable gradient clipping (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`).

---

### 2.3 LSTM: Long Short-Term Memory

**LSTM** (Hochreiter & Schmidhuber, 1997) — the pillar of pre-transformer sequence modeling. Introduces a **cell state** $\mathbf{c}_t$ (separate from the hidden state $\mathbf{h}_t$) that runs through the sequence like a conveyor belt, with **gates** to add/remove information.

**Gates and cell update:**

$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}$$
$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}$$
$$\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(candidate cell)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(cell update)}$$
$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(hidden state)}$$

Where:
- **$[\mathbf{h}_{t-1}, \mathbf{x}_t]$** = concatenation of previous hidden state and current input.
- **Sigmoid gates** produce vectors in $[0, 1]$ — elementwise "how much to keep / add / expose."
- **$\odot$** = elementwise multiplication.

**Why it fixes vanishing gradients.** The cell update is $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \dots$ — nearly-additive when $\mathbf{f}_t \approx 1$. Gradients can flow across many timesteps with the derivative $\prod_t \mathbf{f}_t$, which if $\mathbf{f}_t \approx 1$ doesn't vanish. The gates learn *when* to preserve state and when to update.

**Initial biases.** Initializing the forget-gate bias to +1 (rather than 0) makes the gate open by default, encouraging long-range memory at start.

**Practical LSTM tips:**

- **Bidirectional** LSTMs for sequence-labeling tasks.
- **Layer-normalized LSTMs** improve stability.
- **Dropout** between layers of a stacked LSTM (not within — that would break temporal signal).
- Standard training: Adam + gradient clipping + early stopping.

---

### 2.4 GRU: The Simpler Cousin

**Gated Recurrent Unit** (Cho et al., 2014) — fewer parameters than LSTM, often equal or better performance on small datasets. Merges the forget and input gates into a single "update" gate; combines cell and hidden state.

$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z) \quad \text{(update gate)}$$
$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r) \quad \text{(reset gate)}$$
$$\tilde{\mathbf{h}}_t = \tanh(W_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h) \quad \text{(candidate hidden)}$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

- **$\mathbf{z}_t$** — how much of the new candidate vs the old state to use.
- **$\mathbf{r}_t$** — how much of the previous state to include when computing the candidate.
- **No separate cell state.**

**LSTM vs GRU:**

- LSTM has more parameters (4 gates vs 3), slightly more expressive.
- GRU is faster, often as good or better on small-to-medium tasks.
- On very long sequences, LSTM sometimes edges out.
- Empirically the gap is small; pick either and don't obsess.

**PyTorch:** `nn.LSTM`, `nn.GRU` — same interface, both take `(batch, seq_len, input_size)` inputs.

---

### 2.5 Sequence-to-Sequence Models and Their Limits

**Seq2seq (Sutskever et al., 2014)** — encoder-decoder RNN architecture that launched neural machine translation.

**Encoder** — reads the source sequence into a fixed-size vector (the final hidden state):

$$\mathbf{h}_1^{\text{enc}}, \ldots, \mathbf{h}_T^{\text{enc}}, \quad \text{context} = \mathbf{h}_T^{\text{enc}}$$

**Decoder** — RNN initialized with the context, generates the target sequence one token at a time:

$$\mathbf{h}_t^{\text{dec}} = \text{RNN}(\mathbf{h}_{t-1}^{\text{dec}}, y_{t-1}^{\text{pred}})$$
$$P(y_t \mid y_{<t}, x) = \text{softmax}(W \mathbf{h}_t^{\text{dec}})$$

**Teacher forcing** — during training, feed the ground-truth previous token $y_{t-1}$ (not the model's prediction) as input at each step. Speeds up training but creates a train/inference mismatch ("exposure bias").

**Beam search** — at inference, keep the top-$B$ partial hypotheses at each step instead of greedy top-1. Better quality but slower.

**The bottleneck.** Compressing an entire sentence into a single fixed-size context vector is a hard bottleneck. Long sentences suffer — you lose information about the beginning by the end. This bottleneck **directly motivated attention** (Bahdanau et al., 2014):

Instead of one context, let the decoder **attend to all encoder hidden states** at each decoding step, computing a weighted sum based on relevance to the current decoding position.

This is where the transformer story begins. Module 09 unpacks attention; Module 10 unpacks transformers.

**The fatal flaws of RNNs:**

1. **Sequential computation.** $\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$; no parallelism across the sequence. Training is slow.
2. **Long-range dependencies.** Even LSTM/GRU struggle past a few hundred timesteps in practice.
3. **Information bottleneck** through the hidden state.

Transformers fix all three. But you should still know how RNNs work — they explain what problem attention solves.

**Modern non-attention alternatives:**

- **Temporal convolutional networks (TCN)** — 1D dilated convolutions over the sequence. Parallel, but bounded receptive field.
- **State space models (S4, S5, Mamba)** — new architectures that revisit the RNN idea with modern tricks (linear recurrence, structured state matrices). Competitive with transformers at very long context.
- **Linear RNNs / RWKV** — RNN-transformer hybrids for long-context inference efficiency.

---

## 3. Mental Models & Analogies

### 3.1 The "Reading a Book with a Notebook" Model

An RNN reading a sentence is like reading a book with a **single small notebook**:

- After each sentence, you jot down a summary in the notebook.
- To keep your notes concise, you have to overwrite/refine as you read.
- By page 300, your notebook has been rewritten hundreds of times; the story from page 5 is either forgotten or muddled.

**Vanilla RNN** = a very small notebook with limited space. Long books turn into vague summaries; the details of early events are lost.

**LSTM/GRU** = a notebook with a set of **stickers** ("keep this quote", "don't need this any more", "add this detail") that let you preserve specific bits of information across many pages. The forget gate is a "cross this out" sticker; the input gate is an "add this to my notes" sticker.

**Attention (spoiler for Module 09)** = access to the whole book at any point. You're not limited to a summary; you can flip back and re-read any earlier page as needed. This is why attention beat RNNs.

### 3.2 The "Conveyor Belt with Loading Docks" Model (LSTM's Cell State)

Think of the LSTM cell state as a **long conveyor belt** running through the network from timestep 1 to $T$:

- The belt carries information forward with minimal friction. Its "identity default" — if no gate does anything, information persists unchanged.
- At each station (timestep), three sets of workers operate:
  - The **forget-gate worker** decides what to remove from the belt (multiply each item by 0 to 1).
  - The **input-gate worker** decides what new items to place on the belt (adds new elements weighted by their gate values).
  - The **output-gate worker** peeks at the belt through a tanh window and decides how much to publicize as the hidden state $\mathbf{h}_t$.

The belt itself is nearly linear across time — that's why gradients survive. Nonlinearity is applied only to the peek-out ($\tanh(\mathbf{c}_t)$) and to computing gate values. The core memory pipeline is a friendly gradient highway.

![IMG-RNN-01](/3%20—%20Deep%20Learning/images/IMG-RNN-01.jpg)
![IMG-RNN-01](/3%20—%20Deep%20Learning/images/IMG-RNN-02.jpg)
> **Caption:** RNN unrolls over time; LSTM adds gated memory via a cell state.
> **Placement:** Sections 2.1–2.3.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Vanilla RNNs Can Learn Long Sequences If Trained Long Enough"

Nope. Even with unlimited training, vanishing gradients through hundreds of timesteps mean early tokens have essentially zero influence on late-token gradients. Vanilla RNN is limited to ~10–20 timestep dependencies in practice. LSTMs and GRUs can stretch that to a few hundred with care. Beyond that, transformers or state-space models win.

### 4.2 "Longer LSTMs Are Better"

Diminishing returns. A single LSTM layer is often as good as a 3-layer stack. Attention/transformers benefit from depth in a way RNNs mostly don't. Stacked LSTMs also multiply training time proportionally.

### 4.3 "Teacher Forcing Is Just How You Train an Autoregressive Model"

Teacher forcing creates an unavoidable **train-test mismatch**: at training, the decoder always sees the correct previous token; at inference, it sees its own (possibly wrong) previous token. Errors compound. Mitigations:

- **Scheduled sampling** — randomly use model's own predictions during training.
- **Beam search** at inference — mitigates by exploring more.
- **Curriculum + noise injection.**

This isn't unique to RNNs — modern autoregressive language models (GPT-family) have the same issue.

---

## 5. Self-Assessment Bank (Sequence Models)

### Questions

**Q1 (Short answer).** Write the update equation for a vanilla RNN.

**Q2 (Multiple choice).** Weight sharing in an RNN means:
A. All RNN layers share weights.
B. The same weight matrices $W_{hh}, W_{xh}$ are used at every timestep.
C. RNNs use fewer parameters than CNNs.
D. Every neuron shares weights with its neighbors.

**Q3 (Short answer).** Explain why vanilla RNNs suffer from vanishing gradients on long sequences.

**Q4 (Multiple choice).** The forget gate in an LSTM:
A. Determines which parts of the cell state to drop.
B. Zeroes out the input.
C. Applies dropout.
D. Resets the hidden state.

**Q5 (Short answer).** How does LSTM's cell state help with long-range dependencies?

**Q6 (Multiple choice).** Compared to LSTM, a GRU:
A. Has more parameters.
B. Has fewer parameters and a simpler update, often with comparable performance.
C. Requires no gates.
D. Doesn't have a hidden state.

**Q7 (Short answer).** What is "teacher forcing" and what problem does it create?

**Q8 (Multiple choice).** In encoder-decoder RNN translation, the bottleneck that motivated attention was:
A. GPU memory limits.
B. Compressing the entire source sequence into a single fixed-size vector.
C. Slow softmax.
D. Vocabulary size.

**Q9 (Short answer).** Why is gradient clipping standard when training RNNs?

**Q10 (Multiple choice).** Compared to a transformer processing the same sequence, an RNN:
A. Is faster during training because it can parallelize.
B. Processes tokens sequentially — no parallelism across the sequence dimension.
C. Has better long-range modeling.
D. Uses more memory during training.

---

### Answer Key & Detailed Explanations

**A1.** $\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h)$; output $\mathbf{y}_t = W_{hy} \mathbf{h}_t + \mathbf{b}_y$ (or a task-specific head).

**A2. B.** The same weight matrices $W_{hh}$, $W_{xh}$, $W_{hy}$ are applied at every timestep — analogous to a conv filter being applied at every spatial position. This gives the RNN both its ability to handle arbitrary-length sequences and its parameter efficiency.

**A3.** The gradient of $\mathbf{h}_T$ with respect to $\mathbf{h}_1$ is a product of $T$ Jacobians, each involving $\text{diag}(\tanh'(\cdot)) \cdot W_{hh}$. If the maximum singular value of $W_{hh}$ (times $\tanh'$, at most 1) is $< 1$, the product shrinks exponentially with $T$. Signal from timestep 1 becomes invisible to timestep 100.

**A4. A.** The forget gate is a sigmoid vector in $[0, 1]$ that elementwise multiplies the previous cell state. A gate value near 0 forgets that dimension; near 1 preserves it.

**A5.** The cell state update $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ is nearly additive: if $\mathbf{f}_t \approx 1$, the cell state is preserved and gradients flow through unimpeded. LSTMs learn *when* to preserve vs update, giving gradients an unobstructed highway when they need to reach far back.

**A6. B.** GRU has three gates (update, reset, candidate) vs LSTM's four (forget, input, candidate, output). No separate cell state. Fewer parameters, faster; typically comparable or slightly better on small datasets. The gap is task-dependent and usually small.

**A7.** Teacher forcing = during training, feed the **ground-truth** previous token as the RNN's input at each step, rather than the model's own prediction. Problem: at inference, the model *is* using its own predictions, and errors from wrong predictions compound. Mismatch between train and inference is called **exposure bias**. Mitigations: scheduled sampling, beam search, larger models with better predictions.

**A8. B.** The encoder produces one fixed-size vector (the final hidden state) to summarize the entire source sequence. For long sentences, information about the beginning is lost by the time the decoder starts producing tokens. Attention (Bahdanau 2014, Luong 2015) removed this bottleneck by letting the decoder look at *all* encoder hidden states, weighted by learned relevance.

**A9.** Gradient clipping caps the total gradient norm at some value (typically 1.0 or 5.0). Without it, RNNs frequently produce enormous gradient spikes when the sequence contains a specific pattern that amplifies through $W_{hh}$; these spikes send parameters to bad regions and cause NaN losses. Clipping is close-to-free insurance.

**A10. B.** RNN's hidden state at $t$ depends on $t - 1$ — inherently sequential. Cannot parallelize across the sequence dimension. Transformers replace this sequential dependency with attention, which is highly parallelizable (a matrix multiply). That's the key architectural advantage that let transformers scale to enormous datasets.

---

## 6. Practice Prompts

1. **From scratch.** Implement a vanilla RNN cell (forward and BPTT) in NumPy. Train it on a synthetic sequence-copy task (input a random binary sequence, output the same sequence). Observe the length limit where it fails.
2. **LSTM from scratch.** Implement an LSTM cell in NumPy or PyTorch (no `nn.LSTM`). Verify against `torch.nn.LSTM` on a small task.
3. **Compare depths / gating.** On IMDB sentiment classification, compare a 1-layer RNN, 1-layer LSTM, 1-layer GRU, and 2-layer LSTM. Report test accuracy and per-epoch time.
4. **Gradient clipping demo.** Train an LSTM without clipping; observe the occasional NaN loss and inspect the gradient norm. Add clipping; watch training stabilize.
5. **Seq2seq translation.** Build a small encoder-decoder LSTM for character-level translation (e.g., short English → French pairs). Compare with and without a naïve attention mechanism.

---

## 7. References

- Hochreiter & Schmidhuber, ["Long Short-Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf) (1997).
- Cho et al., ["Learning Phrase Representations using RNN Encoder-Decoder"](https://arxiv.org/abs/1406.1078) (2014) — GRU.
- Sutskever, Vinyals & Le, ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215) (2014).
- Bahdanau, Cho & Bengio, ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473) (2014) — attention.
- Chris Olah, ["Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — the canonical explainer.
- Karpathy, ["The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (2015).
- Gu et al., ["Efficiently Modeling Long Sequences with Structured State Spaces"](https://arxiv.org/abs/2111.00396) (S4, 2021); Gu & Dao, ["Mamba"](https://arxiv.org/abs/2312.00752) (2023).
