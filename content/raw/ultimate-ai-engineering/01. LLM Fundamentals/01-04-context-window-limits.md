# 01.04 — Context Window & Limits

> **Module 1: LLM Fundamentals** · Subtopic 4 of 4
> **Prerequisites:** 01.01 (RoPE, KV cache, attention cost), 01.02 (prefill/decode, prefix caching), 01.03 (token counting).
> **Outcome:** you understand *why* context is limited (memory, compute, positional generalisation, training distribution). You can extend it, compress it, and evaluate the *effective* context rather than the advertised number. You can also engineer application-level context management that stays within budget, stays cheap, and stays accurate.

---

## 1. The Four Walls of Context

```
                 ┌────────────────────────────────────────────────────────────┐
  advertised     │ 1. MEMORY     KV cache grows linearly in T (per sequence)   │
  context  ────► │ 2. COMPUTE    prefill attention grows as T²                  │
  window         │ 3. POSITIONS  RoPE angles never seen in training → drift     │
                 │ 4. LEARNING   few long training docs → weak long-range use   │
                 └────────────────────────────────────────────────────────────┘
                              ▼
                 EFFECTIVE context (what the model can reliably USE)
                 is usually well below the advertised number
```

### 1.1 Memory

From 01.01:

$$
\text{KV bytes} = 2\cdot L\cdot n_{kv}\cdot d_h\cdot b\cdot T\cdot B
$$

For a 70B-class GQA model (≈ 320 KiB/token in bf16), one 128k-token request needs ≈ 40 GiB of KV, which is more than half of an 80 GB GPU *after* the weights are sharded. Long context therefore trades directly against concurrency.

### 1.2 Compute

For a whole causal prefill of $T$ tokens, attention costs about $2\,L\,T^2\,d$ FLOPs (the $QK^\top$ and $AV$ products, each $2T^2d$ per layer, halved by causality). The linear layers cost $2NT$. Using $N\approx 12Ld^2$ for a dense model, the per-token ratio of attention to linear FLOPs is

$$
\frac{2\,L\,T\,d}{2\cdot 12\,L\,d^2} = \frac{T}{12\,d}
$$

With $d = 4096$, attention FLOPs **equal** the entire rest of the model at $T\approx 49\text{k}$ and dominate beyond that (the exact crossover depends on GQA, FFN width, and causal masking, but the order of magnitude holds). This is why TTFT rises superlinearly for long prompts, even with FlashAttention.

### 1.3 Positional generalisation

A model trained at length $L_{\text{train}}$ has only seen RoPE rotation angles $m\theta_i$ for $m < L_{\text{train}}$. Beyond that, the low-frequency dimensions (whose period exceeds $L_{\text{train}}$) enter **unseen angle regimes** and attention patterns break down. Perplexity typically explodes just past the training length.

### 1.4 Learning

Even with positions fixed, models must be *trained* to use long-range dependencies. Natural long documents are rare, and most training tokens are short. Two consequences:
- **"Lost in the middle"** (Liu et al.): accuracy is U-shaped over the position of the relevant information. It is best at the start (primacy) and end (recency) and degrades in the middle.
- **Degradation as input grows**: benchmarks such as RULER show that retrieval, multi-hop tracing, and aggregation tasks degrade well before the advertised window, even when simple needle-in-a-haystack passes.

![LLM-06](/01.%20LLM%20Fundamentals/images/LLM-06.jpg)


---

## 2. Extending the Context Window (RoPE Scaling)

Let $s = L_{\text{target}}/L_{\text{train}}$ be the scale factor, with RoPE frequencies $\theta_i = b^{-2i/d_h}$.

### 2.1 Position Interpolation (PI)

Compress positions so that they stay within the trained range:

$$
m \;\mapsto\; \frac{m}{s} \qquad\Longleftrightarrow\qquad \theta_i \mapsto \frac{\theta_i}{s}
$$

This is simple and stable, and needs only brief fine-tuning (≈ 1k steps). The downside is that it also squashes the **high-frequency** dimensions, which encode fine local order. Nearby tokens become harder to distinguish, and short-context quality suffers slightly.

### 2.2 NTK-aware scaling

Change the base instead, so that high frequencies are barely touched while low frequencies are interpolated:

$$
b' = b\cdot s^{\,d_h/(d_h-2)}
$$

This works partially **without fine-tuning**. "Dynamic NTK" recomputes $s$ from the current sequence length at inference time.

### 2.3 YaRN

YaRN interpolates **per frequency band**. Define each dimension's wavelength $\lambda_i = 2\pi/\theta_i$ and the ratio $r_i = L_{\text{train}}/\lambda_i$ (how many full rotations fit in the training context). Then apply a ramp:

$$
\gamma(r) = \begin{cases} 0 & r < \alpha \\ 1 & r > \beta \\ \dfrac{r-\alpha}{\beta-\alpha} & \text{otherwise}\end{cases}
\qquad\quad
\theta_i' = \big(1-\gamma(r_i)\big)\,\frac{\theta_i}{s} + \gamma(r_i)\,\theta_i
$$

- High-frequency dimensions (many rotations, $r>\beta$) are **not interpolated**, which preserves local order.
- Low-frequency dimensions ($r<\alpha$) are **fully interpolated** (PI).
- Between the two, the scheme blends linearly. Typical Llama settings are $\alpha=1$, $\beta=32$.

YaRN also applies an **attention temperature** to counter the entropy increase over longer sequences. The logits are scaled by $1/t$, where

$$
\sqrt{1/t} = 0.1\,\ln s + 1
$$

Fine-tuning needs are small (a fraction of a percent of pretraining tokens). YaRN is widely supported in serving engines through a `rope_scaling` configuration.

### 2.4 LongRoPE and progressive extension

LongRoPE *searches* for non-uniform per-dimension rescale factors (evolutionary search), then extends in stages (e.g. 256k → 2M). Production long-context models usually combine RoPE rescaling (often a raised base, e.g. $10^6$) with **staged continued pretraining** on progressively longer, up-sampled long documents, plus synthetic long-range tasks.

> ⚠️ Serving engines must apply the *same* scaling the model was fine-tuned with. Static YaRN at inference also changes short-context behaviour slightly. Some deployments enable scaling only when prompts exceed a threshold, which trades that effect against consistency.

---

## 3. Architectural Approaches to Long Context

| Approach | Mechanism | Cost | Trade-off |
|---|---|---|---|
| **Sliding-window attention** (Longformer local, Mistral 7B) | Each token attends to the last $W$ tokens; stacked layers give receptive field ≈ $W\cdot L$ | $O(TW)$, KV bounded by $W$ | Weak exact long-range recall; usually interleaved with some global layers |
| **Global + local + random** (BigBird, Longformer) | Sparse pattern with a few global tokens | $O(T)$ | Kernel complexity; mostly encoder-era |
| **Attention sinks / StreamingLLM** | Keep the first $k$ tokens + a recent window in the KV cache | Constant memory | Unbounded streaming, but no recall of evicted content |
| **Ring attention / context parallelism** | Shard the sequence across $P$ devices; rotate KV blocks around a ring, overlapping communication with compute | Exact attention; memory per device $\propto T/P$ | Needs fast interconnect; for training or very long prefill |
| **State-space models** (Mamba) | Recurrent $h_t = \bar A_t h_{t-1} + \bar B_t x_t,\ y_t = C_t h_t$ with input-dependent (selective) parameters | $O(T)$ time, **constant** state | Fixed-size state limits exact retrieval ("associative recall") |
| **Hybrids** (attention + SSM or linear-attention layers) | Mostly linear layers plus a few full-attention layers | Much smaller KV | Increasingly common in efficient long-context models |
| **MLA / aggressive GQA** | Smaller KV per token | Linear, lower constant | See 01.01 §3 |


![LLM-07](/01.%20LLM%20Fundamentals/images/LLM-07.jpg)


---

## 4. KV Cache Compression at Inference

- **Quantization:** FP8 or INT8 KV is near-lossless on many models. **KIVI** uses 2-bit with *per-channel* keys (keys have outlier channels) and *per-token* values.
- **Eviction:**
  - **H2O** keeps "heavy-hitter" tokens with high cumulative attention, plus recent tokens.
  - **SnapKV** uses an observation window at the end of the prompt to decide which prompt KV entries to keep per head.
  - Eviction is risky for tasks that need exact recall of arbitrary tokens. Always evaluate on retrieval tasks, not only perplexity.
- **Sharing and offloading:** cross-layer KV sharing, CPU or SSD offload with prefetch (connector libraries that tier the KV cache), and reuse of KV across requests through prefix caching.

---

## 5. Application-Level Context Engineering

Most production gains come from here, not from bigger windows.

### 5.1 Budget arithmetic (every request)

$$
T_{\text{system}} + T_{\text{tools}} + T_{\text{retrieved}} + T_{\text{history}} + T_{\text{user}} + T_{\text{template}} + T_{\text{max\_output}} \;\le\; (1-\epsilon)\cdot T_{\text{window}}
$$

Reserve output tokens *before* packing the input. For reasoning models, the output budget must include hidden reasoning tokens.

### 5.2 Strategies, by cost and risk

1. **Don't send it:** prune tool outputs (return summaries or IDs plus a fetch tool), strip boilerplate, deduplicate retrieved chunks.
2. **Retrieve instead of stuffing (RAG):** send only the top-k relevant chunks, placing the most important at the **start or end**.
3. **Compact history:** a rolling summary of older turns, keeping the last *k* turns verbatim. Treat tool-call and tool-result pairs as atomic.
4. **Structured memory:** extract durable facts into a store and retrieve them per turn (MemGPT-style hierarchical memory).
5. **Long context as a last resort:** whole-document reasoning where retrieval fails (cross-references, global aggregation).

### 5.3 Long context vs RAG

| Dimension | Long context | RAG |
|---|---|---|
| Cost per query | High ($\propto T$; prefix caching helps for repeated documents) | Low |
| TTFT | High for large $T$ | Low (retrieval adds ~10–100 ms) |
| Global reasoning over a corpus | Strong, within effective length | Weak, since the answer may span unretrieved chunks |
| Freshness and scale | Bounded by the window | Scales to millions of documents |
| Failure mode | Middle-position neglect, distraction | Retrieval miss (the model never sees the answer) |

The best systems combine both: retrieval to select, long context to reason over a generous, well-ordered selection.

### 5.4 Prompt/prefix caching discipline

Both provider-side prompt caching and self-hosted prefix caching require an **exact byte-identical prefix**:
- Order segments from most stable to most volatile.
- Never inject timestamps, random IDs, or reordered tool lists early in the prompt.
- **Compaction breaks the cache** for everything after the edit point. Compact rarely and in large steps (e.g. when reaching 80% of the budget) rather than trimming one message per turn.

### 5.5 Reference: token-budget context packer

```python
from dataclasses import dataclass
from typing import Callable, Optional


class ContextOverflow(Exception):
    pass


@dataclass
class Segment:
    role: str                   # system | user | assistant | tool
    text: str
    priority: int = 0           # higher = keep longer
    pinned: bool = False        # never dropped (system prompt, latest user turn)
    group: Optional[str] = None # atomic unit, e.g. a tool call and its result
    tokens: int = 0


class ContextPacker:
    def __init__(self, count_tokens: Callable[[str], int], window: int, max_output: int,
                 per_message_overhead: int = 6, safety: float = 0.02):
        self.count = count_tokens
        self.overhead = per_message_overhead        # template tokens per message
        self.limit = int(window * (1 - safety)) - max_output

    def cost(self, s: Segment) -> int:
        if not s.tokens:
            s.tokens = self.count(s.text) + self.overhead
        return s.tokens

    def pack(self, segments: list[Segment],
             summarize: Optional[Callable[[list[Segment], int], str]] = None) -> list[Segment]:
        """segments are chronological; returns a chronological list within budget."""
        if sum(self.cost(s) for s in segments) <= self.limit:
            return segments

        groups: dict[str, list[Segment]] = {}
        order: list[str] = []
        for i, s in enumerate(segments):
            key = s.group or f"__{i}"
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(s)

        gcost = {k: sum(self.cost(s) for s in g) for k, g in groups.items()}
        pinned = {k for k in order if any(s.pinned for s in groups[k])}
        budget = self.limit - sum(gcost[k] for k in pinned)
        if budget < 0:
            raise ContextOverflow("pinned content alone exceeds the budget")

        # Keep the highest priority first; break ties by recency (later index wins).
        candidates = sorted(
            ((max(s.priority for s in groups[k]), idx, k)
             for idx, k in enumerate(order) if k not in pinned),
            reverse=True)
        keep, dropped = set(pinned), []
        for _, _, k in candidates:
            if gcost[k] <= budget:
                keep.add(k)
                budget -= gcost[k]
            else:
                dropped.append(k)

        out = [s for k in order if k in keep for s in groups[k]]

        if dropped and summarize and budget > 64:
            dropped_segs = [s for k in order if k in dropped for s in groups[k]]
            summary = Segment("system", summarize(dropped_segs, budget - self.overhead), pinned=True)
            if self.cost(summary) <= budget:
                # insert after the leading system messages so the stable prefix is preserved
                pos = next((i for i, s in enumerate(out) if s.role != "system"), len(out))
                out.insert(pos, summary)
        return out
```

In production, run summarization **asynchronously** when the history crosses a threshold, so no request pays for the extra LLM call. Keep summaries append-only so that prefixes stay cacheable.

---

## 6. Evaluating Effective Context

| Benchmark / method | Measures |
|---|---|
| **Needle-in-a-haystack** (single needle) | Literal retrieval at (length × depth). Necessary but far from sufficient |
| **RULER** | Multi-needle, multi-key/value, variable tracking, aggregation, QA; reports the *effective* length where performance holds above a threshold |
| **LongBench** (and successors) | Real long-document tasks: QA, summarization, code |
| **Lost-in-the-middle protocol** | Accuracy vs position of the gold document among distractors |
| **Your own task at production lengths** | The only number that matters for your product |

**Protocol tips:** vary depth *and* length. Use distractors that are semantically similar to the needle. Include tasks that require *using* the information (multi-hop reasoning), not only quoting it. Measure TTFT and cost on the same grid.

---

## 7. Production Challenges & How to Solve Them

| Challenge | Solution |
|---|---|
| **Context-length errors in production** | Count tokens with the exact tokenizer after templating (01.03); reserve output; enforce the budget in the gateway, not only in the app |
| **Truncation corrupts structure** | Never cut mid-JSON, mid-code-block, or between a tool call and its result; truncate at semantic units (the `group` field above) |
| **Quality decay in long threads** | Rolling compaction; restate the task and constraints near the end of the prompt (recency); pin critical instructions in the system prompt |
| **Relevant facts ignored mid-context** | Rerank and place the best evidence first and last; cut distractors; ask the model to quote the evidence it used, then verify |
| **Cost explosion with agents** | Tool outputs dominate. Summarize or paginate them, store full results out of context behind an ID and a fetch tool, and cap per-step output |
| **TTFT on long prompts** | Prefix caching of static documents; chunked prefill; smaller retrieval sets; distill long instructions |
| **Serving OOM at long context** | Per-tier max-length limits; KV quantization; route long requests to dedicated high-memory replicas |
| **Context-extension regressions** | Evaluate short-context tasks after YaRN/PI changes; enable scaling conditionally where supported |

---

## 8. Hands-On Projects

### Project 1 — "Infinite Chat" Context Manager with Measurable Recall

**User stories**
- *As a chat-product engineer*, I want conversations that can run indefinitely without context errors, and that still remember critical facts from early turns, so that long user sessions don't degrade.
- *As a finance owner*, I want prompt-cache hit rates and token costs tracked, so that context management does not quietly double spend.

**Acceptance criteria**
1. A service (FastAPI, or Spring Boot calling a model endpoint) wraps any OpenAI-compatible model. It uses §5.5-style packing, atomic tool-call groups, async rolling summarization at an 80% threshold, and a fact-extraction memory store (e.g. SQLite + embeddings).
2. **Property test:** over 10,000 randomly generated conversations (random lengths, tool calls, large tool outputs), the packed prompt **never** exceeds the budget and never splits a tool-call group.
3. **Recall eval:** plant 20 facts across a synthetic conversation 5× longer than the window; ≥ 90% are answered correctly at the end (vs a naive truncation baseline you report).
4. Metrics are exported: tokens in/out per turn, compaction events, cache-hit tokens (from the engine or provider usage fields), and P95 added latency < 50 ms (excluding async summarization).
5. A written ADR explains ordering decisions made for cache stability.

**Step-by-step**
1. Implement `Segment`, `ContextPacker`, and a message store keyed by conversation ID.
2. Add `summarize()` using the model itself, with a strict prompt: preserve entities, decisions, numbers, and open tasks; output ≤ N tokens.
3. Add fact extraction per turn (structured JSON output, from 01.02 §6), embed the facts, and retrieve the top-k per new turn into a pinned "memory" segment.
4. Write Hypothesis property tests for criterion 2.
5. Build the synthetic long-conversation generator with planted facts and questions. Run the naive, summary-only, and summary+memory variants.
6. Instrument with OpenTelemetry and Prometheus. Dashboard cost and cache-hit rate.

---

### Project 2 — Context Extension Lab: PI vs NTK vs YaRN

**User stories**
- *As an ML engineer*, I want to extend an open model's context 4× and prove it works on retrieval *and* on short tasks, so that I understand the real trade-offs of RoPE scaling.

**Acceptance criteria**
1. Base: a small open RoPE model (≈ 0.5–1.5B) with a known training length $L$.
2. Implements PI, NTK-aware, and YaRN by modifying the model's inverse frequencies (your own code, checked against the library config paths).
3. Before any fine-tuning, plots perplexity vs position up to 4L on long documents (e.g. PG-19 or a long-code corpus) for base, PI, NTK, and YaRN.
4. Fine-tunes the PI and YaRN variants for ≤ 400 steps at length 4L, using gradient checkpointing, FlashAttention, and optionally LoRA with trainable norms/embeddings.
5. **Passkey retrieval** at lengths {L, 2L, 4L} × 10 depths: the fine-tuned YaRN model reaches ≥ 95% accuracy at 4L.
6. A short-context regression check (a few standard benchmarks at ≤ L) shows ≤ 2 points of degradation, and the result is reported honestly if it is exceeded.

**Step-by-step**
1. Load the model; locate `inv_freq` in the rotary embedding module; write `apply_pi(s)`, `apply_ntk(s)`, and `apply_yarn(s, alpha=1, beta=32)`, including the attention temperature.
2. Build the evaluation data: long documents chunked at 4L, and a passkey generator (a random 5-digit number hidden in filler text at depth d, with a question at the end).
3. Measure the zero-shot curves (criterion 3).
4. Fine-tune with packed 4L sequences (FlashAttention varlen), a small learning rate, and a short warmup.
5. Re-run the passkey grid and plot heatmaps (length × depth).
6. Serve the result with vLLM using the matching `rope_scaling` configuration, and confirm that the engine's outputs match your Hugging Face outputs.

---

### Project 3 — Long Context vs RAG Benchmark Harness

**User stories**
- *As an AI architect*, I need evidence for when to use long context and when to use retrieval for our document QA, measured on accuracy, latency, and cost, so that design decisions aren't based on anecdotes.

**Acceptance criteria**
1. Corpus: ≥ 200 long documents in your domain (e.g. policy documents, legislation, technical manuals). A question set of ≥ 300 items with gold answers and gold evidence spans, covering single-fact, multi-hop, and aggregation questions.
2. Pipelines: (a) full-document long context; (b) RAG (hybrid BM25 + embeddings, a reranker, top-k in {5, 10, 20}); (c) RAG with 32k tokens of ordered context (hybrid).
3. **Position study:** for (a), reproduce the lost-in-the-middle curve by moving the gold evidence across 10 depths.
4. Measures accuracy (exact-match or LLM-judge with a calibrated rubric; the judge is validated against 50 human labels, with agreement reported), P50/P95 TTFT, E2E latency, and cost per query.
5. Delivers a decision matrix (question type × pipeline) and a recommended default architecture, with confidence intervals.

**Step-by-step**
1. Build the corpus loader and chunker (by headings or sections; 300–800 tokens with overlap).
2. Generate questions (LLM-assisted, then human-reviewed), tag them by type, and store the gold spans.
3. Implement the pipelines behind a common interface: `answer(question) -> {answer, evidence, usage, timings}`.
4. For the position study, construct prompts with the gold document at a controlled depth among k distractors.
5. Run everything against one served model, logging token usage and timings per call.
6. Analyse with bootstrap confidence intervals; write the decision matrix and recommendation.

---

## 9. Foundational Papers (exact titles)

**Context extension**
- Chen et al., 2023 — *Extending Context Window of Large Language Models via Positional Interpolation*
- Peng et al., 2023 — *YaRN: Efficient Context Window Extension of Large Language Models*
- Ding et al., 2024 — *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*
- Xiong et al., 2023 — *Effective Long-Context Scaling of Foundation Models*
- Fu et al., 2024 — *Data Engineering for Scaling Language Models to 128K Context*
- Mohtashami & Jaggi, 2023 — *Landmark Attention: Random-Access Infinite Context Length for Transformers* (passkey retrieval test)

**Long-context architectures**
- Beltagy et al., 2020 — *Longformer: The Long-Document Transformer*
- Zaheer et al., 2020 — *Big Bird: Transformers for Longer Sequences*
- Jiang et al., 2023 — *Mistral 7B*
- Liu, Zaharia & Abbeel, 2023 — *Ring Attention with Blockwise Transformers for Near-Infinite Context*
- Xiao et al., 2023 — *Efficient Streaming Language Models with Attention Sinks*
- Gu & Dao, 2023 — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Dao & Gu, 2024 — *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*

**KV compression**
- Zhang et al., 2023 — *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*
- Li et al., 2024 — *SnapKV: LLM Knows What You are Looking for Before Generation*
- Liu et al., 2024 — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*

**Evaluation and usage**
- Liu et al., 2023 — *Lost in the Middle: How Language Models Use Long Contexts*
- Hsieh et al., 2024 — *RULER: What's the Real Context Size of Your Long-Context Language Models?*
- Bai et al., 2023 — *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*
- Xu et al., 2023 — *Retrieval meets Long Context Large Language Models*
- Packer et al., 2023 — *MemGPT: Towards LLMs as Operating Systems*

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **vLLM / SGLang** | `rope_scaling` configs, prefix caching, chunked prefill, KV-cache quantization |
| **flash-attn (varlen)**, **FlexAttention** | Efficient long-sequence training and custom sparse masks |
| **Hugging Face `transformers` / TRL / PEFT** | Long-context fine-tuning with LoRA and gradient checkpointing |
| **RULER, LongBench repos**, **lm-evaluation-harness** | Standardised long-context evaluation |
| **Provider prompt-caching features** | Cost and latency reduction for stable prefixes; read `usage` fields for cached-token counts |
| **LlamaIndex / LangChain / Haystack** (or your own) | RAG plumbing for Project 3; build the core yourself first |
| **OpenTelemetry + Prometheus/Grafana** | Token, cost, and cache-hit observability |
