# 01.02 — Inference & Decoding

> **Module 1: LLM Fundamentals** · Subtopic 2 of 4
> **Prerequisites:** 01.01 (KV cache, attention cost), basic probability, GPU memory hierarchy (HBM vs SRAM), async I/O.
> **Outcome:** you can explain every millisecond of an LLM request, and choose and tune a serving engine against latency SLOs. You can also implement sampling, speculative decoding, and constrained decoding correctly.

---

## 1. Anatomy of a Request

```
 client ──► gateway ──► scheduler ──► [ PREFILL ] ──► [ DECODE × N ] ──► detokenize/stream ──► client
                              │            │                  │
                              │    all prompt tokens     one token per step
                              │    in parallel           per sequence
                              │    COMPUTE-bound         MEMORY-BANDWIDTH-bound
                              │    writes KV cache       reads ALL weights + KV each step
                              ▼
                     admission control, batching, KV block allocation, preemption
```

### 1.1 The metrics that define "fast"

| Metric | Definition | Dominated by |
|---|---|---|
| **TTFT** (time to first token) | Arrival → first output token | Queueing + prefill ($\propto$ prompt length, and $T^2$ at long context) |
| **TPOT / ITL** (time per output token / inter-token latency) | Mean or distribution of gaps between tokens | Decode step time, which depends on batch size and interference from other requests' prefills |
| **E2E latency** | $\text{TTFT} + (n_{\text{out}}-1)\cdot\text{TPOT}$ | Mostly output length |
| **Throughput** | Output tokens/s across all requests | Batch size |
| **Goodput** | Requests/s that **meet the SLO** (e.g. P99 TTFT < 500 ms and P99 ITL < 50 ms) | What you actually optimise |

Always report **P50, P90, and P99**, never just averages. Chat users feel TTFT. Agents and batch jobs feel E2E latency and cost per token.

---

## 2. Prefill vs Decode: The Roofline View

**Arithmetic intensity** $I$ = FLOPs per byte moved from HBM. A kernel is memory-bound when $I$ is below the GPU's **ridge point**, $\text{peak FLOP/s} / \text{peak bandwidth}$. For an H100 SXM running dense bf16 that is roughly $989\,\text{TFLOP/s} / 3.35\,\text{TB/s} \approx 295$ FLOP/byte.

For a matrix multiply of weights $W \in \mathbb{R}^{d\times d}$ (bf16, 2 bytes per element) against $B$ tokens:

$$
I \approx \frac{2\,B\,d^2}{2\,d^2} = B \quad \text{FLOP/byte}
$$

- **Prefill:** $B = $ number of prompt tokens, often thousands, so $I \gg 295$ and the work is **compute-bound**.
- **Decode:** $B = $ the number of sequences in the batch (one new token each). At batch 1, $I \approx 1$, so the work is **severely memory-bound**.

**Upper bound on single-stream decode speed:**

$$
\text{tok/s}_{\max} \approx \frac{\text{HBM bandwidth}}{\text{bytes of weights} + \text{bytes of KV read per step}}
$$

*Example:* an 8B model in bf16 (16 GB) on 3.35 TB/s gives at most about 200 tok/s at batch 1, before any overhead. INT4 weights (≈ 4.5 GB with scales) raise that ceiling about 3.5×. **This is why weight quantization speeds up decode even when no FLOPs are saved.**

Batching amortises one weight read over $B$ sequences. Throughput rises almost linearly until the step becomes compute-bound, or until the KV cache runs out of memory. KV reads grow with $B\cdot T$ and are *not* amortised, which is why long-context decode stays memory-bound even at high batch.


![LLM-04](/01.%20LLM%20Fundamentals/images/LLM-04.jpg)

---

## 3. KV Cache Management & Batching

### 3.1 Static batching vs continuous batching

With **static batching**, a batch runs until its longest sequence finishes, and the GPU idles on finished slots. **Continuous (iteration-level) batching** (Orca) re-forms the batch on every decode step: finished sequences leave immediately and waiting ones join. Throughput typically improves several-fold on workloads with varied output lengths.

### 3.2 PagedAttention

Naive KV allocation reserves `max_len` contiguous slots per request. The result is internal fragmentation, and typically most reserved KV memory goes unused. **PagedAttention** (vLLM) splits the KV cache into fixed-size **blocks** (e.g. 16 tokens) and gives each sequence a **block table** mapping logical to physical blocks, much like OS virtual memory:

```
Sequence A logical blocks:  [0][1][2]        Physical KV pool (GPU):
Sequence B logical blocks:  [0][1]           ┌──┬──┬──┬──┬──┬──┬──┬──┐
                                             │A0│B0│A1│  │B1│A2│  │  │
Block table A: 0→0, 1→2, 2→5                 └──┴──┴──┴──┴──┴──┴──┴──┘
Block table B: 0→1, 1→4                      waste ≤ 1 partially-filled block per sequence
```

What this buys you:
- Near-zero fragmentation.
- **Copy-on-write sharing** for parallel sampling and beam search.
- Cheap **preemption**, either by swapping blocks to CPU or by dropping and recomputing.

### 3.3 Prefix caching

Requests often share prefixes: system prompts, few-shot examples, RAG documents, and multi-turn history. **Automatic prefix caching** hashes each full block, chaining the hash with its prefix, and reuses matching blocks. **RadixAttention** (SGLang) stores prefixes in a radix tree with LRU eviction. A cache hit skips prefill for the shared part, which cuts TTFT roughly in proportion to the hit length.

> **Design rule:** put stable content first (system prompt → tools → static docs → history → new user turn). A timestamp or request ID at the top of the prompt destroys prefix reuse, both in your own engine and in provider-side prompt caching.

### 3.4 Chunked prefill and prefill/decode disaggregation

One long prefill scheduled into a decode batch stalls every other stream's next token, and ITL spikes. Two mitigations:

- **Chunked prefill** (Sarathi-Serve): split prefills into chunks (e.g. 512–2048 tokens) and co-schedule them with decode steps under a per-step **token budget**. This smooths ITL and slightly delays TTFT.
- **Disaggregation** (DistServe, Splitwise): run prefill and decode on **separate GPU pools** and transfer the KV cache between them over NVLink or RDMA. Each pool can be tuned independently (e.g. more tensor parallelism for prefill, larger batches for decode). The cost is KV transfer and more operational complexity.

---

## 4. Decoding Algorithms — Exact Definitions

Let the logits be $z \in \mathbb{R}^V$ at the current step.

### 4.1 Temperature

$$
p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

As $\tau \to 0$ this becomes argmax (greedy). $\tau > 1$ flattens the distribution. Treat $\tau = 0$ as an explicit greedy branch; never divide by zero.

### 4.2 Truncation samplers

- **Top-k:** keep the $k$ highest-probability tokens and renormalise.
- **Top-p / nucleus** (Holtzman et al.): keep the smallest set $V_p$ with $\sum_{i\in V_p} p_i \ge p$.
- **Min-p:** keep tokens with $p_i \ge p_{\min}\cdot \max_j p_j$. The threshold scales with model confidence, so it is aggressive when the model is confident and permissive when it is uncertain. It holds up better than top-p at high temperature.

**Order of operations matters.** Common engines apply penalties → temperature → top-k → top-p → min-p → sample, but engines differ. Document the order your stack uses; it changes outputs.

### 4.3 Repetition controls

- **Frequency and presence penalties** (OpenAI-style), where $c_j$ is the count of token $j$ generated so far:

$$
z_j \leftarrow z_j - \alpha_{\text{freq}}\cdot c_j - \alpha_{\text{pres}}\cdot \mathbb{1}[c_j > 0]
$$

- **Repetition penalty** (CTRL-style, multiplicative): $z_j \leftarrow z_j/\theta$ if $z_j > 0$, else $z_j\cdot\theta$, applied to every token already seen.

### 4.4 Beam search

Beam search keeps the $b$ best partial sequences by cumulative log-probability, with length normalisation $\frac{1}{|y|^\alpha}\sum\log p$. It is good for constrained, low-entropy tasks (translation, some code infilling) and bad for open-ended text, where it produces bland, repetitive output (the "beam search curse"). It is rarely used in chat serving.

### 4.5 Reference sampler

```python
import torch
import torch.nn.functional as F


def sample_next(
    logits: torch.Tensor,            # (B, V) fp32
    prev_tokens: torch.Tensor,       # (B, T_gen) generated ids so far, -1 padded
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    freq_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    logits = logits.float().clone()
    B, V = logits.shape

    # 1. penalties
    if freq_penalty or presence_penalty:
        counts = torch.zeros(B, V + 1, device=logits.device)
        counts.scatter_add_(1, prev_tokens.clamp(min=-1) + 1,
                            torch.ones_like(prev_tokens, dtype=torch.float))
        counts = counts[:, 1:]                        # drop the padding bucket
        logits -= freq_penalty * counts + presence_penalty * (counts > 0).float()

    # 2. greedy short-circuit
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits /= temperature

    # 3. top-k
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, V), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    # 4. top-p (nucleus): keep tokens until cumulative mass first reaches top_p
    if top_p < 1.0:
        sorted_p, idx = probs.sort(dim=-1, descending=True)
        cum = sorted_p.cumsum(dim=-1)
        drop_sorted = (cum - sorted_p) >= top_p      # the token that crosses p is kept
        drop = torch.zeros_like(drop_sorted).scatter(1, idx, drop_sorted)
        probs = probs.masked_fill(drop, 0.0)

    # 5. min-p
    if min_p > 0.0:
        thresh = min_p * probs.max(dim=-1, keepdim=True).values
        probs = probs.masked_fill(probs < thresh, 0.0)

    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
```

---

## 5. Speculative Decoding

Decode is memory-bound, so verifying $\gamma+1$ tokens in **one target forward pass** costs about the same as generating one. Speculative decoding exploits this: a cheap **draft** proposes $\gamma$ tokens and the **target** verifies them in parallel.

### 5.1 The acceptance rule (lossless)

Let the draft propose $x \sim q(\cdot)$ and let the target distribution be $p(\cdot)$.

1. Accept $x$ with probability $\min\!\left(1,\ \dfrac{p(x)}{q(x)}\right)$.
2. On rejection, sample from the residual $\;p'(x) = \dfrac{\max(0,\ p(x) - q(x))}{\sum_{x'}\max(0,\ p(x')-q(x'))}$ and stop.
3. If all $\gamma$ drafts are accepted, sample one **bonus** token from $p$ at position $\gamma+1$.

This procedure yields samples distributed **exactly** as $p$. Speculative decoding changes speed, not quality. That property is the reason it is safe to deploy.

### 5.2 Expected speedup

With per-token acceptance rate $\alpha$ (treated as i.i.d.), the expected number of tokens produced per target pass is

$$
\mathbb{E}[\text{tokens/step}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}
$$

Speedup is approximately $\dfrac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$, where $c$ is the draft-to-target cost ratio. **The practical implication:** the gain is largest at low batch, low temperature, and on predictable domains (code, structured output, RAG answers that copy from context). At large batch the target is closer to compute-bound, verification is no longer "free", and gains shrink or turn negative.

### 5.3 Draft strategies

| Strategy | Draft source | Notes |
|---|---|---|
| Separate small model | Same-family small LM | Must share the tokenizer |
| **Prompt lookup / n-gram** | Copies n-grams from the prompt | Zero cost; excellent for RAG, editing, and code |
| **Medusa** | Extra decoding heads on the target, tree verification | Needs head training |
| **EAGLE family** | Lightweight autoregressive head operating on target hidden features | Among the strongest; supported in major engines |
| **MTP** (multi-token prediction) | Heads trained with the model (e.g. DeepSeek-V3) | Free drafts if the model shipped with them |

### 5.4 Reference implementation (educational: no KV cache, batch 1)

```python
import torch
import torch.nn.functional as F


@torch.no_grad()
def speculative_generate(target, draft, input_ids, max_new_tokens=128, gamma=4, temperature=1.0):
    """HF-style causal LMs sharing a tokenizer. Returns ids and the acceptance rate."""
    assert temperature > 0, "use a separate greedy-match variant for temperature=0"
    ids, n_start = input_ids, input_ids.shape[1]
    proposed = accepted_total = 0

    while ids.shape[1] - n_start < max_new_tokens:
        L = ids.shape[1]
        # 1. draft gamma tokens autoregressively
        draft_ids, q_probs = ids, []
        for _ in range(gamma):
            q = F.softmax(draft(draft_ids).logits[:, -1, :].float() / temperature, dim=-1)
            tok = torch.multinomial(q, 1)
            q_probs.append(q[0])
            draft_ids = torch.cat([draft_ids, tok], dim=1)

        # 2. one target pass scores positions L-1 .. L+gamma-1 (predicting L .. L+gamma)
        t_logits = target(draft_ids).logits[0, L - 1:, :].float()
        p_probs = F.softmax(t_logits / temperature, dim=-1)         # (gamma+1, V)

        # 3. accept / reject left to right
        n_acc, new_tok = 0, None
        for i in range(gamma):
            tok = draft_ids[0, L + i]
            ratio = p_probs[i, tok] / q_probs[i][tok]
            if torch.rand((), device=ratio.device) < ratio.clamp(max=1.0):
                n_acc += 1
            else:
                residual = (p_probs[i] - q_probs[i]).clamp(min=0)
                new_tok = torch.multinomial(residual / residual.sum(), 1)
                break
        if new_tok is None:                                         # all accepted → bonus token
            new_tok = torch.multinomial(p_probs[gamma], 1)

        proposed += gamma
        accepted_total += n_acc
        ids = torch.cat([ids, draft_ids[:, L:L + n_acc], new_tok.view(1, 1)], dim=1)

    return ids[:, : n_start + max_new_tokens], accepted_total / max(proposed, 1)
```

A production version keeps KV caches for both models, rolls back the draft cache on rejection, batches tree-structured drafts, and fuses verification into the engine's attention kernel.

---

## 6. Structured / Constrained Decoding

Structured decoding guarantees output that conforms to a JSON Schema, regex, or context-free grammar by **masking invalid tokens** before sampling:

$$
z_j \leftarrow \begin{cases} z_j & j \in \mathcal{A}(s_t) \\ -\infty & \text{otherwise}\end{cases}
$$

Here $\mathcal{A}(s_t)$ is the set of tokens allowed from grammar state $s_t$.

- **Regex / JSON Schema → FSM** (Outlines, Willard & Louf): precompute, for each FSM state, which vocabulary tokens are valid transitions. The per-step cost is then an index lookup.
- **CFGs → pushdown automata** (XGrammar, llguidance): handle recursion such as nested JSON. They split the vocabulary into context-independent tokens (precomputed) and context-dependent ones (checked at runtime), and overlap mask computation with the GPU forward pass.

```python
def apply_grammar_mask(logits: torch.Tensor, allowed_token_ids: torch.Tensor) -> torch.Tensor:
    mask = torch.full_like(logits, float("-inf"))
    mask[..., allowed_token_ids] = 0.0
    return logits + mask
```

**Production caveats**
- Masking distorts the model's distribution. If the prompt does not already steer toward the schema, quality drops (the model is forced down low-probability paths). Always *also* describe the schema in the prompt.
- Tokenizer and grammar boundary mismatches: one token can span several grammar symbols. Use a library that handles this correctly; do not hand-roll it.
- Compile grammars ahead of time and cache them by schema hash. Compile time for large schemas can exceed the generation time.

---

## 7. Quantization for Inference

Affine quantization of a tensor $x$ to $b$ bits with scale $s$ and zero-point $z$:

$$
x_q = \mathrm{clamp}\!\left(\left\lfloor \frac{x}{s}\right\rceil + z,\ q_{\min},\ q_{\max}\right),
\qquad \hat x = s\,(x_q - z),
\qquad s = \frac{x_{\max} - x_{\min}}{2^b - 1}
$$

Scales can be per-tensor, per-channel, or **per-group** (e.g. 128 weights per scale, the standard for INT4).

| Method | What | When |
|---|---|---|
| **Weight-only INT4/INT8** (GPTQ, AWQ) | Weights quantized, activations bf16 | Memory-bound decode at low batch; the biggest latency win |
| **GPTQ** | Second-order (Hessian-based), layer-wise error compensation | Strong 4-bit quality, needs calibration data |
| **AWQ** | Protects salient channels by scaling, guided by activation magnitudes | Robust and fast to produce |
| **W8A8 (SmoothQuant)** | Migrates activation outliers into weights, then INT8 GEMMs | Compute-bound, high-batch serving |
| **FP8 (E4M3) W8A8** | Native on Hopper and later GPUs | Near-lossless on many models; common default for high-throughput serving |
| **KV-cache quantization** (FP8/INT8/INT4) | Compresses the cache | Doubles or quadruples the number of sequences that fit |

**Always evaluate** on your own task suite (not only perplexity), plus long-context and tool-calling evals. Quantization damage is often task-specific.

---

## 8. Parallelism for Serving

| Type | Splits | Communication | Use when |
|---|---|---|---|
| **Tensor parallel (TP)** | Each matmul across GPUs (column/row split) | 2 all-reduces per layer, latency-sensitive | The model doesn't fit on one GPU; needs NVLink within a node |
| **Pipeline parallel (PP)** | Layers across GPUs or nodes | Point-to-point activations | Spanning nodes; adds pipeline bubbles |
| **Data parallel (replicas)** | Whole model copies | None | Scaling throughput once one replica fits |
| **Expert parallel (EP)** | MoE experts across GPUs | All-to-all per MoE layer | Large MoE models |
| **Context parallel (CP)** | Sequence dimension | Ring exchange of KV | Very long prompts (see 01.04) |

**Rule of thumb:** use the smallest TP degree that fits the weights plus enough KV cache for your target concurrency, then scale out with replicas. Higher TP lowers per-token latency but loses efficiency to communication.

---

## 9. Measuring It Yourself — Streaming Load Tester

This script works against any OpenAI-compatible endpoint (vLLM, SGLang, TensorRT-LLM, llama.cpp server, and others).

```python
import asyncio, json, time, statistics, random
import httpx

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "your-model"


async def one_request(client: httpx.AsyncClient, prompt: str, max_tokens: int):
    body = {"model": MODEL, "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    token_times = []
    async with client.stream("POST", URL, json=body, timeout=300) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                token_times.append(time.perf_counter())  # a chunk ≈ a token for most servers
    if not token_times:
        return None
    ttft = token_times[0] - t0
    itls = [b - a for a, b in zip(token_times, token_times[1:])]
    return ttft, itls, token_times[-1] - t0, len(token_times)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


async def run(qps: float, duration_s: float, prompts: list[str], max_tokens=256):
    results, tasks = [], []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000)) as client:
        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            tasks.append(asyncio.create_task(one_request(client, random.choice(prompts), max_tokens)))
            await asyncio.sleep(random.expovariate(qps))   # Poisson arrivals, not a fixed loop
        results = [r for r in await asyncio.gather(*tasks) if r]
    ttfts = [r[0] for r in results]
    itls = [x for r in results for x in r[1]]
    toks = sum(r[3] for r in results)
    print(f"QPS={qps}  n={len(results)}  out_tok/s={toks / duration_s:.1f}")
    print(f"TTFT  p50={pct(ttfts,50)*1e3:.0f}ms  p99={pct(ttfts,99)*1e3:.0f}ms")
    print(f"ITL   p50={pct(itls,50)*1e3:.1f}ms  p99={pct(itls,99)*1e3:.1f}ms  mean={statistics.mean(itls)*1e3:.1f}ms")


if __name__ == "__main__":
    asyncio.run(run(qps=4, duration_s=60, prompts=["Explain paged attention in detail."]))
```

Use **Poisson arrivals** and realistic prompt and output length distributions, sampled from production logs. A closed-loop "N concurrent users" test hides queueing collapse.

---

## 10. Production Challenges & How to Solve Them

| Challenge | Solution pattern |
|---|---|
| **High TTFT** | Prefix caching with stable prompt ordering; chunked prefill; shorter prompts (01.04); disaggregated prefill; more TP on the prefill pool |
| **ITL spikes / jitter** | Chunked prefill with a per-step token budget; separate pools for long-prompt traffic; cap max prompt length per tier |
| **KV OOM & preemption storms** | Set `gpu_memory_utilization`, `max_num_seqs`, and max model length deliberately; KV quantization; admission control that queues instead of admitting and then preempting |
| **Cost per token** | FP8/INT4 weights; larger batches for offline work; speculative decoding for latency-bound tiers only; right-size the model per task (routing) |
| **Non-determinism at temperature 0** | Results change with batch composition because floating-point reductions are not associative and kernels pick different split strategies per batch size. Use batch-invariant kernels where the engine supports them, a fixed seed per request, and accept (and test for) small divergence |
| **Cancellations & disconnects** | Propagate client aborts to the engine so it frees KV blocks immediately; enforce a server-side `max_tokens` |
| **Streaming correctness** | Incremental detokenization that buffers incomplete UTF-8 sequences (see 01.03) |
| **Tail latency under burst** | Queue-depth-based autoscaling (not GPU utilisation); load shedding with 429 + Retry-After; per-tenant rate limits |
| **Observability** | Export per-request TTFT, ITL, queue time, prompt and output lengths, cache-hit tokens, preemptions, and KV utilisation; trace IDs across gateway → engine |

---

## 11. Hands-On Projects

### Project 1 — Mini Serving Engine: Continuous Batching + Paged KV Cache

**User stories**
- *As an inference engineer*, I want to build a scheduler and block manager myself, so that I can read vLLM and SGLang internals fluently and debug their behaviour.
- *As an operator*, I want the engine to preempt gracefully under memory pressure, so that it never crashes with OOM.

**Acceptance criteria**
1. Serves a small Llama-style model (your 01.01 model or a ~0.5–1B open model) through an async API with token streaming.
2. **Block manager:** fixed block size (16), a free list, per-sequence block tables, and wasted KV slots ≤ `block_size − 1` per live sequence (verified by a test).
3. **Scheduler:** iteration-level batching with a per-step token budget and chunked prefill. New requests join mid-flight.
4. **Preemption:** under an artificially small KV pool, requests are preempted by recompute and still complete with correct output (compared to an unconstrained run with greedy decoding).
5. **Prefix caching:** hash-chained full blocks are reused. A benchmark with a shared 2k-token system prompt shows TTFT reduced by ≥ 50% on cache hits.
6. With Poisson load and mixed output lengths, throughput is ≥ 3× a static-batching baseline at the same P99 TTFT.

**Step-by-step**
1. **Data structures.** `Sequence(id, tokens, block_table, status)`, `BlockManager(num_blocks, block_size)` with `allocate`, `append_slot`, `free`, and `can_allocate`.
2. **Paged attention in PyTorch.** Store K/V as `(num_blocks, block_size, n_kv, d_h)`. For each sequence, gather its blocks through the block table into a contiguous tensor, then call SDPA. This is slow but correct; optionally write a Triton kernel later.
3. **Scheduler loop.** Each step: (a) keep running decodes, (b) admit waiting prefills while the token budget and free blocks allow, (c) if `append_slot` fails, preempt the lowest-priority (newest) sequence.
4. **Model runner.** Flatten the batch into one ragged forward pass (varlen) or pad per step. Sample with the §4.5 sampler.
5. **API.** A FastAPI endpoint with SSE streaming, backed by a per-request `asyncio.Queue` fed by the engine loop.
6. **Benchmark** with the §9 load tester. Plot throughput vs P99 TTFT for static vs continuous batching.

---

### Project 2 — SLO-Driven Capacity Planning with vLLM / SGLang

**User stories**
- *As a platform lead*, I need the cheapest configuration that meets P99 TTFT < 800 ms and P99 ITL < 60 ms at peak load, so that I can size the GPU budget for a product launch.

**Acceptance criteria**
1. The workload is modelled from a realistic prompt and output length distribution (e.g. a sampled public chat dataset, or synthetic log-normal lengths). The generator is checked into the repo.
2. Sweeps at least these dimensions: engine (vLLM vs SGLang), precision (bf16 / FP8 / INT4 weight-only), `max_num_seqs`, chunked prefill on/off, prefix caching on/off, and TP = 1/2.
3. For each config, finds the **max QPS meeting the SLO** by binary search over offered load. Outputs a Pareto chart (QPS at SLO vs GPU cost/hour).
4. A quality gate: every quantized config is evaluated on a 200-item task-specific eval, and is rejected if it scores > 2 points below bf16.
5. Final deliverable: a one-page recommendation with cost per 1M output tokens for the winning config.

**Step-by-step**
1. Launch the engine in Docker with pinned versions, and record the exact CLI flags per run.
2. Extend the §9 load tester with a length-distribution sampler and a JSON results writer.
3. Automate the sweep (a Python driver that restarts the server per config, waits for `/health`, warms up, and runs 3 × 2-minute trials).
4. Binary-search offered QPS: raise it until the P99 SLO fails, then bisect.
5. Run evals with `lm-evaluation-harness` or your own harness against the same endpoint.
6. Write the recommendation and include what you would monitor after launch.

---

### Project 3 — Speculative Decoding Lab + Guaranteed-Valid JSON

**User stories**
- *As a latency-focused engineer*, I want to measure when speculative decoding helps, so that I enable it only where it pays off.
- *As an application developer*, I need 100% schema-valid JSON from an open model, so that downstream parsers never fail.

**Acceptance criteria**
1. The §5.4 algorithm is implemented with a draft/target pair from the same family. A statistical test (χ² over 10k samples from a short prompt) shows the output distribution matches target-only sampling (p > 0.01).
2. Reports acceptance rate $\alpha$ and wall-clock speedup across γ ∈ {2, 4, 6, 8}, temperature ∈ {0.0, 0.7, 1.0}, and domains {code, chat, RAG-with-context}. Compares measured tokens per step against $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$.
3. Adds **prompt-lookup** drafting and shows it beating the small-model draft on the RAG domain.
4. Enables engine-native speculative decoding (e.g. an EAGLE-style or n-gram method in vLLM or SGLang) and shows the speedup shrinking as batch size grows.
5. Constrained decoding with a nested JSON Schema achieves 100% parse-and-validate success over 1,000 generations, vs an unconstrained baseline rate that you report.

**Step-by-step**
1. Pick a pair (e.g. a ~0.5B and a ~7B model from the same family). Verify the tokenizers are identical.
2. Implement §5.4 and add a greedy variant (accept while argmaxes match).
3. Build the evaluation grid and log α per position to see how acceptance decays within a draft.
4. Implement n-gram prompt lookup: find the last k tokens in the prompt and propose what followed them there.
5. Benchmark the engine's built-in speculative decoding at batch sizes 1, 8, 32, 64.
6. Use the engine's structured-output option (backed by XGrammar, Outlines, or llguidance) with a schema, and validate with `jsonschema`.

---

## 12. Foundational Papers (exact titles)

**Serving systems**
- Yu et al., 2022 — *Orca: A Distributed Serving System for Transformer-Based Generative Models*
- Kwon et al., 2023 — *Efficient Memory Management for Large Language Model Serving with PagedAttention*
- Zheng et al., 2024 — *SGLang: Efficient Execution of Structured Language Model Programs*
- Agrawal et al., 2024 — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- Zhong et al., 2024 — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*
- Patel et al., 2024 — *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*
- Pope et al., 2022 — *Efficiently Scaling Transformer Inference*

**Sampling**
- Fan et al., 2018 — *Hierarchical Neural Story Generation* (top-k)
- Holtzman et al., 2019 — *The Curious Case of Neural Text Degeneration* (nucleus)
- Keskar et al., 2019 — *CTRL: A Conditional Transformer Language Model for Controllable Generation* (repetition penalty)
- Nguyen et al., 2024 — *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*

**Speculative decoding**
- Leviathan et al., 2023 — *Fast Inference from Transformers via Speculative Decoding*
- Chen et al., 2023 — *Accelerating Large Language Model Decoding with Speculative Sampling*
- Cai et al., 2024 — *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*
- Li et al., 2024 — *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*

**Structured generation**
- Willard & Louf, 2023 — *Efficient Guided Generation for Large Language Models*
- Dong et al., 2024 — *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models*

**Quantization**
- Dettmers et al., 2022 — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*
- Frantar et al., 2022 — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- Xiao et al., 2022 — *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*
- Lin et al., 2023 — *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
- Micikevicius et al., 2022 — *FP8 Formats for Deep Learning*

**Parallelism**
- Shoeybi et al., 2019 — *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*

## 13. Essential Tooling

| Tool | Role |
|---|---|
| **vLLM** | PagedAttention, continuous batching, prefix caching, speculative decoding, structured outputs, OpenAI-compatible server |
| **SGLang** | RadixAttention, fast structured generation, strong multi-turn and agentic throughput |
| **TensorRT-LLM** (+ Triton Inference Server) | Maximum NVIDIA performance, in-flight batching, FP8 |
| **llama.cpp / GGUF**, **Ollama** | CPU, edge, and Apple-silicon inference; quantized local serving |
| **XGrammar / Outlines / llguidance** | Grammar-constrained decoding backends |
| **llm-compressor, AutoAWQ, GPTQ tooling** | Producing quantized checkpoints |
| **Engine benchmark CLIs, NVIDIA GenAI-Perf, Locust/k6** | Load and latency testing |
| **Prometheus + Grafana, OpenTelemetry** | Engine metrics and request tracing |
| **Nsight Systems** | Finding CPU scheduling gaps between GPU kernels |

> Pin exact engine versions in your projects. Flags, defaults, and supported speculative methods change between releases, so read the release notes for the versions you pin.
