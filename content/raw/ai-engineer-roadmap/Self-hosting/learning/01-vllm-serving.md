# 01 — vLLM Serving

> **Module goal:** Master the dominant inference engine for open LLMs. Understand what vLLM actually does under the hood (PagedAttention, continuous batching, prefix caching), the flags that matter, and how to squeeze 5–20× more throughput out of a GPU than a naive Hugging Face pipeline.

---

## 1. Executive Summary & Core Concepts

**vLLM** is an open-source LLM serving engine developed at UC Berkeley (Sky Lab) that has become the default choice for serving open-weight models in 2024–2026. Alternatives (Text Generation Inference / TGI, SGLang, llama.cpp) exist and are appropriate in specific contexts, but vLLM dominates general-purpose serving because of three engineering wins:

1. **PagedAttention** — a memory management scheme for the KV cache modeled on OS-level virtual memory paging. Eliminates the fragmentation that limits naive KV storage. Enables running many more concurrent requests in the same VRAM.

2. **Continuous batching (a.k.a. iteration-level scheduling)** — instead of batching requests together and waiting for the slowest one, vLLM batches at the *token* level. New requests join the batch dynamically; finished sequences leave. Massive throughput gains for realistic mixed-length workloads.

3. **Prefix caching** — identical prompt prefixes across requests are computed once and reused. If 100 requests share the same 6k-token system prompt, the KV state for those 6k tokens is computed once, not 100 times. Matches Anthropic-style prompt caching semantically.

Plus a mature ecosystem:
- OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
- Tensor parallelism across GPUs (`--tensor-parallel-size N`)
- LoRA adapter loading at request time (`--enable-lora`)
- Structured output (JSON schema, regex) via the Outlines/xgrammar integrations
- Prometheus metrics endpoint
- Full support for all major open models (Llama, Qwen, Mistral, Gemma, DeepSeek, Phi, Command R)

**Rule of thumb:** if you're serving an open LLM in production and you can't articulate a specific reason not to, use vLLM.

**Alternatives, briefly:**
- **TGI (Text Generation Inference)** — Hugging Face's serving engine. Similar features, tighter Hugging Face ecosystem integration; slightly less throughput on most benchmarks.
- **SGLang** — newer, research-oriented; excellent for complex multi-step LLM programs (structured generation, RAG). Rising.
- **llama.cpp / Ollama** — CPU + consumer-GPU focused; GGUF quantization; great for local desktop / Mac usage; not a production high-throughput server.
- **Managed platforms** (Together AI, Fireworks, Anyscale, Groq) — someone else runs vLLM (or similar) for you and charges per token. Sweet spot when you want open-model economics without the infra work.

---

## 2. Deep-Dive Breakdown

### 2.1 PagedAttention — Why vLLM Is Fast

Traditional LLM serving allocates a fixed-size KV cache per sequence. Two problems:

- **Internal fragmentation:** you allocate for `max_seq_len` but most sequences don't reach it. Wasted VRAM.
- **External fragmentation:** as sequences of different lengths come and go, VRAM gets Swiss-cheesed and you can't fit new requests.

PagedAttention treats KV cache like a paged memory system: fixed-size blocks (say 16 tokens each), allocated on demand, addressed via a block table per sequence. Analogous to OS virtual memory + physical page tables.

Result:
- Near-zero fragmentation
- Sequences with different lengths share VRAM efficiently
- Multiple sequences can even share KV cache blocks (prefix caching leverages this — see 2.4)

You don't tune PagedAttention directly. It's always on. Just know: this is why you can fit ~10× more concurrent requests in the same GPU than a naive implementation.

### 2.2 Continuous Batching

Old (bad) approach: **static batching**. You wait for N requests to arrive; process them together; the batch finishes when the *longest* sequence completes; new requests wait for the next batch cycle.

Problems:
- Short requests wait for long ones — high tail latency
- If N-1 sequences finished 500 tokens ago but the Nth still has 1000 tokens to go, the whole GPU is running for one request

vLLM's **continuous batching** (aka iteration-level scheduling) schedules at the token level. On every forward pass:
- Sequences that finished are removed
- New pending requests are added if there's capacity
- Everyone shares the compute of that single forward pass

Result: throughput scales sub-linearly with concurrent load (more requests → more efficient batching) instead of super-linearly (more requests → more waiting).

For a mixed-workload production service, continuous batching yields **3-10× throughput vs static batching** at the same latency.

### 2.3 Launching vLLM

The simplest possible launch:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

vLLM downloads the model from Hugging Face on first launch, warms it up, and exposes an OpenAI-compatible server. Test:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    }'
```

The important flags:

| Flag | Purpose | When to change |
|------|---------|----------------|
| `--model` | Model identifier (HF hub or local path) | Always |
| `--tensor-parallel-size N` | Split model across N GPUs | Multi-GPU serving |
| `--gpu-memory-utilization 0.9` | Fraction of VRAM vLLM can use | Increase if OOM-free (0.85-0.95); decrease if sharing GPU |
| `--max-model-len 8192` | Max context length | Reduce to save VRAM if you don't need long context |
| `--dtype bfloat16` | Compute dtype | `bfloat16` for Ampere+, `float16` for older, `float32` never |
| `--quantization awq` / `gptq` | Load a pre-quantized model | Model was quantized (see Module 02) |
| `--enable-prefix-caching` | Prefix caching on | Nearly always in production |
| `--enable-lora` | Support LoRA adapters | Serving fine-tuned adapters |
| `--served-model-name` | Alias for `model` in API | UX; make the model name match your client's expectation |
| `--api-key` | Require Bearer auth | Any exposed deployment |
| `--host 0.0.0.0` | Bind on all interfaces | Docker / cloud deployment |

Production launch example (7B on a single 24 GB GPU):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --served-model-name llama-3.1-8b \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --api-key $VLLM_API_KEY \
    --disable-log-requests
```

### 2.4 Prefix Caching

If two requests share a system prompt (or any leading tokens), vLLM's prefix cache computes the KV state for those tokens once and reuses it.

Enable with `--enable-prefix-caching`. Then:

```python
# Every request shares this 4k-token system prompt
SYSTEM = "You are a legal assistant. [4000-token instruction body...]"

# First request: full forward on the system prompt (~200ms)
r1 = client.chat.completions.create(model="llama-3.1-8b", messages=[
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Analyze this contract clause..."},
])

# Subsequent requests: system prompt KV is cached, only new tokens are computed (~5ms overhead)
r2 = client.chat.completions.create(model="llama-3.1-8b", messages=[
    {"role": "system", "content": SYSTEM},  # exact bytes match r1
    {"role": "user", "content": "Different question..."},
])
```

The cache is content-addressed (prefix hash), so any request with an identical byte prefix hits. Evictions happen on VRAM pressure (LRU-ish).

Prefix cache dramatically improves both **latency** (5x faster TTFT on cache hit) and **throughput** (the GPU isn't spending 90% of its time re-encoding the same system prompt across all concurrent users).

### 2.5 LoRA Adapters via `--enable-lora`

If you've fine-tuned with LoRA (fine-tuning track), you can serve multiple adapters against one base model:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules legal=./adapters/legal-classifier medical=./adapters/medical-summarizer \
    --max-loras 2 \
    --max-lora-rank 32 \
    --port 8000
```

Requests specify which adapter to use via the `model` field:

```python
r = client.chat.completions.create(
    model="legal",   # names the LoRA module, not the base
    messages=[...],
)
```

Trade-offs:
- ~2-5% latency overhead per request
- One base model in VRAM + adapters (adapters are small, ~50-200 MB each)
- N adapters served against 1 base = multi-tenant fine-tune serving

For maximum throughput on a single fine-tune: **merge and serve as the base** (fine-tuning Module 05). For multi-tenant fine-tune serving or A/B experimentation: **enable-lora** is the right call.

### 2.6 Deployment: Docker + Cloud GPU

vLLM's official Docker image:

```bash
docker run --gpus all --rm \
    -v $HF_HOME:/root/.cache/huggingface \
    -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.9 \
    --api-key $VLLM_API_KEY
```

On RunPod / Modal / any cloud with GPU:
- Use a "GPU cloud" instance with a modern card (A10G, L4, A100, H100)
- Persistent volume for the model cache (huge — HF downloads can be 15+ GB)
- Health check `/health` returns 200 when ready
- Metrics endpoint at `/metrics` for Prometheus scraping (vLLM emits many; `vllm_num_requests_running`, `vllm_prompt_tokens_total`, `vllm_generation_tokens_total`, `vllm_time_to_first_token_seconds`, etc.)

`[IMG-SH01-01]` — *Prompt: Diagram of vLLM's core innovations. Top half: "PagedAttention" — show a KV cache visualized as small blocks of 16 tokens, with block-tables per sequence pointing to physical blocks (like OS memory pages). Contrast with "Static allocation" showing large fixed pre-allocated slots with mostly-empty regions in each. Bottom half: "Continuous batching" — a timeline showing three request lanes; new requests join a shared batch as older ones finish; contrast with "Static batching" where all requests wait for the longest one before restart. Include a small "prefix cache" panel showing two requests sharing a system prompt with only the divergent tail computed twice. Clean technical illustration with warm colors.*

---

## 3. Mental Models & Analogies

### Model 1: Serving as a Restaurant Kitchen

**Static batching** = the kitchen fills a table of 8 diners, seats them all together, and won't take new orders until all 8 have finished eating. The diner who ordered a salad watches the diner ordering steak eat, and the kitchen sits idle until the last diner leaves.

**Continuous batching** = the kitchen is a rolling brunch. Each cook (a token generation step) checks the dining room: who needs another course? Who's done? Who just arrived? Fill trays, serve, repeat. Nobody waits for the slowest table. The kitchen operates near peak utilization.

**PagedAttention** = the kitchen's freezer used to allocate a fixed section per menu item, wasting space on rarely-ordered ones. Now it uses a paged storage system: small standardized bins, allocated on demand, with a lookup table so cooks know where anything lives. Same freezer, dramatically more capacity.

**Prefix caching** = the sauces and stocks are pre-batched at the start of service. When 40 orders share the same base sauce, you make it once, not 40 times.

All four combine to explain why vLLM produces 5-20× the throughput of a naive Hugging Face pipeline.

### Model 2: Airline vs. Bus

Static batching is like flying an airline that only takes off when all seats are full — great for the airline's fuel efficiency, terrible for the passenger. Continuous batching is like a bus that stops every few minutes: you never wait long to board, capacity fills up dynamically, and if the bus is full, the next one comes soon.

vLLM turns your LLM into that always-departing bus. The economics: bus companies serve more passengers per mile than fully-booked airplanes with delayed departures. So does continuous batching.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Setting `--gpu-memory-utilization` too aggressively.**
Default is 0.90. Tempting to set 0.95 or 0.98 to squeeze more. But vLLM allocates the KV cache pool at startup based on this value; if activations later need slightly more (long generation, batched requests), you OOM mid-request. Symptom: works for 5 minutes, then a big request crashes it. Fix: back off to 0.85-0.90; monitor `vllm_gpu_cache_usage_perc` and only push higher if you're consistently under-using.

**Pitfall #2 — Ignoring `--max-model-len`.**
vLLM defaults to the model's advertised context length (Llama 3.1: 128k). If you set no cap, vLLM allocates the KV cache for that max — even if your traffic uses only 2-4k of context. Result: much smaller concurrent batch, worse throughput, no user benefit. **Set `--max-model-len` to what you actually serve.** If your prompts + max_tokens rarely exceed 8k, set it to 8192. You may see 2-3× throughput improvement.

**Pitfall #3 — Not sizing the machine for the model.**
A 70B model in bf16 is ~140 GB. On a single 80 GB A100 it doesn't fit; on 2× 80 GB with `--tensor-parallel-size 2` it fits. On 4× 40 GB A100s it fits. On a single H100 80GB in 4-bit quantization (~35 GB) it fits. Newcomers often try to fit a 70B on a 40 GB card and get "CUDA out of memory" before the first request. Rule: **look up the model's fp16/bf16 size in GB (~2× param count), pick a card or set of cards that fits it plus ~30% for KV cache/activations.**

Or use tensor parallelism (`--tensor-parallel-size N`) if you have N smaller GPUs. Best when N ∈ {2, 4, 8}; requires model dimension divisibility.

---

## 5. Self-Assessment Bank

**Q1 (MC):** vLLM's PagedAttention is analogous to which OS-level concept?
A) File system journaling
B) Virtual memory paging with a page table per process
C) TCP congestion control
D) CPU scheduling with priority queues

**Q2 (short):** Compare static batching and continuous batching. When would each's tail-latency behavior favor the other?

**Q3 (MC):** For serving Llama 3.1 8B in bf16 on a single 24 GB GPU, a reasonable `--gpu-memory-utilization` is:
A) 0.30
B) 0.85–0.92
C) 0.99
D) 1.00

**Q4 (short):** Explain what `--enable-prefix-caching` does and give a workload where it saves the most.

**Q5 (MC):** For serving a 70B fp16 model, the minimum hardware is roughly:
A) One 24 GB GPU
B) One 40 GB GPU
C) Two 80 GB GPUs (with `--tensor-parallel-size 2`)
D) 32 GB CPU RAM

**Q6 (short):** You want to serve two fine-tuned LoRA adapters against a single base model. Which vLLM flag enables this and what's the trade-off?

**Q7 (MC):** Setting `--max-model-len` far above your actual context usage:
A) Makes no difference
B) Wastes VRAM allocated for KV cache pool, reducing concurrent-request capacity
C) Speeds up inference
D) Improves quality

**Q8 (short):** Your vLLM instance works for 5 minutes and then OOMs on a long request. Most likely cause and fix?

**Q9 (MC):** Which of these is NOT typically an advantage of vLLM over a naive HuggingFace transformers pipeline?
A) Higher throughput
B) Better memory utilization
C) OpenAI-compatible API
D) Better model quality

**Q10 (short):** Describe how you'd deploy vLLM to production with health checks, metrics, and secrets — enough for a 60-second interview answer.

---

### Answer Key

**A1: B.** PagedAttention treats the KV cache as fixed-size blocks addressed through per-sequence block tables, exactly analogous to OS virtual memory pages with per-process page tables. Enables near-zero fragmentation and lets multiple sequences share KV blocks (used by the prefix cache).

**A2:** **Static batching** — batches requests together and processes them until the *longest* sequence finishes. Simpler to implement, but latency for short requests is dominated by the slowest one in the batch. Static wins when all requests have similar length (e.g., a uniform classification workload) — the tail-latency penalty vanishes if the tail *is* the median.

**Continuous batching** — scheduled at the token level; new requests join and finished ones leave every forward pass. Latency for a request depends on its own length, not the batch's; throughput scales sub-linearly with concurrency. Continuous wins for **mixed-length workloads** (real chat traffic) where tail latency of static would be very bad. Continuous can lose on a burst of identical short requests where the overhead of dynamic scheduling exceeds the batching benefit — rare in practice.

**A3: B.** 0.85-0.92 is the safe range. 0.30 wastes VRAM. 0.99 leaves no headroom for activations under load — will OOM on real traffic. 1.00 will OOM immediately. Default is 0.90 for a reason.

**A4:** `--enable-prefix-caching` stores the KV state of computed prompt prefixes and reuses them for later requests with matching byte-prefixes. Saves the most on **workloads with a shared system prompt across many requests** — a common example: a customer support chat that reuses a 4-6k token system prompt on every request. Cache hit → the model skips computing that prefix (5x faster TTFT, higher throughput). Also helpful for RAG when the retrieved context is often similar across a session (though semantic prefix matching, not just byte-exact, is not yet standard).

**A5: C.** 70B fp16 model weights = ~140 GB. Doesn't fit on a single 40 GB or 80 GB GPU. Needs either two 80 GB (2 × H100 80GB, or 2 × A100 80GB) with `--tensor-parallel-size 2` shard weights across them, or four 40 GB (2×A100 40GB with tp=2 fits weights at ~35 GB each but no KV cache room; 4×A100 40GB works comfortably with tp=4). Alternative: aggressive quantization (AWQ int4 → ~35 GB), then a single 80 GB card holds weights + KV.

**A6:** `--enable-lora`, combined with `--lora-modules name1=/path1 name2=/path2` to declare adapters. Each request specifies the adapter via the `model` field. Trade-off: ~2-5% latency overhead per request, plus additional VRAM for each adapter (small — 50-200 MB each). Advantages: one base model in memory serves many fine-tunes; A/B test cheaply; multi-tenant model-per-customer serving. For a single production fine-tune with max throughput, prefer merging and serving as the base model directly.

**A7: B.** `--max-model-len` bounds the KV cache pool size at startup. Setting it to 128k when your traffic uses 4k means the KV pool is 32× larger than needed, drastically reducing how many concurrent sequences fit in VRAM. Fix: set it to your realistic max (`prompt_length + max_new_tokens` p99 + margin). Can dramatically improve throughput with no user-facing downside for typical chat workloads.

**A8:** Most likely: **`--gpu-memory-utilization` set too aggressively.** vLLM pre-allocated a large KV cache pool at startup; smaller requests worked fine, but a long generation combined with active batch demand pushed activation memory beyond what remained. Fix: reduce `--gpu-memory-utilization` (e.g., 0.90 → 0.85). Also consider reducing `--max-model-len` if it's set far above realistic p99 usage. Related cause: `--max-num-batched-tokens` and `--max-num-seqs` — if these are too high, vLLM tries to pack too much into one forward pass. Monitor `vllm_gpu_cache_usage_perc` and `vllm_num_requests_running`.

**A9: D.** Model quality is unchanged — vLLM produces bit-identical outputs to the reference implementation (modulo floating-point non-determinism from batched vs unbatched execution, which is minor). Throughput, memory, and API compatibility are all wins. Quality is not.

**A10:** Deploy the official Docker image (`vllm/vllm-openai`) on a GPU-enabled cloud host (RunPod / Modal / Fly.io GPU / GCP GPU / EC2 g5). Mount a persistent volume for the HF model cache to skip re-downloads on restart. Env vars: `HF_TOKEN` and a `VLLM_API_KEY` for bearer auth. Bind `--host 0.0.0.0`, expose the port behind TLS. Health check: `GET /health`. Metrics: scrape `/metrics` with Prometheus; alert on `vllm_num_requests_running` climbing (queue saturation) or `vllm_time_to_first_token_seconds` p99 climbing. Front with a small FastAPI proxy that adds auth, rate limiting, and OpenAI-compatible passthrough (or use vLLM's built-in auth directly). For rollback, keep the previous model version's container image ready and use blue-green swap at the load balancer.

---

**Related modules:**
- `learning/02-quantization.md` — reducing VRAM further with 8-bit or 4-bit models
- `learning/03-openai-compatible-api.md` — the API layer on top of vLLM
- `builds/01-self-host-llama-api.md` — hands-on end-to-end
- `../fine-tuning/learning/05-fine-tuning-ops.md` — LoRA adapters this module serves

**Practice prompts:**
1. Launch vLLM with Llama 3.1 8B Instruct on the smallest GPU you can afford. Benchmark tokens/sec at concurrency 1, 4, 16, and 64. Plot the curve.
2. Turn `--enable-prefix-caching` on and off with an identical workload featuring a 4k-token shared system prompt. Compare TTFT.
3. Serve two LoRA adapters (real or dummy) via `--enable-lora`. Verify each returns its expected behavior.
4. Set `--max-model-len` to 128000, run a benchmark; set to 8192, run again. Compare throughput.

**References:**
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — the vLLM paper
- vLLM docs — https://docs.vllm.ai
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (2022) — origin of iteration-level scheduling
- Nvidia H100 / A100 specs; RunPod / Modal pricing pages
- Together AI, Fireworks, Anyscale docs (managed vLLM)
