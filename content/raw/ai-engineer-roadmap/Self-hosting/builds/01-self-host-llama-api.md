# Self-Hosting Build 01 — Self-Host Llama 3.1 8B Behind an OpenAI-Compatible API

> **Deliverable:** A production-ish deployment of Llama 3.1 8B Instruct (or a fine-tuned variant), served via vLLM with prefix caching, quantized appropriately for the target GPU, behind a FastAPI wrapper exposing a clean OpenAI-compatible API — plus a router layer that lets your application switch between this self-hosted endpoint and Anthropic with a single config change. Deployed to a rented cloud GPU, with metrics, health checks, and a cost model showing the break-even point.
>
> **Time:** 8-12 hours end-to-end. Longest part is getting vLLM launched cleanly on the chosen platform.
>
> **What you'll be able to say:** "I self-hosted Llama 3.1 8B behind an OpenAI-compatible API on a rented A10G. Handles 60 concurrent users, ~1200 tokens/sec throughput, TTFT p95 under 400ms with prefix caching on. Cost per million output tokens is $1.60 vs $15 for Sonnet — so for our high-volume classification workload the break-even was ~30M tokens/month, and we're doing 200M."

---

## 1. Project Overview

Where this fits: you have (or expect to have) enough LLM traffic that the Anthropic/OpenAI bill is uncomfortable. You have (or can rent) a GPU. You want to run an open model that's *good enough* on your specific task at a fraction of the cost.

### Architecture

`[IMG-SH-BUILD-01]` — *Prompt: A production architecture for a self-hosted LLM API. Left: application services (your FastAPI apps, agents, RAG services) → sending requests to a "provider-router" abstraction layer. From the router, two paths: (1) top path — Anthropic API cloud icon for high-quality/complex requests; (2) bottom path — a "self-hosted vLLM" node (rented A10G / L4 GPU) with Llama 3.1 8B AWQ int4 quantization, prefix caching enabled, OpenAI-compatible API. In front of the vLLM node: a small FastAPI wrapper adding auth, rate limits, and observability. Prometheus scraping metrics from vLLM. Alertmanager. Grafana dashboard visible in the corner showing tokens/sec, TTFT, cost/hour. Config file on left showing `provider: "vllm-selfhost"` toggle. Clean modern MLOps diagram in muted blue/orange palette.*

### Prerequisites

- **GPU access**: RunPod / Modal / Vast.ai / Lambda / cloud vendor. Pick a card with ≥20 GB VRAM (A10G, L4, A100 40GB, RTX 4090).
- **HuggingFace account** with Llama 3.1 access granted (meta-llama/Llama-3.1-8B-Instruct is a gated model — accept the license on HF first).
- **Docker Hub / GHCR** for image storage (optional; can build from source on the GPU host).
- **Python 3.11+** locally for testing.
- **~$5-30 of GPU credits** for the build.

### Repo layout

```
selfhost-llm/
├── pyproject.toml
├── README.md
├── docker/
│   └── Dockerfile               # vLLM + FastAPI wrapper
├── vllm/
│   └── launch.sh                # vLLM launch command
├── proxy/
│   ├── app.py                   # FastAPI wrapper adding auth/rate-limit/metrics
│   ├── metrics.py
│   ├── auth.py
│   └── router.py                # Provider-router logic
├── benchmarks/
│   ├── benchmark_throughput.py  # tokens/sec at various concurrency levels
│   ├── benchmark_latency.py     # TTFT and end-to-end latency
│   └── benchmark_quality.py     # small MMLU sample
├── configs/
│   └── prod.yaml
└── deploy/
    ├── runpod-launch.sh
    └── modal-app.py
```

---

## 2. Step-by-Step Milestones

### Milestone 1 — Local proof of concept (2 hours)

Get vLLM working locally with a small model to build muscle memory. Use Qwen2.5-0.5B or Phi-3-mini if your local machine has no GPU (they run acceptably on CPU for testing).

```bash
# On a GPU box (or CPU with a small model)
pip install vllm

# Launch server
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-mini-4k-instruct \
    --port 8000 \
    --dtype auto

# Test
curl http://localhost:8000/v1/chat/completions \
    -H "content-type: application/json" \
    -d '{
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "messages": [{"role":"user","content":"say hi"}]
    }'
```

**Exit criterion:** you can hit the API and get a completion. Metrics endpoint at `/metrics` returns Prometheus text.

### Milestone 2 — Launch Llama 3.1 8B on rented GPU (2 hours)

Choose a platform. **RunPod** is a good default for its simplicity:

1. Create a RunPod account; add ~$20 credit.
2. Launch a pod: template "PyTorch 2.4", GPU "1× RTX A5000 24GB" or "1× A10G 24GB" ($0.40-0.50/hr), 50 GB volume.
3. SSH in; verify GPU with `nvidia-smi`.
4. Install and launch vLLM:

```bash
# On the RunPod pod
export HF_TOKEN="hf_xxxx"  # your HF token with Llama 3.1 access
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --api-key $VLLM_API_KEY
```

5. Expose the port. RunPod: use "TCP" exposed port and note the public URL. Modal: `modal.web_endpoint`. Fly.io: fly proxy or public IPv6.
6. From your laptop:

```bash
curl https://<runpod-public-url>:8000/v1/chat/completions \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    -H "content-type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role":"user","content":"What is 2+2?"}]
    }'
```

**Exit criterion:** you get a response from your Llama 3.1 running on a rented GPU. Latency is under 3 seconds for a small request.

### Milestone 3 — Quantize for tighter VRAM / higher throughput (1 hour)

Optional but instructive. Serve an AWQ int4 version:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-3.1-8B-Instruct-AWQ \
    --quantization awq_marlin \
    --host 0.0.0.0 --port 8001 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --api-key $VLLM_API_KEY
```

Notes:
- Find a pre-quantized AWQ version on HuggingFace by searching "Llama-3.1-8B-Instruct-AWQ" (multiple options; check download counts and update dates).
- `--quantization awq_marlin` uses the fastest AWQ kernel path (requires Ampere+ GPUs).
- Verify quality on 20 sample prompts vs bf16 baseline — should be within 1-2%.

**Exit criterion:** AWQ version runs; VRAM usage roughly halved; throughput similar or higher; quality within 1-2% on your sample prompts.

### Milestone 4 — Build the FastAPI wrapper with auth, rate-limit, metrics (2-3 hours)

vLLM's built-in server is fine for internal use. For anything exposed, wrap it in a thin FastAPI proxy that adds:
- Bearer-token auth against your own key store (not just the shared vLLM API key)
- Rate limits per API key (RPM + TPM)
- Request logging and Prometheus metrics
- Cost accounting (Module 04 in Capstone 04 style)

```python
# proxy/app.py
import time, os, redis, httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from prometheus_client import make_asgi_app, Counter, Histogram

app = FastAPI(title="LLM Proxy")
VLLM_URL = os.environ["VLLM_UPSTREAM_URL"]     # http://localhost:8000
VLLM_API_KEY = os.environ["VLLM_API_KEY"]
r = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"))

REQUESTS = Counter("proxy_requests_total", "Proxy requests", ["endpoint", "status"])
LATENCY = Histogram("proxy_request_duration_seconds", "Request duration", ["endpoint"])
TOKENS_IN = Counter("proxy_tokens_in_total", "Input tokens", ["key_id"])
TOKENS_OUT = Counter("proxy_tokens_out_total", "Output tokens", ["key_id"])

async def check_auth(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    presented = auth[7:]
    key_id = r.get(f"key:{presented}")
    if not key_id:
        raise HTTPException(401, "invalid token")
    return key_id.decode()

async def rate_limit(key_id: str, rpm: int = 60):
    now = time.time()
    key = f"rl:{key_id}:{int(now//60)}"
    n = r.incr(key)
    r.expire(key, 120)
    if n > rpm:
        raise HTTPException(429, headers={"Retry-After": "60"})

@app.post("/v1/chat/completions")
async def chat(request: Request, key_id: str = Depends(check_auth)):
    await rate_limit(key_id)
    body = await request.body()
    is_stream = b'"stream":true' in body or b'"stream": true' in body

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=300) as client:
        if is_stream:
            # Proxy streaming
            async def gen():
                async with client.stream(
                    "POST", f"{VLLM_URL}/v1/chat/completions",
                    headers={"authorization": f"Bearer {VLLM_API_KEY}", "content-type": "application/json"},
                    content=body,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            LATENCY.labels(endpoint="/v1/chat/completions").observe(time.perf_counter() - start)
            REQUESTS.labels(endpoint="/v1/chat/completions", status=200).inc()
            return StreamingResponse(gen(), media_type="text/event-stream")

        # Non-streaming
        r_up = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            headers={"authorization": f"Bearer {VLLM_API_KEY}", "content-type": "application/json"},
            content=body,
        )
        LATENCY.labels(endpoint="/v1/chat/completions").observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint="/v1/chat/completions", status=r_up.status_code).inc()
        payload = r_up.json()
        usage = payload.get("usage", {})
        TOKENS_IN.labels(key_id=key_id).inc(usage.get("prompt_tokens", 0))
        TOKENS_OUT.labels(key_id=key_id).inc(usage.get("completion_tokens", 0))
        return JSONResponse(payload, status_code=r_up.status_code)

@app.get("/healthz")
async def health():
    async with httpx.AsyncClient(timeout=2) as client:
        try:
            r_up = await client.get(f"{VLLM_URL}/health")
            return {"status": "ok" if r_up.status_code == 200 else "degraded"}
        except Exception:
            return {"status": "down"}

app.mount("/metrics", make_asgi_app())
```

**Exit criterion:** requests to your proxy work; auth is enforced; rate limits fire correctly; metrics show up at `/metrics`.

### Milestone 5 — Benchmarking (2 hours)

Measure throughput, latency, and cost.

```python
# benchmarks/benchmark_throughput.py
import asyncio, time, httpx, statistics

BASE_URL = "http://localhost:8080/v1"  # your proxy
API_KEY = "your-key"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

async def one_request(client, prompt: str):
    start = time.perf_counter()
    r = await client.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": [{"role":"user","content":prompt}], "max_tokens": 200},
    )
    elapsed = time.perf_counter() - start
    data = r.json()
    return elapsed, data.get("usage", {}).get("completion_tokens", 0)

async def concurrent_benchmark(concurrency: int, n_requests: int):
    prompts = [f"Write a paragraph about topic {i}." for i in range(n_requests)]
    async with httpx.AsyncClient(timeout=300) as client:
        sem = asyncio.Semaphore(concurrency)
        async def guarded(prompt):
            async with sem:
                return await one_request(client, prompt)
        start = time.perf_counter()
        results = await asyncio.gather(*[guarded(p) for p in prompts])
        wall = time.perf_counter() - start
    total_out = sum(r[1] for r in results)
    latencies = [r[0] for r in results]
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "wall_sec": wall,
        "tokens_out": total_out,
        "tokens_per_sec": total_out / wall,
        "p50_latency": statistics.median(latencies),
        "p95_latency": sorted(latencies)[int(0.95 * len(latencies))],
    }

async def main():
    for c in [1, 4, 16, 32, 64]:
        r = await concurrent_benchmark(c, n_requests=c * 4)
        print(f"c={c:3}: {r['tokens_per_sec']:.0f} tok/s, "
              f"p50={r['p50_latency']:.2f}s, p95={r['p95_latency']:.2f}s")

asyncio.run(main())
```

**Expected results** (Llama 3.1 8B AWQ int4, A10G 24GB):

| Concurrency | Tokens/sec | p50 latency | p95 latency |
|-------------|-----------|-------------|-------------|
| 1  | ~70   | 2.5s | 3.0s |
| 4  | ~250  | 2.5s | 3.5s |
| 16 | ~800  | 3.0s | 5.0s |
| 32 | ~1200 | 4.0s | 8.0s |
| 64 | ~1400 | 6.0s | 12.0s |

Throughput scales sub-linearly; latency degrades with load. The concurrency sweet spot is usually the point just before p95 climbs sharply (here: ~32).

### Milestone 6 — Cost model + break-even analysis (1 hour)

The economics story:

```
Cost inputs:
- GPU rental: A10G ~$0.50/hr on RunPod, ~$0.60-0.80 on cloud vendors
- Llama 3.1 8B AWQ throughput: ~1000-1500 output tokens/sec at reasonable concurrency
- Assume 30% average utilization over a month
- Hours per month: 730

Effective tokens/sec: 1200 * 0.30 = 360 tokens/sec average
Monthly output tokens: 360 * 60 * 60 * 730 = ~945 million

Monthly GPU cost: $0.50 * 730 = $365
Cost per million output tokens (self-hosted): $365 / 945 = $0.39

Reference: Anthropic Sonnet current price (approximate)
- Input: $3 / M
- Output: $15 / M
- Cache reads: ~$0.30 / M

Break-even for a typical workload (say 500 input, 300 output tokens per request):
- Sonnet: (500 * $3 + 300 * $15) / 1M = $0.006 per request
- Self-hosted: (500 * $0.05 + 300 * $0.39) / 1M = $0.00014 per request  (roughly)
- Sonnet is ~40× more expensive per request

But you pay for GPU whether utilized or not.
- Below ~5-10M output tokens/month, Sonnet's per-token pricing wins
- Above that, self-hosted wins by a growing margin

Rule of thumb break-even: ~30-100M tokens/month for an 8B on a single card.
For 70B models: ~200M-1B tokens/month depending on hardware.
```

Real practice: cost this against **your actual workload** — token mix, average utilization, hardware choice. If self-hosted is cheaper *on paper* but your utilization is 3% because your traffic is bursty, Sonnet's per-request pricing wins.

### Milestone 7 — Provider router (2-3 hours)

Build the abstraction that lets you swap providers with config (Module 03 pattern):

```python
# proxy/router.py
from openai import AsyncOpenAI
import os

class LLMProvider:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "vllm-selfhost":
            self.client = AsyncOpenAI(
                base_url=os.environ["VLLM_URL"],
                api_key=os.environ["VLLM_API_KEY"],
            )
            self.default_model = "meta-llama/Llama-3.1-8B-Instruct"
        elif provider == "anthropic-compat":
            self.client = AsyncOpenAI(
                base_url="https://api.anthropic.com/v1",
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            self.default_model = "claude-sonnet-5"
        elif provider == "openai":
            self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.default_model = "gpt-4o-mini"
        else:
            raise ValueError(provider)

    async def chat(self, messages, **kwargs):
        return await self.client.chat.completions.create(
            model=kwargs.pop("model", None) or self.default_model,
            messages=messages,
            **kwargs,
        )

# Application usage:
provider = LLMProvider(os.environ.get("LLM_PROVIDER", "anthropic-compat"))
resp = await provider.chat([{"role":"user","content":"..."}], max_tokens=200)
```

Set `LLM_PROVIDER=vllm-selfhost` in a config → identical business logic routes to your self-hosted Llama. Change to `anthropic-compat` → back to Sonnet. Test in canary before flipping in prod.

### Milestone 8 — Docker + deploy (1-2 hours)

```dockerfile
# docker/Dockerfile — the FastAPI proxy
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev
COPY proxy/ ./proxy/
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080
CMD ["uvicorn", "proxy.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Deployment options:
- **RunPod**: run vLLM in one pod, proxy in the same pod on a different port; expose proxy publicly
- **Modal**: `modal.web_endpoint` for the proxy; `modal.gpu.T4` (or A10G) function for vLLM
- **Fly.io GPU**: `fly deploy` — Fly's new GPU offering supports A10 and L40S
- **Cloud vendor**: EC2 g5.xlarge or GCP L4 instance with a systemd service for vLLM + one for the proxy

Whatever platform: **health checks** wired up, **secrets** in a manager, **metrics** scraped by Prometheus, **logs** structured JSON.

### Milestone 9 — Failover strategy (1-2 hours)

Since you now have two providers behind one interface, wire failover:

```python
# router.py — with failover
class LLMProvider:
    def __init__(self, primary: str, fallback: str = None):
        self.primary = self._make(primary)
        self.fallback = self._make(fallback) if fallback else None

    async def chat(self, messages, **kwargs):
        try:
            return await asyncio.wait_for(
                self.primary.chat.completions.create(messages=messages, **kwargs),
                timeout=30,
            )
        except (asyncio.TimeoutError, Exception) as e:
            log.warning("primary_failed", error=str(e))
            if self.fallback:
                return await self.fallback.chat.completions.create(messages=messages, **kwargs)
            raise
```

Set primary = self-hosted, fallback = Anthropic. If your vLLM has an outage, requests transparently roll over. Add a circuit breaker on top so you don't hammer a broken primary for every request.

### Milestone 10 — Monitoring & runbook (1 hour)

Standard production observability (Month 6 Module 09 pattern) plus specifically for the self-host:

Metrics to watch:
- `vllm_num_requests_running` — active batch size
- `vllm_gpu_cache_usage_perc` — KV cache pressure
- `vllm_time_to_first_token_seconds` (p95, p99)
- `vllm_e2e_request_latency_seconds` (p95, p99)
- `vllm_prompt_tokens_total`, `vllm_generation_tokens_total`
- `proxy_tokens_in_total`, `proxy_tokens_out_total` by key_id
- Cost/hour derived from tokens (input × input_price + output × output_price where you compute your own effective price)
- GPU utilization, VRAM used, temperature (via DCGM exporter)

Alerts:
- GPU utilization sustained > 95% (need more capacity)
- KV cache usage sustained > 90% (may OOM soon)
- Failover rate > 5% (primary is unstable)
- p95 TTFT > 2s (users perceive lag)

Runbook entries:
- "vLLM pod restarted" → check GPU health with `nvidia-smi`, review model download progress, verify Hugging Face access
- "OOM during long generation" → reduce `--max-model-len` or `--gpu-memory-utilization`
- "Sudden throughput drop" → check prefix cache hit rate; a system prompt change invalidates the cache
- "Failover firing constantly" → primary is degraded, dig into vLLM logs

---

## 3. Expected Outcomes

| Metric | Value |
|--------|-------|
| Throughput at c=32 | ~1200 output tokens/sec |
| TTFT p95 (prefix cache hit) | ~400ms |
| TTFT p95 (cold, 4k system prompt) | ~1.5s |
| End-to-end p95 | ~5-8s for 300 output tokens |
| Cost per M output tokens (30% util) | ~$1-2 |
| Cost per M output tokens (Sonnet) | $15 |
| Break-even monthly token volume | ~30M output tokens |
| VRAM used (AWQ int4) | ~7 GB (fits comfortably on A10G 24GB) |
| Adapter swap latency | ~500ms (if `--enable-lora`) |

---

## 4. Extensions

- **LoRA multi-tenancy.** Serve N fine-tuned adapters against 1 base with `--enable-lora`. Route requests to the right adapter based on `X-Tenant-ID` header.
- **Speculative decoding.** vLLM supports draft-model speculative decoding for 2-3× throughput. Use a smaller "draft" model (Llama 3.2 1B) to propose tokens, verified by the 8B.
- **Multi-node deployment.** For 70B models, tensor-parallel across 2 A100 40GBs.
- **Cost dashboard.** Build a small Streamlit/Grafana page showing $/hour, break-even projection, and split of self-hosted vs Anthropic traffic.
- **Auto-scaling.** Scale GPU nodes up/down based on `vllm_num_requests_waiting`. RunPod, Modal, and cloud vendors all support this.
- **Migrate a fraction of traffic.** In production, route 10% of easy-classification requests to self-host, keep 90% on Sonnet. Measure quality delta, cost delta, latency delta. Increase self-host share if the trade is favorable.

---

## 5. Deliverables Checklist

- [ ] vLLM launches cleanly on the target GPU; API reachable
- [ ] AWQ quantized version tested; quality within 2% of bf16 on sample prompts
- [ ] FastAPI proxy with auth, rate-limiting, metrics, tokens accounting
- [ ] Throughput benchmark results committed to repo (`benchmarks/results.md`)
- [ ] Cost model spreadsheet or notebook (`analysis/cost-model.ipynb`)
- [ ] Provider router that swaps between self-hosted and Anthropic via config
- [ ] Failover logic tested (kill vLLM, verify Anthropic takes over)
- [ ] Dockerfile + deployment scripts for the platform of choice
- [ ] Monitoring dashboard (Grafana) with the panels listed in Milestone 10
- [ ] Runbook with 4-6 alert entries
- [ ] README explaining the whole system and the break-even analysis
- [ ] Blog post version (portfolio-optional but very high value)

---

## 6. Interview Talking Points

- **"Why self-host?"** Cost. At our scale (~200M output tokens/month), self-hosted 8B on an A10G runs ~$0.50-2/M output tokens vs Sonnet's $15/M. Break-even was ~30M tokens/month; we're well past it. Also latency (self-host TTFT is faster on short prompts) and control (updates, rollbacks, no external outages).
- **"Why Llama 3.1 8B?"** Task complexity fits. On our classification workload we lose ~2% accuracy vs Sonnet but gain 30× cost efficiency. For harder queries we route to Anthropic via the provider router.
- **"Why vLLM?"** PagedAttention + continuous batching = 5-20× throughput vs naive HF pipeline. Prefix caching cuts our shared-system-prompt cost by ~80%. Widely supported model coverage. Emerging OSS ecosystem around it.
- **"How does failover work?"** Primary is self-hosted; on timeout or error, we fall back to Anthropic. Circuit breaker prevents hammering a broken primary. p99 error rate has been < 0.1% since we shipped.
- **"What breaks?"** GPU OOM if `--gpu-memory-utilization` too aggressive; runaway concurrent requests if we don't rate-limit; prefix cache misses tank TTFT when a system prompt changes and everyone re-computes it. All monitored + alerted.
- **"How do you evaluate quality?"** Same golden set + LLM-judge pipeline (Month 6 Module 08). Every candidate model (bf16 vs AWQ, 8B vs 70B, different fine-tunes) goes through the same battery before hitting prod.
- **"What would you do next?"** Try speculative decoding for another 2-3× throughput; add a 70B AWQ option on demand for the top 5% hardest queries; multi-tenant LoRA serving for per-customer fine-tunes.

---

## 7. References

- vLLM docs — https://docs.vllm.ai
- Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (2023)
- Runpod / Modal / Lambda / Fly.io GPU pricing pages
- llama.cpp for CPU/edge alternative
- LiteLLM — a production provider router (alternative to hand-rolled) — https://docs.litellm.ai
- Anthropic OpenAI-compatible endpoint docs
- TheBloke, hugging-quants, LoneStriker, QuantFactory — pre-quantized model repositories
