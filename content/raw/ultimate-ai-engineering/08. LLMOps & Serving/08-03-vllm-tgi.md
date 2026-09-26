# 08.03 — vLLM / TGI (Self-Hosted Inference Engines in Production)

> **Module 8: LLMOps & Serving** · Subtopic 3 of 5
> **Prerequisites:** 01.01 §3.2 (KV-cache sizing), **01.02 (required: prefill/decode, PagedAttention, continuous batching, chunked prefill, speculative decoding, parallelism)**, 08.01–08.02, Kubernetes, Prometheus.
> **Outcome:** you can deploy, size, tune, observe, autoscale, and upgrade a production inference engine on Kubernetes; migrate off Hugging Face TGI; and decide between vLLM, SGLang, TensorRT-LLM, and local engines with measured evidence.

> **Status check (September 2026) — read first.**
> - **Hugging Face TGI is in maintenance mode.** The TGI docs state: *"text-generation-inference is now in maintenance mode. Going forward, we will accept pull requests for minor bug fixes, documentation improvements and lightweight maintenance tasks."* Hugging Face recommends **vLLM** and **SGLang** (plus llama.cpp and MLX for local use). This subtopic therefore covers TGI mainly as a **migration source**.
> - **vLLM** moves fast. Its latest release at the time of writing is **v0.30.0 (Sep 22, 2026)**, with a new model-runner generation now the default. Flags, defaults, and metric names change between releases, so **pin versions and read the release notes** before upgrading. The flags below are long-standing ones; verify them with `vllm serve --help` on your pinned version.

---

## 1. Engine Landscape

| Engine | Strengths | Consider when |
|---|---|---|
| **vLLM** | Broadest model and hardware support, PagedAttention, continuous batching, prefix caching, speculative decoding, structured outputs, multi-LoRA, OpenAI-compatible server, large ecosystem (production-stack, llm-d, KServe) | The default choice for most self-hosted serving |
| **SGLang** | RadixAttention prefix reuse, fast structured generation, strong multi-turn and agentic throughput | Heavy prefix sharing, structured outputs, agent workloads |
| **TensorRT-LLM** (+ Triton / NVIDIA NIM) | Peak NVIDIA performance, FP8/FP4 kernels, in-flight batching | Maximum throughput on NVIDIA, with engine-build workflows |
| **llama.cpp / Ollama / MLX** | CPU, Apple silicon, edge, GGUF quantization | Local, edge, and developer machines |
| **TGI** | Mature HF integration (historical) | **Maintenance mode** — migrate to vLLM or SGLang |

**Decide with a bake-off on your workload** (Project 3 of 01.02; §5 below). Rankings flip with model architecture, prompt/output lengths, prefix sharing, quantization, and GPU generation.

---

## 2. vLLM Architecture (Operator's View)

```
 clients ─► OpenAI-compatible API server (FastAPI; /v1/chat/completions, /v1/completions, /v1/embeddings, /health, /metrics)
              │  tokenisation, chat templates, tool/reasoning parsers, structured-output grammar compilation
              ▼
          ENGINE CORE (separate process) ── scheduler: continuous batching, chunked prefill, token budget,
              │                              preemption, priorities; KV cache manager: paged blocks,
              │                              prefix-cache hashing, optional offload tiers
              ▼
          WORKERS / model runners (one per GPU; TP/PP/DP/EP groups) ── CUDA graphs, attention backends,
                                                                        quantized kernels, spec-decode drafts
```

**Key operational facts:**
- **Startup is slow:** weight download, loading, CUDA-graph capture, and grammar and kernel warm-up take minutes for large models. Pre-cache the weights (a PVC or a node-local cache) and set generous startup probes.
- **Memory is pre-allocated:** `--gpu-memory-utilization` reserves a fraction of the GPU. What remains after weights and activations becomes the **KV-cache pool**, which determines concurrency.
- **The OpenAI-compatible API** lets your gateway (08.01) treat self-hosted models like any provider.

---

## 3. Sizing: From GPU Memory to Concurrency

$$
M_{\text{KV}} \approx u \cdot M_{\text{GPU}} \cdot N_{\text{TP}} - M_{\text{weights}} - M_{\text{activations+graphs}},
\qquad
\text{KV tokens} = \frac{M_{\text{KV}}}{2 \cdot L \cdot n_{kv} \cdot d_h \cdot b_{\text{kv}}}
$$

$$
\text{max concurrent sequences} \approx \frac{\text{KV tokens}}{\bar T_{\text{context}}}
$$

Here $u$ is `--gpu-memory-utilization` and $b_{\text{kv}}$ the bytes per KV element (2 for bf16, 1 for an FP8 KV cache). The per-token KV formula comes from 01.01 §3.2.

```python
import math


def vllm_capacity(gpu_mem_gb: float, tp: int, util: float, weights_gb: float, layers: int, kv_heads: int,
                  head_dim: int, kv_bytes: int = 2, activation_overhead_gb_per_gpu: float = 2.0,
                  block_size: int = 16, avg_context_tokens: int = 4000) -> dict:
    usable = util * gpu_mem_gb * tp - weights_gb - activation_overhead_gb_per_gpu * tp
    if usable <= 0:
        return {"fits": False, "reason": "weights + overhead exceed the memory budget"}
    per_token = 2 * layers * kv_heads * head_dim * kv_bytes
    kv_tokens = int(usable * 1e9 // per_token)
    return {"fits": True, "kv_cache_gb": round(usable, 1), "kv_bytes_per_token": per_token,
            "kv_tokens": kv_tokens, "kv_blocks": kv_tokens // block_size,
            "max_concurrent_at_avg_ctx": kv_tokens // avg_context_tokens}
```

*Example:* a 70B-class GQA model (L = 80, 8 KV heads, d_h = 128) in FP8 weights (~70 GB) on 2 × 80 GB GPUs at u = 0.9 leaves ≈ 70 GB of KV after overhead. That is ≈ 214k tokens in bf16 KV, or ≈ 428k with an FP8 KV cache — about 53 vs 107 concurrent 4k-token conversations. **KV dtype and GQA dominate concurrency.**

---

## 4. Configuration Reference (long-standing flags; verify on your version)

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --served-model-name legis-70b \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-lora --max-loras 4 --max-lora-rank 32 \
  --lora-modules fiscal=/models/lora/fiscal drafting=/models/lora/drafting \
  --api-key "$VLLM_API_KEY"
```

| Flag | Controls | Tuning guidance |
|---|---|---|
| `--tensor-parallel-size` / `--pipeline-parallel-size` / `--data-parallel-size` | Parallelism (01.02 §8) | Smallest TP that fits weights + enough KV; scale out with replicas or DP |
| `--max-model-len` | Maximum context | Don't set higher than needed: it bounds the worst-case KV per sequence |
| `--gpu-memory-utilization` | Pre-allocated GPU fraction | 0.85–0.95; leave headroom for spikes and other processes |
| `--max-num-seqs` | Maximum concurrent sequences | Higher → throughput ↑, ITL ↑; tune against the SLO |
| `--max-num-batched-tokens` | Per-step token budget (chunked prefill) | Lower → smoother ITL, slower TTFT for long prompts; higher → the reverse |
| `--enable-prefix-caching` | KV reuse across requests (a default in recent versions) | Essential for chat and agents; keep prompt prefixes stable (02.02 §6.2) |
| `--kv-cache-dtype fp8` | KV precision | ~2× concurrency; validate quality on long-context evals (08.04) |
| `--quantization` | Weight quantization method (often auto-detected from the checkpoint) | Use pre-quantized checkpoints (08.04) |
| `--speculative-config '{...}'` | Speculative decoding (draft model, n-gram, EAGLE-style, MTP) | Helps at low batch; measure at your production concurrency (01.02 §5) |
| `--enable-lora`, `--max-loras`, `--lora-modules` | Multi-adapter serving | Many adapters on one base model; watch the adapter-swap overhead |
| Structured-output backend options | Grammar engine (XGrammar/llguidance/…) | Pre-warm common schemas (02.04 §7) |

Recent releases add **admission-control** options (e.g. caps on queued requests or tokens) and other scale-out features. Check the release notes for the version you pin.

---

## 5. Benchmarking and Tuning Workflow

1. **Workload model:** the prompt- and output-length distributions from production traces (06.04), the prefix-sharing ratio, and the arrival pattern (Poisson + peaks).
2. **Benchmark tools:** vLLM's built-in serving benchmark (`vllm bench serve` in recent versions), or your 01.02 §9 load tester against the OpenAI endpoint.
3. **Find the max QPS at the SLO:** binary-search the offered load for each configuration.
4. **Sweep:** `max-num-seqs`, `max-num-batched-tokens`, KV dtype, speculative decoding on/off, TP degree, and quantized vs bf16 weights.
5. **Quality gate:** every config change affecting numerics (quantization, KV dtype, spec-decode variants) passes the eval suite (06.01).

```python
def max_qps_at_slo(measure, slo: dict, lo: float = 0.1, hi: float = 100.0, iters: int = 12) -> dict:
    """measure(qps) -> {'p99_ttft_s', 'p99_itl_s', 'error_rate'} from a load-test run. Binary search the offered load."""
    def ok(m):
        return (m["p99_ttft_s"] <= slo["p99_ttft_s"] and m["p99_itl_s"] <= slo["p99_itl_s"]
                and m["error_rate"] <= slo.get("error_rate", 0.001))
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = measure(mid)
        if ok(m):
            best, lo = (mid, m), mid
        else:
            hi = mid
    return {"max_qps": round(best[0], 2) if best else 0.0, "at": best[1] if best else None}
```

---

## 6. Kubernetes Deployment

### 6.1 Deployment essentials

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: vllm-legis-70b, labels: { app: vllm, model: legis-70b } }
spec:
  replicas: 2
  selector: { matchLabels: { app: vllm, model: legis-70b } }
  template:
    metadata: { labels: { app: vllm, model: legis-70b } }
    spec:
      nodeSelector: { gpu.class: h100 }
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.30.0          # PIN the version
          args: ["meta-llama/Llama-3.3-70B-Instruct", "--served-model-name", "legis-70b",
                 "--tensor-parallel-size", "2", "--max-model-len", "32768",
                 "--gpu-memory-utilization", "0.90", "--kv-cache-dtype", "fp8"]
          env:
            - { name: HF_HOME, value: /models/hf }
            - { name: VLLM_API_KEY, valueFrom: { secretKeyRef: { name: vllm-secrets, key: api-key } } }
          ports: [{ containerPort: 8000 }]
          resources: { limits: { nvidia.com/gpu: "2" } }
          startupProbe:   { httpGet: { path: /health, port: 8000 }, periodSeconds: 10, failureThreshold: 90 }
          readinessProbe: { httpGet: { path: /health, port: 8000 }, periodSeconds: 5 }
          livenessProbe:  { httpGet: { path: /health, port: 8000 }, periodSeconds: 15, failureThreshold: 4 }
          volumeMounts:
            - { name: model-cache, mountPath: /models }
            - { name: shm, mountPath: /dev/shm }
      volumes:
        - { name: model-cache, persistentVolumeClaim: { claimName: model-cache-rwx } }
        - { name: shm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
```

### 6.2 Routing and autoscaling

- **Load balancing:** round-robin wastes prefix caches. Use **prefix- or session-aware routing** so that conversations and shared prefixes hit the replica holding their KV. vLLM's production-stack router, llm-d's inference scheduler (Gateway API Inference Extension), and KServe integrations all provide this.
- **Autoscale on the queue, not the CPU:** use KEDA or HPA on `vllm:num_requests_waiting` (or on KV-cache usage) via Prometheus. GPU pods take minutes to become ready, so keep a **warm minimum** and scale ahead of known peaks (session calendars; 08.05).
- **Disaggregated prefill/decode** (01.02 §3.4) and KV offload tiers are available through llm-d, production-stack, and vLLM's KV connectors. Adopt them when long prompts cause ITL interference at scale.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: { name: vllm-legis-70b }
spec:
  scaleTargetRef: { name: vllm-legis-70b }
  minReplicaCount: 2
  maxReplicaCount: 8
  cooldownPeriod: 600
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        query: sum(vllm:num_requests_waiting{model_name="legis-70b"})
        threshold: "8"
```

### 6.3 Observability

vLLM exposes Prometheus metrics at `/metrics`. Common series include:
- running and waiting requests;
- KV-cache usage;
- prefix-cache queries and hits;
- the TTFT, inter-token-latency, and E2E histograms;
- prompt and generation token counters;
- preemptions.

**Metric names have changed between engine generations**, so build the dashboards from your pinned version's `/metrics` output. Alert on the waiting-queue depth, KV usage > 90% with preemptions, TTFT P95, and error rates. Correlate them with gateway traces (06.04).

```python
def parse_prom(text: str) -> dict[str, float]:
    """Minimal Prometheus text parser: sums samples per metric name (ignores labels)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, _, value = line.rpartition(" ")
        name = name_labels.split("{", 1)[0]
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


def engine_health(metrics: dict[str, float], waiting_key="vllm:num_requests_waiting",
                  kv_key="vllm:kv_cache_usage_perc", preempt_key="vllm:num_preemptions_total") -> list[str]:
    alerts = []
    if metrics.get(waiting_key, 0) > 8:
        alerts.append("queue building: scale out or shed load")
    if metrics.get(kv_key, 0) > 0.9:
        alerts.append("KV cache > 90%: expect preemptions; reduce max-num-seqs/max-model-len or add capacity")
    if metrics.get(preempt_key, 0) > 0:
        alerts.append("preemptions observed")
    return alerts
```

---

## 7. Migrating from TGI to vLLM

| TGI concept / flag | vLLM equivalent (approximate) | Notes |
|---|---|---|
| `--model-id` | positional model argument | Same HF model IDs |
| `--num-shard` | `--tensor-parallel-size` | |
| `--max-input-tokens` / `--max-total-tokens` | `--max-model-len` (total context) | vLLM bounds the total; validate input limits at the gateway |
| `--max-concurrent-requests` | `--max-num-seqs` (+ gateway rate limits) | Semantics differ: vLLM queues beyond the running set |
| `--max-batch-prefill-tokens` | `--max-num-batched-tokens` | Chunked-prefill token budget |
| `--quantize` | `--quantization` (or auto from the checkpoint) | Prefer pre-quantized checkpoints |
| Messages API (`/v1/chat/completions`) | OpenAI-compatible server | Check tool-call and reasoning parser settings per model family |
| Grammar / JSON guidance | Structured outputs (JSON Schema, regex, grammar) | Re-test schemas (02.03 Project 2) |
| `/metrics` | `/metrics` (different names) | Rebuild the dashboards and alerts |

**Migration plan:**
1. Inventory the models, flags, and client features in use (streaming, tools, grammars, LoRA).
2. Stand up vLLM in **shadow** (08.05), mirror the traffic, and diff the outputs and latencies.
3. Run the quality evals (06.01) and the structured-output conformance suite.
4. Canary through the gateway (08.01), then ramp.
5. Decommission TGI, keeping a rollback window.

---

## 8. Upgrades and Operations

- **Pin** the image tag (`vllm/vllm-openai:vX.Y.Z`), the model revision, and the tokenizer and chat template (01.03 §8).
- **Upgrade like a model change:** performance benchmarks + quality evals + structured-output tests + canary. Defaults change (schedulers, attention backends, prefix caching), so re-tune after upgrading.
- **Multi-LoRA hygiene:** version the adapters, and load and unload them via the API or configuration with evals per adapter.
- **Sleep and wake** (a supported feature) frees GPU memory between bursts or RL phases on shared clusters. Measure the wake latency before relying on it for serving.
- **Security:** API keys or mTLS in front of the engine, no public exposure, and network policies (07.02 §7).

---

## 9. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **OOM at startup** | CrashLoop on load | Lower `--max-model-len` or `--gpu-memory-utilization`; FP8 weights/KV; more TP |
| **Preemption storms** | ITL spikes, throughput collapse under load | Capacity via `vllm_capacity`; lower `max-num-seqs`; FP8 KV; admission control at the gateway |
| **Poor prefix-cache hit rate** | High TTFT despite repeated prompts | Stable prefixes (02.02 §6.2); prefix-aware routing across replicas |
| **Slow scale-up** | Queues grow for minutes during spikes | Warm minimum; predictive scaling; pre-cached weights; a smaller fallback model via the gateway |
| **Upgrade regressions** | Latency or quality changes after a version bump | Pinned versions; benchmark + eval gates; canary |
| **Tool-call / reasoning parsing issues** | Broken tool calls from open models | Correct per-family parser flags and chat templates; contract tests |
| **TGI end-of-life risk** | No new features or models | Migrate per §7 |
| **Opaque performance** | Can't explain latency | Engine metrics + gateway traces + Nsight for deep dives |

---

## 10. Hands-On Projects

### Project 1 — Production vLLM on Kubernetes with Queue-Based Autoscaling

**User stories**
- *As a platform engineer*, I want a self-hosted model endpoint for the legislative assistant that meets P95 TTFT < 1 s and P95 ITL < 60 ms at peak, scales with demand, and survives node loss.

**Acceptance criteria**
1. Sizing with `vllm_capacity` for the chosen model and GPUs, validated against the engine's reported KV capacity (within ±10%).
2. A pinned-version Deployment (§6.1) with pre-cached weights, startup/readiness probes, network policies, API-key auth, and PodDisruptionBudgets.
3. Prefix-aware routing (production-stack router or llm-d), and KEDA autoscaling on the waiting queue with a warm minimum. Scale-up time measured.
4. Dashboards and alerts from `/metrics`; `engine_health`-style rules; traces from the gateway.
5. A load test at 1.5× the forecast peak with a node-kill chaos test: SLO adherence, preemptions, and recovery time reported.

**Step-by-step**
1. Choose the model and quantization (08.04); compute the capacity; pick the TP degree.
2. Build the manifests and deploy to a GPU node pool; pre-populate the model cache.
3. Deploy the router and KEDA; configure the Prometheus scraping.
4. Tune `max-num-seqs` and `max-num-batched-tokens` with `max_qps_at_slo` sweeps.
5. Run the load and chaos tests; document the runbook.

---

### Project 2 — TGI → vLLM Migration with Parity Testing

**User stories**
- *As the owner of an existing TGI deployment*, I want to move to a supported engine without changing client behaviour or degrading quality.

**Acceptance criteria**
1. An inventory of the TGI configuration and the client features used; a mapping document per §7.
2. A vLLM deployment in shadow mode mirroring production requests (with redaction), and an output diff report: exact-match rate at temperature 0, judge equivalence for the non-identical outputs, and latency comparison.
3. Structured-output and tool-calling conformance tests pass on vLLM.
4. A canary rollout at 5% → 50% → 100% via the gateway, with guardrail metrics (06.03), and rollback tested.
5. Post-migration: TGI decommissioned, dashboards rebuilt, costs compared.

**Step-by-step**
1. Document the TGI setup; build the vLLM configuration equivalents.
2. Implement traffic mirroring in the gateway; store both outputs.
3. Run the diff and judge comparisons; fix configuration mismatches (chat templates, stop tokens, parsers).
4. Canary and ramp.
5. Decommission and report.

---

### Project 3 — Multi-LoRA Serving for Department-Specific Adapters

**User stories**
- *As an ML engineer*, I want one base model to serve several fine-tuned adapters (fiscal notes, bill drafting style, constituent correspondence) with per-adapter quality gates and SLOs, instead of one GPU deployment per fine-tune.

**Acceptance criteria**
1. ≥ 3 LoRA adapters (trained in a later fine-tuning module, or public adapters for practice) served with `--enable-lora`, with the adapters registered by name and version.
2. The gateway routes requests to adapters by task. Per-adapter eval suites run in CI, and adapters are promoted only when they pass the gates.
3. Load tests with a mixed-adapter workload: throughput and latency vs single-adapter and vs separate deployments; the adapter-swap overhead is quantified.
4. Hot adapter updates without restarting the engine (where supported), with a rollback path.
5. A cost comparison: multi-LoRA vs dedicated deployments.

**Step-by-step**
1. Prepare the adapters and their eval sets.
2. Configure vLLM with LoRA support; register the adapters.
3. Extend the gateway with adapter routing and versioning.
4. Run the load tests with mixed traffic; measure.
5. Implement the promotion and rollback workflow; report.

---

## 11. Papers & Reading (exact titles)

- Kwon et al., 2023 — *Efficient Memory Management for Large Language Model Serving with PagedAttention*
- Yu et al., 2022 — *Orca: A Distributed Serving System for Transformer-Based Generative Models*
- Agrawal et al., 2024 — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- Zhong et al., 2024 — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*
- Zheng et al., 2024 — *SGLang: Efficient Execution of Structured Language Model Programs*
- Sheng et al., 2023 — *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*
- Chen et al., 2023 — *Punica: Multi-Tenant LoRA Serving*
- vLLM docs and release notes (pin your version); Hugging Face TGI docs (maintenance-mode notice); llm-d and vLLM production-stack documentation

## 12. Essential Tooling

| Tool | Role |
|---|---|
| **vLLM** (`vllm serve`, benchmarks, KV connectors) | Primary engine |
| **SGLang**, **TensorRT-LLM / Triton / NIM**, **llama.cpp / Ollama** | Alternatives for specific workloads |
| **vLLM production-stack**, **llm-d**, **KServe** | Kubernetes serving stacks, routing, disaggregation |
| **KEDA / HPA + Prometheus Adapter** | Queue-based autoscaling |
| **Prometheus + Grafana, DCGM exporter** | Engine and GPU metrics |
| **Nsight Systems** | Deep performance analysis |
