# 08.02 — Caching, Latency & Cost

> **Module 8: LLMOps & Serving** · Subtopic 2 of 5
> **Prerequisites:** 01.02 (TTFT/TPOT, prefix caching, speculative decoding), 02.02 §6 (prompt-caching break-even), 04.02 §7 (safe cache keys), 06.03–06.04 (online metrics, tracing), 08.01 (routing).
> **Outcome:** you can decompose any LLM feature's latency and cost into attributable parts, apply the right cache at the right layer without serving wrong answers, engineer for tail latency, and run an optimisation programme that cuts cost per successful task and P95 latency with measured, guarded changes.

> **Scope:** earlier modules explained the *mechanisms* (prefix/KV caching in 01.02, provider prompt caching economics in 02.02 §6, safe retrieval and semantic caches in 04.02 §7). This subtopic is the **system-level engineering and FinOps** view that ties them together.

---

## 1. Where the Milliseconds and Dollars Go

```
 request ─► gateway ─► guardrails ─► planner LLM ─► retrieval (‖ legs) ─► rerank ─► answer LLM ─► tools ─► answer LLM ─► guards ─► stream
 latency:   5 ms        20 ms         400 ms          90 ms                 50 ms      TTFT 900 ms   150 ms   TPOT×n_out    async      …
 cost:      –           classifier $  small-model $   embed $ + infra       GPU $      large-model $ API $   large-model $ judge $
```

**End-to-end latency** for a sequential chain of LLM calls:

$$
T_{\text{E2E}} = \sum_{\text{stages}} T_{\text{stage}},\qquad
T_{\text{LLM call}} \approx T_{\text{queue}} + T_{\text{net}} + \underbrace{T_{\text{prefill}}(n_{\text{in}}^{\text{uncached}})}_{\text{TTFT}} + (n_{\text{out}} - 1)\cdot \text{TPOT}
$$

**The single most important observation:** output tokens dominate the latency of most LLM calls. At 50 ms per output token, 400 output tokens take 20 s, while the prefill of 4,000 input tokens takes well under a second on a warm, cached prefix. **Shorter outputs are the biggest latency lever.**

**Perceived latency** differs from E2E latency. Streaming makes TTFT the dominant user-facing metric for chat, while agents and batch jobs care about E2E latency (06.03 §2).

---

## 2. The Cache Hierarchy

| Layer | What's cached | Hit condition | Wins | Risks |
|---|---|---|---|---|
| **Client / UI** | Rendered answers for repeat views | Same conversation state | Instant repeats | Stale after updates |
| **Exact response cache** (gateway) | Full response for (model, params, prompt, versions) | Byte-identical request | Zero model cost | Low hit rate for chat; leaks if not scoped by tenant/ACL |
| **Semantic cache** | Response for a *similar* query | Embedding similarity ≥ τ **and** identifier/constraint equality (04.02 §7) | Cheap answers for FAQs | Wrong answers for near-miss queries; must be scoped and versioned |
| **Retrieval / tool-result cache** | Search results, tool outputs | Same normalised query + filters + ACL + index version | Lower retrieval latency, fewer API calls | Staleness; ACL leaks |
| **Embedding cache** | Query/document vectors | Same text + model ID | Lower embedding cost | Model migration invalidation (03.01 §6) |
| **Provider prompt cache** | KV of a stable prompt prefix | Exact prefix match at breakpoints | Large input-cost and TTFT reductions (02.02 §6) | Busted by prefix edits and tool-order changes |
| **Engine prefix cache** (self-hosted) | KV blocks by prefix hash / radix tree (01.02 §3.3) | Shared prefix across requests | TTFT and GPU savings | Evictions under memory pressure |
| **KV offload tiers** (CPU RAM / SSD / remote) | Evicted KV blocks | Prefix reuse beyond GPU memory | Long-document and multi-turn reuse | Transfer bandwidth; complexity |

### 2.1 Cache correctness rules

1. **The key includes everything that changes the answer:** model and version, prompt version, parameters (temperature, tools, schema), retrieval/index version, **tenant and ACL scope**, and the as-of date for temporal content.
2. **Never cache across users or tenants** unless the content is provably public.
3. **Cache only deterministic-enough outputs** (temperature 0, factual lookups). Avoid caching creative or personalised outputs.
4. **Invalidate** on source changes (CDC events, 03.03 §5) and on version bumps. TTLs are a backstop, not the strategy.
5. **Measure the hit rate *and* the correctness** (sampled judge comparisons of cached vs fresh answers).

```python
import hashlib
import json
import time


class ResponseCache:
    def __init__(self, ttl_s: float = 3600):
        self.ttl, self.store = ttl_s, {}

    @staticmethod
    def key(model: str, params: dict, messages: list[dict], versions: dict, tenant: str, acl: list[str]) -> str:
        payload = {"m": model, "p": params, "msg": messages, "v": versions, "t": tenant, "acl": sorted(acl)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, k: str):
        hit = self.store.get(k)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        self.store.pop(k, None)
        return None

    def put(self, k: str, value, cacheable: bool):
        if cacheable:
            self.store[k] = (time.time(), value)

    def invalidate(self, predicate) -> int:
        """predicate(value) -> bool; e.g. drop answers citing a changed document."""
        dead = [k for k, (_, v) in self.store.items() if predicate(v)]
        for k in dead:
            del self.store[k]
        return len(dead)
```

---

## 3. Tail Latency Engineering

### 3.1 Fan-out amplifies tails

If a request makes $k$ parallel calls, each with latency CDF $F$, the request waits for the slowest:

$$
P(T_{\max} \le t) = F(t)^k \quad\Rightarrow\quad P(\text{any call exceeds its P99}) = 1 - 0.99^k
$$

With 10 parallel retrieval legs, tool calls, or sub-agents, **~10% of requests** see at least one P99-latency call (Dean & Barroso, *The Tail at Scale*).

### 3.2 Techniques

- **Hedged requests:** after a delay $d$ (e.g. the P95), issue a duplicate to another replica or provider and take the first to respond. Latency becomes $\min(X_1,\ d + X_2)$, and extra load ≈ $P(X_1 > d)$ (~5% at P95).
- **Timeouts with degradation:** each stage has a budget. On timeout, skip optional stages (reranker → fused order; 03.04 §6.3) instead of failing.
- **Deadline propagation:** pass the remaining budget downstream (headers, MCP/A2A metadata). Stages skip work that can't finish in time.
- **Parallelise independent stages:** retrieval legs; the planner running in parallel with speculative retrieval (04.04 §8).
- **Warm paths:** keep prompt caches warm for stable prefixes (tools + system prompt), and warm engine replicas before traffic shifts (08.05).

```python
import numpy as np


def p_any_exceeds(k: int, per_call_quantile: float = 0.99) -> float:
    return 1 - per_call_quantile ** k


def simulate_hedging(n: int = 100_000, mu: float = 0.0, sigma: float = 0.6, hedge_at_quantile: float = 0.95,
                     seed: int = 0) -> dict:
    """Lognormal call latency (seconds). Returns P50/P99 without and with hedging, plus the extra-load rate."""
    rng = np.random.default_rng(seed)
    x1 = rng.lognormal(mu, sigma, n)
    x2 = rng.lognormal(mu, sigma, n)
    d = np.quantile(x1, hedge_at_quantile)
    hedged = np.where(x1 > d, np.minimum(x1, d + x2), x1)
    q = lambda a, p: float(np.quantile(a, p))
    return {"p50": round(q(x1, .5), 3), "p99": round(q(x1, .99), 3),
            "p50_hedged": round(q(hedged, .5), 3), "p99_hedged": round(q(hedged, .99), 3),
            "extra_load": round(float((x1 > d).mean()), 3)}


def e2e_latency_mc(stages: list[dict], n: int = 50_000, seed: int = 0) -> dict:
    """stages: [{'kind': 'seq'|'par', 'samplers': [callable(rng, n) -> array]}]. Monte Carlo E2E percentiles."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    for st in stages:
        samples = np.stack([s(rng, n) for s in st["samplers"]])
        total += samples.max(0) if st["kind"] == "par" else samples.sum(0)
    return {p: round(float(np.quantile(total, p / 100)), 3) for p in (50, 95, 99)}
```

---

## 4. Cost Engineering

### 4.1 Unit economics

Price per **successful task**, not per token:

$$
C_{\text{task}} = \frac{1}{s}\sum_{\text{calls } j}\Big(T^{\text{in,uncached}}_j p^{\text{in}} + T^{\text{cache\_write}}_j p^{\text{write}} + T^{\text{cache\_read}}_j p^{\text{read}} + (T^{\text{out}}_j + T^{\text{reasoning}}_j)\,p^{\text{out}}\Big) + C_{\text{retrieval}} + C_{\text{tools}} + C_{\text{guards}}
$$

where $s$ is the task success rate (06.03). A cheaper model with a lower success rate can cost *more* per success.

```python
def call_cost(usage: dict, prices: dict) -> float:
    """usage: input_tokens (uncached), cache_creation_input_tokens, cache_read_input_tokens, output_tokens,
    reasoning_tokens (if billed separately from output). prices: USD per 1M tokens."""
    m = 1e-6
    return (usage.get("input_tokens", 0) * prices["in"] * m
            + usage.get("cache_creation_input_tokens", 0) * prices.get("cache_write", prices["in"]) * m
            + usage.get("cache_read_input_tokens", 0) * prices.get("cache_read", prices["in"]) * m
            + (usage.get("output_tokens", 0) + usage.get("reasoning_tokens", 0)) * prices["out"] * m)


def cost_per_success(records: list[dict], prices_by_model: dict) -> dict:
    """records: one per task: {'feature', 'tenant', 'success': bool, 'calls': [{'model', 'usage'}], 'other_usd'}."""
    agg: dict[str, dict] = {}
    for r in records:
        c = sum(call_cost(x["usage"], prices_by_model[x["model"]]) for x in r["calls"]) + r.get("other_usd", 0.0)
        a = agg.setdefault(r["feature"], {"usd": 0.0, "tasks": 0, "successes": 0})
        a["usd"] += c
        a["tasks"] += 1
        a["successes"] += int(r["success"])
    return {f: {**a, "usd_per_task": round(a["usd"] / a["tasks"], 5),
                "usd_per_success": round(a["usd"] / max(a["successes"], 1), 5)} for f, a in agg.items()}
```

### 4.2 The optimisation playbook (roughly in ROI order)

| # | Lever | Typical effect | Guard against |
|---|---|---|---|
| 1 | **Measure and attribute** cost per feature, tenant, and stage from traces (06.04) | Reveals where the money goes | — |
| 2 | **Prompt-cache the stable prefix** (tools, system, static docs; deterministic ordering) | Large input-cost and TTFT cuts on multi-turn and agent traffic | Cache-busting edits; check the hit rate |
| 3 | **Trim context:** tighter retrieval, dedup, observation masking, tool-result shaping (02.02, 04.02 §6) | Fewer input tokens; often *better* quality | Recall loss — evaluate |
| 4 | **Shorten outputs:** concise formats, structured outputs, stop sequences, `max_tokens` | Big latency and output-cost cuts | Completeness — evaluate |
| 5 | **Route and cascade** to smaller models (08.01) | 30–70% on mixed traffic | Quality floor via eval + A/B |
| 6 | **Batch APIs** for offline work (evals, backfills, labelling) — many providers discount batch traffic (e.g. 50%) | Halves offline costs | Latency tolerance (hours) |
| 7 | **Effort / thinking budgets** for reasoning models | Controls reasoning-token spend | Quality on hard slices |
| 8 | **Self-host high-volume, stable workloads** (08.03–08.04) | Lower marginal cost at high utilisation | Ops cost; utilisation reality |
| 9 | **Speculative decoding / quantization** (self-hosted) | Latency and GPU cost cuts | Quality validation (08.04) |
| 10 | **Response and semantic caching** where safe | Zero-cost repeats | Correctness and scoping (§2.1) |

Every lever is a change that needs an **eval gate** (06.01) and **online guardrails** (06.03). Cost wins that quietly reduce quality are regressions.

### 4.3 Budgets and FinOps

- Show costs back per team or feature, with a budget per tenant (07.02 §6 `CostBucket`) and alerts on spend velocity.
- Forecast from traffic × cost per task. Model the legislative-session seasonality explicitly.
- Review the top cost drivers monthly. Track **cost per successful task** as a first-class KPI next to quality.

---

## 5. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Unattributed spend** | "The bill doubled" with no explanation | Per-call usage in traces; cost per feature/tenant/stage dashboards |
| **Cache hit rate collapse** | TTFT and cost jump after a deploy | Diff rendered prefixes; freeze tool ordering; monitor cache_read ratios (02.02 §6.2) |
| **Wrong cached answers** | Users see answers for similar-but-different questions | Full keys, including versions, tenant, and ACL; identifier guards for semantic caches; CDC invalidation |
| **Tail latency from fan-out** | P99 ≫ P95 | Hedging, timeouts with degradation, deadline propagation, fewer parallel dependencies |
| **Long outputs** | Slow responses, high output costs | Output-length budgets; concise formats; structured outputs; summaries with "expand" affordances |
| **Reasoning-token surprises** | Cost spikes on reasoning models | Effort controls; route easy queries to non-reasoning paths; per-request caps |
| **Optimisations degrade quality** | Complaints after a cost-cutting release | Eval gates + A/B guardrails for every cost change |
| **Offline jobs at online prices** | Evals and backfills are expensive | Batch APIs; off-peak self-hosted capacity |

---

## 6. Hands-On Projects

### Project 1 — Cost and Latency Attribution with an Optimisation Sprint

**User stories**
- *As the product owner*, I want to know exactly what each feature costs per successful task and where the latency goes, then cut both without hurting quality.

**Acceptance criteria**
1. Every model, retrieval, tool, and guard call emits usage and latency in traces (06.04). `cost_per_success` dashboards per feature, tenant, and stage, with daily trends.
2. A latency breakdown per stage (P50/P95/P99), plus an `e2e_latency_mc` model calibrated to the measured stage distributions (predicted vs observed E2E within ±10%).
3. An optimisation sprint applying ≥ 5 levers from §4.2, each behind a flag, eval-gated (06.01), and A/B-tested or canaried (06.03).
4. Targets: −40% cost per successful task and −30% P95 E2E latency, with non-inferior quality (CI-based).
5. A before/after report with attribution of the gains to each lever.

**Step-by-step**
1. Instrument the usage and cost attributes; build the dashboards.
2. Fit the stage latency distributions; build the Monte Carlo model.
3. Rank the levers by expected ROI; implement them one by one behind flags.
4. Gate each with evals; roll out with canaries or A/B tests.
5. Write the report.

---

### Project 2 — Multi-Layer Cache with Correctness Guarantees

**User stories**
- *As a platform engineer*, I want caching at every sensible layer with zero cross-tenant leaks and measurable correctness.

**Acceptance criteria**
1. Layers: an exact response cache (`ResponseCache` with full keys), a retrieval-result cache, an embedding cache, provider prompt caching (breakpoints per 02.02 §6.2), and a semantic cache for a public FAQ slice only (with an identifier guard).
2. Invalidation on document-change events (CDC → Kafka → cache `invalidate` by the cited document IDs) and on version bumps. TTLs as a backstop.
3. Tests: property tests showing that keys differ whenever the tenant, ACL, model, prompt version, index version, or as-of date differ. A cross-tenant leak test suite (0 leaks).
4. Correctness sampling: 1% of cache hits re-computed fresh and judged for equivalence; the alert threshold for divergence is < 1%.
5. Reports the hit rates, savings, and latency improvements per layer.

**Step-by-step**
1. Implement the caches (Redis) with key builders per layer.
2. Wire the CDC-driven invalidation (03.03 §5).
3. Write the property and leak tests.
4. Add correctness sampling with a validated judge (06.01 §4).
5. Build the dashboards; tune the TTLs.

---

### Project 3 — Tail-Latency Engineering for an Agentic Pipeline

**User stories**
- *As an SRE*, I want the P99 latency of the research agent within 2× of its P50, even when providers or tools have bad moments.

**Acceptance criteria**
1. Deadline propagation through all stages (HTTP headers, MCP `_meta`, A2A metadata), with stage-level budgets and graceful degradation paths.
2. Hedged requests for LLM calls (hedge at the observed P95 TTFT), with measured extra load; validated against `simulate_hedging` predictions.
3. Timeouts with fallbacks for the optional stages (reranker, groundedness guard → async flagging).
4. Load tests with injected latency (Toxiproxy) show P99/P50 ≤ 2 under a 5% slow-call rate, at < 10% extra cost.
5. The streaming UX metrics — TTFT, time to first *useful* content (first citation or first sentence), and abandonment — improve relative to the baseline.

**Step-by-step**
1. Map the pipeline's stages and their dependencies; set the budgets.
2. Implement the deadline context and propagation; add degradation paths.
3. Implement hedging in the gateway (08.01) with per-model delays.
4. Run the fault-injection load tests; tune.
5. Measure the UX metrics; report.

---

## 7. Foundational Papers & Reading (exact titles)

- Dean & Barroso, 2013 — *The Tail at Scale*
- Kwon et al., 2023 — *Efficient Memory Management for Large Language Model Serving with PagedAttention*
- Zheng et al., 2024 — *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention prefix reuse)
- Gim et al., 2024 — *Prompt Cache: Modular Attention Reuse for Low-Latency Inference*
- Liu et al., 2024 — *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving*
- Bang, 2023 — *GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings*
- Chen, Zaharia & Zou, 2023 — *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*
- Provider docs: prompt caching, batch processing, usage fields (verify current prices and multipliers)

## 8. Essential Tooling

| Tool | Role |
|---|---|
| **Redis / Valkey** | Response, retrieval, and embedding caches |
| **Provider prompt caching + batch APIs** | Input-cost and offline-cost reduction |
| **vLLM/SGLang prefix caching, LMCache-style KV offload** | Self-hosted KV reuse |
| **OpenTelemetry + Grafana / ClickHouse** | Cost and latency attribution |
| **k6 / Locust + Toxiproxy** | Load and tail-latency testing |
| **LLM gateways** (08.01) | Hedging, budgets, caching hooks |
