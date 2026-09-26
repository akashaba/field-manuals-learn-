# 08.01 — Model Selection & Routing

> **Module 8: LLMOps & Serving** · Subtopic 1 of 5
> **Prerequisites:** 06.01 (evals, paired tests), 06.03 (online metrics), 01.02 (latency metrics), 04.03 §5 (calibration), 05.02 §5 (reliability math).
> **Outcome:** you can choose models with evidence (quality on *your* tasks, latency, cost, and governance constraints), compute total cost of ownership for API vs self-hosted options, and build routing layers — rules, learned routers, cascades, fallbacks — that cut cost and improve resilience without silently degrading quality.

---

## 1. Model Selection Is a Multi-Objective Decision

No single "best model" exists. There is a **Pareto frontier** over quality, latency, and cost, constrained by governance:

| Dimension | Measure |
|---|---|
| **Quality** | Your eval suite (06.01): task success per slice, groundedness, safety, tool-use accuracy |
| **Latency** | TTFT and TPOT at P50/P95 under realistic load and prompt lengths (01.02 §1.1) |
| **Cost** | $ per successful task, not per token (reasoning tokens, retries, and cache effects included) |
| **Capabilities** | Context length, tool calling, structured outputs, vision, languages, fine-tunability |
| **Governance** | Data residency and retention, compliance authorisations for your sector, contract terms, model deprecation schedules, licences (open weights) |
| **Operations** | Rate limits, availability history, regional capacity, support, version pinning |

```
 quality ▲                         ● large reasoning model
         │                  ● large model
         │           ● mid model  (Pareto frontier: nothing is both cheaper AND better)
         │     ● small model
         │  ○ fine-tuned small  ← often beats larger general models on narrow tasks
         │ ● tiny
         └──────────────────────────────────────────────►  cost per successful task
```

### 1.1 Selection workflow

1. **Requirements:** quality bars per slice, latency SLOs, budget, and hard governance constraints. The governance constraints eliminate options first.
2. **Shortlist** 4–8 candidates across tiers (small / mid / frontier; API / open weights).
3. **Evaluate on your data** (06.01): the same prompts, adapted minimally per model. Report paired differences with CIs.
4. **Measure latency and cost** with your real prompt and output length distributions (01.02 §9 load tester).
5. **Compute the frontier** and pick per *task*, not per company. Different pipeline stages often want different models: planner, extractor, answerer, judge.
6. **Document** the decision memo: evidence, assumptions, and triggers for re-evaluation (a new model release, a price change, a deprecation notice).

```python
def pareto_frontier(cands: list[dict], quality="quality", cost="cost") -> list[dict]:
    """Keep candidates not dominated (no other is at least as good on both and strictly better on one)."""
    front = []
    for c in cands:
        dominated = any(o[quality] >= c[quality] and o[cost] <= c[cost] and
                        (o[quality] > c[quality] or o[cost] < c[cost]) for o in cands)
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: c[cost])
```

---

## 2. Total Cost of Ownership: API vs Self-Hosted

**API cost per successful task:**

$$
C_{\text{API}} = \frac{\sum_{\text{calls}} \big(T_{\text{in}}^{\text{uncached}}p_{\text{in}} + T_{\text{in}}^{\text{cached}}p_{\text{cache}} + T_{\text{out}}p_{\text{out}}\big)}{\text{success rate}}
$$

**Self-hosted cost per 1M output tokens** (at a throughput that meets the SLO):

$$
C_{\text{self}} = \frac{N_{\text{GPU}} \cdot c_{\text{GPU/h}}}{\text{tokens/h at SLO} \cdot u} \times 10^6 + C_{\text{ops}}
$$

Here $u$ is the average utilisation, and it is the term everyone forgets. A GPU serving 20% of its peak throughput costs 5× more per token. $C_{\text{ops}}$ covers the engineering on-call, upgrades, monitoring, and idle capacity for peaks.

**Break-even monthly volume** (API vs a fixed self-hosted cluster):

$$
V^* = \frac{\text{monthly cluster cost} + \text{monthly ops cost}}{c_{\text{API per token}}}
$$

Self-hosting wins with **high, steady volume**, **strict data-control requirements**, or **customisation needs** (fine-tunes, special decoding). APIs win with **spiky or low volume**, **frontier-quality needs**, and **small teams**.

```python
def self_host_cost_per_m(gpus: int, gpu_hour_usd: float, tokens_per_s_at_slo: float,
                         utilization: float, ops_usd_per_month: float = 0.0,
                         hours_per_month: float = 730) -> float:
    tokens_per_month = tokens_per_s_at_slo * 3600 * hours_per_month * utilization
    monthly = gpus * gpu_hour_usd * hours_per_month + ops_usd_per_month
    return monthly / tokens_per_month * 1e6


def breakeven_tokens_per_month(gpus, gpu_hour_usd, ops_usd_per_month, api_usd_per_m, hours=730):
    return (gpus * gpu_hour_usd * hours + ops_usd_per_month) / api_usd_per_m * 1e6
```

---

## 3. Routing

A **router** picks the model (or path) per request. Typical routers save 30–70% of cost on mixed traffic, because most requests don't need the most capable model.

### 3.1 Router types

| Type | How | Pros | Cons |
|---|---|---|---|
| **Static by task** | Each pipeline stage is pinned to a model | Simple, predictable | No per-request adaptivity |
| **Rules** | Route on features: intent, prompt length, presence of tools or attachments, tenant tier | Transparent | Brittle; needs maintenance |
| **Learned router** | A classifier predicts whether the cheap model's answer will be acceptable (trained on preference or eval data; e.g. RouteLLM) | Adapts to the query distribution | Needs labelled data; drifts |
| **Cascade** | Try the cheap model → verify (confidence, validator, judge) → escalate if needed | Uses actual output quality | Adds latency on escalations |
| **Ensemble / voting** | Several models; aggregate | Higher quality | Highest cost |

### 3.2 Cascade economics

Let the cheap model cost $c_s$ and the large one $c_\ell$. A verifier accepts the cheap answer with probability $a$ and itself costs $c_v$. Then:

$$
\mathbb{E}[\text{cost}] = c_s + c_v + (1-a)\,c_\ell, \qquad
\mathbb{E}[\text{quality}] = a\,q_{s|\text{acc}} + (1-a)\,q_\ell
$$

The cascade beats "always large" on cost when $c_s + c_v < a\,c_\ell$. Quality holds only if the verifier accepts almost exclusively the cheap answers that are actually good, i.e. it has **high precision on acceptance**. Choose the acceptance threshold on a validation set to hit a quality floor.

```python
def cascade_eval(conf: list[float], small_ok: list[bool], large_ok: list[bool], tau: float,
                 c_small: float, c_large: float, c_verify: float = 0.0) -> dict:
    """conf: verifier/router confidence that the small answer is acceptable (calibrated)."""
    n = len(conf)
    accept = [c >= tau for c in conf]
    quality = sum(s if a else l for a, s, l in zip(accept, small_ok, large_ok)) / n
    cost = sum(c_small + c_verify + (0 if a else c_large) for a in accept) / n
    return {"tau": tau, "quality": round(quality, 4), "cost": round(cost, 6),
            "accept_rate": round(sum(accept) / n, 3)}


def choose_cascade_threshold(conf, small_ok, large_ok, c_small, c_large, c_verify=0.0,
                             max_quality_drop=0.01):
    base_q = sum(large_ok) / len(large_ok)
    options = [cascade_eval(conf, small_ok, large_ok, t / 100, c_small, c_large, c_verify) for t in range(0, 101)]
    ok = [o for o in options if o["quality"] >= base_q - max_quality_drop]
    return min(ok, key=lambda o: o["cost"]) if ok else None
```

### 3.3 Learned routers

Train a classifier $r(x) \approx P(\text{cheap model is sufficient} \mid x)$ from:
- **Paired eval data:** run both models on logged queries and label the pairs where the cheap model was as good (via a judge validated against humans; 06.01 §4).
- **Preference data:** human or arena-style comparisons (RouteLLM used such data, with augmentation).

Features range from simple (length, intent, keywords, the presence of code or math) to query embeddings (a logistic regression or small MLP on top), up to a small LLM classifier. **Calibrate** the router (04.03 §5), then pick the threshold with `choose_cascade_threshold` logic. **Monitor** the router's acceptance rate and the quality of routed traffic online (06.03). Routers drift when traffic changes.

---

## 4. Resilience: Fallbacks, Circuit Breakers, Hedging

Providers have outages, rate limits, and latency spikes. Make routing **health-aware**:
- **Fallback chains** by capability tier: `primary → same-tier alternative provider → degraded (smaller model plus a "limited mode" notice) → queue/async`.
- **Circuit breakers** per provider and model: open after an error-rate threshold; half-open probes to recover.
- **Retries** with exponential backoff and jitter, only for retryable errors (429, 5xx, timeouts). Respect `Retry-After`.
- **Hedged requests** for tail latency: if no first token arrives within the P95 TTFT, send a duplicate to another replica or provider and use whichever responds first. Cost rises by roughly the hedge rate (~5%).
- **Prompt portability:** a fallback model is only a fallback if your prompts, tools, and structured outputs work on it. **Evaluate the fallback path regularly** (06.01), and keep per-model prompt variants in the registry (02.01 §7.3).

```python
import random
import time


class CircuitBreaker:
    def __init__(self, fail_threshold: float = 0.5, window: int = 20, cooldown_s: float = 30.0, min_calls: int = 5):
        self.th, self.window, self.cooldown, self.min_calls = fail_threshold, window, cooldown_s, min_calls
        self.results: list[bool] = []
        self.opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.cooldown:
            return True                                     # half-open: let a probe through
        return False

    def record(self, ok: bool):
        self.results = (self.results + [ok])[-self.window:]
        if self.opened_at is not None and ok:
            self.opened_at, self.results = None, [True]     # probe succeeded: close
            return
        fails = self.results.count(False)
        if len(self.results) >= self.min_calls and fails / len(self.results) >= self.th:
            self.opened_at = time.monotonic()


def route_with_fallback(request, chain: list[tuple[str, callable]], breakers: dict[str, CircuitBreaker],
                        retryable=(TimeoutError, ConnectionError), max_retries: int = 1):
    errors = []
    for name, call in chain:
        br = breakers[name]
        if not br.allow():
            errors.append((name, "circuit open"))
            continue
        for attempt in range(max_retries + 1):
            try:
                out = call(request)
                br.record(True)
                return {"model": name, "output": out, "errors": errors}
            except retryable as e:
                br.record(False)
                errors.append((name, type(e).__name__))
                time.sleep(min(2.0, 0.05 * 2 ** attempt) * random.random())
            except Exception as e:                          # non-retryable: move to the next model
                br.record(False)
                errors.append((name, type(e).__name__))
                break
    raise RuntimeError(f"all routes failed: {errors}")
```

---

## 5. The LLM Gateway

Centralise model access behind a **gateway** instead of scattering SDK calls across services:

```
 services ──► LLM GATEWAY ──► providers (Anthropic, OpenAI, Azure, Bedrock, Vertex) / self-hosted vLLM
               • unified API (OpenAI-compatible or internal) + per-model adapters
               • routing: rules, learned router, cascades, fallbacks, circuit breakers
               • authN, per-team keys, quotas & cost budgets (07.02 §6)
               • caching hooks (08.02), guardrail hooks (07.01)
               • logging/tracing with gen_ai.* attributes (06.04), cost attribution per team/feature
               • version pinning & deprecation tracking
```

Options include open-source proxies (LiteLLM, Envoy AI Gateway, agentgateway), commercial gateways, cloud-native gateways, or an in-house service (e.g. Spring Cloud Gateway with custom filters). Whatever you choose: **pin model versions** explicitly, never "latest" aliases in production.

---

## 6. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Leaderboard-driven choices** | A top benchmark model underperforms on your tasks | Evaluate on your data, per slice (06.01–06.02) |
| **Hidden costs** | The "cheap" model costs more per successful task | Measure cost per success, including retries, reasoning tokens, and escalations |
| **Router quality drift** | Complaints rise after a traffic shift | Online monitoring of routed-traffic quality; periodic re-labelling and retraining |
| **Unevaluated fallbacks** | Outage → the fallback breaks structured outputs | Scheduled fallback-path evals; per-model prompt variants; contract tests |
| **Provider outages and rate limits** | Error spikes, 429 storms | Circuit breakers, multi-provider fallback, backoff with jitter, queues |
| **Tail latency** | P99 far above P95 | Hedged requests, timeouts, smaller models for latency-critical steps |
| **Governance violations** | Sensitive data sent to a non-approved endpoint | Gateway policy: data-classification routing, endpoint allowlists, audit |
| **Self-hosting at low utilisation** | GPU bills without savings | TCO with realistic utilisation; hybrid (self-host the base load, API for peaks) |

---

## 7. Hands-On Projects

### Project 1 — Model Bake-Off and Selection Memo

**User stories**
- *As the AI platform lead*, I need an evidence-based choice of models for each stage of the legislative assistant (planner, retriever-reranker, answerer, judge), within budget, governance, and latency constraints.

**Acceptance criteria**
1. ≥ 6 candidates across tiers (API and open weights), filtered first by governance requirements (data residency and retention, compliance posture), with the rationale documented.
2. Each candidate is evaluated on the 06.02 golden set per stage, with paired CIs vs the incumbent; latency and cost are measured under realistic load.
3. A Pareto frontier per stage (`pareto_frontier`) and a TCO analysis for self-hosting the best open-weights option (`self_host_cost_per_m`, `breakeven_tokens_per_month`) at measured throughput and a realistic utilisation.
4. A decision memo: per-stage choices, expected monthly cost, risks, and re-evaluation triggers.
5. The chosen configuration is pinned in the gateway, with the per-model prompt variants in the registry.

**Step-by-step**
1. Write the requirements and constraints; shortlist.
2. Adapt the prompts minimally per model; run the evals with output caching.
3. Load-test the latency and throughput (API and self-hosted vLLM, 08.03).
4. Compute the frontiers and the TCO.
5. Write the memo; configure the gateway.

---

### Project 2 — Cascade and Learned Router with a Quality Floor

**User stories**
- *As a FinOps-minded engineer*, I want to cut model spend by ≥ 40% while keeping answer quality within 1 point of the all-large-model baseline.

**Acceptance criteria**
1. Data: ≥ 3,000 logged queries answered by both a small and a large model, labelled acceptable or not by a validated judge (06.01 §4), with a human-labelled audit subset.
2. Routers:
   - (a) rules;
   - (b) an embedding + logistic-regression router;
   - (c) a cascade with a verifier (validator + small judge) after the small model.
   All are calibrated.
3. Thresholds are chosen with `choose_cascade_threshold` at a quality floor of −1 point, and evaluated on held-out data: cost reduction, quality with CI, added latency (P50/P95), and escalation rate.
4. An online A/B test (06.03) with guardrail metrics, showing non-inferior quality and the cost savings.
5. Drift monitoring for router acceptance and quality, with a retraining playbook.

**Step-by-step**
1. Build the paired dataset (batch APIs for cost); label it; validate the judge.
2. Train and calibrate the routers; build the verifier.
3. Choose the thresholds and evaluate offline.
4. Deploy in the gateway behind a flag; run the A/B test.
5. Add the monitoring and write the playbook.

---

### Project 3 — Resilient Multi-Provider LLM Gateway

**User stories**
- *As an SRE*, I want model access that survives provider outages, rate limits, and latency spikes, with per-team budgets and full observability.

**Acceptance criteria**
1. A gateway (LiteLLM/Envoy AI Gateway, or Spring Cloud Gateway + custom filters) exposing an internal API with adapters for ≥ 2 providers plus self-hosted vLLM.
2. Resilience: fallback chains by tier, per-route `CircuitBreaker`, retries only for retryable errors, hedged requests on TTFT timeouts, and `Retry-After` handling.
3. Governance: data-classification tags on requests route sensitive data only to approved endpoints; per-team keys, quotas, and cost budgets.
4. Observability: gen_ai.* spans, and cost attribution per team, feature, and model (06.04).
5. Chaos tests: provider 100% errors, 30% 429s, P99 latency spikes, and partial outage of the self-hosted engine. Measures the availability and latency SLO adherence during each; fallback correctness is verified with contract tests on structured outputs.

**Step-by-step**
1. Stand up the gateway; implement the adapters and a unified request/response schema.
2. Implement the routing, breakers, retries, and hedging; unit-test with fake providers.
3. Add the policy engine for data classification and budgets.
4. Instrument with OpenTelemetry; build the dashboards.
5. Run the chaos scenarios (a fault-injection proxy such as Toxiproxy); document the results and tune.

---

## 8. Foundational Papers & Reading (exact titles)

- Ong et al., 2024 — *RouteLLM: Learning to Route LLMs with Preference Data*
- Chen, Zaharia & Zou, 2023 — *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*
- Ding et al., 2024 — *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*
- Hu et al., 2024 — *RouterBench: A Benchmark for Multi-LLM Routing System*
- Jiang, Ren & Lin, 2023 — *LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion*
- Dean & Barroso, 2013 — *The Tail at Scale*
- Nygard — *Release It!* (circuit breakers, bulkheads; book)
- Liang et al., 2022 — *Holistic Evaluation of Language Models* (multi-metric evaluation framing)

## 9. Essential Tooling

| Tool | Role |
|---|---|
| **LiteLLM, Envoy AI Gateway, agentgateway, Portkey** | LLM gateways and proxies |
| **Spring Cloud Gateway + Resilience4j** | JVM gateway with circuit breakers, retries, rate limiters |
| **RouteLLM** | Learned router training and evaluation |
| **Inspect AI / promptfoo** | Cross-model evaluation |
| **vLLM benchmark tools, k6/Locust** | Latency and throughput measurement |
| **Toxiproxy / Chaos Mesh** | Fault injection for resilience tests |
| **OpenTelemetry + FinOps dashboards** | Cost attribution and routing observability |
