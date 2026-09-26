# 08.05 — Production Lifecycle (Releasing, Operating, and Governing LLM Systems)

> **Module 8: LLMOps & Serving** · Subtopic 5 of 5
> **Prerequisites:** 06.01–06.04 (evals, datasets, online metrics, tracing), 07.01–07.03 (guardrails, security), 08.01–08.04. Familiarity with CI/CD, GitOps, Kubernetes, and SRE practice.
> **Outcome:** you can version an LLM application as one deployable unit, gate every change with the right evals, roll it out progressively with automatic rollback, run it against SLOs with burn-rate alerting, survive provider deprecations and surges, handle LLM-specific incidents, and satisfy governance requirements (NIST AI RMF, and, for Montana public-sector work, **HB 178 (2025)**) with engineering controls rather than paperwork.

---

## 1. Why LLMOps Is Not Just MLOps

| Classic service | Classic ML | **LLM application** |
|---|---|---|
| Behaviour = code | Behaviour = code + weights + features | Behaviour = code + **model (often a vendor's)** + decoding params + **prompts** + **tool schemas** + **retrieval index + embedding model** + **guard policy** + serving config |
| Tests are deterministic | Offline metrics on a fixed test set | **Stochastic outputs**; quality needs graded evals (06.01) with CIs |
| Dependencies change when you upgrade them | Data drift | **Vendors change or retire models underneath you**; prompts drift in effect as models change |
| Cost ≈ fixed infra | Training ≫ inference | **Inference cost scales per token** and can run away (agent loops, long contexts) |
| Failures are errors | Failures are wrong predictions | Failures include **fluent wrong answers, unsafe actions, injected behaviour**, and data leaks |

**The core discipline:** everything that determines behaviour is **versioned, hashed, evaluated, and deployed as one unit**, and every production request is **traceable to that unit** (06.04).

```
  ┌─────────── DESIGN ───────────┐   ┌──────── BUILD ────────┐   ┌──────── EVALUATE ────────┐
  │ use-case risk tier, SLOs,    │──►│ bundle = code+model+   │──►│ offline evals, safety,   │
  │ data/ACL map, HB 178 checks  │   │ prompts+tools+index+   │   │ perf, contract tests     │
  └──────────────────────────────┘   │ guards+serving (hash)  │   └────────────┬─────────────┘
          ▲                          └────────────────────────┘                │ gates pass
          │                                                                     ▼
  ┌──────── LEARN ────────┐   ┌────────────── OPERATE ──────────────┐   ┌──── RELEASE ─────┐
  │ traces → triage →     │◄──│ SLOs + burn alerts, autoscale,      │◄──│ shadow → canary  │
  │ datasets (06.02),     │   │ drift monitors, cost budgets,       │   │ → progressive,   │
  │ postmortems, retire   │   │ incidents, kill switches            │   │ auto-rollback    │
  └───────────────────────┘   └─────────────────────────────────────┘   └──────────────────┘
```

---

## 2. The Release Unit: A Bundle Manifest

Deploy and roll back **bundles**, not individual prompts or model strings. The bundle's digest is stamped on every trace (`app.bundle` attribute; 06.04), so any output can be tied to exactly what produced it. **Compatibility rules** encode which gates each kind of change needs, and which combinations are invalid.

```python
import hashlib, json
from dataclasses import dataclass, field, asdict

@dataclass(frozen=True)
class Bundle:
    """Everything that determines behaviour. Deploy/rollback this as ONE versioned unit."""
    app: str
    model: str                      # provider/model@snapshot, or registry URI + digest
    decoding: dict                  # temperature, max_tokens, reasoning effort, ...
    prompts: dict                   # prompt name -> content hash (prompt registry)
    tools: dict                     # tool name -> schema hash / MCP server version
    embedding_model: str
    index: str                      # vector index build id (built WITH embedding_model)
    guard_policy: str               # guardrail policy version (07.01)
    eval_suite: str                 # frozen eval dataset hash (06.02)
    serving: dict = field(default_factory=dict)   # engine image, quant scheme, flags (08.03/08.04)

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

# component -> (required gates, extra required actions)
RULES = {
    "model":           ({"offline_eval", "safety_eval", "perf_test", "canary"}, set()),
    "decoding":        ({"offline_eval", "canary"}, set()),
    "prompts":         ({"offline_eval", "canary"}, set()),
    "tools":           ({"offline_eval", "safety_eval", "contract_test", "canary"}, set()),
    "embedding_model": ({"retrieval_eval", "offline_eval", "canary"}, {"reindex"}),
    "index":           ({"retrieval_eval", "canary"}, set()),
    "guard_policy":    ({"safety_eval", "guard_eval"}, set()),
    "eval_suite":      (set(), {"rebaseline"}),
    "serving":         ({"perf_test", "parity_eval", "canary"}, set()),
}

def plan_release(old: Bundle, new: Bundle) -> dict:
    a, b = asdict(old), asdict(new)
    changed = sorted(k for k in a if k != "app" and a[k] != b[k])
    gates, actions, errors = set(), set(), []
    for k in changed:
        g, act = RULES[k]
        gates |= g; actions |= act
    if "embedding_model" in changed and "index" not in changed:
        errors.append("embedding_model changed but index not rebuilt: vectors from different models are not comparable")
    if "eval_suite" in changed and len(changed) > 1:
        errors.append("do not change the eval suite in the same release as the system under test")
    return dict(from_=old.digest(), to=new.digest(), changed=changed,
                gates=sorted(gates), actions=sorted(actions), errors=errors)

base = Bundle(app="bill-assistant", model="azure-openai/gpt-x@2026-05-01", decoding={"temperature": 0.2},
              prompts={"system": "a1f3", "summarize": "9c2e"}, tools={"search_bills": "v3", "get_statute": "v2"},
              embedding_model="text-embed-3@v1", index="idx-2026-09-01", guard_policy="gp-12",
              eval_suite="ev-7", serving={})
p1 = plan_release(base, Bundle(**{**asdict(base), "prompts": {"system": "a1f3", "summarize": "77bd"}}))
print(p1)
assert p1["changed"] == ["prompts"] and not p1["errors"]
p2 = plan_release(base, Bundle(**{**asdict(base), "embedding_model": "text-embed-4@v1"}))
print(p2)
assert "reindex" in p2["actions"] and p2["errors"]
p3 = plan_release(base, Bundle(**{**asdict(base), "embedding_model": "text-embed-4@v1", "index": "idx-2026-09-20"}))
assert not p3["errors"] and "retrieval_eval" in p3["gates"]
```

**Where each component lives:**
- **Prompts** live in a registry (Git and a prompt-management tool) and are addressed by content hash, never edited in place in production.
- **Models** are referenced by **dated snapshot** (`@2026-05-01`), never by floating alias (`-latest`).
- **Indexes** are immutable builds with blue/green aliases (03.03 §9).
- **Guard policies** are policy-as-code (07.01).
- **The bundle itself** is a Git-tracked YAML/JSON file. The deploy system (Argo CD / Flux) promotes a *digest* through environments.

**Blue/green for indexes and embedding changes.** Build `idx-new` with the new embedding model alongside `idx-old`. Run retrieval evals, then atomically swap the alias together with the bundle. Keep `idx-old` until the rollback window closes. **Never** query an index with a different embedding model than the one it was built with; `plan_release` enforces this.

---

## 3. CI/CD: Gates per Stage

```
 PR opened ──► unit tests · schema/contract tests for tools (05.01) · prompt lint (template vars, token budget)
            └► SMOKE EVAL: 50–100 items, fast grader, fail on hard invariants (JSON validity, refusals, PII)
 merge ─────► FULL OFFLINE EVAL (06.01): paired vs prod bundle, per-slice CIs, judge + deterministic graders
            ├► SAFETY: guard eval + injection/red-team suite (07.03 §6), OWASP checks (07.02)
            ├► PERF (if model/serving changed): max QPS at SLO, TTFT/ITL p95 (08.03 §5)
            └► COST: projected $/request on replayed traffic (08.02 §4.1) ≤ budget
 release ───► SHADOW (optional, no user impact) → CANARY 1–5% → 25% → 50% → 100%, auto-rollback on gate breach
 post ──────► daily fixed-eval "drift canary" on prod bundle; weekly sample → human review → dataset (06.02)
```

**Gate design rules:**
- Gates compare the **candidate vs the current production bundle on the same items**, using a paired design (06.01 §3). Absolute thresholds rot as the dataset changes.
- **Hard invariants** (schema validity, PII leakage = 0, tool-authorization tests) are pass/fail. **Soft metrics** (quality, helpfulness) use non-inferiority margins with CIs, per slice (08.04 §4).
- **The eval suite is versioned separately** and never changed in the same release as the system (`plan_release` error). Otherwise you cannot tell whether the model or the ruler moved.
- **Budget the evals themselves.** A full judge-based eval costs real money, so cache per (item, bundle digest) and run the full suite only on merge.

```yaml
# .github/workflows/llm-release.yml (sketch). The same shape works in Azure DevOps/GitLab.
on: { pull_request: {}, push: { branches: [main] } }
jobs:
  smoke:
    if: github.event_name == 'pull_request'
    steps:
      - run: make test contract-test prompt-lint
      - run: python -m evals.run --suite smoke --bundle bundle.yaml --baseline prod --fail-on hard
  full:
    if: github.ref == 'refs/heads/main'
    steps:
      - run: python -m release.plan --from prod --to bundle.yaml > plan.json        # plan_release()
      - run: python -m evals.run --suite full --gates "$(jq -r '.gates|join(",")' plan.json)"
      - run: python -m release.publish --bundle bundle.yaml --require-report evals/report.json
```

---

## 4. Progressive Delivery and Automatic Rollback

**Shadow** sends a copy of traffic to the candidate and discards its output. It is good for performance, cost, and diffing outputs, but **not for tools with side effects**: stub write tools, or run shadow only on read paths.

**Canary** serves real users a small share. Route by **stable user/session hash** so each user gets a consistent experience and the metrics are not contaminated across arms (06.03). Keep a **holdback** for long-running quality comparisons.

The canary analysis should use **one-sided tests on regressions** (errors, negative feedback, guard blocks), latency and cost ratios, and a **minimum sample size** before any decision:

```python
import math

def two_prop_z(x1, n1, x2, n2):
    """z for p2 - p1 (pooled): positive = canary worse on a 'bad' rate."""
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    return (x2 / n2 - x1 / n1) / se

def canary_decision(base, canary, min_n=500, z_crit=2.33, p95_slack=1.15, max_cost_ratio=1.10):
    """base/canary: dict(n, errors, bad_feedback, guard_blocks, p95_ms, cost_per_req).
    Returns ('wait'|'promote'|'rollback', reasons)."""
    if canary["n"] < min_n:
        return "wait", [f"n={canary['n']} < {min_n}"]
    reasons = []
    for k in ("errors", "bad_feedback", "guard_blocks"):
        z = two_prop_z(base[k], base["n"], canary[k], canary["n"])
        if z > z_crit:
            reasons.append(f"{k}: {canary[k]/canary['n']:.3%} vs {base[k]/base['n']:.3%} (z={z:.1f})")
    if canary["p95_ms"] > p95_slack * base["p95_ms"]:
        reasons.append(f"p95 {canary['p95_ms']}ms > {p95_slack}x baseline {base['p95_ms']}ms")
    if canary["cost_per_req"] > max_cost_ratio * base["cost_per_req"]:
        reasons.append(f"cost/req {canary['cost_per_req']:.4f} > {max_cost_ratio}x baseline")
    return ("rollback", reasons) if reasons else ("promote", [])

base = dict(n=20000, errors=40, bad_feedback=300, guard_blocks=120, p95_ms=2100, cost_per_req=0.0041)
ok   = dict(n=2000, errors=5, bad_feedback=31, guard_blocks=13, p95_ms=2200, cost_per_req=0.0043)
bad  = dict(n=2000, errors=6, bad_feedback=58, guard_blocks=12, p95_ms=2150, cost_per_req=0.0042)
print(canary_decision(base, ok)); print(canary_decision(base, bad))
assert canary_decision(base, ok)[0] == "promote" and canary_decision(base, bad)[0] == "rollback"
assert canary_decision(base, dict(ok, n=100))[0] == "wait"
```

The bad canary is rolled back on thumbs-down rate (2.9% vs 1.5%, z = 4.7) even though its error rate and latency look fine. **Quality regressions in LLM apps rarely show up as errors.**

When you look at the results repeatedly during a rollout, use **sequential tests** (mSPRT, 06.03), or fix the look schedule with alpha spending. **Rollback must be one action**: repoint the alias to the previous bundle digest. The previous bundle's index, prompts, and serving image must therefore still exist (retention ≥ rollback window).

---

## 5. Living with Vendor Models: Deprecation and Drift

| Risk | Control |
|---|---|
| **Scheduled retirement** (providers publish deprecation/retirement dates for snapshots) | **Model inventory with retirement dates** (§9) and an alert at T−90/T−30 days. A migration playbook: candidate bake-off (08.01 Project 1) → prompt re-tune → full gates → canary. Budget ~2–6 engineer-weeks per major model migration |
| **Silent behaviour change** under an alias or "improved" snapshot | Pin dated snapshots. Run a **daily drift canary**: a fixed 200-item eval on the prod bundle, alerting when score or output-length distribution shifts (EWMA/PSI, 06.03). Chen, Zaharia & Zou (2023) documented large month-to-month behaviour shifts in hosted models |
| **Regional/deployment differences** (e.g. Azure OpenAI regional deployments vs the vendor API) | Treat each deployment as a distinct `model` value in the bundle. Run parity evals before failover (08.01 §4) |
| **Rate-limit or quota change** | Capacity SLOs on provider TPM/RPM, multi-deployment routing, batch offload (08.02) |
| **Prompt tuned to one model** | Keep prompt variants per model family in the registry. Evals decide, not intuition |
| **Open-weights model licence change** | Record the licence in the AI-BOM (07.02 §2). Legal review on upgrade |

**Migration pattern (dual-run).** Run old and new snapshots side by side:
1. **Offline paired eval** on the frozen suite.
2. **Shadow diff** on live traffic: semantic diff, length, refusals, tool-call divergence.
3. **Canary.**
4. **Keep the old deployment warm** until the rollback window ends.

---

## 6. Capacity and Autoscaling Through Demand Surges

LLM replicas are **slow to start**: minutes to pull a large image, load weights, and capture CUDA graphs (08.03 §6). They are also **expensive to leave idle**. Naive proportional scaling on a noisy signal **flaps**: it adds replicas that never become ready before they are removed. Three fixes make scaling work:

1. **Hysteresis and cooldowns.** Scale up quickly, scale down slowly and only when load fits comfortably at the smaller size.
2. **Step limits.**
3. **A predictive floor from a known calendar.** For a legislature this comes from the published session calendar: session days, floor-session gavel-in, committee and transmittal deadlines, and bill-introduction cut-offs. Montana's Legislature meets in regular session in **odd-numbered years** (90 legislative days, convening in January), so the next demand season is **January 2027**. Plan capacity, quota increases, and reserved GPUs months ahead.

```python
import math
import numpy as np

class Autoscaler:
    """Concurrency-based replica controller with hysteresis, cooldowns, step limits and a predictive floor."""
    def __init__(self, per_replica_capacity, min_r=1, max_r=20, target_util=0.7,
                 up_cooldown_s=60, down_cooldown_s=600, down_band=0.15, max_step_up=4, floor_fn=None):
        self.cap, self.min_r, self.max_r = per_replica_capacity, min_r, max_r
        self.target, self.band = target_util, down_band
        self.up_cd, self.down_cd, self.max_step = up_cooldown_s, down_cooldown_s, max_step_up
        self.floor_fn = floor_fn or (lambda t: 0)
        self.replicas, self.last_up, self.last_down = min_r, -1e9, -1e9

    def step(self, t, in_flight):
        desired = math.ceil(in_flight / (self.cap * self.target)) if in_flight else self.min_r
        floor = max(self.min_r, self.floor_fn(t))
        desired = min(self.max_r, max(floor, desired))
        r = self.replicas
        if desired > r and t - self.last_up >= self.up_cd:
            self.replicas = min(desired, r + self.max_step); self.last_up = t
        elif desired < r and t - self.last_down >= self.down_cd and t - self.last_up >= self.down_cd:
            if in_flight <= (r - 1) * self.cap * self.target * (1 - self.band) or r - 1 < floor:
                self.replicas = max(desired, r - 1, floor); self.last_down = t
        self.replicas = max(self.replicas, floor)
        return self.replicas

class Naive:
    def __init__(self, cap, target=0.7, max_r=20): self.cap, self.t, self.max_r = cap, target, max_r
    def step(self, t, in_flight): return min(self.max_r, max(1, math.ceil(in_flight / (self.cap * self.t))))

def run(ctrl, load, cap, startup_s=300, dt=15, surge=(8 * 3600, 8.5 * 3600)):
    """Replicas become ready startup_s after being requested; scale-down removes pending ones first."""
    pods = [-1e9]
    events = over = over_surge = gpu_s = 0
    for i, L in enumerate(load):
        t = i * dt
        want = ctrl.step(t, L)
        if want != len(pods):
            events += 1
            pods = pods + [t + startup_s] * (want - len(pods)) if want > len(pods) else sorted(pods)[:want]
        ready = sum(1 for r in pods if r <= t)
        if L > ready * cap:
            over += dt
            over_surge += dt * (surge[0] <= t < surge[1])
        gpu_s += dt * len(pods)
    return dict(scale_events=events, overload_min=over / 60, surge_overload_min=over_surge / 60,
                gpu_hours=round(gpu_s / 3600, 1))

rng = np.random.default_rng(0)
dt = 15
t = np.arange(0, 86400, dt)
load = 20 + 60 * np.exp(-((t / 3600 - 11) ** 2) / 8)                          # daytime hump
in_session = (t / 3600 > 8) & (t / 3600 < 9.5)                                 # floor session 08:00-09:30
load = load + 140 * in_session * np.clip((t / 3600 - 8) * 60, 0, 1)           # gavel-in surge (1-min ramp)
load = np.maximum(0, load + rng.normal(0, 12, len(t)))                        # noisy signal
cap = 16                                                                       # concurrent seqs/replica at SLO
floor_fn = lambda s: 16 if 7.75 <= s / 3600 < 9.75 else 0                    # pre-warm from session calendar

res = {name: run(c, load, cap) for name, c in [("naive", Naive(cap)), ("hysteresis", Autoscaler(cap)),
                                                ("hysteresis+predictive", Autoscaler(cap, floor_fn=floor_fn))]}
for k, v in res.items():
    print(f"{k:22s} {v}")
assert res["hysteresis"]["scale_events"] < res["naive"]["scale_events"] / 3
assert res["hysteresis+predictive"]["surge_overload_min"] < res["hysteresis"]["surge_overload_min"]
```

**Results over one simulated session day** (5-minute replica start-up; surge at the 08:00 gavel):

| Controller | Scale events | Overload (all day) | Overload during surge |
|---|---|---|---|
| Naive | **4,248** | **812 min**; replicas are removed before they ever become ready | 5.75 min |
| Hysteresis | 216 | 10.5 min | 6.0 min, still caught by the start-up delay |
| **+ predictive floor** | 212 | 4.5 min | **0 min** |

Hysteresis costs more GPU-hours than naive (≈159 vs ≈101), but naive's GPU-hours buy nothing because its pods never become ready. Tune `down_cooldown_s` against the idle cost.

**Production translation:**
- Use KEDA/HPA `behavior` (`scaleUp.stabilizationWindowSeconds` small; `scaleDown` large with `policies` limiting pods per period) for hysteresis.
- Implement the predictive floor as a **KEDA cron scaler** that raises `minReplicaCount` for calendar windows, or as a CronJob patching the floor.
- Shorten start-up with pre-pulled images (DaemonSet), weights on a local NVMe cache or a fast PVC, and **warm pool** nodes. Cluster-autoscaler GPU node provisioning often dominates (5–15 min), so keep node headroom during session.
- On provider APIs, "capacity" means **quota**: request TPM increases and provisioned throughput ahead of session, and route overflow across deployments (08.01 §4).
- **Load-shed gracefully.** Define degraded modes such as smaller model, no reranker, shorter answers, or queued "we'll notify you" responses for bulk, and trigger them from queue depth *before* SLOs burn.

---

## 7. SLOs and Burn-Rate Alerting

**SLIs for an LLM application.** Measure at the gateway, per route and tier:

| SLI | Good event definition | Typical SLO |
|---|---|---|
| Availability | Request returns a non-error, non-timeout response | 99.5–99.9% |
| **TTFT** | First token ≤ 1.5 s (interactive) | 95–99% |
| **ITL / streaming smoothness** | p95 inter-token gap ≤ 80 ms within request | 95% |
| End-to-end (agents) | Task completes ≤ N s | 90–95% |
| **Quality SLI** | Online grader pass: groundedness check passes, schema-valid, no guard violation (06.03, 07.01) | 97–99% |
| Cost | Request cost ≤ per-request budget | 99% |

**Error budget:** with target $S$, the budget is $1-S$. The **burn rate** over window $w$ is

$$
\text{burn}(w) \;=\; \frac{\text{bad}(w)/\text{total}(w)}{1 - S},
$$

so burning at rate $b$ exhausts a 30-day budget in $30/b$ days. **Multi-window, multi-burn-rate alerts** (Google SRE Workbook) page when a significant fraction of budget is burning *now*. Each alert requires both a long window (significance) and a short window (still happening, so it resets quickly):

| Severity | Long / short window | Burn threshold | Budget consumed at trigger |
|---|---|---|---|
| Page | 1 h / 5 min | 14.4 | 2% |
| Page | 6 h / 30 min | 6 | 5% |
| Ticket | 3 d / 6 h | 1 | 10% |

```python
import random

def burn_rate(bad, total, slo):
    return (bad / total) / (1 - slo) if total else 0.0

ALERTS = [("page", 3600, 300, 14.4), ("page", 6 * 3600, 1800, 6.0), ("ticket", 3 * 86400, 6 * 3600, 1.0)]

def evaluate_alerts(events, now, slo=0.995):
    """events: list of (t_seconds, is_bad). Fires when BOTH long and short windows exceed the threshold."""
    fired = []
    for sev, long_w, short_w, thr in ALERTS:
        rates = []
        for w in (long_w, short_w):
            win = [b for t, b in events if now - w < t <= now]
            rates.append(burn_rate(sum(win), len(win), slo))
        if min(rates) > thr:
            fired.append((sev, long_w // 60, round(rates[0], 1), round(rates[1], 1)))
    return fired

random.seed(0)
ev = [(t, random.random() < 0.003) for t in range(0, 7 * 3600, 2)]                 # healthy: 0.3% bad
ev += [(t, random.random() < 0.30) for t in range(7 * 3600, 7 * 3600 + 1200, 2)]   # 20-min incident: 30% bad
now = 7 * 3600 + 1200
print("during incident:", evaluate_alerts(ev, now))
assert any(a[0] == "page" for a in evaluate_alerts(ev, now))
assert evaluate_alerts(ev, 7 * 3600) == []
ev += [(t, random.random() < 0.003) for t in range(now, now + 1800, 2)]
print("30 min after recovery:", evaluate_alerts(ev, now + 1800))
assert not any(a[1] == 60 for a in evaluate_alerts(ev, now + 1800))              # the fast page has reset
```

**Results:**
- **Before the incident:** a healthy 0.3% bad rate against a 99.5% SLO (burn 0.6) fires nothing.
- **During the 20-minute incident:** the 1 h/5 min **page** fires (burn 19.7 and 47.0), plus a ticket.
- **30 minutes after recovery:** the page has **auto-resolved** because its short window is clean, while the ticket (slow budget burn) stays open for follow-up.

In Prometheus this is `recording rules` for ratios over each window plus `alert` rules requiring both windows. Use the same pattern for the **quality SLI**, since a quality incident is still an incident.

---

## 8. Incident Management for LLM Systems

| Incident type | Detection | First action (runbook) | Durable fix |
|---|---|---|---|
| **Provider outage/degradation** | Error/latency burn, circuit breakers (08.01 §4) | Failover chain; degrade to smaller/self-hosted model; status banner | Multi-provider parity evals; quota headroom |
| **Quality regression** (bad release, vendor drift) | Quality SLI burn, thumbs-down spike, drift canary | **Roll back bundle digest**; freeze releases | Missing eval slice → add to 06.02 dataset; stricter gate |
| **Prompt-injection exploit / unsafe action** | Guard alerts, anomalous tool-call patterns, taint-policy denials (07.03) | **Kill switch** for the affected tool/capability; revoke tokens; preserve traces | Architectural containment (Rule of Two), new red-team cases |
| **Sensitive-data exposure** | PII detectors on outputs, ACL audit (07.02) | Disable route; purge caches (08.02 §2.1) and logs per policy; notify per data-breach policy | ACL-aware retrieval tests, cache-key tenancy |
| **Cost runaway** (agent loops, retry storms, abuse) | Spend burn rate vs budget, tokens/request anomaly | Per-tenant hard caps; disable offending agent loop; rate-limit | Step/token budgets in agents (05.02), budget alerts (08.02 §4.3) |
| **Capacity exhaustion** (session surge) | `num_requests_waiting`, TTFT burn | Load-shed to degraded modes; raise the floor; burst to provider | Predictive floor, reserved capacity, quota plan (§6) |
| **Index/data staleness** | Freshness SLI (age of newest indexed bill version) | Trigger incremental reindex; banner "data as of …" | CDC pipeline (Kafka) with lag alerts |

**Build these before the incident:**
- **Kill switches:** per tool, per model, and per feature, as feature flags read at request time and audited.
- **Degraded modes** that have themselves been tested.
- **Trace retention** sufficient for forensics: prompts and outputs kept under a redaction policy, with access controlled (06.04).

**Blameless postmortems** for LLM incidents answer:
- which bundle digest;
- which eval slice *should* have caught it;
- which dataset items are added (06.02);
- which gate or alert changes.

---

## 9. Governance, Compliance, and FinOps as Engineering Controls

**AI system inventory** is the backbone. It is one record per production AI use, generated from bundle manifests rather than maintained by hand. Each record holds:
- purpose;
- risk tier;
- owner;
- model(s) with vendor, snapshot, and retirement date;
- data categories;
- users (public or internal);
- decision impact;
- human-review points;
- eval report link;
- incidents;
- cost centre.

**Frameworks and how to operationalise them:**
- **NIST AI RMF 1.0** (NIST AI 100-1) has four functions: **Govern, Map, Measure, Manage**. The **Generative AI Profile** (NIST AI 600-1) lists GenAI-specific risks (confabulation, information integrity, information security, data privacy, harmful bias, value-chain/component integration, and others) with suggested actions. Map each to controls in this curriculum: evals (Measure), guardrails and containment (Manage), inventory and ownership (Govern), and use-case risk assessment (Map).
- **ISO/IEC 42001** (AI management systems) applies if your organisation certifies.
- **Model/system cards** (Mitchell et al.) are generated per bundle: intended use, out-of-scope uses, eval results by slice, known limitations, human-oversight design.

**Montana public-sector requirements.** Montana **HB 178 (2025)** limits state and local government use of AI. The enrolled text:
- **prohibits** use for cognitive behavioural manipulation, for classification causing unlawful discrimination or disparate impact, for malicious purposes, and for surveillance of public spaces (with narrow exceptions);
- **requires disclosure** when a government entity publishes AI-generated material not reviewed by qualified personnel, or operates a public-facing AI interface;
- **requires human review:** when an AI system produces a recommendation or decision that could affect a person's rights, duties, or privileges, a trained human in an appropriate responsible position must review it and be able to reject or modify it.

Engineering translations (confirm the applicability and interpretation with legal counsel):

| Requirement | Control you build |
|---|---|
| Disclosure for public-facing AI and unreviewed AI-generated publications | Bundle flag `public_facing: true` makes a disclosure banner mandatory (a CI check). A publication workflow records a `reviewed_by` field, and unreviewed output carries an "AI-generated" label |
| Human review of rights-affecting recommendations/decisions | Risk-tier routes through HITL approval (05.06). The UI shows sources and uncertainty and allows edit/reject. **Audit log** of reviewer identity (Azure AD), the decision, and any modification |
| Prohibited uses | Use-case intake checklist at the Design stage; the inventory records the determination; guard policies block out-of-scope requests (07.01) |
| Training for reviewers | The inventory links each approver role to a training record; approval UI gated by an Azure AD group |

**Records and retention.** Legislative and public-records obligations may apply to prompts, outputs, and approval logs. Set retention per record class together with records management, and make redaction (07.01) and retention consistent across traces, caches, and backups.

**FinOps:**
- **Showback/chargeback per department and route:** cost attributed per request from traces (08.02 §4.1), rolled up by the inventory's cost centre.
- **Budgets with burn alerts** (same math as §7, with dollars as the "bad events").
- **Unit metrics** such as cost per bill summary and cost per resolved question.
- **Quarterly model re-selection:** rerun 08.01's Pareto analysis. Prices fall fast, so last quarter's optimum is often dominated.

---

## 10. The Learning Loop

```
 prod traces (06.04) ──► sampling: random + guard-flagged + thumbs-down + low-confidence + novel clusters
        │                                        │
        ▼                                        ▼
 online metrics / SLOs                 human triage (labels, root cause: retrieval? prompt? model? tool?)
        │                                        │
        └──────────► dataset versions (06.02) ◄──┘  ── new eval items, red-team cases, regression tests
                               │
                               ▼
                 fixes as bundle changes (§2) → gates (§3) → canary (§4) → prod
```

Measure the loop itself:
- **time from user-reported failure to regression test** (target: days);
- the **share of incidents caught by an existing eval**, which should rise over time;
- **eval-to-online agreement**: do offline wins predict online wins (06.03)?

---

## 11. Production Challenges and Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **Unversioned prompt edits** | "Nothing was deployed" but behaviour changed | Prompt registry + content hashes in the bundle; prod edits blocked; the trace carries the bundle digest |
| **Eval suite drift** | Scores jump when the dataset is edited | Version the suite separately; rebaseline in its own release (`plan_release` rule) |
| **Rollback impossible** | Old index deleted or old snapshot retired | Retain previous bundle artefacts through the rollback window; blue/green indexes |
| **Vendor retirement surprise** | Forced migration in weeks | Inventory with retirement dates, T−90 alerts, dual-run migration playbook |
| **Flapping autoscaler** | Replicas churn; never ready; TTFT spikes | Hysteresis, cooldowns, step limits, predictive floors (§6) |
| **Alert fatigue** | Pages for brief blips; missed slow burns | Multi-window burn-rate alerts (§7); quality SLIs alongside availability |
| **Silent quality incidents** | Errors flat, users unhappy | Quality SLI + canary analysis on feedback and guard metrics (§4) |
| **Shadow traffic side effects** | Duplicate emails or writes from the shadow arm | Stub write tools in shadow; read-only shadow |
| **Governance as paperwork** | Inventory stale; audits painful | Generate the inventory, model cards, and disclosures from bundle manifests and CI |
| **Cost creep** | Spend rises with no traffic growth | Unit-cost dashboards per route; budget burn alerts; quarterly re-selection |

---

## 12. Hands-On Projects

### Project 1 — Bundle-Based Release Pipeline with Automatic Rollback

**User stories**
- *As a developer*, I want to change a prompt, tool, or model through a PR and have the pipeline run exactly the gates that change requires.
- *As the on-call engineer*, I want a single command (or an automatic trigger) to roll back to the previous known-good bundle.

**Acceptance criteria**
- A `bundle.yaml` holds all components of §2. Its digest is stamped on every OTel trace (`app.bundle`).
- CI runs `plan_release` and executes only the required gates. Changing the embedding model without an index rebuild fails CI. Changing the eval suite together with any other component fails CI.
- Rollout proceeds shadow → 5% → 25% → 100% via Argo Rollouts (or a gateway weight). `canary_decision` runs every 10 minutes, and a planted regression (a prompt edit that raises the thumbs-down rate) is **auto-rolled back within 30 minutes**.
- Rolling back restores the previous prompts, index alias, and serving image, verified by traces showing the old digest.

**Step-by-step**
1. Define the bundle schema, then store prompts by hash in Git and build the registry loader.
2. Add digest stamping to the gateway (06.04's tracing middleware).
3. Write `release/plan.py` (`plan_release`) and wire gate jobs to it: smoke, full eval, safety, perf.
4. Configure an Argo Rollouts `AnalysisTemplate` whose metric provider calls your canary-analysis service (`canary_decision` over Prometheus/trace metrics).
5. Game day: ship the planted regression, observe the automatic rollback, and write the postmortem.

### Project 2 — SLOs, Burn-Rate Alerts, and an LLM Incident Program

**User stories**
- *As the service owner*, I want pages only when users are materially affected, including by quality regressions, and runbooks that let any on-call engineer respond.

**Acceptance criteria**
- SLOs are defined for availability, TTFT, ITL, and a **quality SLI** (online groundedness/schema pass rate) for two routes, and error budgets are published.
- Prometheus recording and alert rules implement the three-tier multi-window burn-rate policy (§7). They are validated by replaying synthetic incident traces: a 20-minute outage pages within 5 minutes, and a slow 3-day burn opens a ticket without paging.
- Runbooks exist for the seven incident types in §8. Each has a tested **kill switch** or degraded mode, exercised in a game day.
- Two game days are completed (provider outage and prompt-injection exploit) with blameless postmortems. At least 10 new eval/red-team items are added to 06.02 datasets.

**Step-by-step**
1. Instrument SLIs at the gateway and export good/total counters per route.
2. Generate the burn-rate rules (Sloth or Pyrra can generate them from an SLO spec), then unit-test them with `promtool test rules` using synthetic series.
3. Implement kill switches as feature flags checked per request, with an audit log of every toggle.
4. Write runbooks with exact commands and dashboards, and run the game days.
5. Feed the postmortem action items into datasets and gates.

### Project 3 — Legislative-Session Readiness: Capacity, Inventory, and HB 178 Controls

**User stories**
- *As IT leadership*, I want confidence that AI services will hold their SLOs through the 2027 session's peaks, with documented compliance for government AI use.
- *As a legislator's staffer*, I want to know when content is AI-generated and whether a person reviewed it.

**Acceptance criteria**
- **Capacity plan:**
  - a demand forecast from prior-session traffic (or proxies such as bill-introduction and hearing calendars);
  - a replica/quota plan per route with 30% headroom;
  - the predictive floor wired to the session calendar (KEDA cron scaler);
  - a load test at 1.5× forecast peak passing the SLOs.
- **Degraded modes** are documented and tested. Load shedding engages before TTFT SLO burn exceeds 6×.
- **AI inventory** is generated from bundles for every production AI use. Each entry records purpose, risk tier, data categories, owner, model snapshot and retirement date, human-review points, and eval link.
- **HB 178 controls:**
  - a CI check that `public_facing` bundles render a disclosure;
  - a publication workflow recording `reviewed_by`;
  - HITL approval with an Azure AD group gate and an audit log for rights-affecting recommendations;
  - a use-case intake checklist covering the prohibited uses.

  All of these are reviewed with counsel.
- **Model retirement risk:** every vendor model used during the session is confirmed supported past session end, or migrated beforehand.

**Step-by-step**
1. Pull the prior-session traffic and calendar, fit a demand model (daily and weekly seasonality plus calendar events), and convert it to replicas and quota using 08.03 §3 and 08.04 §5.
2. Configure KEDA (queue-based scaling plus a cron floor) and pre-warm the node pools. Run the 1.5× load test.
3. Build the inventory generator (bundle manifests → inventory records → model cards) and publish it internally.
4. Implement the disclosure and review controls in the UI and API. Add CI checks and audit-log dashboards.
5. Run a tabletop exercise before session: surge, provider outage, and an injection attempt. Sign off with a readiness review.

---

## 13. Foundational Papers, Standards, and Reading (exact titles)

- Sculley et al., 2015 — *Hidden Technical Debt in Machine Learning Systems*
- Breck et al., 2017 — *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction*
- Shankar et al., 2022 — *Operationalizing Machine Learning: An Interview Study*
- Paleyes, Urma & Lawrence, 2022 — *Challenges in Deploying Machine Learning: a Survey of Case Studies*
- Mitchell et al., 2019 — *Model Cards for Model Reporting*
- Gebru et al., 2021 — *Datasheets for Datasets*
- Chen, Zaharia & Zou, 2023 — *How is ChatGPT's behavior changing over time?*
- Shankar et al., 2024 — *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences*
- Beyer et al. (eds.), 2016 — *Site Reliability Engineering: How Google Runs Production Systems*; Beyer et al. (eds.), 2018 — *The Site Reliability Workbook* (ch. "Alerting on SLOs")
- NIST, 2023 — *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (NIST AI 100-1)
- NIST, 2024 — *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile* (NIST AI 600-1)
- ISO/IEC 42001:2023 — *Information technology — Artificial intelligence — Management system*
- Montana Legislature, 2025 — **HB 178**, *An Act Limiting the Use of Artificial Intelligence Systems by State and Local Government…* (enrolled bill, Chapter 427)

## 14. Essential Tooling

| Tool | Use |
|---|---|
| **Argo CD / Flux**, **Argo Rollouts / Flagger** | GitOps promotion of bundle digests; canary with analysis and auto-rollback |
| **KEDA** (queue + cron scalers), HPA `behavior`, Karpenter/cluster-autoscaler | Hysteresis, predictive floors, GPU node provisioning |
| **Prometheus + Alertmanager**, **Sloth / Pyrra / OpenSLO** | SLO specs → multi-window burn-rate rules |
| **Langfuse / Phoenix / LangSmith / Braintrust** (06.04) | Traces with bundle digests, datasets, prompt registry, online evals |
| **OpenFeature / LaunchDarkly / Unleash** | Kill switches, degraded modes, canary flags |
| **MLflow / model registries**, **Azure AI Foundry** deployments | Model/snapshot registry, deployment metadata, retirement tracking |
| **PagerDuty / Opsgenie**, incident tooling | Paging, runbooks, postmortems |
| **OpenCost / Kubecost**, cloud cost exports | GPU and token FinOps, showback |

**Image prompt (lifecycle):** *"Circular lifecycle infographic for an LLM application: six segments — Design (risk tier, SLOs), Build (versioned bundle icon with hash), Evaluate (gates, charts with confidence intervals), Release (shadow → canary → full traffic slider), Operate (SLO gauge, autoscaling servers, alarm bell), Learn (traces flowing into dataset). Centre label 'bundle digest traces every output'. Clean flat vector, muted blues and greens, white background, sans-serif labels."*

**Image prompt (burn-rate alerts):** *"Time-series chart: error-budget remaining (%) over 30 days declining; overlaid short and long window burn-rate lines; red 'PAGE' markers where both 1h and 5m burn exceed 14.4, amber 'TICKET' marker for slow 3-day burn; shaded incident window. Clean technical style, labelled axes."*
