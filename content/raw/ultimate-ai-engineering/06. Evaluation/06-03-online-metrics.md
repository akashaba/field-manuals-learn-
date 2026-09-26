# 06.03 — Online Metrics

> **Module 6: Evaluation** · Subtopic 3 of 4
> **Prerequisites:** 06.01 (offline evals, statistics), 06.02 (datasets), 01.02 §1.1 (latency metrics), 05.06 §7 (human feedback signals), hypothesis testing and variance.
> **Outcome:** you can define the online metrics that tell you whether an LLM feature actually works for users, run trustworthy controlled experiments (sample-ratio checks, variance reduction, valid sequential monitoring, interleaving for ranking changes), and monitor production for drift and quality regressions with statistically sound alerting.

---

## 1. Why Offline Isn't Enough

Offline evals (06.01) measure a system against a fixed dataset. Production adds everything the dataset missed:
- New intents and phrasing.
- Real user behaviour: follow-ups, abandonment, workarounds.
- Interaction effects with the UI.
- Latency tolerance.
- Distribution drift, including new laws, new sessions, and new documents.

**Online metrics close the loop**, and they keep offline evals honest. If offline gains don't move online outcomes, the dataset or the metric is wrong.

---

## 2. A Metric Hierarchy for LLM Features

```
                     ┌──────────────────────────────────────────┐
  NORTH STAR         │ task success: user's goal achieved       │  e.g. "research question resolved without
                     │ (measured by behaviour, not opinions)    │   escalation or re-asking within 24h"
                     └───────────────────┬──────────────────────┘
          ┌──────────────────────────────┼──────────────────────────────┐
  GUARDRAILS (must not regress)          │            DIAGNOSTICS (explain movement)
  • safety/policy violation rate         │            • acceptance / copy / insert rate
  • hallucination rate (judged sample)   │            • edit distance of accepted drafts (05.06 §7)
  • P95 TTFT & E2E latency (01.02)       │            • regenerate / rephrase rate
  • cost per successful task             │            • thumbs up/down rate & comments
  • error / timeout rate                 │            • citation click-through
  • escalation queue SLA (05.06)         │            • abandonment mid-stream
                                         │            • tool error rate, retrieval k-shortfall
```

### 2.1 Implicit vs explicit signals

| Signal | Pros | Pitfalls |
|---|---|---|
| **Thumbs up/down, ratings** | Direct | Tiny response rates (often < 1–5%); selection bias toward extremes; UI placement changes the rates |
| **Acceptance / copy / insert** | Strong behavioural evidence of usefulness | Users may accept, then fix; pair with edit distance |
| **Edit distance on accepted outputs** | Measures residual work | Needs capture of the final artifact |
| **Regenerate / immediate rephrase** | A strong negative signal | Sometimes exploratory; segment by intent |
| **Abandonment during streaming** | Latency or quality dissatisfaction | Confounded with the user getting what they needed early |
| **Follow-up questions** | Could be engagement *or* failure | Classify follow-ups (clarification vs new topic) with a small model |
| **Task-completion events** | Closest to real value | Needs product instrumentation (filed, published, resolved) |

### 2.2 LLM-judged online quality

Sample production traffic (e.g. 1–2%, stratified by intent) and run **validated** judges (06.01 §4) for faithfulness, policy compliance, and answer quality. Redact PII first. Use PPI (06.01 §4.5) with a small, ongoing human-labelled stream to keep the estimates unbiased as the judges drift.

---

## 3. Controlled Experiments (A/B Tests) for LLM Features

### 3.1 Design choices

- **Randomisation unit:** usually the **user** (or account), not the request. LLM features have carry-over effects (learning, trust, conversation state). Analyse at the randomisation unit, or use cluster-robust statistics.
- **Triggering:** analyse users who were actually exposed (e.g. who asked a question), and apply the same trigger condition in both arms.
- **Duration:** full weekly cycles, plus legislative-calendar effects (session vs interim). Watch for **novelty effects**: plot the treatment effect by days since first exposure.
- **Guardrails:** pre-declare the guardrail metrics and their non-inferiority margins (06.01 §6).

### 3.2 Sample-ratio mismatch (SRM): check first, always

If you assigned 50/50 but observe 50.8/49.2 with large n, **the experiment is broken**. Common causes are bots, redirects, crashes in one arm, or caching. Test with a chi-square goodness of fit:

$$
\chi^2 = \sum_{g}\frac{(O_g - E_g)^2}{E_g}
$$

A very small p-value (e.g. < 0.001) means stop and investigate before reading any metric.

### 3.3 Variance reduction with CUPED

Using a pre-experiment covariate $X$ (e.g. the user's task-success rate in the previous 4 weeks) that is independent of the treatment:

$$
Y^{\text{cuped}} = Y - \theta\,(X - \bar X),\qquad \theta = \frac{\mathrm{Cov}(Y, X)}{\mathrm{Var}(X)},\qquad
\mathrm{Var}(\bar Y^{\text{cuped}}) = (1 - \rho^2_{XY})\mathrm{Var}(\bar Y)
$$

With $\rho = 0.5$, the variance drops by 25%, which is equivalent to 33% more traffic.

### 3.4 Peeking and sequential testing

Checking a fixed-horizon test daily and stopping at the first $p < 0.05$ **inflates false positives dramatically**. Either commit to a horizon, or use **always-valid** inference. The **mixture sequential probability ratio test (mSPRT)** for a mean difference with (estimated) variance $\sigma^2$ and mixing variance $\tau^2$ rejects $H_0$ when

$$
\Lambda_n = \sqrt{\frac{\sigma^2}{\sigma^2 + n\tau^2}}\;\exp\!\left(\frac{n^2\tau^2\,\bar d_n^{\,2}}{2\sigma^2(\sigma^2 + n\tau^2)}\right) \;\ge\; \frac{1}{\alpha}
$$

Here $\bar d_n$ is the running mean of per-pair differences and $\sigma^2$ their variance. You can monitor it continuously without inflating the error rate.

```python
import math
import random
from statistics import NormalDist

import numpy as np
from scipy import stats


def srm_check(observed: list[int], expected_ratio: list[float], threshold: float = 1e-3) -> dict:
    total = sum(observed)
    expected = [total * r / sum(expected_ratio) for r in expected_ratio]
    chi2, p = stats.chisquare(observed, expected)
    return {"chi2": round(float(chi2), 2), "p": float(p), "srm": bool(p < threshold)}


def two_prop_diff(k_a: int, n_a: int, k_b: int, n_b: int, conf: float = 0.95) -> dict:
    pa, pb = k_a / n_a, k_b / n_b
    se = math.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    diff = pb - pa
    pooled = (k_a + k_b) / (n_a + n_b)
    z_stat = diff / math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    return {"diff": diff, "ci": (diff - z * se, diff + z * se), "p": 2 * (1 - NormalDist().cdf(abs(z_stat)))}


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


def msprt_lambda(diffs: np.ndarray, sigma2: float, tau2: float) -> float:
    n, dbar = len(diffs), float(np.mean(diffs))
    return math.sqrt(sigma2 / (sigma2 + n * tau2)) * math.exp(
        (n * n * tau2 * dbar * dbar) / (2 * sigma2 * (sigma2 + n * tau2)))
```

### 3.5 Interleaving for retrieval and ranking changes

To compare two rankers (e.g. a new reranker from 04.03), **interleaving** mixes both result lists into one list per query, then credits whichever ranker contributed the clicked or cited items. Because every user acts as their own control, it needs far less traffic than an A/B test (Chapelle et al., 2012). **Team-draft interleaving:** the rankers take turns (a coin flip decides who picks first each round), each adding its highest-ranked item that isn't already in the list.

For RAG, "clicks" can be **citations the generator actually used**, or source links users open.

```python
def team_draft_interleave(a: list[str], b: list[str], k: int, rng: random.Random) -> tuple[list[str], dict]:
    out, team, ia, ib = [], {}, 0, 0
    while len(out) < k and (ia < len(a) or ib < len(b)):
        first = "A" if rng.random() < 0.5 else "B"
        for who in (first, "B" if first == "A" else "A"):
            src, idx = (a, ia) if who == "A" else (b, ib)
            while idx < len(src) and src[idx] in team:
                idx += 1
            if idx < len(src) and len(out) < k:
                out.append(src[idx])
                team[src[idx]] = who
                idx += 1
            if who == "A":
                ia = idx
            else:
                ib = idx
    return out, team


def interleaving_winner(team: dict[str, str], clicked: list[str]) -> str:
    ca = sum(team.get(d) == "A" for d in clicked)
    cb = sum(team.get(d) == "B" for d in clicked)
    return "A" if ca > cb else "B" if cb > ca else "tie"
```

### 3.6 Adaptive allocation (bandits) — with care

Thompson sampling over prompt or model variants can shift traffic toward better variants during the test. It suits **low-risk, short-feedback** decisions (e.g. which suggestion template gets accepted). Beware of delayed and noisy rewards, and of losing clean causal estimates. Keep a fixed-allocation holdout when you need an unbiased effect size.

```python
def thompson_choose(successes: dict[str, int], failures: dict[str, int], rng: np.random.Generator) -> str:
    draws = {v: rng.beta(successes[v] + 1, failures[v] + 1) for v in successes}
    return max(draws, key=draws.get)
```

---

## 4. Shadow, Canary, and Staged Rollouts

| Stage | What | Measures |
|---|---|---|
| **Shadow** | The new version runs on live inputs, but outputs aren't shown | Output diffs, judge scores, latency and cost — no user risk |
| **Canary** | 1–5% of traffic, with automatic rollback | Guardrails (errors, latency, safety), SRM |
| **A/B** | A randomised experiment at meaningful traffic | North star + guardrails with CIs |
| **Ramp** | 25% → 50% → 100%, with monitoring | Stability at scale (cost, rate limits, cache hit rates) |

**Automatic rollback triggers** should be guardrail breaches with statistical backing (e.g. an EWMA or a sequential test on the error rate), not single spikes.

---

## 5. Drift and Quality Monitoring

### 5.1 Input drift

The **Population Stability Index** compares a current distribution $q$ against a reference $p$ over bins (intents, document types, embedding clusters):

$$
\text{PSI} = \sum_{b}(q_b - p_b)\ln\frac{q_b}{p_b}
$$

A common rule of thumb reads < 0.1 as stable, 0.1–0.25 as moderate shift, and > 0.25 as major shift. For free text, bin by **intent classifier labels** or **embedding k-means clusters**, and also track the distance between embedding centroids and the rate of new clusters.

### 5.2 Output and quality drift

Track these over time:
- Response length.
- Refusal rate.
- Citation rate.
- Abstention rate ("not found in sources").
- Tool error rate.
- Judge scores on the sampled traffic.
- **Cost and latency per successful task.**

Provider model updates, index changes, and prompt edits all show up here first.

### 5.3 Alerting without alert fatigue

Use **control charts** rather than static thresholds. The **EWMA** chart smooths the metric, $z_t = \lambda x_t + (1-\lambda)z_{t-1}$, with control limits

$$
\mu_0 \pm L\,\sigma\sqrt{\frac{\lambda}{2-\lambda}\big(1-(1-\lambda)^{2t}\big)}
$$

It detects small, persistent shifts (e.g. a 2-point drop in faithfulness) faster than Shewhart charts, with fewer false alarms from single spikes.

```python
def psi(ref_counts: dict[str, int], cur_counts: dict[str, int], eps: float = 1e-6) -> float:
    bins = set(ref_counts) | set(cur_counts)
    rt, ct = sum(ref_counts.values()), sum(cur_counts.values())
    total = 0.0
    for b in bins:
        p = max(ref_counts.get(b, 0) / rt, eps)
        q = max(cur_counts.get(b, 0) / ct, eps)
        total += (q - p) * math.log(q / p)
    return total


def ewma_alerts(xs: list[float], mu0: float, sigma: float, lam: float = 0.2, L: float = 3.0) -> list[int]:
    z, alerts = mu0, []
    for t, x in enumerate(xs, start=1):
        z = lam * x + (1 - lam) * z
        width = L * sigma * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
        if abs(z - mu0) > width:
            alerts.append(t)
    return alerts
```

---

## 6. Pairwise Preference at Scale (Arena-Style)

When comparing many variants on open-ended quality, collect **pairwise preferences** (from users, experts, or validated judges) and fit a **Bradley–Terry** model:

$$
P(i \succ j) = \sigma(\beta_i - \beta_j)
$$

This gives a ranking with confidence intervals (via bootstrap), as in Chatbot Arena. Control for style confounders (e.g. length, formatting) as covariates, because preferences are known to reward verbosity (06.01 §4.1).

---

## 7. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Offline–online mismatch** | Offline wins don't move users | Revisit the dataset representativeness (06.02) and the north-star definition; online judged sampling |
| **Sparse explicit feedback** | Few thumbs; biased | Behavioural signals (acceptance, edits, regenerate); judged sampling with PPI |
| **Broken experiments** | Implausible results | SRM checks first; A/A tests; consistent triggering |
| **Peeking** | "Significant" results that don't replicate | Fixed horizons, or mSPRT / always-valid p-values |
| **Low power** | Inconclusive tests | CUPED; user-level metrics with less variance; interleaving for ranking changes; longer runs |
| **Novelty effects** | Early gains fade | Effect by exposure day; longer tests; holdouts |
| **Silent drift** | Quality erodes over weeks | PSI on intents and clusters; EWMA on judged quality; version annotations on dashboards |
| **Alert fatigue** | On-call ignores alerts | Control charts; alert on guardrails with statistical backing; runbooks |

---

## 8. Hands-On Projects

### Project 1 — Online Metrics and Feedback Instrumentation

**User stories**
- *As a product owner*, I want a live dashboard that tells me whether the legislative assistant helps users complete their research, not just whether they clicked thumbs up.

**Acceptance criteria**
1. A metric spec (north star, guardrails, diagnostics) with precise definitions, event schemas, and owners.
2. Front-end and back-end instrumentation for: answer shown, stream abandoned, copy/insert, regenerate, rephrase within 60 s (classified by a small model), citation clicked, thumbs + comment, and task-completion events (e.g. memo filed). Events are joined to trace IDs (06.04).
3. A 1–2% stratified judged-quality sample (faithfulness, policy) with PPI correction from a weekly human-labelled batch.
4. Dashboards per intent and release, with version annotations; the latency, cost, and success-per-cost panels.
5. A validation study: correlate the diagnostics with the north star at user-week level, and drop or keep metrics based on the evidence.

**Step-by-step**
1. Write the metric spec and event schemas (a JSON Schema per event).
2. Implement the instrumentation (React front-end events, back-end OpenTelemetry spans) into an event pipeline (Kafka → warehouse).
3. Build the judged-sampling job with redaction and PPI.
4. Build the dashboards (Grafana, Superset, or Metabase).
5. Run the metric validation analysis, and prune the metrics.

---

### Project 2 — Experimentation Framework for LLM Features

**User stories**
- *As an ML team*, we want to A/B test prompt, model, and retrieval changes with trustworthy statistics and automatic guardrail protection.

**Acceptance criteria**
1. User-level deterministic assignment (hashing with an experiment salt), exposure logging, and trigger definitions.
2. The analysis library covers: SRM check (`srm_check`), two-proportion and mean differences with CIs, CUPED (`cuped`), cluster-robust SEs, Holm correction for the secondary metrics, and mSPRT monitoring (`msprt_lambda`) for guardrails.
3. **Simulation validation:** A/A tests (false-positive rate ≈ α); peeking vs mSPRT comparison showing the naive FPR inflation; CUPED variance reduction measured on synthetic data with known correlation.
4. Interleaving support for retrieval and reranking experiments, with a power comparison against A/B on the same simulated effect.
5. An experiment report template (hypothesis, primary metric, guardrails, results with CIs, decision), used for one real or realistic experiment.

**Step-by-step**
1. Implement the assignment service and exposure events.
2. Implement the statistics library; unit-test it against scipy/statsmodels.
3. Build the simulation harness for A/A tests, peeking, CUPED, and interleaving power.
4. Integrate with the dashboards; add automatic guardrail alerts.
5. Run an experiment (e.g. new reranker vs old), and write the report.

---

### Project 3 — Drift and Quality Monitoring with Incident Runbooks

**User stories**
- *As the on-call engineer*, I want to be alerted when input mix, output behaviour, or judged quality shifts meaningfully, with enough context to act.

**Acceptance criteria**
1. Daily PSI over intent labels and embedding clusters (k-means fitted on a reference month), plus new-cluster detection with example queries.
2. EWMA charts (`ewma_alerts`) for judged faithfulness, refusal rate, abstention rate, tool error rate, P95 latency, and cost per task. The control-limit parameters are tuned on historical data for a target false-alarm rate.
3. Alerts include the drifting slices, example traces (06.04), and recent change annotations (model, prompt, index, dataset versions).
4. Runbooks for the top 5 alert types, with diagnosis steps and rollback criteria.
5. A backtest: replay 3 months of historical metrics with injected drifts (e.g. a provider model update, an index outage) and report the detection delay and false alarms.

**Step-by-step**
1. Build the reference distributions and clustering; implement PSI and new-cluster detection.
2. Implement the EWMA monitors; tune λ and L on history.
3. Wire the alerts to the paging and chat tools with context payloads.
4. Write the runbooks.
5. Run the backtest with synthetic drift injection, and tune.

---

## 9. Foundational Papers & Reading (exact titles)

**Experimentation**
- Kohavi, Tang & Xu, 2020 — *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing* (book)
- Deng et al., 2013 — *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED)
- Fabijan et al., 2019 — *Diagnosing Sample Ratio Mismatch in Online Controlled Experiments: A Taxonomy and Rules of Thumb for Practitioners*
- Johari et al., 2017 — *Peeking at A/B Tests: Why it matters, and what to do about it*
- Howard et al., 2021 — *Time-uniform, nonparametric, nonasymptotic confidence sequences*

**Interleaving and click evaluation**
- Radlinski, Kurup & Joachims, 2008 — *How Does Clickthrough Data Reflect Retrieval Quality?*
- Chapelle et al., 2012 — *Large-Scale Validation and Analysis of Interleaved Search Evaluation*

**Bandits and preferences**
- Russo et al., 2018 — *A Tutorial on Thompson Sampling*
- Bradley & Terry, 1952 — *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*
- Chiang et al., 2024 — *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*

**Monitoring**
- Roberts, 1959 — *Control Chart Tests Based on Geometric Moving Averages* (EWMA)
- Rabanser, Günnemann & Lipton, 2019 — *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **GrowthBook / Statsig / Eppo / Optimizely** (or in-house) | Assignment, exposure, experiment analysis |
| **scipy / statsmodels** | Tests, cluster-robust SEs |
| **Kafka + ClickHouse / BigQuery / Snowflake** | Event pipelines and metric computation |
| **Grafana / Superset / Metabase** | Dashboards and alerts |
| **Evidently / NannyML / whylogs** | Drift monitoring |
| **Langfuse / Phoenix / LangSmith** | Online judged evaluation on traces |
| **Prometheus + Alertmanager** | Guardrail alerting |
