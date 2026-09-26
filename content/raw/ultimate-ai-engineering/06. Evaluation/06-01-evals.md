# 06.01 — Evals

> **Module 6: Evaluation** · Subtopic 1 of 4
> **Prerequisites:** 02.01 §7 (paired tests for prompts), 03.04 §8 and 04.03 §5 (relevance judging, calibration), 05.02 §8 (pass^k, trajectory checks), basic statistics (binomial, bootstrap, hypothesis tests).
> **Outcome:** you can design an evaluation program for any LLM system: what to measure, with which graders, how many examples, and how to report results with honest uncertainty. You can build and validate LLM judges, correct for judge error, combine small human labels with large judged sets (prediction-powered inference), and gate releases on evals in CI.

> **Consolidation note:** earlier modules introduced evaluation pieces in context (prompt A/B tests, retrieval metrics, reranker calibration, agent pass^k). This subtopic is the **general framework and statistics** that all of them share.

---

## 1. What an Eval Is (and What It's For)

An **eval** is a function from (system version, dataset) to a set of scores with uncertainty, computed reproducibly:

$$
\mathrm{Eval}(\text{system}_v, D) = \Big\{ \hat\mu_m \pm \text{CI}_m \Big\}_{m \in \text{metrics}},\qquad \hat\mu_m = \frac{1}{|D|}\sum_{x \in D} g_m\big(x,\ \text{system}_v(x)\big)
$$

Here $g_m$ is a **grader**. The purposes differ, and so do the designs:

| Eval type | Question | Dataset | Cadence |
|---|---|---|---|
| **Capability** | Can the system do X at all? How well? | Hard, diverse, growing | Model or architecture changes |
| **Regression** | Did this change break anything? | Stable golden set + past-bug cases | Every PR / release (CI gate) |
| **Safety / policy** | Does it refuse what it must, and not over-refuse? | Adversarial + borderline sets | Every release + red-team cycles |
| **Component** | Is retrieval / reranking / tool selection working? | Stage-specific labels | When that component changes |
| **End-to-end** | Does the user get a correct, grounded, useful outcome? | Realistic tasks with end states | Every release |
| **Online** | Does it work for real users? | Live traffic (06.03) | Continuous |

**Principle:** evals exist to support **decisions** (ship or don't, choose A or B, where to invest). Design each eval backwards from the decision it informs, and from the smallest difference that would change that decision.

---

## 2. Graders

| Grader | How | Strengths | Weaknesses |
|---|---|---|---|
| **Deterministic assertions** | Exact or normalised match, regex, JSON Schema, numeric tolerance, "contains citation X", unit tests on code, database end-state checks | Cheap, reproducible, unambiguous | Only for checkable properties |
| **Reference-based metrics** | Field F1 (02.03 §7), token or span overlap (04.01 §6), execution accuracy (SQL) | Objective against gold | Needs gold labels; brittle for free text |
| **Model-graded (LLM judge)** | Rubric scoring, pairwise preference, reference-guided grading | Scales to open-ended outputs | Biases; must be validated (§4) |
| **Human** | Expert rating or preference | Ground truth for subjective quality | Slow, costly, inconsistent without guidelines |
| **Trajectory checks** | Required and forbidden tool calls, step efficiency (05.02 §8.2) | Catches unsafe or wasteful paths | Needs structured traces (06.04) |

**Decompose quality into checkable parts wherever possible.** "Is this bill summary good?" becomes:
- Every cited section exists → deterministic.
- The effective date matches the source → deterministic.
- No unsupported claims → judge, per sentence.
- Plain-language readability → judge + formula.

Each part becomes a separate metric, so a regression points to its cause.

```python
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class EvalCase:
    id: str
    input: Any
    expected: Any = None
    slice: str = "all"
    cluster: str | None = None          # e.g. source document — for clustered SEs (§3.2)
    meta: dict = field(default_factory=dict)


Grader = Callable[[EvalCase, Any], float]          # returns a score in [0, 1]


def exact(case: EvalCase, out: Any) -> float:
    norm = lambda s: " ".join(str(s).lower().split())
    return float(norm(out) == norm(case.expected))


def regex_all(*patterns: str) -> Grader:
    return lambda case, out: float(all(re.search(p, str(out), re.I) for p in patterns))


def numeric_close(tol: float) -> Grader:
    def g(case, out):
        try:
            return float(abs(float(out) - float(case.expected)) <= tol)
        except (TypeError, ValueError):
            return 0.0
    return g


def cites_only_known(known_ids: set[str]) -> Grader:
    def g(case, out):
        cited = set(re.findall(r"\[([A-Z]{1,3}-\d+(?::s\d+)?|\d+-\d+-\d+)\]", str(out)))
        return float(bool(cited) and cited <= known_ids)
    return g


def run_eval(system: Callable[[Any], Any], cases: list[EvalCase], graders: dict[str, Grader],
             cache: dict | None = None, system_version: str = "v0") -> list[dict]:
    """Returns one row per (case, metric). Caches outputs by (system_version, case.id)."""
    cache = {} if cache is None else cache
    rows = []
    for c in cases:
        key = (system_version, c.id)
        if key not in cache:
            cache[key] = system(c.input)
        out = cache[key]
        for name, g in graders.items():
            rows.append({"case": c.id, "slice": c.slice, "cluster": c.cluster or c.id,
                         "metric": name, "score": g(c, out)})
    return rows
```

---

## 3. Statistics: Reporting Uncertainty Honestly

### 3.1 Confidence intervals for rates

For a pass rate $\hat p = k/n$, prefer the **Wilson interval** over the normal approximation, which fails near 0 or 1 and for small $n$:

$$
\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \;\pm\; \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}
$$

### 3.2 Clustered data

Eval items often come in **clusters**: several questions about the same document, several turns of one conversation, several paraphrases of one intent. The items are then correlated, and naive standard errors are **too small** (Miller, 2024). Use cluster-robust standard errors:

$$
\mathrm{SE}_{\text{clustered}}(\bar x) = \frac{1}{n}\sqrt{\sum_{c=1}^{C}\Big(\sum_{i \in c}(x_i - \bar x)\Big)^2}
$$

Alternatively, bootstrap by **resampling clusters**, not items.

### 3.3 Comparing two systems

Use **paired** designs: the same items, both systems (02.01 §7.2 covers McNemar's test and the paired bootstrap). Report the **difference with a confidence interval**, not just two scores.

### 3.4 How many examples? (power analysis)

For a paired binary comparison, let $\pi_d$ be the expected fraction of discordant items (one system right, the other wrong) and $\delta$ the smallest difference in accuracy you care about. The required number of items is approximately:

$$
n \approx \frac{\Big(z_{1-\alpha/2}\sqrt{\pi_d} + z_{1-\beta}\sqrt{\pi_d - \delta^2}\Big)^2}{\delta^2}
$$

*Example:* $\pi_d = 0.15$, $\delta = 0.03$, $\alpha = 0.05$, power 0.8 gives $n \approx 1{,}300$. **Most "we improved 2 points on 200 examples" claims are noise.** Either invest in larger sets, or make decisions that are only sensitive to larger effects.

### 3.5 Non-determinism and multiple comparisons

- **Run-to-run variance:** sampling (temperature > 0), and even temperature-0 non-determinism (01.02 §10), change the scores. Run $k$ repetitions and report the mean ± the standard deviation across runs. For agents, report pass^k (05.02 §8).
- **Multiple comparisons:** testing 20 metrics × 5 slices produces "significant" noise. Pre-register the primary metric, and correct the rest with **Holm** (family-wise error) or **Benjamini–Hochberg** (false discovery rate).

```python
import math
import random
from collections import defaultdict
from statistics import NormalDist


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def clustered_se(scores: list[float], clusters: list[str]) -> float:
    n, mean = len(scores), sum(scores) / len(scores)
    sums = defaultdict(float)
    for s, c in zip(scores, clusters):
        sums[c] += s - mean
    return math.sqrt(sum(v * v for v in sums.values())) / n


def cluster_bootstrap_ci(scores, clusters, iters=5000, conf=0.95, seed=0):
    by = defaultdict(list)
    for s, c in zip(scores, clusters):
        by[c].append(s)
    keys, rng, means = list(by), random.Random(seed), []
    for _ in range(iters):
        sample = [x for k in (rng.choice(keys) for _ in keys) for x in by[k]]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = int((1 - conf) / 2 * iters)
    return means[lo], means[iters - lo - 1]


def paired_sample_size(p_discordant: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    return math.ceil((za * math.sqrt(p_discordant) + zb * math.sqrt(p_discordant - delta ** 2)) ** 2 / delta ** 2)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Returns name -> rejected (significant) under Holm's step-down procedure."""
    order = sorted(pvals, key=pvals.get)
    m, result, still = len(order), {}, True
    for i, name in enumerate(order):
        still = still and pvals[name] <= alpha / (m - i)
        result[name] = still
    return result
```

---

## 4. LLM-as-Judge, Done Rigorously

### 4.1 Known biases

| Bias | Description | Mitigation |
|---|---|---|
| **Position bias** | In pairwise mode, prefers the first (or second) answer | Judge both orders; count only consistent wins (04.03 §4.2) |
| **Verbosity bias** | Prefers longer answers | Rubrics that penalise padding; length-controlled comparisons; report length alongside |
| **Self-preference** | Prefers outputs from its own model family | Use a different judge family, or multiple judges |
| **Leniency / scale drift** | Scores bunch at the top; drift across judge versions | Anchored rubrics with examples per score level; pin the judge version |
| **Reference anchoring** | Over-weights surface similarity to the reference | Grade against criteria, with the reference as evidence only |

### 4.2 Judge design checklist

- **Narrow criteria:** one judge per criterion (faithfulness, completeness, tone), not "overall quality".
- **Binary or low-cardinality scales** with **anchored definitions** ("2 = every claim supported by a cited source; 1 = …").
- **Reasoning before the verdict** (02.03 §3.1), with structured output.
- **Provide the evidence:** sources, reference answers, and the rubric. For faithfulness, judge each claim against the source rather than the whole answer.
- **Validate** against human labels before trusting the judge, and again whenever the judge model or prompt changes.

### 4.3 Validating a judge

On a human-labelled set (typically 100–300 items), measure agreement:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $p_o$ is the observed agreement and $p_e$ the chance agreement. Also report the **judge's sensitivity and specificity**, treating human labels as truth. Compare judge–human agreement with **human–human** agreement: a judge rarely needs to be better than your annotators agree with each other.

### 4.4 Correcting pass rates for judge error

If the judge has sensitivity $se$ (true passes judged as passes) and specificity $sp$, the raw judged pass rate $\hat q$ is biased. The **Rogan–Gladen** correction estimates the true rate:

$$
\hat\pi = \frac{\hat q + sp - 1}{se + sp - 1}
$$

This matters when comparing systems whose failure modes the judge detects unevenly.

### 4.5 Prediction-powered inference (PPI): many judged items + few human labels

Suppose $n$ items have both human labels $Y$ and judge scores $f(X)$, and $N \gg n$ items have only judge scores. **PPI** (Angelopoulos et al., 2023) combines them into an unbiased estimate with a tighter interval than humans alone:

$$
\hat\theta_{\text{PP}} = \underbrace{\frac{1}{N}\sum_{j=1}^{N} f(\tilde X_j)}_{\text{judge on large set}} + \underbrace{\frac{1}{n}\sum_{i=1}^{n}\big(Y_i - f(X_i)\big)}_{\text{bias correction ("rectifier")}},
\qquad
\widehat{\mathrm{Var}} = \frac{\sigma^2_{f}}{N} + \frac{\sigma^2_{Y-f}}{n}
$$

The better the judge, the smaller $\sigma^2_{Y-f}$, and the more you gain. PPI never trusts the judge blindly: its bias is measured and removed.

```python
def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    labels = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def judge_sens_spec(human: list[int], judge: list[int]) -> tuple[float, float]:
    tp = sum(h == 1 and j == 1 for h, j in zip(human, judge))
    tn = sum(h == 0 and j == 0 for h, j in zip(human, judge))
    pos, neg = sum(human), len(human) - sum(human)
    return tp / pos, tn / neg


def rogan_gladen(judged_rate: float, se: float, sp: float) -> float:
    return min(1.0, max(0.0, (judged_rate + sp - 1) / (se + sp - 1)))


def ppi_mean(y_labeled: list[float], f_labeled: list[float], f_unlabeled: list[float], conf=0.95):
    n, N = len(y_labeled), len(f_unlabeled)
    rect = [y - f for y, f in zip(y_labeled, f_labeled)]
    mean_f = sum(f_unlabeled) / N
    mean_r = sum(rect) / n
    var = lambda xs: sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)
    se = math.sqrt(var(f_unlabeled) / N + var(rect) / n)
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    est = float(mean_f + mean_r)
    return est, (float(est - z * se), float(est + z * se))
```

---

## 5. Evaluating Specific System Types (Pointers)

| System | Core metrics | Where covered |
|---|---|---|
| Prompted classifier / extractor | Accuracy, macro-F1, field F1, hallucination and omission rates | 02.01 §7, 02.03 §7 |
| Retrieval | Recall@k, nDCG@10, evidence recall @ budget | 03.01 §4, 03.04 §8, 04.01 §6, 04.02 §8 |
| RAG answers | Correctness, faithfulness per claim, citation validity, abstention precision/recall | 04.02 §1, 04.03 Project 3 |
| Agents | Task success on end state, pass^k, trajectory rules, cost | 05.02 §8 |
| Safety | Attack success rate, over-refusal rate, policy-violation rate | 02.01 §6, 05.03/05.04 red teams |
| Generation quality | Pairwise preference (judge + human), Elo/Bradley-Terry across variants | §4, 06.03 |

---

## 6. Evals in the Development Loop

```
  error analysis of traces (06.04) ──► new failure category ──► add cases to dataset (06.02)
          ▲                                                              │
          │                                                              ▼
  online metrics (06.03) ◄── ship ◄── CI gate: primary metric ≥ baseline − margin (paired CI)
                                         + no slice regression beyond tolerance
                                         + safety metrics non-inferior
```

**CI gating rules that work:**
- **Non-inferiority, not "any drop fails".** Fail the build if the lower bound of the paired difference's confidence interval is below $-\Delta$, where $\Delta$ is the tolerated margin. This avoids flaky failures from noise.
- **Separate fast and full suites:** a deterministic smoke suite on every PR (minutes), and a full judged suite nightly or pre-release.
- **Cache system outputs** by (system version, case) so that re-grading is cheap. **Pin judge versions.**
- **Every production bug becomes an eval case** before the fix is merged.

```python
def non_inferiority_gate(diff_ci: tuple[float, float], margin: float) -> dict:
    lo, hi = diff_ci
    return {"pass": lo >= -margin, "improved": lo > 0,
            "reason": "ok" if lo >= -margin else f"CI lower bound {lo:.3f} < -{margin}"}
```

---

## 7. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Vibes-based releases** | Regressions discovered by users | Golden set + CI gate; every bug becomes a case |
| **Noise mistaken for progress** | Flip-flopping "improvements" | Paired tests; CIs; power analysis; repeated runs |
| **Correlated items** | Overconfident CIs | Clustered SEs / cluster bootstrap |
| **Untrusted judges** | Scores disagree with experts | Narrow rubrics; validation (κ, sens/spec); Rogan–Gladen; PPI |
| **Judge drift** | Scores shift without system changes | Pin judge versions; re-validate on change; anchor examples |
| **Metric gaming** | The metric improves, users don't | Multiple complementary metrics; human spot checks; online metrics (06.03) |
| **Eval cost** | Full suites too slow or expensive | Output caching; tiered suites; compact subsets via IRT (06.02) |
| **Overfitting to the eval set** | Great offline, weak in production | Hidden holdouts; periodic refresh from production (06.02) |

---

## 8. Hands-On Projects

### Project 1 — Evaluation Framework and CI Gate for a Legislative Assistant

**User stories**
- *As a team shipping a RAG + agent assistant*, we want every change evaluated on correctness, grounding, safety, and cost, with a build gate that only fails for real regressions.

**Acceptance criteria**
1. The framework (`EvalCase`, graders, `run_eval`, output caching) supports deterministic graders, reference metrics, LLM judges (structured outputs), and trajectory checks.
2. Suites: a smoke suite (≥ 50 deterministic cases, < 5 minutes) and a full suite (≥ 500 cases across ≥ 6 slices, including safety and unanswerable questions).
3. The report shows Wilson CIs per metric and slice, clustered SEs where items share documents, paired differences vs baseline with CIs, and Holm-corrected secondary metrics.
4. The CI gate uses `non_inferiority_gate` on the primary metric and safety metrics, with slice tolerances. The report is posted to PRs.
5. A process: every production incident gets an eval case, linked to its trace (06.04).

**Step-by-step**
1. Define the metric tree (primary, secondary, guardrails), and write the graders.
2. Build the runner with concurrency, retries, and SQLite output caching keyed by system version.
3. Implement the statistics module (§3) and HTML/Markdown reporting.
4. Wire the smoke suite into PR CI and the full suite into nightly and pre-release pipelines.
5. Backfill cases from past bugs; document the process.

---

### Project 2 — Judge Validation, Bias Study, and PPI Estimation

**User stories**
- *As an evaluation lead*, I need to know how far we can trust our LLM judges, and I want estimates that stay unbiased even when the judges aren't perfect.

**Acceptance criteria**
1. Human-labels ≥ 300 outputs (two annotators on 100 for human–human κ) for faithfulness (binary, per answer) and helpfulness (pairwise).
2. Judges: ≥ 2 judge models × 2 prompt designs (holistic vs claim-level faithfulness). Reports κ, sensitivity/specificity, and agreement relative to human–human κ.
3. Bias measurements: position bias (swap consistency), verbosity bias (preference vs length difference), and self-preference (judging outputs from its own family vs others).
4. On 5,000 unlabelled outputs, compares the estimates of the pass rate: raw judge, Rogan–Gladen corrected, human-only (300), and PPI. Reports the CI widths and the effective sample-size gain.
5. Recommendations: the judge configuration and the recalibration cadence.

**Step-by-step**
1. Sample outputs across two system versions; write the labelling guidelines; label them.
2. Implement the judge prompts with structured outputs; run all configurations.
3. Compute the agreement metrics and the bias diagnostics.
4. Implement `rogan_gladen` and `ppi_mean`; compare the estimates and CIs.
5. Write the report.

---

### Project 3 — Statistical Rigor Audit of Existing Evals

**User stories**
- *As an engineering manager*, I want to know which past "improvements" were real, and how big our eval sets need to be for future decisions.

**Acceptance criteria**
1. Collects ≥ 10 past eval comparisons (from your team, or reproduced from public model cards / papers with per-item data).
2. For each, recomputes: Wilson CIs, paired tests, clustered SEs where applicable, and multiple-comparison corrections. Classifies each as supported / not supported / inconclusive.
3. Runs a variance study: the same system run 10× at production settings, quantifying run-to-run SD per metric.
4. A power-analysis table: the required n for detecting 1, 2, 3, and 5-point changes at your observed discordance rates (`paired_sample_size`).
5. An eval policy document: minimum n, primary-metric pre-registration, repetition counts, reporting template.

**Step-by-step**
1. Gather the per-item results (or re-run the systems) for the historical comparisons.
2. Apply the §3 toolkit; tabulate the conclusions.
3. Run the repeated-run variance study.
4. Compute the power tables.
5. Write and socialise the policy.

---

## 9. Foundational Papers & Reading (exact titles)

**Statistics for evals**
- Miller, 2024 — *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*
- Dror et al., 2018 — *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*
- Card et al., 2020 — *With Little Power Comes Great Responsibility*
- Wilson, 1927 — *Probable Inference, the Law of Succession, and Statistical Inference*
- Holm, 1979 — *A Simple Sequentially Rejective Multiple Test Procedure*
- Benjamini & Hochberg, 1995 — *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*
- Angelopoulos et al., 2023 — *Prediction-Powered Inference*
- Rogan & Gladen, 1978 — *Estimating Prevalence from the Results of a Screening Test*

**LLM judges**
- Zheng et al., 2023 — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
- Liu et al., 2023 — *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*
- Kim et al., 2023 — *Prometheus: Inducing Fine-grained Evaluation Capability in Language Models*
- Panickssery, Bowman & Feng, 2024 — *LLM Evaluators Recognize and Favor Their Own Generations*
- Dubois et al., 2024 — *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*
- Min et al., 2023 — *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*
- Shankar et al., 2024 — *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences*

**Benchmarks and methodology**
- Liang et al., 2022 — *Holistic Evaluation of Language Models* (HELM)
- Ribeiro et al., 2020 — *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **Inspect AI** | Eval framework: tasks, solvers, scorers, agents, sandboxes |
| **promptfoo** | Prompt/app regression testing with assertions and CI integration |
| **OpenAI Evals / lm-evaluation-harness / HELM** | Benchmark-style evaluation |
| **DeepEval, RAGAS** | RAG and LLM-app metrics (validate the judges) |
| **Langfuse / Phoenix / LangSmith / Braintrust** | Datasets, experiments, judge runs, comparison UIs |
| **scipy.stats, statsmodels, ppi_py** | Statistical tests, cluster-robust SEs, PPI |
