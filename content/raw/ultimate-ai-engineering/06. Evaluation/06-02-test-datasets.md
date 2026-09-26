# 06.02 — Test Datasets

> **Module 6: Evaluation** · Subtopic 2 of 4
> **Prerequisites:** 06.01 (graders, power analysis), 03.01 (embeddings), 04.01 §6 (gold evidence spans), basic sampling theory and logistic regression.
> **Outcome:** you can build evaluation datasets that are **representative, discriminative, trustworthy, and durable**. That means sourcing from production and experts, labelling with measured agreement, generating synthetic cases that are actually useful, preventing contamination and leakage, choosing informative items with item response theory, and governing datasets as versioned assets.

---

## 1. The Dataset Is the Specification

An eval can only be as good as its data. The dataset defines *what "good" means* for your system, so its composition is a product decision.

```
          ┌──────────── sources ────────────┐        ┌──────── quality gates ────────┐
 prod logs (sampled, redacted) ─┐           │        │ dedupe · PII scrub ·          │
 domain experts (hand-written) ─┼─► candidate pool ──►│ label + adjudicate ·          │──► versioned dataset
 red team / adversarial ────────┤           │        │ agreement ≥ target ·          │     + datasheet
 synthetic (LLM-generated) ─────┤           │        │ contamination check ·         │     + splits
 past incidents / bug reports ──┘           │        │ difficulty/IRT screening      │
          └──────────────────────────────────┘        └───────────────────────────────┘
```

### 1.1 Properties of a good eval set

| Property | Meaning | How to get it |
|---|---|---|
| **Representative** | Mirrors the production distribution (or is explicitly reweighted to it) | Sample from logs; post-stratify (§2.2) |
| **Covering** | Includes the rare but important slices: edge cases, unanswerable questions, adversarial inputs | A coverage matrix; targeted authoring |
| **Discriminative** | Separates better systems from worse ones | Remove saturated or impossible items; IRT screening (§6) |
| **Trustworthy labels** | Labels are correct and consistent | Guidelines; multiple annotators; agreement; adjudication (§3) |
| **Clean** | No duplicates, no leakage into prompts or training data | Dedupe; contamination checks; group splits (§5) |
| **Durable** | Stays meaningful as the system and the world change | Versioning, refresh cadence, freshness metadata (§7) |

---

## 2. Sourcing from Production

### 2.1 Sampling strategies

| Strategy | Use for |
|---|---|
| **Uniform random** | Estimating production-level quality |
| **Stratified by slice** (intent, document type, language, user segment) | Guaranteeing coverage of small but important slices |
| **Failure-enriched** (low judge score, negative feedback, escalations, errors) | Regression and capability sets focused on weaknesses |
| **Uncertainty sampling** (low model confidence, judge disagreement) | Label budget where it's most informative |
| **Novelty sampling** (far from existing items in embedding space) | New topics and distribution shift |

### 2.2 Reweighting enriched samples

If you oversample hard slices, the raw mean **over-represents** them. Estimate production-level accuracy with **post-stratification**, where $W_h$ is slice $h$'s share of production traffic and $\bar y_h$ its sample mean:

$$
\hat\mu = \sum_{h} W_h\,\bar y_h,\qquad
\mathrm{Var}(\hat\mu) \approx \sum_h W_h^2\,\frac{s_h^2}{n_h}
$$

```python
import math
import random
from collections import defaultdict


def stratified_sample(records: list[dict], key: str, quotas: dict[str, int], seed: int = 0) -> list[dict]:
    rng, by = random.Random(seed), defaultdict(list)
    for r in records:
        by[r[key]].append(r)
    out = []
    for stratum, q in quotas.items():
        pool = by.get(stratum, [])
        out.extend(rng.sample(pool, min(q, len(pool))))
    return out


def poststratified_mean(scores_by_slice: dict[str, list[float]], traffic_share: dict[str, float]) -> tuple[float, float]:
    total_w = sum(traffic_share[h] for h in scores_by_slice)
    mu, var = 0.0, 0.0
    for h, ys in scores_by_slice.items():
        w = traffic_share[h] / total_w
        m = sum(ys) / len(ys)
        s2 = sum((y - m) ** 2 for y in ys) / (len(ys) - 1) if len(ys) > 1 else 0.0
        mu += w * m
        var += w * w * s2 / len(ys)
    return mu, math.sqrt(var)
```

### 2.3 Privacy

Production data contains personal and sensitive information. Before items enter an eval set:
- Redact or pseudonymise PII (named-entity detection plus deterministic replacement, so references stay consistent).
- Check consent and retention policies.
- Restrict access to the unredacted originals.
- Record the provenance (source, date, redaction version) in the item metadata.

---

## 3. Labelling

### 3.1 Guidelines first

Write **labelling guidelines** before labelling: definitions, decision rules for edge cases, and positive *and* negative examples per label. Pilot them on 30–50 items with 2–3 annotators, revise, and repeat until agreement stabilises.

### 3.2 Measuring agreement

- **Cohen's κ** for two annotators (06.01 §4.3).
- **Fleiss' κ** for several annotators with categorical labels. With $N$ items, $n$ ratings per item, and $n_{ij}$ raters assigning item $i$ to category $j$:

$$
P_i = \frac{1}{n(n-1)}\sum_j n_{ij}(n_{ij}-1),\quad \bar P = \frac{1}{N}\sum_i P_i,\quad p_j = \frac{1}{Nn}\sum_i n_{ij},\quad \bar P_e = \sum_j p_j^2,\quad
\kappa = \frac{\bar P - \bar P_e}{1 - \bar P_e}
$$

- **Krippendorff's α** for missing ratings and ordinal or interval scales.

**Interpretation:** agreement is a property of *the task and the guidelines*, not only of the annotators. A low κ means the construct is ambiguous: fix the guidelines or split the label. As a rule of thumb, κ < 0.6 on a supposedly objective label means the label isn't ready for use in gating.

### 3.3 Adjudication and label noise

- Disagreements go to an expert adjudicator, and the reasons are recorded to refine the guidelines.
- Estimate the residual label noise by re-labelling a random 5–10%. Public benchmarks contain meaningful label-error rates (Northcutt et al., 2021).
- Track "gold" items with confirmed labels. Use them for annotator QA and judge validation.

```python
def fleiss_kappa(counts: list[list[int]]) -> float:
    """counts[i][j] = number of raters assigning item i to category j (same n raters per item)."""
    N, n = len(counts), sum(counts[0])
    P_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts]
    P_bar = sum(P_i) / N
    p_j = [sum(row[j] for row in counts) / (N * n) for j in range(len(counts[0]))]
    P_e = sum(p * p for p in p_j)
    return (P_bar - P_e) / (1 - P_e)
```

---

## 4. Synthetic Test Data

LLM-generated test cases fill coverage gaps cheaply: rare intents, adversarial phrasings, new documents without real queries. They are also easy to get wrong.

### 4.1 A generation pipeline that produces useful items

```
 seeds (documents, intents, personas, difficulty levels, failure categories from error analysis)
   │
   ▼  generate with explicit variation axes (persona × intent × difficulty × phrasing style)
 candidates
   │  filter 1: dedupe (near-duplicate removal, §5.1)
   │  filter 2: validity — answerable from the source? (round-trip retrieval + judge), or explicitly unanswerable
   │  filter 3: label — gold answer/evidence produced AND verified (a second model or a human)
   │  filter 4: difficulty — keep items that at least one strong baseline gets wrong, or that separate systems (§6)
   ▼
 human spot-check (≥ 10%) ──► accepted synthetic set (flagged `synthetic=true`)
```

### 4.2 Known failure modes of synthetic sets

- **Too easy and too clean.** Generated questions echo the source's wording, so lexical retrieval looks great. Counter this with paraphrase and persona instructions, and by measuring lexical overlap with the source.
- **Low diversity.** Items collapse onto a few templates. Measure **distinct-n** and embedding dispersion, and compare them with real queries.
- **Generator bias.** Items favour the generating model's style. When comparing systems, avoid generating with the same model family you evaluate, or balance the generators.
- **Label errors.** A generated "gold" answer can be wrong. Always verify.

**Rule:** report synthetic and real items **separately**, and never gate releases on synthetic data alone.

```python
def distinct_n(texts: list[str], n: int = 2) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = t.lower().split()
        g = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        grams.update(g)
        total += len(g)
    return len(grams) / total if total else 0.0


def lexical_overlap(question: str, source: str) -> float:
    q = set(question.lower().split())
    return len(q & set(source.lower().split())) / len(q) if q else 0.0
```

---

## 5. Cleanliness: Duplicates, Contamination, Leakage

### 5.1 Near-duplicate removal (MinHash)

Near-duplicates inflate your confidence and skew slices. For shingle sets $A, B$, the Jaccard similarity is $J = |A\cap B|/|A\cup B|$. MinHash estimates it from signatures: $\Pr[h_{\min}(A) = h_{\min}(B)] = J$.

```python
import zlib

import numpy as np

_P = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> set[int]:
    toks = text.lower().split()
    return {zlib.crc32(" ".join(toks[i:i + k]).encode()) for i in range(max(1, len(toks) - k + 1))}


def minhash(sh: set[int], num_perm: int = 128, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _P, num_perm, dtype=np.uint64)
    b = rng.integers(0, _P, num_perm, dtype=np.uint64)
    x = np.array(sorted(sh), dtype=np.uint64)[:, None]
    return ((x * a + b) % _P).min(axis=0)                  # uint64 arithmetic wraps; fine as a hash family


def dedupe(texts: list[str], threshold: float = 0.8, num_perm: int = 128) -> list[int]:
    """Return indices to keep (O(n^2) signature comparison; use LSH banding at scale)."""
    sigs = [minhash(shingles(t), num_perm) for t in texts]
    keep = []
    for i, s in enumerate(sigs):
        if all((s == sigs[j]).mean() < threshold for j in keep):
            keep.append(i)
    return keep
```

### 5.2 Contamination

**Contamination** means the eval items (or their answers) appeared in the data the system was trained or tuned on, or they appear in its **prompts** (few-shot examples, retrieved documents built from the same source). For LLM apps, the most common contamination is **self-inflicted**:
- Few-shot examples copied from the test set.
- Prompt tuning against the test set (02.01 §5 — use a separate development set).
- A fine-tuning corpus that includes test documents.
- Synthetic training data generated from the test set's documents.

Detection:
- **n-gram overlap** (e.g. 13-grams) between the test items and the training, prompt, and example corpora.
- **Embedding similarity** above a threshold.
- **Canary strings** embedded in private test sets.
- For third-party models, **membership signals** (e.g. models completing test items verbatim; order-sensitivity tests).

```python
def ngram_set(text: str, n: int = 13) -> set[tuple]:
    t = text.lower().split()
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def contamination_report(test_items: dict[str, str], corpus: list[str], n: int = 13) -> dict[str, float]:
    """Fraction of each test item's n-grams found anywhere in the corpus."""
    corpus_grams = set().union(*(ngram_set(c, n) for c in corpus)) if corpus else set()
    out = {}
    for k, text in test_items.items():
        g = ngram_set(text, n)
        out[k] = len(g & corpus_grams) / len(g) if g else 0.0
    return out
```

### 5.3 Leakage through splits

When items share a source (multiple questions per bill, multiple turns per conversation), a random split puts siblings on both sides. Tuning on development items then leaks into the test results. Use **group splits** (by document, conversation, or user) and **temporal splits** (test on items newer than anything used for tuning) whenever the deployment faces the future.

```python
import hashlib


def group_split(items: list[dict], group_key: str, test_frac: float = 0.2, dev_frac: float = 0.1,
                salt: str = "v1") -> dict[str, list[dict]]:
    """Deterministic split by group hash — the same group always lands in the same split."""
    out = {"train": [], "dev": [], "test": []}
    for it in items:
        h = int(hashlib.sha256(f"{salt}:{it[group_key]}".encode()).hexdigest(), 16) / 2 ** 256
        split = "test" if h < test_frac else "dev" if h < test_frac + dev_frac else "train"
        out[split].append(it)
    return out
```

---

## 6. Choosing Informative Items: Item Response Theory

Not all items are equally useful. Some everyone solves, some nobody solves, and some are mislabelled. **Item response theory** (IRT) models the probability that system $j$ (ability $\theta_j$) answers item $i$ (difficulty $b_i$, discrimination $a_i$) correctly:

$$
P(y_{ij} = 1) = \sigma\big(a_i(\theta_j - b_i)\big),
\qquad
\mathcal{I}_i(\theta) = a_i^2\,P_i(\theta)\big(1 - P_i(\theta)\big)
$$

The **item information** $\mathcal{I}_i(\theta)$ is highest when the item's difficulty matches the ability level of the systems you are comparing. Fit IRT to a response matrix (many systems or configurations × items). Then:
- **Retire saturated items** (very low $b$, since all systems solve them) into a cheap regression suite.
- **Flag suspicious items:** negative discrimination often means a wrong label.
- **Build compact evals:** select the items with the highest total information near your systems' ability range. This can cut eval cost several-fold with little loss in ranking fidelity (tinyBenchmarks).

```python
def fit_2pl(Y: np.ndarray, iters: int = 2000, lr: float = 0.05, l2: float = 0.01, seed: int = 0):
    """Y: (systems, items) binary matrix. Joint MAP estimation by gradient ascent (educational)."""
    rng = np.random.default_rng(seed)
    S, I = Y.shape
    theta = rng.normal(0, 0.1, S)
    b = rng.normal(0, 0.1, I)
    log_a = np.zeros(I)
    for _ in range(iters):
        a = np.exp(log_a)
        z = a[None, :] * (theta[:, None] - b[None, :])
        p = 1 / (1 + np.exp(-z))
        r = Y - p                                             # d loglik / dz
        theta += lr * ((r * a[None, :]).mean(1) - l2 * theta)
        b += lr * ((-r * a[None, :]).mean(0) - l2 * b)
        log_a += lr * ((r * (theta[:, None] - b[None, :])).mean(0) * a - l2 * log_a)
        theta -= theta.mean()                                 # identifiability: centre abilities
    return theta, b, np.exp(log_a)


def most_informative(b: np.ndarray, a: np.ndarray, theta_range: np.ndarray, k: int) -> np.ndarray:
    p = 1 / (1 + np.exp(-a[None, :] * (theta_range[:, None] - b[None, :])))
    info = (a[None, :] ** 2 * p * (1 - p)).sum(0)
    return np.argsort(-info)[:k]
```

---

## 7. Governance: Datasets as Versioned Assets

- **Immutable versions:** every release of a dataset (`legis-qa@2026.09.1`) is content-addressed. Scores are always reported *with* the dataset version.
- **Datasheet / data card:** purpose, sources, collection dates, sampling, labelling process, agreement, known gaps, PII handling, licence, intended and prohibited uses.
- **Splits are fixed and documented:** dev (tune freely), test (report), and a **hidden holdout** (rarely run, access-controlled) to detect overfitting to the test set.
- **Refresh cadence:** add new failure categories from error analysis (06.04) and new production slices; retire saturated items; re-validate labels when policies or laws change (legal facts go stale).
- **Ownership:** a named owner per dataset, reviews for changes (pull requests on the dataset repository), and a changelog.

```python
import hashlib
import json
from collections import defaultdict


def dataset_manifest(name: str, version: str, items: list[dict], meta: dict) -> dict:
    """Content-addressed manifest: any change to any item changes the dataset hash."""
    item_hashes = sorted(hashlib.sha256(json.dumps(it, sort_keys=True).encode()).hexdigest() for it in items)
    ds_hash = hashlib.sha256("".join(item_hashes).encode()).hexdigest()
    slices = defaultdict(int)
    for it in items:
        slices[it.get("slice", "all")] += 1
    return {"name": name, "version": version, "sha256": ds_hash, "n_items": len(items),
            "slices": dict(slices), "synthetic_share": sum(bool(it.get("synthetic")) for it in items) / len(items),
            **meta}
```

---

## 8. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Unrepresentative set** | Offline scores don't predict production | Sample from logs; post-stratify; compare slice shares with traffic |
| **Missing rare-but-critical cases** | Incidents in slices never tested | Coverage matrix; targeted authoring; incident → test case |
| **Ambiguous labels** | Low agreement; judges can't be validated | Guidelines, pilots, adjudication; split ambiguous labels |
| **Synthetic set too easy** | Great scores, poor reality | Paraphrase and persona variation; overlap and diversity metrics; difficulty filtering; report separately |
| **Contamination** | Suspiciously high scores; mismatch with fresh data | n-gram/embedding checks; canaries; strict dev/test separation; temporal splits |
| **Leakage via siblings** | Tuned systems ace the test but not new documents | Group and temporal splits |
| **Saturation** | Everyone scores ~100%; no signal | IRT screening; retire into the regression suite; add harder items |
| **Stale labels** | Correct answers marked wrong after the law changed | Freshness metadata; scheduled re-validation; as-of dates on items |

---

## 9. Hands-On Projects

### Project 1 — Golden Dataset for Legislative QA

**User stories**
- *As the evaluation owner*, I want a trustworthy, versioned dataset of real legislative questions with gold answers and evidence, so that every system change is judged against the same bar.

**Acceptance criteria**
1. ≥ 600 items: 60% sampled from redacted production logs (stratified by intent and document type), 25% expert-written for coverage gaps, and 15% adversarial or unanswerable. Each has gold answers, gold evidence spans (04.01 §6), an as-of date, and a slice.
2. Guidelines with a pilot; Fleiss' κ ≥ 0.7 on the answer-correctness labels (3 annotators on 100 items); an adjudication log.
3. Near-duplicate removal (MinHash), a contamination report against the prompt examples and fine-tuning corpora, and group splits by bill/document (`group_split`) into dev, test, and a hidden holdout.
4. A datasheet plus `dataset_manifest`, stored in a dataset repository with PR-based changes.
5. The production-level accuracy estimate uses post-stratification with a CI, alongside per-slice results.

**Step-by-step**
1. Define the slice taxonomy and quotas; sample the logs; run PII redaction.
2. Write the guidelines; pilot; iterate; label; adjudicate.
3. Run the dedupe and contamination checks; build the splits.
4. Write the datasheet; publish version 1; set up the change process.
5. Evaluate the current system; report the post-stratified and per-slice results.

---

### Project 2 — Synthetic Test Generation Pipeline with Quality Gates

**User stories**
- *As a RAG engineer*, I want to generate test questions for newly ingested documents automatically, but only keep items that are valid, diverse, and challenging.

**Acceptance criteria**
1. A pipeline following §4.1 with explicit variation axes (persona × intent × difficulty × phrasing) and all four filters, plus human spot checks.
2. For ≥ 200 new documents, it generates ≥ 1,000 candidates and reports the pass rate of each filter.
3. Quality comparison against real queries: distinct-2, mean lexical overlap with the source, embedding dispersion, and system accuracy on synthetic vs real items (the gap quantified).
4. Label accuracy: a human audit of 100 accepted items (the target is ≥ 95% correct gold labels).
5. Recommendations: which slices synthetic data can cover reliably, and which need humans.

**Step-by-step**
1. Build the generator prompts with structured outputs (02.03), seeded by documents and error categories.
2. Implement the filters: `dedupe`, round-trip answerability via your 04.02 retriever plus a judge, gold verification by a second model, and difficulty via baseline failures.
3. Compute the diversity and overlap metrics; compare them with the real set.
4. Run the human audit.
5. Integrate into the ingestion pipeline (new documents → new candidate tests → review queue).

---

### Project 3 — Contamination, Saturation, and a Compact IRT-Selected Eval

**User stories**
- *As an evaluation lead*, I want to know whether our test set is still informative and uncontaminated, and I want a much cheaper subset that ranks system variants the same way.

**Acceptance criteria**
1. Contamination audit: n-gram and embedding checks against all prompt, example, and fine-tuning corpora, plus a canary-string check. Contaminated items are quarantined.
2. A response matrix of ≥ 15 system variants (models × prompts × retrieval configs) × ≥ 500 items. Fits a 2PL IRT model (`fit_2pl`, or `py-irt` / `girth`).
3. Flags saturated items (solved by ≥ 95% of variants) and suspicious items (negative discrimination); human review confirms or fixes their labels.
4. Builds a compact subset of 20% of the items via `most_informative`. Rank correlation (Kendall τ) with the full-set ranking is ≥ 0.9 on held-out variants.
5. Retirement and refresh policy, documented and applied (version 2 of the dataset).

**Step-by-step**
1. Run the contamination checks and quarantine.
2. Generate the variants and collect the per-item correctness matrix (cached outputs).
3. Fit the IRT model; inspect the difficulty and discrimination distributions.
4. Select the compact subset; validate the ranking fidelity with held-out variants.
5. Publish the new dataset version with the changelog.

---

## 10. Foundational Papers & Reading (exact titles)

**Documentation and governance**
- Gebru et al., 2018 — *Datasheets for Datasets*
- Pushkarna, Zaldivar & Kjartansson, 2022 — *Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI*

**Labelling and agreement**
- Fleiss, 1971 — *Measuring nominal scale agreement among many raters*
- Artstein & Poesio, 2008 — *Inter-Coder Agreement for Computational Linguistics*
- Northcutt, Athalye & Mueller, 2021 — *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks*

**Contamination and leakage**
- Sainz et al., 2023 — *NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark*
- Golchin & Surdeanu, 2023 — *Time Travel in LLMs: Tracing Data Contamination in Large Language Models*
- Oren et al., 2023 — *Proving Test Set Contamination in Black Box Language Models*
- Kapoor & Narayanan, 2023 — *Leakage and the Reproducibility Crisis in Machine-Learning-based Science*
- Lee et al., 2022 — *Deduplicating Training Data Makes Language Models Better*
- Broder, 1997 — *On the Resemblance and Containment of Documents* (MinHash)

**Item selection and dynamic benchmarks**
- Lalor, Wu & Yu, 2016 — *Building an Evaluation Scale using Item Response Theory*
- Rodriguez et al., 2021 — *Evaluation Examples are not Equally Informative: How should that change NLP Leaderboards?*
- Polo et al., 2024 — *tinyBenchmarks: evaluating LLMs with fewer examples*
- Kiela et al., 2021 — *Dynabench: Rethinking Benchmarking in NLP*
- Perez et al., 2022 — *Discovering Language Model Behaviors with Model-Written Evaluations*

## 11. Essential Tooling

| Tool | Role |
|---|---|
| **Label Studio / Argilla / Prodigy** | Annotation, adjudication, agreement tracking |
| **Presidio** | PII detection and anonymisation |
| **datasketch** | MinHash/LSH deduplication at scale |
| **py-irt / girth** | Item response theory fitting |
| **DVC / lakeFS / Hugging Face Datasets (private)** | Dataset versioning and distribution |
| **Langfuse / Phoenix / LangSmith / Braintrust datasets** | Promoting traces to dataset items (06.04) |
| **pandas / Polars** | Slicing, stratification, reporting |
