# 08 — Model Evaluation (Production)

> **Module goal:** Move from "does my model work on the test set?" to "is my model **still** working on the users I have **today**, and how would I know if it silently degraded?" Master offline evaluation, online evaluation, LLM-as-judge, human-in-the-loop, and continuous evaluation pipelines that gate deploys and page you on drift.

---

## 1. Executive Summary & Core Concepts

Model evaluation in production is not the sklearn `classification_report` you ran once during Month 2. It is a **continuous pipeline** with four modes:

| Mode | When it runs | Purpose | Example |
|------|-------------|---------|---------|
| **Offline eval** | Pre-deploy (CI) + nightly | Regression gate; catch quality drops before users do | Hold-out set, golden set, synthetic set |
| **Online eval** | Live traffic | Real-world performance; detect drift | A/B test, shadow traffic, canary |
| **Human eval** | Sampled from prod, or annotation runs | Ground truth for hard tasks; calibrate LLM judges | RLHF label queue, red-team drills |
| **LLM-as-judge** | Continuous, cheap | Scale human judgment to full traffic | GPT/Claude scoring responses on rubric |

The core insight: **models decay silently.** No exception is thrown when a classifier's precision drops from 0.92 to 0.71. Users churn, revenue drops, the on-call sees nothing. The only defense is *evaluation as a first-class product surface* — dashboards, alerts, deploy gates.

Key metric categories:

**Classical ML:**
- Classification: precision, recall, F1, ROC-AUC, calibration
- Regression: MAE, RMSE, R², MAPE (be careful — divides by zero)
- Ranking: NDCG@k, MRR, MAP

**LLM-specific (open-ended generation):**
- **Reference-based** (compare to gold answer): BLEU, ROUGE, BERTScore, exact-match
- **Reference-free** (LLM-as-judge or rubric): faithfulness, helpfulness, safety, style
- **Task-specific**: for RAG — faithfulness, answer relevance, context precision/recall (RAGAS); for code — pass@k on unit tests; for extraction — field-level F1

**Drift metrics (statistical, no labels needed):**
- **Input drift** — has the input distribution shifted? (PSI, KL divergence, KS test)
- **Output drift** — has the model's prediction distribution shifted?
- **Latency/error drift** — engineering-adjacent proxies for quality

**The golden set** — a small (100–1000), carefully curated, high-quality labeled set that never changes and represents production edge cases you must not regress on. Every deploy passes through it in CI. When you find a new production failure, add it to the golden set (with the fix as the new label). This is your durable memory of quality.

**Eval hierarchy** (cheapest → most expensive):
1. Unit tests on prompts/rubrics (< 1s, run on every PR)
2. Golden set eval (~1–10 min, run on every PR)
3. Full held-out eval + LLM-judge scoring (~15–60 min, nightly)
4. Human review of sampled prod traffic (hours, weekly)
5. A/B test on real users (days–weeks, per major release)

---

## 2. Deep-Dive Breakdown

### 2.1 The Golden Set: Your Deploy Gate

A golden set is small and precious. Rules:

1. **Curated, not sampled.** Deliberately include edge cases: adversarial inputs, minority classes, known past bugs, historical bad user experiences. If sampling randomly captures 5% negatives, that's fine — but *supplement* it with the failure cases.
2. **Never train on it.** Not in fine-tuning data, not in few-shot examples. Treat it like sealed evidence.
3. **Versioned like code.** Live in the repo (or DVC/MLflow artifact) with review on changes.
4. **Failing a golden case blocks merge.** Not a warning — a red X on the PR.

Example structure (`evals/golden_set.jsonl`):
```jsonl
{"id": "gs-001", "input": "What year did I file taxes last?", "expected": "REFUSAL: pii", "category": "safety-refusal"}
{"id": "gs-002", "input": "Summarize this 2-line email: ...", "expected_contains": ["hello", "meeting"], "category": "summarization"}
{"id": "gs-003", "input": "What's 2+2?", "expected": "4", "category": "basic-correctness"}
{"id": "gs-042", "input": "<past-bug-that-once-caused-data-leak>", "expected_not_contains": ["ssn", "@"], "category": "regression"}
```

In CI:
```python
# evals/run_golden.py
import json, sys
from my_service import answer

fails = []
for line in open("evals/golden_set.jsonl"):
    case = json.loads(line)
    out = answer(case["input"])
    if "expected" in case and out.strip() != case["expected"]:
        fails.append((case["id"], out))
    if "expected_contains" in case and not all(s in out for s in case["expected_contains"]):
        fails.append((case["id"], out))
    if "expected_not_contains" in case and any(s in out for s in case["expected_not_contains"]):
        fails.append((case["id"], f"LEAKED: {out}"))

if fails:
    print(f"{len(fails)} golden cases failed:")
    for fid, msg in fails: print(f"  {fid}: {msg}")
    sys.exit(1)
```

### 2.2 LLM-as-Judge — Scaling Human Judgment

Human labeling is the gold standard but doesn't scale. **LLM-as-judge** uses a stronger model to grade the output of your production model against a rubric. The pattern:

```python
JUDGE_RUBRIC = """You are an impartial evaluator. Given a user question and a model answer,
score the answer on:
- Faithfulness (1-5): did it stay grounded in the provided context?
- Helpfulness (1-5): did it actually answer what was asked?
- Safety (pass/fail): did it avoid PII exposure, medical/legal advice, etc.?

Return strict JSON: {"faithfulness": int, "helpfulness": int, "safety": "pass"|"fail", "reasoning": str}"""

def judge(question: str, context: str, answer: str) -> dict:
    resp = judge_client.messages.create(
        model="claude-opus-5",  # judge model stronger than judged model
        max_tokens=500,
        system=JUDGE_RUBRIC,
        messages=[{"role": "user", "content": f"Q: {question}\nContext: {context}\nA: {answer}"}],
    )
    return json.loads(resp.content[0].text)
```

**LLM-judge best practices:**
- **Use a stronger judge** than the judged model (Opus judging Sonnet's output).
- **Calibrate against human labels.** Take 100 human-labeled examples, score them with the judge, measure agreement (Cohen's κ). Iterate on the rubric until κ > 0.7.
- **Pairwise beats absolute.** Judges are noisy at absolute scores; they're more reliable at "A vs B, which is better?" This is the RLHF preference-model insight.
- **Ask for reasoning first** ("chain-of-thought"), then the score. Improves calibration.
- **Beware position bias, verbosity bias, sycophancy bias.** Judges often prefer the first answer, the longer answer, and answers matching perceived user opinion. Randomize order; length-normalize; remove leading context.
- **Judges cost money.** Tier: judge 100% of low-traffic critical paths, sample 5–10% of high-traffic paths.

### 2.3 Online Evaluation: Shadow, Canary, A/B

**Shadow traffic** — new model receives copies of real requests; its outputs are logged but not returned to users. Zero user risk. Compare shadow metrics to production. Cost: 2× inference.

```python
async def handle(request):
    prod_response = await prod_model(request)
    # Fire-and-forget shadow call
    asyncio.create_task(shadow_and_log(request, prod_response))
    return prod_response

async def shadow_and_log(req, prod_resp):
    try:
        shadow_resp = await new_model(req)
        await eval_store.write({
            "req_id": req.id,
            "prod": prod_resp,
            "shadow": shadow_resp,
            "prod_score": await judge(req, prod_resp),
            "shadow_score": await judge(req, shadow_resp),
        })
    except Exception as e:
        log.warning("shadow_failed", error=str(e))  # never crash prod
```

**Canary** — new model receives a small fraction of real traffic (1% → 5% → 25% → 100%). Compare user-facing metrics between cohorts. Automatic rollback on error/latency/quality regression.

**A/B test** — split traffic by user (deterministic hash on `user_id`), measure business KPIs (retention, task completion, thumbs-up rate). Requires statistical rigor: define primary metric ahead of time, compute sample size for detectable effect, don't peek.

### 2.4 Drift Detection

Even without labels, you can detect *something changed*.

**Population Stability Index (PSI)** for a single feature:
$$
\text{PSI} = \sum_i (P_i^{prod} - P_i^{ref}) \cdot \ln\frac{P_i^{prod}}{P_i^{ref}}
$$
where $P_i$ is the proportion of samples in bin $i$. Rule of thumb: PSI < 0.1 stable; 0.1–0.25 moderate drift; > 0.25 significant drift.

**Kolmogorov–Smirnov test** for continuous features: the maximum vertical distance between the empirical CDFs of the reference and production distributions.

**For LLMs / text inputs:**
- Track **input length distribution** (mean, p95, p99)
- Track **detected language mix**
- Track **top-N n-gram frequencies** vs. baseline
- Track **embedding-space centroid distance** — embed a batch of inputs, compare centroid vs. rolling reference

Emit each as a gauge; alert if beyond baseline for N hours.

`[IMG-08-01]` — *Prompt: A three-panel diagram showing model quality over time. Panel 1: "Reference distribution" — a smooth bell curve of input feature X, blue. Panel 2: "Production week 4 — no drift" — same bell curve overlaid in orange, PSI = 0.03 label. Panel 3: "Production week 12 — drift detected" — shifted bell curve, PSI = 0.31 label, red alert badge saying "Retrain candidate". Clean minimalist scientific diagram style.*

### 2.5 RAG Evaluation Specifically (RAGAS)

For retrieval-augmented systems, four metrics matter:

| Metric | Question | Method |
|--------|----------|--------|
| **Context precision** | Are retrieved chunks relevant? | LLM-judges each chunk vs. question; averages |
| **Context recall** | Did retrieval get all needed info? | Compares retrieved context vs. ground-truth answer, LLM-judges |
| **Faithfulness** | Is the answer grounded in retrieved context? | Extracts claims from answer, verifies each against context |
| **Answer relevance** | Does the answer address the question? | Generates candidate questions from the answer, embeds against original |

The [RAGAS](https://github.com/explodinggradients/ragas) library packages all four; each is an LLM-judge under the hood.

```python
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from datasets import Dataset

data = Dataset.from_dict({
    "question": [...],
    "contexts": [[...], ...],
    "answer": [...],
    "ground_truth": [...],
})
result = evaluate(data, metrics=[context_precision, context_recall, faithfulness, answer_relevancy])
# Log to MLflow / dashboard / alert if any dips below threshold
```

Run RAGAS on a golden RAG set in CI. Alert if any metric drops > 5 points from the last known good.

---

## 3. Mental Models & Analogies

### Model 1: The Clinical Trial Pipeline

Drug development goes through phases: preclinical (petri dish, animal), Phase I (a few humans, safety), Phase II (dozens, efficacy), Phase III (thousands, real-world), Phase IV (post-market surveillance forever). Each phase filters out failures at increasing cost.

Model evaluation is identical:
- **Preclinical** = unit tests on your prompts and rubrics
- **Phase I** = golden set in CI
- **Phase II** = full offline eval, LLM-judged
- **Phase III** = canary or A/B with real users
- **Phase IV** = continuous production monitoring, drift detection, human review sampling — *this never stops*

Skipping a phase to save time is what causes both drug recalls and production incidents. Post-market surveillance (drift monitoring) is what separates responsible engineering from cargo-cult MLOps.

### Model 2: The Sports Officiating Crew

A referee (production model) makes calls in real time. A **video assistant referee (VAR)** reviews close calls (LLM-judge on flagged responses). The **league's disciplinary committee** reviews controversies days later (human review of sampled prod traffic). And every off-season, the **rules committee** revises the rulebook based on emerging patterns (retraining or prompt-tuning based on drift).

Note that the referee never grades themselves. The judge is always a separate, more expensive-per-decision actor. And the rulebook is never revised mid-game — you don't retrain in prod. Each role in the crew has a distinct latency-cost-accuracy tradeoff; a well-run league uses all of them.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Eval-set leakage: the model has already seen your test data.**
You held out 20% for evaluation. But the training data included web crawls of Stack Overflow, and your "held-out" questions were paraphrases of top-voted questions. The eval accuracy is 95% because the model *memorized*, not learned. This is rampant with LLMs — modern base models have seen most public benchmarks. **Fix:** for LLM eval, generate *fresh* eval inputs (synthetic, or newly collected post the training cutoff), or use closed benchmarks. Never blindly trust an academic-benchmark score on a general-purpose LLM.

**Pitfall #2 — Optimizing the metric, not the goal.**
BLEU-4 correlates weakly with actual usefulness of a summary. F1 on a noisy label distribution rewards the model that best matches the noise. Latency at p50 hides tail user pain. **Every metric is a proxy** for a business or user goal, and Goodhart's Law will bite: once you optimize the proxy, it decouples from the goal. Guardrails: (1) keep multiple metrics, especially orthogonal ones (quality *and* latency *and* cost); (2) regularly do human review to sanity-check the metric still tracks reality; (3) treat every large metric jump with suspicion — did you improve, or did you find a shortcut?

**Pitfall #3 — Trusting the LLM-judge without calibration.**
"GPT-4 said the answer is 5/5, ship it." LLM-judges have documented biases: they prefer the first option in pairwise comparisons (position bias), longer answers (verbosity bias), and answers styled like their own outputs (self-preference). They also *hallucinate scores* just as they hallucinate content. **Fix:** every LLM-judge in production must have a calibration report — how does it correlate with human labels on your data? What's Cohen's κ? Refresh the calibration whenever you change the judge model. And never let the same model both generate and judge — that's an open-book test.

---

## 5. Self-Assessment Bank

**Q1 (MC):** Your golden set has 200 examples. You add a new one every time a bug is fixed in production. It now has 500 examples. This growth is:
A) A problem — the set is drifting from its original purpose
B) Correct — the set encodes durable memory of production edge cases
C) A cost issue — CI is now slower
D) Suspicious — you're overfitting to the golden set

**Q2 (short):** Your team wants to test a new prompt version. Describe the difference between shadow traffic, canary, and A/B test, and when to use each.

**Q3 (MC):** Cohen's κ between your LLM-judge and human labels is 0.42 across 100 samples. What should you conclude?
A) The judge is calibrated — ship it
B) The judge is unreliable for this task — refine the rubric or use a stronger judge
C) The humans are unreliable
D) κ doesn't apply to LLM-judges

**Q4 (short):** Explain PSI (Population Stability Index) in one sentence and give a rule-of-thumb threshold at which you'd act.

**Q5 (MC):** For a RAG system, "faithfulness" measures:
A) Whether the retrieved chunks are relevant to the question
B) Whether the generated answer is grounded in the retrieved context (no hallucination)
C) Whether the answer is factually correct in absolute terms
D) Whether the retrieval reranker is accurate

**Q6 (short):** Name three biases that affect LLM-as-judge and one mitigation for each.

**Q7 (MC):** Your model's accuracy on the held-out set is 96%, but user thumbs-down rate has crept from 5% to 15% over three months while accuracy stayed flat. What is the most likely explanation?
A) The model has a bug
B) Distribution drift — production inputs no longer resemble the held-out set
C) Users are being harsher
D) The thumbs-down button is broken

**Q8 (MC):** A "shadow deploy" is:
A) A deploy behind a feature flag off by default
B) A new version that receives copies of production traffic but doesn't return responses to users
C) A canary at 1% traffic
D) An A/B test with 50/50 split

**Q9 (short):** Design a CI gate for LLM model changes. What must run, what must pass, what's advisory-only?

**Q10 (MC):** Why is pairwise LLM-judging often preferred over absolute scoring?
A) It's cheaper — one call instead of two
B) Judges are more reliable at "A vs B" comparisons than at absolute 1–5 scores
C) It doesn't require a rubric
D) It works offline

---

### Answer Key

**A1: B.** The golden set is intentionally a growing, curated record of *edge cases and past failures.* It's not a random test set; it's institutional memory. Overfitting to it (D) is a real risk only if you use it as a *training* signal — for evaluation-only, it's fine that the model performs well because you've explicitly encoded the cases you care about. Do periodically prune obsolete cases and keep the total tractable for CI runtime.

**A2:**
- **Shadow traffic:** new model receives copies of prod requests, outputs logged but discarded. Zero user risk, but doubles inference cost. Use when the new model is unproven — you want to compare metrics without any user impact.
- **Canary:** new model serves a small fraction of real traffic (start 1%). Users see the output; you monitor SLIs and roll back if they regress. Use when you've cleared shadow and want to expose to reality gradually.
- **A/B test:** deterministic split of users, measure a *business* metric (retention, task completion) with statistical rigor over days/weeks. Use when the technical metrics look fine but you need to confirm the change moves the product needle.
Typical order: shadow → canary → A/B for a major change.

**A3: B.** Cohen's κ of 0.42 is "moderate agreement" — borderline for casual work, unacceptable for production quality gating. Landis & Koch's rough guide: < 0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, > 0.80 almost perfect. Below ~0.7 you have a lot of noise. Actions: rewrite the rubric to be less ambiguous, add worked examples in-context, try a stronger judge model, or split the task into multiple simpler judge questions.

**A4:** PSI measures how much a feature's distribution has shifted between a reference set and a current window; it's a weighted log-ratio of bin proportions. Rule of thumb: **PSI > 0.25 is significant drift** worth investigating (data pipeline change, real user behavior shift, or upstream data-source change) — often a retraining/re-eval trigger. 0.1–0.25 is a yellow light.

**A5: B.** Faithfulness in RAG = the answer's claims are traceable to the retrieved context — i.e., no hallucination beyond what was retrieved. It's about *grounding to context*, not absolute truth. (A) is context precision; (C) is factual accuracy (which requires ground truth); (D) is a retrieval metric, not a generation metric.

**A6:** Three biases + mitigation each:
1. **Position bias** — judges prefer the first option in pairwise. *Mitigation:* randomize order per call, or run twice with swapped order and average.
2. **Verbosity bias** — judges rate longer answers higher, regardless of quality. *Mitigation:* include length-neutral criteria in the rubric; consider normalizing by token count; explicitly instruct the judge to ignore length.
3. **Self-preference / stylistic bias** — a judge from provider X prefers responses from provider X's models. *Mitigation:* use a different provider or family as judge; validate with human labels; ensemble multiple judges.
4. **Sycophancy** — judges align with perceived user opinion in the prompt. *Mitigation:* strip user-provided context of preference cues before judging; frame the judge as impartial.

**A7: B.** The held-out set is frozen from an earlier snapshot; if user behavior drifts (new topics, new vocabulary, new intents), the held-out set no longer represents the *current* user distribution. Model performs great on the old distribution, poorly on the new one. Fix: refresh the eval set from recent production (with fresh labels), track drift metrics, and consider re-training or updating retrieval corpora.

**A8: B.** Shadow deploys route copies of production traffic to a new version whose responses are logged and evaluated but *not returned to users*. It gives you real-input eval with zero user exposure — at the cost of doubled inference for the shadowed fraction.

**A9:** Sample CI gate design:
- **Must run + must pass** (blocks merge): (1) unit tests on prompt-building logic; (2) golden set — 100% pass required; (3) safety eval — 100% pass on refusal-required cases; (4) latency sanity — p95 within 20% of baseline on a fixed load-test payload.
- **Must run + must not regress > threshold** (blocks merge): (5) full held-out eval (LLM-judged); alert if any metric drops > 5 points; (6) RAGAS if it's a RAG change.
- **Advisory only** (reports to PR, doesn't block): (7) cost per request delta; (8) response length delta; (9) refusal rate delta.
- **Post-merge** (before general availability): (10) shadow traffic for N hours; (11) canary at 1% → 5% → 25%; (12) A/B test on the primary KPI.

**A10: B.** LLM judges are notoriously noisy at absolute scoring — same input can get 3/5 or 5/5 across runs. They are much more consistent at "which of these two is better?" This is why RLHF is built on preference data, not absolute ratings, and why systems like Chatbot Arena use pairwise Elo. (A) is factually wrong — pairwise still requires the judge to see both. (C, D) are unrelated.

---

**Related modules:**
- `learning/05-mlflow.md` — where you store eval runs and models
- `learning/04-ci-cd.md` — where the golden-set gate lives
- `learning/09-llm-observability.md` — LangSmith / LangFuse for continuous eval on prod traces
- `../llm-engineering/13-rag-evaluation.md` — RAG-specific detail
- `../agents-production/12-agent-evaluation.md` — agent-specific eval

**Practice prompts:**
1. Take your Month 4 RAG build's test set. Wire RAGAS on it. Post the four scores.
2. Write a `judge()` function with a rubric for one of your builds. Score 20 outputs, label 20 yourself, compute Cohen's κ.
3. Add a golden-set CI job to your last build's GitHub Actions workflow. Deliberately break something to see the gate fire.

**References:**
- Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023) — landmark paper on LLM-judge biases
- RAGAS docs — https://docs.ragas.io
- OpenAI Evals — https://github.com/openai/evals
- Hugging Face `lighteval` — modern eval harness
- Chip Huyen, *Designing Machine Learning Systems* — Ch. 8 (Model Deployment) and Ch. 9 (Continual Learning)
