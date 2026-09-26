# RAG — Evaluation — Master Study Guide

> **Track:** LLM Engineering · **Module:** 18 (final learning module)
> **Prerequisites:** Modules 12–17.
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every technique in modules 12–17 makes claims: "this chunker is better," "this reranker helps," "this prompt reduces hallucination." **Evaluation is how you verify those claims.** Without it:

- You optimize on vibes.
- You ship changes that quietly regress quality.
- You cannot defend architecture decisions to your team.
- You have no early-warning system when production degrades.

**Fundamental principles you must own:**

1. **RAG has two failure modes:** retrieval failure and generation failure. Evaluate each separately, then together.
2. **You need an eval dataset** — question + expected answer + expected source. Curate 50–500 labeled examples for offline eval.
3. **Metrics fall into two buckets:** retrieval metrics (recall@k, MRR, NDCG) and generation metrics (faithfulness, answer relevance, context precision/recall).
4. **LLM-as-judge is the modern default** for open-ended evaluation. Use it carefully.
5. **Regression testing** should be automatic — every change re-runs the eval.
6. **Production evaluation** is different from offline: sample, tag, and monitor over time.

If you retain nothing else: **measure or you're not engineering. RAG without an eval is a rumor.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Building an Eval Dataset

Your eval dataset is a set of **triples**: `(question, expected_answer, expected_source_chunk_ids)`. Optionally also a **ground-truth answer** written by an expert.

**Sources of eval data:**

- **Curated manually** — for the first version, hand-write 50–100 questions with expected sources. Highest quality; lowest scale.
- **Generated with an LLM from source docs** — prompt: "For each chunk, produce 3 questions this chunk answers, with the expected answer." Auto-generates hundreds of eval examples. Lower quality; needs human review.
- **Extracted from real user queries** — if you have a live system, sample production questions. Highest realism.
- **Public benchmarks** — MTEB retrieval subsets, MS MARCO, KILT, RAG-Bench, HotpotQA. Domain-general.

**Criteria for a good eval dataset:**

- **Diverse** — cover all your question types and document types.
- **Verifiable** — each answer must be checkable against the sources.
- **Realistic** — matches the actual user distribution.
- **Sized right** — 50–200 for iteration; 500+ for CI regression testing.

**Example eval row (JSONL):**

```json
{
  "question": "How soon after starting do employees get health coverage?",
  "expected_answer": "Health benefits start on the first of the month following your start date.",
  "expected_sources": ["benefits.pdf::chunk_12"],
  "difficulty": "easy",
  "category": "hr_policy"
}
```

**Difficulty and category tags** enable slicing metrics (see 2.4).

**Auto-generate with quality control:**

```python
def generate_qa_from_chunk(chunk):
    prompt = f"""Given the following passage, write ONE clear question this passage answers,
    and provide the answer verbatim from the passage.

    Passage: {chunk.text}

    Return JSON: {{"question": "...", "answer": "..."}}"""
    return json.loads(llm(prompt))
```

Then **review manually** — LLM-generated eval questions can be tautological, ambiguous, or leaky. Discard low-quality ones.

---

### 2.2 Retrieval Metrics

Given a query and a set of expected source chunks, measure how well retrieval surfaces them.

**Recall@k** — fraction of expected sources present in top-k retrieved:

$$\text{Recall@}k = \frac{|\text{expected} \cap \text{top-}k|}{|\text{expected}|}$$

Typical bar: **Recall@5 or Recall@10** ≥ 0.9 for a well-tuned RAG.

**Precision@k** — fraction of top-k that are expected sources. Less useful for RAG since retrieval is intentionally wide.

**MRR (Mean Reciprocal Rank)** — average of $1/\text{rank}$ of the first correct hit. Rewards getting the right answer high in the list.

$$\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q}$$

**NDCG@k (Normalized Discounted Cumulative Gain)** — weights higher-ranked results more, and supports graded relevance (not just binary):

$$\text{DCG@}k = \sum_{i=1}^k \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

Where $\text{IDCG}$ is the DCG of the ideal ranking. Values in $[0, 1]$; higher = better.

**Hit@k** — 1 if any expected source is in top-k, else 0. Averaged, tells you the fraction of queries where at least one right chunk was retrieved. Rougher than recall but useful.

**Interpretation:**

- Recall@10 tells you "does the answer exist in the retrieved set?"
- MRR tells you "how high is the first correct answer ranked?"
- NDCG penalizes low-ranked correct answers more than high-ranked ones.

Use **Recall@k as your primary retrieval metric**. Report others too for a fuller picture.

---

### 2.3 Generation Metrics — The RAGAS Framework

RAGAS (Es et al., 2023) proposed a set of metrics tailored for RAG systems, most of which use LLM-as-judge.

**1. Faithfulness** — do the generated claims follow from the retrieved context?

$$\text{Faithfulness} = \frac{|\text{claims supported by context}|}{|\text{total claims in answer}|}$$

Procedure:
1. Extract atomic claims from the answer with an LLM.
2. For each claim, ask an LLM: "Does the context entail this claim? yes/no."
3. Score = fraction of yes.

Low faithfulness = hallucinated claims. **This is the most critical RAG metric.**

**2. Answer relevance** — does the answer address the question?

Procedure:
1. Given the answer, ask an LLM to generate the *question* this answer answers.
2. Compute embedding similarity between the generated question(s) and the actual question.
3. Score = mean cosine similarity.

Low answer relevance = answer is off-topic or overly generic.

**3. Context precision** — of the retrieved context, how much is relevant to the question?

$$\text{Precision@}k = \frac{\sum_{i=1}^{k} \frac{\text{num relevant chunks in top-}i \cdot v_i}{i}}{\text{total relevant chunks}}$$

where $v_i$ = 1 if chunk $i$ is relevant, 0 otherwise. High context precision = retrieval brought relevant stuff, and brought it near the top.

**4. Context recall** — of the info needed to answer, how much is in the retrieved context?

Procedure: for each sentence in the ground-truth answer, an LLM checks whether it's attributable to the retrieved context.

**5. Answer semantic similarity** — how close is the generated answer to the ground-truth answer, semantically?

Cosine similarity of embeddings.

**6. Answer correctness** — a hybrid combining factual overlap with semantic similarity.

**RAGAS in code:**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

ds = Dataset.from_dict({
    "question": [...],
    "answer": [...],           # your system's answers
    "contexts": [[...], ...],  # retrieved contexts per question
    "ground_truth": [...],     # expected answers
})

result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
```

**Interpretation targets:**
- Faithfulness > 0.9 — most answers strictly follow context.
- Context precision > 0.6 — most retrieved chunks are on-topic.
- Context recall > 0.85 — most needed info is retrieved.
- Answer relevance > 0.85 — answers address questions.

Below these thresholds, iterate on the corresponding stage.

---

### 2.4 LLM-as-Judge (Careful Use)

Many RAG metrics — faithfulness, correctness, coherence, style — require semantic judgment that classical metrics can't provide. **LLM-as-judge** uses a strong LLM as an evaluator.

**Simple LLM-as-judge:**

```python
def judge(question, answer, context, ground_truth):
    prompt = f"""On a scale of 1-5, rate how well the answer addresses the question,
    is consistent with the context, and matches the ground truth. Return only a number.

    Question: {question}
    Context: {context}
    Ground truth: {ground_truth}
    Answer: {answer}

    Rating:"""
    return int(llm(prompt, model="gpt-4o").strip())
```

**Better patterns:**

- **Explicit criteria** — "Rate on these axes separately: factual accuracy (1-5), completeness (1-5), style (1-5)."
- **Rubrics** — describe what each score level means, so ratings are consistent.
- **Reasoning first, then rating** — ask the LLM to reason about the answer before giving a score. Improves reliability.
- **Multiple judges** — average or majority vote across 2–3 LLM judges.
- **Pairwise comparison** — instead of absolute rating, compare two answers ("A vs B: which is better?"). Often more reliable.

**Known biases:**
- **Position bias** — model prefers the first option. Randomize order.
- **Verbosity bias** — model prefers longer answers. Control for length.
- **Self-preference bias** — a model may prefer answers from itself; cross-model judging is better.
- **Style bias** — model prefers polished prose over correct but plain answers.

**Correlate with humans.** Regularly (weekly or per-release), have a human rate a sample; compute correlation between LLM judge and human. If correlation drops below ~0.7, revise the judge prompt or upgrade the judge model.

**When *not* to use LLM-as-judge:** when a cheap classical metric (exact match, ROUGE, embedding similarity) is sufficient. Don't burn LLM tokens to grade what a regex could grade.

---

### 2.5 Production Evaluation and Monitoring

Offline eval is your regression-testing safety net. Production eval is your **live signal**.

**Sampling.** For every N production requests, log:
- Question, retrieved chunks (with IDs and scores), generated answer, timestamps, model version, embedding version.
- User feedback if collected (thumbs up/down, rating, follow-up).
- Retrieval trace: query embeddings, filters applied, fusion result.

**Sample a subset for LLM-as-judge** — grade faithfulness, relevance, and answer quality on 1–5% of production traffic.

**Metrics to track over time:**

- **Faithfulness distribution** — is it drifting down? Investigate.
- **Refusal rate** — sudden jumps signal retrieval issues.
- **Retrieval hit rate** — for cases where you know the right chunk, is it in top-k?
- **User feedback ratio** — thumbs up / (up + down).
- **Latency and cost** — p50, p95, p99 latency; cost per query.

**Alerting:**

- Faithfulness < 0.85 → investigate.
- Refusal rate spike (>2× baseline) → embedding model or index broken?
- Latency p95 > SLA → scale up retrieval or infer.

**A/B testing.**

- Model changes, prompt changes, embedding upgrades — always A/B test on a slice of traffic.
- Use offline eval to gate the initial deploy; use live eval to confirm no regressions.

**Regression tests in CI.** Add your eval dataset to CI:

```python
def test_rag_regression():
    ds = load_eval_dataset()
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    assert result["faithfulness"] >= baseline["faithfulness"] - 0.02
    assert result["answer_relevancy"] >= baseline["answer_relevancy"] - 0.02
```

Every PR runs the eval; regressions block merges.

**Human-in-the-loop.** Users who complain / thumbs-down are your highest-signal signal. Route thumbs-downs to human review; auto-add failing cases to your eval set. Your eval set grows and improves over time.

---

## 3. Mental Models & Analogies

### 3.1 The "Two-Stage Report Card" Model

Grading a RAG system is like grading a research paper: you can't just look at the conclusion. You need two grades:

1. **Research quality (retrieval)** — did the author gather the right sources?
2. **Writing quality (generation)** — did the author argue faithfully from those sources?

A paper can fail on either axis:
- Bad research, good writing → confident, well-argued nonsense.
- Good research, bad writing → correct info but muddled or drifting off-topic.

Faithfulness catches the second failure; retrieval metrics catch the first. Together they diagnose RAG the way a two-part rubric grades a paper.

### 3.2 The "Speedometer + Odometer" Model

Offline eval is your **odometer** — total quality over your labeled set, run periodically. Gates deploys, catches regressions.

Live production eval is your **speedometer** — real-time quality on real users. Detects drift and outages.

You need both. A car with only an odometer is fine at inspection but doesn't tell you when you're speeding right now. A car with only a speedometer tells you the current state but not the total wear.

Together: offline eval sets the bar; production eval keeps you above it. Any RAG system without both is flying blind.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Model Answered — Ship It"

Anecdotal quality checks miss regressions. A change that helps 5 anecdotal test cases can hurt 30 others you didn't test. Always run a labeled eval before shipping — even 30 questions with expected answers is better than none.

### 4.2 "LLM-as-Judge Is Ground Truth"

It isn't. LLM judges are biased (position, verbosity, style, self-preference) and inconsistent (~5–15% variance between runs on the same input). Use them as a *scalable proxy* for human judgment, and periodically calibrate against real human ratings. Report metrics as approximate.

### 4.3 "Eval Just Measures Averages; We Ship When Average Is Fine"

Averages hide catastrophic failures. Slice metrics by category, difficulty, source type, tenant, question length. A model that averages 0.9 faithfulness but drops to 0.4 on legal questions is dangerous. Always look at slices, not just aggregates. Track worst-case p5 or p1, not just mean.

---

## 5. Self-Assessment Bank (RAG Evaluation)

### Questions

**Q1 (Short answer).** Name the two categories of RAG failure and their corresponding metrics.

**Q2 (Multiple choice).** Recall@10 = 0.9 means:
- (a) 10% of retrieved chunks are relevant.
- (b) On average, 90% of expected relevant chunks are in the top-10 retrieved.
- (c) The model is 90% accurate.
- (d) 90% of chunks embed correctly.

**Q3 (Short answer).** Describe RAGAS's "faithfulness" metric and how it's computed.

**Q4 (Multiple choice).** LLM-as-judge is:
- (a) Perfectly reliable ground truth.
- (b) A scalable but biased proxy for human judgment; use with calibration.
- (c) Only for open-source models.
- (d) A synonym for cross-encoder reranking.

**Q5 (Short answer).** Why is it critical to slice eval metrics by category / difficulty rather than only report the aggregate?

**Q6 (Multiple choice).** MRR (Mean Reciprocal Rank) rewards:
- (a) All correct results equally.
- (b) Getting the first correct result high in the ranking.
- (c) Retrieving many results.
- (d) Slow retrieval.

**Q7 (Short answer).** Describe two failure modes an LLM-as-judge can have.

**Q8 (Multiple choice).** A production RAG system's faithfulness suddenly drops from 0.92 to 0.75 over one week. First step:
- (a) Do nothing; it's noise.
- (b) Investigate retrieval — is the right chunk still being retrieved? Are embeddings stale? Did a doc source get corrupted?
- (c) Fine-tune the LLM.
- (d) Switch models.

**Q9 (Short answer).** Give one strategy to build an initial eval dataset when you have no user queries yet.

**Q10 (Multiple choice).** Regression testing your RAG in CI means:
- (a) Rerunning eval on every PR; failing merges if metrics degrade beyond a threshold.
- (b) Only testing when the model changes.
- (c) Manually reviewing every query.
- (d) Skipping tests to speed up CI.

---

### Answer Key & Detailed Explanations

**A1.** **Retrieval failures**: right chunk not returned or ranked too low. Metrics: **Recall@k, MRR, NDCG@k, Hit@k**. **Generation failures**: right chunk returned but answer is hallucinated, off-topic, or wrong. Metrics: **faithfulness, answer relevance, context precision/recall, answer correctness**.

**A2. (b).** Recall@10 = 0.9 means on average, 90% of the expected relevant chunks (from your labeled eval set) appear within the top-10 retrieved. Higher = better retrieval.

**A3.** Faithfulness measures the fraction of the model's claims that are supported by the retrieved context. Procedure: (1) use an LLM to extract atomic claims from the answer; (2) for each claim, use an LLM to judge whether the context entails it (yes/no); (3) faithfulness = fraction of "yes". Low faithfulness = hallucination.

**A4. (b).** LLM judges scale to thousands of grades cheaply but have known biases (position, verbosity, style, self-preference). They're a proxy for human judgment. Regularly calibrate against a sample of human ratings; if correlation drops, revise the judge prompt or upgrade the model.

**A5.** Averages hide failures on important subsets. A model with 0.9 aggregate faithfulness may have 0.4 on legal questions or on specific tenants — dangerous if that's a critical use case. Slicing by category, difficulty, source, tenant, and question length reveals hidden risk. Track worst-case slices, not just mean.

**A6. (b).** MRR = mean of $1/\text{rank of first correct hit}$. Getting the right answer at rank 1 scores 1.0; rank 5 scores 0.2; not in top-k scores 0. It rewards *high-ranked correct answers* more than low-ranked ones.

**A7.** Any two: (1) **Position bias** — prefers the first candidate in pairwise comparisons; fix by randomizing order. (2) **Verbosity bias** — prefers longer answers; control for length. (3) **Style bias** — prefers polished prose over plain-but-correct answers. (4) **Self-preference** — favors answers similar to what the judge model would generate; use a different model for judging. (5) **Consistency drift** — ratings vary run-to-run on the same input; run multiple times and average.

**A8. (b).** Faithfulness drops usually mean retrieval regressed — the LLM has less good context to work from, so it hallucinates more. Check: retrieval hit rate on your eval set, embedding model version, index freshness, source doc corruption, filter changes. Don't jump to LLM changes without diagnosing.

**A9.** (1) **Auto-generate** questions from your documents using an LLM ("write 3 questions this chunk answers"); manually review and clean. (2) **Domain SME** writes 50–100 questions by hand. (3) Use **public benchmarks** in your domain (MS MARCO, KILT, HotpotQA). (4) **Alpha users** — ship to a small group first, log their questions, use as eval seeds. Any one plus rationale.

**A10. (a).** Add your labeled eval dataset to CI. Every PR that changes prompts, embeddings, retrieval, or the model runs the eval and reports metrics vs baseline. Regressions beyond a threshold (say -0.02) fail the merge. Prevents silent quality drops.

---

## 6. Practice Prompts

1. **Build an eval set.** From your Build corpus, generate 50 (question, answer, expected_source) triples via LLM. Manually review and clean. Save as JSONL.
2. **RAGAS integration.** Install `ragas`; wire it into your RAG. Report faithfulness, answer relevancy, context precision, context recall on your eval set.
3. **LLM-as-judge calibration.** Rate 30 answers with (a) LLM judge, (b) yourself. Compute Pearson correlation. If below 0.7, improve the judge prompt.
4. **Slice metrics.** Tag your eval questions by category (easy / hard, HR / legal / IT). Report faithfulness per slice. Identify the worst-performing slice.
5. **CI regression test.** Add your eval as a pytest that fails if faithfulness drops below a baseline. Make a "hurting" change and confirm CI catches it.

---

## 7. References

- Es et al., ["RAGAS: Automated Evaluation of Retrieval Augmented Generation"](https://arxiv.org/abs/2309.15217) (2023).
- Zheng et al., ["Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) (2023) — biases and mitigations.
- RAGAS docs: [docs.ragas.io](https://docs.ragas.io/).
- LangSmith, Braintrust, Arize Phoenix, Weights & Biases — production LLM eval and observability platforms.
- Chen et al., ["Benchmarking Large Language Models in Retrieval-Augmented Generation"](https://arxiv.org/abs/2309.01431) (2023) — RGB benchmark.
- MTEB retrieval subset: [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (filter to Retrieval).
