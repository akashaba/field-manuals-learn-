# 04.03 — Reranking

> **Module 4: RAG** · Subtopic 3 of 5
> **Prerequisites:** **03.04 §6** (cross-encoders, LLM rerankers, latency budgeting — the introduction), 03.01 (contrastive training), 02.01 §7 (paired tests), 04.02 (retrieval ceiling, dynamic k).
> **Outcome:** you understand ranking losses mathematically, can train and distil domain rerankers, can implement pointwise, pairwise, listwise, and setwise LLM reranking with bias controls, can **calibrate** reranker scores so they drive RAG decisions (how many chunks, when to abstain), and can serve rerankers within a latency budget.

---

## 1. Why Reranking Works — and What It Should Optimise in RAG

First-stage retrievers score the query and document **independently** ($s = \langle f(q), g(d)\rangle$, or BM25). A reranker scores them **jointly**, $s = h(q, d)$, with full token-level interaction. It sees negation, qualifiers, exact numbers, and whether the passage *answers* the question or merely mentions the topic.

```
 first stage (recall@100 high, precision low)     reranker (precision@5 high)
 ┌─────────────────────────────────────┐          ┌───────────────────────────┐
 │ 100–200 candidates, cheap scoring   │ ───────► │ joint q–d scoring, top 5–20│ ──► context budget (04.02 §6)
 └─────────────────────────────────────┘          └───────────────────────────┘
```

**In RAG, topical relevance is not the target — usefulness for answering is.** A passage that restates the question is topically relevant but useless. A definitions section that the answer depends on may look off-topic. Train and evaluate rerankers against **evidence labels** ("needed to answer"), not topical labels, and:
- Rerank against the **standalone, rewritten question** (04.04), not the raw conversational turn.
- Include **metadata** in the reranker input (section path, document status, date). "Enacted, 2025" vs "introduced bill, 2021" matters.
- For instruction-following rerankers, state the domain preferences explicitly: "Prefer current enacted statutes over bills and commentary."

---

## 2. Learning-to-Rank Losses (the Math You Need)

Let $s_i = h(q, d_i)$ be the model score and $y_i$ the graded label.

**Pointwise (binary cross-entropy):**

$$
\mathcal{L}_{\text{point}} = -\sum_i \Big[y_i \log \sigma(s_i) + (1 - y_i)\log\big(1 - \sigma(s_i)\big)\Big]
$$

This is simple and yields probability-like scores, but it ignores that only the *order* matters.

**Pairwise (RankNet):** model $P(d_i \succ d_j) = \sigma(s_i - s_j)$ over pairs with $y_i > y_j$:

$$
\mathcal{L}_{\text{RankNet}} = -\sum_{(i,j):\, y_i > y_j} \log \sigma(s_i - s_j)
$$

**LambdaRank:** scale each pair's gradient by the change in NDCG from swapping the pair, which focuses learning on the top of the list:

$$
\lambda_{ij} = \frac{-\,|\Delta \text{NDCG}_{ij}|}{1 + e^{\,s_i - s_j}}
$$

**Listwise (ListNet / softmax cross-entropy):**

$$
\mathcal{L}_{\text{ListNet}} = -\sum_i \mathrm{softmax}(\mathbf{y})_i \,\log \mathrm{softmax}(\mathbf{s})_i
$$

**Localized Contrastive Estimation (LCE)** — the standard cross-encoder recipe (Gao et al., 2021) — uses one positive and $n$ **hard negatives sampled from the first-stage retriever you will actually rerank**:

$$
\mathcal{L}_{\text{LCE}} = -\log \frac{e^{s^+}}{e^{s^+} + \sum_{k=1}^{n} e^{s_k^-}}
$$

**Distillation (MarginMSE):** match the teacher's *margin* rather than its raw scores. This transfers knowledge from a large cross-encoder or LLM to a small, fast model (Hofstätter et al., 2020):

$$
\mathcal{L}_{\text{MarginMSE}} = \Big( (s^+ - s^-) - (t^+ - t^-) \Big)^2
$$

```python
import torch
import torch.nn.functional as F


def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """scores, labels: (B, n). Mean over pairs with labels_i > labels_j."""
    diff = scores[:, :, None] - scores[:, None, :]
    mask = (labels[:, :, None] > labels[:, None, :]).float()
    return (F.softplus(-diff) * mask).sum() / mask.sum().clamp(min=1)     # -log σ(x) = softplus(-x)


def listnet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -(F.softmax(labels.float(), dim=-1) * F.log_softmax(scores, dim=-1)).sum(-1).mean()


def lce_loss(pos: torch.Tensor, negs: torch.Tensor) -> torch.Tensor:
    """pos: (B,), negs: (B, n). Positive is class 0."""
    logits = torch.cat([pos[:, None], negs], dim=1)
    return F.cross_entropy(logits, torch.zeros(len(pos), dtype=torch.long, device=pos.device))


def margin_mse_loss(s_pos, s_neg, t_pos, t_neg) -> torch.Tensor:
    return F.mse_loss(s_pos - s_neg, t_pos - t_neg)
```

---

## 3. Cross-Encoder Rerankers in Practice

### 3.1 Training recipe for a domain reranker

1. **Candidates:** run your *production* first stage (hybrid, 04.02) on the training queries and keep the top 50–200. Training on candidates from a different retriever teaches the wrong distinctions.
2. **Labels:** human evidence labels where you have them. Otherwise use an **LLM judge validated against human labels** (03.04 Project 3), with graded 0–3 labels and reasoning before the grade.
3. **Loss:** LCE with 7–15 hard negatives per positive, optionally plus MarginMSE from a stronger teacher (a large cross-encoder or an LLM pointwise score).
4. **Input format:** `query [SEP] section_path | title | status | date [SEP] passage`, truncated to 256–512 tokens.
5. **Evaluation:** nDCG@10 and evidence recall at the RAG budget on a human-labelled test set, sliced by query type. Include the first-stage-only baseline.

### 3.2 Serving

The cost per query is roughly

$$
T_{\text{rerank}} \approx \frac{n_{\text{cand}}\cdot(\ell_q + \ell_d)}{\text{throughput (tokens/s)}} + T_{\text{overhead}}
$$

Levers: rerank depth $n_{\text{cand}}$ (gains usually flatten beyond 50–100), passage truncation $\ell_d$, model size (distil), batching, fp16/INT8 with ONNX Runtime or TensorRT, and **score caching** by `(query_hash, chunk_id, chunk_version, model_version)` for repeated queries. Always set a **timeout with fallback** to the fused first-stage order (03.04 §6.3).

---

## 4. LLM Rerankers

LLMs rerank well zero-shot, follow instructions, and can reason about usefulness. They are also slow, expensive, and **position-biased**.

| Strategy | Calls for n candidates | Strengths | Weaknesses |
|---|---|---|---|
| **Pointwise** (yes/no log-prob or 0–3 grade) | $n$ (parallel) | Simple, parallel, calibratable | Scores are not comparable across queries without calibration; ignores other candidates |
| **Pairwise** (PRP: "Which passage is more relevant, A or B?") | $O(n^2)$ all-pairs, or $O(nk)$ with a sliding or sorting variant | Robust; both orders cancel position bias | Many calls |
| **Listwise** (RankGPT: output a permutation of a window) | $\lceil (n - w)/s \rceil + 1$ per pass | Sees the candidates together; strong quality | Position bias; output parsing; context-length limits |
| **Setwise** (pick the best of a set of $c$; tournament or heap sort) | ≈ $O\big(\frac{n}{c-1} + k \log_c n\big)$ | Fewer calls than pairwise; good quality/cost | More complex orchestration |
| **Reasoning rerankers** (think before judging) | as above, more tokens | Better on hard, reasoning-heavy relevance | Latency; use only on the final top 10–20 |

### 4.1 Sliding-window listwise reranking

Process windows from the **bottom of the list upward**, so strong candidates "bubble" to the top in one pass. With window $w$ and stride $s$, one pass guarantees that the top $w - s$ positions are fully sorted relative to everything seen.

```python
from typing import Callable, Sequence


def sliding_window_rerank(query: str, docs: Sequence[str],
                          rank_window: Callable[[str, list[str]], list[int]],
                          window: int = 20, stride: int = 10) -> list[int]:
    """rank_window(query, window_docs) -> permutation of range(len(window_docs)), best first.
    Returns a permutation of range(len(docs)), best first."""
    order = list(range(len(docs)))
    end = len(order)
    start = max(0, end - window)
    while True:
        idx = order[start:end]
        perm = rank_window(query, [docs[i] for i in idx])
        if sorted(perm) != list(range(len(idx))):             # malformed output: keep the window's order
            perm = list(range(len(idx)))
        order[start:end] = [idx[p] for p in perm]
        if start == 0:
            return order
        end -= stride
        start = max(0, end - window)
```

### 4.2 Controlling position bias

- **Pairwise:** ask both orders (A,B) and (B,A). Count a win only when the two answers are consistent; otherwise treat it as a tie.
- **Listwise:** **permutation self-consistency** (Tang et al., 2023). Rerank several random shuffles of the same candidates, then aggregate the rankings (Borda count, or a Kemeny approximation).
- **Measure the bias:** for a fixed set of relevant passages, plot selection frequency against input position. A slope means your prompt or model needs these controls.

```python
import random
from collections import defaultdict
from typing import Callable


def prp_allpairs(query, docs, prefer: Callable[[str, str, str], str]) -> list[int]:
    """prefer(query, a, b) -> 'A' | 'B'. Both orders are asked; inconsistent answers count as ties."""
    score = defaultdict(float)
    n = len(docs)
    for i in range(n):
        for j in range(i + 1, n):
            ab, ba = prefer(query, docs[i], docs[j]), prefer(query, docs[j], docs[i])
            if ab == "A" and ba == "B":
                score[i] += 1
            elif ab == "B" and ba == "A":
                score[j] += 1
            else:
                score[i] += 0.5
                score[j] += 0.5
    return sorted(range(n), key=lambda i: -score[i])


def permutation_self_consistency(query, docs, rerank: Callable[[str, list[str]], list[int]],
                                 n_shuffles: int = 5, seed: int = 0) -> list[int]:
    """Borda aggregation over rerankings of shuffled inputs."""
    rng, n = random.Random(seed), len(docs)
    points = defaultdict(float)
    for _ in range(n_shuffles):
        perm = list(range(n))
        rng.shuffle(perm)
        ranked = rerank(query, [docs[p] for p in perm])        # indices into the shuffled list
        for pos, local in enumerate(ranked):
            points[perm[local]] += n - pos
    return sorted(range(n), key=lambda i: -points[i])
```

### 4.3 Practical guidance

- Use an LLM reranker as a **final stage** on ≤ 20–30 candidates, after a cross-encoder. Or use it **offline**, to label training data for a cross-encoder you then serve.
- Force a structured output (a JSON array of IDs, via 02.03) and validate it as a permutation. Fall back gracefully.
- Use short, stable passage IDs (`[1]`…`[20]`) rather than long IDs. This reduces errors and tokens.
- Cache by `(query, candidate set hash, model, prompt version)`.

---

## 5. Calibration: Turning Scores into Decisions

RAG needs rerankers to answer **"how many chunks?"** and **"is anything relevant at all?"** (04.02 §6). Raw scores (logits, similarities) are not probabilities, and their scale shifts across queries and model versions. Calibrate on a labelled development set.

**Platt scaling:**

$$
\hat p(\text{relevant} \mid s) = \sigma(a\,s + b), \qquad (a, b) = \arg\min_{a,b} \sum_i \mathrm{BCE}\big(y_i,\ \sigma(a s_i + b)\big)
$$

(Isotonic regression is the non-parametric alternative when you have more data.) Then choose the threshold $\tau$ for a **target precision** on the development set, and evaluate the results on held-out data:

- keep chunks with $\hat p \ge \tau$;
- **abstain** when none pass;
- report the **expected calibration error**, $\text{ECE} = \sum_b \frac{|B_b|}{N}\,|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)|$.

```python
import numpy as np


def fit_platt(scores: np.ndarray, labels: np.ndarray, iters: int = 200, lr: float = 0.1):
    """Fit p = sigmoid(a*s + b) by gradient descent on log loss (standardised scores for stability)."""
    mu, sd = scores.mean(), scores.std() or 1.0
    z = (scores - mu) / sd
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b)))
        g = p - labels
        a -= lr * float((g * z).mean())
        b -= lr * float(g.mean())
    return lambda s: 1 / (1 + np.exp(-(a * (np.asarray(s) - mu) / sd + b)))


def threshold_for_precision(probs: np.ndarray, labels: np.ndarray, target: float = 0.9) -> float:
    order = np.argsort(-probs)
    tp = np.cumsum(labels[order])
    prec = tp / np.arange(1, len(order) + 1)
    ok = np.where(prec >= target)[0]
    return float(probs[order][ok.max()]) if len(ok) else 1.0


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo) & (probs <= hi)
        if m.any():
            ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(ece)
```

**Recalibrate** whenever the reranker model, the input format, or the first-stage retriever changes. Monitor the score distributions in production for drift.

---

## 6. Diversity and Redundancy After Reranking

Rerankers score each candidate independently, so the top 5 may be five near-copies of the same paragraph (amended and original versions, or overlapping chunks). Before assembling the context:
- **Deduplicate** by chunk overlap (offsets) and near-duplicate text (02.02 §8).
- **Merge siblings into parents** (04.01 §4.1).
- **MMR** over reranker scores and embeddings (02.01 §3.1) for multi-aspect questions.

---

## 7. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Reranker trained on the wrong candidates** | Offline gains don't show in production | Mine negatives from the *production* first stage; retrain when the retriever changes |
| **Topical ≠ useful** | Top chunks restate the question | Evidence-based labels; metadata in the input; instruction-following rerankers |
| **Latency blow-ups** | P99 over the SLO at peak | Cap the depth; truncate passages; distilled model; batching; timeout fallback |
| **LLM position bias** | The first/last candidates are over-selected | Both-order pairwise; permutation self-consistency; bias measurement |
| **Malformed listwise output** | Missing or duplicate IDs | Structured output; permutation validation; fallback to the input order |
| **Uncalibrated thresholds** | Too many or too few chunks; wrong abstentions | Platt/isotonic calibration; precision-targeted thresholds; ECE monitoring |
| **Redundant top-k** | Context full of near-duplicates | Dedup, parent merging, MMR |
| **Cost of LLM reranking** | Token bills | Use the LLM only for the final 10–20 candidates, or offline for labels; cache |

---

## 8. Hands-On Projects

### Project 1 — Train and Distil a Domain Cross-Encoder

**User stories**
- *As a RAG engineer*, I want a small reranker trained on our legislative queries that matches a large model's quality at a fraction of the latency, so that we can rerank 100 candidates under 60 ms.

**Acceptance criteria**
1. Training data: ≥ 10k queries (real + synthetic, filtered), with candidates from the production hybrid retriever (04.02) and labels from a validated LLM judge. Test set: ≥ 300 human-labelled queries.
2. Trains a small cross-encoder (~20–150M parameters) with LCE (§2) and hard negatives, and a variant with MarginMSE distillation from a large cross-encoder or LLM teacher.
3. Reports nDCG@10, MRR@10, and evidence recall at 2k tokens against: the first stage only, an off-the-shelf small reranker, the large teacher, and your distilled student. The student reaches ≥ 95% of the teacher's nDCG@10.
4. Serving: an ONNX/INT8 export with measured P50/P95 latency for 100 candidates on the target hardware, plus a timeout fallback.
5. End-to-end RAG answer accuracy improves over no reranking, with paired significance.

**Step-by-step**
1. Generate the candidate pools and judge labels (batch APIs for cost), and validate the judge on 200 human pairs.
2. Implement the training loop with `lce_loss` and `margin_mse_loss` (sentence-transformers `CrossEncoder` or your own).
3. Train the teacher-scored variant; tune the negatives count and the truncation length.
4. Export to ONNX, quantise, and benchmark with batching.
5. Integrate behind the retrieval API and run the RAG evaluation.

---

### Project 2 — LLM Reranker Lab: Strategies, Bias, and Cost

**User stories**
- *As an applied scientist*, I want to know which LLM reranking strategy gives the best quality per dollar and how strong position bias is, so that we configure the final rerank stage responsibly.

**Acceptance criteria**
1. Implements pointwise (log-prob and graded), pairwise (PRP all-pairs and a sliding variant), listwise (sliding window, §4.1), and setwise reranking with the same LLM.
2. On a BEIR dataset and your domain set: nDCG@10, the number of LLM calls, input and output tokens, cost, and latency for n ∈ {20, 50, 100}.
3. Position-bias study: selection rate vs input position, with and without both-order pairing and permutation self-consistency (3 and 5 shuffles).
4. Robustness: the rate of malformed outputs and the effectiveness of fallbacks.
5. A recommendation table: strategy × depth for three latency and cost tiers.

**Step-by-step**
1. Build a prompt library per strategy, with structured outputs and ID-based passages.
2. Implement `sliding_window_rerank`, `prp_allpairs`, `permutation_self_consistency`, and a setwise tournament.
3. Cache all LLM calls by `(prompt hash)` to make the experiments reproducible and cheap to re-run.
4. Run the grid; compute the metrics and costs; plot quality against cost.
5. Write the analysis, including the bias plots.

---

### Project 3 — Calibrated Reranker Gate: Dynamic k and Abstention in RAG

**User stories**
- *As a product owner*, I want the assistant to use only the chunks that matter, and to say "the provided sources don't cover this" instead of guessing, so that attorneys trust it.

**Acceptance criteria**
1. Calibrates the reranker with Platt and isotonic regression on a development set, and reports ECE before and after (target ECE < 0.05).
2. Implements a gate: keep chunks with $\hat p \ge \tau$ (τ chosen for target precisions of 0.8, 0.9, and 0.95); abstain when none pass.
3. The test set includes ≥ 100 **unanswerable** questions (the evidence is absent from the corpus) alongside answerable ones.
4. Metrics: answer accuracy on answerable questions, abstention precision and recall on unanswerable ones, hallucination rate (judge-validated), and average context tokens. Compared with a fixed top-k baseline.
5. Monitoring: score-distribution drift alerts and a documented recalibration procedure.

**Step-by-step**
1. Collect the reranker scores and labels on the development set; fit the calibrators (`fit_platt`, isotonic via scikit-learn).
2. Choose τ with `threshold_for_precision`, and validate it on held-out data.
3. Add the gate to the RAG pipeline, plus an abstention response template.
4. Build the unanswerable set by removing the gold documents or writing out-of-scope questions.
5. Evaluate against the baseline, and set up the drift monitoring.

---

## 9. Foundational Papers (exact titles)

**Learning to rank**
- Burges et al., 2005 — *Learning to Rank using Gradient Descent* (RankNet)
- Burges, 2010 — *From RankNet to LambdaRank to LambdaMART: An Overview*
- Cao et al., 2007 — *Learning to Rank: From Pairwise Approach to Listwise Approach* (ListNet)
- Liu, 2009 — *Learning to Rank for Information Retrieval* (monograph)

**Neural rerankers**
- Nogueira & Cho, 2019 — *Passage Re-ranking with BERT*
- Nogueira et al., 2020 — *Document Ranking with a Pretrained Sequence-to-Sequence Model*
- Gao, Dai & Callan, 2021 — *Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline*
- Hofstätter et al., 2020 — *Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation*
- Lin, Nogueira & Yates, 2021 — *Pretrained Transformers for Text Ranking: BERT and Beyond*

**LLM reranking**
- Sun et al., 2023 — *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents*
- Qin et al., 2023 — *Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting*
- Zhuang et al., 2024 — *A Setwise Approach for Effective and Highly Efficient Zero-Shot Ranking with Large Language Models*
- Tang et al., 2023 — *Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models*
- Pradeep, Sharifymoghaddam & Lin, 2023 — *RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!*
- Weller et al., 2025 — *Rank1: Test-Time Compute for Reranking in Information Retrieval*

**Calibration**
- Platt, 1999 — *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*
- Guo et al., 2017 — *On Calibration of Modern Neural Networks*

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **sentence-transformers `CrossEncoder`** | Training and inference for cross-encoders |
| **RankLLM**, **rerankers** (Python) | LLM and cross-encoder reranking implementations |
| **Hosted rerank APIs** (Cohere, Voyage, Jina, cloud providers) | Managed rerankers; check latency, limits, data policies |
| **ONNX Runtime / TensorRT / Text Embeddings Inference (rerank endpoint)** | Fast serving |
| **LightGBM (lambdarank)** | Feature-based learning to rank |
| **scikit-learn** (`CalibratedClassifierCV`, `IsotonicRegression`) | Calibration |
| **ranx / ir_measures** | Metrics and significance |
