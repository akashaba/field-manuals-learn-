# 03.04 — Hybrid Retrieval

> **Module 3: Representations** · Subtopic 4 of 4
> **Prerequisites:** 03.01 (dense embeddings, IR metrics), 03.02 (ANN), 03.03 (storage and filters), 02.02 §3 (RRF and context assembly), basic probability.
> **Outcome:** you can build a multi-stage retrieval system that combines lexical, learned-sparse, dense, and late-interaction signals. You fuse them with methods chosen on evidence, add rerankers within a latency budget, route queries by type, and evaluate the whole cascade with trustworthy relevance labels.

> **Relationship to 02.02:** 02.02 introduced RRF as a way to assemble context from several sources. This subtopic is the retrieval-engineering deep dive: *why* the signals are complementary, all the major scoring models, fusion trade-offs, reranking cascades, and evaluation.

---

## 1. Why Hybrid: Complementary Failure Modes

| Query | Dense-only failure | Lexical-only failure |
|---|---|---|
| `"HB 123"`, `"2-18-303"`, `"ORA-01555"`, a person's name | Identifiers and rare tokens get blurred into a generic region of vector space | ✔ exact match wins |
| "Can a county refuse to fund the sheriff's office?" | ✔ paraphrase / semantic match | Vocabulary mismatch ("refuse to fund" vs "appropriation denial") |
| New jargon after the embedding model's training cutoff | Poorly placed in vector space | ✔ matches literally |
| Misspellings, morphology | ✔ fairly robust | Fails without analyzers or fuzziness |
| Out-of-domain corpora (zero-shot) | Often degrades sharply | BM25 is a strong, robust baseline (BEIR) |

Hybrid retrieval exploits the fact that errors are **weakly correlated**. Candidates missed by one retriever are often found by the other, so the union has higher recall, and a good fusion or reranker recovers the precision.

```
                          ┌──────────── query understanding ─────────────┐
 user query ─────────────►│ normalise · detect identifiers · rewrite/     │
                          │ expand · route · build filters (ACL, tenant)  │
                          └───┬─────────────┬───────────────┬────────────┘
                              ▼             ▼               ▼
 STAGE 1 (recall)        BM25 / BM25F   learned sparse   dense ANN        each returns top ~100–1000
 ~10–50 ms               (inverted idx) (SPLADE)        (HNSW/IVF)
                              └─────────────┼───────────────┘
                                            ▼
 FUSION                         RRF · convex combination · learned                → top ~100–200
                                            ▼
 STAGE 2 (precision)        late interaction (ColBERT) or cross-encoder rerank    → top ~20–50
 ~30–150 ms                                 ▼
 STAGE 3 (optional)         LLM listwise rerank / dedup / diversity (MMR)         → top ~5–10 → LLM context
```

![LLM-03-2](/03.%20Representations/images/LLM-03-2.jpg)

---

## 2. Lexical Retrieval: BM25 Done Properly

### 2.1 The scoring function

For query terms $t \in q$, term frequency $f(t,d)$, document length $|d|$, and average document length $\overline{|d|}$:

$$
\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{IDF}(t)\cdot \frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\overline{|d|}}\right)},
\qquad
\mathrm{IDF}(t) = \ln\!\left(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\right)
$$

- $k_1 \in [0.9, 2.0]$ controls **term-frequency saturation** (repeating a term has diminishing returns).
- $b \in [0, 1]$ controls **length normalisation** (0.75 is the default; lower it for corpora where long documents are legitimately more relevant).
- The IDF variant shown (Lucene's) never goes negative.

**BM25F** handles multi-field documents (title, headings, body, citations): combine the field term frequencies *before* saturation, $\tilde f(t,d) = \sum_{\phi} w_\phi\, f_\phi(t,d)$, with per-field length normalisation. This beats summing per-field BM25 scores.

### 2.2 Analysis is where lexical search is won or lost

- **Identifier-aware tokenisation.** A standard analyzer splits `2-18-303` into `2`, `18`, `303`, which destroys the most valuable query type in legal and legislative search. Add a **keyword** or **pattern-tokenised** subfield for citations, bill numbers, and codes, and search it with a boost.
- **Stemming and lemmatisation** for recall on prose fields. Keep an unstemmed field for exact phrases.
- **Synonym expansion** for domain abbreviations (`DUI ↔ driving under the influence`), applied at query time so it can be edited without reindexing.
- **Phrase and proximity boosts** for multi-word terms of art.

```python
import math
import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"[A-Za-z]{1,3}-\d+|\d+(?:-\d+)+|\w+")   # keep 'HB-123', '2-18-303' intact


def analyze(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: dict[str, str]):
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.len: dict[str, int] = {}
        for doc_id, text in docs.items():
            tf = Counter(analyze(text))
            self.len[doc_id] = sum(tf.values())
            for term, f in tf.items():
                self.postings[term].append((doc_id, f))
        self.N = len(docs)
        self.avgdl = sum(self.len.values()) / max(self.N, 1)
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for term in set(analyze(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, f in self.postings[term]:          # term-at-a-time over the inverted index
                norm = self.k1 * (1 - self.b + self.b * self.len[doc_id] / self.avgdl)
                scores[doc_id] += idf * f * (self.k1 + 1) / (f + norm)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]
```

Production engines add **dynamic pruning** (WAND, Block-Max WAND/MaxScore) to skip documents that cannot enter the top-k, compressed postings, and caching.

---

## 3. Learned Sparse Retrieval (SPLADE)

Learned sparse models keep the **inverted-index** execution model but *learn* term weights **and expansions**. A document about "appropriations" can receive weight on "budget" and "funding" even if those words never appear in it.

SPLADE uses a masked-language-model head. For input tokens $i$ and vocabulary entries $j$, with MLM logits $w_{ij}$:

$$
w_j = \max_{i \in \text{tokens}} \log\!\big(1 + \mathrm{ReLU}(w_{ij})\big),
\qquad
s(q, d) = \sum_{j \in V} w_j^{(q)}\, w_j^{(d)}
$$

Sparsity is enforced with the **FLOPS regulariser**, which penalises terms that are active across many documents (the cost of posting lists):

$$
\mathcal{L}_{\text{FLOPS}} = \sum_{j \in V}\Big(\frac{1}{B}\sum_{d \in \text{batch}} w_j^{(d)}\Big)^2
$$

```python
import torch


@torch.inference_mode()
def splade_encode(texts, mlm_model, tokenizer, top_k: int | None = 256):
    """Returns a list of {token: weight} sparse vectors from an MLM-head model (SPLADE-style)."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    logits = mlm_model(**enc).logits                                    # (B, T, V)
    w = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
    w = w.max(dim=1).values                                             # (B, V) max-pool over tokens
    out = []
    for row in w:
        nz = row.nonzero().squeeze(-1)
        vals = row[nz]
        if top_k and len(nz) > top_k:
            keep = vals.topk(top_k).indices
            nz, vals = nz[keep], vals[keep]
        out.append({tokenizer.convert_ids_to_tokens(int(i)): float(v) for i, v in zip(nz, vals)})
    return out
```

**Serving:** store the weights in an inverted index (OpenSearch neural sparse, Elasticsearch sparse vectors, Qdrant sparse vectors, pgvector `sparsevec`, Vespa). SPLADE-class models often beat BM25 substantially in-domain and keep lexical matching of identifiers. **The cost** is a transformer pass per document at index time, larger postings than BM25, and sometimes a query encoder at search time (inference-free variants use only a query-side lookup).

---

## 4. Late Interaction (ColBERT)

ColBERT keeps **one vector per token** and scores with **MaxSim**:

$$
s(q, d) = \sum_{i=1}^{|q|} \max_{j = 1}^{|d|} \langle \mathbf{E}_{q_i}, \mathbf{E}_{d_j} \rangle
$$

It is more expressive than a single vector (each query term finds its best match), and much cheaper than a cross-encoder (documents are pre-encoded).

```python
def maxsim(Q: torch.Tensor, D: torch.Tensor, d_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Q: (Lq, dim), D: (N, Ld, dim), both L2-normalised; d_mask: (N, Ld) bool. Returns (N,) scores."""
    sim = torch.einsum("qh,nlh->nql", Q, D)                             # (N, Lq, Ld)
    if d_mask is not None:
        sim = sim.masked_fill(~d_mask[:, None, :], float("-inf"))
    return sim.max(dim=-1).values.sum(dim=-1)
```

**Costs and engineering:** storage scales with the number of tokens (hundreds of vectors per passage). ColBERTv2 uses residual compression (~20–40 bytes per token vector), and PLAID uses centroid-based candidate generation to make search fast. In practice ColBERT is used either as a **first-stage** retriever (with PLAID) or as a **reranker** over fused candidates. The second option is simpler, because you only store token vectors for the candidates you rerank, or you compute them on the fly.

---

## 5. Fusion

### 5.1 Rank-based: RRF

$\mathrm{RRF}(d) = \sum_r \frac{1}{k + \mathrm{rank}_r(d)}$, with $k \approx 60$ (see 02.02 §3.1 for the implementation). It needs no score calibration and is robust. **Weaknesses:** it throws away score *magnitudes* — a document that one retriever found overwhelmingly relevant gets the same credit as a marginal one at the same rank — and it is sensitive to $k$.

### 5.2 Score-based: convex combination

$$
s(d) = \alpha\cdot \phi\big(s_{\text{dense}}(d)\big) + (1-\alpha)\cdot \phi\big(s_{\text{lex}}(d)\big)
$$

$\phi$ is a normalisation function: min-max over the candidate list, z-score, or **theoretical min-max** (dividing by the score's known bounds, e.g. $[-1,1]$ for cosine). Documents missing from one list receive that list's minimum normalised score.

**Evidence:** a systematic analysis (Bruch et al., 2023) found that a **tuned convex combination generally outperforms RRF** and needs only a small number of labelled queries to tune α. RRF remains a good zero-data default.

```python
import numpy as np


def normalize(scores: dict[str, float], method: str = "minmax", bounds=None) -> dict[str, float]:
    if not scores:
        return {}
    v = np.array(list(scores.values()), dtype=float)
    if method == "minmax":
        lo, hi = v.min(), v.max()
    elif method == "theoretical":
        lo, hi = bounds
    elif method == "zscore":
        mu, sd = v.mean(), v.std() or 1.0
        return {k: (s - mu) / sd for k, s in scores.items()}
    else:
        raise ValueError(method)
    rng = (hi - lo) or 1.0
    return {k: (s - lo) / rng for k, s in scores.items()}


def convex_fusion(dense: dict[str, float], lexical: dict[str, float], alpha: float = 0.6,
                  method: str = "minmax", dense_bounds=(-1.0, 1.0)) -> list[tuple[str, float]]:
    nd = normalize(dense, method, dense_bounds if method == "theoretical" else None)
    nl = normalize(lexical, "minmax" if method == "theoretical" else method)   # BM25 is unbounded
    fill_d, fill_l = (min(nd.values()) if nd else 0.0), (min(nl.values()) if nl else 0.0)
    ids = set(nd) | set(nl)
    fused = {i: float(alpha * nd.get(i, fill_d) + (1 - alpha) * nl.get(i, fill_l)) for i in ids}
    return sorted(fused.items(), key=lambda kv: -kv[1])


def tune_alpha(queries, dense_fn, lex_fn, qrels, metric, grid=np.linspace(0, 1, 11)):
    """Pick alpha maximising mean metric on a dev set. metric(ranked_ids, qrels_for_q) -> float."""
    best = max(grid, key=lambda a: np.mean([
        metric([d for d, _ in convex_fusion(dense_fn(q), lex_fn(q), a)], qrels[q]) for q in queries]))
    return float(best)
```

### 5.3 Learned fusion and query-adaptive weights

- **Learning to rank** (LightGBM LambdaMART) over features: each retriever's score and rank, BM25 per field, document freshness, authority, and query features (length, identifier present).
- **Query-adaptive α:** route identifier-heavy queries toward lexical and natural-language questions toward dense (§7). Even a regex-based router captures most of the gain in identifier-rich domains.

---

## 6. Reranking

### 6.1 Cross-encoders

Concatenate `[CLS] query [SEP] document [SEP]`, pass it through a transformer, and output a relevance score. Full query–document attention makes cross-encoders the most accurate practical rerankers (monoBERT, monoT5, modern multilingual rerankers). **Cost:** one forward pass per candidate. Reranking 100 candidates × 256–512 tokens is ~25–50k tokens per query, which is tens of milliseconds on a GPU for a small model and too slow for large ones.

### 6.2 LLM rerankers

- **Pointwise:** ask "Is this passage relevant?" and use the log-prob of "yes" as the score.
- **Listwise:** give the LLM a window of candidates and ask for an ordering (RankGPT), using sliding windows for more than ~20 candidates. The strongest zero-shot quality and the ability to follow **instructions** ("prefer enacted statutes over bills"), but high latency and cost. Use them as a final stage on ≤ 20–30 candidates, or offline to create training labels.

### 6.3 Latency budgeting

$$
T_{\text{total}} \approx T_{\text{rewrite}} + \max(T_{\text{BM25}}, T_{\text{sparse}}, T_{\text{dense}}) + T_{\text{fuse}} + T_{\text{rerank}}(n_{\text{cand}}, \bar\ell) + T_{\text{LLM-rerank}}
$$

Run the first-stage retrievers **in parallel**. The main levers are the rerank depth $n_{\text{cand}}$ and the candidate length $\bar\ell$ (rerank passages, not whole documents). Measure nDCG@10 as a function of rerank depth: gains usually flatten after 50–100 candidates.

```python
import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class Retriever:
    name: str
    search: Callable[[str, int], Awaitable[dict[str, float]]]   # query, k -> {doc_id: score}
    k: int = 100


async def hybrid_search(query: str, retrievers: list[Retriever],
                        fuse: Callable[[dict[str, dict[str, float]]], list[tuple[str, float]]],
                        rerank: Callable[[str, list[str]], Awaitable[list[tuple[str, float]]]] | None = None,
                        rerank_depth: int = 50, final_k: int = 10, timeout_s: float = 0.2):
    async def run(r: Retriever):
        try:
            return r.name, await asyncio.wait_for(r.search(query, r.k), timeout_s)
        except asyncio.TimeoutError:
            return r.name, {}                                  # degrade gracefully, don't fail the query
    results = dict(await asyncio.gather(*(run(r) for r in retrievers)))
    fused = fuse(results)
    if rerank is None:
        return fused[:final_k], results
    reranked = await rerank(query, [d for d, _ in fused[:rerank_depth]])
    return reranked[:final_k], results
```

---

## 7. Query Understanding

- **Identifier detection and routing:** regexes for bill numbers, statute citations, case numbers, and error codes. When found, add an exact-match lexical clause with a high boost, and shift α toward lexical.
- **Rewriting for multi-turn conversations:** rewrite "what about the Senate version?" into a standalone query using the chat history (an LLM call with a strict output schema; 02.03).
- **Expansion:**
  - **HyDE:** generate a hypothetical answer, embed it, and search with that vector.
  - **Query2doc:** append a generated pseudo-document to the query for BM25.
  - **Multi-query:** run N paraphrases and fuse the results.

  These help recall on vague queries, but add latency and can drift, so gate them by query type.
- **Filters from language:** "bills from the 2025 session about water rights" → `session=2025` plus the topic query. Extract filters with structured output, then **validate** them against allowed values.

```python
import re

ID_PATTERNS = {
    "bill": re.compile(r"\b(?:HB|SB|HJ|SJ)[\s-]?\d{1,4}\b", re.I),
    "code_section": re.compile(r"\b\d{1,2}-\d{1,3}-\d{1,4}\b"),
}


def route_query(q: str) -> dict:
    ids = {name: [m.group(0).upper().replace(" ", "-") for m in p.finditer(q)] for name, p in ID_PATTERNS.items()}
    ids = {k: v for k, v in ids.items() if v}
    words = len(q.split())
    if ids and words <= 4:
        return {"alpha": 0.2, "exact_terms": ids, "expand": False}       # mostly an identifier lookup
    if ids:
        return {"alpha": 0.4, "exact_terms": ids, "expand": False}       # identifier + intent
    return {"alpha": 0.7, "exact_terms": {}, "expand": words >= 6}       # natural-language question
```

---

## 8. Evaluating Hybrid Systems

### 8.1 Build a trustworthy test collection

1. **Queries:** sample real queries (logs, support tickets, attorney requests), stratified by type — identifier, keyword, natural-language question, multi-hop. Add synthetic queries only to fill gaps, and keep them labelled as synthetic.
2. **Pooling:** for each query, take the union of the top-k results from *every* system you will compare (BM25, dense, hybrid, reranked), and judge the whole pool. Unjudged documents create bias against new systems.
3. **Graded judgments** (0–3), with written guidelines. Measure inter-annotator agreement on a subset.
4. **LLM-assisted judging:** LLM judges with a careful rubric can approximate human preferences well. Validate the judge on a human-labelled subset (Cohen's κ), audit disagreements, and never let the same model both *generate* answers and *judge* its own retrieval without calibration.

### 8.2 Report the cascade, not just the end

| Stage | Metric | Why |
|---|---|---|
| Each first-stage retriever | Recall@100/@1000 | Upper bound on what later stages can do |
| Fusion | Recall@k at rerank depth, nDCG@10 | Did fusion keep the union's recall? |
| Reranker | nDCG@10, MRR@10 | Precision at the top |
| End-to-end RAG | Answer accuracy / groundedness | What users experience (later module) |
| Per query type | All of the above, sliced | Hybrid gains concentrate in specific slices |

Use **paired significance tests** (randomisation tests or paired bootstrap) across queries (see 02.01 §7.2). `ranx` and `ir_measures` provide these.

---

## 9. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Identifiers not found** | "HB 123" returns unrelated bills | Identifier-aware analyzers, keyword subfields, exact-match boosts, routing (§2.2, §7) |
| **Fusion hurts precision** | Hybrid worse than dense on some slices | Tune α per query type; convex over RRF when you have labels; add a reranker |
| **Score scale drift** | Fusion quality changes after a model or index update | Rank-based fusion or per-query normalisation; re-tune after every model change |
| **Reranker latency** | P99 blows the SLO | Cap rerank depth; truncate passages; smaller/distilled rerankers; GPU batching; timeouts with fallback to the fused order |
| **One retriever down or slow** | Whole query fails | Per-retriever timeouts; degrade to the available signals (§6.3 code) |
| **Keeping two indexes consistent** | Lexical and vector results disagree on freshness | One ingestion pipeline writing both (03.03 §5), or a single engine that supports both |
| **Eval set bias** | New system "loses" because its results were never judged | Pool across all systems; judge new results before comparing |
| **Query rewriting drift** | Rewritten query loses the user's constraint | Keep the original query in the lexical leg; validate extracted filters; A/B test rewriting |

---

## 10. Hands-On Projects

### Project 1 — Hybrid Legislative Search Engine

**User stories**
- *As a legislative attorney*, I want to type either a citation ("2-18-303", "HB 45") or a plain-English question and get the right sections at the top, so that I can research faster than keyword search allows.
- *As the search owner*, I want per-query-type metrics, so that we know hybrid search helps every slice.

**Acceptance criteria**
1. The corpus includes statutes, bills, and fiscal notes (or another identifier-rich domain), indexed in OpenSearch/Elasticsearch with identifier-aware analyzers and BM25F over title, headings, body, and citations. The dense index uses pgvector or the engine's native kNN.
2. A labelled test set of ≥ 300 queries across ≥ 4 query types, with pooled graded judgments (§8.1).
3. Systems compared: BM25F, dense, RRF, tuned convex fusion, convex + cross-encoder rerank (depth 50), and query routing (§7) + fusion + rerank.
4. Reports Recall@100, nDCG@10, and MRR per query type, with paired significance tests and P50/P95 latency. The final system beats both single retrievers on every slice or explains why not.
5. A search API (Spring Boot or FastAPI) implementing the §6.3 async cascade with timeouts and graceful degradation.

**Step-by-step**
1. Design the index mappings, with a keyword subfield and a pattern tokenizer for citations and bill IDs. Verify with the `_analyze` API.
2. Build the dense index with your 03.01 fine-tuned model (or a strong baseline model).
3. Implement `route_query`, `convex_fusion`, and `tune_alpha` (per query type), plus an RRF baseline.
4. Add a cross-encoder reranker service (sentence-transformers `CrossEncoder`, or a hosted rerank API) with batching.
5. Create the judgment pool and label it (humans + a validated LLM judge).
6. Run the evaluation with `ranx`, and write up the per-slice results and latency.

---

### Project 2 — Fusion and Reranking Trade-Off Study

**User stories**
- *As a retrieval engineer*, I want to know which fusion method and reranker depth give the best nDCG per millisecond on our data, so that we configure the cascade on evidence.

**Acceptance criteria**
1. Uses two datasets: your domain set plus a public BEIR dataset (e.g. SciFact, FiQA, or TREC-COVID).
2. Fusion methods: RRF with k ∈ {10, 30, 60, 100}; convex with min-max, z-score, and theoretical normalisation (α tuned on 20%, 50%, and 100% of the dev queries); and LambdaMART over rank and score features.
3. Rerankers: none, a small cross-encoder, a large cross-encoder, and an LLM listwise reranker on the top 20. Depths {20, 50, 100, 200}.
4. Outputs a Pareto frontier of nDCG@10 vs P95 latency, and analyses how many labelled queries convex fusion needs to beat RRF.
5. All results come with confidence intervals and are reproducible from a single config file.

**Step-by-step**
1. Precompute and cache the first-stage runs (BM25, dense, SPLADE) as TREC run files.
2. Implement the fusion variants over the cached runs (fast iteration, no re-querying).
3. Train LambdaMART with LightGBM on features from the runs, using cross-validation by query.
4. Run the rerankers with batching and measure latency on fixed hardware.
5. Plot the results and write the analysis.

---

### Project 3 — Relevance Judgments at Scale with LLMs, and a Domain Reranker

**User stories**
- *As a search team with no labelling budget*, we want reliable relevance labels generated with LLM assistance, validated against a small human set, so that we can evaluate systems and train a domain reranker.

**Acceptance criteria**
1. Human-labels ≥ 200 query–document pairs (graded 0–3) with guidelines. Measures inter-annotator κ on 50 of them.
2. An LLM judge prompt with a rubric, structured output (02.03), and a reasoning-before-grade field. Reports agreement with humans (κ and a confusion matrix), and iterates the prompt until κ ≥ the human–human agreement or explains the gap.
3. Labels ≥ 20k pairs from pooled candidates with the validated judge. Estimates the cost and the label-noise rate.
4. Fine-tunes a cross-encoder on the LLM labels. On the human-labelled test set it beats the off-the-shelf reranker by ≥ 3 nDCG@10 points (with significance).
5. Documents the bias risks (the judge prefers its own style, position bias) and the mitigations used (shuffled order, multiple judges, calibration).

**Step-by-step**
1. Write the labelling guidelines, and label with two annotators.
2. Build the judge prompt; run it on the human set; compute κ; iterate.
3. Generate the pools from BM25, dense, and hybrid runs; judge them in batch mode (use the batch APIs for cost).
4. Train the reranker (sentence-transformers `CrossEncoder`, with binary or graded loss) using hard negatives from the pools.
5. Evaluate on the human test set, and write up the cost, quality, and risks.

---

## 11. Foundational Papers (exact titles)

**Lexical and sparse**
- Robertson & Zaragoza, 2009 — *The Probabilistic Relevance Framework: BM25 and Beyond*
- Broder et al., 2003 — *Efficient Query Evaluation using a Two-Level Retrieval Process* (WAND)
- Ding & Suel, 2011 — *Faster Top-k Document Retrieval Using Block-Max Indexes*
- Nogueira et al., 2019 — *Document Expansion by Query Prediction*
- Formal et al., 2021 — *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*
- Formal et al., 2021 — *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval*
- Lin, 2021 — *A Proposed Conceptual Framework for a Representational Approach to Information Retrieval*

**Dense, late interaction, and hybrid**
- Khattab & Zaharia, 2020 — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*
- Santhanam et al., 2022 — *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*
- Santhanam et al., 2022 — *PLAID: An Efficient Engine for Late Interaction Retrieval*
- Kuzi et al., 2020 — *Leveraging Semantic and Lexical Matching to Improve the Recall of Document Retrieval Systems: A Hybrid Approach*
- Chen et al., 2022 — *Out-of-Domain Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models*
- Bruch, Gai & Ingber, 2023 — *An Analysis of Fusion Functions for Hybrid Retrieval*
- Cormack, Clarke & Büttcher, 2009 — *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*

**Reranking**
- Nogueira & Cho, 2019 — *Passage Re-ranking with BERT*
- Nogueira et al., 2020 — *Document Ranking with a Pretrained Sequence-to-Sequence Model*
- Sun et al., 2023 — *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents*
- Pradeep, Sharifymoghaddam & Lin, 2023 — *RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!*
- Burges, 2010 — *From RankNet to LambdaRank to LambdaMART: An Overview*

**Query understanding**
- Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE)
- Wang, Yang & Wei, 2023 — *Query2doc: Query Expansion with Large Language Models*

**Evaluation**
- Thakur et al., 2021 — *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*
- Thomas et al., 2023 — *Large language models can accurately predict searcher preferences*
- Upadhyay et al., 2024 — *UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor*
- Lin, Nogueira & Yates, 2021 — *Pretrained Transformers for Text Ranking: BERT and Beyond* (book)

## 12. Essential Tooling

| Tool | Role |
|---|---|
| **OpenSearch / Elasticsearch** | BM25F, analyzers, kNN, sparse vectors, hybrid queries with normalisation processors |
| **Vespa** | Multi-phase ranking, tensors, ColBERT-style ranking in the engine |
| **Pyserini / Anserini** | Reproducible BM25/SPLADE baselines and TREC tooling |
| **SPLADE models, OpenSearch neural-sparse models** | Learned sparse encoders |
| **RAGatouille / PyLate / ColBERT repo** | Late-interaction indexing and reranking |
| **sentence-transformers `CrossEncoder`**, hosted rerank APIs | Reranking |
| **LightGBM (lambdarank)** | Learned fusion and learning to rank |
| **ranx / ir_measures / pytrec_eval** | Metrics, fusion baselines, significance tests |
