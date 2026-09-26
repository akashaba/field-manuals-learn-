# 04.02 — Hybrid Search (as the RAG Retrieval Layer)

> **Module 4: RAG** · Subtopic 2 of 5
> **Prerequisites:** **03.04 (required)** — BM25/BM25F, SPLADE, ColBERT, RRF vs convex fusion, cascades, query routing. Also 03.03 (pgvector, ACLs, CDC), 04.01 (chunks and metadata).
> **Outcome:** you can implement hybrid retrieval *inside real engines* (Postgres, OpenSearch, Elasticsearch), apply filters and ACLs correctly across both legs, handle temporal and versioned corpora, size the retrieved context dynamically, cache safely, and **attribute RAG failures** to retrieval or to generation.

> **Scope:** 03.04 covered *how* lexical, sparse, dense, and late-interaction scoring work and how to fuse them. This subtopic covers **operating hybrid search as the retrieval layer of a RAG system**, where the consumer is an LLM with a token budget, not a human scanning ten blue links.

---

## 1. Retrieval Is the Ceiling

Let $R$ be the event "sufficient evidence was retrieved into the context" and $C$ the event "the answer is correct":

$$
P(C) = \underbrace{P(R)}_{\text{retrieval}}\cdot \underbrace{P(C \mid R)}_{\text{generation}} + \big(1 - P(R)\big)\cdot \underbrace{P(C \mid \neg R)}_{\text{parametric / lucky}}
$$

For domain-specific, fresh, or proprietary knowledge (legislation, internal policy), $P(C \mid \neg R) \approx 0$, and the answer rate is capped by $P(R)$. **Diagnose every failure into one of four quadrants** before changing anything:

```
                        answer correct            answer wrong
                  ┌─────────────────────────┬─────────────────────────────┐
 evidence         │  ✔ success              │  GENERATION failure          │
 retrieved        │                         │  (ignored/misread evidence,  │
                  │                         │   distraction, bad prompt)   │
                  ├─────────────────────────┼─────────────────────────────┤
 evidence         │  PARAMETRIC / lucky     │  RETRIEVAL failure           │
 NOT retrieved    │  (risky: unverifiable,  │  (fix chunking, hybrid,      │
                  │   may be stale)         │   rewriting, reranking)      │
                  └─────────────────────────┴─────────────────────────────┘
```

```python
from collections import Counter


def attribute_failures(records: list[dict]) -> dict:
    """records: {'evidence_ids': set, 'retrieved_ids': set, 'correct': bool, 'slice': str}.
    Evidence counts as retrieved when ALL gold evidence ids are present (use 'any' for lenient)."""
    quad = Counter()
    by_slice: dict[str, Counter] = {}
    for r in records:
        hit = r["evidence_ids"] <= r["retrieved_ids"]
        q = ("success" if r["correct"] else "generation_failure") if hit else \
            ("parametric_or_lucky" if r["correct"] else "retrieval_failure")
        quad[q] += 1
        by_slice.setdefault(r.get("slice", "all"), Counter())[q] += 1
    n = sum(quad.values())
    return {"overall": {k: round(v / n, 3) for k, v in quad.items()},
            "by_slice": {s: dict(c) for s, c in by_slice.items()},
            "retrieval_ceiling": round((quad["success"] + quad["generation_failure"]) / n, 3)}
```

**Rule of thumb:** if retrieval failures dominate, work on this subtopic, 04.01, 04.03, and 04.04. If generation failures dominate, work on prompting, context assembly, and verification (02.01, 02.02).

---

## 2. Hybrid Search in Real Engines

### 2.1 Postgres only (pgvector + full-text), fused with RRF in SQL

This is attractive when your metadata, ACLs, and transactions already live in Postgres (03.03). You get one system and one consistency model.

```sql
-- $1 query embedding, $2 query text, $3 tenant, $4 caller groups, $5 as-of date
WITH dense AS (
    SELECT chunk_id, rank() OVER (ORDER BY embedding <=> $1::halfvec) AS rnk
    FROM chunks
    WHERE tenant_id = $3 AND acl_groups && $4::text[]
      AND valid_from <= $5 AND ($5 < valid_to OR valid_to IS NULL)
    ORDER BY embedding <=> $1::halfvec
    LIMIT 100
),
lexical AS (
    SELECT chunk_id, rank() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rnk
    FROM chunks, websearch_to_tsquery('english', $2) AS q
    WHERE tsv @@ q
      AND tenant_id = $3 AND acl_groups && $4::text[]                   -- SAME filters as dense
      AND valid_from <= $5 AND ($5 < valid_to OR valid_to IS NULL)
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT 100
)
SELECT chunk_id,
       COALESCE(1.0 / (60 + d.rnk), 0) + COALESCE(1.0 / (60 + l.rnk), 0) AS rrf_score
FROM dense d FULL OUTER JOIN lexical l USING (chunk_id)
ORDER BY rrf_score DESC
LIMIT 30;
```

**Caveats:**
- `ts_rank_cd` is **not BM25**: it has no proper IDF or term-frequency saturation. For true BM25 inside Postgres, use a BM25 extension (e.g. ParadeDB `pg_search`), or keep a search engine for the lexical leg.
- Add identifier handling: a `citations text[]` column with a GIN index, and an exact-match clause boosted in fusion (03.04 §2.2).
- Set `hnsw.iterative_scan` for the dense leg so that restrictive filters still return enough rows (03.03 §4.2).

### 2.2 OpenSearch: hybrid query + normalisation pipeline

```json
PUT /_search/pipeline/rag-hybrid
{
  "description": "min-max normalise and weight lexical vs dense",
  "phase_results_processors": [
    { "normalization-processor": {
        "normalization": { "technique": "min_max" },
        "combination":   { "technique": "arithmetic_mean", "parameters": { "weights": [0.35, 0.65] } }
    } }
  ]
}

GET /chunks/_search?search_pipeline=rag-hybrid
{
  "size": 30,
  "query": {
    "hybrid": {
      "queries": [
        { "bool": {
            "should": [
              { "multi_match": { "query": "county sheriff funding HB 45",
                                 "fields": ["title^2", "section_path^1.5", "content"] } },
              { "terms": { "citations": ["HB-45"], "boost": 5 } }
            ],
            "filter": [ { "term": { "tenant_id": 42 } }, { "terms": { "acl_groups": ["legal", "staff"] } } ]
        } },
        { "knn": { "embedding": { "vector": [0.012, -0.034, 0.101], "k": 100,
            "filter": { "bool": { "filter": [ { "term": { "tenant_id": 42 } },
                                              { "terms": { "acl_groups": ["legal", "staff"] } } ] } } } } }
      ]
    }
  }
}
```

The `score-ranker-processor` provides RRF instead of score normalisation. **Verify the processor names, parameters, and subquery limits for your OpenSearch version**; the hybrid features have evolved across 2.x and 3.x.

### 2.3 Elasticsearch retrievers (9.x)

```json
GET /chunks/_search
{
  "retriever": {
    "linear": {
      "retrievers": [
        { "retriever": { "standard": { "query": { "bool": {
              "must":   { "match": { "content": "county sheriff funding" } },
              "filter": [ { "term": { "tenant_id": 42 } } ] } } } },
          "weight": 1.0 },
        { "retriever": { "knn": { "field": "embedding", "query_vector": [0.012, -0.034, 0.101],
              "k": 100, "num_candidates": 400, "filter": { "term": { "tenant_id": 42 } } } },
          "weight": 2.0 }
      ],
      "normalizer": "minmax",
      "rank_window_size": 100
    }
  }
}
```

Swap `"linear"` for `"rrf"` (with `rank_constant`, default 60, and `rank_window_size`) when you have no labels to tune the weights. Weighted RRF is supported through a per-retriever `weight`.

### 2.4 Which to use

| Situation | Choice |
|---|---|
| Metadata/ACLs in Postgres, ≤ tens of millions of chunks, a small team | **Postgres** (pgvector + a BM25 extension or full-text) |
| Lexical quality is critical (legal, citations), or you need rich analyzers and aggregations | **OpenSearch/Elasticsearch** hybrid |
| Complex multi-phase ranking in the engine (ColBERT, learned features) | **Vespa** |
| Existing dedicated vector DB with sparse vectors | Its native hybrid API (e.g. Qdrant prefetch + fusion) |

---

## 3. Filters Must Be Identical Across Legs

If the dense leg applies ACL or tenant filters but the lexical leg doesn't (or vice versa):
- **Security:** unauthorised chunks can enter the fused list (03.03 §6).
- **Ranking bias:** fusion mixes a filtered list with an unfiltered one, so rank positions are no longer comparable.

**Engineering pattern:** build filters once as a typed object, compile it into each engine's syntax, and **contract-test** that every leg receives the same predicate. Also watch for **k-shortfall** under restrictive filters. If the dense leg returns 7 results and the lexical leg 100, fusion is dominated by the lexical leg. Log per-leg result counts.

---

## 4. Multi-Source Retrieval

RAG over several heterogeneous sources (statutes, bills, FAQs, past memos, tickets) needs **source-aware assembly**:

- **Per-source retrieval + quotas:** retrieve from each source, then allocate context budget by a per-source share and a minimum (e.g. statutes ≥ 50%, FAQs ≤ 15%).
- **Authority ordering:** put primary sources (enacted law) before secondary ones (memos) in the context, and say so in the prompt.
- **Source-aware fusion:** normalise scores per source before merging, because sources have different score distributions.

```python
def allocate_by_source(ranked: dict[str, list[tuple[str, float, int]]], budget: int,
                       shares: dict[str, float], minimum: dict[str, int] | None = None):
    """ranked: source -> [(chunk_id, score, tokens)] best first. Returns chosen chunk ids.
    Pass 1 fills each source's share (at least its minimum); pass 2 fills leftovers by global score."""
    minimum = minimum or {}
    chosen, used, leftovers = [], 0, []
    for src, items in ranked.items():
        cap = max(int(budget * shares.get(src, 0.0)), minimum.get(src, 0))
        spent = 0
        for cid, score, toks in items:
            if spent + toks <= cap and used + toks <= budget:
                chosen.append(cid)
                spent += toks
                used += toks
            else:
                leftovers.append((score, cid, toks))
    for score, cid, toks in sorted(leftovers, reverse=True):
        if used + toks <= budget:
            chosen.append(cid)
            used += toks
    return chosen, used
```

---

## 5. Temporal and Versioned Corpora

Legislation, policies, and product documentation change. "What is the penalty?" means *as of when*?

### 5.1 Validity intervals ("as-of" retrieval)

Store `valid_from` / `valid_to` on every chunk version (e.g. effective date and repeal/amendment date). Default to **today**, and let query rewriting extract explicit dates ("in 2019", "before the 2023 session"; 04.04). Filter on the interval (§2.1 SQL) so that superseded text never competes with current text.

### 5.2 Recency as a soft signal

For news-like sources, where newer is usually better but old items remain valid, blend in an exponential decay with half-life $h$:

$$
s'(d) = s(d)\cdot\Big(1 - w + w\cdot e^{-\lambda\,\Delta t(d)}\Big), \qquad \lambda = \frac{\ln 2}{h}
$$

Here $w \in [0,1]$ is the maximum recency influence. **Never** use a recency decay on authoritative law (use validity intervals instead). Old statutes that are still in force are not less relevant.

```python
import math
from datetime import date


def recency_adjust(results: list[tuple[str, float, date]], today: date, half_life_days: float,
                   weight: float = 0.3) -> list[tuple[str, float]]:
    lam = math.log(2) / half_life_days
    out = [(cid, s * (1 - weight + weight * math.exp(-lam * (today - d).days))) for cid, s, d in results]
    return sorted(out, key=lambda t: -t[1])
```

---

## 6. How Much to Retrieve: Dynamic k and Abstention

A fixed top-k wastes tokens on easy queries and starves hard ones. Better options:

- **Token budget** instead of k (04.01 §6.1).
- **Score-gap cut-off:** after reranking (04.03), stop at the largest relative drop in score.
- **Calibrated threshold:** keep chunks with calibrated relevance probability ≥ τ (04.03 §5).
- **Abstention:** if nothing passes the threshold, say "not found in the sources" rather than letting the model improvise. Measure abstention precision (was the evidence really absent?).

```python
def dynamic_cut(scores: list[float], min_k: int = 2, max_k: int = 12, min_score: float | None = None,
                rel_drop: float = 0.35) -> int:
    """Return how many (sorted, descending) results to keep."""
    k = min(len(scores), max_k)
    if min_score is not None:
        k = min(k, sum(1 for s in scores if s >= min_score))
        if k == 0:
            return 0                                          # abstain
    for i in range(max(min_k, 1), k):
        prev, cur = scores[i - 1], scores[i]
        if prev > 0 and (prev - cur) / prev >= rel_drop:
            return i
    return k
```

---

## 7. Caching Without Wrong Answers

| Cache | Key | Risk |
|---|---|---|
| Query embedding | normalised query text + `model_id` | Low |
| Retrieval results | normalised query + **filters + ACL groups + as-of date** + index version | Leaking results across users if ACLs are missing from the key |
| **Semantic cache** (reuse an answer for a *similar* query) | Embedding similarity above a threshold | **High**: "penalty under HB 45" ≈ "penalty under HB 46" in embedding space |

**Guard semantic caches with an identifier check.** Require that the numbers, citations, dates, and named entities extracted from the two queries match exactly. Also scope every cache by tenant and ACL, and invalidate on index-version bumps.

```python
import hashlib
import json
import re

IDENT = re.compile(r"\b(?:HB|SB|HJ|SJ)[\s-]?\d+\b|\b\d+(?:-\d+)+\b|\b\d{4}\b|\$?\d[\d,.]*", re.I)


def identifiers(q: str) -> tuple[str, ...]:
    return tuple(sorted(m.group(0).upper().replace(" ", "-") for m in IDENT.finditer(q)))


def retrieval_cache_key(query: str, filters: dict, acl_groups: list[str], index_version: str) -> str:
    norm = " ".join(query.lower().split())
    payload = json.dumps({"q": norm, "f": filters, "acl": sorted(acl_groups), "v": index_version},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def semantic_cache_hit(new_q: str, cached_q: str, similarity: float, threshold: float = 0.95) -> bool:
    return similarity >= threshold and identifiers(new_q) == identifiers(cached_q)
```

---

## 8. Measuring the Retrieval Layer for RAG

| Metric | Definition | Use |
|---|---|---|
| **Evidence recall @ budget** | Fraction of gold evidence (spans or IDs) inside the retrieved context within $B$ tokens | Primary retrieval metric for RAG |
| **Context precision** | Fraction of retrieved tokens or chunks that are relevant | Distraction and cost |
| **Retrieval ceiling** | Share of questions whose evidence was retrieved (§1) | Upper bound on accuracy |
| **Per-leg contribution** | Share of gold evidence found only by the lexical leg, only by the dense leg, or by both | Justifies (or removes) a leg |
| **k-shortfall rate** | Queries where a leg returned fewer results than requested | Filter/ACL health |
| **Latency per leg** | P50/P95 | Budget and timeouts |

Frameworks such as RAGAS and ARES provide LLM-judged context precision and recall when gold evidence isn't labelled. Validate the judge against human labels (03.04 §8.1).

---

## 9. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Accuracy plateau** | Prompt tweaks stop helping | Failure attribution (§1); if retrieval-bound, improve recall |
| **Inconsistent filters across legs** | ACL leaks; odd rankings | A single filter object compiled per engine; contract tests; per-leg count logging |
| **Superseded text retrieved** | Answers cite repealed law | Validity intervals + as-of filtering; version metadata in the context |
| **Recency decay on authoritative sources** | Old-but-valid law ranked low | Decay only on news-like sources; intervals for law |
| **Semantic cache serves the wrong bill** | Correct-looking answer for the wrong identifier | Identifier-equality guard; ACL-scoped keys; version invalidation |
| **Hybrid slower than either leg** | Latency budget exceeded | Parallel legs with timeouts (03.04 §6.3); smaller candidate windows; approximate filters |
| **One source dominates the context** | Answers ignore primary law | Source quotas and authority ordering (§4) |
| **Too much or too little context** | Wasted tokens, or missing evidence | Budget-based retrieval, dynamic cut, calibrated thresholds, abstention |

---

## 10. Hands-On Projects

### Project 1 — Postgres-Only Hybrid RAG vs OpenSearch Hybrid

**User stories**
- *As a tech lead* with Postgres already in production, I want to know whether a Postgres-only hybrid retrieval layer is good enough, or whether we need a search engine, so that we don't add infrastructure without evidence.

**Acceptance criteria**
1. Both systems index the same chunks (04.01) and embeddings. Postgres uses pgvector + full-text (and optionally a BM25 extension). OpenSearch uses BM25F + kNN + a normalisation pipeline.
2. The retrieval API (Spring Boot or FastAPI) applies identical tenant, ACL, and as-of filters to both legs in both systems, verified by contract tests.
3. On ≥ 300 labelled questions (sliced by identifier, keyword, and natural-language queries): evidence recall at 2k/4k tokens, per-leg contribution, P50/P95 latency, and the attribution quadrants of the full RAG answers.
4. The operational comparison covers setup effort, resource footprint at 10M chunks (extrapolated), backup and restore, and consistency with the source data.
5. A recommendation, with the specific query slices where each system wins.

**Step-by-step**
1. Load the chunk tables and index the same data into OpenSearch through the 03.03 CDC pipeline (a second sink).
2. Implement the SQL of §2.1 and the pipeline and query of §2.2 behind a common `Retriever` interface.
3. Implement the typed filter object and its compilers to SQL and OpenSearch DSL, with contract tests.
4. Run the retrieval evaluation, then end-to-end RAG with a fixed generator prompt.
5. Compute `attribute_failures` per slice, and write the decision memo.

---

### Project 2 — Temporal ("As-Of") Legislation RAG

**User stories**
- *As a legislative attorney*, I want to ask "what did 45-5-207 say in 2019?" or "what changed in this section after the 2023 session?", and get answers from the correct version with citations.

**Acceptance criteria**
1. Ingests ≥ 3 historical versions of a code (or another versioned corpus) with `valid_from`/`valid_to` per section version, derived from session laws or effective dates.
2. Query rewriting extracts explicit dates or sessions into an `as_of` filter (validated against the available range). The default is today.
3. Supports diff questions by retrieving two versions and presenting both, labelled with their dates.
4. On ≥ 150 time-sensitive questions: version accuracy (the cited version is correct) ≥ 95%, and zero answers that cite superseded text as current.
5. Contrasts the system with a baseline that has no temporal filtering, and quantifies the stale-citation rate.

**Step-by-step**
1. Build the version table and compute validity intervals. Handle the "effective upon passage" and "applies retroactively" edge cases explicitly.
2. Extend the retrieval SQL or DSL with interval filters (§2.1).
3. Implement date extraction with structured output (02.03) and validation.
4. Build the diff flow: retrieve the section at t₁ and t₂ and render them side by side in the context.
5. Evaluate against the baseline, and write up the failure cases.

---

### Project 3 — RAG Failure-Attribution and Retrieval Health Dashboard

**User stories**
- *As a RAG owner*, I want a live dashboard showing whether failures come from retrieval or generation, per query slice, so that the team works on the right problem every sprint.

**Acceptance criteria**
1. An offline evaluation job runs nightly on a labelled set of ≥ 300 questions. It computes evidence recall @ budget, context precision, the attribution quadrants (`attribute_failures`), per-leg contribution, and k-shortfall rate.
2. Online sampling: 1% of production queries are judged by a validated LLM judge for context relevance and answer groundedness, with PII redaction.
3. A dashboard (Grafana or Streamlit) with trends per slice and per release, and annotations for index, model, and prompt versions.
4. Alerts: the retrieval ceiling drops by > 3 points week over week, or k-shortfall exceeds 5%.
5. One documented case where the dashboard redirected engineering effort (e.g. from prompt tuning to chunking).

**Step-by-step**
1. Define the evaluation record schema (question, gold evidence, retrieved IDs, answer, judge outputs, versions).
2. Build the offline runner, reusing the §1 and §8 metrics, and store the results in Postgres or ClickHouse.
3. Build the online sampler with trace IDs (OpenTelemetry), a redaction step, and the judge.
4. Build the dashboards and alerts.
5. Run for a few releases, and write the retrospective.

---

## 11. Foundational Papers & Reading (exact titles)

- Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Izacard & Grave, 2021 — *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*
- Ram et al., 2023 — *In-Context Retrieval-Augmented Language Models*
- Shi et al., 2023 — *REPLUG: Retrieval-Augmented Black-Box Language Models*
- Barnett et al., 2024 — *Seven Failure Points When Engineering a Retrieval Augmented Generation System*
- Es et al., 2023 — *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
- Saad-Falcon et al., 2023 — *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*
- Bruch, Gai & Ingber, 2023 — *An Analysis of Fusion Functions for Hybrid Retrieval*
- Kasai et al., 2023 — *RealTime QA: What's the Answer Right Now?* (temporal questions)
- Bang, 2023 — *GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings*
- Engine docs: pgvector hybrid search; OpenSearch hybrid query, normalization and score-ranker processors; Elasticsearch `rrf` and `linear` retrievers

## 12. Essential Tooling

| Tool | Role |
|---|---|
| **PostgreSQL + pgvector (+ ParadeDB `pg_search` or full-text)** | Single-database hybrid retrieval |
| **OpenSearch** (hybrid query, search pipelines) / **Elasticsearch** (retrievers) | Search-engine hybrid |
| **Qdrant / Weaviate / Vespa** hybrid APIs | Alternatives with native fusion |
| **RAGAS, ARES, DeepEval, TruLens** | RAG evaluation (validate judges) |
| **ranx** | Offline fusion experiments and significance |
| **OpenTelemetry + Grafana / Langfuse / Phoenix** | Tracing and dashboards |
| **Redis / Postgres** | Retrieval and embedding caches with safe keys |
