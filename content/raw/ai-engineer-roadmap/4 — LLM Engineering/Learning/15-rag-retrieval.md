# RAG — Retrieval — Master Study Guide

> **Track:** LLM Engineering · **Module:** 15
> **Prerequisites:** Modules 12–14.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** With chunks stored and embeddings indexed, **retrieval** is the run-time act of pulling relevant chunks for a specific query. It's the pivot point between "we have data" and "the model gets the right context." Fundamentally simple in the naive case ("embed query, top-k"); rich with practical patterns in production.

**Fundamental principles you must own:**

1. **Not every user question is a good retrieval query.** Transform queries when helpful — expand, decompose, or rewrite.
2. **Top-k alone is rarely optimal.** Combine with metadata filters, hybrid search, and reranking.
3. **Recall matters more than precision** at the retrieval stage — the reranker will filter later.
4. **Retrieval failure modes are diagnosable** — missing chunks, wrong chunks, ambiguous query.
5. **Two-stage retrieval** (fast dense + BM25 → slow reranker) is the modern standard.
6. **Retrieval is a place to inject business logic** — permissions, freshness, source authority.

If you retain nothing else: **retrieval is a pipeline of transforms, not a single call. Design each stage deliberately.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Basic Retrieval Pipeline

Naive top-k retrieval:

```python
def retrieve(query, k=10):
    q_emb = embed(query)
    hits = vector_db.search(q_emb, top_k=k)
    return [hit.content for hit in hits]
```

**Modern pipeline** (a template):

```
Query
  ↓
[Query preprocessing: rewrite, expand, decompose]
  ↓
[Retrieval: dense + BM25 in parallel]
  ↓
[Fusion: RRF or convex combination]
  ↓
[Optional: rerank top-M with a cross-encoder or LLM]
  ↓
Top-k final chunks
  ↓
[Format for LLM: prepend metadata, group by source, etc.]
```

Every stage is optional in principle; every stage adds quality when done right.

**Recommended defaults for a serious RAG system:**
- Retrieve `M = 30–100` from dense + BM25 in parallel.
- Fuse with RRF.
- Rerank top `M` down to `k = 3–10`.
- Format with citations and section context.

---

### 2.2 Query Transformation

User questions are often bad retrieval queries. Transforms improve recall.

**Query rewriting** — turn "what did they say about the project?" into a clearer standalone query using conversation history:

```python
def rewrite_query(user_query, history):
    prompt = f"""Given this conversation, rewrite the last user question as a standalone
    query for retrieval. Include necessary context from history.

    History: {history}
    Last user question: {user_query}

    Standalone query:"""
    return llm(prompt)
```

Critical for chatbots — otherwise "who's the CEO?" retrieves generic content about CEOs, not "CEO of ACME".

**Query expansion** — add synonyms, paraphrases, related terms:

```python
def expand_query(query):
    prompt = f"""Given a search query, provide 3 paraphrases that a search system might find useful.
    Query: {query}
    Paraphrases (one per line):"""
    variants = llm(prompt).splitlines()
    return [query] + variants
```

Then retrieve for each variant; deduplicate results.

**HyDE (Hypothetical Document Embeddings, Gao et al., 2022)** — instead of embedding the question, embed a **hypothetical answer** the LLM invents. The answer's embedding often lands closer to real documents than the question's does:

```python
def hyde(query):
    hypothetical_answer = llm(f"Write a paragraph answering: {query}")
    return embed(hypothetical_answer)   # use this for retrieval
```

Strong on out-of-domain queries where a question and its answer have different vocabulary.

**Query decomposition** — split complex questions into sub-questions, retrieve each, combine:

```python
def decompose(query):
    prompt = f"""Break this question into standalone sub-questions:
    Question: {query}
    Sub-questions (one per line):"""
    return llm(prompt).splitlines()
```

"When was the company founded and who was the first CEO?" → ["When was the company founded?", "Who was the first CEO?"] — retrieve each independently, then generate with combined context.

**Multi-query retrieval** (LangChain pattern) — generate N related queries via LLM, retrieve for each, union the results. Improves recall on ambiguous queries.

**Trade-off:** Every LLM-based query transform adds latency (a small model call) and cost. Use them where quality matters more than a 200ms latency spike.

---

### 2.3 Hybrid Retrieval and Fusion

Covered briefly in Module 13. Re-emphasize:

**Dense + BM25 in parallel, then RRF-fuse:**

```python
def hybrid_retrieve(query, k=30):
    dense_hits = dense_search(embed(query), top_k=k)
    bm25_hits  = bm25_search(query, top_k=k)
    fused = rrf_fusion([dense_hits, bm25_hits], k=60)  # RRF constant
    return fused[:k]

def rrf_fusion(ranked_lists, k=60):
    scores = defaultdict(float)
    for hits in ranked_lists:
        for rank, doc in enumerate(hits):
            scores[doc.id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

**When hybrid dominates dense:** domain-specific vocabulary, proper nouns, code identifiers, product IDs, scientific notation, legal citations. **Almost always: use hybrid over dense-alone.**

**Modern hybrid: SPLADE, ColBERT.**
- **SPLADE** — learned sparse retrieval; represents docs as sparse term-weight vectors. Combines BM25's exact-match power with learned relevance.
- **ColBERT (Late Interaction)** — per-token embeddings; scores by max-similarity across token pairs. Higher recall than single-vector dense; more storage.

For most teams, **dense + BM25 with RRF is enough**. Reach for SPLADE / ColBERT when quality demands it.

---

### 2.4 Filtering and Access Control

**Metadata filters** should apply at retrieval time — never trust post-hoc filtering to be correct or complete.

Common filters:
- **User permissions**: `WHERE access_level <= user.role`.
- **Time**: `WHERE created_at > NOW() - INTERVAL '90 days'`.
- **Tenant**: `WHERE tenant_id = ?`.
- **Source type**: `WHERE source IN ('docs', 'wiki')`.
- **Language**: `WHERE language = user.locale`.

**Permission-aware retrieval — key patterns:**

1. **Pre-filter at query time.** Fastest, safest.
2. **Namespace per tenant.** Physically separate indexes.
3. **Row-level security** (in Postgres/pgvector) — DB enforces the filter regardless of the query.
4. **Post-filter as a defense in depth** — even if pre-filter fails, drop docs the user shouldn't see.

**Never** trust the LLM to respect access control — always enforce at the retrieval or database layer.

**Freshness filtering.** For time-sensitive info, boost recent docs:

```python
# Rerank by combined semantic + freshness score
score = 0.7 * semantic_similarity + 0.3 * exp(-days_old / 90)
```

Simple but effective for news, policy updates, or evolving product docs.

---

### 2.5 Retrieval Failure Modes and Diagnostics

When RAG "hallucinates" or gives wrong answers, retrieval is usually the cause. Common failure modes:

**1. The relevant chunk wasn't retrieved.**
- Query terminology mismatch (user says "auth", docs say "authentication").
- Chunk too small — key context split across chunks.
- Embedding model doesn't cover your domain.
- Filter excluded the right doc.

**Diagnostic:** given a known answer, is the answering chunk in your top-k? (Recall@k at the chunk level.)

**2. The relevant chunk was retrieved but ranked below top-k.**

**Diagnostic:** run with `k = 100`; check position of the right chunk.

**Fix:** reranker (Module 16) or query expansion.

**3. The wrong chunk was retrieved and looks plausible.**
- Embedding-space "false neighbors" — chunks that use similar language but are about different things.
- The chunk is on-topic but doesn't answer the specific question.

**Fix:** more precise chunking, filters, rerankers, or grounded-generation prompts that let the model refuse.

**4. Multiple relevant chunks were retrieved but they contradict.**
- Older + newer versions of the same info.
- Multiple sources with different claims.

**Fix:** deduplicate, freshness boost, or explicitly ask the model to reconcile.

**5. Ambiguous query.**
- "How does it work?" — no clear referent.

**Fix:** query rewriting with conversation history; ask user for clarification.

**Diagnostic playbook.** When RAG fails, always check:

1. What did the retrieval return? (Log the top-k chunks with scores.)
2. Was the right chunk in there? (Search your corpus manually.)
3. What was the query embedded as? (Log query.)
4. Are filters excluding good candidates?
5. Would a rerank / a different top-k / a different embedding fix it?

Instrument retrieval logging in every RAG system. You cannot debug what you can't see.

---

## 3. Mental Models & Analogies

### 3.1 The "Detective Investigation" Model

Retrieval is a detective questioning witnesses:

- **Naive top-k** = interviewing the first 10 people who happen to be in the room. Often useful, sometimes irrelevant.
- **Hybrid retrieval** = mixing witnesses who saw the event (dense: semantic) with those who know the specific people involved (BM25: exact-name matching).
- **Query rewriting** = clarifying the question first — "wait, when I say 'the incident,' I mean the one from March 3rd."
- **Query decomposition** = breaking a complex case into sub-questions ("Who was there?" and "What was the sequence of events?").
- **HyDE** = imagining what a good witness statement would sound like, then finding the real witnesses whose statements match that shape.
- **Filters** = only interviewing employees, not customers; only Q1 2026 witnesses; only people cleared to speak.

A good detective doesn't just ask one question. Neither should your retrieval pipeline.

### 3.2 The "Big Net, Small Net" Model

Retrieval is like fishing:

- **First net (retrieval)** is coarse but wide: cast for many candidates. You'd rather have too many than miss the right one. Optimize for **recall**.
- **Second net (reranker, Module 16)** is fine and small: sift the caught fish for the good ones. Optimize for **precision**.

This two-stage pattern is the modern default. Retrieval fires 100 candidates fast; a reranker (slower, better) narrows to 5. Both nets are needed.

Common mistake: trying to optimize the first net for precision. Result: you miss fish altogether. Better: cast wide, filter later.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The User's Query Is the Query"

Especially in multi-turn chat, the user's raw question is often incomplete: "what about him?" — who's "him"? A raw query embed will return junk. **Always rewrite** queries in a chat context. This one change often lifts retrieval quality by 20%.

### 4.2 "Bigger k Solves Every Retrieval Issue"

Larger $k$ increases recall but also floods the LLM with more (mostly-irrelevant) context. Costs go up; quality often *drops* due to "lost in the middle." Use large $k$ at the *retrieval* stage, then rerank down to a small $k$ for the LLM. Don't just dump 30 chunks into the prompt.

### 4.3 "One Embedding Model, One Query, One Retrieval Call"

Modern systems retrieve multiple ways and fuse. Dense + BM25. Query + expansions. Original query + rewritten query. Multi-query. Each additional retrieval call adds recall (and latency). Trade-off consciously; instrument to see what actually helps.

---

## 5. Self-Assessment Bank (Retrieval)

### Questions

**Q1 (Short answer).** Why is query rewriting important in a multi-turn chatbot?

**Q2 (Multiple choice).** HyDE (Hypothetical Document Embeddings):
- (a) Embeds the query directly.
- (b) Uses an LLM to write a hypothetical answer, then embeds that for retrieval.
- (c) Uses hyperbolic geometry.
- (d) Requires no LLM.

**Q3 (Short answer).** Explain the recall-vs-precision tradeoff in retrieval, and how a two-stage retrieval+rerank pipeline resolves it.

**Q4 (Multiple choice).** For a corpus with lots of proper nouns and product IDs, the retrieval strategy most likely to succeed is:
- (a) Dense-only.
- (b) BM25-only.
- (c) Dense + BM25 hybrid with RRF fusion.
- (d) Query rewriting alone.

**Q5 (Short answer).** Give three diagnostic questions you'd ask when a RAG system returns a wrong answer.

**Q6 (Multiple choice).** For permission-scoped retrieval (users only see docs they have access to), the safest place to enforce the filter is:
- (a) In the LLM prompt ("only use docs the user can see").
- (b) At the retrieval / database layer with pre-filtering.
- (c) After generation as a post-check.
- (d) In documentation.

**Q7 (Short answer).** Describe query decomposition and one use case.

**Q8 (Multiple choice).** Retrieving `k=30` at the retrieval stage but passing only `k=5` to the LLM (after reranking) is:
- (a) Wasteful.
- (b) A standard two-stage pipeline — high recall from retrieval, high precision from reranking.
- (c) An error.
- (d) Only for very large models.

**Q9 (Short answer).** What is "multi-query retrieval" and how does it improve recall?

**Q10 (Multiple choice).** A retrieval failure diagnosis showed the correct chunk was returned at rank 45 (top 100), but the LLM only sees top 5. The best fix is:
- (a) Retrieve more (`k=100`) and send more to the LLM.
- (b) Add a reranker to bring the correct chunk into the top 5.
- (c) Rewrite the chunks.
- (d) Increase the model context.

---

### Answer Key & Detailed Explanations

**A1.** User queries in chat are often context-dependent: "who is he?" or "what about that one?" — these are impossible to retrieve well because the referent is in earlier turns. Query rewriting uses the LLM to produce a standalone, self-contained retrieval query from the last user turn + history. Without rewriting, retrieval degrades sharply in multi-turn conversations.

**A2. (b).** HyDE: (1) ask an LLM to write a hypothetical answer to the query, (2) embed the hypothetical answer, (3) use that embedding for retrieval. Because the hypothetical answer resembles real documents in vocabulary and structure, its embedding often lands closer to relevant docs than the query embedding does.

**A3.** Retrieval should have **high recall** — return every plausibly-relevant chunk, even at the cost of including some irrelevant ones. Reranking then has **high precision** — pick the actually-best ones from the retrieval set. A one-stage system forced to be both is bad at both. Two stages let each specialize: cheap-and-wide retrieval → expensive-and-narrow reranking.

**A4. (c).** Dense embeddings often fail to distinguish specific product IDs, exact names, or technical tokens; BM25 excels there. Combining both catches semantic queries and exact-term queries. RRF fusion is a robust score-agnostic combiner.

**A5.** (1) **Was the right chunk in top-k?** (log retrieval; check recall). (2) **Did filters exclude the right chunk?** (log applied filters; check without filter). (3) **Was the query embedded correctly?** (log query; verify asymmetric prefix if needed). (4) **Would a reranker fix it?** (retrieve top 100; check rank of the right chunk). (5) **Is the chunk too small / lacking context?** (inspect the chunk vs the expected answer). Any three plus rationale.

**A6. (b).** Enforce at retrieval / DB layer — pre-filter or row-level security in the database. The LLM cannot be trusted to respect complex access rules, and any prompt-based rule is bypassable. Defense in depth: also filter at retrieval AND enforce at the DB level.

**A7.** Query decomposition breaks a complex/compound question into simpler sub-questions, retrieves for each independently, and combines results. **Use case**: "Compare Alice's Q1 revenue to Bob's Q2 expenses" — decompose into "Alice Q1 revenue" and "Bob Q2 expenses"; retrieve each; combine at generation.

**A8. (b).** Standard two-stage: retrieve wide (k=30 → high recall), rerank to narrow (k=5 → high precision). The reranker is the "small net" that turns coarse retrieval into precise context for the LLM.

**A9.** Multi-query retrieval uses an LLM to generate several related queries (paraphrases, expansions, reformulations) from the user's question, retrieves for each, and unions the results. Increases recall because different phrasings surface different chunks; effective for ambiguous or under-specified queries. Cost: 3–5× retrieval calls plus one small LLM call.

**A10. (b).** The correct chunk is retrievable but ranked too low. Options: (1) increase `k` to include rank 45 (wasteful — 40 useless chunks for the LLM); (2) add a reranker that reorders top-100 into a better top-5 (efficient — the reranker knows semantic relevance better than the retriever). Option (b) is standard.

---

## 6. Practice Prompts

1. **Query rewriter.** Build a small LLM-based query rewriter that takes chat history + latest user turn and outputs a standalone retrieval query. Measure recall improvement on a chat corpus.
2. **HyDE test.** On an out-of-domain benchmark, compare (a) query embedding, (b) HyDE-based retrieval. Report recall@10 for both.
3. **Multi-query.** Implement multi-query retrieval (generate 3 paraphrases, retrieve top-10 each, union). Measure vs single-query.
4. **Filter-aware retrieval.** Add per-tenant filtering to your pgvector setup. Simulate cross-tenant queries and verify isolation.
5. **Failure log.** Build a retrieval-log dashboard: for every query, store the top-k with scores, query rewrite, and filters. When users report bad answers, use it to diagnose.

---

## 7. References

- Gao et al., ["Precise Zero-Shot Dense Retrieval without Relevance Labels"](https://arxiv.org/abs/2212.10496) (2022) — HyDE.
- Ma et al., ["Query Rewriting for Retrieval-Augmented Large Language Models"](https://arxiv.org/abs/2305.14283) (2023).
- Formal et al., ["SPLADE"](https://arxiv.org/abs/2107.05720) (2021).
- Khattab et al., ["ColBERT"](https://arxiv.org/abs/2004.12832) (2020).
- LangChain multi-query retriever docs and tutorials.
