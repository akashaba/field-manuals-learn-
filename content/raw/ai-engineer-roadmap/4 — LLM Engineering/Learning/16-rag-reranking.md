# RAG — Reranking — Master Study Guide

> **Track:** LLM Engineering · **Module:** 16
> **Prerequisites:** Modules 12–15.
> **Time budget:** ~4–6 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** After retrieval returns 30–100 candidate chunks, you can't send them all to the LLM (context bloat, "lost in the middle," cost). You need to **narrow to the 3–10 best**. Rerankers are the second-stage models that do this well.

Rerankers use richer computations than the first-stage retriever — typically **cross-encoders** that see the query and each candidate *together*, producing a fine-grained relevance score. They're slower but far more precise.

A reranker is one of the highest-impact additions to a RAG system. On many benchmarks, adding a reranker improves top-3 accuracy by 10–30 percentage points over dense-only retrieval.

**Fundamental principles you must own:**

1. **Bi-encoder retrieval + cross-encoder reranking** — the modern two-stage pattern.
2. **Cross-encoders are more accurate but ~100× slower** than bi-encoders per pair — you can only afford to run them on a small candidate set.
3. **LLM-as-reranker** is a strong (expensive) alternative — score candidates via a small LLM.
4. **Reranking is a filter, not a magnifier** — it can't rescue candidates that retrieval missed.
5. **Diversity matters** — the top-k should not be near-duplicates.
6. **Position (order) matters** — put the best chunks at the top and bottom of the LLM prompt, not the middle.

If you retain nothing else: **retrieve wide, rerank narrow. The reranker is what makes RAG feel like it's reading, not fishing.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Bi-Encoders vs Cross-Encoders

**Bi-encoder (used in retrieval, Modules 02, 13):**
- Encode query and each document *independently* into vectors.
- Score = cosine similarity of the two vectors.
- Extremely fast — encode documents once, encode queries at query time.
- Approximate — the vectors compress rich interactions into a single similarity score.

**Cross-encoder (used in reranking):**
- Feed the query AND a document **together** through a single transformer.
- Output a scalar relevance score from the [CLS] token or similar.
- **Much more accurate** — model sees fine-grained token-level interactions.
- Much slower — you can't precompute; every (query, doc) pair requires a forward pass.

Mathematical framing:

$$\text{bi-encoder score}(q, d) = \cos(f(q), f(d))$$

$$\text{cross-encoder score}(q, d) = g([q; d])$$

Bi-encoder factors the joint over $q$ and $d$; cross-encoder doesn't. That's why cross is more accurate and slower.

**Late interaction (ColBERT)** — hybrid: encode each token separately (like bi-encoder for precomputation), but score by max-similarity over token pairs (like cross-encoder for fine-grained). Middle ground: better than bi-encoder, slower and more storage.

---

### 2.2 Popular Reranker Models

**Cross-encoder open-source:**
- `BAAI/bge-reranker-v2-m3` — strong multilingual, long-context.
- `BAAI/bge-reranker-v2-gemma` — larger, higher quality.
- `mixedbread-ai/mxbai-rerank-large-v1` — competitive open reranker.
- `cross-encoder/ms-marco-MiniLM-L-12-v2` — classic, small.
- `Alibaba-NLP/gte-reranker-large` — recent strong option.

**Managed APIs:**
- **Cohere Rerank** (`rerank-english-v3.0`, `rerank-multilingual-v3.0`) — de facto standard for hosted rerankers. Simple API, strong quality.
- **Voyage Rerank** (`rerank-2`) — competitive alternative.
- **Jina Rerank** — open + hosted variants.
- **NVIDIA NeMo Retriever** — enterprise-grade.

**Practical usage (Cohere example):**

```python
import cohere
co = cohere.Client(api_key)

results = co.rerank(
    query=query,
    documents=[chunk.text for chunk in candidates],
    top_n=5,
    model="rerank-english-v3.0",
)
reranked = [(candidates[r.index], r.relevance_score) for r in results.results]
```

**Practical usage (open-source, sentence-transformers CrossEncoder):**

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

pairs = [(query, chunk.text) for chunk in candidates]
scores = reranker.predict(pairs)
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:5]
```

**Latency:** typical reranker takes ~10ms per (query, doc) pair on GPU, ~50–200ms on CPU. For 30 candidates, ~300ms to a few seconds. Acceptable for RAG-scale, prohibitive for real-time search of millions.

---

### 2.3 LLM-as-Reranker

**Idea:** use a small (fast, cheap) LLM to score each candidate for relevance.

**Pointwise LLM reranker** — score each doc individually:

```python
def llm_score(query, doc):
    prompt = f"""On a scale of 0–10, how relevant is this document to the query?

    Query: {query}
    Document: {doc}

    Return only a number, no explanation."""
    return int(llm(prompt, model="gpt-4o-mini").strip())
```

**Pairwise LLM reranker** — compare pairs:

```python
def llm_prefer(query, doc_a, doc_b):
    prompt = f"""Given the query, which document is more relevant? Return "A" or "B".
    Query: {query}
    A: {doc_a}
    B: {doc_b}"""
```

Tournament-style ranking builds up a sort.

**Listwise LLM reranker** — pass all candidates, ask for ranking:

```python
def llm_rank(query, docs):
    numbered = "\n".join(f"[{i}] {d}" for i, d in enumerate(docs))
    prompt = f"""Rank these documents by relevance to the query. Return a comma-separated
    list of indices, most relevant first.

    Query: {query}
    Documents:
    {numbered}"""
    return [int(x) for x in llm(prompt).split(",")]
```

**Trade-offs:**
- **Quality:** LLM rerankers (especially listwise with GPT-4/Claude) rival dedicated cross-encoders.
- **Cost:** Significantly higher per rerank than a fine-tuned cross-encoder.
- **Latency:** slower.
- **Simplicity:** no dedicated model to host.

**Use when:** you already pay for an LLM anyway; volume is low; quality is paramount; you want fewer moving parts.

Modern managed rerankers (Cohere Rerank, Voyage) sit between: dedicated cross-encoders, simple API, low latency, competitive with LLM rerankers.

---

### 2.4 Diversity and Deduplication

Top-k relevance-only ranking can produce highly similar chunks — e.g., 5 near-duplicate snippets of the same paragraph. The LLM sees no new information from chunks 2–5.

**Solution: diversity-aware selection.**

**Maximal Marginal Relevance (MMR)** — Carbonell & Goldstein, 1998:

$$\text{MMR}(d_i) = \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \cdot \max_{d_j \in \text{selected}} \text{Sim}(d_i, d_j)$$

At each step, pick the doc that maximizes MMR. High relevance to query, low similarity to already-selected.

- $\lambda = 1$: pure relevance (same as top-k).
- $\lambda = 0$: pure diversity (ignore relevance).
- $\lambda = 0.5$–$0.7$: balanced, typical.

**Cluster-and-sample.** Cluster the top-M by embedding similarity; sample one representative per cluster.

**Deduplication by exact / near-exact hash.** Strip stop words, hash the remaining; drop hits with the same hash.

**Group-by-source constraint.** Take at most N chunks per document — prevents one long document from monopolizing.

Diversity is especially important when the LLM needs to *summarize* across sources or *compare* — not when it needs to answer from one specific fact.

---

### 2.5 Post-Rerank Ordering and Context Packing

After reranking, you have your top-k chunks with scores. How you **order them in the prompt** matters more than most realize.

**Best practices:**

1. **Highest-relevance at the top or bottom.** "Lost in the middle" says the model pays less attention to central content. Two workable orders:
   - **Descending relevance** (top of prompt = most relevant): natural for readers, good for models.
   - **Ascending relevance** ("Most Recent First" pattern — bottom = most relevant): the model sees the best chunk right before the question. Some studies show this helps.
   - Empirically the difference is small; pick one and be consistent.

2. **Group by source.** If multiple chunks come from the same document, keep them adjacent — the model has an easier time recognizing continuity.

3. **Include metadata inline.** For each chunk, prepend the source and section:
   ```
   [Source: policies.pdf, Section: Reset Procedures, page 12]
   To reset the device, ...
   ```

4. **Numbered references.** For citations, number each chunk `[1]`, `[2]`, ... and ask the model to cite by number. Makes citation extraction trivial.

5. **Delimit clearly.** Use consistent separators (---, ###, XML tags) so the model doesn't merge chunks visually.

**A well-formatted context block:**

```
You are answering questions using ONLY the following sources.

<source id="1" doc="handbook.pdf" section="Onboarding" page="4">
New employees complete orientation in their first week.
</source>

<source id="2" doc="handbook.pdf" section="Onboarding" page="5">
The orientation includes ...
</source>

<source id="3" doc="benefits.pdf" section="Health" page="12">
Health benefits start on the first of the month following ...
</source>

If the sources don't contain the answer, say "I don't know based on the provided documents."
Cite sources by their id.

Question: How soon after starting do employees get health coverage?
```

**Token budget.** Reserve tokens for the LLM's output. If your context is 6000 tokens and you want 1000 output tokens, ensure your model's context is >7000. Trim chunks if needed (see Module 09 compression).

---

## 3. Mental Models & Analogies

### 3.1 The "First-Round vs Final-Round Interview" Model

Retrieval is the **first-round interview**: screen many resumes fast, imperfect but broad. Quality: 60–80% precision.

Reranking is the **final-round interview**: dig deep on the shortlist. Slow but incisive. Quality: 90%+ precision on the shortlist.

You can't final-round every resume — too slow. You can't first-round everyone into the offer stage — too imprecise. Two stages fit their strengths.

**LLM-as-reranker** is like having a senior partner do the final round: expensive but authoritative.

### 3.2 The "Signal + Noise Filter" Model

Retrieval is a **wide antenna** — picks up lots of signal AND noise. Reranker is a **noise filter** — attenuates the irrelevant, amplifies the relevant. The output is the same signal the antenna picked up, just less noisy.

Notice the implication: **the reranker can't create signal that wasn't there.** If retrieval missed the right chunk entirely, reranking can't rescue you. The reranker is a filter, not a magnifier. That's why retrieval recall matters more than retrieval precision — recall bounds what the reranker can do.

If your rerank output feels random or bad, retrieval failed. Debug retrieval first.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Reranker Will Fix Bad Retrieval"

It won't. If the correct chunk isn't in your top-100 (or wherever you cap retrieval), the reranker can't magic it in. Improve retrieval first (better embeddings, hybrid, query rewriting, larger $k$), then rerank the wide set.

### 4.2 "Rerank Everything from the Vector DB"

Cross-encoders are slow. Reranking 10k candidates takes minutes. **Retrieve top ~30–100 first, then rerank those.** The retrieval stage cuts candidates to a manageable size; the rerank stage refines them.

Managed APIs like Cohere have hard caps (100–1000 docs per call) that force this pattern.

### 4.3 "Top-k by Score Is Always Diverse Enough"

If all top-k are near-duplicates of the same paragraph, the LLM sees no new information. On queries where the LLM needs multiple perspectives or facts, deduplicate or apply MMR. Common in FAQ corpora with paraphrased answers.

---

## 5. Self-Assessment Bank (Reranking)

### Questions

**Q1 (Short answer).** State the key difference between a bi-encoder and a cross-encoder.

**Q2 (Multiple choice).** Cross-encoders are typically:
- (a) Faster and less accurate than bi-encoders.
- (b) Slower and more accurate — they see the query and doc together for fine-grained interactions.
- (c) The same as bi-encoders.
- (d) Only for classification.

**Q3 (Short answer).** Why do we retrieve 30–100 candidates but only rerank those, rather than reranking the whole corpus?

**Q4 (Multiple choice).** Cohere Rerank is a managed API for:
- (a) A cross-encoder reranker.
- (b) A bi-encoder embedding model.
- (c) A vector database.
- (d) A tokenizer.

**Q5 (Short answer).** What is MMR (Maximal Marginal Relevance), and why is it useful?

**Q6 (Multiple choice).** After reranking, placing the highest-relevance chunk in the middle of the context is:
- (a) Optimal for attention.
- (b) Sub-optimal — "lost in the middle" effect means models pay less attention there.
- (c) Standard practice.
- (d) Only for GPT-family models.

**Q7 (Short answer).** In one sentence, why does the reranker help even if retrieval already returned "the right chunk somewhere in top-100"?

**Q8 (Multiple choice).** LLM-as-reranker:
- (a) Uses an LLM to score candidates for relevance — high quality, high cost.
- (b) Is always faster than cross-encoders.
- (c) Doesn't work.
- (d) Only supports pointwise scoring.

**Q9 (Short answer).** Why should you cite sources with numbered IDs (e.g., `[Source 1]`) in the prompt?

**Q10 (Multiple choice).** If your reranker's top-5 are all near-duplicate chunks of the same paragraph, the fix is:
- (a) Use a stronger reranker.
- (b) Add diversity — MMR, cluster-and-sample, or dedup by hash.
- (c) Retrieve more candidates.
- (d) Give up.

---

### Answer Key & Detailed Explanations

**A1.** A **bi-encoder** encodes query and document *independently* into vectors, then scores by vector similarity. A **cross-encoder** feeds query and document *together* through a transformer and outputs a scalar relevance score. Bi-encoders are fast (precompute doc vectors); cross-encoders are much more accurate (see joint interactions) but too slow to run on the full corpus.

**A2. (b).** Cross-encoders see the joint interaction between query and doc token-by-token, so they capture subtle relevance signals a bi-encoder loses when compressing each side into an independent vector. Slower because you can't precompute; every (q, d) pair needs a forward pass.

**A3.** Cross-encoders are ~100× slower per pair than bi-encoders. Reranking the whole corpus (millions of docs) per query is infeasible. Two-stage design: cheap bi-encoder retrieval narrows to ~30–100 candidates; expensive cross-encoder rerank refines those to 3–10 for the LLM.

**A4. (a).** Cohere Rerank is a managed cross-encoder reranker (multilingual and English variants). You pass a query and up to ~1000 candidate documents; it returns them ranked by relevance.

**A5.** MMR = Maximal Marginal Relevance. At each step, pick the doc that maximizes $\lambda \cdot \text{Sim}(d, q) - (1 - \lambda) \cdot \max_{d' \in \text{selected}} \text{Sim}(d, d')$. Balances relevance to the query with diversity from already-selected docs. Useful when the LLM needs to see different perspectives or facts, not near-duplicates.

**A6. (b).** "Lost in the middle" (Liu et al., 2023) — LLMs pay less attention to information in the middle of long contexts. Best positions are top or bottom. For top-k results, place the highest-relevance at the top (or bottom, per your prompt template) — never bury the best in the middle.

**A7.** Retrieval's top-100 rankings are imprecise — the truly best chunks may be at ranks 40, 15, 72. Sending "top-100" to the LLM is impractical (cost, context, "lost in the middle"). The reranker reorders the 100 so the *truly best* land in top-5, ready for the LLM.

**A8. (a).** LLM-as-reranker uses an LLM to score candidates — pointwise (per-doc), pairwise (compare two), or listwise (rank all). Often high quality (competitive with dedicated cross-encoders); high cost per rerank. Not always faster (usually slower) — LLM inference costs latency per token.

**A9.** Numbered source IDs let you (1) instruct the model to cite specific sources in its answer (`"Based on [Source 3], ..."`), (2) extract citations post-generation with simple regex, and (3) verify each claim by finding its source. Enables downstream citation validation, source click-through, and trust.

**A10. (b).** Reranking by relevance alone rewards similar duplicates equally. Add a diversity step: **MMR** rebalances relevance and diversity per pick; **cluster-and-sample** picks one representative per cluster; **hash-based dedup** drops obvious near-duplicates first. Choose based on your corpus.

---

## 6. Practice Prompts

1. **Reranker A/B test.** With and without a cross-encoder reranker (`bge-reranker-v2-m3` or Cohere Rerank), measure top-3 accuracy on 100 labeled queries. Report the delta.
2. **LLM-vs-cross-encoder.** For 20 queries, rerank with (a) `bge-reranker-v2-m3`, (b) GPT-4o-mini listwise. Compare quality and cost.
3. **MMR implementation.** Implement MMR from scratch. Show how varying $\lambda$ from 0.3 to 0.9 changes selected diversity.
4. **Position experiment.** For the same top-5 chunks, place the best chunk at position 1, 3, or 5 in the prompt. Compare answer quality.
5. **Citation output.** Prompt the LLM to answer with `[Source N]` citations. Post-process to render citations as links to the original doc.

---

## 7. References

- Nogueira & Cho, ["Passage Re-ranking with BERT"](https://arxiv.org/abs/1901.04085) (2019) — the first paper on cross-encoder reranking.
- Cohere Rerank docs: [docs.cohere.com/reference/rerank](https://docs.cohere.com/reference/rerank).
- Xiao et al., ["FlagEmbedding / BGE-reranker"](https://github.com/FlagOpen/FlagEmbedding).
- Carbonell & Goldstein, ["Maximal Marginal Relevance"](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) (1998).
- Sun et al., ["Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"](https://arxiv.org/abs/2304.09542) (2023) — LLM-as-reranker.
