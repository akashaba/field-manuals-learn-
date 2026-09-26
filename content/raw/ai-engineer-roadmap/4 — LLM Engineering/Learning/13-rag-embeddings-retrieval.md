# RAG — Embeddings for Retrieval — Master Study Guide

> **Track:** LLM Engineering · **Module:** 13
> **Prerequisites:** Modules 02 (embeddings basics) and 12 (chunking).
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Module 02 covered *what* embeddings are. This module is the **RAG-specific** view: which embedding model to pick for retrieval, how to embed queries vs documents, how to combine dense embeddings with sparse retrieval, and how to keep an embedding index fresh over time.

Getting embeddings wrong is like using a bad compass. Even with perfect chunking and a fast index, if your embedding model doesn't semantically place similar-meaning texts near each other for **your domain**, retrieval fails at the vector-space level.

**Fundamental principles you must own:**

1. **Not all embeddings are equal.** Choice matters more for retrieval than for downstream classification.
2. **Query and document embeddings often need different treatment** — asymmetric embedding.
3. **Dimensionality is a cost/quality tradeoff** — bigger = better usually, but 8×+ storage and 2×+ compute per lookup.
4. **Cosine similarity is the workhorse metric.** Nearly always the right choice for text embeddings.
5. **Hybrid retrieval (dense + BM25) is often better** than dense alone, especially for domain-specific vocabulary.
6. **Embedding models drift** — your index will slowly go stale as content, terminology, or embedding models evolve.

If you retain nothing else: **pick a strong embedding model for your domain, benchmark it on your data, and combine with sparse retrieval by default.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Choosing an Embedding Model

**Popular options (as of 2026 — always check MTEB leaderboard):**

**Proprietary APIs:**
- `text-embedding-3-small` / `text-embedding-3-large` (OpenAI) — solid general-purpose; supports **dimension truncation** (Matryoshka).
- Voyage AI embeddings (`voyage-3`, `voyage-code-3`, `voyage-law-2`) — often best-in-class; domain-tuned models available.
- Cohere `embed-v3` — strong for multilingual and retrieval.

**Open-source (via `sentence-transformers` or the Hub):**
- `BAAI/bge-large-en-v1.5` — long-time strong performer.
- `BAAI/bge-m3` — multilingual + long context, versatile.
- `intfloat/e5-large-v2` / `e5-mistral-7b-instruct` — top MTEB scorers, larger model.
- `mixedbread-ai/mxbai-embed-large-v1` — competitive open model.
- `Alibaba-NLP/gte-large-en-v1.5` — strong retrieval-focused.
- `all-MiniLM-L6-v2` — tiny (384-d, 90MB), decent quality — great for baseline / edge / cost-sensitive.

**Selection criteria:**

- **MTEB leaderboard** — [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is the community benchmark. Filter by "Retrieval" task.
- **Task fit** — MTEB has task categories: retrieval, classification, clustering, STS. For RAG, look at retrieval subscores.
- **Domain fit** — legal, medical, code, multilingual — specialized models often beat general.
- **Dimensionality** — 256–1536 typical. Larger = better, more storage and compute.
- **Context length** — 512, 8192, 32k+. Long-context embedding models handle bigger chunks.
- **License** — some models restrict commercial use.
- **Cost** — API models charge per token; self-hosted has GPU overhead.

**Benchmark on your data.** Never trust MTEB averages alone. Build a small evaluation set from your domain (see Module 18) and measure recall@k, MRR, NDCG on your actual queries. The right model for you may not be the top MTEB model.

---

### 2.2 Query vs Document Embedding (Asymmetric)

Some embedding models (E5-family, BGE-family, Voyage instruct-family) treat query and document embeddings **differently** — they were trained on asymmetric retrieval and expect a "prompt" or "instruction" prefix.

**E5 example:**

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/e5-large-v2")

# Documents
doc_embeddings = model.encode(
    ["passage: The Pythagorean theorem is a² + b² = c².",
     "passage: The mitochondrion is the powerhouse of the cell."],
    normalize_embeddings=True,
)

# Queries
query_embedding = model.encode(
    ["query: What is the Pythagorean theorem?"],
    normalize_embeddings=True,
)
```

The `"query:"` and `"passage:"` prefixes are **part of the model's contract**. Skipping them measurably degrades retrieval.

**BGE instruction prompt (for BGE reranker/embeddings):**

```python
query = "Represent this sentence for searching relevant passages: What is X?"
```

**OpenAI, Voyage, Cohere** typically use different endpoints (`input_type="query"` vs `"document"`). Always read the model card / API docs.

**Practical rule:** **read the model card** before deploying. Half the "why is my retrieval bad" incidents are missing prefixes.

---

### 2.3 Cosine Similarity, Normalization, and Distance Metrics

**Cosine similarity** is the workhorse for text embeddings:

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

Bounded in $[-1, 1]$; higher = more similar; invariant to vector magnitude.

**Normalize once, then use dot product.** If you unit-normalize embeddings ($\|\mathbf{u}\|_2 = 1$), cosine similarity reduces to a simple dot product. This is much faster in a vector DB and mathematically identical:

```python
def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

# After normalization, cosine sim == dot product
```

Most modern embedding models produce roughly unit-length vectors already or offer a `normalize=True` option.

**Euclidean (L2) distance** — for normalized vectors, monotonically related to cosine similarity: $\|\mathbf{u} - \mathbf{v}\|^2 = 2 - 2 \mathbf{u} \cdot \mathbf{v}$. So L2 and cosine rank identically for normalized embeddings.

**Dot product (unnormalized)** — used by some models (e.g., certain SPLADE-family or task-specific models). Preserves magnitude information; useful when magnitude encodes importance.

**Manhattan (L1)** — rarely used for text embeddings.

**Practical guidance:**
- Text embeddings → **cosine (or dot on normalized)**.
- Almost never L1 or unnormalized L2.
- If unsure, normalize and use dot product — best of all worlds.

---

### 2.4 Dense + Sparse Hybrid Retrieval

**Dense retrieval** — embedding-based semantic search. Great for paraphrase, semantic, cross-lingual queries.

**Sparse retrieval** — BM25, TF-IDF. Great for exact-term matches (product IDs, proper nouns, technical jargon).

**Hybrid retrieval** combines both — often outperforms either alone.

**BM25 quick primer.** BM25 (Best Match 25) is the classic IR scoring function:

$$\text{BM25}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 (1 - b + b \cdot |D| / \bar{L})}$$

Where:
- **$f(q, D)$** = frequency of term $q$ in document $D$.
- **$|D|$** = document length.
- **$\bar L$** = average document length.
- **$k_1$**, **$b$** = tunable parameters (typically 1.5 and 0.75).
- **$\text{IDF}(q)$** = inverse document frequency.

Key property: BM25 scores exact term matches highly and penalizes long documents (normalization).

**Reciprocal Rank Fusion (RRF)** — a simple, robust fusion:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$

Where $k$ is a constant (typically 60). Each retriever ranks documents; RRF combines ranks — no need to align raw scores.

**Alternative fusion methods:**
- **Convex combination** — $\alpha \cdot s_{\text{dense}} + (1 - \alpha) \cdot s_{\text{sparse}}$; must normalize scores.
- **Learned fusion** — train a linear or neural combiner on labeled query-doc relevance data.

**Implementations:**
- **Elasticsearch / OpenSearch** — BM25 + kNN vector search built in.
- **pgvector + PostgreSQL FTS** — for pgvector deployments.
- **Weaviate, Qdrant, Milvus, Vespa** — many support hybrid natively.
- **LangChain / LlamaIndex** provide `EnsembleRetriever` for do-it-yourself hybrid.

**When to use hybrid:**
- Corpora with lots of proper nouns, product names, code identifiers.
- Domain vocabulary that embedding models weren't trained on.
- User queries that are keyword-like (short, entity-heavy).

Empirically, hybrid retrieval improves recall@k by 5–20% on many benchmarks over dense alone.

---

### 2.5 Index Freshness and Model Migration

**The staleness problem:**
- Docs are added or edited over time.
- Chunk contents change.
- Old embeddings become inconsistent with new docs.
- Embedding model versions get upgraded.

**Freshness patterns:**

**Incremental updates.** Embed and index new/changed documents; delete old versions.

```python
def upsert_document(doc_id, chunks):
    delete_by_document_id(doc_id)
    embeddings = model.encode([c.text for c in chunks], normalize_embeddings=True)
    index.upsert([(f"{doc_id}::{i}", emb, chunk.metadata)
                  for i, (emb, chunk) in enumerate(zip(embeddings, chunks))])
```

**Change detection.** Store a hash of each document's content; skip re-embedding unchanged docs.

**Backfill.** When adding metadata, updating chunk sizes, or backfilling old docs — a scheduled batch job. Use batch embedding APIs for cost.

**Model migration.** When you want to switch embedding models:

1. **Never** mix embeddings from different models in the same index — they live in incompatible spaces.
2. Build a new index with the new model.
3. Migrate traffic gradually (or A/B test).
4. Once validated, deprecate the old index.

**Cost-effective embedding:**
- **Batch API** (OpenAI, Voyage) — 50% off for async batches.
- **Local embedding models** — for high-volume, use a GPU or a beefy CPU with a small model like `bge-small-en-v1.5`.
- **Cache embeddings** — never re-embed the same text; hash-key your cache.

**Multi-vector for future-proofing.** Store the raw chunk text alongside the embedding. If you switch embedding models later, you can re-embed from stored text without re-parsing source docs.

**Matryoshka Representation Learning (MRL).** Some models (OpenAI text-embedding-3, Nomic) are trained so their embeddings can be **truncated** without losing much quality:

```python
# Full-dim
emb = openai.embeddings.create(model="text-embedding-3-large", input=text)
# Truncated to 256 for storage / speed
emb_short = normalize(emb[:256])
```

Let you trade quality for storage/compute per query. Very handy in production.

---

## 3. Mental Models & Analogies

### 3.1 The "Zip-Code + Full-Address" Model (Hybrid Retrieval)

Imagine finding someone in a country by asking:

- **Dense retrieval** = knowing what someone is *like* — their profession, interests, personality. Broad, semantic. If someone asks "who's a musician in Helena", you find musicians by their musical vibe, not their name.
- **Sparse retrieval (BM25)** = knowing exact identifiers — a full name, a phone number, a house address. Precise for terms; useless for meaning.

**Hybrid** = use both. If you know both the profession and the name, you'll find the person faster than with either alone. Some queries want exact matches; some want semantic proximity. Combine.

Real-world: a customer typing "SKU 4587-B" needs BM25 to match that exact SKU. A customer typing "the blue one with the rounded corners" needs dense retrieval. Hybrid handles both.

### 3.2 The "Two Languages" Model (Asymmetric Embedding)

Think of query and document embeddings as two dialects of the same language. Some embedding models are trained to speak both — but only if you tell them which dialect a piece of text belongs to (via the `passage:` or `query:` prefix).

Skip the dialect marker and the model tries to guess — often wrongly, blending both into a mush. That's why "why is my retrieval bad" so often traces back to a missing prefix.

Bilingual analogue: if you ask an English-Chinese interpreter to translate without saying which direction ("English to Chinese, please"), they may guess wrong. Give them the direction and they nail it.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Best MTEB Model Is the Best for Me"

MTEB averages over many tasks — some of which are irrelevant to your use case. A model that dominates classification but is only mediocre on retrieval might rank high overall. **Filter to retrieval, then benchmark on YOUR data.** The gap between "best MTEB" and "best for you" is often 10–30% recall@k.

### 4.2 "Cosine Similarity of 0.85 Is Great"

Cosine values are not calibrated across models. One model's 0.85 is another model's 0.65 for the same semantic similarity. Look at the *distribution* of similarities in your corpus — a value that seems high may be typical for that model's embedding space.

Use **relative** cosine (rank by similarity) and validate with recall@k on labeled data, not absolute thresholds.

### 4.3 "One Embedding Per Chunk Is Enough"

For many use cases, no. Consider **multi-vector indexing**:
- Embed the chunk's text.
- Embed a summary of the chunk.
- Embed hypothetical questions the chunk answers.
- Index all vectors pointing to the same chunk.

This dramatically improves retrieval for questions that don't lexically or semantically match the chunk's phrasing but *conceptually* match its purpose. Trade-off: more storage.

---

## 5. Self-Assessment Bank (Embeddings for Retrieval)

### Questions

**Q1 (Short answer).** Why is it critical to benchmark embedding models on **your** data, not rely on the MTEB average?

**Q2 (Multiple choice).** For E5 embeddings, when embedding a document, you should:
- (a) Just pass the raw text.
- (b) Prefix with `"passage: "`.
- (c) Prefix with `"query: "`.
- (d) Prefix with the document ID.

**Q3 (Short answer).** Explain why unit-normalizing embeddings lets you replace cosine similarity with a dot product.

**Q4 (Multiple choice).** Hybrid dense + sparse retrieval is typically better than dense alone when:
- (a) All queries are natural language paraphrases.
- (b) The corpus has many exact-term matches (product IDs, technical terms, proper nouns).
- (c) The corpus is small.
- (d) You use a large embedding model.

**Q5 (Short answer).** Describe Reciprocal Rank Fusion (RRF) in one paragraph.

**Q6 (Multiple choice).** Matryoshka embeddings enable:
- (a) Faster training.
- (b) Truncating a full-dim embedding to a lower dim without significant quality loss.
- (c) Larger vocabulary.
- (d) Multi-modal fusion.

**Q7 (Short answer).** Why must you never mix embeddings from two different models in the same vector index?

**Q8 (Multiple choice).** For a query "SKU 4587-B is defective" against a product catalog:
- (a) Dense retrieval will find it perfectly.
- (b) BM25 or hybrid retrieval likely outperforms dense-only because "SKU 4587-B" is an exact term.
- (c) Neither will work.
- (d) Semantic chunking is required.

**Q9 (Short answer).** What is "multi-vector indexing" and when does it help?

**Q10 (Multiple choice).** If your embedding model changes (upgrade), the correct migration approach is:
- (a) Just start using the new model; the vectors are compatible.
- (b) Build a new index with the new model; migrate traffic; deprecate the old one.
- (c) Average old and new vectors.
- (d) Restart your database.

---

### Answer Key & Detailed Explanations

**A1.** MTEB averages over many tasks (retrieval, classification, clustering, STS). A model that wins on average may be mediocre on retrieval, or bad on your specific domain (legal, medical, code). Always build a small labeled eval set from your actual queries and measure recall@k / MRR / NDCG. The best-for-you may be 10-30 points off the MTEB leaderboard.

**A2. (b).** E5 was trained with `passage:` prefix for documents and `query:` prefix for queries. Skipping the prefixes silently degrades retrieval by 5–20% because you're using the model out of its trained regime.

**A3.** Cosine similarity: $\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$. If $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$ (both unit-normalized), the denominator is 1, and cosine reduces to $\mathbf{u} \cdot \mathbf{v}$ — a plain dot product. Vector DBs run dot products faster than cosine, so pre-normalizing lets you use the faster path.

**A4. (b).** BM25 handles exact-term matches (product IDs, proper nouns, code identifiers, jargon) far better than dense embeddings. Hybrid combines exact-term precision (BM25) with semantic understanding (dense) — best of both.

**A5.** RRF is a rank-based fusion: for each retriever, rank the documents; for each doc, sum $\frac{1}{k + \text{rank}_r}$ across retrievers (typical $k = 60$). Final ranking = sum. Advantages: no need to normalize raw scores across retrievers; robust to score scale differences; simple and often close to optimal.

**A6. (b).** Matryoshka models are trained so that the first $d$ dimensions of a full embedding form a good approximation on their own. So you can store `emb[:256]` from a 3072-dim embedding and lose modest quality — trading storage for a small quality drop. OpenAI's `text-embedding-3` supports this via the `dimensions` parameter.

**A7.** Different models produce embeddings in different vector spaces. Cosine similarity across models is meaningless — they're not comparable. Mixing them gives garbage retrieval. Always keep one model per index; migrate by building a new index.

**A8. (b).** "SKU 4587-B" is a specific token/entity that dense embedding models often can't distinguish from other similar-looking IDs. BM25's exact-term matching dominates for this kind of query. Hybrid retrieval combining both is safest.

**A9.** Multi-vector indexing stores multiple embeddings per chunk — the chunk text, a generated summary, hypothetical questions the chunk answers — each pointing to the same chunk record. Retrieval matches against all of them. Helps when user queries don't lexically or semantically match the chunk's exact wording, but do match its conceptual purpose. Cost: more storage.

**A10. (b).** New embeddings live in a different vector space; old-vs-new vectors are incompatible. Build a fresh index with the new model, migrate traffic (A/B tested), and deprecate the old. Don't attempt to average or blend embeddings from different models.

---

## 6. Practice Prompts

1. **Model benchmarking.** Build an eval set of 50 query-document pairs from your domain. Benchmark 3 embedding models (e.g., `all-MiniLM-L6-v2`, `bge-large-en-v1.5`, `text-embedding-3-small`). Report recall@10.
2. **Hybrid retrieval.** Set up dense (BGE) + BM25 retrieval on a Wikipedia subset. Fuse with RRF. Compare recall@10 to dense-only.
3. **Asymmetric embedding test.** Use E5 with and without prefixes; compare recall@10 on the same eval set. Confirm the gap.
4. **Matryoshka.** Embed the same corpus with `text-embedding-3-large` at full 3072 dims and truncated to 256/512/1024. Compare retrieval quality vs storage.
5. **Multi-vector indexing.** For each chunk, additionally embed 3 auto-generated questions. Index all vectors. Compare retrieval quality to single-vector.

---

## 7. References

- MTEB Leaderboard: [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
- Wang et al., ["Text Embeddings by Weakly-Supervised Contrastive Pre-training"](https://arxiv.org/abs/2212.03533) (2022) — E5.
- Xiao et al., ["C-Pack: Packaged Resources To Advance General Chinese Embedding"](https://arxiv.org/abs/2309.07597) (2023) — BGE.
- Robertson & Zaragoza, ["The Probabilistic Relevance Framework: BM25 and Beyond"](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (2009).
- Cormack et al., ["Reciprocal Rank Fusion Outperforms Condorcet"](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (2009).
- Kusupati et al., ["Matryoshka Representation Learning"](https://arxiv.org/abs/2205.13147) (2022).
