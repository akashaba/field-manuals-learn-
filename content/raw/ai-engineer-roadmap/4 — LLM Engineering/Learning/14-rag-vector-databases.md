# RAG — Vector Databases — Master Study Guide

> **Track:** LLM Engineering · **Module:** 14
> **Prerequisites:** Modules 02, 12, 13.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** A vector database is where your embeddings live and are searched. Choose well and retrieval is fast, cheap, and correct at scale; choose poorly and you're rebuilding your search stack at 10× users.

The critical insight: **vector search is approximate**. Exact k-nearest-neighbors on N vectors is O(N) per query — infeasible past a few hundred thousand. Real vector databases use Approximate Nearest Neighbor (**ANN**) algorithms — trading a tiny bit of recall for orders-of-magnitude speed.

**Fundamental principles you must own:**

1. **Vector DB = ANN index + metadata store + filter pipeline** — three components, not just one.
2. **HNSW dominates in-memory ANN** for balance of speed and recall.
3. **IVF+PQ dominates on-disk ANN** for very large collections.
4. **Metadata filtering (payload filters) is essential** for permission-aware / date-scoped retrieval.
5. **Every DB has different tradeoffs** — hosted vs self-hosted, latency vs recall, cost vs features.
6. **Postgres+pgvector is often enough** — don't over-engineer with a specialized DB you don't need.

If you retain nothing else: **pick the smallest vector DB that meets your recall, latency, and filtering requirements. Upgrade only when metrics demand it.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The ANN Problem and Common Algorithms

**Exact k-NN**: for each query, compute distance to every indexed vector, sort, return top-k. Time complexity **O(N · d)** per query. For N=100M, d=768 float32, that's ~300GB scanned per query — impossible.

**ANN**: sacrifice a small amount of recall for massive speedups. Common algorithms:

**HNSW (Hierarchical Navigable Small World)** — Malkov & Yashunin, 2016. Builds a multi-layer graph where each layer is a proximity graph over a subset of vectors, denser at the bottom.

- **Query:** navigate from top layer (few nodes, coarse) to bottom (all nodes, fine), following greedy edges.
- **Recall:** typically 95–99% at reasonable settings.
- **Latency:** O(log N) per query.
- **Memory:** ~1.5× the raw vectors (graph edges are stored).
- **Rebuild:** slow but incremental additions are fast.

**Key tunables:**
- `M` — max edges per node (typical: 16). Higher = better recall, more memory.
- `efConstruction` — build-time search width (typical: 200). Higher = better graph, slower build.
- `efSearch` — query-time search width (typical: 40–200). Higher = better recall, higher latency.

HNSW is the **default in-memory** ANN algorithm — used by Qdrant, Weaviate, Milvus (HNSW mode), FAISS-HNSW, pgvector.

**IVF (Inverted File Index)** — Hervé Jégou et al.

- Partition all vectors into $k$ Voronoi cells using k-means centroids.
- **Query:** find the $n_\text{probe}$ nearest centroids; search only within those cells.
- **Recall:** depends on `nprobe`; higher = more cells searched = better recall.
- **Advantage:** works well on disk (large-scale, don't fit in RAM).

**PQ (Product Quantization)** — compresses vectors from float32 to codes.

- Split each vector into $M$ subvectors.
- Cluster each subvector space into $2^k$ centroids (e.g., 256 for 8-bit codes).
- Represent each vector as $M$ 8-bit codes (M bytes per vector).
- Query with **asymmetric distance** using precomputed lookup tables.

**IVF+PQ**: partition with IVF, compress within cells with PQ. Enables billion-scale search on modest hardware. Used at scale (Facebook, Spotify, etc.).

**Other algorithms:**
- **LSH (Locality-Sensitive Hashing)** — older, simpler, less accurate at same speed. Rarely used.
- **DiskANN (Microsoft)** — disk-resident graph index for billion-scale.
- **ScaNN (Google)** — asymmetric quantization with tree-based routing.
- **Annoy (Spotify)** — random projection forests. Simple but less performant than HNSW.

**Choose HNSW** for < 100M vectors that fit in RAM. Choose **IVF+PQ** or **DiskANN** for larger scale. Choose **flat (exact)** for < 100k vectors — good enough and simpler.

---

### 2.2 Popular Vector Databases (2026 snapshot)

**Self-hosted / open source:**

- **pgvector** (Postgres extension) — vector storage + HNSW/IVFFlat inside Postgres. **Enormous productivity win** if you already use Postgres: transactions, joins with your relational data, familiar tooling. Now often good enough for production.
- **Qdrant** — Rust-native, fast, HNSW-based. Strong filtering; great DX; good for both prototypes and prod.
- **Weaviate** — schema-first, GraphQL-native. Good hybrid search integrations.
- **Milvus** — very large-scale, Kubernetes-native. Complex to operate but scales.
- **FAISS** — Facebook's library (not a full DB). Fast, flexible. Roll your own persistence.
- **Chroma** — Python-native, prototype-friendly. Less production-grade at scale.
- **Vespa** — Yahoo's engine; extremely powerful (dense + sparse + BM25 + tensor operations). Steeper learning curve.
- **Redis** (with `RediSearch`) — vector search as a Redis module.

**Managed / hosted:**

- **Pinecone** — managed only, popular in the early RAG era. Great DX; more expensive at scale.
- **Weaviate Cloud** — managed Weaviate.
- **Qdrant Cloud** — managed Qdrant.
- **MongoDB Atlas Vector Search** — vector search inside Atlas.
- **Elasticsearch / OpenSearch** — mature engines with vector fields.
- **Vertex AI Matching Engine** (Google), **AWS OpenSearch Serverless** — cloud-native offerings.

**Selection heuristic:**

| Situation | Try first |
|-----------|-----------|
| Already using Postgres, < 10M vectors | pgvector |
| Standalone service, dev-friendly, self-hosted | Qdrant |
| Managed, no ops team | Pinecone or Qdrant Cloud |
| Very large scale (100M+) | Milvus / Vespa / DiskANN |
| Mostly BM25 with some vector needs | Elasticsearch/OpenSearch |
| Prototype / notebook | Chroma or FAISS |

---

### 2.3 Metadata Filtering (Payload Filters)

Retrieval isn't only about semantic similarity. Real systems need to filter by:

- Access control ("only docs this user can see").
- Time ranges ("only after Jan 2026").
- Document type or tag ("only technical docs, not marketing").
- Language, region, source.

**Two approaches:**

**Pre-filter** — filter candidates *before* the vector search. Good when the filter is highly selective (< 5% of docs pass); the ANN index may lose recall if too few candidates remain.

**Post-filter** — do the vector search first; filter the top-K results. Good when filters are broad; the ANN can leverage its full index.

**Hybrid (best)** — modern DBs (Qdrant, Milvus 2.4+, Weaviate) implement **filterable ANN**: during graph traversal, only follow edges to nodes passing the filter. Efficient at any selectivity.

**Schema design:**

- **Structured fields** (dates, tags, IDs) → typed columns / payload fields with indexes.
- **Free-text fields** (title, category) → separate FTS index alongside.
- **High-cardinality fields** (user IDs) → hash or bucket first if the DB struggles.

**Multi-tenancy.** For a SaaS app with many tenants:
- **Namespace per tenant** — many DBs support this natively. Fastest and simplest.
- **Metadata field with a filter** — works but doesn't scale well past 10k+ tenants.
- **Separate collections** — for a small number of large tenants.

---

### 2.4 Indexing, Persistence, and Operational Concerns

**Indexing is O(N)** at ingestion time. Building HNSW on 10M vectors takes minutes to hours.

**Incremental updates:**
- **HNSW** supports incremental adds efficiently.
- **IVF** requires periodic re-clustering as new data drifts from the original centroids.
- **Deletions** are typically soft (mark as deleted, filter at query time); periodic compaction cleans up.

**Persistence:**
- On-disk backed indexes (Qdrant, pgvector) survive restarts.
- Pure in-memory indexes (naive FAISS) require re-loading — slow startup at scale.

**Snapshots and backups.** Every serious deployment needs:
- Regular snapshots of the vector index.
- Backups of the source documents (so you can re-embed from source).
- Versioning of the embedding model — a snapshot without the model version is useless.

**Scaling:**
- **Vertical** — bigger machine (more RAM for HNSW).
- **Horizontal** — sharding by ID or metadata. Query hits all shards; results merged.
- **Read replicas** — for read-heavy workloads.

**Cost math (self-hosted, rough):**
- HNSW memory: ~$1.5 \times d \times 4 \text{ bytes}$ per vector (raw vectors + graph edges). For 10M docs × 1024-dim: ~60 GB RAM.
- IVF+PQ: 100–200× more compact (16–32 bytes per vector on disk).

**Hosted cost math:**
- Pinecone-style: ~$0.10–1 per million vectors per month, plus query costs.
- Do the math before committing — self-hosting on a modest VM often beats hosted at scale.

---

### 2.5 Practical Setup (pgvector Example)

Because pgvector is the "just use Postgres" default for most teams, here's a working setup:

```sql
-- 1. Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create table
CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    section_title TEXT,
    source_url TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    metadata JSONB
);

-- 3. Create HNSW index for cosine similarity
CREATE INDEX chunks_embedding_idx ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Indexes on metadata for filtering
CREATE INDEX chunks_document_idx ON chunks (document_id);
CREATE INDEX chunks_created_idx ON chunks (created_at);
CREATE INDEX chunks_metadata_gin ON chunks USING gin (metadata);
```

Query:

```sql
SELECT content, source_url, section_title,
       1 - (embedding <=> $1::vector) AS similarity
FROM chunks
WHERE metadata @> '{"language": "en"}'::jsonb
  AND created_at > NOW() - INTERVAL '30 days'
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

Where `<=>` is the cosine distance operator (lower = more similar).

**Tuning `ef_search`** (query-time HNSW parameter):

```sql
SET hnsw.ef_search = 100;  -- higher = better recall, higher latency
```

**Python client:**

```python
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect(...)
register_vector(conn)
with conn.cursor() as cur:
    cur.execute(
        "SELECT content, 1 - (embedding <=> %s::vector) AS sim "
        "FROM chunks "
        "WHERE document_id = %s "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s",
        (query_vec, doc_id, query_vec, k),
    )
    results = cur.fetchall()
```

Postgres gives you: transactions, joins with users/permissions tables, familiar backups, replication, and JSONB metadata — all in one system.

---

## 3. Mental Models & Analogies

### 3.1 The "Library Neighborhoods" Model

Think of a vector space as a city. Vectors are houses; nearby houses have similar meanings. Exact k-NN is walking to every house in the city to find the closest.

**HNSW** builds a **hierarchical bus network**: express lines between distant neighborhoods (top layer, few stops), local lines within neighborhoods (bottom layer, all stops). To find the nearest house, take express lines to the right neighborhood, then walk locally. Log-time trips.

**IVF** builds **postal zones** by clustering. Every house belongs to one zone (the nearest centroid). To find the nearest house, first find the $n_{\text{probe}}$ closest zones; then search inside them. Linear in the total zone size, but way smaller than the city.

**PQ** compresses each house's address into a **product zip code** — instead of 768 numbers you store 32 codes. Approximate address; enormously smaller.

Real DBs mix these: coarse routing (IVF) + compressed addresses (PQ) + graph shortcuts (HNSW). Which combo you use depends on scale.

### 3.2 The "Filter Then Search" Model

Combined filters + vector search is like finding "the closest coffee shop that's open now and takes cards" — three constraints simultaneously.

- **Pre-filter** = ask staff which shops are open and take cards; then find closest from that list. Good if only 3 shops qualify.
- **Post-filter** = find the 100 closest shops; then keep only those open and taking cards. Good if 60% of shops qualify.
- **Filterable ANN** = walk toward the closest shop, but skip anyone who fails the filter. Best of both.

Modern vector DBs implement the third. Older ones force you to pick pre- or post-.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "We Need a Specialized Vector DB from Day One"

For most teams: no. Postgres + pgvector handles up to ~10M vectors with sub-second queries at reasonable settings — often plenty. Specialized DBs make sense at very large scale, or when the specialized DB's features (filterable ANN, multi-tenancy, managed hosting) genuinely matter. Starting simple avoids operational overhead.

### 4.2 "ANN Is Random — Sometimes It Misses"

ANN is *approximate*, not random. With good algorithms and good settings, recall is 95–99% consistently. If your ANN is returning wildly different results, either recall is set too low (raise `efSearch` in HNSW) or your query vectors are close to many neighbors (an ambiguous query). Measure recall vs exact on a benchmark — don't trust vibes.

### 4.3 "Filtering Reduces Recall / Changes Results"

Depends on the DB. In **filterable ANN** (Qdrant, Milvus, modern pgvector), the filter is applied during traversal — no recall loss. In older systems, pre-filter followed by ANN can hurt recall (if the filter is very selective, the ANN loses its neighbors). Always benchmark: for each of your filter patterns, measure recall vs exact.

---

## 5. Self-Assessment Bank (Vector Databases)

### Questions

**Q1 (Short answer).** In one paragraph, explain why exact k-NN is impractical past a few hundred thousand vectors.

**Q2 (Multiple choice).** HNSW's `M` parameter controls:
- (a) Number of vectors in the index.
- (b) Number of edges per node in the graph — higher = better recall, more memory.
- (c) Query timeout.
- (d) Vector dimension.

**Q3 (Short answer).** Give the tradeoff between IVF+PQ and HNSW for a 100M-vector index.

**Q4 (Multiple choice).** In pgvector, the `<=>` operator computes:
- (a) Euclidean distance.
- (b) Cosine distance (1 - cosine similarity).
- (c) Dot product.
- (d) Manhattan distance.

**Q5 (Short answer).** Why is pgvector often the right first choice for a team that already runs Postgres?

**Q6 (Multiple choice).** Filterable ANN vs pre-filter followed by ANN:
- (a) Same thing.
- (b) Filterable ANN applies the filter during graph traversal, preserving recall; pre-filter first can hurt recall if the filter is highly selective.
- (c) Filterable ANN is always slower.
- (d) Pre-filter is faster.

**Q7 (Short answer).** Why must every vector-index snapshot be paired with its embedding-model version?

**Q8 (Multiple choice).** For a SaaS RAG service with 1000 tenants, best practice for isolation is:
- (a) One giant table with a `tenant_id` metadata filter.
- (b) Native namespace/collection per tenant when the DB supports it.
- (c) Separate database per tenant.
- (d) No isolation.

**Q9 (Short answer).** Explain what Product Quantization does at a high level and its main benefit.

**Q10 (Multiple choice).** Increasing HNSW's `efSearch` from 40 to 200 will:
- (a) Slow queries and increase recall.
- (b) Speed up queries.
- (c) Reduce memory usage.
- (d) Change the index structure.

---

### Answer Key & Detailed Explanations

**A1.** Exact k-NN scans every vector at query time — O(N · d) work. For 10M vectors at 768 dims, that's 30GB read plus 7.5 billion float ops per query. Well over 1 second on a single machine; unworkable at query rates > 1/s. ANN algorithms trade a tiny recall cost (typically 1–5%) for orders-of-magnitude speedups.

**A2. (b).** `M` = max edges per node in the HNSW graph. Higher M → denser graph → better recall but more memory (~M links × 8 bytes per node). Typical 16.

**A3.** **IVF+PQ** compresses vectors 100–200× (16–32 bytes each) and can search on disk, so it scales to billions of vectors with modest RAM. **Recall is lower** than HNSW (typical 85–95%). **Query latency** is competitive on disk.
**HNSW** keeps full-precision vectors and a graph in RAM. Higher recall (95–99%), lower query latency, but ~1.5× the raw vector size in RAM. For 100M vectors × 1024 dims: ~600 GB RAM for HNSW vs ~2–3 GB for IVF+PQ. HNSW wins on quality; IVF+PQ wins on scale/cost.

**A4. (b).** `<=>` in pgvector is **cosine distance** = 1 - cosine similarity. Lower is closer. Other operators: `<->` is L2 distance, `<#>` is negative inner product.

**A5.** Zero new infrastructure — same Postgres server, same backups, same monitoring, same permissions. Joins between vector search and relational data (users, permissions, timestamps) are trivial SQL. Transactions cover both writes to source docs and their embeddings. HNSW inside pgvector is fast enough for up to ~10M–50M vectors. Only reach for a specialized DB when you have specific pgvector limits (extremely high write throughput, > 100M vectors, or specific features).

**A6. (b).** Filterable ANN applies the filter *during* graph traversal — the index skips filtered-out nodes as it navigates. Recall stays high regardless of filter selectivity. Pre-filter followed by ANN over the remaining candidates can miss neighbors if the filter is very selective (the ANN's graph structure is designed assuming the full corpus).

**A7.** Embeddings from different models are in incompatible spaces. Restoring a snapshot with a different embedding model produces garbage — vectors from model A can't be searched with queries embedded by model B. Store the model version alongside the snapshot; on restore, verify it matches.

**A8. (b).** Native namespaces (or "collections" / "tenants") give the DB explicit multi-tenancy support: separate physical partitions, per-tenant limits, faster queries (search only in the tenant's namespace), easy deletion. Filter-based isolation (a) works but is slower and error-prone at scale. Separate DBs (c) works but is operational overhead — only for a small number of very large tenants.

**A9.** PQ splits each vector into $M$ subvectors and quantizes each subvector to one of $2^k$ centroids (typically $k = 8$, i.e., 256 centroids per subvector). Each vector is stored as $M$ 8-bit codes. **Benefit**: 100–200× compression, enabling billion-scale vector search on modest hardware. **Cost**: some quality loss in similarity computation (asymmetric distance uses lookup tables).

**A10. (a).** `efSearch` is the query-time search width — how many candidates the HNSW graph explores. Higher values search more paths → better recall, higher latency. The tradeoff is smooth; tune based on your recall/latency requirements.

---

## 6. Practice Prompts

1. **pgvector setup.** Install pgvector locally (Docker), embed 100k Wikipedia paragraphs, build HNSW index, benchmark query latency and recall vs an exact `ORDER BY embedding <=> query` baseline.
2. **Filter benchmark.** Insert vectors with a `language` tag. Run queries with and without filter; measure recall against exact filter+search. Try `ef_search` = 40 vs 200.
3. **HNSW parameter sweep.** Vary `M` (8, 16, 32) and `efConstruction` (64, 200, 400). Plot build time, memory, recall@10 tradeoffs.
4. **DB comparison.** Deploy the same corpus in pgvector, Qdrant, and Chroma. Compare query latency, features, and DX. Write a decision matrix.
5. **IVF+PQ scale test.** Build an IVF+PQ index with `faiss` on 10M synthetic vectors. Compare RAM usage and recall vs a flat index.

---

## 7. References

- Malkov & Yashunin, ["Efficient and robust approximate nearest neighbor search using HNSW"](https://arxiv.org/abs/1603.09320) (2016).
- Jegou et al., ["Product Quantization for Nearest Neighbor Search"](https://ieeexplore.ieee.org/document/5432202) (2011).
- pgvector docs: [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
- Qdrant docs: [qdrant.tech/documentation](https://qdrant.tech/documentation).
- Milvus docs: [milvus.io/docs](https://milvus.io/docs).
- Facebook FAISS: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss).
- ANN Benchmarks: [ann-benchmarks.com](http://ann-benchmarks.com/).
