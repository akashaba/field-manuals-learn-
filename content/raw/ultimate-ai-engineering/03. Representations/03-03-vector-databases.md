# 03.03 — Vector Databases

> **Module 3: Representations** · Subtopic 3 of 4
> **Prerequisites:** 03.01 (embeddings, versioning), 03.02 (ANN indexes, filtered search), database fundamentals (WAL, MVCC, indexes, replication), Kafka or another event log, Kubernetes basics.
> **Outcome:** you can design the data layer of a retrieval system end to end: the schema, ingestion via change data capture, consistency and deletion guarantees, access control, multi-tenancy, sharding and capacity, and operations. You can also choose between Postgres + pgvector, a dedicated vector database, or a search engine with measured evidence.

---

## 1. What a Vector Database Adds to an ANN Library

FAISS or hnswlib give you an **index**. A production system also needs:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ API: upsert / delete / query (vector + filters + hybrid) / scroll / snapshot │
├──────────────┬──────────────────┬─────────────────┬───────────────────────────┤
│ Durability   │ Metadata &       │ Query engine    │ Distribution              │
│ WAL, fsync,  │ filtering        │ planner, fan-out│ sharding, replication,    │
│ snapshots    │ payload indexes  │ merge top-k,    │ rebalancing, consistency  │
│              │ ACL fields       │ rescoring       │ levels                    │
├──────────────┴──────────────────┴─────────────────┴───────────────────────────┤
│ Index lifecycle: build (async), incremental insert, deletes/tombstones,       │
│ compaction, rebuild, tiering (RAM ↔ SSD ↔ object storage)                     │
├───────────────────────────────────────────────────────────────────────────────┤
│ Ops & security: authN/Z, multi-tenancy, quotas, encryption, backup/restore,   │
│ metrics, audit                                                                 │
└───────────────────────────────────────────────────────────────────────────────┘
```

**The core design tension:** ANN indexes (especially graphs) are expensive to build and poor at in-place updates, while databases must accept continuous writes and deletes with bounded freshness. Every vector database is an answer to that tension.

---

## 2. Internal Architecture: The Segment Model

Most dedicated vector databases (Milvus, Qdrant, Weaviate, Lucene-based engines) use an LSM-like **segment** design:

```
 writes ──► WAL (durable) ──► GROWING segment (in memory, small; brute-force or light index)
                                    │ size/time threshold
                                    ▼
                               SEALED segment ──► background INDEX BUILD (HNSW / IVF-PQ / DiskANN)
                                    │
                         deletes = tombstone bitmap per segment (filtered out at query time)
                                    │
                         COMPACTION merges small segments, drops tombstoned rows, rebuilds index
                                    │
                         (optional) tier sealed segments to SSD / object storage

 query ──► planner ──► fan-out to all segments on all shards ──► per-segment top-k ──► merge ──► rescore
```

**Consequences you will observe in production:**

- **Freshness lag:** a newly written vector is searchable via the growing segment immediately (or after a short flush), but is *indexed* later. Heavy write bursts produce many unindexed segments, and brute-force costs rise.
- **Delete cost:** deletes are cheap to issue but make queries slower (more tombstones to filter) until compaction runs. High churn requires scheduled compaction.
- **Query latency ∝ number of segments.** Tune segment sizes. Too many small segments are slow, and too few large ones make rebuilds slow.
- **Consistency levels:** some systems let you choose per query — strong (read your writes), bounded staleness, session, or eventual. Strong consistency costs latency, because the query must wait for the WAL to be applied.

**Postgres + pgvector** takes a different path: vectors are ordinary columns under MVCC, and HNSW/IVFFlat are index access methods. You get transactions, joins, row-level security, and point-in-time recovery for free. Writes update the index synchronously (slower inserts at scale), and deletes are cleaned up by `VACUUM`.

---

## 3. Choosing a System

| Option | Architecture | Strengths | Watch out for |
|---|---|---|---|
| **Postgres + pgvector** | Extension; HNSW & IVFFlat; `vector`, `halfvec`, `sparsevec`, `bit` types; iterative index scans | Transactions, joins, RLS, existing ops/backup tooling; one system for metadata + vectors | Single-node index memory; HNSW build time at 10M+ rows; filtered queries need iterative scans; index in shared buffers |
| **Qdrant** | Rust; segments; HNSW with payload-aware filtering; quantization; sparse vectors | Strong filtered search, multi-tenancy features, compact ops | Distributed-mode operations; RAM sizing with payload indexes |
| **Milvus** | Cloud-native, disaggregated (query/data/index nodes, object storage) | Very large scale; many index types incl. GPU and DiskANN | Operational complexity (several components) |
| **Weaviate** | Go; HNSW + BM25 hybrid; modules | Built-in hybrid search, multi-tenancy | Memory footprint at scale |
| **Elasticsearch / OpenSearch** | Lucene HNSW (+ quantization) inside a search engine | Best-in-class BM25, aggregations, mature ops; hybrid in one query | Lucene segment merges; memory; vector features vary by version and license |
| **Vespa** | Search engine with tensors, HNSW, rich ranking phases | Complex ranking (multi-phase, ColBERT-style), real-time updates | Learning curve |
| **LanceDB / Lance format** | Columnar files on object storage; embedded or server | Cheap, versioned datasets; great for batch/ML workflows | Online high-QPS serving characteristics |
| **Managed / serverless** (Pinecone, Turbopuffer, Azure AI Search, MongoDB Atlas, cloud Postgres) | Vendor-operated; object-storage-backed or clustered | Zero ops, elastic | Cost at scale, data residency, feature and lock-in constraints |

> Feature sets change quickly. The table reflects the general architecture. **Verify current features, limits, and licenses** for the versions you would deploy.

**Decision heuristics:**
1. **≤ ~10–50M vectors, relational metadata, strong consistency and ACLs matter, and a team that already runs Postgres** → start with **pgvector**. It is the lowest operational risk.
2. **Lexical search quality is central** (legal, legislative, and code search) → **OpenSearch/Elasticsearch or Vespa**, or Postgres + a search engine side by side (03.04).
3. **100M+ vectors, high filtered QPS, many tenants** → a dedicated vector database (Qdrant, Milvus) or a managed service. Prove it with a bake-off (Project 2).

---

## 4. Data Modelling

### 4.1 Schema (pgvector)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    doc_id        text PRIMARY KEY,
    tenant_id     int  NOT NULL,
    source_uri    text NOT NULL,
    doc_version   int  NOT NULL,
    status        text NOT NULL,              -- e.g. introduced | enacted | repealed
    acl_groups    text[] NOT NULL,            -- who may read
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    chunk_id        text PRIMARY KEY,         -- deterministic: doc_id || ':' || doc_version || ':' || ordinal
    doc_id          text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tenant_id       int  NOT NULL,
    doc_version     int  NOT NULL,
    ordinal         int  NOT NULL,
    section_path    text,                     -- "Title 2 > Ch 18 > 2-18-303"
    content         text NOT NULL,
    content_hash    bytea NOT NULL,           -- sha256(normalised content + preprocess_version)
    model_id        text NOT NULL,            -- embedding model + version (03.01 §6.1)
    acl_groups      text[] NOT NULL,          -- denormalised for single-table filtering
    embedding       halfvec(1024) NOT NULL,   -- fp16 halves memory; index limit 4,000 dims
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX chunks_embedding_hnsw ON chunks
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX chunks_tenant ON chunks (tenant_id);
CREATE INDEX chunks_acl    ON chunks USING gin (acl_groups);
CREATE INDEX chunks_tsv    ON chunks USING gin (tsv);            -- lexical leg of hybrid (03.04)
```

**Modelling rules:**
- **Deterministic IDs** (derived from the document ID, version, and ordinal) make upserts idempotent and deletes exact.
- **Denormalise the filter fields** (tenant, ACL, status, dates) onto the chunk row. Joins inside an ANN query defeat the index.
- **Store `model_id`** and never mix models in one index (03.01 §6.1). During migrations, use a separate column or table per model.
- **Keep the source text or a pointer** so you can re-embed and rerank without reprocessing the original documents.

### 4.2 Filtered queries with iterative scans

```sql
-- Per session / transaction
SET hnsw.ef_search = 100;
SET hnsw.iterative_scan = relaxed_order;   -- keep scanning until enough rows pass the filters
SET hnsw.max_scan_tuples = 20000;          -- bound the work (latency guard)

WITH candidates AS MATERIALIZED (
    SELECT chunk_id, doc_id, section_path, content,
           embedding <=> $1::halfvec AS distance
    FROM chunks
    WHERE tenant_id = $2
      AND acl_groups && $3::text[]            -- caller's groups
    ORDER BY embedding <=> $1::halfvec
    LIMIT 50
)
SELECT * FROM candidates ORDER BY distance LIMIT 10;   -- relaxed_order may be slightly out of order
```

### 4.3 Binary quantization + rescoring inside Postgres

```sql
CREATE INDEX chunks_embedding_bq ON chunks
    USING hnsw ((binary_quantize(embedding)::bit(1024)) bit_hamming_ops);

SELECT chunk_id, content FROM (
    SELECT chunk_id, content, embedding
    FROM chunks
    ORDER BY binary_quantize(embedding)::bit(1024) <~> binary_quantize($1::halfvec)
    LIMIT 200                                        -- oversample by Hamming distance
) c
ORDER BY embedding <=> $1::halfvec                   -- exact rescoring
LIMIT 10;
```

### 4.4 Java / Spring access

```java
@Repository
public class ChunkSearchRepository {
    private final JdbcTemplate jdbc;
    public ChunkSearchRepository(JdbcTemplate jdbc) { this.jdbc = jdbc; }

    public record ChunkHit(String chunkId, String docId, String sectionPath, String content, double distance) {}

    @Transactional(readOnly = true)                      // SET LOCAL applies to this transaction only
    public List<ChunkHit> search(float[] queryVec, int tenantId, String[] groups, int k) {
        jdbc.execute("SET LOCAL hnsw.ef_search = 100");
        jdbc.execute("SET LOCAL hnsw.iterative_scan = relaxed_order");
        var vec = new com.pgvector.PGhalfvec(queryVec);  // pgvector-java; register types on the connection
        return jdbc.query(con -> {
            var ps = con.prepareStatement("""
                SELECT chunk_id, doc_id, section_path, content, embedding <=> ? AS distance
                FROM chunks
                WHERE tenant_id = ? AND acl_groups && ?
                ORDER BY embedding <=> ?
                LIMIT ?""");
            ps.setObject(1, vec);
            ps.setInt(2, tenantId);
            ps.setArray(3, con.createArrayOf("text", groups));
            ps.setObject(4, vec);
            ps.setInt(5, k);
            return ps;
        }, (rs, i) -> new ChunkHit(rs.getString(1), rs.getString(2), rs.getString(3),
                                   rs.getString(4), rs.getDouble(5)));
    }
}
```

---

## 5. Ingestion: Keeping the Index Consistent with the Source of Truth

### 5.1 Change data capture pipeline

```
 source DB (documents) ──► Debezium CDC ──► Kafka topic docs.changes (key = doc_id → per-doc ordering)
                                                     │
                                 ┌───────────────────▼────────────────────┐
                                 │ ingest worker (consumer group)         │
                                 │ 1. fetch doc @ version                 │
                                 │ 2. normalise + chunk + contextual hdrs │
                                 │ 3. hash chunks; skip unchanged hashes  │
                                 │ 4. embed changed chunks (batched)      │
                                 │ 5. single transaction:                 │
                                 │    upsert new/changed chunks           │
                                 │    delete chunks of old versions       │
                                 │ 6. commit Kafka offset AFTER db commit │
                                 └───────────────────┬────────────────────┘
                                                     ▼
                                            vector store (+ DLQ for poison messages)
```

**Guarantees and how to achieve them:**

| Requirement | Mechanism |
|---|---|
| No lost updates | At-least-once consumption; commit offsets after the DB transaction |
| No duplicates | Idempotent upserts keyed by deterministic `chunk_id`; `content_hash` skips no-op re-embeds |
| Per-document ordering | Kafka key = `doc_id`; ignore events older than the stored `doc_version` |
| Deletes propagate | Tombstone events → delete all chunks of the document; periodic reconciliation job (source IDs vs index IDs) |
| Right-to-erasure (GDPR etc.) | Hard delete + `VACUUM` (Postgres) or compaction (vector databases) + purge from backups per policy; verify with an audit query |
| Embedding cost control | Re-embed only changed chunks (hash diff); batch by length; rate-limit the backfill |

```python
import hashlib
from dataclasses import dataclass
from typing import Callable, Protocol


class VectorStore(Protocol):
    def current_version(self, doc_id: str) -> int | None: ...
    def existing_hashes(self, doc_id: str) -> dict[str, bytes]: ...
    def apply(self, doc_id: str, version: int, upserts: list[dict], delete_ids: list[str]) -> None: ...


@dataclass
class DocEvent:
    doc_id: str
    version: int
    deleted: bool
    text: str = ""
    meta: dict | None = None


def chunk_hash(text: str, preprocess_version: str) -> bytes:
    return hashlib.sha256((preprocess_version + "\x00" + " ".join(text.split())).encode()).digest()


def handle_event(ev: DocEvent, store: VectorStore, chunker: Callable[[str], list[str]],
                 embed: Callable[[list[str]], list[list[float]]], model_id: str,
                 preprocess_version: str = "pp-3") -> dict:
    cur = store.current_version(ev.doc_id)
    if cur is not None and ev.version < cur:
        return {"status": "stale_event_ignored"}
    old = store.existing_hashes(ev.doc_id)                      # chunk_id -> hash
    if ev.deleted:
        store.apply(ev.doc_id, ev.version, [], list(old))
        return {"status": "deleted", "removed": len(old)}

    chunks = chunker(ev.text)
    ids = [f"{ev.doc_id}:{ev.version}:{i}" for i in range(len(chunks))]
    hashes = [chunk_hash(c, preprocess_version) for c in chunks]
    reusable = set(old.values())
    to_embed = [i for i, h in enumerate(hashes) if h not in reusable]
    vecs = dict(zip(to_embed, embed([chunks[i] for i in to_embed]))) if to_embed else {}

    upserts = [{"chunk_id": ids[i], "ordinal": i, "content": chunks[i], "content_hash": hashes[i],
                "model_id": model_id, "embedding": vecs.get(i), "reuse_embedding_from_hash": i not in vecs,
                **(ev.meta or {})} for i in range(len(chunks))]
    stale = [cid for cid in old if cid not in set(ids)]
    store.apply(ev.doc_id, ev.version, upserts, stale)          # ONE transaction
    return {"status": "upserted", "chunks": len(chunks), "embedded": len(to_embed), "deleted": len(stale)}
```

The `reuse_embedding_from_hash` flag lets the store copy an existing vector for an unchanged chunk whose ID changed with the version. This saves embedding cost on small edits to large documents.

---

## 6. Security and Access Control

**Rule: enforce authorisation inside the retrieval query, never after generation.** If unauthorised chunks reach the LLM's context, they can leak into the answer regardless of any post-filtering.

- **Document-level ACLs** as filter fields (`acl_groups && :user_groups`), computed from the identity provider's group claims (e.g. Azure AD / Entra ID groups in the token).
- **Postgres row-level security** makes this a database guarantee rather than an application convention:

```sql
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_and_acl ON chunks
    USING (tenant_id = current_setting('app.tenant_id')::int
           AND acl_groups && string_to_array(current_setting('app.groups'), ','));
-- per request, inside the transaction:
-- SELECT set_config('app.tenant_id', '42', true), set_config('app.groups', 'legal,staff', true);
```

  RLS predicates are applied *after* the index returns candidates, so combine RLS with **iterative scans** (§4.2). Otherwise restrictive policies return fewer than k rows.
- **Tenant isolation tests:** automated tests that attempt cross-tenant reads through every API path.
- **Embeddings are sensitive data** (03.01 §6.3): encryption at rest, access logging, and deletion propagation.
- **Prompt injection through stored content** (02.01 §6): the vector store is a supply chain for your context. Record provenance on every chunk.

---

## 7. Multi-Tenancy Patterns

| Pattern | Isolation | Efficiency | Best for |
|---|---|---|---|
| **Shared index + tenant filter** | Logical (filter) | Best packing | Many small tenants; needs good filtered search |
| **Partition per tenant** (table partitions, partition keys, tenant shards) | Stronger; per-tenant index | Good | Medium tenants with skewed sizes |
| **Collection/index per tenant** | Strong | Overhead per collection | Few large tenants; per-tenant configs |
| **Database/cluster per tenant** | Physical | Worst | Regulated tenants, data residency |

Mix patterns by tier: small tenants share an index, and large or regulated tenants get dedicated partitions or clusters. Enforce **quotas** (vectors, QPS, storage) and monitor **noisy neighbours** (per-tenant P99 latency).

---

## 8. Capacity Planning and Scaling

Per-replica memory for an HNSW deployment:

$$
\text{RAM} \approx N\,\big(d \cdot b_{\text{vec}} + M_0 \cdot 4 \cdot \phi + p\big)\cdot(1 + h)
$$

where $b_{\text{vec}}$ is the bytes per dimension, $M_0 = 2M$ is the number of layer-0 links, $\phi \approx 1.1$ accounts for upper layers and allocator overhead, $p$ is the payload and metadata bytes held in RAM, and $h$ is headroom (0.3–0.5).

$$
\text{shards} = \left\lceil \frac{\text{RAM}}{\text{usable RAM per node}} \right\rceil,\qquad
\text{replicas} = \max\!\left(2,\ \left\lceil \frac{\text{QPS}_{\text{peak}}}{\text{QPS per replica at SLO}} \right\rceil\right)
$$

`QPS per replica at SLO` must be **measured** with realistic filters and concurrent writes. Every query fans out to all shards, so per-query latency is bounded by the *slowest* shard. Tail latency grows with the shard count.

```python
import math


def plan_capacity(n_vectors: int, dim: int, bytes_per_dim: float, M: int = 16,
                  payload_bytes: int = 200, headroom: float = 0.4, node_ram_gb: float = 64,
                  usable_frac: float = 0.75, peak_qps: float = 200, qps_per_replica: float = 150):
    per_vec = dim * bytes_per_dim + 2 * M * 4 * 1.1 + payload_bytes
    total_gb = n_vectors * per_vec * (1 + headroom) / 1e9
    shards = math.ceil(total_gb / (node_ram_gb * usable_frac))
    replicas = max(2, math.ceil(peak_qps / qps_per_replica))
    return {"bytes_per_vector": round(per_vec), "total_ram_gb": round(total_gb, 1),
            "shards": shards, "replicas": replicas, "nodes": shards * replicas}
```

**Scaling levers, in order of cost:** compress vectors (fp16 → int8 → binary + rescoring; 03.01 §3) → tune `M` and `ef_search` → partition by tenant or time so that queries touch fewer shards → add replicas for QPS → add shards for size → move to disk-based indexes (03.02 §5.2).

---

## 9. Operations

- **Index builds:** in pgvector, raise `maintenance_work_mem` so the graph fits during the build (otherwise it is much slower) and use parallel maintenance workers. Build new indexes concurrently. For large reloads, bulk-load first and build the index after.
- **Rebuild strategy:** blue/green indexes (build a new index or collection → validate → switch an alias).
- **Backups:** Postgres PITR (WAL archiving) or vector-database snapshots to object storage. **Restore drills** on a schedule, measuring RTO (restore + index warm-up time).
- **Monitoring:**
  - Query latency P50/P99 by tenant and filter type; result-count < k rate.
  - Index freshness lag (event time → searchable).
  - Ingest throughput and DLQ size; tombstone ratio and compaction backlog.
  - Memory and disk; cache hit rates.
  - **Recall canary:** a fixed query sample compared against exact search on a snapshot.

```python
import numpy as np


def recall_canary(query_vecs: np.ndarray, ann_search, exact_search, k: int = 10,
                  threshold: float = 0.9) -> dict:
    """Run on a schedule; alert if recall vs exact drops below threshold."""
    rec = [len(set(ann_search(q, k)) & set(exact_search(q, k))) / k for q in query_vecs]
    r = float(np.mean(rec))
    return {"recall_at_k": round(r, 4), "p10": round(float(np.percentile(rec, 10)), 4),
            "alert": r < threshold}
```

---

## 10. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Stale or ghost results** | Deleted or edited content still retrieved | Tombstone events, versioned chunk IDs, reconciliation job, compaction/VACUUM monitoring |
| **ACL leaks** | Users see content they shouldn't | Filter inside the query; RLS; isolation tests; never post-filter after generation |
| **Filtered queries return too few rows** | Empty answers for restrictive users or tenants | Iterative scans; selectivity-aware planning (03.02 §6); partitions for big tenants |
| **Write amplification on re-embed** | Embedding bills spike on minor edits | Hash-diff chunks; reuse vectors; batch; rate-limit backfills |
| **Slow index builds** | Deploys blocked for hours | More `maintenance_work_mem`/parallel workers; blue/green builds; GPU builds in dedicated engines |
| **Tail latency with many shards** | P99 grows as you scale out | Fewer, larger shards; tenant-aware routing; replicas; hedged requests |
| **Noisy tenants** | Other tenants' latency degrades | Quotas, per-tenant partitions, rate limits |
| **Unverified backups** | Restore fails when needed | Scheduled restore drills with RTO measurement |

---

## 11. Hands-On Projects

### Project 1 — Production pgvector Retrieval Store with CDC, ACLs, and RLS

**User stories**
- *As a backend engineer* on a legislative platform, I want document changes in our system of record reflected in vector search within seconds, with no duplicates or ghosts, so that attorneys always search current text.
- *As a security officer*, I want retrieval to respect document ACLs at the database layer, so that no one can retrieve content outside their groups.

**Acceptance criteria**
1. Postgres 16+ with pgvector (≥ 0.8), the §4.1 schema, an HNSW index on `halfvec`, and RLS (§6).
2. The CDC pipeline follows §5.1: Debezium → Kafka (key = `doc_id`) → Spring Boot or Python worker with `handle_event` semantics, a DLQ, and offset commit after the transaction.
3. **Freshness SLO:** P95 event-to-searchable time < 10 s under 50 documents/s of updates.
4. **Correctness tests:**
   - Replaying the whole topic twice yields identical row counts and hashes.
   - Deleting a source document removes all its chunks within the SLO.
   - A reconciliation job reports zero drift.
5. **Security tests:** ≥ 20 cross-tenant and ACL attack cases (API parameter tampering, direct SQL via the app role) all fail. Filtered queries for low-privilege users return k results via iterative scans.
6. Everything is deployed on Kubernetes (Helm or Kustomize), with dashboards for latency, freshness lag, and DLQ size.

**Step-by-step**
1. Stand up Postgres, Kafka, and Debezium with Docker Compose for development, and Kubernetes manifests for staging.
2. Create the schema and indexes; tune `maintenance_work_mem`; load an initial corpus in bulk before building the index.
3. Implement the worker (Spring Kafka or confluent-kafka) with idempotent upserts and version checks.
4. Implement the RLS policies and a request filter that sets `app.tenant_id` and `app.groups` from the JWT claims.
5. Write the correctness, chaos (kill worker mid-batch), and security test suites.
6. Add Micrometer or Prometheus metrics and Grafana dashboards, and write a runbook.

---

### Project 2 — Vector Database Bake-Off on Your Workload

**User stories**
- *As a platform architect*, I need evidence-based selection between pgvector, a dedicated vector database, and a search engine for our 20M-chunk corpus, so that we don't migrate twice.

**Acceptance criteria**
1. Candidates: pgvector, Qdrant, OpenSearch (or Elasticsearch), and Milvus (or a managed service), all on equivalent hardware on Kubernetes or VMs.
2. The workload replays **real** query and write distributions: vector-only, filtered (3 selectivity buckets), and hybrid (where supported), plus concurrent upserts and deletes at 5% of QPS.
3. Measures:
   - Recall@10 vs exact search.
   - QPS at P99 < 50 ms.
   - Ingest throughput and index build time.
   - Freshness lag.
   - Memory and disk footprint.
   - Restore time from backup.
   - Operational effort (hours to set up, number of components).
4. A weighted decision matrix with cost projections at 20M and 200M chunks.
5. All configs and scripts are reproducible (Terraform/Helm + benchmark code in the repo).

**Step-by-step**
1. Freeze the dataset and embeddings, and compute exact ground truth.
2. Deploy each engine with vendor-recommended settings, then tune `M`/`ef` to a common recall target (e.g. 0.95).
3. Build a workload driver (Python asyncio or k6) with Poisson arrivals and mixed operations.
4. Run each scenario for ≥ 15 minutes after warm-up, 3 repetitions.
5. Run failure drills: kill a node, restore from a snapshot.
6. Write the decision document.

---

### Project 3 — Multi-Tenant Vector Platform with Quotas and a Recall Canary

**User stories**
- *As a platform team*, we want to offer vector search as an internal service to many product teams, with isolation, quotas, and SLOs, so that each team doesn't run its own database.

**Acceptance criteria**
1. A service API (REST/gRPC) supporting tenant provisioning, upsert, delete, query (vector + filters), and snapshot/restore. Tier-based placement: shared index for small tenants, dedicated partitions for large ones (§7).
2. Enforces per-tenant quotas (vector count, QPS, storage) with clear errors. Rate limiting uses a token bucket.
3. A recall canary per tenant tier (`recall_canary`) plus freshness probes, exported as metrics, with alerts.
4. A noisy-neighbour test: one tenant at 10× load doesn't push other tenants' P99 beyond their SLO.
5. A disaster-recovery drill: restore a tenant from snapshot to a point in time, and measure RTO and RPO.

**Step-by-step**
1. Choose a backend (pgvector with partitioning, or Qdrant/Milvus) and design the tenant → placement mapping.
2. Implement the control plane (tenants, quotas, placement) and the data plane (query routing).
3. Implement the canary job with sampled queries per tenant, exact search on a snapshot, and metric export.
4. Build the load and noisy-neighbour tests and tune the isolation (rate limits, partitioning).
5. Write the SLO document and the runbooks.

---

## 12. Foundational Papers & Reading (exact titles)

- Wang et al., 2021 — *Milvus: A Purpose-Built Vector Data Management System*
- Guo et al., 2022 — *Manu: A Cloud Native Vector Database Management System*
- Pan, Wang & Li, 2024 — *Survey of Vector Database Management Systems*
- Zhang et al., 2023 — *VBase: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity*
- Wei et al., 2020 — *AnalyticDB-V: A Hybrid Analytical Engine Towards Query Fusion for Structured and Unstructured Data*
- Patel et al., 2024 — *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data*
- O'Neil et al., 1996 — *The Log-Structured Merge-Tree (LSM-Tree)*
- Kleppmann, 2017 — *Designing Data-Intensive Applications* (book: replication, partitioning, CDC, consistency)
- pgvector README and CHANGELOG (HNSW/IVFFlat parameters, iterative scans, types); Debezium documentation (outbox pattern, ordering)

## 13. Essential Tooling

| Tool | Role |
|---|---|
| **PostgreSQL + pgvector** (+ `pgvector-java`, `pgvector-python`) | Transactional vector store with RLS |
| **Qdrant / Milvus / Weaviate / OpenSearch / Vespa** | Dedicated and search-engine alternatives |
| **Debezium + Kafka (+ Kafka Connect)** | Change data capture and ordered event streams |
| **Spring Kafka / Spring JDBC**, **confluent-kafka (Python)** | Ingest workers |
| **VectorDBBench**, **k6**, **Locust** | Benchmarking and load testing |
| **Prometheus + Grafana**, **OpenTelemetry** | Metrics, tracing, and canary dashboards |
| **Helm / Kustomize / Terraform** | Reproducible deployments |
| **pgBackRest / WAL-G**, engine snapshot APIs | Backups and point-in-time recovery |
