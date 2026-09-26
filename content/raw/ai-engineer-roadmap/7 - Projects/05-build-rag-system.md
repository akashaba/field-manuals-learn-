# Capstone Project 05 — Build a RAG System

> **Deliverable:** A retrieval-augmented generation service that lets you "chat with your documents." The corpus is a real one — the Montana Legislative bills PDF corpus, or your own document collection. The system ingests documents, chunks them, embeds them, stores them in pgvector, retrieves top-K on query, reranks, and generates grounded answers with citations. Includes eval with RAGAS.
>
> **Time:** 8–12 hours.
>
> **What you'll be able to say:** "I built a full RAG system: PDF ingestion, semantic chunking, pgvector index, hybrid retrieval, cross-encoder reranking, grounded generation with citations, and RAGAS eval on a golden set. Faithfulness 0.87, answer relevance 0.82. I know why my retrieval hit rate is 78% and what my next tuning move would be."

---

## 1. Project Overview

RAG has become the default way to make LLMs useful with private data. The pipeline has a lot of moving parts, and each is a potential quality lever.

### Pipeline

```
[PDF/HTML/MD documents]
        │
        ▼
[Ingest] parse → clean → strip boilerplate
        │
        ▼
[Chunk] split into 300–800 token chunks with 50-token overlap
        │
        ▼
[Embed] each chunk → vector via text-embedding-3-large (or Voyage AI, or open model)
        │
        ▼
[Index] pgvector table (chunk_id, embedding, text, doc_id, metadata)
        │
        ▼
[Query time]
    ┌───────────────────────────────────┐
    │ user query                        │
    │   │                                │
    │   ▼                                │
    │ embed query                       │
    │   │                                │
    │   ▼                                │
    │ vector search top-50              │
    │   │                                │
    │   ▼                                │
    │ hybrid: + BM25 top-50 → union     │
    │   │                                │
    │   ▼                                │
    │ rerank (cross-encoder) → top-5    │
    │   │                                │
    │   ▼                                │
    │ prompt LLM with retrieved context │
    │   │                                │
    │   ▼                                │
    │ answer + citations                 │
    └───────────────────────────────────┘
```

### Architecture
![IMG-CAP05-01](/7%20-%20Projects/images/IMG-CAP05-01.jpg)


### Prerequisites

- Python 3.11+
- Postgres + pgvector extension (via Docker: `pgvector/pgvector:pg16`)
- Anthropic API key (or OpenAI)
- Cohere API key (optional, for their reranker) OR a local cross-encoder
- Corpus: 20–100 PDFs of your choice

---

## 2. Step-by-Step Implementation

### Step 1 — Set up pgvector

```bash
# docker-compose.yml
version: "3.9"
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: dev
    ports: ["5432:5432"]
    volumes: ["./pgdata:/var/lib/postgresql/data"]
```

```sql
-- schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source_path TEXT,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER,
    embedding VECTOR(3072),  -- text-embedding-3-large dimension
    metadata JSONB
);

CREATE INDEX chunks_embedding_idx ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX chunks_text_gin ON chunks
  USING gin (to_tsvector('english', text));  -- for BM25-ish full-text search
```

**Why HNSW?** Approximate nearest-neighbor index; O(log n) query at billion scale with acceptable recall. Alternative is IVF-Flat (cheaper build, less recall on smaller sets). Cosine similarity is the standard metric for text embeddings.

### Step 2 — Ingest and chunk (`ingest.py`)

```python
"""ingest.py — parse, chunk, embed, insert."""
import re, uuid, hashlib
from pathlib import Path
from typing import Iterator
from pypdf import PdfReader
import tiktoken
from openai import OpenAI  # or Voyage AI / Anthropic when available

enc = tiktoken.encoding_for_model("text-embedding-3-large")
openai = OpenAI()  # embeddings — OpenAI or Voyage; Anthropic doesn't have first-party embeddings yet

def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # Normalize whitespace, strip headers/footers roughly
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> Iterator[str]:
    """Sliding window over the token sequence."""
    tokens = enc.encode(text)
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        yield enc.decode(chunk_tokens)
        if end == len(tokens):
            break
        start = end - overlap

def embed_batch(texts: list[str], model="text-embedding-3-large") -> list[list[float]]:
    resp = openai.embeddings.create(input=texts, model=model)
    return [d.embedding for d in resp.data]

def ingest_pdf(path: Path, conn) -> str:
    """Ingest one PDF; return document_id."""
    text = read_pdf(path)
    chunks = list(chunk_text(text))
    embeddings = []
    # Batch embed to save API calls
    for i in range(0, len(chunks), 50):
        embeddings.extend(embed_batch(chunks[i:i+50]))

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (title, source_path) VALUES (%s, %s) RETURNING id",
            (path.stem, str(path)),
        )
        doc_id = cur.fetchone()[0]
        for i, (text, vec) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """INSERT INTO chunks (document_id, chunk_index, text, tokens, embedding, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (doc_id, i, text, len(enc.encode(text)), vec, {"source": str(path)}),
            )
    conn.commit()
    return doc_id
```

**Why chunk at ~500 tokens with 50 overlap?** Too small = context is fragmentary; too large = retrieval brings in irrelevant text and burns prompt tokens. Overlap prevents a topic sentence from being split at a chunk boundary. Empirically 300–800 tokens works for most corpora; tune on your data.

**Why token-based chunking (not chars/sentences)?** Character length doesn't map to embedding capacity (`text-embedding-3-large` has an 8191-token limit). Sentence-based can produce chunks anywhere from 50 to 5000 tokens depending on writing style. Token-count-based is predictable.

**Why batch the embeddings?** Each API call has fixed overhead; batching to 50–100 items per call reduces cost and wall-time dramatically.

### Step 3 — Hybrid retrieval (`retrieve.py`)

```python
"""retrieve.py — vector + BM25 union."""
import psycopg
from typing import NamedTuple

class Chunk(NamedTuple):
    id: str
    text: str
    score: float
    doc_title: str

def vector_search(query_vec: list[float], conn, top_k: int = 50, tenant_id=None) -> list[Chunk]:
    with conn.cursor() as cur:
        # Cosine distance: <=> ; smaller = more similar
        cur.execute(
            """
            SELECT c.id, c.text, 1 - (c.embedding <=> %s::vector) AS score, d.title
            FROM chunks c JOIN documents d ON c.document_id = d.id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, top_k),
        )
        return [Chunk(*row) for row in cur.fetchall()]

def bm25_search(query_text: str, conn, top_k: int = 50) -> list[Chunk]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.text,
                   ts_rank_cd(to_tsvector('english', c.text),
                              plainto_tsquery('english', %s)) AS score,
                   d.title
            FROM chunks c JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.text) @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query_text, query_text, top_k),
        )
        return [Chunk(*row) for row in cur.fetchall()]

def hybrid_search(query_text: str, query_vec, conn, top_k: int = 50) -> list[Chunk]:
    v = vector_search(query_vec, conn, top_k=top_k)
    b = bm25_search(query_text, conn, top_k=top_k)
    # Reciprocal rank fusion — cheap, effective
    scores = {}
    for rank, c in enumerate(v):
        scores.setdefault(c.id, {"chunk": c, "s": 0.0})["s"] += 1 / (60 + rank)
    for rank, c in enumerate(b):
        scores.setdefault(c.id, {"chunk": c, "s": 0.0})["s"] += 1 / (60 + rank)
    fused = sorted(scores.values(), key=lambda x: -x["s"])
    return [x["chunk"] for x in fused[:top_k]]
```

**Why hybrid?** Vector search catches semantic similarity ("football" ≈ "soccer"); BM25 catches exact-match keywords and rare terms that embeddings blur ("HR-0451", "§ 7-8-201(3)(b)"). The union recovers cases where either alone would miss. Reciprocal Rank Fusion (RRF) is the standard cheap combiner — no need to calibrate scales.

### Step 4 — Reranking (`rerank.py`)

```python
"""rerank.py — cross-encoder rerank."""
from sentence_transformers import CrossEncoder

# Loaded once at startup
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    if not chunks:
        return []
    pairs = [(query, c.text) for c in chunks]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda p: -p[1])
    return [c for c, _ in ranked[:top_k]]
```

**Why rerank?** Bi-encoders (embedding-based retrieval) are fast but imprecise — they encode query and document independently. Cross-encoders concatenate query + document and run them through a Transformer together — much more accurate at picking the actually relevant chunk. Trade-off: too slow to score millions, perfect for scoring the top-50 the bi-encoder returned.

Alternative: Cohere Rerank (managed API, no local model needed):
```python
import cohere
co = cohere.ClientV2()
r = co.rerank(query=query, documents=[c.text for c in chunks], top_n=5, model="rerank-english-v3.0")
top = [chunks[hit.index] for hit in r.results]
```

### Step 5 — Grounded generation (`answer.py`)

```python
"""answer.py — prompt-and-cite."""
import anthropic

SYSTEM = """You are a research assistant. Answer the user's question using ONLY the provided context.

Rules:
1. If the context doesn't contain the answer, say "I don't have enough information to answer that."
2. Every claim must be attributed to a source in the form [chunk_id]. Multiple sources per claim allowed.
3. Do not fabricate. Do not use outside knowledge.
4. Be concise but complete.
"""

def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[chunk_{i}] (from: {c.doc_title})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return f"<context>\n{context}\n</context>\n\nQuestion: {query}"

async def answer(query: str, chunks: list[Chunk], client: anthropic.AsyncAnthropic) -> dict:
    prompt = build_prompt(query, chunks)
    resp = await client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        temperature=0.2,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": resp.content[0].text,
        "sources": [{"chunk_id": c.id, "title": c.doc_title, "text": c.text} for c in chunks],
        "usage": resp.usage.model_dump(),
    }
```

**Why "only from context, cite everything"?** Two of RAG's most common failures — hallucination (making up facts) and ungrounded elaboration (using training-data knowledge). This system prompt closes both. Citations force retrievability: users can verify.

**Why `temperature=0.2`?** Slight sampling variation for readability without introducing hallucination-style drift. Pure 0 sometimes produces terse or robotic answers.

**Why cache the system prompt?** It's stable across requests; cache_control ~10× cheaper on cached tokens.

### Step 6 — End-to-end orchestration (`rag.py`)

```python
"""rag.py — the top-level RAG function."""
async def rag_answer(query: str, conn, llm, top_k_retrieve: int = 50, top_k_rerank: int = 5):
    # 1. Embed query
    q_vec = openai.embeddings.create(input=query, model="text-embedding-3-large").data[0].embedding
    # 2. Hybrid retrieval
    candidates = hybrid_search(query, q_vec, conn, top_k=top_k_retrieve)
    # 3. Rerank
    ranked = rerank(query, candidates, top_k=top_k_rerank)
    # 4. Generate
    return await answer(query, ranked, llm)
```

### Step 7 — FastAPI service

```python
"""app.py"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import psycopg, anthropic

STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["conn"] = psycopg.connect("postgresql://postgres:dev@localhost/rag")
    STATE["llm"] = anthropic.AsyncAnthropic()
    yield
    STATE["conn"].close()

app = FastAPI(lifespan=lifespan)

class Q(BaseModel):
    query: str

@app.post("/ask")
async def ask(q: Q):
    return await rag_answer(q.query, STATE["conn"], STATE["llm"])
```

### Step 8 — Evaluation with RAGAS

```python
"""eval_ragas.py — run RAGAS on a golden set."""
from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall, faithfulness, answer_relevancy
)
from datasets import Dataset

# golden_set.jsonl format:
# {"question": "...", "ground_truth": "...", "reference_context_titles": ["doc-a", "doc-b"]}

async def run_eval(conn, llm, golden_set: list[dict]):
    rows = []
    for case in golden_set:
        result = await rag_answer(case["question"], conn, llm)
        rows.append({
            "question": case["question"],
            "answer": result["answer"],
            "contexts": [s["text"] for s in result["sources"]],
            "ground_truth": case["ground_truth"],
        })
    ds = Dataset.from_list(rows)
    scores = evaluate(ds, metrics=[
        context_precision, context_recall, faithfulness, answer_relevancy,
    ])
    return scores
```

Interpretation:
- **Context precision** (0-1): were the retrieved chunks actually relevant?
- **Context recall** (0-1): did retrieval get *all* the info needed?
- **Faithfulness** (0-1): does the answer stick to retrieved context (no hallucination)?
- **Answer relevance** (0-1): does the answer address the question?

Ship-quality bar for a first cut: all four > 0.75. Below that, tune the pipeline.

`[IMG-CAP05-02]` — *Prompt: A bar chart evaluation dashboard. Four bars labeled Context Precision (0.81), Context Recall (0.78), Faithfulness (0.87), Answer Relevance (0.82), each colored differently, y-axis 0-1. Below the bars, an annotation strip: "Retrieve top-50 → rerank top-5 → temperature=0.2 → gpt-embed-3-large → pgvector HNSW". Include a small "vs. baseline (no rerank)" comparison group with lower bars (0.68 / 0.72 / 0.79 / 0.76) shown in muted color to show the rerank uplift. Clean data-viz style, minimal chart junk.*

---

## 3. Failure Modes and How to Diagnose

**Symptom: model says "I don't have enough info" when it should have an answer.**
- Check retrieval hit rate: did the right chunk make it into the top-K? If not, retrieval is the bug.
- Try larger `top_k_retrieve`, better chunking, or query expansion (paraphrase before embedding).

**Symptom: hallucinated citations, or answer contradicts context.**
- Faithfulness low. Strengthen the system prompt ("do not use outside knowledge"); reduce temperature; try a stronger generator model.

**Symptom: retrieval brings in irrelevant chunks that push out relevant ones.**
- Context precision low. Add reranking (probably biggest single improvement); tune chunk size; try a better embedding model.

**Symptom: fast latency but wrong answers.**
- You skipped reranking. Add it — 200-400ms extra latency, big quality gain.

**Symptom: retrieval works on some queries, not others.**
- Different query classes need different retrieval strategies. Add query classification: exact-lookup ("what is section 5.2?") → BM25-heavy; conceptual ("how does X work?") → embedding-heavy.

---

## 4. Cost & Performance Considerations

Per query cost breakdown (approximate, at Anthropic Sonnet + OpenAI embeddings):
- Query embedding: ~$0.00001
- Vector search: $0 (self-hosted pgvector)
- Reranking (Cohere or local): $0.0001–0.001 depending
- Generation with 3k tokens of retrieved context + 500 output: ~$0.02

At scale: **prompt-cache the system instructions** (saves ~30-40% of generation cost); **cache the answer** for repeat queries (Module 11); **use Haiku** for simple factual lookups.

Latency budget:
- Query embed: 100-200ms
- Vector + BM25 retrieval: 20-50ms
- Rerank (top-50 → top-5): 100-400ms
- Generation (streamed): TTFT 400-800ms, full 3-8s

---

## 5. Extensions

- **Chunk metadata filtering:** date, source type, author. Filter *before* vector search.
- **HyDE (Hypothetical Document Embedding):** ask LLM to write a hypothetical answer, embed that, retrieve. Improves recall on abstract queries.
- **Multi-query rewriting:** generate 3 paraphrases of the user's question, retrieve for each, union results.
- **Parent-document retrieval:** embed small chunks but return their parent sections. Precision of small, context of large.
- **Query classification:** route factual → BM25; conceptual → hybrid; multi-hop → agent-with-search.
- **Structured citation output:** parse `[chunk_id]` tokens from the answer; return highlighted-source UI.

---

## 6. Interview Talking Points

- **"Walk me through a RAG system you built."** Use the pipeline diagram; talk chunking → embedding → hybrid retrieval → rerank → grounded generation → eval.
- **"How do you evaluate RAG?"** RAGAS's four metrics; golden set of Q-A-context triples; watch faithfulness especially.
- **"What are the failure modes of RAG?"** Retrieval miss (chunk isn't in top-K); irrelevant retrieval (chunk is there but ignored or displaced); hallucination despite good context; poor query understanding.
- **"How do you tune chunk size?"** Empirically on your corpus; 300-800 tokens usually; sensitive to document type (statutes benefit from smaller, narrative from larger).
- **"Why rerank if the vector search is already ranked?"** Bi-encoder is approximate; cross-encoder is much more accurate but only affordable on the top-K.
- **"How would you handle a corpus that updates?"** Version the corpus; include `corpus_version` in the cache key so old cached answers evict on refresh; support incremental ingestion (upsert by document hash).

---

## 7. References

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) — original RAG paper
- RAGAS documentation — https://docs.ragas.io
- pgvector README — https://github.com/pgvector/pgvector
- HyDE paper — Gao et al., 2022
- Anthropic's cookbook — RAG recipes with prompt caching
- Cohere Rerank docs
