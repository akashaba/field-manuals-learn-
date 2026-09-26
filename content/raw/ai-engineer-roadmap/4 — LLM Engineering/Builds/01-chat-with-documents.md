# Build #1 — "Chat with Your Documents" System (Advanced)

> **Track:** LLM Engineering · **Build:** 01
> **Prerequisites:** All 18 LLM-Engineering learning modules + prior builds.
> **Time budget:** ~50–70 hours over ~3 weeks.
> **Deliverable:** A public GitHub repo containing a **production-grade RAG system** — not a toy chatbot.
> **User directive:** *"But don't stop at a basic chatbot."*

---

## 1. Executive Summary

Anyone can wire OpenAI + Chroma + a Jupyter notebook and call it "chat with your PDF." That is not this build. This build is the **RAG system a real team would rely on**:

- Structured ingestion pipeline with change detection.
- Hybrid retrieval (dense + BM25) with cross-encoder reranking.
- Multi-strategy retrieval (query rewriting, multi-query, HyDE fallback).
- Grounded generation with citations, refusal, and structured output.
- Streaming responses with citation postprocessing.
- Full evaluation (RAGAS + your own eval set) in CI.
- Cost + latency observability.
- Multi-tenant permission-aware retrieval.
- Prompt caching, model routing, semantic caching.
- Docker + CI/CD; production-ready FastAPI service.

At the end, you should be able to hand the repo to a hiring manager and say: **"this is what a senior engineer's RAG looks like."**

---

## 2. Learning Outcomes

By the end you will have practiced:

1. Ingesting mixed-format documents (PDF, Markdown, HTML, code) with structure-aware chunking.
2. Building and maintaining a hybrid retrieval index (dense + BM25) in pgvector.
3. Applying query transformation (rewriting, multi-query, HyDE) to lift recall.
4. Adding a cross-encoder reranker to lift top-k precision.
5. Designing grounded-generation prompts with citations, refusal, and JSON structured output.
6. Streaming responses through FastAPI + SSE with client-side citation rendering.
7. Setting up RAGAS-based evaluation as a CI regression gate.
8. Instrumenting cost and latency; adding prompt caching and semantic caching.
9. Multi-tenant, permission-aware retrieval with row-level security.
10. Model routing (small → large fallback) for cost efficiency.

---

## 3. Scope

**Pick a corpus you actually care about.** Suggestions aligned with day-to-day life:

### 3.1 Option A — Montana Legislative Corpus
> Bill drafts, statutes, committee reports for one session. Complex structure (sections, subsections, references), formal language, real citations required.

### 3.2 Option B — Personal Knowledge Base
> Your notes, journals, papers, emails (Obsidian, Notion export, PDFs). Answer "what have I read about X?" or "what did I decide about Y?"

### 3.3 Option C — Open-Source Project Documentation
> Docs + issues + PRs from a big open-source project (Kubernetes, React, PyTorch). Answer developer questions from actual project state.

### 3.4 Option D — Academic Papers
> ~500 papers in a specific area (e.g., soccer analytics, algorithmic trading). Answer with citations to specific papers/sections.

**Requirements:**
- ≥ 100 documents; ≥ 1M total tokens.
- Mixed content types (text, tables, sometimes code or math).
- Realistic questions you actually want answered.

---

## 4. Repository Structure

```
chat-with-docs/
├── .github/
│   └── workflows/
│       ├── ci.yml               # lint + tests + eval regression
│       └── docker.yml           # build + push image
├── configs/
│   ├── prod.yaml
│   ├── dev.yaml
│   └── eval.yaml
├── data/
│   ├── raw/                     # source docs (gitignored)
│   ├── processed/               # normalized chunks (small sample committed)
│   └── eval/
│       └── eval_set.jsonl       # 100+ labeled Q/A/source triples
├── docs/
│   ├── architecture.md
│   ├── model_card.md
│   ├── data_dictionary.md
│   └── decisions.md             # ADRs
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_experiments.ipynb
│   └── 03_retrieval_evaluation.ipynb
├── src/
│   └── chat_docs/
│       ├── __init__.py
│       ├── config.py            # pydantic-settings
│       ├── ingest/
│       │   ├── loader.py        # PDF, MD, HTML, code loaders
│       │   ├── chunker.py       # structure-aware chunkers
│       │   ├── enricher.py      # metadata extraction, headers
│       │   └── indexer.py       # embed + upsert
│       ├── retrieval/
│       │   ├── dense.py         # pgvector cosine search
│       │   ├── bm25.py          # Postgres FTS or Elastic
│       │   ├── hybrid.py        # RRF fusion
│       │   ├── query_transform.py  # rewrite, multi-query, HyDE
│       │   ├── reranker.py      # cross-encoder or Cohere Rerank
│       │   └── filters.py       # permissions, freshness
│       ├── generation/
│       │   ├── prompt.py        # grounded-generation templates
│       │   ├── llm.py           # provider-agnostic wrapper (OpenAI/Anthropic/Local)
│       │   ├── streaming.py     # SSE streaming with citation postprocessing
│       │   └── router.py        # model routing / fallback
│       ├── cache/
│       │   ├── prompt_cache.py  # provider prompt caching
│       │   └── semantic_cache.py # response caching by embedding similarity
│       ├── observability/
│       │   ├── logging.py       # structured logs
│       │   ├── metrics.py       # Prometheus / stdout
│       │   └── tracing.py       # OpenTelemetry
│       ├── eval/
│       │   ├── ragas_eval.py    # RAGAS runner
│       │   ├── custom_metrics.py # citation validation, refusal correctness
│       │   └── judge.py         # LLM-as-judge
│       └── api/
│           ├── main.py          # FastAPI app
│           ├── schemas.py       # Pydantic
│           ├── auth.py          # api-key + user context
│           └── routes/
│               ├── chat.py      # POST /v1/chat
│               ├── ingest.py    # POST /v1/documents
│               └── admin.py     # health, metrics, evals
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_prompts.py
│   ├── test_api.py
│   └── test_eval_regression.py  # runs eval; asserts thresholds
├── artifacts/
│   └── config_v1.json           # deployment version
├── Dockerfile
├── docker-compose.yml           # includes pgvector, redis
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 5. Milestones & Acceptance Criteria

### M1 — Kickoff & Scaffolding (2–3h)
- [ ] Repo initialized on GitHub with license.
- [ ] `docker-compose.yml` runs Postgres + pgvector + Redis locally.
- [ ] `pyproject.toml` with pinned deps.
- [ ] Pre-commit + CI + Makefile targets: `setup`, `ingest`, `serve`, `test`, `eval`, `docker`.

### M2 — Ingestion Pipeline (5–7h)
- [ ] Loaders for PDF (unstructured or pypdf), Markdown, HTML, code.
- [ ] Structure-aware chunking per content type.
- [ ] Metadata extraction (title, section, page, timestamps, source).
- [ ] Change detection: content hash per doc; skip unchanged.
- [ ] Idempotent — rerunning doesn't duplicate.

### M3 — Indexing (4–5h)
- [ ] `pgvector` schema with HNSW index, JSONB metadata, source-doc FK.
- [ ] Postgres FTS index for BM25.
- [ ] Embedding pipeline using `text-embedding-3-large` (or BGE self-hosted).
- [ ] Batch API used for cost efficiency on the initial full ingest.
- [ ] Row-level security for permission-aware retrieval.

### M4 — Baseline RAG (3–4h)
- [ ] End-to-end: query → dense retrieve → top-k → LLM → answer.
- [ ] Bare-bones prompt (no citations yet).
- [ ] Sanity test: retrieve for 10 hand-crafted questions; inspect quality.

### M5 — Hybrid Retrieval (4–5h)
- [ ] BM25 retrieval via Postgres FTS.
- [ ] RRF fusion of dense + BM25.
- [ ] Bench: recall@10 dense vs hybrid on 50 eval queries. Expect +5–15% recall.

### M6 — Query Transformation (4–5h)
- [ ] Chat-history-aware query rewriter (LLM).
- [ ] Multi-query expansion (generate 3 paraphrases).
- [ ] HyDE fallback for out-of-domain queries.
- [ ] Cache rewritten queries by hash to avoid recomputation.

### M7 — Reranking (2–3h)
- [ ] Cross-encoder reranker (Cohere Rerank or `bge-reranker-v2-m3`).
- [ ] Retrieve `k=30`, rerank to `k=5`.
- [ ] Bench: top-3 accuracy vs baseline. Expect meaningful lift.

### M8 — Grounded Generation (5–6h)
- [ ] Production-quality grounded-generation prompt.
- [ ] Structured JSON output: `{answer, citations[], answered, confidence}`.
- [ ] Explicit refusal permitted and tested.
- [ ] Cite each claim with `[N]` markers + optional quote for verifiability.

### M9 — Streaming + Citation Rendering (3–4h)
- [ ] Server-Sent Events (SSE) endpoint streaming JSON structured output tokens.
- [ ] Client-side: render inline `[N]` markers as clickable citations; render source footer.
- [ ] Verify streaming produces citations correctly even mid-stream.

### M10 — Caching (3–4h)
- [ ] **Prompt caching** on the system prompt + source-doc portion (Anthropic style).
- [ ] **Semantic response caching** — hit rate visible in metrics.
- [ ] **Embedding cache** — never re-embed the same text (Redis + hash key).

### M11 — Model Routing (2–3h)
- [ ] Small model (Claude Haiku / GPT-4o-mini) tries first.
- [ ] Validate output: cited claims, non-empty, JSON valid.
- [ ] Escalate to larger model on validation failure.
- [ ] Log which model handled each request.

### M12 — Evaluation (5–6h)
- [ ] Curate 100+ labeled Q/A/source triples for your corpus.
- [ ] RAGAS suite: faithfulness, answer_relevancy, context_precision, context_recall.
- [ ] Custom: citation validation (do cited quotes appear in sources?).
- [ ] Refusal correctness: for "unanswerable" questions, does the model refuse?
- [ ] Baseline metrics documented in `docs/decisions.md`.

### M13 — Observability (3–4h)
- [ ] Structured logs (JSON) with request_id, user_id, retrieved chunk IDs, model, latency, cost.
- [ ] Prometheus metrics endpoint.
- [ ] Optional: OpenTelemetry traces to Jaeger / Honeycomb.
- [ ] Cost dashboard showing spend by endpoint and model.

### M14 — API + Auth (3–4h)
- [ ] FastAPI with `/v1/chat`, `/v1/documents`, `/v1/eval`, `/healthz`, `/readyz`, `/metrics`.
- [ ] API-key auth + `user_id` extraction for permission filtering.
- [ ] Rate limiting per key.
- [ ] Pydantic schemas; documented in OpenAPI.

### M15 — Multi-Tenant / Permission-Aware (3–4h)
- [ ] Postgres row-level security or namespace-per-tenant.
- [ ] Every retrieval scoped to `user_id`'s allowed docs.
- [ ] Test: cross-tenant queries return no data.

### M16 — Testing & CI (3–4h)
- [ ] Unit tests for every module.
- [ ] Integration tests for the full pipeline.
- [ ] `test_eval_regression.py` runs RAGAS on the eval set; fails CI if metrics regress > threshold.
- [ ] Coverage ≥ 75% on `src/chat_docs/`.

### M17 — Docker + Deploy (3–4h)
- [ ] Multi-stage Dockerfile < 2 GB.
- [ ] `docker compose` runs full stack (API + pgvector + redis).
- [ ] CI builds and pushes to GHCR.
- [ ] Optional: deploy to Fly.io / Render / Railway with a live demo URL.

### M18 — README + Demo (2–3h)
- [ ] README with quickstart (< 15 min from clone to running query).
- [ ] Architecture diagram.
- [ ] Sample queries with expected results.
- [ ] Link to model card and eval report.

---

## 6. Non-negotiables

1. **Hybrid retrieval + reranking** — dense-only is not acceptable.
2. **Citations** — every answer cites; every citation is machine-checkable.
3. **Refusal** — the system says "I don't know" when it should.
4. **Structured output** — JSON with well-defined fields (answer, citations, confidence, answered).
5. **Streaming** — user-facing latency feels < 1 second.
6. **Evaluation in CI** — no PR merges if metrics regress beyond threshold.
7. **Cost + latency logged** on every call.
8. **Permission-aware retrieval** — filters enforced at the DB layer, not the prompt.
9. **Prompt caching enabled** on any large stable prefix.
10. **All secrets in env vars**, never committed.

---

## 7. Nice-to-Haves (stretch)

- **Agentic multi-step retrieval** — the model can call `retrieve()` as a tool iteratively.
- **Cross-encoder fine-tuning** on your corpus for a task-specific reranker.
- **LoRA fine-tuning** of a small local model on your instruction data.
- **UI**: Next.js chat interface with inline citations and source viewer.
- **Slack / Teams bot** wrapper.
- **Human-in-the-loop**: thumbs-down auto-adds cases to eval set.
- **Automated eval on production traffic sample** — nightly job scores 1% of prod queries.
- **A/B testing framework** — traffic split between two configs, metrics compared.

---

## 8. Evaluation Rubric

| Area | 1 (poor) | 3 (ok) | 5 (excellent) |
|------|----------|--------|---------------|
| Ingestion | one file type | multiple types | structure-aware + metadata + change detection |
| Retrieval | dense-only | hybrid | hybrid + query transform + reranker |
| Grounding | free-form answer | prompt says "use context" | full grounded prompt + refusal + structured JSON |
| Citations | none | referenced in text | inline `[N]` + verifiable quotes + rendered UI |
| Streaming | not implemented | plain text stream | SSE with structured chunks + client rendering |
| Caching | none | response cache | prompt + semantic + embedding caches |
| Cost/latency | not measured | logged | dashboard + alerts + model routing |
| Multi-tenant | none | filter in code | row-level security or namespace isolation |
| Evaluation | anecdotal | one metric | RAGAS suite + custom + CI regression |
| Docs | short README | + architecture + model card | + decisions + eval report + reproducibility |

Aim for 4+ across every row before shipping.

---

## 9. Related Files

- `../goals.md`
- `../learning/12-rag-chunking.md` through `../learning/18-rag-evaluation.md`
- `../learning/03-llm-apis.md`, `../learning/04-prompting.md`, `../learning/05-structured-outputs.md`
- `../learning/10-cost-latency-optimization.md`
- `../../deep-learning/builds/01-image-text-classification.md`

---

## 10. Final Advice

**"Don't stop at a basic chatbot"** — the user's directive echoes throughout every module of this track. The distinction between a demo and a system:

- A demo asks: "does it work sometimes?"
- A system asks: "does it work reliably? cheaply? measurably? correctly? at scale?"

Every non-negotiable above is a "does it work" question the user's team will ask when they inherit this. Ship as if you're handing it to a colleague who trusts it in production tomorrow.

That colleague may be your future self at your next job. Build for them.
