# Month 4 — LLM Engineering 🤖 — Goals

> **Track:** LLM Engineering (Module 4 of the broader learning journey)
> **Prerequisites:** Months 1–3 (Foundations, ML, DL — especially the attention/transformers/LLMs modules).
> **Meta-goal:** *Ship production LLM systems — not toy chatbots.*

---

## 1. The North Star

The single sentence that governs every decision this month:

> **"When someone hands me a real business problem — a pile of PDFs, a support inbox, a policy manual, an internal API — I can design and ship an LLM system that solves it: right model, right prompting, right retrieval, right structure, right cost, right latency, right evaluation. Not a demo. A system that colleagues rely on."**

Every module below is scored against that sentence.

---

## 2. Terminal Learning Objectives (mastery, not exposure)

### 2.1 Tokens & Embeddings
- **LO-1.1** Explain BPE/tokenization; predict token counts; diagnose tokenization-induced bugs.
- **LO-1.2** Use embedding models (OpenAI, `sentence-transformers`, BGE, Cohere) fluently; pick the right one.
- **LO-1.3** Compute cosine similarity; understand embedding dimensionality and its cost/quality tradeoff.

### 2.2 Working with LLM APIs
- **LO-2.1** Call OpenAI, Anthropic, Google, and open-weight (via together.ai / groq / etc.) APIs interchangeably.
- **LO-2.2** Handle streaming, retries, rate limits, structured errors, cost accounting.
- **LO-2.3** Cache aggressively — prompt caching, KV-cache-aware batching, semantic cache.

### 2.3 Prompt Engineering
- **LO-3.1** Apply zero-shot, few-shot, chain-of-thought, and multi-step prompting techniques where each is appropriate.
- **LO-3.2** Write system prompts that reliably shape model behavior.
- **LO-3.3** Diagnose bad prompts via output analysis; iterate methodically.

### 2.4 Structured Outputs & Tools
- **LO-4.1** Get reliable JSON out of LLMs — grammar-constrained decoding, JSON mode, function calling.
- **LO-4.2** Design tool schemas the model understands; validate tool calls.
- **LO-4.3** Build multi-tool loops (function → observe → next call) safely.

### 2.5 Production Concerns
- **LO-5.1** Pick a model given a spec (quality/cost/latency/features required).
- **LO-5.2** Manage context windows — packing, summarization, sliding, retrieval offload.
- **LO-5.3** Optimize cost and latency — caching, batching, quantization, speculative decoding awareness.
- **LO-5.4** Use HuggingFace `transformers`, `datasets`, and the Hub confidently.

### 2.6 Retrieval-Augmented Generation
- **LO-6.1** Chunk documents intelligently — token-aware, structure-aware, semantic-aware.
- **LO-6.2** Build and maintain a vector index (pgvector, Qdrant, FAISS) with metadata.
- **LO-6.3** Combine dense + sparse retrieval (hybrid, BM25 + embeddings).
- **LO-6.4** Rerank with cross-encoders or LLM-as-judge to improve top-k precision.
- **LO-6.5** Ground generation in retrieved context with citations; refuse when the context is insufficient.
- **LO-6.6** Evaluate RAG systems with faithfulness, context precision/recall, answer relevance metrics.

---

## 3. Deliverable

You will produce **one** portfolio-grade artifact:

**Build #1 — "Chat with Your Documents" System (Advanced).** A full RAG-based document Q&A system with hybrid retrieval, reranking, citations, streaming, evaluation, and cost/latency observability. See `builds/01-chat-with-documents.md`.

**The user was explicit: "But don't stop at a basic chatbot."** The build spec reflects that — the deliverable is a production-grade system, not a Jupyter demo.

---

## 4. Definition of Done (per topic)

A topic is **done** when you can:

- [ ] Explain the concept to a non-ML engineer teammate in 5 minutes.
- [ ] Implement or configure it in code that runs.
- [ ] Diagnose one common failure mode from lived experience.
- [ ] Score ≥ 8/10 on the module's Self-Assessment Bank.

---

## 5. Anti-goals (what you are *not* doing here)

- **You are not fine-tuning foundation models this month.** LoRA on a small model, maybe. Pretraining? Absolutely not.
- **You are not implementing a transformer from scratch.** Month 3 covered that. This month is about **using** LLMs.
- **You are not chasing the newest model.** Understanding the tradeoffs > owning the shiniest name.
- **You are not building an agent framework from scratch.** Use LangGraph, DSPy, or write minimal glue code. Framework wars are a distraction.
- **You are not evaluating with vibes.** Every model change must be justified by a metric you can defend.

---

## 6. Cadence (suggested)

| Week | Focus |
|------|-------|
| 1 | Tokens, embeddings, LLM APIs, prompting basics |
| 2 | Structured outputs, tool calling, streaming, model selection |
| 3 | Context windows, cost/latency, HuggingFace, RAG chunking + embeddings + vector DBs |
| 4 | RAG retrieval + reranking + grounded generation + evaluation — **Build #1** |

Keep the build immovable.

---

## 7. Success Signals

You'll know Month 4 is behind you when:

- You reach for `tiktoken`/`Anthropic.count_tokens()` before predicting whether a prompt fits.
- You automatically add a JSON schema to structured-output requests; you never `try: json.loads`.
- You reject models by their capability profile (function calling, structured output, context length, cost), not by hype.
- You feel physical discomfort when someone says "let's just embed everything and cosine search it" without asking about chunking, hybrid search, or reranking.
- You have opinions on `all-MiniLM-L6-v2` vs `bge-large-en-v1.5` vs `text-embedding-3-small` — and can defend them.
- You know your RAG faithfulness score, not just "it seems to work."
- When someone says "the model hallucinated," you have a diagnostic checklist ready.

---

## 8. Related Files

- `learning/01-tokens.md` … `learning/18-rag-evaluation.md`
- `builds/01-chat-with-documents.md`
- `self-assessment.md`
