# Multi-Query Retrieval - Code Walkthrough

Step-by-step guide to `example.js`: LLM query decomposition (Qwen via node-llama-cpp), parallel retrieval helpers, RRF and weighted fusion, and all eight examples.

---

## Table of Contents

1. [What is Multi-Query Retrieval?](#what-is-multi-query-retrieval)
2. [Setup and Configuration](#setup-and-configuration)
3. [Config, Helpers, and Logging](#config-helpers-and-logging)
4. [Initialization](#initialization)
5. [LLM Query Decomposition](#llm-query-decomposition)
6. [Knowledge Base and Vector Store](#knowledge-base-and-vector-store)
7. [Example 1: Query Decomposition](#example-1-query-decomposition)
8. [Example 2: Query Expansion and Parallel Retrieval](#example-2-query-expansion-and-parallel-retrieval)
9. [Example 3: Reciprocal Rank Fusion](#example-3-reciprocal-rank-fusion-rrf)
10. [Example 4: Perspective-Based Retrieval](#example-4-perspective-based-retrieval)
11. [Example 5: Multi-Part Questions](#example-5-multi-part-questions)
12. [Example 6: Weighted Query Fusion](#example-6-weighted-query-fusion)
13. [Example 7: Adaptive Strategy](#example-7-adaptive-strategy)
14. [Example 8: Deduplication Strategies](#example-8-deduplication-strategies)
15. [Main Runner](#main-runner)
16. [Quick Reference](#quick-reference)

---

## What is Multi-Query Retrieval?

Using **multiple search queries** instead of one to improve coverage and recall.

| Single query | Multi-query |
|--------------|-------------|
| One embedding, one search | Decompose or expand into several queries |
| Can overfit to one phrasing | Different phrasings retrieve different docs |
| May miss aspects of a complex question | Sub-queries cover each aspect; results fused (e.g. RRF) |

**Benefits:** Better coverage, handles complex questions, robust to phrasing, improved recall.

---

## Setup and Configuration

| Component | Source |
|-----------|--------|
| Embedding model | `models/bge-small-en-v1.5.Q8_0.gguf` |
| LLM (decomposition) | `models/Qwen3-1.7B-Q8_0.gguf` |
| Vector store | `embedded-vector-db`, dim 384, namespace `"multi_query"` |
| Config | `config.js` |
| Helpers | `multi-query-retrieval.js`: `retrieveParallel`, `deduplicateById`, `deduplicateByMaxScore`, `rrfFuse` |
| Logger | `logger.js`: `createLogger`, `timeAsync`, `metrics` |

**Imports and destructuring from config:**

```javascript
const { dim: DIM, maxElements: MAX_ELEMENTS, namespace: NS, rrfK, topKPerQuery } = config;
const logger = createLogger({ logLevel: config.logLevel });
```

Paths: `EMBEDDING_MODEL_PATH`, `LLM_MODEL_PATH` point under `models/`.

---

## Config, Helpers, and Logging

**`config.js`** - Tunables (env overrides in `ENV_MAP`): `dim`, `maxElements`, `namespace`, `rrfK`, `maxSubQueries`, `topKPerQuery`, `timeoutMs`, `retries`, `concurrency`, `logLevel`.

**`multi-query-retrieval.js`:**

| Function | Purpose |
|----------|---------|
| `retrieveParallel(queries, options)` | Run each query in parallel. Options: `getEmbedding`, `search`, `namespace`, `topK`, `retries`, `timeoutMs`, `concurrency`. Returns `Array<Array<result>>`. |
| `deduplicateById(results)` | One result per document ID (first occurrence). |
| `deduplicateByMaxScore(results)` | One result per ID (highest similarity), sorted by score. |
| `rrfFuse(resultLists, k)` | RRF; returns one list with `rrfScore`, sorted descending. |

**`logger.js`:** `logger.timeAsync(label, fn)`, `logger.metrics({ queryCount, totalResults, uniqueAfterDedup, durationMs })`.

---

## Initialization

**Embedding:** `initializeEmbeddingModel()` - `getLlama` → `loadModel(EMBEDDING_MODEL_PATH)` → `createEmbeddingContext()`.

**LLM:** `initializeLLM()` - `getLlama` → `loadModel(LLM_MODEL_PATH)` → `createContext()` → `new LlamaChatSession({ contextSequence })`.

Both created once in the main runner; `chatSession` is passed only into Example 1.

---

## LLM Query Decomposition

**Fallback** (when LLM fails or parse fails):

```javascript
const FALLBACK_SUBQUERIES = [
    "What is web application architecture?",
    "How to make web applications scalable?",
    "What are testing best practices?",
    "How to implement automated testing?",
];
```

**`parseSubQueriesFromLLM(raw)`:** Tries JSON array or object (`sub_queries` / `subQueries` / `queries`); else numbered/bullet lines. Returns `null` if nothing valid.

**`decomposeQueryWithLLM(complexQuery, chatSession)`:** Prompt asks for 3–5 sub-questions as a JSON array. Calls `chatSession.prompt(prompt, { maxTokens: 400 })`, parses with `parseSubQueriesFromLLM`. If at least 2 queries: `parsed.slice(0, config.maxSubQueries)`; else `FALLBACK_SUBQUERIES`.

---

## Knowledge Base and Vector Store

**`createKnowledgeBase()`** - Returns 25 `Document` instances with `id`, `category`, `topic`, `subtopic` (frontend, backend, database, testing, devops).

**`addDocumentsToStore(vectorStore, embeddingContext, documents)`** - For each document: `getEmbeddingFor(doc.pageContent)`, then `vectorStore.insert(NS, doc.metadata.id, vector, metadata)` with `metadata = { content: doc.pageContent, ...doc.metadata }`.

---

## Example 1: Query Decomposition

**Goal:** One complex query → LLM sub-queries → single-query baseline vs multi-query + RRF; show that multi-query yields more diverse, RRF-ranked results.

**Complex query (in code):**

```javascript
const complexQuery = "What are the best practices for designing scalable architectures and implementing effective testing strategies?";
```

**Flow:**

1. **Sub-queries** - `llmSubQueries = await decomposeQueryWithLLM(complexQuery, chatSession)`. Build `subQueries` by taking up to `config.maxSubQueries - 1` from `llmSubQueries`. Log under "Decomposed Sub-Queries:".
2. **Single-query baseline** - Embed `complexQuery`, search with `topK: 3`. Log 3 results (similarity, topic, 60-char content). Wrapped in `logger.timeAsync("single-query", ...)`.
3. **Multi-query** - `resultLists = await retrieveParallel(subQueries, { getEmbedding, search, namespace: NS, topK: 4, retries, timeoutMs })`. Wrapped in `logger.timeAsync("multi-query", ...)`.
4. **Fusion** - `fusedResults = rrfFuse(resultLists, rrfK)`.
5. **Display** - "Results (N unique documents, RRF-ranked):", first 8 items; score = `doc.rrfScore ?? doc.similarity`, then topic and 60-char content.
6. **Key insight** - Single query count vs multi-query count; documents supporting multiple aspects rank higher.

---

## Example 2: Query Expansion and Parallel Retrieval

**Goal:** Compare sequential vs parallel execution for expanded phrasings; deduplicate by ID.

**Query and expansion:**

```javascript
const originalQuery = "How to optimize database queries?";
const expandedQueries = [
    originalQuery,
    "What are database performance optimization techniques?",
    "Ways to improve database query speed",
    "Database indexing and query optimization strategies"
];
```

**Flow:**

1. **Sequential** - For each `expandedQueries`: embed, search `topK: 3`, push into `seqResults`. Measure `seqTimeMs`.
2. **Parallel** - `parResultLists = await retrieveParallel(expandedQueries, { ..., topK: 3 })`. Measure `parTimeMs`. `parResults = parResultLists.flat()`.
3. **Dedup** - `uniqueResults = deduplicateById(parResults)`.
4. **Log** - `logger.metrics({ queryCount, totalResults, uniqueAfterDedup, durationMs: parTimeMs })`. Print "Unique Results (N documents):", first 4 (topic, subtopic, 70-char content). Key insight: coverage and speedup.

---

## Example 3: Reciprocal Rank Fusion (RRF)

**Goal:** Several query variations in parallel → per-query results → RRF fuse → single ranked list.

**Queries:**

```javascript
const queries = [
    "React performance optimization",
    "How to make React apps faster?",
    "React rendering optimization techniques"
];
```

**Flow:**

1. **Parallel** - `allResults = await retrieveParallel(queries, { ..., topK: topKPerQuery })`.
2. **Per-query** - Log "Individual Query Results:", for each list "Query N:" and first 3 (rank, id, topic).
3. **RRF** - `fusedResults = rrfFuse(allResults, rrfK)`.
4. **Display** - "Fused Results (RRF):", first 5 with `rrfScore`, id, topic, 60-char content. Key insight: RRF formula, multi-list docs rank higher, `k` from `rrfK`.

---

## Example 4: Perspective-Based Retrieval

**Goal:** Same topic from different angles → results per perspective → combined unique count.

**Topic and perspectives:**

```javascript
const topic = "microservices architecture";
const perspectives = [
    { query: "What are the benefits of microservices?", label: "Benefits" },
    { query: "What are the challenges of microservices?", label: "Challenges" },
    { query: "How to implement microservices?", label: "Implementation" },
    { query: "When to use microservices vs monolith?", label: "Comparison" }
];
```

**Flow:**

1. **Parallel** - `perspectiveQueries = perspectives.map(p => p.query)`. `perspectiveResultLists = await retrieveParallel(perspectiveQueries, { ..., topK: 2 })`.
2. **Structure** - `perspectiveResults = perspectives.map((p, i) => ({ perspective: p.label, results: perspectiveResultLists[i] ?? [] }))`.
3. **Display** - "Results by Perspective:" for each label and its results (topic, category, 70-char content).
4. **Combined** - `allDocs = perspectiveResults.flatMap(pr => pr.results)`, `uniqueDocs = deduplicateById(allDocs)`. Log "Combined Coverage: N unique documents".

---

## Example 5: Multi-Part Questions

**Goal:** One multi-part question → explicit sub-questions → retrieve per part → show results by part.

**Query and parts:**

```javascript
const multiPartQuery = "What's the difference between unit testing and integration testing, and when should I use each?";
const parts = [
    { query: "What is unit testing?", label: "Unit Testing Definition" },
    { query: "What is integration testing?", label: "Integration Testing Definition" },
    { query: "When to use unit testing?", label: "Unit Testing Use Cases" },
    { query: "When to use integration testing?", label: "Integration Testing Use Cases" },
    { query: "Difference between unit and integration testing", label: "Comparison" }
];
```

**Flow:**

1. **Parallel** - `partQueries = parts.map(p => p.query)`. `partResultLists = await retrieveParallel(partQueries, { ..., topK: 2 })`.
2. **Structure** - `partResults = parts.map((p, i) => ({ part: p.label, results: partResultLists[i] ?? [] }))`.
3. **Display** - "Results by Question Part:" for each part (similarity, topic, 80-char content). Key insight: each aspect addressed.

---

## Example 6: Weighted Query Fusion

**Goal:** Main query + related queries with weights → weighted score fusion → main query prioritized.

**Main and related:**

```javascript
const mainQuery = "React state management";
const relatedQueries = [
    { query: "React hooks for state", weight: 0.8 },
    { query: "Redux state management", weight: 0.5 },
    { query: "Context API React", weight: 0.6 }
];
```

**Flow:**

1. **Main** - Embed `mainQuery`, search `topK: 5` → `mainResults`.
2. **Related** - `relatedQueriesOnly = relatedQueries.map(rq => rq.query)`. `relatedResultLists = await retrieveParallel(relatedQueriesOnly, { ..., topK: 3 })`. Zip with weights: `relatedResults = relatedQueries.map((rq, i) => ({ results: relatedResultLists[i] ?? [], weight: rq.weight }))`.
3. **Weighted scores** - `weightedScores = new Map()`. Main: each doc `score = similarity * 1.0`. Related: for each doc add `similarity * weight` to existing or new entry. Sort by score descending → `fusedResults`.
4. **Display** - "Weighted Fusion Results:", first 5 (score, topic, 70-char content).

---

## Example 7: Adaptive Strategy

**Goal:** Classify each test query by complexity → choose strategy (single, expansion, or complex with larger topK) → run and show result count and sample.

**`analyzeQueryComplexity(query)`:** Uses word count, "and"/"or"/"vs"/"compare", trailing "?", "difference"/"compare"/"versus"/"vs":

- **Very Complex** (multi-topic and word count > 15) → "Multi-Part Decomposition"
- **Complex** (comparison) → "Perspective-Based Retrieval"
- **Moderate** (word count > 10 and question) → "Query Expansion"
- **Simple** → "Single Query (no decomposition needed)"

**Test queries:**

```javascript
const testQueries = [
    "What is React?",
    "How do I optimize React performance?",
    "What's the difference between REST and GraphQL?",
    "How do I build a scalable web app with microservices, implement CI/CD, and ensure good test coverage?"
];
```

**Execution per query:**

- **Single Query** - `vectorStore.search(NS, vector, 3)` → `results`.
- **Query Expansion** - `queries = [query, \`How to ${query.replace(/^How (do|to) I /i, "")}\`].slice(0, config.maxSubQueries)`. `retrieveParallel(queries, { ..., topK: 2 })` → `results = deduplicateById(resultLists.flat())`.
- **Otherwise (complex)** - `vectorStore.search(NS, vector, 5)` → `results`.

Log "Results: N documents retrieved", first 2 (topic, category). Key insight: balance speed vs quality by complexity.

---

## Example 8: Deduplication Strategies

**Goal:** Same queries in parallel → three dedup strategies: by ID, by topic+category, by max score.

**Queries:**

```javascript
const queries = [
    "React state management hooks",
    "useState and useEffect in React",
    "React hooks for managing state"
];
```

**Flow:**

1. **Retrieval** - `resultLists = await retrieveParallel(queries, { ..., topK: 4 })`. `allResults = resultLists.flat()`. Log "Total Results: N (with duplicates)".
2. **Strategy 1 - By ID** - `byId = deduplicateById(allResults)`. "Unique documents: N", first 3 (id, topic).
3. **Strategy 2 - By topic + category** - `Map` key `${topic}-${category}`, one value per key → `byTopicCategory`. "Unique topic-category pairs: N", first 3 (topic, category).
4. **Strategy 3 - Max score** - `byMaxScore = deduplicateByMaxScore(allResults)`. "Unique documents (best score): N", first 3 (similarity, topic). Key insight: by ID simple; by topic groups; by max score best for ranking.

---

## Main Runner

**`runAllExamples()`:**

1. Clear console; print header and "What you'll learn" bullets.
2. Load embedding model (`OutputHelper.withSpinner`).
3. Create `VectorDB`, `createKnowledgeBase()`, `addDocumentsToStore(...)`.
4. Load LLM (`withSpinner`), get `chatSession`.
5. Run examples via `OutputHelper.runExample(title, fn)`:
   - Example 1: Query Decomposition (with `chatSession`)
   - Example 2: Query Expansion & Parallel Retrieval
   - Example 3: Reciprocal Rank Fusion
   - Example 4: Perspective-Based Retrieval
   - Example 5: Multi-Part Questions
   - Example 6: Weighted Query Fusion
   - Example 7: Adaptive Strategy
   - Example 8: Deduplication
6. Print "All examples completed successfully!", Key Takeaways, "When to Use Multi-Query" table, Best Practices, Result Fusion Methods.
7. On error: log message and hints (dependencies, model files in `models/`).

**Prerequisites:** `bge-small-en-v1.5.Q8_0.gguf` and `Qwen3-1.7B-Q8_0.gguf` in `models/`.

---

## Quick Reference

| Example | Strategy | Fusion / dedup |
|---------|----------|----------------|
| 1. Decomposition | LLM sub-queries, topK 4, RRF | `rrfFuse(resultLists, rrfK)` |
| 2. Expansion | Fixed expanded list, topK 3 | `deduplicateById` |
| 3. RRF | Fixed query list, topK from config | `rrfFuse` |
| 4. Perspective | Fixed perspectives, topK 2 | `deduplicateById` |
| 5. Multi-part | Fixed parts, topK 2 | Per-part display |
| 6. Weighted | Main + related with weights | Weighted sum, sort |
| 7. Adaptive | By complexity | Single / expansion / more results |
| 8. Dedup | Fixed queries, topK 4 | By ID, by topic+category, by max score |

**Practices:** Prefer `retrieveParallel`; use RRF for multiple ranked lists; use `deduplicateByMaxScore` for best similarity per doc; limit sub-queries (e.g. `maxSubQueries`); adapt strategy to complexity.
