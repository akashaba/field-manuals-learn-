# 11 — Caching

> **Module goal:** Slash latency and cost by not doing work twice. Master HTTP caching, application-layer key-value caching (Redis), LLM-specific caches (prompt cache, semantic cache, embedding cache), and the eternal problem — cache invalidation.

---

## 1. Executive Summary & Core Concepts

**Caching** stores the result of a costly computation and returns it directly on the next equivalent request. Its purposes for AI systems:

1. **Latency** — an LLM call is 500ms–30s; a Redis GET is 0.5ms
2. **Cost** — a prompt hit against Anthropic's built-in prompt cache is roughly 10× cheaper on cached tokens; a full-response cache skips the call entirely
3. **Load shedding** — cached responses don't touch the upstream provider, buying you headroom during traffic spikes
4. **Determinism** — repeated identical prompts return the same answer, which users like

**Cache taxonomy for an AI service:**

| Layer | What's cached | TTL | Backend |
|-------|--------------|-----|---------|
| **HTTP / CDN** | Public GET responses | seconds–days | Cloudflare, Fastly, CloudFront |
| **In-process** | Small, hot lookups | seconds–minutes | Python `functools.lru_cache`, `cachetools` |
| **Distributed KV** | Per-user, cross-replica data | minutes–hours | Redis, Memcached |
| **Prompt cache (provider-side)** | Static prompt prefix | minutes | Anthropic/OpenAI native |
| **LLM response cache** | Full model responses keyed by prompt | hours–days | Redis with prompt hash |
| **Semantic cache** | Fuzzy-matched responses via embedding NN | days | Redis + vector index |
| **Embedding cache** | Text → vector | forever (deterministic) | Postgres/Redis by hash |
| **Retrieval cache** | Query → top-K doc IDs | minutes–hours | Redis |

**Three cache pattern archetypes:**

- **Cache-aside** (lazy) — app reads cache first; on miss, computes, writes to cache. Most common. Simple, but stale possible.
- **Read-through / Write-through** — cache sits inline; every read fills, every write updates cache and DB atomically. Stronger consistency; harder to build.
- **Write-behind** — writes go to cache immediately, DB async. Fast, but data-loss risk on crash.

**Cache correctness = key design + invalidation.** "There are only two hard things in computer science: cache invalidation and naming things." — Phil Karlton. It's true. The key must include every variable that changes the result (model version, prompt version, retrieval corpus version, user locale, tenant ID). Missing any produces cross-tenant leakage or stale answers.

**Cache correctness ≠ cache effectiveness.** Even a correct cache is useless if the hit rate is 2%. You must measure `hits / (hits + misses)` per cache and per key pattern, and iterate.

---

## 2. Deep-Dive Breakdown

### 2.1 The Cache-Aside Pattern in Python

```python
import hashlib, json, redis
from typing import Any, Callable

r = redis.Redis(decode_responses=True)

def cache_key(*parts: Any) -> str:
    """Stable, collision-resistant key from arbitrary parts."""
    return "cache:" + hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]

def cached(ttl_sec: int):
    def deco(fn: Callable):
        async def wrapper(*args, **kwargs):
            key = cache_key(fn.__module__, fn.__name__, args, kwargs)
            hit = r.get(key)
            if hit is not None:
                return json.loads(hit)
            val = await fn(*args, **kwargs)
            r.set(key, json.dumps(val), ex=ttl_sec)
            return val
        return wrapper
    return deco

@cached(ttl_sec=3600)
async def embed(text: str, model: str = "text-embedding-3-large") -> list[float]:
    resp = await client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding
```

Notes:
- **`sort_keys=True`** so `{"a":1,"b":2}` and `{"b":2,"a":1}` hash the same
- Truncated SHA-256 for shorter keys (24 hex chars → 96 bits, collision-safe for practical scale)
- TTL forces eventual freshness even if you forget to invalidate

### 2.2 LLM Response Caching

Caching a full LLM response works when your prompt-and-context is deterministic and reused. Common cases: FAQs, deterministic classification, tool calls with fixed argument spaces.

```python
async def llm_call_cached(system: str, user: str, model: str, prompt_version: str) -> str:
    key = cache_key("llm", model, prompt_version, system, user)
    hit = r.get(key)
    if hit:
        METRICS.cache_hit.labels(type="llm").inc()
        return hit
    METRICS.cache_miss.labels(type="llm").inc()
    resp = await client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.0,   # determinism is important for caching
        max_tokens=1024,
    )
    text = resp.content[0].text
    r.set(key, text, ex=86400)
    return text
```

**Rules for LLM response caching:**
- **Temperature 0** or a fixed seed — non-determinism poisons cache hits.
- **Include prompt_version, model, and any external state** (retrieval corpus version, tool list hash) in the key.
- **Beware personalization.** If the response depends on user history, per-user prefix must be in the key — but then hit rates plummet. Consider a two-tier cache: user-agnostic response (high hit rate) + short personalization step on top.
- **Beware safety.** Caching a bad response replays it. Have a "purge by pattern" tool.
- **Beware time-sensitive answers.** "What's today's weather" cached for 24h is a bug. Add TTLs that reflect data freshness needs, or hash the response for freshness signals ("current" phrases).

### 2.3 Prompt Caching (Provider-Side)

Anthropic (`cache_control`) and OpenAI (automatic prompt caching) support caching a **prompt prefix** — typically a long system prompt, tool definitions, or a large document passed with every request. The provider stores the prefix's internal state and reuses it across requests, charging ~10% of normal input-token cost for the cached portion.

```python
resp = client.messages.create(
    model="claude-sonnet-5",
    system=[
        {"type": "text", "text": LONG_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": TOOL_DEFINITIONS, "cache_control": {"type": "ephemeral"}},
    ],
    messages=[{"role": "user", "content": user_input}],
)
# Response: usage.cache_creation_input_tokens=X, usage.cache_read_input_tokens=Y
```

Rules of thumb:
- Cache anything > 1024 tokens that repeats across requests
- Prefixes must match byte-for-byte from the start — reorganize your prompt so the stable part is at the top
- Cache blocks have a limited TTL (Anthropic: 5 minutes by default, or 1 hour with the extended-cache option); high-frequency use keeps them warm

For a chat app with many concurrent users sharing a large system prompt, prompt cache alone can cut input costs by 80–90%. This is one of the highest-ROI production optimizations available.

### 2.4 Semantic Cache — Fuzzy Match by Embedding

Exact-key caching doesn't help when user queries are semantically identical but textually different: "What's the weather in Paris?" vs "Paris weather?" — different SHA-256, different cache miss.

**Semantic cache** stores previous (embedding, response) pairs; on a new query, embed it, do a nearest-neighbor search, and if the top-K distance is below a threshold, return the cached response.

```python
import numpy as np

def semantic_cache_get(query: str, threshold: float = 0.06) -> str | None:
    q_vec = embed_sync(query)
    # Search vector index (pgvector, Qdrant, etc.)
    results = vector_index.search(q_vec, k=1)
    if not results or results[0].distance > threshold:
        return None
    return results[0].payload["response"]

def semantic_cache_set(query: str, response: str) -> None:
    vector_index.upsert(embed_sync(query), payload={"query": query, "response": response})
```

**Cautions:**
- Threshold tuning is critical. Too tight = few hits. Too loose = wrong-answer hits ("What's Paris weather?" returns cached "What's Prague weather?" answer). Validate on labeled pairs.
- Never semantic-cache safety-critical or personalized answers.
- Log semantic-cache hits with the *original* query and cached query so you can audit for confusions.
- Better on FAQ-style tasks; worse on open generation.

### 2.5 Cache Invalidation

Four strategies (in order of complexity):

1. **TTL (Time-To-Live)** — set an expiry; eventually the cache heals. Simple, but stale within the TTL window. Choose TTL based on staleness tolerance.
2. **Version-in-key** — bump a global (or per-domain) version number; old entries become naturally unreachable (and evicted on TTL). Great when writes are rare and reads are hot.
3. **Explicit invalidation** — on write, delete affected keys. Works for narrow single-object caches. Fragile for aggregates.
4. **Event-based invalidation** — publish an event on the source of truth; a consumer flushes affected cache keys. Highest complexity, best for large multi-service systems.

**LLM-specific version-in-key example:**

```python
CACHE_VERSION = {
    "model": "claude-sonnet-5",
    "prompt": "v3.2.1",       # bump when prompt changes
    "corpus": "docs-2026-09", # bump when RAG corpus refreshes
}

def llm_cache_key(user_input: str) -> str:
    return cache_key("llm", CACHE_VERSION, user_input)
```

When any of the components changes (deploy pushes new prompt version), old cache keys become unreferenced. Set TTL so they self-evict. **This gives you free rollback:** revert the version → old cache is reused.

### 2.6 Observability of Caches

Every cache should emit at least four metrics:
- `cache_hits_total{cache}`
- `cache_misses_total{cache}`
- `cache_size_bytes{cache}` (if bounded)
- `cache_evictions_total{cache, reason}` — LRU, TTL, manual

Hit rate = `hits / (hits + misses)`. Alert on:
- **Hit rate drops** (deploy broke keying; corpus refresh; sudden new query patterns)
- **Evictions climb** (cache size too small; TTL too long relative to memory)
- **P99 latency of GET** — a slow Redis is a red flag

`[IMG-11-01]` — *Prompt: A layered cache diagram for an AI service. From top: "Request enters" → decision diamond "In-process LRU (10ms)" → "Redis distributed cache (0.5ms)" → decision "Semantic cache (embedding search, 20ms)" → "Provider prompt cache (built-in)" → "Full LLM call (2000ms)". Each layer labeled with typical hit rate (60% / 30% / 5% / partial / 5%) and cost per hit. Show fallthrough arrows and populate-on-miss arrows. Modern architecture diagram style.*

---

## 3. Mental Models & Analogies

### Model 1: The Warehouse Racking System

Think of a warehouse: fastest-moving items sit on the picking floor at eye level (in-process, LRU). Slower items are on high shelves accessible with a lift (Redis, another network hop). Reserve stock is in a distant offsite warehouse (the origin service or LLM provider). The picker's job is to always reach for the closest available copy. Meanwhile, inventory management (invalidation) makes sure when a product changes SKU or gets recalled, every location's stock is updated — otherwise you sell customers the old thing.

Multiple cache layers stack the same way: each layer is *closer, faster, smaller*. A miss at layer N falls through to layer N+1. The trick to a well-tuned cache is the **hit rate at the innermost layer** — that's where the latency win compounds.

### Model 2: The Browser History and Tab Cache

Your browser doesn't reload a page every time you navigate back — it caches the DOM tree, images, scripts. Some of those cache entries have `Cache-Control: max-age=3600` (TTL); some are `ETag`-validated (conditional refresh); some are marked `no-store` (auth pages that must not be cached). And when you open incognito, the entire cache is scoped per-window — so your work-account and personal-account tabs don't cross-contaminate.

An LLM cache system is the same. Public FAQ answers: long TTL. User-personalized responses: per-user scope. Regulated PHI/PII responses: `no-store`. And the multi-tenant equivalent of incognito — cache keyed by tenant ID so Company A's cached responses never leak to Company B — is the single most important safety property in a multi-tenant AI service.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Cache-key omissions cause cross-tenant leakage.**
You cache `f"answer:{user_input}"`. User A on tenant Acme asks "What's our policy?"; user B on tenant Zenith asks the same string; both hit the same cache key and B sees Acme's answer. This is a **security breach**, not a bug. **Fix:** every key must include every identity-relevant scope: `tenant_id`, `user_id` (if answer is user-personalized), `locale`, `role`. Better yet, use a helper `key(*, tenant, user, prompt_version, ...)` where those args are keyword-only, so no one forgets one at a call site.

**Pitfall #2 — Caching non-deterministic LLM responses without setting temperature=0.**
You call `client.messages.create(...)` with default temperature, cache the response for 24h. Next user with same prompt gets the same (possibly bad) response, forever. If temperature was 0.7, the cached response is a *sample*, not the "correct" answer. **Fix:** either cache only temperature=0 responses (deterministic given identical prompt+model+version), or cache multiple samples and rotate, or don't cache generation at all — cache only the deterministic parts (embeddings, tool call plans, classifications).

**Pitfall #3 — TTL set to "long" without invalidation plan.**
You cache RAG retrieval results for 24h. You update the corpus at midnight. Users get yesterday's answers all day. **Fix:** use version-in-key with a corpus version that bumps on refresh; old entries evict on their own TTL, new entries reference the new version. Or shorter TTL that matches staleness tolerance. Or explicit flush on corpus swap. The pattern: **your TTL must be shorter than the freshness SLO**, or you need a plan to invalidate faster.

Related pitfall: **cold-cache stampede.** When a hot cache key expires, hundreds of concurrent requests all miss and hit the origin simultaneously (thundering herd on the LLM). Fixes: (a) lock-and-wait — first miss holds a lock, others wait; (b) probabilistic early refresh — refresh with probability rising as TTL nears expiry; (c) stale-while-revalidate — return the stale value while a single background task refreshes.

---

## 5. Self-Assessment Bank

**Q1 (MC):** In cache-aside, the application:
A) Writes to the cache first, then the database
B) Reads cache first; on miss, reads origin and populates cache
C) Only reads from cache; another system populates it
D) Uses the cache as a write buffer

**Q2 (short):** Why is `temperature=0` important when caching LLM responses?

**Q3 (MC):** Which is NOT a valid cache invalidation strategy?
A) TTL / eventual expiry
B) Version-in-key
C) Explicit delete on write
D) Never invalidate; assume everything's static

**Q4 (short):** Name one high-ROI cache layer specific to LLM APIs and explain why it saves cost.

**Q5 (MC):** A semantic cache uses:
A) SHA-256 of the query as the key
B) Embedding-based nearest-neighbor match with a similarity threshold
C) Regex normalization of the query
D) Language detection to pick a locale-specific response

**Q6 (short):** You cache RAG answers with 24h TTL. Your ops team refreshes the retrieval corpus daily at midnight. Sketch a bug this creates and a fix.

**Q7 (MC):** Cache-key design in a multi-tenant AI service MUST include which of the following at minimum?
A) tenant_id
B) prompt_version
C) model
D) All of the above

**Q8 (MC):** Anthropic's prompt caching feature charges cached input tokens at approximately:
A) The same rate as normal
B) ~10% of normal input token rate
C) Zero
D) 2× normal

**Q9 (short):** What is "thundering herd" in caching, and give one mitigation.

**Q10 (MC):** You add caching but your hit rate is 2%. What's the most likely issue?
A) Redis is slow
B) The key includes too many high-cardinality dimensions (e.g., `request_id`, timestamp)
C) The TTL is too short
D) Any of A, B, C — investigate all three

---

### Answer Key

**A1: B.** Cache-aside = the app pattern where the cache sits alongside the origin; app is responsible for read-through logic. Read cache → miss → compute → populate → return. It's the most common because it doesn't require the cache to be a first-class participant in your write path.

**A2:** LLMs with temperature > 0 return different responses to identical prompts. If you cache one such sample, every future user with that prompt gets that specific sample — often not the "best" one and possibly a low-quality outlier. Also, the caller may have assumed sampling variance across users (for diversity or exploration). Temperature=0 (greedy decoding) is deterministic given `(prompt, model, model_version)`, so the cached value is the *canonical* response. If you need diversity, cache multiple samples and rotate, or cache only downstream deterministic decisions (classifications, extractions) rather than raw generations.

**A3: D.** Static data is real — cache-forever is legitimate for immutable content (embeddings of a corpus that never changes; hashes of a file). But it's not a strategy for *dynamic* systems; it just delays the problem. In practice most caches need one of A/B/C.

**A4:** Any of:
- **Provider prompt cache** — Anthropic/OpenAI cache a long stable prefix (system prompt, tool definitions, big document). Cached tokens are billed at ~10% of normal, and latency drops sharply for cached portions. When many concurrent requests share a system prompt, this alone can cut input cost by 80–90%.
- **Embedding cache** — text→vector is deterministic given `(model, text)`. Cache forever. Saves the entire embedding call for repeat inputs (huge for FAQ-heavy or repetitive-ingestion workloads).
- **Semantic response cache** — matches semantically equivalent queries; a 40% hit rate on FAQ apps completely skips the LLM call.
- **Retrieval cache** — same query → same top-K docs (until corpus version changes). Skips the vector DB round-trip.

**A6:** Bug: users get *yesterday's answers* until each cached key hits its 24h TTL — up to 24 hours of staleness after a corpus refresh. Users may see hallucinations grounded in stale context ("our policy is X" when it was updated to Y at midnight). Fixes (any):
1. **Version-in-key**: include `corpus_version` in the cache key. Corpus refresh bumps the version; old keys become unreferenced and evict on TTL. New keys map to fresh answers immediately.
2. **Explicit flush on refresh**: after the midnight corpus swap, delete `rag:*` cache keys.
3. **Shorter TTL** aligned with the refresh cadence (e.g., 12h for a nightly refresh, so worst-case staleness is 12h — still not great, but bounded).
4. **Event-driven**: publish a `corpus_refreshed` event; a listener invalidates affected keys.

**A7: D — all of the above.** Missing `tenant_id` → cross-tenant leakage (security). Missing `prompt_version` → stale answers after a prompt fix. Missing `model` → mixing responses from different models (Sonnet vs Opus) as if equivalent. All three must be in the key; also often: locale, user_id (for personalized), tool-set hash, retrieval corpus version.

**A8: B.** Anthropic (and OpenAI) charge cached input tokens at roughly 10% of normal input pricing (specific ratios vary by model and tier, and cache-creation vs cache-read differ; consult the current pricing page). The break-even is generally reached in 2–3 uses of the same cached prefix — well within a normal request's lifespan for a shared system prompt.

**A9:** **Thundering herd** = when a hot cache key expires (or is invalidated), many concurrent requests all miss simultaneously and all call the origin at once, overloading it. Mitigations: (1) **lock-and-wait** — first miss holds a Redis lock; others wait for it to populate then read; (2) **probabilistic early refresh** — as TTL approaches expiry, a small fraction of requests preemptively refresh the cache while continuing to serve the cached value; (3) **stale-while-revalidate** — on expiry, return the stale value once and kick off a background refresh; subsequent requests during refresh get the stale value; (4) **staggered TTLs** with jitter so unrelated hot keys don't all expire at the same second.

**A10: D.** All three deserve investigation. (A) Slow Redis = high per-request latency, no direct hit-rate effect, but check anyway. (B) High-cardinality dimensions (request_id, session_id, timestamp) means each request produces a unique key that never repeats — a "cache" of one, hit rate → 0. Fix: strip volatile fields from the key. (C) TTL shorter than the expected reuse interval means keys expire before their second hit. Longer than reuse interval only helps if you have enough memory to store them. Also check: correctness of the key function (are you including a mutable object hash that changes each time?), Redis memory eviction (are you being evicted before TTL?), and workload variety (maybe no query genuinely repeats).

---

**Related modules:**
- `learning/10-rate-limiting.md` — cache hits don't consume rate-limit quota (or do; you choose)
- `learning/14-cost-optimization.md` — caching is a top-3 cost lever
- `learning/09-llm-observability.md` — cache metrics on the dashboard
- `../llm-engineering/13-rag-evaluation.md` — caching interacts with corpus versioning

**Practice prompts:**
1. Wrap your LLM call with an in-process LRU cache (`functools.lru_cache` or `cachetools.TTLCache`). Measure hit rate on 500 replayed prompts.
2. Add a Redis-backed cache with a version-in-key. Bump the version; confirm old entries become unreachable.
3. Instrument `cache_hits_total`, `cache_misses_total`. Add both to your Prometheus/Grafana dashboard.
4. Wire Anthropic prompt caching on a long system prompt. Measure cache_read_input_tokens vs cache_creation_input_tokens per request. Compute % cost saved.
5. Build a simple semantic cache with pgvector + Postgres. Tune the threshold on a labeled set.

**References:**
- Redis docs — https://redis.io/docs
- Anthropic prompt caching — https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Fastly and Cloudflare edge caching whitepapers
- Martin Fowler — "Patterns of Enterprise Application Architecture" (Cache-aside chapter)
- Phil Karlton's famous quote — the reason this module exists
