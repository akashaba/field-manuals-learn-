# Memory — Master Study Guide

> **Track:** Agents + Production · **Module:** 05
> **Prerequisites:** Modules 01–04.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** LLMs are **stateless** — they remember nothing between requests. Every "memory" an agent has is engineered: state you decide to persist, structure you impose, retrieval you invoke. Without memory, an agent forgets the user's name between messages, learns nothing from past runs, and can't span sessions.

Building memory well is a rich design problem. There's no single "memory system" — you decide what to remember (facts? messages? preferences? tools used?), how to store it (in-context, DB, vector store, structured JSON), and how to retrieve it (always, on-demand, queried).

**Fundamental principles you must own:**

1. **Memory is a design decision, not a library feature.** Every agent's memory is shaped by its task.
2. **Different memory types serve different purposes** — short-term, long-term, episodic, semantic, procedural.
3. **Storage matters** — messages list, structured JSON, vector store, relational DB. Each has trade-offs.
4. **Retrieval matters** — how you load relevant memory into context per turn.
5. **Cost and privacy** — memory that grows unboundedly costs money and can leak sensitive info.
6. **Forgetting is a feature** — decay, pruning, and summarization keep memory useful.

If you retain nothing else: **memory is engineered state around the LLM — pick the type, storage, and retrieval that fit your task.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Memory Taxonomy

Borrowed loosely from cognitive science; useful for agent design:

**Short-term / working memory** — the current conversation and reasoning trace. Lives in the messages list. Bounded by context length. Cleared between sessions.

**Long-term memory** — information that persists across sessions.
  - **Episodic memory** — records of specific past events ("user asked about X on Tuesday and I answered Y").
  - **Semantic memory** — facts about the world or the user, extracted from experiences ("user prefers metric units; user's spouse's name is X").
  - **Procedural memory** — learned procedures ("when the user asks about weather, I use tool Y and format as Z").

**Domain memory** — data the agent uses to do its job — RAG corpus, tool documentation. Different from user-facing memory.

Each type has different **write** and **read** patterns:

| Type | Write when | Read when |
|------|-----------|-----------|
| Short-term | Every turn | Every LLM call |
| Episodic | End of interaction | On demand / by query |
| Semantic | On extracting fact | Beginning of session / relevance-based |
| Procedural | Rarely (learned patterns) | Beginning of session |
| Domain | Batch ingestion | Retrieval per query |

Modern chatbots (ChatGPT memory, Claude's persistent memory) combine semantic + episodic + domain.

---

### 2.2 Short-Term Memory: The Conversation

Short-term memory is your messages list. Everything the LLM sees each turn. Simple in concept; tricky at scale.

**Growing pains:**

- **Cost** — every turn re-sends the history.
- **"Lost in the middle"** — long histories degrade recall.
- **Context limits** — even a 200k model has a limit.
- **Latency** — TTFT scales with input length.

**Mitigations:**

**Sliding window.** Keep the last N turns. Drop older messages.

```python
def keep_last_n_turns(messages, n=10):
    # Preserve system prompt + last n messages
    system = [m for m in messages if m["role"] == "system"]
    tail = messages[-n:]
    return system + tail
```

Simple but loses context.

**Summarization.** When history exceeds a threshold, summarize the oldest portion into a running summary.

```python
def maybe_summarize(messages, threshold_tokens=8000):
    if count_tokens(messages) < threshold_tokens:
        return messages
    old = messages[:len(messages)//2]
    summary = llm(f"Summarize this conversation, preserving key facts: {old}")
    kept = messages[len(messages)//2:]
    return [{"role": "system", "content": f"Prior conversation summary: {summary}"}, *kept]
```

Preserves gist; loses detail.

**Hierarchical summarization.** Multiple summary levels — recent turns raw, medium-age summarized to a paragraph, ancient summarized to a line.

**Selective retention.** Keep messages that are important; drop the rest.
- Keep every user turn (they're small and carry intent).
- Drop successful tool observations after they've been "used" (the LLM's next message referenced the result).
- Keep decisions the LLM made (they're context for future decisions).

**External offload.** Move detailed state to a database; the LLM sees compact references.

Rule: **profile your token usage per turn.** If short-term memory is > 30% of your token budget, invest in a mitigation.

---

### 2.3 Long-Term Memory: Fact Extraction and Recall

Long-term memory lets an agent remember across sessions: preferences, prior conversations, learned patterns.

**Architectural pattern — the "memory manager":**

1. **Extraction.** After each interaction (or periodically), an LLM extracts durable facts from the conversation.
2. **Storage.** Store facts in a structured format — key-value, JSON, or per-topic markdown files.
3. **Retrieval.** At the start of a session (or per turn), load relevant memory into the system prompt.

**Fact extraction:**

```python
def extract_facts(conversation):
    prompt = f"""Extract durable facts about the user from this conversation.
    Only facts they stated. Return JSON list of {{"topic", "fact"}}.

    Conversation: {conversation}

    Facts:"""
    return json.loads(llm(prompt))
```

**Storage schemas:**

- **Flat key-value** — `{"user_name": "Brian", "unit_preference": "metric"}`. Simple.
- **Per-topic files** — `/user/profile.md`, `/user/preferences.md`, `/topics/soccer.md`. Rich structure, human-readable.
- **Vector store** — embed each fact; retrieve by relevance to current query. Best for large memory.
- **Relational DB** — for typed data with joins.

**Retrieval strategies:**

- **Always load** — every session starts with the full user profile. Best for small stable memory.
- **On-demand tool** — expose a `read_memory(query)` tool; the agent decides when to consult memory.
- **Relevance-based** — embed the current query; retrieve top-k relevant memory chunks.
- **Rule-based** — if user asks about food, load `/topics/food.md`.

**When memory conflicts.** User said "call me Brian" months ago; now says "actually, call me Coach." Resolve by:
- **Recency wins** — newer facts override older ones.
- **Explicit-update wins** — user explicitly changing something overrides implicit past mentions.
- **Confidence scoring** — if you're unsure, ask the user to confirm.

**Privacy.** Long-term memory contains user data. Some categories should NEVER be stored (health details in some jurisdictions, government IDs, minors' info). Consent matters — some products ask before writing to memory.

---

### 2.4 Memory Stores and Retrieval

**Storage backend choices:**

**Structured markdown / JSON files.** Human-readable, git-diffable, easy to inspect.
- Pros: transparent, portable, versionable.
- Cons: no efficient query beyond file-load.
- Fit: personal assistants, small-scale memory.

**Vector store** (pgvector, Qdrant, ...). Store facts as embedded chunks; retrieve by relevance.
- Pros: scales; captures semantic similarity.
- Cons: opaque; approximate; not human-readable.
- Fit: many-topic personal assistants, RAG-like memory.

**Relational DB** (Postgres). Typed schema; strict.
- Pros: query with SQL; joins; transactions.
- Cons: schema evolution effort.
- Fit: business-user data (CRM-like).

**Key-value store** (Redis, DynamoDB). Fast simple lookup.
- Pros: sub-ms reads.
- Cons: no complex queries.
- Fit: session state, active-user profile.

**Graph DB** (Neo4j). Entities and relationships.
- Pros: expresses "X knows Y, works at Z, has interests A, B, C."
- Cons: complexity; niche.
- Fit: social/entity-centric agents (research assistants tracking who-said-what).

**Real systems combine.** Common pattern: profile in structured markdown + vector store for episodic memory + a scratchpad for session state.

**Memory query design:**

Retrieval-based memory looks a lot like RAG (Month 4 modules 12–18). Same tools: chunking (per-fact granularity), embeddings, vector search, reranking. Reuse those skills.

Key differences from RAG:
- **Smaller** than doc corpora — usually thousands, not millions, of facts.
- **User-scoped** — filter by user_id at every query.
- **Updated more frequently** — user's preferences change; write path matters.
- **Higher stakes on wrongness** — remembering a wrong fact is worse than not remembering.

---

### 2.5 Forgetting, Decay, and Consolidation

Unbounded memory grows expensive and noisy. Forgetting is essential.

**Explicit forget.**
- User asks to forget ("forget my previous address"). Honor immediately.
- Time-based deletion. Auto-delete facts older than N months unless refreshed.
- Sensitivity-based auto-delete. PII, one-time codes.

**Consolidation.**
- Merge duplicate or near-duplicate facts.
- Update facts on new evidence rather than adding both old and new.
- Convert "user mentioned X once" into "user cares about X" only after N mentions.

**Decay.**
- Weight relevance by recency. A 2-year-old preference is less confident than a 2-week-old one.
- Reduce weight on unused facts; boost weight on referenced ones.

**Structured writes.** Every fact carries metadata:
- `source` — where did we learn this?
- `first_seen` — when?
- `last_confirmed` — when last mentioned?
- `confidence` — how sure are we?
- `sensitivity` — is this OK to persist?

Retrieval and use of memory can then respect these signals.

**Right-to-be-forgotten.** For regulated data (GDPR, CCPA), users have legal rights. Design memory to support hard deletes, not just "hidden".

**Anti-goal: hoarding.** More memory ≠ better agent. Signal-to-noise matters. Prune aggressively; keep memory small and high-quality.

---

## 3. Mental Models & Analogies

### 3.1 The "Assistant with a Notebook" Model

Your agent is a personal assistant. Its memory is a **notebook** they carry between meetings with you:

- **Short-term memory** = notes from today's meeting — everything discussed, fresh in mind.
- **Long-term memory** = the notebook itself — accumulated notes across all past meetings.
- **Episodic** = "on Nov 3, Brian asked about X" — chronological log.
- **Semantic** = "Brian prefers coffee to tea" — distilled facts.
- **Procedural** = "when Brian asks about the weather, I check three sites and give a comparison" — learned habits.

A good assistant:
- Writes down important things (extraction).
- Skims relevant pages before each meeting (retrieval).
- Discards yesterday's grocery list (forgetting).
- Highlights conflicting notes for confirmation (conflict resolution).
- Never reveals notes from another client (user isolation).

Your memory system is you designing that assistant's habits. Bad habits (hoarding, forgetting critical, cross-mixing users) show up as failures.

### 3.2 The "Filing Cabinet + Sticky Notes" Model

Two-tier memory:

- **Sticky notes on the desk** = short-term memory. What you're working with *right now*. Cluttered? Toss them.
- **Filing cabinet across the room** = long-term memory. Organized (topic-labeled folders); retrievable by index; safely stored.

Some tasks need only the sticky notes (one-off Q&A). Some need frequent trips to the filing cabinet (personal assistant, research over history). The design question is: **what belongs on stickies, what belongs in the cabinet, and when do you go get files?**

Fetching everything from the cabinet every turn is wasteful. Never fetching means you re-discover the same facts every meeting. Design retrieval to load what you need, when you need it.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Just Save Every Turn to Long-Term Memory"

Terrible signal-to-noise. Most turns don't produce durable facts worth remembering. Extract deliberately: after each interaction, use an LLM to distill 0–3 facts worth persisting; discard the rest.

### 4.2 "Load All Memory Into Every Prompt"

Expensive. Grows over time until it eats context. Instead: retrieve *relevant* memory per turn (relevance-based, tool-based, or rule-based). Load the user's profile once at session start; retrieve topic-specific memory as needed.

### 4.3 "Memory Is Just Storage"

Memory is storage + retrieval + consolidation + forgetting + privacy. Missing any of these produces failures:
- Storage without retrieval → memory nobody reads.
- Retrieval without consolidation → stale contradictory facts.
- Consolidation without forgetting → memory grows forever.
- Any without privacy controls → GDPR liability.

---

## 5. Self-Assessment Bank (Memory)

### Questions

**Q1 (Short answer).** Distinguish short-term, episodic, and semantic memory in one sentence each.

**Q2 (Multiple choice).** Agent memory across sessions is fundamentally:
- (a) Built into the model.
- (b) Engineered state around the model — the LLM is stateless.
- (c) Automatic in the API.
- (d) Only possible with fine-tuning.

**Q3 (Short answer).** For a long-running chatbot, name three techniques to keep the messages list from blowing the context window.

**Q4 (Multiple choice).** For a personal assistant with hundreds of small user facts, the recommended storage is:
- (a) One giant JSON blob loaded every turn.
- (b) Structured markdown / vector store with relevance-based retrieval per turn.
- (c) Only session memory; forget between sessions.
- (d) A relational schema with 50 columns.

**Q5 (Short answer).** Describe fact extraction — what it is and when it runs.

**Q6 (Multiple choice).** When memory contains conflicting facts (user said A months ago; now says B), the typical default is:
- (a) Keep both; let the LLM decide.
- (b) Recency wins: newer facts override older ones; explicitly-updated facts override implicit past mentions.
- (c) Discard everything and start over.
- (d) Ask the user every turn.

**Q7 (Short answer).** Why is "forgetting" a feature, not a bug, of a good memory system?

**Q8 (Multiple choice).** Structured metadata per fact (source, first_seen, last_confirmed, confidence, sensitivity):
- (a) Waste of space.
- (b) Enables retrieval by recency/relevance, safe deletion, conflict resolution, and privacy controls.
- (c) Only for legal compliance.
- (d) Slows down memory.

**Q9 (Short answer).** In a multi-user product, how do you prevent memory from leaking across users?

**Q10 (Multiple choice).** Loading the user's profile into the system prompt at session start is:
- (a) Wrong — memory should never be in the system prompt.
- (b) A common pattern for small, stable, always-relevant memory; combine with on-demand retrieval for larger memory.
- (c) Impossible.
- (d) Only works with GPT-4.

---

### Answer Key & Detailed Explanations

**A1.** **Short-term / working memory** — the current conversation and reasoning trace held in the messages list this session. **Episodic memory** — records of specific past events across sessions ("on X date, user asked Y"). **Semantic memory** — distilled facts extracted from events ("user's spouse's name is X", "user prefers metric units").

**A2. (b).** LLMs themselves are stateless — nothing persists between calls. All "memory" is state you engineer around the model: storage, extraction, retrieval, and re-injection into the prompt each turn.

**A3.** (1) **Sliding window** — keep the last N turns. (2) **Summarization** — periodically compress older turns into a running summary. (3) **Hierarchical summaries** — recent full, medium-age paragraph, ancient one-line. (4) **Selective retention** — keep user turns and decisions; drop stale tool observations. (5) **External offload** — move detailed state to a DB; LLM sees compact references.

**A4. (b).** Structured markdown files (per-topic) or vector store with relevance-based retrieval per turn. Loading everything every turn (a) is expensive and dilutes attention. Forgetting each session (c) is amnesia. A 50-column schema (d) is over-designed for evolving personal facts.

**A5.** Fact extraction is the process of distilling durable facts from a conversation — using an LLM to read the conversation and output structured facts worth persisting. Runs periodically: at end of interaction, at scheduled intervals, or on trigger (user says "remember this"). Only writes facts that pass a "worth remembering" filter — durable, non-sensitive, non-transient.

**A6. (b).** Recency wins by default: newer facts override older ones. Explicit-update signals (user says "no, I meant X") override implicit past mentions. Store `last_confirmed` timestamp to enable this. Optionally ask user to confirm when confidence is low.

**A7.** Unbounded memory grows expensive (storage + retrieval cost + context bloat) and noisy (stale facts contradict fresh ones). Forgetting: (1) deletes explicitly-requested items; (2) time-decays stale info; (3) merges duplicates; (4) drops sensitive info per policy. Keeps memory small, relevant, high-signal.

**A8. (b).** Metadata enables: (1) **recency-based retrieval** and decay; (2) **safe deletion** by age or sensitivity; (3) **conflict resolution** (prefer higher-confidence or more recent); (4) **privacy compliance** (delete sensitive data per policy); (5) **provenance** (know where each fact came from). Well worth the small storage overhead.

**A9.** (1) **Per-user storage isolation** — namespace or user_id-scoped queries. (2) **Enforce user_id filter at retrieval layer** (DB-level RLS, or partitioned indexes) — never trust the prompt to enforce. (3) **Test cross-user queries** — send user A's query in user B's session; verify no data leak. (4) **Audit logs** — every memory read should log user_id and query.

**A10. (b).** Common pattern: at session start, load the user's stable profile (name, preferences, key facts) into the system prompt — always in view for the LLM. For larger or more dynamic memory, combine with retrieval tools the agent invokes on demand. Best of both.

---

## 6. Practice Prompts

1. **Fact extractor.** Build an LLM-based fact extractor: given a conversation, output a JSON list of `{topic, fact, confidence}`. Test on 10 sample chats.
2. **Semantic memory search.** Store extracted facts in pgvector. Build a `search_memory(query)` tool your agent can call. Test on 20 queries.
3. **Sliding window vs summarization.** Build a chatbot that runs both strategies on the same history; compare quality and cost per turn.
4. **Cross-user isolation.** Deliberately try to leak memory across users. Add RLS or namespace filters until it can't.
5. **Right to forget.** Implement "forget X" — user asks to forget a specific fact. Delete from storage; add to a "forbidden re-add" list. Test that future extraction respects it.

---

## 7. References

- OpenAI ChatGPT Memory (product feature, 2024).
- Anthropic memory features in Claude.ai (product feature, 2025+).
- Park et al., ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442) (2023) — memory in simulated agents.
- Packer et al., ["MemGPT: Towards LLMs as Operating Systems"](https://arxiv.org/abs/2310.08560) (2023).
- Mem0, Zep, LangMem — open-source memory frameworks.
- Cognitive-architecture papers on episodic vs semantic memory (SOAR, ACT-R) — for background.
