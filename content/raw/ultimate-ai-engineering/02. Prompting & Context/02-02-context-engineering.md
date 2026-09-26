# 02.02 — Context Engineering

> **Module 2: Prompting & Context** · Subtopic 2 of 4
> **Prerequisites:** 01.04 (context limits, token budget, the basic context packer, long context vs RAG), 02.01 (prompt anatomy, injection defence), embeddings and retrieval basics.
> **Outcome:** you design the *system* that decides what the model sees on every call: retrieval assembly, tool context, memory, compaction, caching, and multi-agent isolation. You can measure and debug that system with the same rigour as any other data pipeline.

> **How this builds on 01.04:** 01.04 covered *why* the window is finite and how to fit a single request into it. This subtopic treats context as a **continuously running pipeline** across many turns, tools, and agents, where the main risks are not overflow but **distraction, contradiction, poisoning, and cost**.

---

## 1. Context Is a Compiled Artifact

The model is stateless. Every call it sees exactly one string: the **rendered context**. Everything the system "knows" — policies, history, retrieved facts, tool results, memories, plans — must be selected, transformed, ordered, and rendered into that string. Treat this as a compiler:

```
 SOURCES                    COMPILER PASSES                                   OUTPUT
 ─────────                  ─────────────────────────────────────────         ──────
 system policy  ─┐
 tool registry  ─┤   1. SELECT     which sources apply to this turn?
 memory store   ─┤   2. RETRIEVE   query each source (hybrid search, filters)
 doc index      ─┼─► 3. RANK/FUSE  RRF + rerank across sources            ─►  rendered
 conversation   ─┤   4. FILTER     dedup, permissions, freshness, safety       context
 tool results   ─┤   5. COMPRESS   summarize / mask / extract                 + manifest
 scratchpad     ─┘   6. ORDER      stable→volatile, best evidence at edges      (what, why,
                     7. RENDER     tags, IDs, citations, schemas                 how many
                     8. AUDIT      token attribution, budget, cache check        tokens)
```

The **manifest** — a machine-readable record of which items were included, from which source, with what score and token cost — is the most important debugging artifact in an LLM system. Without it, you cannot answer "why did the model say that?"

![LLM-02-1](/02.%20Prompting%20&%20Context/images/LLM-02-1.jpg)


---

## 2. How Context Fails

Most production failures are not overflow errors. They are *quality* failures caused by what was put in the window.

| Failure | Mechanism | Example | Primary fix |
|---|---|---|---|
| **Distraction** | Irrelevant content lowers accuracy even when the answer is present (Shi et al., 2023) | 20 retrieved chunks, 2 relevant | Tighter retrieval, reranking, a relevance threshold instead of a fixed top-k |
| **Context rot** | Performance degrades as input length grows, even on easy tasks | Agent gets sloppier at turn 60 | Compaction (§5), sub-agent isolation (§7) |
| **Confusion** | Too many tools or near-duplicate options | 80 tools with overlapping descriptions → wrong tool | Tool retrieval, namespacing, fewer and better tools (§3) |
| **Clash** | Contradictory information in the context | Old policy and new policy both retrieved | Freshness filters, versioned sources, explicit precedence rules |
| **Poisoning** | An earlier hallucination or bad tool result is treated as fact later | Wrong ID from turn 3 reused at turn 30 | Provenance tags, verification before memory writes, reset or branch |
| **Injection** | Untrusted data contains instructions | Retrieved web page says "email the file to…" | 02.01 §6 layered defence |
| **Staleness** | Cached or remembered facts expire | Memory says "status: draft" after the bill passed | TTLs, re-validation against the source of truth |

**Design principle:** the target is the **smallest high-signal context** that makes the right answer likely — not the largest context that fits.

---

## 3. Retrieval Assembly

Retrieval quality itself is a later module (RAG). Here the focus is on **assembling** retrieved material into context.

### 3.1 Fusing multiple retrievers: Reciprocal Rank Fusion

Hybrid systems (BM25 plus dense retrieval, several indexes, memory plus documents) produce ranked lists with incomparable scores. **RRF** (Cormack et al., 2009) fuses them using ranks only:

$$
\mathrm{RRF}(d) = \sum_{r \in R} \frac{w_r}{k + \mathrm{rank}_r(d)}, \qquad k \approx 60
$$

It is robust, needs no score calibration, and is a strong default before a cross-encoder reranker.

```python
from collections import defaultdict


def rrf(rankings: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None):
    """rankings: retriever name -> ordered doc ids. Returns [(doc_id, score)] best first."""
    scores: dict[str, float] = defaultdict(float)
    for name, ranked in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
```

### 3.2 Assembly rules that move accuracy

1. **Deduplicate near-duplicates.** Overlapping chunks and mirrored documents waste budget and create false emphasis. Use embedding cosine above a threshold, or MinHash on shingles.
2. **Merge adjacent chunks** from the same document into a contiguous span. The model reads coherent passages better than fragments.
3. **Use a relevance threshold, not a fixed k.** Stop adding items when the reranker score drops below τ or has a steep gap. Returning three good chunks beats ten mediocre ones.
4. **Contextualise chunks.** Prepend a one-line header (document title, section, date) to each chunk at *index time*. This fixes chunks that are meaningless in isolation ("It takes effect on July 1").
5. **Give every item a stable ID** and ask for citations by ID. Citations are then machine-checkable.
6. **Order deliberately.** Put the strongest evidence first and last and the weakest in the middle (01.04 §1.4). Group by document when cross-references matter.
7. **Attach metadata the model needs to resolve clashes:** `date`, `version`, `status`, `authority`. Add an explicit precedence rule: "If documents conflict, prefer the most recent enacted version."

### 3.3 Compressing retrieved text

| Technique | What it does | Trade-off |
|---|---|---|
| **Extractive** (select sentences by relevance) | Keeps verbatim text | Safest for citation; loses connective context |
| **Abstractive** (LLM summarises per chunk; RECOMP) | High compression | Adds a model call; can introduce errors — keep IDs |
| **Token pruning** (LLMLingua family) | Drops low-information tokens using a small LM's perplexity | 2–10× compression; can harm exact quoting |
| **Structured extraction** (pull fields into JSON; 02.03) | Maximal density | Only works when you know the fields in advance |

---

## 4. Tool Context

Tool definitions are **prompts**. Each tool's name, description, and parameter schema consumes tokens on every call and competes for the model's attention.

- **Descriptions are documentation for a new colleague.** State when to use the tool, when *not* to use it, the input formats, and what it returns. Include one example call for tricky tools.
- **Fewer, higher-level tools** (`update_bill_status(bill_id, status)`) outperform many thin API wrappers (`get_bill`, `patch_field`, `commit`). They reduce steps, errors, and context.
- **Tool retrieval.** With more than ~20–30 tools, retrieve the relevant subset per turn (embed the tool descriptions, as in §3.1). Namespacing (`legis.search`, `legis.amend`) helps both retrieval and model disambiguation.
- **Shape tool results for the model, not for the API.** Return only the fields needed, paginate, truncate with a **handle** (`result_id=r42; 1,832 more rows; call fetch_rows(r42, offset)`), and turn raw errors into actionable messages ("`bill_id` must look like `HB-123`; you passed `123`").
- **Stable tool ordering.** Tool definitions sit at the very start of the cache hierarchy. Reordering or editing tools invalidates *everything* cached after them (§6).
- **MCP (Model Context Protocol)** standardises how tools, resources, and prompts are exposed to models. Treat every MCP server as untrusted input (02.01 §6): its descriptions and results enter your context.

---

## 5. Memory and Compaction

### 5.1 Memory taxonomy

```
┌────────────────────┬─────────────────────────────┬──────────────────────────────────┐
│ Type               │ Holds                       │ Typical implementation           │
├────────────────────┼─────────────────────────────┼──────────────────────────────────┤
│ Working            │ Current task state          │ The context window itself        │
│ Episodic           │ What happened (sessions)    │ Summaries + event log, by time   │
│ Semantic           │ Durable facts, preferences  │ Fact store (KV/SQL + embeddings) │
│ Procedural         │ How to do things            │ Instructions, skills, playbooks  │
│ Scratchpad         │ Agent's own notes/plans     │ A file or structured note tool   │
└────────────────────┴─────────────────────────────┴──────────────────────────────────┘
```

### 5.2 Retrieval scoring for memories

Generative Agents (Park et al., 2023) score each memory $m$ for query $q$ at time $t$:

$$
\mathrm{score}(m) = \alpha\cdot \underbrace{\gamma^{\,(t - t_m)/\Delta}}_{\text{recency}} + \beta\cdot \underbrace{\mathrm{imp}(m)}_{\text{importance}} + \delta\cdot \underbrace{\cos(e_q, e_m)}_{\text{relevance}}
$$

Each term is min-max normalised before weighting. Here $\gamma \in (0,1)$ is a decay per interval $\Delta$, and importance is assigned at write time (by a model or by rules).

```python
import math
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Memory:
    text: str
    emb: np.ndarray
    importance: float              # 0..1, assigned at write time
    created: float = field(default_factory=time.time)
    source: str = "conversation"   # provenance
    supersedes: str | None = None  # id of a fact this replaces


def score_memories(q_emb, memories, now=None, alpha=1.0, beta=1.0, delta=1.0,
                   decay=0.995, interval_s=3600.0):
    now = now or time.time()

    def minmax(x):
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.ones_like(x)

    q = q_emb / np.linalg.norm(q_emb)
    rec = [decay ** ((now - m.created) / interval_s) for m in memories]
    imp = [m.importance for m in memories]
    rel = [float(m.emb @ q / np.linalg.norm(m.emb)) for m in memories]
    total = alpha * minmax(rec) + beta * minmax(imp) + delta * minmax(rel)
    return sorted(zip(total.tolist(), memories), key=lambda t: -t[0])
```

**Memory write policy matters more than read policy.**
- Write only **verified or user-stated** facts, with provenance.
- On conflict, **supersede** rather than append (keep history, but retrieve only the latest).
- Put TTLs on volatile facts.
- Never store instructions extracted from untrusted content (that is persistent prompt injection).

### 5.3 Compaction strategies for long-running agents

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Observation masking** | Replace old tool *outputs* with placeholders (`[output of search_docs elided; 3,210 tokens; call id c17]`), and keep the reasoning and actions | Cheap, deterministic, cache-friendly if done in large steps | Loses details that may be needed later (the model can re-call the tool) |
| **LLM summarisation** | Summarise older turns into a structured state (goals, decisions, open items, key IDs) | High compression | Extra call; summary errors become poisoning; breaks the cache |
| **Structured note-taking** | The agent writes a `NOTES.md` or `todo` via a tool and re-reads it | Durable across resets; inspectable | Requires model discipline and prompting |
| **Sub-agent isolation** | Delegate a sub-task to a fresh context; only a condensed result returns | Keeps the main context clean; parallelisable | Coordination overhead; information lost at the boundary |
| **Reset with hand-off** | Start a fresh context from notes + summary + open files | Resets context rot completely | Needs a high-quality hand-off artifact |

Empirical work on software-engineering agents found that simple observation masking can match LLM summarisation on solve rate while cutting cost (Lindenbauer et al., 2025). **Start with masking, measure, then add summarisation only where it helps.**

```python
def mask_observations(messages: list[dict], keep_last: int = 3, min_tokens: int = 200,
                      count=lambda s: len(s) // 4) -> list[dict]:
    """Elide all but the most recent `keep_last` tool results.

    Keeps assistant reasoning and tool calls intact so the trajectory stays coherent.
    Apply in large steps (e.g. when the context passes 70% of budget) to preserve cache prefixes.
    """
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    to_mask = set(tool_idx[:-keep_last]) if keep_last else set(tool_idx)
    out = []
    for i, m in enumerate(messages):
        if i in to_mask and count(m["content"]) >= min_tokens:
            out.append({**m, "content": (
                f"[elided tool output; ~{count(m['content'])} tokens; "
                f"call_id={m.get('tool_call_id', '?')}. Re-run the tool if you need it.]")})
        else:
            out.append(m)
    return out
```

---

## 6. Prompt Caching as a Design Constraint

Caching makes context engineering an **economic** problem. Providers cache exact prefixes. Anthropic uses explicit `cache_control` breakpoints (or top-level automatic caching), with a hierarchy of **tools → system → messages**. OpenAI caches long prefixes automatically. Self-hosted engines use prefix or radix caching (01.02 §3.3).

### 6.1 Break-even arithmetic

Let the prefix be $P$ tokens, the uncached suffix $U$ tokens, the base input price $c$ per token, the write multiplier $w$, the read multiplier $r$, and the hit rate $h$:

$$
\mathbb{E}[\text{cost}] = c\,\big[\,(h\,r + (1-h)\,w)\,P + U\,\big]
\qquad\text{vs.}\qquad c\,(P + U)
$$

Caching pays off when $h\,r + (1-h)\,w < 1$, i.e.

$$
h > h^* = \frac{w - 1}{w - r}
$$

With Anthropic's published multipliers at the time of writing — 5-minute writes $w = 1.25$, reads $r = 0.1$ — $h^* \approx 0.22$. For 1-hour writes ($w = 2$), $h^* \approx 0.53$. Always check current pricing, because read multipliers vary by model. Latency gains (lower TTFT) come on top of the cost gains.

```python
def cache_breakeven(write_mult: float, read_mult: float) -> float:
    return (write_mult - 1) / (write_mult - read_mult)


def expected_cost(prefix_toks, suffix_toks, price_per_tok, hit_rate, write_mult, read_mult):
    eff = hit_rate * read_mult + (1 - hit_rate) * write_mult
    return price_per_tok * (eff * prefix_toks + suffix_toks)
```

### 6.2 Cache-aware context layout

```
[tools (sorted, frozen)] [system policy] [static docs / many-shot examples] ◄─ breakpoint 1
[conversation summary (append-only; changes rarely)]                        ◄─ breakpoint 2
[recent turns ................................................ ]            ◄─ breakpoint 3 (rolling)
[volatile: retrieved chunks for THIS turn, timestamp, user message]         (never cached)
```

**Cache killers:** timestamps or request IDs in the system prompt; non-deterministic JSON key order in tool schemas; tool lists that differ per user; editing an early message during compaction; toggling features that change the system block. Monitor `cache_read_input_tokens / total_input_tokens` as a first-class SLO.

---

## 7. Multi-Agent Context Isolation

Multi-agent systems are primarily a **context-management** technique. Each sub-agent gets a clean, task-specific window, explores widely, and returns a compressed result.

```
                    ┌──────────────────────────┐
                    │ ORCHESTRATOR              │  context: goal, plan, condensed results
                    │ (small, clean window)     │
                    └───┬─────────┬─────────┬──┘
          task spec +   │         │         │   ≤ 1–2k-token result + artifacts by reference
          budget        ▼         ▼         ▼
                 ┌─────────┐┌─────────┐┌─────────┐
                 │ worker A││ worker B││ worker C│   each: fresh context, own tools,
                 │ 60k tok ││ 45k tok ││ 80k tok │   explores deeply, then discards
                 └─────────┘└─────────┘└─────────┘
                         shared artifact store (files / DB) ◄── full outputs live here
```

**Hand-off contract:** the orchestrator sends an objective, constraints, an output schema, and a budget. The worker returns the result, evidence references, confidence, and open questions. Large outputs go to a shared store, and only references go back into the orchestrator's context.

**Costs:** total tokens multiply (several times a single agent), and coordination errors appear at the boundaries. Use multiple agents when sub-tasks are **parallel and separable** (research breadth, independent files). Avoid them when sub-tasks share a lot of state (tightly coupled edits).

---

## 8. Reference: A Context Compiler with a Manifest

```python
from dataclasses import dataclass, field
from html import escape
from typing import Callable


@dataclass
class Item:
    id: str
    source: str               # e.g. "docs", "memory", "tool:search"
    text: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)
    tokens: int = 0


@dataclass
class SourceBudget:
    max_tokens: int
    min_score: float = float("-inf")


class ContextCompiler:
    def __init__(self, count_tokens: Callable[[str], int], budgets: dict[str, SourceBudget],
                 dedup_threshold: float = 0.85):
        self.count, self.budgets, self.dedup_threshold = count_tokens, budgets, dedup_threshold

    @staticmethod
    def _shingles(text: str, n: int = 5) -> set[str]:
        words = text.lower().split()
        return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}

    def _is_dup(self, item: Item, kept: list[Item]) -> bool:
        s = self._shingles(item.text)
        for k in kept:
            t = self._shingles(k.text)
            if s and t and len(s & t) / len(s | t) >= self.dedup_threshold:
                return True
        return False

    def compile(self, items: list[Item]) -> tuple[str, list[dict]]:
        manifest, kept = [], []
        by_source: dict[str, list[Item]] = {}
        for it in items:
            by_source.setdefault(it.source, []).append(it)

        for source, group in by_source.items():
            budget = self.budgets.get(source, SourceBudget(0))
            used = 0
            for it in sorted(group, key=lambda x: -x.score):
                it.tokens = it.tokens or self.count(it.text)
                reason = None
                if it.score < budget.min_score:
                    reason = "below_threshold"
                elif used + it.tokens > budget.max_tokens:
                    reason = "over_budget"
                elif self._is_dup(it, kept):
                    reason = "duplicate"
                if reason is None:
                    kept.append(it)
                    used += it.tokens
                manifest.append({"id": it.id, "source": source, "score": round(it.score, 4),
                                 "tokens": it.tokens, "included": reason is None,
                                 "reason": reason or "ok"})

        # Order: strongest first, weakest in the middle, second-strongest last.
        ranked = sorted(kept, key=lambda x: -x.score)
        ordered = ranked[0::2] + ranked[1::2][::-1]

        blocks = []
        for it in ordered:
            attrs = " ".join(f'{k}="{escape(str(v))}"' for k, v in
                             {"id": it.id, "source": it.source, **it.meta}.items())
            blocks.append(f"<item {attrs}>\n{escape(it.text)}\n</item>")
        return "<context>\n" + "\n".join(blocks) + "\n</context>", manifest
```

Log the manifest with every request. Aggregate it into dashboards showing tokens per source, the inclusion rate, and the duplicate rate. The dashboard is also where you run **context ablations**: remove one source and re-run the eval set.

---

## 9. Observability for Context

- **Log the rendered context** (redacted) plus the manifest for sampled requests. Store it with the prompt version, model version, and cache statistics.
- **Token attribution per source** over time. A slowly growing "tool results" share is an early warning of cost explosion and context rot.
- **Context diffs between turns.** Unexpected changes early in the prompt explain cache-hit drops.
- **Replay:** take a failed production request, recompile its context with the current pipeline, and diff the results. This is how you debug regressions in retrieval or memory.
- **Groundedness checks:** the percentage of cited IDs that exist in the manifest, and the percentage of answer sentences supported by cited items (an NLI or LLM judge).

---

## 10. Production Challenges & How to Solve Them

| Challenge | Solution |
|---|---|
| **Answers degrade in long sessions** | Observation masking at a threshold; structured notes; reset with hand-off; sub-agents for exploratory sub-tasks |
| **Wrong tool chosen** | Fewer, higher-level tools; clearer descriptions with "when not to use"; tool retrieval; eval tool choice as its own metric |
| **Contradictory sources** | Version and status metadata; precedence rules in the system prompt; freshness filters at retrieval |
| **Memory returns stale or wrong facts** | Supersession, TTLs, provenance; re-validate against the system of record before acting |
| **Token cost creeping up** | Per-source budgets; result handles instead of payloads; cache-hit SLO; alerts on tokens-per-task |
| **Cache hit rate collapses after a deploy** | Diff rendered prefixes between versions; freeze tool ordering; move volatile fields after breakpoints |
| **Can't explain an answer** | Manifest + citations by ID + replay tooling |
| **Multi-agent runs cost several times more** | Use only for parallel, separable work; cap worker budgets; return references, not payloads |

---

## 11. Hands-On Projects

### Project 1 — Context Compiler for a Document Assistant, with Ablations

**User stories**
- *As an AI engineer*, I want every model call's context to be built by one inspectable pipeline with a manifest, so that I can explain and debug any answer.
- *As a product owner*, I want to know which context sources actually improve answer quality per token, so that we stop paying for useless context.

**Acceptance criteria**
1. Sources: a document index (BM25 + dense, fused with RRF), a conversation summary, a user-preferences memory, and a tool-result store. Each has a token budget and a score threshold.
2. The compiler implements dedup, adjacent-chunk merging, contextual headers, edge ordering, XML rendering with IDs, and a JSON manifest (§8).
3. An eval set of ≥ 150 questions with gold answers and gold evidence IDs.
4. **Ablations:** full context vs removing each source vs fixed top-k without a threshold vs no dedup. Reports answer accuracy, citation precision/recall, input tokens, and cost.
5. A dashboard (Grafana, or a notebook) of tokens per source and inclusion rates over the eval run.
6. Final config: accuracy within 1 point of the best configuration, at ≤ 70% of its tokens.

**Step-by-step**
1. Index a document corpus (e.g. statutes or technical docs). Chunk by section and add contextual headers at index time.
2. Implement BM25 (`rank_bm25` or OpenSearch) and dense retrieval (FAISS), fused with `rrf` (§3.1), plus an optional cross-encoder reranker.
3. Implement `ContextCompiler` (§8), with an extension that merges adjacent chunks from the same document.
4. Build the answer prompt with citation rules (02.01 §1.2) and structured output that includes `cited_ids` (02.03).
5. Write the eval runner and the citation metrics: cited IDs ∩ gold IDs.
6. Run the ablation grid, plot accuracy vs tokens, and write the recommendation.

---

### Project 2 — Long-Term Memory Service for an Assistant

**User stories**
- *As a returning user*, I want the assistant to remember my stated preferences and ongoing projects accurately, and to update them when they change, so that I don't repeat myself.
- *As a privacy officer*, I want memory writes to be provenance-tracked, deletable, and never populated from untrusted documents.

**Acceptance criteria**
1. A memory service (FastAPI or Spring Boot) with `write`, `search`, `supersede`, `forget`, and `export` endpoints. Each memory has provenance, a timestamp, importance, a TTL, and a supersession link.
2. The write pipeline extracts candidate facts from *user* turns only, deduplicates against existing memories, and resolves conflicts by supersession (with an LLM judge for "same fact, new value?").
3. Retrieval uses the §5.2 scoring with tunable weights and a token budget for injected memories.
4. **Eval:** a synthetic multi-session benchmark (≥ 50 users × 10 sessions) with preference changes, contradictions, and temporal questions ("what was my deadline before it moved?"). Reports accuracy on current-value questions, historical questions, and abstention when a fact is unknown.
5. **Safety tests:** instructions planted in retrieved documents never end up in memory. `forget` removes a fact from all retrieval paths within one request.

**Step-by-step**
1. Design the schema (Postgres + pgvector, or SQLite + FAISS) with `id, user_id, text, embedding, importance, created_at, ttl, source, supersedes, deleted_at`.
2. Build the extraction prompt with a strict output schema: `{facts: [{text, category, importance, is_update_of?}]}`.
3. Conflict resolution: retrieve the top similar memories, ask the judge "same attribute?", and on yes, mark the old one as superseded.
4. Implement `score_memories` (§5.2) and the context injection block with provenance tags.
5. Generate the synthetic benchmark: a scripted user persona with timeline events. Evaluate with exact match plus an LLM judge.
6. Add a privacy test suite and an audit log.

---

### Project 3 — Compaction Strategy Benchmark for a Tool-Using Agent

**User stories**
- *As an agent developer*, I want evidence for which compaction strategy gives the best task success per dollar on long tasks, so that our agents stay reliable past 50+ steps.

**Acceptance criteria**
1. An agent with file-system and search tools solves multi-step tasks (e.g. a set of repository-maintenance tasks, or a research-and-summarise task set) of ≥ 40 tasks, each requiring ≥ 25 tool calls.
2. Strategies compared: no compaction (until failure), observation masking (§5.3), LLM summarisation at 70% of the budget, structured notes (a `NOTES.md` tool), and orchestrator + sub-agents.
3. Metrics per strategy: success rate, total tokens, cost (using real cache pricing and `cache_read` statistics), wall-clock time, and the number of times the agent re-fetched elided information.
4. Each configuration runs ≥ 3 seeds. Differences are reported with confidence intervals.
5. The write-up includes one failure-case analysis per strategy (e.g. a summarisation-induced poisoning example).

**Step-by-step**
1. Build a minimal agent loop with message-list state and the tools. Log every step.
2. Implement the strategies as pluggable `compact(messages) -> messages` functions, triggered by token thresholds.
3. Set up cache breakpoints per §6.2 and record the provider usage fields.
4. Build the task set with automatic success checkers (tests pass, file contains the expected facts).
5. Run the matrix, compute the statistics, and plot cost vs success.
6. Write the recommendation: default strategy, thresholds, and when to escalate to sub-agents.

---

## 12. Foundational Papers & Essential Reading (exact titles)

**Retrieval assembly and compression**
- Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Cormack, Clarke & Büttcher, 2009 — *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*
- Shi et al., 2023 — *Large Language Models Can Be Easily Distracted by Irrelevant Context*
- Liu et al., 2023 — *Lost in the Middle: How Language Models Use Long Contexts*
- Xu, Shi & Choi, 2023 — *RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation*
- Jiang et al., 2023 — *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*
- Jiang et al., 2023 — *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression*

**Memory and agents**
- Park et al., 2023 — *Generative Agents: Interactive Simulacra of Human Behavior*
- Packer et al., 2023 — *MemGPT: Towards LLMs as Operating Systems*
- Shinn et al., 2023 — *Reflexion: Language Agents with Verbal Reinforcement Learning*
- Wu et al., 2024 — *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*
- Lindenbauer et al., 2025 — *The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management*

**Tools**
- Schick et al., 2023 — *Toolformer: Language Models Can Teach Themselves to Use Tools*
- Qin et al., 2023 — *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*
- Patil et al., 2023 — *Gorilla: Large Language Model Connected with Massive APIs*

**Engineering write-ups (not papers, but essential)**
- Anthropic Engineering — *Effective context engineering for AI agents*; *Introducing Contextual Retrieval*; *Writing effective tools for agents*; *How we built our multi-agent research system*
- Anthropic docs — *Prompt caching*; Model Context Protocol specification

## 13. Essential Tooling

| Tool | Role |
|---|---|
| **Provider prompt caching** (Anthropic `cache_control`, OpenAI automatic caching) and usage fields | Cost and latency of stable prefixes |
| **FAISS / pgvector / OpenSearch / Qdrant** | Retrieval for documents, tools, and memories |
| **rank_bm25 / Tantivy / OpenSearch** | Lexical retrieval for hybrid fusion |
| **Cross-encoder rerankers** (sentence-transformers `CrossEncoder`, hosted rerank APIs) | Precision before assembly |
| **LLMLingua** | Prompt compression |
| **Model Context Protocol SDKs** | Standardised tools and resources |
| **Langfuse / Arize Phoenix / OpenTelemetry GenAI conventions** | Tracing rendered contexts, token attribution, replay |
| **Letta (MemGPT), mem0** | Reference memory systems to study (build your own core first) |
