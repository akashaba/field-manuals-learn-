# 05.03 — Agent Memory & State

> **Module 5: Tools & Agents** · Subtopic 3 of 6
> **Prerequisites:** **02.02 §5 (required)** — memory taxonomy, retrieval scoring, write policies, compaction. Also 05.02 §6 (graphs, durable execution), 03.03 (databases, CDC), 02.01 §6 (prompt injection).
> **Outcome:** you can separate **execution state** (what this run is doing) from **memory** (what persists across runs), make agent state durable, replayable, and concurrency-safe with event sourcing and checkpoints, build procedural and episodic memory that measurably improves agents over time, and defend memory against poisoning and privacy failures.

> **Scope:** 02.02 covered *what goes into the context window* from memory (taxonomy, scoring, compaction). This subtopic covers the **storage and lifecycle engineering** behind agents: state models, durability, concurrency, learning from experience, consolidation, and security.

---

## 1. State vs Memory

```
┌──────────────────────── EXECUTION STATE (per run) ─────────────────────────┐
│ goal · plan/checklist · step pointer · messages · tool results (or handles) │  lifetime: one run
│ variables · budget counters · pending approvals · sub-agent handles         │  needs: durability,
└─────────────────────────────────────────────────────────────────────────────┘  replay, concurrency
                 │ distilled at run end (consolidation)                ▲ retrieved at run start
                 ▼                                                     │
┌──────────────────────── MEMORY (across runs) ───────────────────────────────┐
│ semantic: facts & preferences      procedural: skills, playbooks, prompts   │  lifetime: long
│ episodic: past runs, outcomes, reflections (lessons learned)                │  needs: provenance,
└─────────────────────────────────────────────────────────────────────────────┘  quality gates, TTL, deletion
```

**Why the separation matters:** state must be *exact* (resuming a run must not re-send an email), while memory must be *selective* (only verified, useful knowledge should persist). Mixing them produces agents that either forget what they were doing or remember garbage forever.

The **CoALA** framework (Sumers et al., 2023) formalises this: working memory, plus long-term episodic, semantic, and procedural memory, with explicit internal actions (retrieve, reason, learn) alongside external actions (tools).

---

## 2. Execution State Engineering

### 2.1 Typed state with reducers

Define the run state as a schema, and define **how each field merges updates** (a reducer). This is what makes parallel nodes and replays deterministic:

| Field | Reducer | Why |
|---|---|---|
| `messages` | append | History is ordered and additive |
| `plan` | replace | The latest plan wins |
| `findings` | set-union keyed by ID | Parallel workers add results without clobbering each other |
| `budget.tokens` | sum | Every node adds its spend |
| `approvals` | map-merge | Keyed by action hash (05.06) |

### 2.2 Event sourcing: the run is its event log

Store **immutable events** (`PlanCreated`, `ToolCalled`, `ToolReturned`, `ApprovalGranted`, …) and derive the state by folding reducers over them:

$$
S_t = \mathrm{fold}(R,\ S_0,\ [e_1, e_2, \ldots, e_t]) = R(\ldots R(R(S_0, e_1), e_2)\ldots, e_t)
$$

This gives you:
- **Crash recovery:** replay the log and continue from the last completed event.
- **Time travel / debugging:** reconstruct the state at any step.
- **Forking:** branch a run from step $t$ with a different decision (for evaluation, or a human correction).
- **Audit:** a complete, tamper-evident history of what the agent did (hash-chain the events).
- **Snapshots** every $N$ events bound the replay cost.

```python
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int
    type: str
    payload: dict
    prev_hash: str
    hash: str = ""


def _hash(run_id, seq, type_, payload, prev_hash) -> str:
    body = json.dumps([run_id, seq, type_, payload, prev_hash], sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


REDUCERS: dict[str, Callable[[dict, dict], dict]] = {
    "RunStarted":   lambda s, p: {**s, "goal": p["goal"], "messages": [], "findings": {}, "tokens": 0},
    "PlanUpdated":  lambda s, p: {**s, "plan": p["plan"]},
    "MessageAdded": lambda s, p: {**s, "messages": s["messages"] + [p["message"]]},
    "ToolReturned": lambda s, p: {**s, "findings": {**s["findings"], p["call_id"]: p["result"]}},
    "TokensUsed":   lambda s, p: {**s, "tokens": s["tokens"] + p["n"]},
    "RunFinished":  lambda s, p: {**s, "status": p["status"], "answer": p.get("answer")},
}


class ConcurrencyError(Exception):
    pass


class EventStore:
    """In-memory reference. Production: Postgres table with UNIQUE(run_id, seq) or an event store."""

    def __init__(self):
        self.events: dict[str, list[Event]] = {}
        self.snapshots: dict[str, tuple[int, dict]] = {}

    def append(self, run_id: str, type_: str, payload: dict, expected_seq: int) -> Event:
        log = self.events.setdefault(run_id, [])
        if len(log) != expected_seq:                      # optimistic concurrency control
            raise ConcurrencyError(f"expected seq {expected_seq}, log has {len(log)}")
        prev = log[-1].hash if log else "genesis"
        ev = Event(run_id, expected_seq, type_, payload, prev, _hash(run_id, expected_seq, type_, payload, prev))
        log.append(ev)
        return ev

    def verify_chain(self, run_id: str) -> bool:
        prev = "genesis"
        for e in self.events.get(run_id, []):
            if e.prev_hash != prev or e.hash != _hash(e.run_id, e.seq, e.type, e.payload, e.prev_hash):
                return False
            prev = e.hash
        return True

    def state(self, run_id: str, upto: int | None = None) -> dict:
        log = self.events.get(run_id, [])
        end = len(log) if upto is None else upto
        start, s = 0, {}
        snap = self.snapshots.get(run_id)
        if snap and snap[0] <= end:
            start, s = snap
        for e in log[start:end]:
            s = REDUCERS[e.type](s, e.payload)
        return s

    def snapshot(self, run_id: str) -> None:
        n = len(self.events.get(run_id, []))
        self.snapshots[run_id] = (n, self.state(run_id))

    def fork(self, run_id: str, at_seq: int, new_run_id: str) -> None:
        """Copy events [0, at_seq) into a new run (re-hashed under the new id)."""
        for i, e in enumerate(self.events[run_id][:at_seq]):
            self.append(new_run_id, e.type, e.payload, expected_seq=i)
```

### 2.3 Side effects and replay

Replaying the log must **not** re-execute side effects. Two rules:
1. **Record results, don't recompute them.** A `ToolReturned` event stores the output (or a handle to it), and replay reads it. This is exactly how durable execution engines (Temporal) treat activities.
2. **Idempotency keys** on every side-effecting call (05.01 §4.2). If a crash lands between "email sent" and "event written", the retry is deduplicated downstream.

### 2.4 Concurrency

Parallel sub-agents, human edits, and retries can write the same run concurrently:
- **Optimistic concurrency** (expected version / sequence number, as above). Losers re-read and retry.
- **Partition the state** so that parallel workers write disjoint keys (reducers with set-union).
- **Single-writer orchestrator** for shared decisions. Workers emit proposals, and the orchestrator commits them.

---

## 3. Memory That Makes Agents Better Over Time

### 3.1 Episodic memory → lessons (Reflexion, ExpeL)

After each run, store a **compact episode record**: the goal, the key steps, the outcome, and an evaluation signal. After a *failure* with a clear signal (a failed test, a rejected draft, a validation error), have the model write a **lesson**: *"When drafting amendments to Title 2, always check 2-18-101 definitions first — the drafting office rejected a draft for inconsistent terms."* At the start of a similar task, retrieve the relevant lessons.

**Quality gate:** keep lessons only if they are (a) tied to a verified outcome, (b) specific and actionable, and (c) not contradicted by later successes. Keep success and failure counts per lesson.

### 3.2 Procedural memory: skill libraries (Voyager, Agent Workflow Memory)

Store **reusable procedures** — a validated multi-step workflow, a code function, or a prompt template — as *skills*, with preconditions and usage statistics. Retrieve by similarity to the task, and **rank by estimated reliability** using a Beta posterior over the observed outcomes:

$$
\hat r_{\text{skill}} = \mathbb{E}[\theta \mid s, f] = \frac{s + \alpha}{s + f + \alpha + \beta},
\qquad
\mathrm{score} = \cos(\mathbf{e}_{\text{task}}, \mathbf{e}_{\text{skill}}) \cdot \hat r_{\text{skill}}
$$

Here $s$ and $f$ are the successes and failures, and $(\alpha, \beta)$ is a prior (e.g. $(1,1)$). New skills start with the prior's mean reliability. Skills that keep failing sink, and can be retired automatically.

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class Skill:
    name: str
    description: str
    procedure: str                    # steps, code, or a prompt template
    emb: np.ndarray
    successes: int = 0
    failures: int = 0
    retired: bool = False

    def reliability(self, a: float = 1.0, b: float = 1.0) -> float:
        return (self.successes + a) / (self.successes + self.failures + a + b)


class SkillLibrary:
    def __init__(self, retire_below: float = 0.3, min_trials: int = 5):
        self.skills: dict[str, Skill] = {}
        self.retire_below, self.min_trials = retire_below, min_trials

    def add(self, skill: Skill):
        self.skills[skill.name] = skill

    def retrieve(self, task_emb: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        q = task_emb / np.linalg.norm(task_emb)
        scored = [(s.name, float(q @ (s.emb / np.linalg.norm(s.emb))) * s.reliability())
                  for s in self.skills.values() if not s.retired]
        return sorted(scored, key=lambda t: -t[1])[:k]

    def record(self, name: str, success: bool):
        s = self.skills[name]
        if success:
            s.successes += 1
        else:
            s.failures += 1
        if s.successes + s.failures >= self.min_trials and s.reliability() < self.retire_below:
            s.retired = True
```

### 3.3 Semantic memory for agents

These are facts about users, entities, and the environment (02.02 §5.2 covers scoring). Agent-specific concerns:
- **Provenance:** which run and which source (user statement, tool result, web page).
- **Verification level:** verified against a system of record vs inferred.
- **Supersession:** keep history, but retrieve the current value.
- **TTL** for volatile facts.

### 3.4 Consolidation ("sleep-time" processing)

Don't do expensive memory work on the hot path. Run a **background consolidation job** (per user or per agent) that:
- Merges duplicate facts and resolves conflicts.
- Clusters episodes and distils recurring lessons into skills.
- Summarises long histories into profile documents.
- Expires stale items.

Recent work frames this as using idle "sleep-time" compute to pre-process context for future queries (Lin et al., 2025).

**Forgetting** is a feature. Decay rarely used memories with an Ebbinghaus-style retention curve, $R = e^{-t/S}$, where the strength $S$ grows with each successful retrieval and use (MemoryBank). Delete on request, completely (02.02 §5.2; privacy §4).

### 3.5 Memory as tools

Rather than injecting memories automatically, you can give the agent **memory tools** (`memory_search`, `memory_write`, file-based memory directories such as provider memory tools). The model decides what to read and write, as in MemGPT's paging between context and external storage. Combine this with the write gate in §4, because model-initiated writes are exactly where poisoning enters.

---

## 4. Memory Security and Privacy

**Memory poisoning** is persistent prompt injection. An attacker plants content (a web page, a document, a crafted user message) that makes the agent *store* an instruction or a false fact. Every future run then retrieves it (AgentPoison; Chen et al., 2024).

**Controls:**

| Control | Implementation |
|---|---|
| **Source-gated writes** | Only write memories derived from trusted sources (the authenticated user's own statements, verified tool results). Never from retrieved web or document content without verification |
| **Instruction detection** | Reject memory candidates that read as directives ("always…", "ignore…", "send…") when they come from non-user sources; a classifier plus heuristics |
| **Provenance + review** | Store the source and run ID; high-impact memories (procedures, policies) require human approval (05.06) |
| **Isolation** | Per-user and per-tenant namespaces; no cross-user retrieval |
| **Quarantine + expiry** | New memories start "probationary" (lower retrieval weight) until they are corroborated |
| **Right to erasure** | Delete from the primary store, the vector index, summaries/derived artifacts, and backups per policy; audit it |

```python
import re

DIRECTIVE = re.compile(r"\b(always|never|ignore|disregard|you must|from now on|send|forward|email|"
                       r"transfer|execute|run the)\b", re.I)
TRUSTED_SOURCES = {"user_statement", "verified_tool_result", "admin"}


def memory_write_gate(candidate: dict) -> tuple[bool, str]:
    """candidate: {'text', 'source', 'evidence' (optional), 'kind': fact|lesson|procedure}."""
    src, text, kind = candidate.get("source"), candidate.get("text", ""), candidate.get("kind", "fact")
    if src not in TRUSTED_SOURCES:
        if DIRECTIVE.search(text):
            return False, "directive-like content from untrusted source"
        if not candidate.get("evidence"):
            return False, "untrusted source without verifiable evidence"
    if kind == "procedure" and src != "admin":
        return False, "procedures require admin/human approval"
    if len(text) > 1000:
        return False, "too long; summarise before storing"
    return True, "accepted (probationary)" if src not in TRUSTED_SOURCES else "accepted"
```

---

## 5. Evaluating Memory

| Question | Evaluation |
|---|---|
| Does memory recall the right facts? | LongMemEval / LoCoMo-style QA over long multi-session histories: current-value, temporal, multi-hop, and abstention questions |
| Does it improve task performance? | The same task family over repeated episodes: success-rate curve with memory vs without (learning curve) |
| Is it safe? | Poisoning red-team: attack success rate of stored malicious instructions; cross-user leakage tests |
| Is state durable? | Crash-injection tests: resume correctness and no duplicate side effects |
| Cost | Memory tokens per turn, consolidation cost per user per day |

---

## 6. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Lost progress on crash** | Runs restart from scratch | Event sourcing / durable execution; snapshots |
| **Duplicate side effects on resume** | Emails sent twice | Record results in events; idempotency keys |
| **Race conditions** | Parallel workers overwrite findings | Reducers with keyed merges; optimistic concurrency; a single-writer orchestrator |
| **Memory bloat** | Retrieval quality degrades, costs rise | Consolidation, deduplication, decay, TTLs |
| **Stale or contradictory memory** | The agent acts on outdated facts | Supersession, verification against the system of record, provenance-weighted retrieval |
| **Memory poisoning** | Malicious behaviour persists across sessions | Source-gated writes, directive detection, probation, review for procedures |
| **Lessons that don't generalise** | Retrieved lessons mislead | Tie lessons to verified outcomes; track per-lesson success; retire bad ones |
| **Privacy and erasure** | Deleted data resurfaces through summaries | Deletion propagates to derived artifacts; audit queries |

---

## 7. Hands-On Projects

### Project 1 — Durable, Event-Sourced Agent Runtime with Time Travel

**User stories**
- *As an agent platform engineer*, I want every agent run to be crash-safe, replayable, and forkable, so that we can debug production incidents and resume long tasks exactly.

**Acceptance criteria**
1. A Postgres-backed event store (`UNIQUE(run_id, seq)`, hash chain) implementing `append` with optimistic concurrency, `state(upto)`, snapshots every 50 events, `fork`, and `verify_chain`.
2. The 05.02 agent loop emits typed events for every decision, model call, and tool call. Tool results are stored (or referenced), never recomputed on replay.
3. Crash injection: kill the process at random points across 100 runs. All resume correctly, and no side-effecting tool executes twice (verified by downstream logs and idempotency keys).
4. A debug UI (web or CLI): step through a run's state at any event, diff two runs, and fork from step t with an edited decision.
5. A concurrency test: 5 parallel workers append findings; there are no lost updates, and conflicts are retried.

**Step-by-step**
1. Create the events table (`run_id, seq, type, payload jsonb, prev_hash, hash, created_at`) and a snapshots table.
2. Port `EventStore` to Postgres using `INSERT … ` with a unique constraint, and treat the violation as a concurrency error.
3. Instrument the agent loop to emit events, and implement resume = replay + continue.
4. Build the crash-injection harness (a subprocess killed at random via signals).
5. Build the debug/time-travel tool, and document an incident replay.

---

### Project 2 — Skill-Learning Agent: Procedural Memory with Measured Gains

**User stories**
- *As a team running a drafting-assistance agent*, we want it to get better at recurring task types (e.g. "prepare a fiscal-note summary") by reusing procedures that worked, and to stop using ones that didn't.

**Acceptance criteria**
1. A task family with ≥ 5 recurring task types and an automatic success checker per task.
2. `SkillLibrary` (§3.2) with Beta-posterior ranking, automatic retirement, and skill creation from successful trajectories (the procedure is distilled by an LLM and validated by re-running it).
3. Lessons from failures (§3.1) are stored with the quality gate and success tracking.
4. Learning curve: success rate and cost per task over 10 episodes of each type, with skills and lessons vs without. Shows a statistically significant improvement (or reports honestly that there is none).
5. An ablation: skills only, lessons only, both; and the effect of retirement.

**Step-by-step**
1. Build the environment and the checkers (reusing the 05.02 Project 3 harness).
2. Implement skill extraction: after a success, prompt the model to write a reusable procedure with preconditions; test it on a held-out instance before adding it.
3. Implement retrieval at task start (inject the top-k skills and lessons into the context).
4. Run the episodes with multiple seeds and compute the curves with confidence intervals.
5. Analyse which skills helped, which got retired, and why.

---

### Project 3 — Memory Poisoning Red Team and Defences

**User stories**
- *As a security engineer*, I need to know whether attackers can make our assistant persistently misbehave through its memory, and to prove our defences work.

**Acceptance criteria**
1. The target: an assistant with semantic memory (auto-extraction from conversations and documents) and memory tools.
2. An attack suite of ≥ 50 scenarios: instructions in retrieved documents, false facts in web pages, crafted user messages in shared workspaces, and slow poisoning over multiple turns.
3. Metrics: poisoning success rate (a malicious memory is stored **and** later changes behaviour), false rejection rate on legitimate memories, and cross-user leakage (target: 0).
4. Defences are added incrementally — the source gate (`memory_write_gate`), directive detection with a classifier, probationary weighting, human review for procedures — with the metrics after each.
5. A threat model and a residual-risk write-up.

**Step-by-step**
1. Implement the memory pipeline without defences, and log all writes and retrievals.
2. Write the attack generators and the behavioural checks (does a future run follow the planted instruction?).
3. Measure the baseline; add the defences one by one; re-measure.
4. Test deletion completeness (the vector index, summaries, caches).
5. Write the report.

---

## 8. Foundational Papers & Reading (exact titles)

- Sumers et al., 2023 — *Cognitive Architectures for Language Agents*
- Park et al., 2023 — *Generative Agents: Interactive Simulacra of Human Behavior*
- Packer et al., 2023 — *MemGPT: Towards LLMs as Operating Systems*
- Shinn et al., 2023 — *Reflexion: Language Agents with Verbal Reinforcement Learning*
- Zhao et al., 2023 — *ExpeL: LLM Agents Are Experiential Learners*
- Wang et al., 2023 — *Voyager: An Open-Ended Embodied Agent with Large Language Models*
- Wang et al., 2024 — *Agent Workflow Memory*
- Zhong et al., 2023 — *MemoryBank: Enhancing Large Language Models with Long-Term Memory*
- Xu et al., 2025 — *A-MEM: Agentic Memory for LLM Agents*
- Chhikara et al., 2025 — *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*
- Lin et al., 2025 — *Sleep-time Compute: Beyond Inference Scaling at Test-time*
- Maharana et al., 2024 — *Evaluating Very Long-Term Conversational Memory of LLM Agents* (LoCoMo)
- Wu et al., 2024 — *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*
- Chen et al., 2024 — *AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases*
- Fowler — *Event Sourcing* (martinfowler.com); Temporal docs — *Workflow determinism and event history*

## 9. Essential Tooling

| Tool | Role |
|---|---|
| **Temporal / Restate** | Durable execution with recorded activity results |
| **LangGraph checkpointers** (Postgres, Redis) | Graph state persistence, time travel, forking |
| **PostgreSQL (+ pgvector)** | Event store, memory store, vector retrieval in one system |
| **EventStoreDB / Kafka** | Dedicated event logs at scale |
| **Letta (MemGPT), mem0, Zep/Graphiti** | Reference memory systems to study |
| **Provider memory tools** (e.g. Claude's client-side memory tool) | Model-directed memory read/write |
| **OpenTelemetry** | Linking state events to traces |
