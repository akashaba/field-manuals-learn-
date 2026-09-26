# Multi-Agent Systems — Master Study Guide

> **Track:** Agents + Production · **Module:** 07
> **Prerequisites:** Modules 01–06.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Some tasks benefit from **multiple specialized agents** working together. A research assistant might delegate: one agent searches, another summarizes, a third writes citations. A coding system might have a planner agent, an implementer agent, and a critic agent. These architectures — **multi-agent systems** — are among the most-discussed patterns of the modern agent era.

But they're also **often unnecessary**. Anthropic's guidance: "start with the smallest thing that works." A single well-designed agent with good tools often beats a poorly-orchestrated multi-agent system. Multi-agent is powerful *when* the roles are genuinely distinct, the coordination cost is manageable, and evaluation shows a win.

**Fundamental principles you must own:**

1. **Multi-agent adds coordination cost.** Agents must communicate — via messages, shared state, or handoffs.
2. **Specialization can pay off** — but specialization by prompt/tools alone (single agent, phase-shifted) often works as well.
3. **Topologies matter** — hierarchical (supervisor + workers), peer-to-peer, graph, actor.
4. **Handoffs** are the atomic unit of multi-agent coordination.
5. **Shared state is a design decision** — everyone reads/writes vs message passing.
6. **Evaluation is harder** — more traces, more failure modes, more variance.

If you retain nothing else: **multi-agent is a powerful tool. Use it when the roles are genuinely distinct. Not by default.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Why Multi-Agent (Sometimes)

Genuine advantages when roles are distinct:

**1. Specialization.** Each agent has:
- A focused system prompt for one role.
- A narrower tool set (5 tools instead of 30).
- Optionally a specialized model (a coding-tuned model for coding, a smaller model for triage).

**2. Concurrency.** Independent agents can run in parallel — one searches while another analyzes.

**3. Cleaner interfaces.** Each agent has a clear input/output contract, making debugging and eval per-agent tractable.

**4. Reusability.** Well-designed agents become reusable components across systems.

**5. Complexity management.** A task too complex for one agent's context becomes tractable when split.

**Common examples:**

- **Deep Research** — planner + N parallel researcher agents + synthesizer. Each researcher works on a sub-topic; synthesizer combines.
- **Coding agents** — planner + implementer + reviewer + tester. Cursor Composer, Devin, Claude Code all have some multi-agent-like decomposition, often invisible to the user.
- **Marketing content** — writer + editor + fact-checker + SEO optimizer.
- **Software engineering pipelines** — issue triager + implementer + PR reviewer.

**Common example that shouldn't be multi-agent:**

- "Send an email, then check the inbox, then reply" — three tools in one agent, not three agents.
- "Route + answer" — a routing workflow, not multi-agent.
- "Retrieve + generate" — RAG, not multi-agent.

Rule: multi-agent when each role has enough scope to justify its own agent + tool set. Otherwise, just add tools to one agent.

---

### 2.2 Topologies

Different architectures for different coordination needs.

**Hierarchical / Supervisor pattern.** One agent supervises; others do the work.

```
              ┌─── Supervisor ───┐
              │                  │
        Worker A              Worker B
        (search)              (write)
```

- Supervisor decides who works on what.
- Workers report back; supervisor synthesizes.
- Clear control flow; easy to debug.
- Most common pattern; works for most cases.

**Peer-to-peer / Chat.** Agents communicate directly with each other, taking turns.

```
   Agent A  ⇄  Agent B  ⇄  Agent C
```

- More conversation-like; each agent can address any other.
- Higher flexibility, higher variance.
- Used in Microsoft AutoGen conversational patterns.

**Graph / DAG.** Explicit flow graph with typed edges (LangGraph, etc.).

```
   Input → [Planner] → [Fetcher] → [Analyzer] → [Reporter] → Output
                            ↓
                       [Cache Check]
```

- Deterministic control flow with agents as nodes.
- Conditional edges enable branching.
- Structured state passed along edges.
- Best of both worlds when the flow is mostly known.

**Blackboard / Shared state.** All agents read and write to a shared state store.

- Enables coordination without direct messaging.
- Risk: race conditions, stale reads.
- Powerful when agents genuinely need shared world model.

**Actor / message passing.** Each agent is an actor with an inbox; agents send typed messages.

- Ala Erlang / Akka.
- Scales; formal semantics.
- Overhead for simple cases.

**Which to use:**
- **Default: hierarchical.** Simple, effective, debuggable.
- **Predefined flow, some autonomy: graph** (LangGraph).
- **Long open-ended interaction: peer-to-peer** (AutoGen).
- **Coordinating over shared model: blackboard**.

---

### 2.3 Handoffs and Delegation

The atomic operation of multi-agent systems: **one agent decides to hand off to another**.

**Handoff patterns:**

**1. Supervisor dispatches.** Supervisor produces a message like "worker B: research X" and the framework routes it to worker B.

```python
class Supervisor:
    def act(self, state):
        if state.needs_research:
            return HandoffTo("researcher", task=state.query)
        if state.needs_writing:
            return HandoffTo("writer", material=state.results)
        return FinalAnswer(state.results)
```

**2. Direct handoff.** An agent decides itself: "I'm done here; hand this to X." Common in peer-to-peer.

**3. Return handoff.** Worker completes → returns control to supervisor with results.

**4. Escalation handoff.** Worker hits a blocker → hands off to a more capable agent (or human).

**Data passing on handoff:**

- **Full context** — pass the whole conversation. Simple; expensive.
- **Distilled brief** — supervisor writes a scoped prompt for the worker. Cleaner; loses nuance.
- **Structured payload** — JSON with task, inputs, expected output schema. Best for typed pipelines.

**OpenAI's Swarm framework** made handoffs the primitive: agents "return" other agents to yield control. LangGraph does it via graph edges. AutoGen via messages.

**Handoff pitfalls:**

- **Ping-pong** — A hands off to B; B hands back to A; A back to B... Cap handoffs.
- **Broken telephone** — each handoff loses context. Structured payloads help.
- **Wrong recipient** — supervisor picks the wrong worker. Better tool schemas / worker descriptions.

---

### 2.4 Communication and Shared State

Multi-agent systems must coordinate. Options:

**Message passing.** Agents send typed messages via a channel. Common in actor models.

Pros: explicit; testable; no shared-state bugs.
Cons: verbose; complex flows can be hard to trace.

**Shared state store.** All agents read/write a shared object.

Pros: agents can see each other's work without explicit messages.
Cons: race conditions; state pollution; hard to reason about.

**Event bus / pub-sub.** Agents publish events; others subscribe.

Pros: decoupling; scalable.
Cons: order not guaranteed; harder to debug.

**LangGraph state pattern.** Typed state dict; each node reads/writes named fields.

```python
class ResearchState(TypedDict):
    query: str
    plan: list[str]
    results: list[dict]
    draft: str
    final: str
```

Each agent-node modifies specific fields. Clear contract; typed.

**Locking and race conditions.** In parallel multi-agent, concurrent writes to shared state cause bugs. Options:
- Serialize writes through a coordinator.
- Use CRDTs or lock manager.
- Design state as append-only (each agent adds; nothing overwrites).

**Rule:** start with message passing or explicit graph state. Shared mutable state creates hard-to-debug systems.

---

### 2.5 When NOT to Use Multi-Agent

Common anti-patterns:

**1. "Two prompts, one agent" masquerading as multi-agent.** If your "second agent" is really the same LLM with a different prompt and no separate tool set, it's a workflow step. Save yourself the framework overhead.

**2. Splitting for artificial specialization.** If "researcher" and "writer" agents share 90% of their tools, you likely don't need separation. Give one agent both prompts (chain-of-thought or role-swap prompts).

**3. Debugging nightmare from too many agents.** 6-agent systems produce logs where every step has 6 possible actors. Traceability collapses. Start with 2 agents (supervisor + worker); grow only when a win emerges.

**4. Overhead > benefit.** Every agent adds LLM calls. If a single agent with good tools does the job in 5 calls, and a 3-agent system does it in 15, the multi-agent version costs 3× with unclear quality benefit.

**5. Emergent chaos.** Peer-to-peer agents chatting without clear termination can enter conversational loops. Cap turns; keep control flow explicit.

**When multi-agent likely IS the right call:**

- The task has natural distinct roles that a single prompt can't cleanly capture.
- Each role has a **distinct tool set** (~5 tools not shared).
- Parallelism is available and matters.
- Evaluation shows a measurable improvement over single-agent baseline.

Test the single-agent version first. If it works, ship it.

---

## 3. Mental Models & Analogies

### 3.1 The "Small Startup vs Big Corporation" Model

A **single agent with many tools** is like a small startup: everyone does everything, coordination is cheap, decisions happen fast. Works well up to a certain complexity — then it fragments.

A **multi-agent system** is like a corporation: departments (roles), managers (supervisor), org-chart (topology), meetings (handoffs). Enables scale but adds coordination overhead. If you're a small startup, don't put on a corporate structure early — the overhead crushes speed.

Ask: is this task big enough to warrant the corporate structure? Or can two smart people do it?

### 3.2 The "Theater Ensemble" Model

Peer-to-peer multi-agent is a **theater troupe** improvising. Each actor has a role; they take turns; they respond to each other. Great when the collaboration itself produces value (debate, brainstorming, adversarial critique).

Hierarchical multi-agent is a **film production**: director dispatches specialists (cinematographer, editor, sound designer); each does their part; director assembles the film. Higher control, higher quality on structured outputs.

Graph-based multi-agent is a **factory line**: each station has a role; work flows in a defined direction; the graph is the process specification.

Choose the metaphor that matches your task's shape.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "More Agents = Better"

Empirically no. Each extra agent adds LLM calls, coordination logic, failure modes, and evaluation complexity. Start with 1 agent. Add a second when it clearly helps. Rarely justify more than 4–5 in a system.

### 4.2 "Multi-Agent Systems Emerge Interesting Behaviors"

Sometimes, but you're also emerging failure modes — infinite chats, agents agreeing prematurely, one dominant agent monopolizing decisions. Emergence without evaluation is a research demo, not production. Ship what works measurably.

### 4.3 "Frameworks Handle Multi-Agent Complexity for Me"

Frameworks (LangGraph, AutoGen, CrewAI, Swarm) provide scaffolding. They don't magically fix bad decomposition, ambiguous handoffs, or poor eval. Understand the pattern; then choose a framework that implements it well.

---

## 5. Self-Assessment Bank (Multi-Agent Systems)

### Questions

**Q1 (Short answer).** Give two situations where multi-agent likely helps, and one where it likely doesn't.

**Q2 (Multiple choice).** The most common (and safest) multi-agent topology is:
- (a) Peer-to-peer chat.
- (b) Hierarchical (supervisor + workers).
- (c) Fully connected mesh.
- (d) Blackboard.

**Q3 (Short answer).** Describe a "handoff" in multi-agent systems and give one design pattern for what data to pass.

**Q4 (Multiple choice).** For a task where you want parallel independent processing:
- (a) Peer-to-peer chat.
- (b) Supervisor dispatches to N parallel workers; synthesizes results.
- (c) Sequential single agent.
- (d) Blackboard.

**Q5 (Short answer).** What is the "ping-pong" anti-pattern in multi-agent, and how do you defend against it?

**Q6 (Multiple choice).** Shared mutable state across agents is:
- (a) Recommended by default.
- (b) A common source of race conditions and hard-to-debug issues; use message passing or graph state instead.
- (c) Required.
- (d) Impossible in Python.

**Q7 (Short answer).** Before splitting a task into two agents, what should you try first?

**Q8 (Multiple choice).** LangGraph's approach to multi-agent is:
- (a) Fully autonomous agents.
- (b) An explicit graph with agent nodes and typed state passed along edges — mixes deterministic control flow with agent autonomy per node.
- (c) Pure message passing.
- (d) Single-agent only.

**Q9 (Short answer).** How would you measure whether a multi-agent decomposition actually improves over single-agent?

**Q10 (Multiple choice).** For most tasks, the right starting architecture is:
- (a) 5-agent hierarchy from the start.
- (b) Single agent with good tools; add agents only when eval shows a measurable win.
- (c) Peer-to-peer mesh.
- (d) Framework-defined "swarm."

---

### Answer Key & Detailed Explanations

**A1.** **Helps:** (1) Deep Research — a planner + N parallel searchers over distinct sub-topics + a synthesizer. Genuine specialization + parallelism. (2) Code refactor pipeline — a planner + N per-file implementers + a reviewer. **Doesn't help:** "Fetch URL + summarize + email" — this is three tools in one agent, or a workflow, not three agents. Specialization is artificial.

**A2. (b).** Hierarchical (supervisor + workers) is the standard pattern: clear control flow, easy to debug, works for most cases. Peer-to-peer (a) is higher variance. Fully connected (c) doesn't scale. Blackboard (d) is niche.

**A3.** A **handoff** is the transfer of control from one agent to another — supervisor dispatches to worker, or worker completes and returns to supervisor. Design patterns for data passing: (1) **Full context** — pass whole conversation (simple, expensive). (2) **Distilled brief** — supervisor writes a scoped prompt for the worker (cleaner, may lose nuance). (3) **Structured payload** — JSON with task, inputs, expected output schema (best for typed pipelines).

**A4. (b).** Supervisor dispatches to N workers in parallel; workers each handle one sub-task; supervisor synthesizes. Standard pattern for parallel work — think Deep Research.

**A5.** **Ping-pong**: agents A and B hand control back and forth indefinitely without resolving the task. Defenses: (1) hard cap on handoffs; (2) require each handoff to make measurable progress; (3) require a "final answer" signal within N turns; (4) supervisor that mediates handoffs rather than letting workers decide freely.

**A6. (b).** Shared mutable state causes race conditions in parallel execution, stale reads, and hard-to-debug issues. Prefer message passing (explicit, testable) or graph state (typed, controlled updates). If you must share state, make it append-only (each agent adds without overwriting).

**A7.** Try a **single agent with good tools first**. Most "multi-agent" decompositions can be expressed as one agent with a well-scoped tool set. Add a second agent only when there's a distinct role that (a) needs a different system prompt, (b) uses a different tool set, and (c) improves quality/cost measurably per your evals.

**A8. (b).** LangGraph is an explicit state-graph framework. Each node is a step (possibly an agent); edges define transitions; a typed state dict is passed along. Some nodes are deterministic (workflow); some are LLM/agent decisions. Mixes both without full autonomy.

**A9.** (1) Build a **single-agent baseline** for the same task. (2) Build the multi-agent version. (3) Run both on the same eval set of tasks. (4) Compare: task success rate, cost per task, latency per task, error diversity. Multi-agent wins only if it improves ≥ 1 dimension without regressing others significantly. If not, ship the simpler version.

**A10. (b).** Anthropic's guidance and empirical evidence: start with the simplest architecture that could work — usually a single well-designed agent with the right tools. Add agents when task complexity or eval results demand it. Multi-agent frameworks are enablers, not defaults.

---

## 6. Practice Prompts

1. **Baseline vs multi-agent.** For a research task (compare 3 products), build (a) single agent with search+fetch+summarize tools, (b) 3-agent hierarchy (planner + researcher + writer). Compare eval scores, cost, latency.
2. **Supervisor prototype.** Build a simple supervisor + 2 workers in LangGraph. Supervisor decides which worker to invoke each turn. Cap 5 handoffs.
3. **Ping-pong test.** Deliberately design a case that induces ping-pong (worker A always says "ask worker B"; worker B always says "ask worker A"). Add defense: max_handoffs=5, require progress signals.
4. **Structured payload handoff.** Define a Pydantic model for the handoff payload between agents. Validate at each transfer.
5. **Parallel workers.** For a "gather info on 5 companies" task, dispatch 5 workers in parallel via `asyncio.gather`. Compare latency to sequential.

---

## 7. References

- Anthropic, ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) (2024) — sections on multi-agent.
- Wu et al., ["AutoGen"](https://arxiv.org/abs/2308.08155) (2023) — conversational multi-agent.
- OpenAI Swarm (repo).
- LangGraph documentation.
- CrewAI (framework).
- Cognitive-multi-agent papers on Camel, MetaGPT, ChatDev.
