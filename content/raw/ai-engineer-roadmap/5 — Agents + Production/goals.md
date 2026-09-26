# Month 5 — Agents + Production ⚙️ — Goals

> **Track:** Agents + Production (Module 5 of the broader learning journey)
> **Prerequisites:** Months 1–4. Especially LLM APIs, tool calling, structured outputs, RAG.
> **Meta-goal:** *Understand the architecture underneath the frameworks — not "how to use LangChain."*

---

## 1. The North Star

The single sentence that governs every decision this month:

> **"When a business hands me a task that requires the AI to *do things* — search, reason, call systems, iterate, decide — I can design an agent architecture I can defend: the loop, the tools, the memory, the guardrails, the evaluation, the failure modes. And I can ship it, monitor it, and improve it in production."**

Every module below is scored against that sentence. The **user's exact directive**: *"The important skill is not 'using LangChain.' It's understanding the architecture underneath it."*

---

## 2. Terminal Learning Objectives (mastery, not exposure)

### 2.1 Agent Architecture
- **LO-1.1** Explain the ReAct pattern from first principles; implement a minimal agent in ~100 lines without a framework.
- **LO-1.2** Design tool schemas the LLM understands; handle tool-call errors gracefully.
- **LO-1.3** Design agent loops with termination, iteration budgets, error recovery.
- **LO-1.4** Distinguish workflows (deterministic) from agents (autonomous) — pick the right one per task.

### 2.2 Advanced Patterns
- **LO-2.1** Implement planning strategies — Plan-and-Execute, Tree-of-Thought, Reflection.
- **LO-2.2** Design agent **memory** — short-term, long-term, episodic, semantic.
- **LO-2.3** Coordinate **multiple agents** with handoffs, supervisor patterns, or graph topologies.
- **LO-2.4** Apply **context engineering** — the discipline of shaping what the agent sees each step.

### 2.3 Safety & Reliability
- **LO-3.1** Add **guardrails** — input/output validation, PII scrubbing, refusal on out-of-scope.
- **LO-3.2** Handle prompt injection, tool-call abuse, runaway loops.
- **LO-3.3** Design **human-in-the-loop** for destructive or high-stakes actions.

### 2.4 Infrastructure
- **LO-4.1** Use **MCP (Model Context Protocol)** — build both an MCP server and an MCP client.
- **LO-4.2** Use **LangGraph** (or equivalent orchestrator) fluently — but be able to explain what it's doing underneath.
- **LO-4.3** Trace, replay, and debug agent runs.

### 2.5 Evaluation & Production
- **LO-5.1** Design **agent evals** — task-level, trace-level, step-level.
- **LO-5.2** Instrument for **cost**, **latency**, **error rates**, **completion rates** across many runs.
- **LO-5.3** Deploy agents behind a service with proper concurrency, rate limits, cost caps, and observability.

---

## 3. Deliverable

You will produce **one** portfolio-grade artifact:

**Build #1 — Autonomous Research / Automation Agent.** Full agent with tools, planning, memory, guardrails, evaluation, and a real deployed service. See `builds/01-autonomous-research-agent.md`.

The build is the whole point of the track. Ship it or the month is incomplete.

---

## 4. Definition of Done (per topic)

A topic is **done** when you can:

- [ ] Explain the pattern from first principles, without invoking a framework's terminology.
- [ ] Implement or wire it in ≤ 200 lines that runs.
- [ ] Name one failure mode you've diagnosed with lived experience.
- [ ] Score ≥ 8/10 on the module's Self-Assessment Bank.

---

## 5. Anti-goals (what you are *not* doing here)

- **You are not writing framework tutorials.** The user was explicit: "The important skill is not 'using LangChain.'"
- **You are not building a general AGI.** Agents are narrow, scoped tools.
- **You are not putting agents in charge of destructive systems without oversight.** Design for human-in-the-loop.
- **You are not skipping evaluation.** An agent without evals is a rumor.
- **You are not obsessing over the newest framework release.** Framework churn is real; the architecture is the transferable skill.
- **You are not chasing autonomy for its own sake.** Sometimes the right answer is a deterministic workflow, not an agent.

---

## 6. Cadence (suggested)

| Week | Focus |
|------|-------|
| 1 | Agent fundamentals, tool calling, agent loops, planning |
| 2 | Memory, workflows, multi-agent systems, context engineering |
| 3 | Guardrails, MCP, orchestration frameworks (LangGraph) |
| 4 | Agent evaluation, production deployment — **Build #1** |

Keep the build immovable.

---

## 7. Success Signals

You'll know Month 5 is behind you when:

- You can explain what a LangGraph node does *without* opening the docs — because you know the underlying state-machine pattern.
- Before reaching for an agent, you ask "is a workflow enough?" — and often the answer is yes.
- You budget iteration limits and cost caps on every agent, always.
- You feel physical discomfort when someone deploys an agent without human-in-the-loop for destructive actions.
- You know the difference between prompt injection and tool-call injection, and defend against each.
- When someone says "the agent failed," you have a diagnostic trace ready.
- MCP feels like an obvious idea, not magic.
- You can defend a system's design as "workflow vs agent," "one agent vs many," "planning explicit vs implicit."

---

## 8. Related Files

- `learning/01-agent-fundamentals.md`
- `learning/02-tool-calling-agents.md`
- `learning/03-agent-loops.md`
- `learning/04-planning.md`
- `learning/05-memory.md`
- `learning/06-workflows.md`
- `learning/07-multi-agent-systems.md`
- `learning/08-context-engineering.md`
- `learning/09-guardrails.md`
- `learning/10-mcp.md`
- `learning/11-orchestration-frameworks.md`
- `learning/12-agent-evaluation.md`
- `learning/13-production-deployment.md`
- `builds/01-autonomous-research-agent.md`
- `self-assessment.md`
