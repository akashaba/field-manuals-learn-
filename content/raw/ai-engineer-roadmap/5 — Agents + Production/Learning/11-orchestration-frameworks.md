# Orchestration Frameworks (LangGraph & Friends) — Master Study Guide

> **Track:** Agents + Production · **Module:** 11
> **Prerequisites:** Modules 01–10.
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** By this point in the track, you can build agents from scratch (Modules 01–03). Frameworks add **conveniences** on top: state graphs, streaming, checkpointing, human-in-the-loop, deployment, tracing. The user was explicit: "The important skill is not using LangChain. It's understanding the architecture underneath it." **This module covers the architecture underneath the frameworks so you can pick and use them well.**

By 2026, the ecosystem has consolidated around a few patterns:

- **State-graph frameworks** (LangGraph, LlamaIndex Workflows, Semantic Kernel Agent Framework) — explicit graphs of nodes and edges with typed state.
- **Declarative agent frameworks** (DSPy) — describe *what* the agent does; compile to prompts.
- **Multi-agent orchestrators** (AutoGen, CrewAI, OpenAI Swarm, Google ADK) — abstractions for agent-to-agent coordination.
- **DAG execution frameworks** (Prefect, Temporal, Dagster) — general-purpose workflow engines increasingly used for agentic workloads.

**Fundamental principles you must own:**

1. **All frameworks wrap the same underlying loop** you built in Module 01.
2. **State graphs are the emerging default** for structured agents — LangGraph is the reference.
3. **Frameworks trade abstraction for control.** More abstraction = less code; less transparency.
4. **Choose based on team fit** — what your team can operate and debug.
5. **Framework churn is real** — pick something with an active community; don't rewrite for every hype cycle.
6. **You should be able to leave a framework** — architecture skill > framework skill.

If you retain nothing else: **frameworks are tools; the architecture underneath is the transferable skill.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 LangGraph — The State Graph Model

**LangGraph** (from the LangChain team) is the currently-dominant state-graph framework for agent workflows.

**Core idea:** an agent is a **directed graph** of **nodes** operating on a **typed state**. Edges connect nodes; conditional edges branch based on state.

**Anatomy:**

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # append-only messages
    plan: list[str]
    current_step: int
    results: list[dict]

def planner_node(state: AgentState) -> dict:
    plan = call_planner_llm(state["messages"][-1].content)
    return {"plan": plan, "current_step": 0}

def executor_node(state: AgentState) -> dict:
    step = state["plan"][state["current_step"]]
    result = execute_step(step)
    return {"results": state["results"] + [result],
            "current_step": state["current_step"] + 1}

def should_continue(state: AgentState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return "finalize"

def finalize_node(state: AgentState) -> dict:
    answer = synthesize(state["messages"][-1].content, state["results"])
    return {"messages": [{"role": "assistant", "content": answer}]}

graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue,
                             {"executor": "executor", "finalize": "finalize"})
graph.add_edge("finalize", END)

app = graph.compile()
result = app.invoke({"messages": [{"role": "user", "content": "Research X"}]})
```

**What LangGraph gives you:**

- **Typed state** (via `TypedDict`) — every node reads/writes a known shape.
- **Structured control flow** — nodes and edges are explicit; you draw the graph.
- **Checkpointing** — pause/resume/replay agent runs. Powerful for human-in-the-loop and long jobs.
- **Streaming** — stream state updates as they happen.
- **Human-in-the-loop** — pause at a node awaiting external input.
- **Studio / debugging UI** — visualize graphs and traces.

**When LangGraph shines:**
- Complex multi-step agents with branching logic.
- Workflows that mix deterministic control with LLM autonomy.
- Long-running agents needing pause/resume.
- Team collaborations where visualizing the graph aids discussion.

**When LangGraph doesn't earn its weight:**
- Simple sequential ReAct — 100 lines of vanilla Python is just as clear.
- Rapid prototyping where the graph shape isn't stable yet.
- Extremely latency-sensitive apps — the abstraction adds overhead.

**LangChain vs LangGraph.** LangChain is the older library with many integrations (LLM wrappers, vector stores, doc loaders, tool schemas) and higher-level "chains." LangGraph is the newer graph-oriented framework, cleaner for agents. Common pattern: use LangChain's integrations *inside* a LangGraph node — best of both.

---

### 2.2 DSPy — Declarative Agents

**DSPy** (Stanford NLP) takes a radically different angle: describe the agent's **task** and **signatures** declaratively; DSPy compiles prompts (and even fine-tunes) to hit metrics.

```python
import dspy

class SolveTask(dspy.Signature):
    """Solve the given research task."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class ResearchAgent(dspy.Module):
    def __init__(self):
        self.search = dspy.Predict(dspy.Signature(
            "query -> results: list[str]"))
        self.answer = dspy.ChainOfThought(SolveTask)

    def forward(self, question):
        results = self.search(query=question).results
        context = "\n".join(results)
        return self.answer(question=f"{context}\n\n{question}")

# Compile the agent — DSPy optimizes prompts for a metric
compiled = dspy.compile(ResearchAgent(), trainset=labeled_examples, metric=my_metric)
```

**Philosophy:** you write structured programs (modules with signatures); DSPy handles prompt engineering by **compiling** — running the modules against a dev set and optimizing the prompts (few-shot examples, chain-of-thought templates) to maximize a metric.

**When DSPy earns its weight:**
- You have a **labeled eval set** to compile against.
- Multiple related tasks share prompt structure (DRY).
- Prompt engineering is where most quality gains come from.

**When it doesn't:**
- No eval set → nothing to compile against.
- Prompt engineering isn't your bottleneck (data, tools are).
- Steep learning curve; not what your team knows.

Less widely adopted than LangGraph but growing in serious ML teams.

---

### 2.3 Multi-Agent Frameworks (AutoGen, CrewAI, Swarm, Google ADK)

Different frameworks optimize for different multi-agent patterns.

**Microsoft AutoGen (2023).** Peer-to-peer chat between agents. Each agent has a role, tools, and can send messages. Agents "converse" until a termination criterion. Good for research, debate, adversarial critique. Recent versions ship an event-driven core (v0.4+).

**CrewAI.** Role-based crews with hierarchical or sequential process. Simpler mental model than AutoGen; opinionated defaults. Good for straightforward multi-role tasks.

**OpenAI Swarm (2024).** Minimalist. Agents return other agents to hand off control. Very small footprint; opinionated for "supervisor" patterns.

**Google Agent Development Kit (ADK, 2025).** Google's answer. Multi-agent, integrated with Vertex AI, Gemini, and Google Cloud tools.

**Semantic Kernel Agent Framework (Microsoft).** Enterprise .NET/Python framework with strong Azure integration.

**Which to use:**
- **Prototyping multi-agent debate/critique** → AutoGen.
- **Structured multi-role pipelines** → CrewAI.
- **Simple supervisor + workers, minimal magic** → Swarm.
- **On Google Cloud** → Google ADK.
- **On Azure / .NET stacks** → Semantic Kernel.

All are wrappers around the same underlying loop + message-passing pattern (Module 07). Learn one deeply; you'll transfer skills to the others.

---

### 2.4 Workflow Engines (Temporal, Prefect, Dagster)

Non-agent-specific but increasingly used for agentic workloads.

**Temporal.** Durable execution engine. Great for long-running, resumable, retry-heavy workflows. If your agent runs for hours, calls flaky external services, and must survive crashes, Temporal is battle-tested.

**Prefect.** Python-first workflow orchestrator. Good for data pipelines, easy to reason about. Prefect 2+ has native support for observability and retries.

**Dagster.** Data-orchestration heavy; strong for ML pipelines with data dependencies.

**Restate.** Newer entrant, targets durable execution for agents.

**When to add a workflow engine:**
- Long-running agents (hours to days).
- Deep retry logic against flaky external services.
- Need durable state (survives crashes / restarts).
- Complex scheduling / dependencies across agent runs.

**Overkill for:**
- Interactive chatbot with < 30-second turnaround.
- One-off scripts / prototypes.

**Common combination:** LangGraph (agent structure) inside Temporal (durable execution). Or Prefect for scheduling; the agent itself is a task.

---

### 2.5 Choosing a Framework (Or None)

**Decision matrix:**

| Criterion | Option |
|-----------|--------|
| Simple ReAct agent, < 200 LoC | Vanilla Python |
| Structured multi-step with branching | LangGraph |
| Multi-agent chat / debate | AutoGen |
| Role-based crew, sequential | CrewAI |
| Simple supervisor + workers, minimal | OpenAI Swarm |
| Declarative + auto-prompt-optimization | DSPy |
| Long-running durable workflows | Temporal + inner agent library |
| Full stack, Google Cloud | Google ADK |
| Full stack, Azure/.NET | Semantic Kernel |

**Selection rubric:**

1. **Team fit** — what can your team read and maintain?
2. **Ecosystem** — which framework has the integrations you need?
3. **Community activity** — active open-source, roadmap, not abandonware.
4. **Debuggability** — can you trace runs and understand behavior?
5. **Deployment story** — hosting, scaling, observability.
6. **Lock-in cost** — how hard is it to leave? Prefer frameworks that lean on standard patterns.

**When to use no framework:**
- Simple agents (single tool, ReAct, < 500 lines).
- You need extreme latency / cost control.
- Framework overhead outweighs framework benefits.
- Learning — Module 01's from-scratch agent is invaluable.

**Framework churn.** Between 2022 and 2026, we've seen: LangChain, AutoGPT, BabyAGI, LangGraph, AutoGen, CrewAI, Swarm, DSPy, Semantic Kernel, ADK, Restate, LlamaIndex Workflows, MCP as protocol, and many more. Half will consolidate or fade. Skills that transfer: architecture (Modules 01–10 + 12–13), MCP, and the ability to read framework source.

**The user was explicit — but bears repeating.** Framework fluency is *lower on the priority list* than architectural understanding. Learn one deeply for your build; understand the pattern behind it; be ready to switch.

---

## 3. Mental Models & Analogies

### 3.1 The "Web Framework" Model

Recall the evolution of web frameworks: raw sockets → CGI → mod_python → Rails/Django → Flask → FastAPI. Each layer added conveniences: routing, middleware, ORM. **Architecture** — HTTP, REST, request/response, MVC — stayed the same. Devs who understand HTTP can pick up Flask in an afternoon; those who only know Rails idioms struggle without their Rails.

Agent frameworks are following the same trajectory. The **architecture** (loop, tools, state, planning, memory, evaluation) stays the same. The **frameworks** (LangGraph et al.) provide conveniences. Investment in architecture: transferable. Investment in a specific framework: erodes with each new one.

### 3.2 The "IDE for a Language" Model

An IDE (VS Code, IntelliJ, PyCharm) is a productivity multiplier. But if you can only work in an IDE and can't read raw code, you have a shallow skill. If you understand the language deeply, any IDE is a tool.

Same with agent frameworks. LangGraph, DSPy, AutoGen — these are IDEs for agent development. Use them! But if you can't build the same agent in 200 lines of vanilla Python, you don't understand it. Test yourself: read a LangGraph example → know what it's doing under the hood → know why each design decision was made.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Framework Is the Architecture"

The user's central point. LangGraph does NOT define agent architecture. It's an implementation choice. **Understand the state-graph pattern**, and LangGraph becomes obvious; skip that step and you'll be lost when it changes (which it will).

### 4.2 "Frameworks Handle Everything Automatically"

They handle the mechanics — the loop, streaming, checkpointing. They don't handle:
- Tool design (Module 02).
- Prompt engineering (Month 4 Module 04).
- Context engineering (Module 08).
- Guardrails (Module 09).
- Evaluation (Module 12).
- Cost/latency optimization (Month 4 Module 10).

Framework code is the smallest part of a well-designed agent. Everything else is where the quality comes from.

### 4.3 "Pick the Newest Framework"

The newest often lacks community and stability. Established frameworks (LangGraph, AutoGen) have battle-tested docs, examples, StackOverflow answers, and known failure modes. Wait 6 months for hype to settle; pick what teams actually use.

---

## 5. Self-Assessment Bank (Orchestration Frameworks)

### Questions

**Q1 (Short answer).** Explain LangGraph's core abstraction (state graph) in one paragraph.

**Q2 (Multiple choice).** LangGraph's typed state (via `TypedDict`) is:
- (a) Just documentation.
- (b) A shared, structured object read/written by each node — the graph's memory.
- (c) Only for debugging.
- (d) A DB schema.

**Q3 (Short answer).** For an agent that takes hours and must survive crashes, name a framework choice and why.

**Q4 (Multiple choice).** DSPy differs from LangGraph in that:
- (a) DSPy is not for LLMs.
- (b) DSPy is a declarative programming model that auto-compiles prompts / few-shots against an eval set.
- (c) DSPy runs on GPUs only.
- (d) DSPy is deprecated.

**Q5 (Short answer).** Give two situations where you should NOT use an agent framework.

**Q6 (Multiple choice).** OpenAI Swarm's core primitive is:
- (a) Multi-agent chat.
- (b) An agent that returns another agent — modeling handoffs as function returns.
- (c) A state graph.
- (d) A workflow engine.

**Q7 (Short answer).** Why is architectural understanding more transferable than framework fluency?

**Q8 (Multiple choice).** For a simple ReAct agent with 3 tools and no branching, the right choice is:
- (a) LangGraph.
- (b) Vanilla Python (~100 lines).
- (c) AutoGen.
- (d) Temporal.

**Q9 (Short answer).** Describe how LangGraph's checkpointing enables human-in-the-loop.

**Q10 (Multiple choice).** Frameworks (LangGraph, AutoGen, etc.):
- (a) Handle everything about agent design.
- (b) Wrap the underlying loop + state pattern; you still design tools, prompts, guardrails, evals — the bulk of quality work.
- (c) Are unnecessary.
- (d) Only work with OpenAI.

---

### Answer Key & Detailed Explanations

**A1.** LangGraph models an agent as a **directed graph** of **nodes** operating on a **shared typed state**. Each node is a function that reads the current state and returns updates. Edges define transitions (deterministic or conditional based on state). The graph is compiled to an executor. This structure makes agent control flow explicit and inspectable, enables features like streaming and checkpointing, and supports mixing deterministic workflow steps with LLM-decision nodes.

**A2. (b).** The typed state is a shared object with named fields (via `TypedDict`), read and updated by each node. It's the graph's memory: instead of passing arguments through a chain of function calls, all state lives in one dict. This makes state management explicit, testable, and inspectable.

**A3.** **Temporal** (or Restate). Temporal is a durable execution engine: workflow state is persisted; if the process crashes, execution resumes from the last checkpoint automatically. For long-running (hour+) agent runs with external service dependencies, this is a huge win — you don't lose an hour of work to a transient failure. LangGraph nodes can run inside Temporal workflow steps for a combined stack.

**A4. (b).** DSPy is a declarative programming model: you describe your agent as modules with input/output signatures, and DSPy compiles the actual prompts (few-shot examples, chain-of-thought templates) by optimizing them against a labeled eval set. LangGraph is imperative (you write the graph); DSPy is declarative (you write the specification; prompts are compiled).

**A5.** (1) **Simple sequential ReAct agents** (< 200 LoC) — the framework overhead exceeds its benefit. (2) **Rapid prototyping** where the agent shape isn't stable — framework abstractions crystallize decisions you're not ready to make. (3) **Extreme latency/cost sensitivity** — direct function calls beat framework indirection. (4) **Learning** — writing agents from scratch teaches architecture faster than adopting a framework. Any two plus rationale.

**A6. (b).** OpenAI Swarm's minimalist design uses **agent handoffs as function returns**: an agent's function can return another `Agent` object, signaling the framework to hand control to that agent. Very small footprint; models supervisor + workers cleanly.

**A7.** Frameworks come and go. Architecture — the loop, state, tools, planning, memory, guardrails, evals — is invariant across frameworks. A dev who understands architecture can adopt any framework in a week; a dev who only knows Framework X is stranded when it declines. The user's directive: architecture > framework.

**A8. (b).** Simple ReAct with 3 tools is ~100 lines of vanilla Python. LangGraph adds abstraction overhead; AutoGen is for multi-agent; Temporal is for long-running durable work. Match tool to task.

**A9.** LangGraph checkpoints save the state after each node execution. To enable human-in-the-loop, add an "approval" node where execution pauses; the state is checkpointed; the app waits for external input (a human clicks approve/reject via UI). On resumption, the state is loaded from the checkpoint and execution continues from where it stopped. This makes pause/resume/edit natural in the graph model.

**A10. (b).** Frameworks handle mechanics — loops, streaming, state passing, checkpointing. They do NOT design tools, prompts, guardrails, evals — the bulk of quality work. A great framework choice on top of poor tool/prompt/eval design produces a bad agent. The framework is the smallest part of a good system.

---

## 6. Practice Prompts

1. **LangGraph rebuild.** Take your minimal from-scratch agent (Module 01) and rebuild it in LangGraph. Compare code, features, and clarity.
2. **State graph design.** Draw the state graph for your Build #1 agent: nodes, edges, state fields, conditional branches. Verify it's expressible in LangGraph.
3. **DSPy compilation.** Write a small DSPy program (Q&A over a corpus). Provide a small labeled eval set. Compile. Observe how DSPy modified the prompts.
4. **AutoGen multi-agent.** Build a 2-agent debate: proposer + skeptic. Test on a controversial technical question. Observe the conversation.
5. **Framework migration.** Take an agent you built in one framework; port to another. Note what stays the same (architecture) vs what changes (framework idioms).

---

## 7. References

- LangGraph docs: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/).
- DSPy docs: [dspy.ai](https://dspy.ai/).
- Microsoft AutoGen: [microsoft.github.io/autogen](https://microsoft.github.io/autogen/).
- CrewAI: [crewai.com](https://www.crewai.com/).
- OpenAI Swarm: [github.com/openai/swarm](https://github.com/openai/swarm).
- Google Agent Development Kit (ADK).
- Temporal for LLMs: [temporal.io/blog](https://temporal.io/blog).
- Anthropic, ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) — framework-agnostic architecture guide.
