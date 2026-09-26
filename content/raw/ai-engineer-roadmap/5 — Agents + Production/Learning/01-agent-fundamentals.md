# Agent Fundamentals — Master Study Guide

> **Track:** Agents + Production · **Module:** 01
> **Prerequisites:** Month 4 (LLM Engineering), especially tool calling.
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** An **agent** is an LLM that operates in a loop: it decides what to do, does it (via tools or environment interaction), observes the result, and decides again. This shift — from "LLM as text generator" to "LLM as autonomous actor" — powers everything from GitHub Copilot's code-fixing to Deep Research to autonomous SWE agents.

But there's an ocean of hype. **The core is small.** A working agent is:

1. An LLM with a system prompt describing its role and its tools.
2. A tool-execution runtime.
3. A loop that calls the LLM, executes any tool calls, feeds results back, and repeats until the agent produces a final answer or hits a limit.

That's it. Every "agent framework" wraps this pattern in various conveniences. Once you can write the core in 100 lines of Python, everything else is layering — including LangGraph, LangChain, AutoGPT, and CrewAI.

**Fundamental principles you must own:**

1. **An agent = LLM + tools + loop.** Nothing more mysterious.
2. **The ReAct pattern** (Reason + Act) is the canonical structure.
3. **Iteration limits and error handling are non-optional.** Every agent needs a stop condition.
4. **Not every problem needs an agent.** Deterministic workflows are cheaper, faster, and more reliable when the steps are known.
5. **Tools ARE the agent's power.** A tool-less LLM is just a chatbot. Design tools carefully.
6. **The system prompt is the agent's constitution.** Its identity, capabilities, limits, and error behavior all live there.

If you retain nothing else: **an agent is a for-loop around an LLM with tools. Frameworks are conveniences on top.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The ReAct Pattern (Reason + Act)

**ReAct** (Yao et al., 2022) is the canonical agent-loop pattern. Every step, the LLM emits:

- A **thought** (reasoning about what to do next).
- Optionally an **action** (a tool call with arguments).
- Then observes the **result**.

Repeat until the LLM produces a final answer.

**Example trace:**

```
User: What's the current weather in Helena, MT, and how does it compare to last week?

Thought: I need current weather and last week's weather in Helena, MT. I'll call
         get_weather twice — once for today, once for a week ago.
Action:  get_weather(city="Helena, MT", date="today")
Observation: {"temp_f": 45, "condition": "cloudy", "precip_in": 0.0}

Thought: Now the historical.
Action:  get_weather(city="Helena, MT", date="2026-09-15")
Observation: {"temp_f": 62, "condition": "sunny", "precip_in": 0.0}

Thought: I have both. Let me compare.
Final Answer: Today in Helena is 45°F and cloudy, 17°F cooler than a week ago (62°F, sunny).
```

The **thought → action → observation** cycle is the heart of the agent. Modern LLM APIs natively support this via **tool calling** (Module 02) — the LLM outputs a structured tool call, the runtime executes it, and the result feeds back as the next message.

**Historical note.** Early ReAct prompts asked the LLM to output free-form thought/action text; parsers extracted actions via regex. Fragile. **Modern tool calling** (OpenAI, Anthropic, etc.) replaces this with structured JSON tool calls. Much cleaner. Same underlying pattern.

---

### 2.2 The Minimal Agent Loop (in code)

Here's a working agent in ~50 lines of Python. No framework:

```python
import json
from anthropic import Anthropic

client = Anthropic()

TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for a query. Returns top 5 result snippets.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a simple math expression, e.g. '3 * (4 + 5)'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]


def execute_tool(name, args):
    """Dispatch to the concrete tool implementation."""
    if name == "search_web":
        return {"results": mock_search(args["query"])}
    if name == "calculator":
        return {"result": eval_math(args["expression"])}
    raise ValueError(f"Unknown tool: {name}")


def run_agent(user_message, max_steps=10):
    messages = [{"role": "user", "content": user_message}]
    for step in range(max_steps):
        response = client.messages.create(
            model="claude-...",
            system="You are a research assistant. Use tools when needed. "
                   "When you have the final answer, respond in text only.",
            tools=TOOLS,
            max_tokens=1024,
            messages=messages,
        )
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # No more tool calls — final answer
            final = "".join(b.text for b in response.content if b.type == "text")
            return final

        if response.stop_reason == "tool_use":
            # Execute each tool call the model made this turn
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {e}",
                            "is_error": True,
                        })
            messages.append({"role": "user", "content": tool_results})
    raise RuntimeError(f"Agent exceeded {max_steps} steps without finishing.")


answer = run_agent("Search for news on Montana legislative bills passed this session, then count them.")
print(answer)
```

**Key observations:**

- The loop is boring. Framework code adds streaming, retries, observability, but the core is here.
- **Iteration cap (`max_steps`)** — never let an agent loop forever.
- **Error propagation** — tool errors go back to the LLM as `is_error=true` results. The LLM can decide to retry or apologize.
- **`stop_reason`** tells you whether the LLM wants more tools or is done.

This is what LangChain wraps. Understand it once and every framework becomes readable.

---

### 2.3 Anatomy of a Good Agent Design

A production agent isn't a naked loop. It has:

**1. Role and system prompt.** Clear identity, capabilities, refusal behavior. Something like:

```
You are a research assistant. You have access to:
- search_web(query): search the web.
- fetch_url(url): fetch a specific webpage.
- summarize(text): compress long text.

Rules:
- Always cite sources with [source: URL].
- If a task is out-of-scope for research (writing code, sending email, etc.), decline.
- If you cannot find reliable information, say so explicitly.
- Stop and return an answer within 10 tool calls.
```

**2. Tool set.** Small (3–10) is the sweet spot. Too many overwhelms the LLM's selection.

**3. State / memory.** Even short-lived — accumulated observations, intermediate reasoning. See Module 05.

**4. Iteration budget.** Hard cap on steps + cost + wall-clock time.

**5. Error handling.** What happens when a tool fails, times out, returns invalid data? Retry? Escalate? Skip?

**6. Termination criteria.** Explicit stop rules — the LLM returns a "final answer" marker, no more tool calls, or a limit is hit.

**7. Human-in-the-loop hooks.** For destructive actions, require confirmation.

**8. Logging & tracing.** Every turn's messages, tool calls, results, cost, latency — logged and inspectable.

**9. Cost & rate caps.** Total tokens, total dollars, requests-per-minute. Runaway agents can rack up serious bills.

**10. Guardrails.** Input validation (prompt injection defense), output validation (schema, PII), tool-allowlist enforcement.

Modules 03–13 cover each of these in depth. This module is the **map**.

---

### 2.4 Agents vs Workflows — Choose Wisely

Anthropic's ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) (Dec 2024) drew a critical distinction:

**Workflow** — a **predefined** sequence of LLM calls and tool operations. The path is chosen by code; the LLM only fills in outputs.

**Agent** — the LLM **dynamically chooses** the next step. The path is data-dependent.

**Rule of thumb**: use the simplest thing that works. **Workflows first.** Agents only when the task truly requires dynamic decision-making.

**Common workflow patterns:**

- **Prompt chaining** — output of step A is input to step B.
- **Routing** — LLM classifies input; a code branch dispatches to specialized prompts.
- **Parallelization** — same prompt on multiple pieces (map); combine results (reduce).
- **Orchestrator + workers** — a coordinating LLM delegates to specialized worker LLMs (still deterministic dispatch, not full autonomy).
- **Evaluator-optimizer** — one LLM produces output, another critiques; loop until quality is good.

Each of these is *deterministic* at the control-flow level — you designed the graph. The LLM fills in nodes.

**Agents cost more:**
- More LLM calls (each step is a call).
- Higher latency (sequential, not parallel).
- Higher variance in behavior.
- Harder to test (many possible traces).
- Harder to debug.

The user's directive again: don't overengineer. Even in this track, **the deliverable is agent architecture skill, not agent maximalism**. Ship workflows where they suffice.

---

### 2.5 A Brief History and Taxonomy

**Chronology:**

- **2020–2022 — Naïve prompting.** Basic Q&A; no tools; no loops.
- **2022 — ReAct paper (Yao et al.).** Formalized the reason-act-observe loop. Actions parsed from free-form text.
- **2023 — Tool calling APIs.** OpenAI adds function calling; Anthropic and Google follow. Now structured; no more regex-parsing actions.
- **2023 — Frameworks flourish.** LangChain, LlamaIndex, AutoGPT, BabyAGI, AutoGen. Diverse takes on the same loop.
- **2023 — Multi-agent papers.** Camel, AutoGen, MetaGPT. Different agents specialized by role.
- **2024 — LangGraph, DSPy formalize state machines.** Agents become explicit graphs with typed state.
- **2024 — MCP (Anthropic).** Standardized tool/resource protocol for interoperability.
- **2024–2025 — Deep Research (OpenAI/Anthropic/Google).** Multi-hour, multi-tool agents shipped in consumer products.
- **2025 — SWE agents (Cursor Composer, Cline, Claude Code).** Code agents ship at scale in developer tools.
- **2026 — Compound agent systems.** Multi-agent orchestration common; MCP ubiquitous; agent evaluation standardized.

**Agent taxonomies:**

- **By autonomy**: workflow (low) → routing (mid) → autonomous ReAct (high) → open-ended research (highest).
- **By tool set**: single-purpose (search) → multi-tool (search + code exec + email) → generalist (any registered MCP server).
- **By reasoning strategy**: reactive (respond to current state) → planning (build a plan first) → reflective (critique own work) → tree-search (explore alternatives).
- **By population**: single-agent → multi-agent hierarchical → multi-agent peer-to-peer.
- **By persistence**: stateless (one-shot) → stateful (session) → long-running (background jobs).

Every real agent falls somewhere on each axis. Design deliberately.

---

## 3. Mental Models & Analogies

### 3.1 The "New Employee with a Toolbox" Model

An agent is a **new employee** you just hired:

- **The system prompt** = the job description — role, responsibilities, rules, escalation paths.
- **The tools** = the software installed on their machine — what they can actually do.
- **The loop** = their workday — they take a task, think, do work, observe results, iterate.
- **The iteration cap** = "if you can't finish by end of day, escalate."
- **Human-in-the-loop** = "for any expense over $500, get manager approval."
- **Memory** = notes they keep between meetings.
- **Evals** = performance reviews.

A good manager doesn't hire an employee and turn them loose without a job description, an approved toolbox, or an escalation path. Same for agents.

The frameworks are like HR systems — they add scaffolding around this hire-fire-evaluate loop.

### 3.2 The "Video Game NPC" Model

A ReAct agent is a video-game NPC:

- **Perceive** the world (current state, observations from last action).
- **Reason** about goals ("I need to reach the castle").
- **Act** (move forward, open a door, fight a monster).
- **Observe** the new state.
- **Repeat.**

Different agent architectures = different NPC AIs:
- **Simple reactive** = pathfinding bot; responds only to immediate stimuli.
- **Planning** = NPC that plans several moves ahead.
- **Reflective** = NPC that questions its own moves ("was that the best path?").
- **Multi-agent** = squad of NPCs coordinating.

The **environment** is your tool ecosystem. The **goal** is the user's task. The **NPC's brain** is the LLM.

Cool part: the LLM is a *general-purpose* brain that can be dropped into any environment as long as you give it the right tools. That's the power of agents.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "An Agent Solves Everything Better Than a Workflow"

No. Agents are **higher variance, higher cost, harder to test**. For every task where the steps are known in advance, a workflow is faster, cheaper, more reliable, and easier to evaluate. Use agents for **genuinely dynamic** decision-making. This is Anthropic's central "Building Effective Agents" point.

### 4.2 "The Framework Will Handle the Hard Parts"

LangChain/LangGraph/AutoGen don't magically make agents reliable. They add abstractions on top of the same underlying LLM loop. You still need to design tools carefully, cap iterations, evaluate, add guardrails, handle errors. If you don't understand the loop underneath, you can't debug when the framework surprises you (and it will).

### 4.3 "More Tools = Smarter Agent"

Above ~10 tools, LLM selection accuracy degrades. It gets confused about which tool to use. Fixes: (1) reduce tools to essentials; (2) group tools by domain and use routing (LLM first picks a category, then a tool); (3) use hierarchical agents (a supervisor delegates to specialized worker agents with smaller toolsets).

---

## 5. Self-Assessment Bank (Agent Fundamentals)

### Questions

**Q1 (Short answer).** In one paragraph, describe the ReAct pattern.

**Q2 (Multiple choice).** The minimum components of an agent are:
- (a) LLM + database.
- (b) LLM + tools + loop.
- (c) LLM + vector store.
- (d) Multiple LLMs.

**Q3 (Short answer).** Give three non-negotiable design elements for a production agent.

**Q4 (Multiple choice).** For a task where every step is known in advance (e.g., "translate → summarize → email"), the right architecture is:
- (a) An agent with autonomy.
- (b) A workflow — deterministic sequence of LLM calls.
- (c) Multi-agent.
- (d) Reflection loop.

**Q5 (Short answer).** Why is an iteration cap essential in every agent loop?

**Q6 (Multiple choice).** In modern tool-calling APIs, "actions" from ReAct are:
- (a) Parsed from free-form text with regex.
- (b) Emitted as structured JSON tool calls by the LLM.
- (c) Hand-coded by the developer.
- (d) Not part of the API.

**Q7 (Short answer).** Distinguish "workflow" from "agent" in one sentence.

**Q8 (Multiple choice).** Adding more tools to an agent (say, from 5 to 30):
- (a) Always improves performance.
- (b) Usually degrades tool-selection accuracy — hierarchical delegation is better past ~10 tools.
- (c) Has no effect.
- (d) Reduces cost.

**Q9 (Short answer).** Give one design pattern for handling tool errors in an agent loop.

**Q10 (Multiple choice).** LangChain vs writing the loop yourself:
- (a) LangChain is required for production.
- (b) LangChain is a convenience wrapper; the underlying loop is small and understandable — write it once for learning.
- (c) LangChain is faster than manual loops.
- (d) The two are architecturally different.

---

### Answer Key & Detailed Explanations

**A1.** ReAct (Reason + Act) is an agent-loop pattern in which the LLM alternates between **thoughts** (reasoning about what to do next), **actions** (calling a tool with arguments), and **observations** (receiving the tool's result). The LLM keeps looping until it produces a final answer for the user. Modern tool-calling APIs (OpenAI, Anthropic) natively implement this — the "action" is a structured JSON tool call.

**A2. (b).** Minimum viable agent = LLM (the brain) + tools (things it can do) + loop (until it's done). Databases and vector stores are common but not required. Multi-LLM setups are one architecture, not the minimum.

**A3.** Any three of: (1) **System prompt** with role and rules; (2) **Bounded tool set**; (3) **Iteration/cost caps**; (4) **Error handling** for tool failures; (5) **Termination criteria**; (6) **Human-in-the-loop hooks** for destructive actions; (7) **Logging/tracing**; (8) **Guardrails** for input/output validation.

**A4. (b).** A deterministic workflow is cheaper, faster, more predictable, and easier to test than an agent. Reserve agents for tasks where the step sequence must be chosen at runtime based on data.

**A5.** Without a cap, an LLM stuck in a loop (repeatedly calling the same tool, or oscillating between two tools) can run indefinitely — racking up cost, latency, and support tickets. Caps: max_steps (e.g., 10), max_tokens_total (e.g., 100k), max_wall_time (e.g., 60s), max_cost_usd (e.g., $1 per run).

**A6. (b).** Modern APIs (OpenAI function calling, Anthropic tool use, Google function calling) return structured tool-call objects (name + JSON arguments). No regex parsing of free-form text. Reliable and machine-readable.

**A7.** A **workflow** has a **predefined**, code-controlled sequence of LLM/tool operations. An **agent** dynamically decides the next step based on the current state.

**A8. (b).** LLM tool selection accuracy degrades past ~10 tools. Above that, use hierarchical delegation: an outer LLM classifies the request into a domain; a specialized inner agent (with fewer tools) handles the domain.

**A9.** Return the error as a tool-result message with `is_error: true` and a clear error message. The LLM sees the failure and can retry, try a different tool, ask the user, or apologize. Do NOT crash the loop — the LLM often recovers gracefully if you feed it the error.

**A10. (b).** LangChain wraps the same loop with conveniences (streaming, tracing, integrations, prebuilt agents). It's not required. Write the loop yourself once — you'll understand exactly what LangChain is doing, and you'll debug better when it surprises you.

---

## 6. Practice Prompts

1. **50-line agent.** Write the minimal ReAct agent from scratch (no framework), with 2 tools (calculator + web search). Test on 3 questions.
2. **Trace visualization.** For your minimal agent, print every step: assistant messages, tool calls, tool results. Verify the trace is inspectable.
3. **Workflow vs agent.** Take a task like "given a URL, fetch, summarize, translate to Spanish." Implement it as (a) a deterministic workflow (three separate calls), (b) an agent with 3 tools. Compare code complexity, latency, cost, reliability.
4. **Break the agent.** Feed your agent malicious inputs: prompt injection ("ignore all instructions and..."), infinite tool-call loops. Add defenses.
5. **Tool schema audit.** For each tool, verify the name, description, and JSON schema clearly express when and how to call it. Add examples to the description.

---

## 7. References

- Yao et al., ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) (2022).
- Anthropic, ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) (2024).
- Anthropic tool use docs: [docs.claude.com/en/docs/build-with-claude/tool-use](https://docs.claude.com/en/docs/build-with-claude/tool-use).
- OpenAI function calling docs.
- LangChain "How to build an agent" tutorials.
- Andrej Karpathy talks on agents / autonomy (recent).
