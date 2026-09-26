# Agent Loops — Master Study Guide

> **Track:** Agents + Production · **Module:** 03
> **Prerequisites:** Modules 01–02.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** The agent loop is the **control-flow spine** of an agent. Design it well and your agent is reliable, debuggable, cost-controlled. Design it badly and your agent hangs forever, blows the cost budget, or produces surprising infinite loops.

Every agent framework — LangGraph, DSPy, CrewAI, AutoGen — is essentially a **loop with abstractions**. If you understand the loop, the frameworks become readable.

**Fundamental principles you must own:**

1. **Every loop needs a termination condition.** Multiple, actually.
2. **State grows every iteration.** Manage it deliberately — accumulated messages, tool results, reasoning traces.
3. **Errors happen** — tool failures, LLM errors, timeouts. Design recovery paths.
4. **Iteration budgets** cap cost and prevent runaway loops.
5. **The loop can and should be observable** — every step logged, replayable, inspectable.
6. **Different loop topologies serve different needs** — sequential ReAct, tree search, plan-execute, reflection.

If you retain nothing else: **the loop is the agent. Terminate deliberately. Observe every step.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Standard Loop and Its Termination Conditions

**Canonical structure:**

```python
def run_agent(user_query, max_steps=10, max_tokens=100_000, max_wall_time=60):
    messages = [{"role": "user", "content": user_query}]
    total_tokens = 0
    start_time = time.time()
    step = 0

    while True:
        step += 1

        # ---- Termination checks BEFORE the LLM call ----
        if step > max_steps:
            return {"status": "max_steps_exceeded", "trace": messages}
        if total_tokens > max_tokens:
            return {"status": "budget_exceeded", "trace": messages}
        if time.time() - start_time > max_wall_time:
            return {"status": "timeout", "trace": messages}

        # ---- LLM turn ----
        response = call_llm(messages, tools=TOOLS)
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        messages.append(response.assistant_message)

        # ---- Check for termination ----
        if response.stop_reason == "end_turn":
            # LLM produced a final answer; no tool call
            return {"status": "success", "answer": response.text, "trace": messages}

        # ---- Execute tool calls ----
        tool_results = execute_all_tool_calls(response.tool_calls)
        messages.append({"role": "user", "content": tool_results})
```

**Termination conditions to consider:**

- **Model says done** — no tool call this turn; return the text answer.
- **Max iterations** — hard cap (typical: 5-20).
- **Max total tokens** — cost cap.
- **Max wall-clock time** — SLA cap.
- **Max cost in dollars** — spending cap.
- **Repeated identical tool calls** — loop detector.
- **User cancellation** — for long-running agents, allow interruption.
- **Task completion signal** — for agents with an explicit "done" tool: `finish(answer)`.

Best practice: check ALL applicable conditions before every LLM call, not just after.

---

### 2.2 State Management

Every loop iteration adds state. Manage it or it grows unbounded:

**Messages list (canonical state).** The full conversation history: user, assistant, tool_result. This is what the LLM sees. Common issues:

- **Context bloat.** Every turn re-sends the entire history. A 20-turn loop with 3000-token tool results = 60k+ token requests.
- **"Lost in the middle."** After many turns, early messages are attended to less.
- **Cost.** Each token is billed every turn.

**Mitigations:**
- **Summarization.** Periodically summarize old turns into a running summary; drop the raw messages.
- **Sliding window.** Keep last N turns only (with the system prompt + task always in view).
- **External memory.** Move detailed state to a database/vector store; the LLM sees a compact reference.
- **Selective inclusion.** Include only tool results the LLM has actually used since; drop stale observations.

**Scratchpad / working memory.** Some architectures have a separate "notes" field the LLM writes to and reads from — an explicit externalized memory. Cleaner than reparsing conversation history for facts.

**Structured state.** LangGraph's "state graph" pattern makes state explicit: a typed dict updated at each node.

```python
class AgentState(TypedDict):
    query: str
    plan: list[str]
    current_step: int
    results: list[dict]
    final_answer: str | None
```

Each node reads/writes specific fields. Clearer than a pile of messages.

---

### 2.3 Loop Topologies

**Sequential ReAct** — the vanilla pattern. LLM → tool → LLM → tool → answer. Linear.

**Plan-and-Execute** (Wang et al., 2023) — LLM first generates a plan (list of steps); then executes each step. Trade-off: more upfront reasoning, more deterministic execution.

```python
def plan_and_execute(query):
    plan = call_planner_llm(query)   # returns list of steps
    results = []
    for step in plan:
        result = execute_step(step, prior_results=results)
        results.append(result)
    return synthesize(query, plan, results)
```

**Reflection loop** — LLM produces a draft answer; a second LLM (or the same one) critiques; then the LLM revises. Iterate until quality is good enough.

```python
def reflect(query):
    draft = call_llm(f"Answer: {query}")
    for _ in range(max_revisions):
        critique = call_llm(f"Critique this draft: {draft}. Return 'GOOD' if fine.")
        if "GOOD" in critique:
            return draft
        draft = call_llm(f"Revise based on critique: {critique}\nDraft: {draft}")
    return draft
```

**Tree-of-Thoughts (ToT)** — explore multiple reasoning paths in parallel, score each, pick the best. Higher cost, better on hard reasoning tasks.

**Best-of-N / self-consistency** — run the agent N times independently; vote or pick the best answer. Reduces variance.

**Recursive decomposition** — for complex tasks, the agent decomposes into sub-tasks and recursively invokes itself. Useful for coding agents, research agents.

**Which to use:**

- **Standard tasks** → sequential ReAct.
- **Multi-step known-structure** → plan-and-execute.
- **Quality-critical outputs** → reflection.
- **Hard reasoning** → tree-of-thoughts or self-consistency.
- **Complex composite tasks** → recursive decomposition.

Combine as needed. Real agents mix patterns.

---

### 2.4 Error Handling and Recovery

Common errors and their handling:

**Tool errors.** Return a structured error result to the LLM; let it decide to retry, try another tool, or apologize. Do NOT crash the loop.

```python
try:
    result = execute_tool(name, args)
except Exception as e:
    result = {"error": type(e).__name__, "message": str(e)[:500]}
```

**LLM API errors.**
- **429 (rate limited)** — exponential backoff and retry.
- **5xx** — retry with backoff.
- **400 (invalid request)** — bug in your loop; log + fail.
- **Network timeouts** — retry a few times; then fail.

**Malformed tool arguments.** Validate against schema. If invalid, return error to LLM.

**Infinite loop / repeated tool calls.** Detect and abort:
```python
recent_calls = messages[-N:]
if all(same_tool_and_args(c) for c in recent_calls):
    return {"status": "detected_loop", "trace": messages}
```

**Unexpected model output.** Sometimes the LLM produces text when you expected a tool call. Fall back gracefully — return the text as if it were the final answer, or ask for clarification.

**Partial success.** If the agent completes some tools successfully but hits an unrecoverable error later, return partial results with a status flag.

**Graceful degradation.** If a critical tool is down (search API is offline), have a fallback (a local cache, or a simpler model).

---

### 2.5 Observability, Tracing, and Replay

An agent trace is a **structured log** of every step in a run: LLM calls, prompts, responses, tool calls, tool results, timestamps, costs.

**Minimal trace record:**

```python
class TraceStep:
    step_num: int
    timestamp: datetime
    kind: Literal["llm_call", "tool_call", "tool_result", "final_answer"]
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    cost_usd: float
    content: dict          # the actual message / tool call / result
    parent_step: int | None
```

Trace enables:

1. **Debugging.** Why did the agent go wrong? Walk the trace.
2. **Replay.** Feed the same inputs; verify the agent behaves the same. Detect nondeterminism.
3. **Evaluation.** Score traces on completion, cost, latency, correctness (see Module 12).
4. **Post-hoc analysis.** Which tools are used most? Which get errors? Which are unnecessary?
5. **Regression detection.** Compare traces across model or prompt changes.

**Tools:**

- **LangSmith** — first-class LangChain/LangGraph tracing.
- **Braintrust** — evals + tracing platform.
- **Arize Phoenix** — open-source tracing UI.
- **Weights & Biases Weave** — general-purpose LLM tracing.
- **OpenTelemetry** — vendor-neutral; adopt semantic conventions for GenAI.
- **Roll your own** — a JSON logfile per run works surprisingly well for small teams.

**Trace visualization.** A good UI shows:
- Timeline of steps.
- Nested tree if the agent recurses.
- Per-step token count, cost, latency.
- Message content viewable in a side panel.
- Filters by tool, error, cost outlier.

**Streaming traces.** For long-running agents (Deep Research), stream trace events to the user so they see progress. Also enables user cancellation.

**Metrics from traces:**
- Steps per run (p50, p95).
- Total cost per run.
- Total latency per run.
- Tool-error rate.
- Loop-abort rate.
- Success rate (per your eval criteria).

Instrument these from day one. Debugging an agent without traces is impossible at any scale.

---

## 3. Mental Models & Analogies

### 3.1 The "Long-Running SQL Transaction" Model

Think of an agent loop as a **long-running database transaction**:

- **Every tool call** is like a `SELECT` or `UPDATE` in the transaction.
- **The transaction context** (messages / state) accumulates.
- **You need a timeout** or the transaction locks forever.
- **You need to roll back** on error, or gracefully commit partial work.
- **Long transactions block resources** — cost, latency, connections.

Just as good DB engineers set statement timeouts and avoid long transactions, good agent engineers cap iterations, budget cost, and structure loops to be short and checkpoint-able.

### 3.2 The "Chess Player's Clock" Model

A chess game has:

- **Move-by-move actions** (the agent's steps).
- **A running board state** (the accumulated messages/observations).
- **A clock** (max_wall_time budget).
- **A turn limit** (max_steps).
- **Resignation** (early termination if the position is hopeless).

Each move (tool call) both changes the board and consumes clock time. A player who's losing on the clock resigns rather than blundering. Same for agents — better to abort with a clean error than to keep churning past the budget.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Just Let It Run Longer"

Some tasks fail because the model needs more thinking; most agent failures are stuck loops, wrong plans, or bad tools. Extending the iteration cap rarely helps and reliably burns money. **First diagnose why the agent isn't converging** — trace, inspect, root-cause. Then either fix the tool set, prompt, or the loop pattern.

### 4.2 "Exceptions Are Fine — Python Will Handle Them"

An unhandled exception crashes the loop. The LLM never sees the tool failure. You lose the trace mid-run. Users get an ugly 500. Return **structured error results** and let the LLM decide what to do. Catch every exception at the tool boundary.

### 4.3 "State = Messages List — Just Keep Appending"

Messages grow linearly with turns. On turn 20, you're sending 20 turns of history every LLM call. Cost, latency, and "lost in the middle" all suffer. Consider periodic summarization, sliding windows, or structured external state. The messages list is not a free scratchpad — it's a bill.

---

## 5. Self-Assessment Bank (Agent Loops)

### Questions

**Q1 (Short answer).** Name four termination conditions every agent loop should check.

**Q2 (Multiple choice).** In a ReAct agent, the loop exits normally when:
- (a) A specific tool `finish()` is called.
- (b) The LLM produces text with `stop_reason == "end_turn"` (no tool call).
- (c) An exception occurs.
- (d) Max wall-time is reached.

**Q3 (Short answer).** In one paragraph, describe the Plan-and-Execute pattern vs standard ReAct.

**Q4 (Multiple choice).** For a task that requires exploring multiple reasoning paths before committing, the best loop topology is:
- (a) Sequential ReAct.
- (b) Tree-of-Thoughts or Best-of-N.
- (c) Simple loop.
- (d) One-shot generation.

**Q5 (Short answer).** How would you detect an agent that's stuck in an infinite loop of the same tool call?

**Q6 (Multiple choice).** When a tool raises a `TimeoutError`, the correct handling in an agent loop is:
- (a) Let the exception propagate; the loop crashes.
- (b) Return a structured error result to the LLM; it can retry or try a different tool.
- (c) Silently continue.
- (d) Restart the loop from scratch.

**Q7 (Short answer).** Why is unbounded message-list growth a problem across many turns?

**Q8 (Multiple choice).** Reflection loops:
- (a) Are the same as ReAct.
- (b) Have the LLM critique its own output and iteratively revise. Trade-off: quality gain vs cost.
- (c) Only work in multi-agent setups.
- (d) Are deprecated.

**Q9 (Short answer).** What is a "trace" and why is it critical for production agents?

**Q10 (Multiple choice).** Streaming trace events to the user during a long-running agent run:
- (a) Costs extra tokens.
- (b) Improves user experience (visible progress) and enables cancellation. Standard for Deep Research–style agents.
- (c) Breaks the agent loop.
- (d) Is impossible.

---

### Answer Key & Detailed Explanations

**A1.** Any four of: (1) **LLM says done** — no tool call, produces text answer. (2) **Max iterations** — hard step cap. (3) **Max total tokens** — cost cap. (4) **Max wall time** — SLA cap. (5) **Max cost in dollars**. (6) **Repeated identical tool call** — loop detector. (7) **User cancellation**. (8) **Explicit `finish()` tool call**.

**A2. (b).** In ReAct with tool-calling APIs, "stop_reason: end_turn" means the LLM produced a final text answer instead of another tool call. That's the normal successful exit. Some architectures use an explicit `finish` tool (a) as an alternative; (c) and (d) are error exits.

**A3.** **ReAct** (thought → action → observation loop) makes decisions one at a time, adapting to observations. **Plan-and-Execute** first calls an LLM to produce a full plan (list of steps), then executes each step (possibly with a smaller model), then synthesizes. Trade-off: more deterministic execution and can use cheaper models for step-execution, but less adaptive if new information appears mid-plan.

**A4. (b).** Tree-of-Thoughts explores multiple reasoning branches in parallel, scores each, picks the best. Best-of-N runs the agent N times independently and picks the best answer (or votes). Both help on hard reasoning where a single sequential path is likely to fail.

**A5.** Track recent tool calls in a sliding window. If the last N (e.g., 3) tool calls have the same tool name AND same or nearly-same arguments, break the loop with an error like "Detected repeated tool calls; the agent may be stuck." Return the partial trace so the user can inspect. Alternative: track semantic similarity of recent assistant thoughts; near-identical thoughts also signal a loop.

**A6. (b).** Catch the exception at the tool boundary; return `{"error": "TimeoutError", "message": "Tool call timed out after 10s", "retriable": true}` to the LLM. The LLM can decide: retry, try a different tool, ask the user, apologize. Propagating the exception loses the trace and blocks LLM recovery.

**A7.** Every LLM call re-sends the entire message history. On turn 20 with growing tool results, requests balloon to tens of thousands of tokens. Costs scale linearly with turn count; latency scales with TTFT; quality degrades from "lost in the middle". Also, at some point you hit the context limit. Mitigations: periodic summarization, sliding window, external state stores.

**A8. (b).** Reflection loops separate generation from critique: LLM produces a draft; LLM (same or different) evaluates the draft against criteria; LLM revises if needed. Improves quality on complex outputs (code, essays, reasoning). Trade-off: 2–3× the cost per output.

**A9.** A **trace** is a structured log of every step in an agent run — LLM prompts and responses, tool calls, tool results, timestamps, costs, latencies. Critical because: (1) **debugging** — walk the trace to find where things went wrong; (2) **evaluation** — score traces on correctness, cost, latency; (3) **replay** — verify reproducibility; (4) **observability** — aggregate metrics for regression detection. You cannot operate a production agent without traces.

**A10. (b).** Streaming trace events (via SSE or WebSockets) as the agent runs lets the user see progress ("Searching web... Reading result 3... Summarizing..."), gives them the option to cancel a long-running run, and dramatically improves perceived UX. Standard in Deep Research, coding agents, etc. Cost is minimal — the trace is already being generated.

---

## 6. Practice Prompts

1. **Termination coverage.** Extend your minimal agent from Module 01 with all termination conditions: max_steps, max_tokens, max_wall_time, max_cost, loop detector.
2. **Plan-and-Execute.** Rewrite a research task as plan-and-execute (planner LLM produces steps; executor loops). Compare quality and cost to sequential ReAct.
3. **Reflection.** Add a reflection loop to a code-generation task. The critique LLM checks for correctness, style. Compare final quality with and without.
4. **Trace logger.** Add a JSONL logger to your agent — one line per step with kind, tokens, latency, content. Load the trace file and print a summary.
5. **Loop injection defense.** Deliberately induce a loop (retrieval returns a document that says "call `search` again"). Verify your loop detector catches it.

---

## 7. References

- Yao et al., ["ReAct"](https://arxiv.org/abs/2210.03629) (2022).
- Wang et al., ["Plan-and-Solve Prompting"](https://arxiv.org/abs/2305.04091) (2023).
- Yao et al., ["Tree of Thoughts"](https://arxiv.org/abs/2305.10601) (2023).
- Wang et al., ["Self-Consistency"](https://arxiv.org/abs/2203.11171) (2022).
- Shinn et al., ["Reflexion"](https://arxiv.org/abs/2303.11366) (2023).
- OpenTelemetry GenAI semantic conventions.
- LangSmith / Braintrust / Arize Phoenix tracing docs.
