# 05.02 — Agentic Workflows

> **Module 5: Tools & Agents** · Subtopic 2 of 6
> **Prerequisites:** 05.01 (tool runtime and agent loop), 02.01 §4 (reasoning, ReAct, chaining), 02.02 §5–7 (compaction, sub-agent isolation), 04.04 §4 (decomposition), basic probability.
> **Outcome:** you can choose the *least autonomous* architecture that solves a task, implement the standard workflow patterns and agent loops with budgets and termination guarantees, run them on durable orchestration, reason quantitatively about reliability, and evaluate agents with trajectory checks and pass^k.

---

## 1. Workflows vs Agents: The Autonomy Spectrum

```
 deterministic code ◄──────────────────────────────────────────────────────────────► autonomous agent
 ┌───────────────┬──────────────────┬───────────────────────┬──────────────────────┬──────────────────┐
 │ single LLM    │ WORKFLOW         │ WORKFLOW with LLM     │ AGENT in a bounded   │ open-ended agent │
 │ call + tools  │ fixed steps,     │ decisions (routing,   │ loop: model chooses  │ plans, spawns    │
 │               │ LLM in each step │ evaluator loops)      │ tools & when to stop │ sub-agents       │
 └───────────────┴──────────────────┴───────────────────────┴──────────────────────┴──────────────────┘
   predictability, cost control, testability  ◄────────►  flexibility for open-ended tasks
```

- **Workflow:** the *code* defines the control flow, and LLMs perform steps.
- **Agent:** the *model* directs its own process and tool use in a loop, until it decides it is done.

**Rule:** start at the left, and move right only when evals show the task needs it. Most production value comes from workflows, with agents confined to the parts that genuinely require open-ended exploration (research, debugging, multi-step investigation).

---

## 2. The Workflow Patterns

| Pattern | Shape | Use when | Example (legislative) |
|---|---|---|---|
| **Prompt chaining** | A → gate → B → C | The task decomposes into fixed stages; gates check intermediate quality | Extract bill metadata → validate → draft summary → check citations |
| **Routing** | classify → specialised path | Distinct input types need different handling | Route: statute question / bill status / fiscal question / chit-chat |
| **Parallelisation: sectioning** | split → parallel workers → merge | Independent sub-tasks | Analyse each section of a bill in parallel |
| **Parallelisation: voting** | same task × n → aggregate | Higher confidence via diversity | 3 independent reviews of a constitutional-risk flag; majority vote |
| **Orchestrator–workers** | a planner decides the sub-tasks dynamically → workers → synthesis | Sub-tasks aren't known in advance | "Summarise all 2025 bills affecting county budgets" |
| **Evaluator–optimiser** | generate → critique → revise loop | Clear evaluation criteria exist; iteration helps | Draft an amendment → check against drafting rules → revise |

```python
import asyncio
from collections import Counter
from typing import Awaitable, Callable

LLM = Callable[[str], Awaitable[str]]


async def chain(llm: LLM, x: str, steps: list[str], gate: Callable[[int, str], bool] = lambda i, y: True) -> str:
    for i, template in enumerate(steps):
        x = await llm(template.format(input=x))
        if not gate(i, x):
            raise ValueError(f"gate failed after step {i}")
    return x


async def route(classify: Callable[[str], Awaitable[str]], handlers: dict[str, Callable[[str], Awaitable[str]]],
                x: str, default: str) -> str:
    label = await classify(x)
    return await handlers.get(label, handlers[default])(x)


async def parallel_vote(llm: LLM, prompt: str, n: int = 3, parse=lambda s: s.strip()) -> tuple[str, float]:
    outs = [parse(o) for o in await asyncio.gather(*(llm(prompt) for _ in range(n)))]
    best, votes = Counter(outs).most_common(1)[0]
    return best, votes / n


async def orchestrator_workers(plan: Callable[[str], Awaitable[list[str]]], worker: LLM,
                               synthesize: Callable[[str, list[str]], Awaitable[str]], task: str,
                               max_workers: int = 8, concurrency: int = 4) -> str:
    subtasks = (await plan(task))[:max_workers]
    sem = asyncio.Semaphore(concurrency)

    async def run(st):
        async with sem:
            return await worker(st)
    results = await asyncio.gather(*(run(s) for s in subtasks))
    return await synthesize(task, results)


async def evaluator_optimizer(generate: LLM, evaluate: Callable[[str], Awaitable[tuple[bool, str]]],
                              task: str, max_rounds: int = 3) -> dict:
    draft, history = await generate(task), []
    for r in range(max_rounds):
        ok, feedback = await evaluate(draft)
        history.append(feedback)
        if ok:
            return {"output": draft, "rounds": r + 1, "accepted": True}
        draft = await generate(f"{task}\n\nPrevious draft:\n{draft}\n\nFix these issues:\n{feedback}")
    return {"output": draft, "rounds": max_rounds, "accepted": False, "feedback": history}
```

---

## 3. The Agent Loop, Formally

At step $t$ the agent has a context $c_t$ (instructions, history, observations). It samples an action $a_t \sim \pi_\theta(\cdot \mid c_t)$ — a tool call, a message, or **stop** — then receives an observation $o_t$ and updates $c_{t+1} = c_t \oplus (a_t, o_t)$. This is a partially observable decision process in which the policy is a frozen LLM steered by the prompt.

**Every agent loop needs explicit guarantees:**

| Guarantee | Mechanism |
|---|---|
| **Termination** | Max steps, max tool calls, max tokens, max wall-clock, max cost; loop detection |
| **Progress** | Plan or checklist state; detect repeated (action, args) patterns; stall detection |
| **Safety** | Tool permissions, approval gates (05.06), sandboxing (05.01 §5) |
| **Recoverability** | Checkpointed state (05.03); idempotent tools (05.01 §4) |
| **Observability** | A trace per step (model call, tool call, tokens, decision) |

```python
import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    max_steps: int = 25
    max_tool_calls: int = 60
    max_tokens: int = 400_000
    max_seconds: float = 300
    max_usd: float = 2.0
    loop_window: int = 6              # look back this many actions
    loop_repeats: int = 3             # identical action seen this many times → loop

    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    usd: float = 0.0
    started: float = field(default_factory=time.monotonic)
    recent: list[str] = field(default_factory=list)

    def charge(self, tokens: int = 0, usd: float = 0.0, tool_calls: int = 0, step: bool = False):
        self.tokens += tokens
        self.usd += usd
        self.tool_calls += tool_calls
        self.steps += int(step)

    def record_action(self, name: str, args: dict) -> str | None:
        h = hashlib.sha1(f"{name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()[:12]
        self.recent = (self.recent + [h])[-self.loop_window:]
        if self.recent.count(h) >= self.loop_repeats:
            return f"loop detected: {name} repeated {self.loop_repeats}x with identical arguments"
        return None

    def exceeded(self) -> str | None:
        checks = [("steps", self.steps, self.max_steps), ("tool_calls", self.tool_calls, self.max_tool_calls),
                  ("tokens", self.tokens, self.max_tokens), ("usd", self.usd, self.max_usd),
                  ("seconds", time.monotonic() - self.started, self.max_seconds)]
        return next((f"{n} budget exceeded ({v:.4g} > {lim})" for n, v, lim in checks if v > lim), None)
```

When a budget trips, **don't just stop**. Return a structured partial result (what was done, what remains, and why it stopped) so that a human or the caller can continue (05.06).

---

## 4. Planning Strategies

| Strategy | How | Strength | Cost / risk |
|---|---|---|---|
| **ReAct** | Interleave thought → action → observation | Adaptive, simple | Myopic; can wander |
| **Plan-and-execute** | Plan the steps first, execute them, re-plan on failure | Coherent long tasks; a cheaper executor model | Stale plans if the world differs from the assumptions |
| **Reflexion** | After a failure, write a verbal lesson and retry with it in memory | Learns within the episode or across episodes | Needs a reliable failure signal |
| **Tree search** (LATS, ToT) | Explore alternative action branches with value estimates | Hard tasks with a verifiable outcome | Multiplies cost; needs a state reset/fork |
| **Reasoning models + tools** | Interleaved thinking between tool calls | Strong default for complex tasks today | Token cost of thinking; budget it (02.01 §4.5) |

**Practical default:** a reasoning-capable model in a ReAct-style loop, with a lightweight **checklist or plan state** that the agent maintains through a tool (`update_plan`). Add re-planning when a step fails, plus a verifier step before final output.

---

## 5. Reliability Engineering for Agents

### 5.1 Verify-and-retry, quantified

Suppose a step succeeds with probability $p$. A verifier detects a failure with probability $v$ (its recall), and detected failures are retried, up to $r$ times. Undetected failures pass through as final failures, and the verifier is assumed not to reject correct outputs. Then

$$
P_{\text{step}} = p\sum_{k=0}^{r}\big((1-p)\,v\big)^k = p\,\frac{1 - \big((1-p)v\big)^{r+1}}{1 - (1-p)v}
$$

With $p = 0.9$, $v = 0.8$, $r = 2$: $P_{\text{step}} \approx 0.978$. Over 20 steps, $0.9^{20} \approx 0.12$ becomes $0.978^{20} \approx 0.64$. **Cheap, high-recall verifiers are the highest-leverage reliability investment** — schema validation, tests, citation checks, database constraint checks.

```python
def step_success(p: float, v: float, r: int) -> float:
    q = (1 - p) * v
    return p * (1 - q ** (r + 1)) / (1 - q) if q < 1 else p * (r + 1)


def task_success(p: float, v: float, r: int, n_steps: int) -> float:
    return step_success(p, v, r) ** n_steps
```

### 5.2 Other levers

- **Shorter trajectories:** task-level tools, programmatic tool calling (05.01 §2.2).
- **Checkpoints + resume** instead of restarting from scratch (05.03).
- **Error-aware prompting:** tell the agent how to react to tool errors ("If a search returns nothing, broaden the query once, then report the gap").
- **Guardrail steps:** a final verifier (or a human) before irreversible actions (05.06).

---

## 6. Orchestration: Graphs, State Machines, Durable Execution

Hand-written `while` loops don't survive production: process restarts, 30-minute tasks, human approvals that take a day, retries. Model agent control flow as an **explicit graph or state machine** over typed state, and run it on a **durable execution** engine.

```
            ┌────────────┐   needs_research   ┌─────────────┐
 START ───► │  plan      │──────────────────► │  research   │──┐
            └─────┬──────┘                    └─────────────┘  │ (loop ≤ 3)
                  │ ready                            ▲          │
                  ▼                                  └──────────┘
            ┌────────────┐  fails checks  ┌────────────┐
            │  draft     │◄───────────────│  verify    │
            └─────┬──────┘                └─────┬──────┘
                  └──────────────────────────►  │ passes
                                                ▼
                                   ┌────────────────────────┐ approve ┌──────────┐
                                   │ human_review (interrupt)│───────► │ publish  │──► END
                                   └────────────────────────┘ reject → draft
```

```python
from typing import Awaitable, Callable

Node = Callable[[dict], Awaitable[dict]]            # returns a state *update*
Edge = Callable[[dict], str]                         # returns the next node name


async def run_graph(nodes: dict[str, Node], edges: dict[str, Edge], state: dict, start: str,
                    end: str = "END", max_transitions: int = 50, on_step=None) -> dict:
    current, transitions = start, 0
    while current != end:
        transitions += 1
        if transitions > max_transitions:
            raise RuntimeError(f"max transitions exceeded at node '{current}'")
        update = await nodes[current](state)
        state = {**state, **update, "_last": current}
        if on_step:
            on_step(current, state)                     # checkpoint hook (05.03)
        if state.get("_interrupt"):
            return {**state, "_paused_at": current}     # wait for a human or external event (05.06)
        current = edges[current](state)
    return state
```

**Engines:**
- **LangGraph:** typed state, conditional edges, checkpointers, interrupts.
- **Temporal / Restate / Durable Functions:** durable workflows. Deterministic workflow code, with activities for the LLM and tool calls, automatic retries, timers, and **signals** for human input.
- **Step Functions / Cloud Workflows:** managed state machines.

With Temporal, **every LLM call and tool call is an activity**. They are recorded in the workflow history, so a crash replays the history instead of re-calling the model.

---

## 7. Multi-Agent Systems

**Orchestrator–subagent** designs (02.02 §7) give each sub-agent a clean context for a parallel, separable sub-task. **Handoffs** transfer control and context between specialised agents (e.g. triage → billing agent). **Debate or critique** pairs agents to challenge each other's answers.

**Costs and failure modes:**
- **Token multiplier:** multi-agent runs often consume several times more tokens than a single agent. The gain must be worth it (breadth-heavy research is the canonical win).
- **Coordination failures** are the dominant class (Cemri et al., 2025, MAST taxonomy): specification and role ambiguity, inter-agent misalignment (ignored inputs, withheld information, conversation resets), and weak verification and termination.
- **Mitigations:**
  - Precise hand-off contracts (objective, constraints, output schema, budget).
  - A single owner of the final answer.
  - An explicit verifier role.
  - Shared state in a store rather than in chat.
  - Hard budgets per sub-agent.

---

## 8. Evaluating Agents

### 8.1 Outcome metrics that capture reliability

Run each task $n$ times, with $c$ successes:

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \quad(\text{at least one of } k \text{ succeeds}),\qquad
\text{pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}} \quad(\text{all } k \text{ succeed})
$$

pass^k (from τ-bench) measures **consistency**, which is what production users experience. An agent at 80% pass@1 may have pass^8 < 20%.

```python
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0


def pass_hat_k(n: int, c: int, k: int) -> float:
    return comb(c, k) / comb(n, k) if c >= k else 0.0
```

### 8.2 Trajectory evaluation

Outcome-only metrics hide unsafe or wasteful paths. Check the trajectories against rules:
- Required tool calls present (e.g. "must check bill status before summarising").
- Forbidden calls absent ("never call `send_email` without approval").
- Step and cost efficiency.
- Policy compliance (τ-bench-style domain policies).
- Recovery behaviour after injected tool errors.

Use deterministic trajectory checks where possible, and LLM judges (validated) for the qualitative parts.

### 8.3 Environments

Build **simulated environments**: a seeded database, fake external APIs with injectable faults, and an **LLM-simulated user** with a persona and goal (τ-bench style) for conversational agents. Every run must be reproducible from a seed, and every task must have a machine-checkable end state.

---

## 9. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Over-agentification** | An agent where a 3-step workflow would do; flaky and costly | Start with workflows; add autonomy only where evals demand it |
| **Runaway loops** | Repeated identical tool calls; cost spikes | `Budget` with loop detection; structured partial results |
| **Compounding errors** | Long tasks rarely finish | Verifiers + retries (§5.1); shorter trajectories; checkpoints |
| **Lost progress on crashes** | Hours of work restarted | Durable execution; checkpointed state (05.03) |
| **Multi-agent chaos** | Agents contradict each other or duplicate work | Hand-off contracts; single owner; shared state store; verifier role |
| **Unmeasured consistency** | Demo works; users see failures | pass^k on environment tasks; trajectory checks |
| **Opaque behaviour** | Can't explain why it did X | A trace per step (OpenTelemetry GenAI conventions); replay tooling |
| **Latency** | Agents take minutes | Parallel sub-tasks; smaller models for routine steps; streaming progress to the UI |

---

## 10. Hands-On Projects

### Project 1 — Same Task, Three Architectures: Workflow vs Agent vs Multi-Agent

**User stories**
- *As a tech lead*, I want evidence of how much autonomy our "bill impact report" feature actually needs, so that we ship the most reliable and cheapest design.

**Acceptance criteria**
1. Task: given a bill ID, produce an impact report (affected code sections, fiscal implications, related bills, a plain-language summary) with citations.
2. Three implementations sharing the same tools (05.01):
   - (a) A prompt-chaining workflow with gates.
   - (b) A single ReAct agent with a plan tool and a `Budget`.
   - (c) Orchestrator–workers (one worker per aspect).
3. On ≥ 60 bills, reports: accuracy by rubric (validated LLM judge + human spot checks), citation validity, pass^3, tokens, cost, P50/P95 latency, and failure taxonomy counts.
4. Fault injection: 10% of tool calls fail or time out; measure each architecture's recovery.
5. A recommendation, including where the hybrid (workflow with an agentic sub-step) wins.

**Step-by-step**
1. Build the tools and a seeded test environment (a database snapshot plus a fault injector).
2. Implement the three architectures using §2–3 code, with tracing.
3. Write the rubric and the judge prompt; validate on 20 human-graded reports.
4. Run each architecture 3 times per bill; compute pass^k and the other metrics.
5. Analyse the failures (MAST-style categories) and write the recommendation.

---

### Project 2 — Durable Research Agent with Budgets and Human Checkpoints

**User stories**
- *As a policy analyst*, I want to launch a long research task ("compare how neighbouring states regulate X"), close my laptop, and come back to a finished, cited brief — with the agent pausing to ask me when a scope decision is needed.

**Acceptance criteria**
1. Built on Temporal (Java or Python SDK) or LangGraph with a Postgres checkpointer. Every LLM and tool call is an activity or checkpointed node.
2. An orchestrator–workers design with per-worker budgets and loop detection. The orchestrator synthesises with citations.
3. Crash tests: kill the worker process at random points in 20 runs. All runs complete without repeating completed activities (verified by call logs).
4. Human checkpoints: the agent raises a scope question as an interrupt/signal and resumes when answered. Timeouts default to a safe choice (05.06).
5. A trace UI showing the steps, token spend per worker, and budget status in real time.

**Step-by-step**
1. Define the workflow state and graph (plan → research workers → synthesise → verify → review).
2. Implement the activities: search, fetch, extract (with structured outputs), and LLM calls with retry policies.
3. Add `Budget` checks per worker, and structured partial results on exhaustion.
4. Implement the interrupt (a Temporal signal, or a LangGraph interrupt) and the resume API.
5. Run the crash-injection tests and document the results.

---

### Project 3 — Agent Evaluation Harness with a Simulated User and pass^k

**User stories**
- *As an AI QA engineer*, I want a reproducible environment in which conversational agents are tested against policies with simulated users, so that regressions in consistency are caught before release.

**Acceptance criteria**
1. The environment: a seeded database (e.g. constituent service requests or bill-tracking subscriptions), domain tools, a written policy document, and ≥ 50 tasks with goal end states.
2. An LLM user simulator with a persona, goal, and hidden information revealed only when asked. Seeds make the runs reproducible.
3. Metrics: pass@1, pass^k (k = 1..5), policy-violation rate (deterministic checks), trajectory efficiency, and cost.
4. A CI integration: a nightly run against the main branch, with alerts when pass^3 drops by more than 5 points.
5. A comparison of two models or two prompt versions with confidence intervals.

**Step-by-step**
1. Design the database schema, the tools, and the policy. Write the tasks with the expected final database state.
2. Implement the user simulator (a system prompt with the persona and goal; stop when the goal is met or the user gives up).
3. Implement the runner: n = 5 trials per task; record the trajectories; compute `pass_hat_k`.
4. Add the trajectory rule checks (required and forbidden calls).
5. Wire the harness into CI with reports and alerts.

---

## 11. Foundational Papers & Reading (exact titles)

- Yao et al., 2023 — *ReAct: Synergizing Reasoning and Acting in Language Models*
- Wang et al., 2023 — *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models*
- Shinn et al., 2023 — *Reflexion: Language Agents with Verbal Reinforcement Learning*
- Zhou et al., 2023 — *Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models*
- Wang et al., 2023 — *A Survey on Large Language Model based Autonomous Agents*
- Wu et al., 2023 — *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*
- Hong et al., 2023 — *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework*
- Du et al., 2023 — *Improving Factuality and Reasoning in Language Models through Multiagent Debate*
- Cemri et al., 2025 — *Why Do Multi-Agent LLM Systems Fail?*
- Yao et al., 2024 — *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*
- Jimenez et al., 2023 — *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- Chen et al., 2021 — *Evaluating Large Language Models Trained on Code* (unbiased pass@k estimator)
- Anthropic Engineering — *Building effective agents* (2024); *How we built our multi-agent research system* (2025)
- OpenAI — *A practical guide to building agents* (2025)

## 12. Essential Tooling

| Tool | Role |
|---|---|
| **LangGraph** | Graph/state-machine agents with checkpointers and interrupts |
| **Temporal** (Java/Python/Go SDKs), **Restate**, **Azure Durable Functions** | Durable execution for long-running agents |
| **OpenAI Agents SDK / Claude Agent SDK / Google ADK / Spring AI** | Agent frameworks (handoffs, tools, guardrails) |
| **DSPy** | Optimising multi-step LM programs |
| **Inspect AI**, **τ-bench**, **SWE-bench harness** | Agent evaluation environments |
| **OpenTelemetry GenAI conventions**, **Langfuse / Phoenix / LangSmith** | Tracing trajectories |
