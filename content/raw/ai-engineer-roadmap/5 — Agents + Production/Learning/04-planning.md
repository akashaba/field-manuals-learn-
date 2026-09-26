# Planning — Master Study Guide

> **Track:** Agents + Production · **Module:** 04
> **Prerequisites:** Modules 01–03.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Reactive agents (pure ReAct) work well for short, local decisions but stumble on **multi-step, interconnected tasks**: "research this topic across 10 sources and produce a report" or "refactor this codebase into microservices." Planning is the technique of **thinking ahead** — deciding *what* to do, not just doing *the next thing*.

Planning is what separates "helpful assistant" from "capable agent." Every high-performing agentic system (Deep Research, Devin, Claude Code, Cursor Composer) plans in some form.

**Fundamental principles you must own:**

1. **Planning ≠ execution.** Two distinct phases; often use different models or contexts.
2. **Explicit plans are debuggable.** You can inspect them, edit them, replan when they fail.
3. **Plans should adapt.** Static plans fail; re-planning on new information is the difference between fragile and robust.
4. **Different tasks need different plan structures** — linear, tree, graph, hierarchical.
5. **The right amount of planning depends on the task's coupling.** Loose sub-tasks: plan up-front. Tight coupling: interleaved planning and execution.
6. **Reflection = planning applied to output**, not action. A distinct pattern worth its own place in your toolkit.

If you retain nothing else: **planning is the discipline of thinking farther ahead than the next tool call. Its cost buys you robustness on complex tasks.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Plan-and-Execute

The classic planning pattern (Wang et al., 2023 + related work):

**1. Planning phase.** An LLM (often the strongest available) produces a **plan** — a structured list of steps.

**2. Execution phase.** Each step is executed, often by a smaller/cheaper model with a narrower toolset.

**3. Synthesis.** Results are combined into a final answer.

```python
def plan_and_execute(query):
    plan = generate_plan(query)   # ← strong model
    results = []
    for step in plan:
        result = execute_step(step, prior_results=results)  # ← cheaper model
        results.append(result)
    return synthesize(query, plan, results)
```

**Plan output format.** Structured (JSON) beats free-form:

```json
{
  "steps": [
    {"id": 1, "description": "Search for recent papers on X", "tool": "search_web", "depends_on": []},
    {"id": 2, "description": "For each top-5 paper, fetch abstract", "tool": "fetch_url", "depends_on": [1]},
    {"id": 3, "description": "Summarize findings", "tool": "llm", "depends_on": [2]}
  ]
}
```

Explicit dependencies enable **parallel execution** of independent steps.

**Benefits:**
- **Debuggability** — you can inspect the plan before running.
- **Cost** — execution can use cheaper models per step.
- **Parallelism** — independent steps run in parallel.
- **Determinism** — same query → similar plan → similar cost.

**Drawbacks:**
- **Rigidity** — the plan may miss context that only emerges during execution.
- **Fixed depth** — hard to expand a step into sub-steps.

Fixes: **replanning** (see 2.4), and **plan tree structures** (see 2.3).

---

### 2.2 ReAct vs Plan-and-Execute — When to Use What

| Situation | Prefer |
|-----------|--------|
| Short task (2–5 steps) | ReAct |
| Long task (10+ steps) | Plan-and-Execute |
| Steps depend on prior results | ReAct (interleaved planning) |
| Steps mostly independent | Plan-and-Execute (parallelizable) |
| Uncertain what steps are needed | ReAct with periodic replanning |
| Predictable task structure | Plan-and-Execute |
| Cost-critical | Plan-and-Execute (cheaper execution model) |
| Adaptivity-critical | ReAct |
| Debuggability-critical | Plan-and-Execute (plan is inspectable) |

**Hybrid: ReWOO** (Reasoning WithOut Observations, Xu et al., 2023) — plan produces steps with **placeholder references** ("use result of step 1"); execution fills in. Reduces LLM turns compared to pure ReAct while retaining some adaptivity.

---

### 2.3 Task Decomposition and Tree Search

For genuinely complex tasks, a linear plan is insufficient. **Hierarchical decomposition**: break a goal into sub-goals, sub-goals into sub-sub-goals, and so on.

**Example — "Refactor this monolith into microservices":**

```
Level 0: Refactor the monolith
├── Level 1a: Identify service boundaries
│    ├── Analyze module dependencies
│    ├── Interview stakeholders on domain boundaries
│    └── Propose service map
├── Level 1b: Extract shared utilities
│    ├── Identify cross-cutting concerns
│    ├── Move to a common library
│    └── Update imports
├── Level 1c: Implement each service
│    ├── Service A: (recurses)
│    ├── Service B: (recurses)
│    └── ...
└── Level 1d: Migrate deployment
```

Each level is a plan; each leaf is an executable action.

**Recursive agent invocation.** An agent handles the top level; whenever it hits a sub-goal, it recursively invokes a child agent (possibly itself) with a narrower scope. Common in coding agents.

**Tree-of-Thoughts (ToT)** (Yao et al., 2023) — for a hard reasoning problem, generate multiple candidate next-thoughts at each step; score each; keep top-K; recurse. Explores a tree of reasoning paths rather than a single line.

```python
def tot_search(problem, depth, breadth):
    frontier = [initial_thought(problem)]
    for step in range(depth):
        candidates = []
        for state in frontier:
            children = generate_next_thoughts(state, k=breadth)
            candidates.extend(children)
        scores = [evaluate(c) for c in candidates]
        frontier = top_k(candidates, scores, k=breadth)
    return best(frontier)
```

Cost is high (breadth × depth LLM calls) but works well on games, math, planning problems.

**Monte Carlo Tree Search (MCTS)** — for problems with a clear reward signal, use classical MCTS with LLM as the rollout policy. Emerging pattern in code agents.

---

### 2.4 Replanning and Adaptivity

Plans meet reality. Results, errors, and unexpected findings mean the original plan may not fit anymore. **Replanning** patterns:

**1. Fixed-cadence replanning.** After every N steps, revisit the plan.

```python
def adaptive_plan(query, replan_every=3):
    plan = generate_plan(query)
    executed = []
    while remaining_steps(plan):
        for step in next_batch(plan, replan_every):
            result = execute(step)
            executed.append((step, result))
        # Replan based on what we've learned
        plan = revise_plan(query, plan, executed)
    return synthesize(query, executed)
```

**2. On-error replanning.** When a step fails or returns unexpected results, re-plan the remaining steps.

**3. Emergent-goal replanning.** When execution reveals a new sub-goal (an unexpected question, a needed clarification), inject it into the plan.

**4. Dynamic step expansion.** Any step in the plan can be *expanded* into sub-steps when it's too coarse. The plan grows during execution.

**Trade-off.** Each replan is another LLM call; frequent replanning is expensive. Balance:
- **Stable tasks** — plan once, execute, minimal replanning.
- **Exploratory tasks** — replan often; the plan is a hypothesis.

**Reflexion** (Shinn et al., 2023) — after each attempt, an LLM reflects on failure modes and writes "lessons" that inform the next attempt. Effectively a memory of what didn't work.

---

### 2.5 Reflection: Planning Applied to Output

**Reflection** is planning-in-reverse: the agent generates output, then critiques its own work, then revises.

**Basic reflection loop:**

```python
def reflect(query, max_revisions=3):
    draft = generate(query)
    for _ in range(max_revisions):
        critique = critique_draft(query, draft)
        if is_acceptable(critique):
            return draft
        draft = revise(query, draft, critique)
    return draft
```

**Design patterns:**

**Actor-Critic split.** Two roles: **actor** produces; **critic** evaluates. Sometimes the same model with different prompts; sometimes different models (a cheap actor, an expensive critic; or vice versa).

**Structured critique.** The critique is a checklist or rubric, not free-form:

```json
{
  "correctness": {"pass": false, "issue": "Cited source [3] but claim contradicts it."},
  "completeness": {"pass": true},
  "citations": {"pass": true},
  "style": {"pass": false, "issue": "Overly verbose."}
}
```

**Reflection converges (usually) in 2–3 iterations.** More revisions typically don't help — diminishing returns.

**Where reflection shines:**
- **Code generation** — critic runs tests / lints; actor fixes.
- **Report writing** — critic checks structure, argument flow, citations.
- **Reasoning problems** — critic verifies logic.

**Where it doesn't help much:**
- Simple factual Q&A.
- Tasks where the correct answer is unambiguous the first time.

**Reflection + tools = self-verification.** Combine reflection with executable checks: run tests, run code, search for citations. Grounds the critic in real evidence rather than the critic's own judgment.

**Constitutional AI** and **RLAIF** are related — using an AI to critique AI outputs against principles. Same skeleton; applied at training vs inference time.

---

## 3. Mental Models & Analogies

### 3.1 The "Project Manager" Model

An agent that plans is a **project manager**:

- **Scope the project** = plan generation.
- **Break into milestones** = task decomposition.
- **Assign to team members** = tool/step execution.
- **Check-ins and updates** = periodic replanning.
- **Retros** = reflection.

A PM who plans nothing is a chaos monkey. A PM who plans everything up front and never revisits is Waterfall in a world of surprises. The good PM: **plan enough to know direction; adapt as reality speaks; retro to learn.**

Agents follow the same discipline.

### 3.2 The "Chess Engine" Model (Tree Search)

Reactive agents play **fast chess** — best move given the current position, no lookahead.

Plan-and-execute plays **regular chess** — decides on a plan of attack, executes.

Tree-of-Thoughts / MCTS play **serious chess** — explore many candidate move sequences, score positions, pick the best branch.

Real chess engines do all three by phase:
- Opening: known-plan lookup (like a workflow).
- Middlegame: search + evaluation (like ToT).
- Endgame: computed table (like a deterministic function).

Real agents likewise mix. Don't get religious about one planning style — use what fits.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Just Plan Everything Up Front"

Static plans fail on discovery-heavy tasks. If you learn something in step 3 that invalidates step 5, a fixed plan wastes steps 5–N. Design for **adaptivity**: fixed-cadence replanning, on-error replanning, or interleaved plan-execute (ReAct).

### 4.2 "Reflection Loops Solve Everything"

They usually help modestly (5–15% quality lift) at 2–3× cost. On simple factual tasks, reflection adds cost with no gain. Test on your task before enabling by default. **Diminishing returns after 2–3 revisions.**

### 4.3 "Planning Makes the Agent Deterministic"

It reduces variance but doesn't eliminate it. Plans are LLM outputs — different runs produce slightly different plans. Combine with structured output (JSON schemas), few-shot examples, and temperature=0 for maximum determinism, but never expect 100% reproducibility.

---

## 5. Self-Assessment Bank (Planning)

### Questions

**Q1 (Short answer).** Contrast Plan-and-Execute and ReAct in one paragraph.

**Q2 (Multiple choice).** For a task with 20 independent sub-tasks that can each be completed in isolation, the best pattern is:
- (a) Sequential ReAct.
- (b) Plan-and-Execute with parallel step execution.
- (c) Reflection loop.
- (d) One-shot generation.

**Q3 (Short answer).** Give three benefits of an explicit plan (as a JSON structure) over an implicit ReAct trace.

**Q4 (Multiple choice).** Tree-of-Thoughts:
- (a) Same as ReAct.
- (b) Explores multiple reasoning branches in parallel, scores each, keeps top-k. Higher cost, better on hard reasoning.
- (c) Only for image generation.
- (d) Doesn't require an LLM.

**Q5 (Short answer).** Describe fixed-cadence replanning and on-error replanning.

**Q6 (Multiple choice).** A reflection loop typically improves quality by:
- (a) 50%+
- (b) 5-15% at 2-3× the cost.
- (c) No effect.
- (d) Only on classification tasks.

**Q7 (Short answer).** For a monolithic-to-microservices refactoring task, why would hierarchical (recursive) decomposition beat a single flat plan?

**Q8 (Multiple choice).** ReWOO (Reasoning WithOut Observations) uses:
- (a) No LLM.
- (b) A plan with placeholder references between steps, filled in during execution — reduces LLM turns vs ReAct.
- (c) Reflection only.
- (d) Vector search.

**Q9 (Short answer).** Give one type of task where reflection loops help a lot, and one where they don't.

**Q10 (Multiple choice).** For a task with 3–5 short interdependent steps, the best planning pattern is usually:
- (a) Sequential ReAct (planning is implicit in each turn).
- (b) Plan-and-execute.
- (c) MCTS.
- (d) Best-of-N.

---

### Answer Key & Detailed Explanations

**A1.** **ReAct** interleaves reasoning and acting: at each step the LLM chooses the next action based on the current state. Adaptive but many LLM turns. **Plan-and-Execute** separates planning from execution: an LLM first generates the full plan; then each step is executed (often by a smaller model). Less adaptive to surprises during execution, but debuggable (plan is inspectable), parallelizable (independent steps run concurrently), and cheaper (execution can use smaller models).

**A2. (b).** With independent sub-tasks, Plan-and-Execute plus parallel execution is the fit — plan generates 20 steps; executor runs them concurrently; results synthesized. Sequential ReAct would run them one at a time, wasting time.

**A3.** (1) **Debuggability** — you can inspect the plan before running; humans can review. (2) **Parallelism** — explicit dependencies enable concurrent execution of independent steps. (3) **Cost** — execution can use cheaper models per step; only planning needs a strong model. (4) **Reproducibility** — plan is a structured artifact you can log, diff, and replay. (5) **Human-in-the-loop** — plan can be reviewed / edited before execution. Any three plus rationale.

**A4. (b).** Tree-of-Thoughts generates multiple candidate next-thoughts at each step, scores them (via LLM or heuristic), keeps top-k, and recurses. Explores a tree of reasoning paths rather than one line. Costs breadth × depth times a linear pass but works well on games, math, and planning problems.

**A5.** **Fixed-cadence replanning**: after every N steps, revisit the plan — the LLM sees what's been done and what's left, and may revise remaining steps. **On-error replanning**: when a step fails or returns unexpected results, re-plan the remaining steps. Both add adaptivity to plan-and-execute; both cost extra LLM calls.

**A6. (b).** Reflection typically improves output quality by 5-15% on tasks where the LLM benefits from critique (code, essays, complex reasoning). Cost roughly triples (generate + critique + revise). Diminishing returns after 2–3 iterations. On simple factual tasks, no benefit.

**A7.** The refactoring has natural hierarchy: identifying boundaries, extracting utilities, implementing each service, migrating deployment — each of which decomposes further. A single flat plan would either be too coarse (steps are too big to execute) or too long (impractical to plan 500 leaf-level steps up front). Hierarchical / recursive decomposition lets each level plan its own scope and delegate concrete work down.

**A8. (b).** ReWOO's planner produces a plan where step N can reference "the result of step M" symbolically. The executor fills in placeholders as steps complete. Reduces total LLM turns compared to full ReAct (planner is called once) while preserving some structural adaptivity.

**A9.** **Helps:** code generation (critic runs tests / lints; actor fixes) — measurable quality gain. Also report writing, complex reasoning. **Doesn't help:** simple factual Q&A ("what's the capital of France") — first answer is correct; critique adds cost without change.

**A10. (a).** Short (3–5 step) sequences with interdependence favor ReAct — each step's decision uses the previous step's result. Full plan-and-execute would over-engineer; the plan and the trace end up equivalent, and plan-and-execute pays extra for the initial planning call.

---

## 6. Practice Prompts

1. **Plan-and-Execute.** Build a research agent using plan-and-execute for a "compare three products" query. Log the plan and execute steps. Compare quality and cost to sequential ReAct.
2. **Parallel plan execution.** For a plan with 5 independent steps, execute them in parallel with `asyncio.gather`. Measure latency vs sequential.
3. **Reflection.** Add a code-writing agent with a critic that runs `pytest` and reports failures. Iterate until tests pass or max 3 revisions.
4. **Replan on error.** In a plan-and-execute agent, when a step returns an unexpected result, call the planner again with the trace and let it produce a revised plan.
5. **Tree-of-Thoughts.** Implement ToT for a game-of-24 style problem. Compare accuracy vs single-shot reasoning.

---

## 7. References

- Wang et al., ["Plan-and-Solve Prompting"](https://arxiv.org/abs/2305.04091) (2023).
- Xu et al., ["ReWOO: Decoupling Reasoning from Observations"](https://arxiv.org/abs/2305.18323) (2023).
- Yao et al., ["Tree of Thoughts"](https://arxiv.org/abs/2305.10601) (2023).
- Shinn et al., ["Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366) (2023).
- Wei et al., ["Chain-of-Thought Prompting"](https://arxiv.org/abs/2201.11903) (2022).
- LangGraph documentation on plan-and-execute agents.
