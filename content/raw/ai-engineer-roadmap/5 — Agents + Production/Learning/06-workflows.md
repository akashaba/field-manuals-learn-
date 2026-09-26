# Workflows — Master Study Guide

> **Track:** Agents + Production · **Module:** 06
> **Prerequisites:** Modules 01–05.
> **Time budget:** ~5–7 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** The user's directive: understand the architecture, not the framework. And a huge chunk of production "AI systems" that people call agents are actually **workflows** — deterministic pipelines of LLM calls and tool operations. Anthropic's ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) argues bluntly: **prefer workflows; use agents only when needed.**

Why? Workflows are cheaper, faster, more testable, more reliable. Agent autonomy is powerful but costly. The engineering wisdom: **match the design to the task's actual variability.**

**Fundamental principles you must own:**

1. **Workflow = predefined control flow.** The code decides steps; the LLM fills in outputs.
2. **Agent = dynamic control flow.** The LLM decides steps.
3. **Common workflow patterns exist** — prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
4. **Compose workflows and agents.** Real systems mix.
5. **Workflows scale better** — deterministic control flow is easy to test, parallelize, cache.
6. **Deterministic ≠ inflexible** — a good workflow adapts via routing, conditionals, and re-runs.

If you retain nothing else: **before reaching for an agent, ask if a workflow can do the job. Often it can.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Prompt Chaining

The simplest pattern: **output of step A is input to step B**. A pipeline of LLM calls.

**Example — write a blog post:**

```python
def write_blog_post(topic):
    outline = llm(f"Outline a blog post on: {topic}")
    draft = llm(f"Write the post using this outline:\n{outline}")
    edited = llm(f"Edit for clarity and tone:\n{draft}")
    title = llm(f"Suggest a compelling title for:\n{edited}")
    return {"title": title, "content": edited}
```

**When to use:**
- Task naturally decomposes into sequential sub-tasks.
- Each step's output is well-scoped and clear.

**Advantages:**
- Each step has a focused prompt → higher quality on each.
- Cheaper models can handle intermediate steps.
- Testable in isolation.
- Easy to debug — inspect each step's output.

**Guardrails to add:**
- **Validate intermediate outputs** — reject a bad outline before wasting a draft-generation call.
- **Structured outputs at boundaries** — pass JSON between steps, not free-form text.
- **Retry per step** — a failed edit doesn't lose the draft.

**Anti-pattern:** chaining 10+ LLM calls when the whole thing could be one call. Each additional call adds latency, cost, and failure surface. Chain when each step adds distinct value.

---

### 2.2 Routing

**Route** the input to a **specialized handler** based on classification.

**Example — customer support:**

```python
def route_ticket(user_message):
    category = classify(user_message, categories=[
        "billing", "technical", "account", "sales", "other"
    ])
    match category:
        case "billing":  return billing_handler(user_message)
        case "technical": return technical_handler(user_message)
        case "account":   return account_handler(user_message)
        case "sales":     return sales_handler(user_message)
        case _:           return generic_handler(user_message)
```

**When to use:**
- Different question types benefit from different prompts / tools / models.
- You want to control per-category behavior explicitly.

**Advantages:**
- Each handler is optimized for its category.
- Easy to add new categories.
- Simple to reason about.

**The classifier can be:**
- **Rules / regex** — fast but brittle.
- **Small LLM** — flexible; cheap.
- **Embedding + kNN** — best when training data is labeled.

**Multi-label routing.** A ticket may involve billing AND technical. Options:
- Route to a combined handler.
- Route to both, fuse outputs.
- Route hierarchically (main topic first, sub-topic within).

**Handoff to human.** For high-value or high-risk categories, route to a human. Route + escalate.

---

### 2.3 Parallelization

Break work into **independent** pieces, run them in parallel, combine.

**Two flavors:**

**Sectioning.** Decompose a task into sub-tasks that can each be solved independently. Combine at the end.

```python
async def analyze_document(doc):
    tasks = [
        analyze_sentiment(doc),
        extract_entities(doc),
        summarize(doc),
        classify_topic(doc),
    ]
    sentiment, entities, summary, topic = await asyncio.gather(*tasks)
    return {"sentiment": sentiment, "entities": entities, "summary": summary, "topic": topic}
```

**Voting / ensembling.** Run the same task multiple times (different prompts, temperatures, models); combine.

```python
async def robust_classify(text):
    candidates = await asyncio.gather(*[classify(text, prompt=p) for p in PROMPT_VARIANTS])
    return majority_vote(candidates)
```

**When to use:**
- Independent sub-tasks → sectioning.
- Correctness-critical, want to reduce variance → voting.

**Advantages:**
- Massively lower latency (max instead of sum).
- Higher quality (ensembling).

**Costs:**
- More LLM calls.
- Combining results is its own engineering problem.

**Tools:** `asyncio.gather` in Python, `Promise.all` in JS. Every major LLM SDK supports async.

---

### 2.4 Orchestrator-Workers

An **orchestrator LLM** breaks a task into subtasks; **worker LLMs** each handle a subtask; orchestrator synthesizes.

Unlike a pure agent, the *decomposition* is dynamic but the *execution* is delegated to workers with narrow scope.

**Example — code refactoring:**

```
Orchestrator: "Refactor this file into modules."
→ Analyzes the file, produces a plan:
   - Extract auth logic into auth.py
   - Extract db logic into db.py
   - Update imports in main.py
→ Dispatches each sub-task to a worker with the relevant slice of the file
→ Workers each edit their slice
→ Orchestrator reviews and combines
```

**When to use:**
- The task has natural sub-tasks discoverable only by inspection.
- Sub-tasks are somewhat independent (workers don't need to coordinate).

**Distinction from agent:** the orchestrator plans; workers *execute what it says*, not decide new plans. Cheaper and more predictable than a full multi-agent.

**Distinction from parallelization:** decomposition is dynamic (LLM-decided), not static.

**Anthropic's example — coder agents:** orchestrator reads the request, delegates coding tasks to worker LLMs, aggregates results.

---

### 2.5 Evaluator-Optimizer (Reflection at Workflow Scale)

One LLM produces output; a second LLM evaluates; iterate until quality is good enough.

**Example — translation refinement:**

```python
def translate_and_refine(text, target_lang):
    for i in range(3):
        translation = translate(text, target_lang)
        critique = evaluate(text, translation, criteria=[
            "accuracy", "fluency", "tone", "cultural appropriateness"
        ])
        if critique.overall_score > 0.9:
            return translation
        # Use critique to guide next attempt
        text_with_hints = augment_with_critique(text, critique)
    return translation
```

**When to use:**
- Iterative refinement measurably improves quality.
- You can articulate clear evaluation criteria.
- The evaluator can reliably assess.

**Guardrails:**
- Cap the loop (max 2–3 iterations; diminishing returns).
- Different models for actor and critic when possible (reduces self-preference bias).
- Structured critiques — dimensions, scores, actionable feedback.

**Use cases:**
- Code generation with test-run critique.
- Report writing with citation-check critique.
- Complex translations.
- Content moderation with policy-adherence critique.

---

## 3. Mental Models & Analogies

### 3.1 The "Assembly Line vs Craftsperson" Model

A **workflow** is an assembly line — stations in fixed order, each does one thing, product moves forward. Predictable. Efficient. Adds up.

An **agent** is a craftsperson — decides what to do next based on the piece, may go back and rework, may improvise. Slower per unit; better for one-of-a-kind items.

**Design question:** is your product commoditizable (repetitive with known steps) or bespoke (unique requirements)? Assembly-line for the first; craftsperson for the second.

Most "AI products" that seem novel actually turn out to be commoditizable once you look closely. Ship an assembly line if you can.

### 3.2 The "Recipe vs Improvisation" Model

- **Prompt chaining** = following a recipe step by step.
- **Routing** = choosing the right recipe from a cookbook.
- **Parallelization** = a team cooking multiple dishes at once, then plating.
- **Orchestrator-workers** = a head chef delegating stations.
- **Evaluator-optimizer** = a critic tasting each course and asking the chef to redo.

Full **agent autonomy** is a chef inventing a menu on the spot based on what's in the fridge. Sometimes worth it, often overkill.

Design your kitchen around the food you actually serve.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Everything Should Be an Agent"

The hype says agents. Reality: many tasks are workflows in disguise. Ask: does the *sequence of steps* depend on the input? If no, it's a workflow. Ship workflows where they suffice. Save agent complexity for tasks that genuinely need dynamic decision-making.

### 4.2 "Prompt Chaining Is Just an Agent With Fewer Steps"

No. In an agent, the LLM chooses each next step. In prompt chaining, YOU chose them. This distinction shows up in testability (chaining is deterministic; agent is variable) and cost (agent pays for reasoning about steps; chaining doesn't).

### 4.3 "Workflows Are Inflexible"

Not if you design them well. Routing gives per-input variation. Conditional branches let workflow adapt. Re-runs and evaluator loops add adaptivity. A well-designed workflow can handle 80% of task variance without going full agent — and the remaining 20% can fall through to an agent branch.

---

## 5. Self-Assessment Bank (Workflows)

### Questions

**Q1 (Short answer).** Define workflow vs agent in one sentence each.

**Q2 (Multiple choice).** For a task where every input goes through the same 3 steps (analyze → summarize → format), the right choice is:
- (a) An autonomous agent.
- (b) A prompt-chaining workflow.
- (c) Multi-agent system.
- (d) Reflection loop.

**Q3 (Short answer).** Describe the routing pattern and one advantage over a monolithic prompt handling all categories.

**Q4 (Multiple choice).** Parallelization via `asyncio.gather` on independent sub-tasks:
- (a) Doubles cost.
- (b) Reduces latency from sum-of-parts to max-of-parts, at slight cost overhead.
- (c) Increases latency.
- (d) Only works for GPT.

**Q5 (Short answer).** Explain the orchestrator-worker pattern and how it differs from pure multi-agent systems.

**Q6 (Multiple choice).** Evaluator-optimizer loops typically:
- (a) Improve quality significantly with unlimited iterations.
- (b) Improve quality 5-15% at 2-3× cost; diminishing returns after ~3 iterations.
- (c) Are only for classification.
- (d) Reduce cost.

**Q7 (Short answer).** Give three benefits of a workflow over an agent for a suitable task.

**Q8 (Multiple choice).** In evaluator-optimizer, using different models for actor and critic:
- (a) Doesn't matter.
- (b) Reduces self-preference bias in critique.
- (c) Slows down inference.
- (d) Increases variance.

**Q9 (Short answer).** When would a workflow with a small agent branch (workflow that falls through to an agent for edge cases) be a better design than a pure agent?

**Q10 (Multiple choice).** Anthropic's ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) advice is essentially:
- (a) Always use agents.
- (b) Prefer workflows; use agents only when task truly requires dynamic decision-making.
- (c) Use frameworks.
- (d) Don't use LLMs.

---

### Answer Key & Detailed Explanations

**A1.** **Workflow**: LLM/tool operations are orchestrated by predefined code control flow (paths decided by code). **Agent**: LLM dynamically decides the next step based on state.

**A2. (b).** Fixed 3-step sequence per input is a textbook prompt-chaining workflow. Deterministic, easy to test, cheap. An agent would over-engineer this.

**A3.** **Routing**: classify the input, then dispatch to a specialized handler (prompt/tools/model). **Advantage over monolithic**: each handler can be optimized for its category (specialized prompts, narrower tool sets, cheaper models for simple categories). Easier to add categories over time. Testable per-handler.

**A4. (b).** `asyncio.gather` fires N requests concurrently; wall time is max latency of the individual tasks, not their sum. Cost stays the same as sequential (still N LLM calls). Best pattern for embarrassingly parallel work.

**A5.** **Orchestrator-workers**: an orchestrator LLM decomposes a task into subtasks (dynamic decomposition), then dispatches each to a worker LLM with a narrow scope. Workers execute what the orchestrator said, they don't re-plan. Difference from pure multi-agent: workers are deterministic executors of the orchestrator's plan, not autonomous agents making their own decisions.

**A6. (b).** Evaluator-optimizer typically improves output quality by 5–15% at 2–3× cost per output. Diminishing returns after ~3 iterations. Great for quality-critical outputs (code, essays, translation), overkill for simple tasks.

**A7.** (1) **Testability** — deterministic control flow means unit tests. (2) **Cost** — fewer LLM calls per task; smaller models per step. (3) **Latency** — pipelines can be parallelized; no LLM overhead for reasoning about the next step. (4) **Debuggability** — inspect each step in isolation. (5) **Reliability** — deterministic; the same input produces the same behavior. Any three plus rationale.

**A8. (b).** Cross-model evaluation reduces self-preference bias — a model tends to rate its own outputs more favorably. Using a different model (or a specialized critic model) for evaluation produces more honest scores. Also, splitting actor and critic can use a cheap actor + expensive critic OR expensive actor + cheap critic depending on cost sensitivity.

**A9.** When 80% of inputs follow a predictable pattern (workflow handles them fast, cheap, reliably) but 20% are unusual cases needing dynamic decision-making. Route the 80% through the workflow; fall through to an agent for the unusual 20%. Best of both: majority get workflow speed/cost/reliability; edge cases get agent flexibility.

**A10. (b).** The Anthropic post argues explicitly: workflows are simpler and often sufficient; agents are for tasks that genuinely require dynamic decision-making. Frameworks are convenience layers, not architecture. Start simple.

---

## 6. Practice Prompts

1. **Workflow vs agent.** Take a task like "given a URL, fetch, extract, summarize, translate, tweet-ify." Implement it as: (a) sequential prompt chaining, (b) parallel where possible, (c) an autonomous agent. Compare cost, latency, and reliability.
2. **Routing.** Build a support-ticket router: LLM classifier → specialized handler per category. Compare quality to a single monolithic prompt trying to handle everything.
3. **Orchestrator-workers.** Build an "analyze and improve this codebase" task where an orchestrator identifies improvements and workers implement each one. Compare to a single autonomous agent doing the whole thing.
4. **Evaluator-optimizer.** Implement code generation with a critic that runs pytest. Iterate up to 3 times. Compare final pass rate to no-critique baseline.
5. **Hybrid pipeline.** Design a system: routing → per-category prompt chaining → fallback to agent for uncategorized. Test on 50 mixed inputs.

---

## 7. References

- Anthropic, ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) (2024).
- LangGraph patterns docs.
- Chase & Chen, ["State of Agents"](https://blog.langchain.dev) blog series.
- Andrew Ng, ["The batch"](https://www.deeplearning.ai/the-batch/) coverage of workflows vs agents.
