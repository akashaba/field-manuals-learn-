# Agent Evaluation — Master Study Guide

> **Track:** Agents + Production · **Module:** 12
> **Prerequisites:** Modules 01–11 + Month 4 Module 18 (RAG eval).
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Agents amplify LLM behavior across many steps. A model that's 95% accurate per turn becomes an agent that's $0.95^{10} \approx 60\%$ accurate over 10 steps — even before you factor in tool errors. **Evaluation is what tells you if your agent works reliably enough to ship, and what tells you when a change makes it better or worse.**

Agent eval is harder than LLM eval because:

- The output is a **trajectory** (many steps), not a single response.
- The right answer often requires **checking intermediate reasoning + final output + tool usage**.
- Cost, latency, and step count are as important as correctness.
- LLM randomness compounds across steps → high variance across runs.
- Evaluations require designing task suites, ground-truth traces, and grading rubrics.

**Fundamental principles you must own:**

1. **Task-level** (did the agent complete the task?), **trace-level** (was the process reasonable?), and **step-level** (was each decision correct?) all matter.
2. **You need eval datasets** — hand-curated tasks with expected outcomes.
3. **LLM-as-judge is the pragmatic default** for open-ended tasks; use with calibration.
4. **Regression tests in CI** — every code change re-runs the eval suite.
5. **Production monitoring** picks up what offline eval misses.
6. **Cost, latency, iteration count are metrics too** — track alongside quality.

If you retain nothing else: **agents without evals are rumors. Design your eval suite before you scale your agent.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Levels of Agent Evaluation

**Task-level (outcome).** Did the agent produce the correct final answer / accomplish the task? Binary or graded.

Examples:
- Q&A: is the answer factually correct?
- Coding: do the generated tests pass?
- Automation: was the right email sent to the right person?

**Trace-level (process).** Was the reasoning path reasonable? Correct answer via lucky guess ≠ good agent.

Examples:
- Did the agent take the shortest reasonable path?
- Did it use appropriate tools?
- Did it avoid unnecessary detours?
- Did it self-correct on tool errors?

**Step-level (decisions).** For each step, was the decision correct?

Examples:
- Was the right tool selected?
- Were arguments valid?
- Did the LLM correctly interpret the previous tool's result?

**Meta-metrics.** Cost, latency, step count, error rate, refusal rate, human intervention rate. These are quality signals even when the task succeeded.

**Which to use:**
- **Task-level always** — bottom line.
- **Trace-level when process matters** — regulated domains, safety, reproducibility.
- **Step-level when debugging or optimizing tool selection**.
- **Meta-metrics for production monitoring**.

---

### 2.2 Building an Agent Eval Dataset

An eval dataset is a set of **(task, expected outcome, optional grading criteria)** triples.

**Sources:**

- **Hand-curated seed set.** For your first version, 20–50 hand-written tasks covering easy/hard, common/edge cases.
- **Real user queries.** Sample from production traffic (with consent / anonymization). Highest signal.
- **Generated with LLMs.** Given a corpus / capability list, prompt an LLM to invent test tasks. Requires human review.
- **Public benchmarks.** SWE-bench (coding agents), WebArena (web navigation), τ-bench (retail agents), AgentBench (general), GAIA (research assistant). Domain-general.

**Task specification:**

```json
{
  "id": "task_042",
  "task": "Find the current temperature in Helena, MT and compare it to last week's average.",
  "category": "research_easy",
  "expected_final_answer_pattern": {"has_current_temp": true, "has_last_week_comparison": true},
  "expected_tool_calls": ["get_weather", "get_historical_weather"],
  "forbidden_tool_calls": ["send_email"],
  "max_cost": 0.10,
  "max_latency_seconds": 30,
  "must_cite_sources": true
}
```

**Difficulty and categories.** Tag tasks:
- **Difficulty**: easy / medium / hard.
- **Category**: research / write / decide / automate.
- **Coverage**: happy path, edge cases, error recovery, ambiguity.

Slice metrics by tag; catch specific-slice regressions.

**Golden traces (optional).** For step-level eval, hand-craft a "reference trace" — the ideal step sequence. Compare produced traces to golden.

**Size:** 50–200 for iteration; 500–2000 for CI regression. Don't obsess over volume; quality of coverage matters more.

---

### 2.3 Grading Strategies

**Exact match / regex.** For extractive tasks where a specific string is the answer. Cheap, deterministic. Limited to well-scoped outputs.

**Semantic similarity.** Embed the produced answer and the reference; cosine similarity. Rough; misses factual errors when the wording matches.

**Rule-based validators.** For structured outputs — did the JSON have the required fields, did the citation appear, did the tool call use the right endpoint. Deterministic and cheap.

**Programmatic checks.** For coding agents: run the tests. For automation agents: check the desired state was reached (email sent, ticket created).

**LLM-as-judge.** A stronger LLM evaluates the agent's output. Most flexible; most expensive; needs calibration.

**Human evaluation.** Gold standard for subjective / novel tasks. Slow; expensive; irreplaceable for the toughest cases.

**Combining:**

For each task, use the cheapest grader that works. Rule-based first; regex second; LLM-as-judge for open-ended; human for the hardest 5% you sample manually.

**Rubric-based LLM-as-judge:**

```python
def grade_task(task, response, criteria):
    prompt = f"""Grade this agent's response on the following criteria:
    Task: {task}
    Criteria: {json.dumps(criteria)}

    Response: {response}

    For each criterion, return "pass", "fail", or "partial" with 1-sentence reasoning.
    Return JSON: {{"criterion_name": {{"verdict": ..., "reason": ...}}, ...}}
    """
    return json.loads(llm(prompt, model="claude-opus"))
```

Combine per-criterion verdicts into a task-level score.

**Calibration** — regularly (weekly) sample ~30 tasks and have a human grader score them. Compute agreement between LLM judge and human. If <70%, revise the judge prompt or upgrade the judge model.

---

### 2.4 Trace-Level and Step-Level Eval

**Trace scoring** measures the process, not just the outcome.

**Trace metrics:**

- **Step count** — did the agent finish in reasonable steps? (Low = good.)
- **Tool-selection accuracy** — did each step choose the "right" tool? (Requires golden traces or expert review.)
- **Argument accuracy** — did each tool call have correct arguments?
- **Error-recovery success** — when a tool failed, did the agent recover?
- **Redundancy** — did the agent repeat itself unnecessarily?
- **Ambiguity handling** — did the agent ask for clarification when needed?

**Tools:**

**LangSmith.** LangChain's tracing platform; captures every step and lets you build evals over them.

**Braintrust.** Eval + tracing platform; strong on LLM-as-judge patterns.

**Arize Phoenix.** Open-source; supports OpenTelemetry.

**Weights & Biases Weave.** General-purpose LLM tracing.

**Roll your own.** For small teams, JSONL trace logs + a Python analyzer is enough.

**Step-level judging (LLM as judge on individual steps):**

```python
def judge_step(step, task_context, prior_steps):
    prompt = f"""In the context of this task and prior steps, was the following step's tool
    call appropriate?

    Task: {task_context}
    Prior steps: {prior_steps}
    This step's tool call: {step.tool_name}({step.args})

    Reply "appropriate", "wrong_tool", "bad_args", or "unnecessary" with 1-sentence reason."""
```

Great for diagnosing where an agent goes off the rails, especially when task-level results are inconsistent.

---

### 2.5 CI Regression, Production Monitoring, and Continuous Eval

**CI regression testing.** Every PR runs the eval suite:

```yaml
# .github/workflows/eval.yml
- name: Run agent eval
  run: python -m eval.run_suite --dataset eval/dataset.jsonl --output eval/results.json

- name: Assert regression thresholds
  run: python -m eval.assert_baseline --results eval/results.json --baseline eval/baseline.json --tolerance 0.02
```

**Baseline metrics.** Store `baseline.json` in the repo: task-level pass rate, avg cost, avg latency, step count p95. CI fails a merge if any regresses beyond tolerance.

**Slice metrics.** Aggregate not just averages but per-category:

```
Overall:           92% pass (baseline 91%, ok)
- easy:             98%
- medium:           89%
- hard:             75% (baseline 80%, REGRESSION)
```

Slice regression is more informative than overall.

**Production monitoring.** Offline eval catches known cases. Production sees the real distribution.

**Log every agent run** with:
- Task ID (if applicable).
- User ID (with privacy).
- Total steps.
- Total cost.
- Total latency.
- Final answer.
- Full trace.
- User feedback (thumbs, follow-ups).

**Sample for LLM-as-judge review.** 1–5% of production runs graded automatically per week. Track:
- Task success rate.
- Faithfulness / correctness.
- Rate of refusal.
- Rate of human escalation.

**Alerting.**
- Success rate drops >5% week-over-week → investigate.
- Cost per run spikes → check for infinite loops.
- Latency p95 breaches SLA → scale, cache, or route.

**Trace-mining.** Cluster failures: which task categories fail most? Which tool errors are most common? What's the modal failure mode this month?

**Human-in-the-loop feedback.** Users who thumbs-down (or engage a support flow) are your highest-signal data source. Route those traces to a review queue; add failing cases to your eval dataset.

**Continuous learning loop:**

1. Deploy → collect prod traces.
2. Sample or user-flag failures → analyze.
3. Add representative failures to eval dataset.
4. Iterate on prompt / tool / retrieval to fix.
5. CI regression catches if the fix breaks other cases.
6. Deploy → repeat.

This is how agents get better over months. Without it: agents stay at initial quality forever.

---

## 3. Mental Models & Analogies

### 3.1 The "Long-Distance Runner" Model

Evaluating a single LLM response is like timing a 100m sprint — one number matters, over quickly.

Evaluating an agent is like grading a marathon:
- **Task-level** — did they finish?
- **Trace-level** — did they follow a reasonable route?
- **Step-level** — did they run efficiently, hydrate, pace correctly?
- **Meta-metrics** — total time, calories burned, injuries.

A runner who finishes by cheating (took a car for 5 km) fails the trace check even if their task-level ("finished the marathon") passes.

Similarly, an agent that gets the right answer via 20 wasted tool calls has task success but trace failure. That's still a failure by trace standards. Design evals to catch this.

### 3.2 The "Airline Ops Center" Model

Airlines don't just measure "did the plane land safely" — they monitor:
- Flight completion rate (task).
- Fuel efficiency (cost).
- On-time performance (latency).
- Route adherence (trace).
- Turbulence handling (error recovery).
- Fleet-wide anomalies (production monitoring).

Every flight generates data; ops teams sample, aggregate, and act. An airline that only checked "did the plane land safely" would miss slow degradation.

Agent evals in production are ops-center dashboards for your agent fleet. Design accordingly: multiple metrics, sliced views, alerts, and a feedback loop from anomalies back into training data.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Task Succeeded — the Agent Is Good"

Single-run task success is high variance. Same agent, same input, different day → different result. Run each eval task **multiple times** (say 3–5), report success rate + confidence interval. A 100% one-run "success" that's actually 40% success on repeated runs is a landmine.

### 4.2 "I'll Evaluate When It's Ready"

Agents without evals grow entangled quickly. Every prompt tweak becomes "did I break anything?" — impossible to know. Build a 20-task eval on day one. Extend as you scale. It's cheap insurance and doubles as documentation of what the agent should do.

### 4.3 "LLM-as-Judge Is Ground Truth"

Same lesson from Month 4: LLM judges have biases (position, verbosity, style, self-preference), inconsistency (~5–15% variance), and can miss subtle failures. Use them for scale; calibrate against human graders regularly; be skeptical of small deltas.

---

## 5. Self-Assessment Bank (Agent Evaluation)

### Questions

**Q1 (Short answer).** Name the three levels of agent evaluation and one meta-metric.

**Q2 (Multiple choice).** For an agent that gets the right answer via 20 wasted tool calls, the correct interpretation is:
- (a) Task success; ship it.
- (b) Task success but trace-level failure — investigate why the agent took so long.
- (c) Fatal error.
- (d) Rewrite the model.

**Q3 (Short answer).** How does an agent's success rate typically degrade across steps? Give the intuition.

**Q4 (Multiple choice).** Public benchmarks for agents include:
- (a) MTEB
- (b) SWE-bench (coding), WebArena (web), τ-bench (retail), GAIA (research).
- (c) ImageNet.
- (d) None exist yet.

**Q5 (Short answer).** Give three fields you'd include in an eval-task specification.

**Q6 (Multiple choice).** Programmatic grading (running the produced code's tests) is preferable to LLM-as-judge when:
- (a) Never.
- (b) The task has a machine-checkable outcome; programmatic is cheap and deterministic.
- (c) Only for math.
- (d) The tests are unreliable.

**Q7 (Short answer).** Why should you run each eval task multiple times before comparing agent versions?

**Q8 (Multiple choice).** For CI regression testing:
- (a) Run eval on every push; fail merges when metrics regress beyond a threshold.
- (b) Skip eval to speed up CI.
- (c) Only run when model changes.
- (d) Only after deploy.

**Q9 (Short answer).** Describe how you'd design a feedback loop from production failures to your eval suite.

**Q10 (Multiple choice).** For a production agent, the metric MOST likely to detect a runaway-cost bug is:
- (a) Task success rate.
- (b) Cost per run (with alert on p95 or mean anomaly).
- (c) Answer relevance.
- (d) Model version.

---

### Answer Key & Detailed Explanations

**A1.** **Task-level** — did the agent produce the right final answer / accomplish the task? **Trace-level** — was the process reasonable (right tools, right sequence, appropriate handling of errors)? **Step-level** — for each step, was the decision correct? **Meta-metrics** — cost, latency, step count, refusal rate, error rate, human intervention rate.

**A2. (b).** Task success without trace-level failure signals waste, unnecessary cost, high latency, and often reflects a bug or suboptimal design (repeated identical tool calls, wandering plan). Fix by inspecting the trace, tightening the prompt, or adding loop detection.

**A3.** If each step is right with probability $p$, an agent needs $N$ correct steps: overall $p^N$. For $p = 0.95, N = 10$: $\approx 0.60$. Compounding errors mean small per-step improvements have outsized effects on task-level success. This is why tool design, prompt clarity, and step-level correctness matter — they all raise $p$.

**A4. (b).** Modern agent benchmarks include: SWE-bench (fix real GitHub issues), WebArena (web navigation), τ-bench (retail agent tasks), AgentBench (general), GAIA (research assistant), MLE-bench (ML engineering), OSWorld (desktop control), and others. Emerging area with active benchmark development.

**A5.** Any three: `task` (the input), `expected_final_answer` or `pattern`, `expected_tool_calls` (or forbidden calls), `max_cost`, `max_latency`, `must_cite_sources`, `category`, `difficulty`, `grading_criteria`, `golden_trace` (optional).

**A6. (b).** Programmatic grading is deterministic, cheap, and machine-checkable — perfect for coding agents (run tests), automation (verify DB state), or structured extraction (schema check). LLM-as-judge is for open-ended tasks where programmatic checks are infeasible.

**A7.** LLM randomness means the same agent produces different traces on repeated runs. A single-run "success" that's actually 40% success rate is a landmine. Run each task N times (say 3–5); report success rate + variance. Compare distributions, not single points. This detects regressions or improvements that are more than noise.

**A8. (a).** Every PR that touches prompts, tools, retrieval, or the agent loop runs the eval suite in CI. Compare to baseline; fail if regression exceeds tolerance (say -0.02 on pass rate). Prevents silent quality drops that would only be caught in production.

**A9.** (1) Log every production run with trace, user feedback, and final answer. (2) Sample failures — user thumbs-downs, escalations, errors. (3) Human review of a subset. (4) Add the representative failure tasks to your eval dataset. (5) Iterate on prompts / tools / retrieval; verify the failure is fixed in eval. (6) CI regression catches breakage of other cases. (7) Ship; repeat. This is how agents keep improving over months.

**A10. (b).** Cost per run instrumentation with anomaly alerts catches runaway loops before they explode. Task success (a) is quality, not cost. Answer relevance (c) same. Model version (d) is a config field.

---

## 6. Practice Prompts

1. **20-task seed set.** For your Build #1 agent, hand-write 20 eval tasks covering easy/medium/hard and different task categories. Include expected outcomes and grading criteria.
2. **Programmatic grader.** For a coding agent, hook up a grader that runs `pytest` on the generated code and returns pass/fail.
3. **LLM-as-judge with calibration.** Grade 30 tasks with LLM-as-judge; grade the same 30 yourself; compute agreement. If < 70%, revise the judge prompt.
4. **CI regression.** Add an eval regression test to your CI; make a "hurting" change; verify CI catches it.
5. **Production trace mining.** Log 100 production runs. Cluster failures (LLM-as-judge tags them). Identify the top 3 failure modes; add them to your eval set.

---

## 7. References

- Yao et al., ["τ-bench"](https://arxiv.org/abs/2406.12045) (2024) — retail agent benchmark.
- Jimenez et al., ["SWE-bench"](https://arxiv.org/abs/2310.06770) (2023) — coding agent benchmark.
- Zhou et al., ["WebArena"](https://arxiv.org/abs/2307.13854) (2023) — web navigation.
- Mialon et al., ["GAIA"](https://arxiv.org/abs/2311.12983) (2023) — research assistant.
- LangSmith / Braintrust / Arize Phoenix docs on agent eval.
- Anthropic, ["Agent Evaluations"](https://www.anthropic.com/research) posts.
