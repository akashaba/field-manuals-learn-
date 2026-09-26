# Prompt Engineering — Master Study Guide

> **Track:** LLM Engineering · **Module:** 04
> **Prerequisites:** Modules 01–03.
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Prompt engineering is the **highest-leverage skill in LLM engineering.** The same model, given a bad vs excellent prompt, can produce a garbage vs production-ready response. This is real engineering work — repeatable, measurable, iterable — not "prompt whispering."

**What "prompt engineering" actually is:**

- Writing clear, unambiguous instructions.
- Providing worked examples (few-shot).
- Structuring input so the model can parse it.
- Steering reasoning with chain-of-thought or scratchpads.
- Constraining output format.
- Iterating based on measured failures.

**Fundamental principles you must own:**

1. **Clarity beats cleverness.** Verbose, explicit prompts outperform "clever" one-liners.
2. **Examples > descriptions.** Show, don't tell.
3. **Structure the input.** Use delimiters (XML tags, markdown, JSON) so the model knows what's what.
4. **System prompts steer role.** User prompts drive the task.
5. **Iterate against a test set**, not vibes. Small evals catch regressions.
6. **Prompt engineering has a ceiling.** If a stronger model or fine-tuning is needed, prompts alone won't get you there.

If you retain nothing else: **treat prompts like code — version them, test them, review them, and never trust "it worked on my machine."**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Anatomy of a Good Prompt

A strong prompt has (roughly):

1. **Role / persona.** "You are a senior technical writer for a legal firm."
2. **Task statement.** "Rewrite the following contract clause in plain English."
3. **Context / background.** "The audience is small-business owners."
4. **Constraints.** "Max 100 words. No jargon. Use active voice."
5. **Format specification.** "Return valid JSON with fields `plain_text` and `key_terms`."
6. **Examples (few-shot).** 2–5 (input, output) pairs.
7. **The actual input.** Delimited clearly.

**Bad prompt:**

> "Simplify this contract."

**Better prompt:**

```
You are a senior technical writer for a legal firm. Your job is to rewrite legal
clauses in plain English for small-business owners.

Rules:
- Maximum 100 words.
- No legalese ("hereinafter", "whereas").
- Active voice.
- Preserve every legal obligation from the original.

Return JSON with this exact schema:
{
  "plain_text": "the rewritten clause",
  "key_terms": ["list of important terms to explain further"]
}

Here is the clause:
<clause>
{{ clause_text }}
</clause>
```

**Delimiters matter.** Wrapping the input in a clear tag (`<clause>...</clause>`, `<article>...</article>`, or triple backticks) prevents prompt injection from user content bleeding into instructions.

**System vs user split:**
- **System**: role, rules, tone, format, safety guardrails. Stable across sessions.
- **User**: the specific task and input for this turn.

Reason to keep them split: the system prompt gets cached (prompt caching); it also gets less influenced by conversational drift.

---

### 2.2 Zero-Shot, Few-Shot, and Chain-of-Thought

**Zero-shot** — describe the task; don't show examples. Works when the task is common or the model is strong.

```
Classify the following email as spam or not-spam.
Email: {{ email }}
Answer:
```

**Few-shot** — include $k$ (typically 3–8) worked examples. Dramatic quality boost when the task is unusual or the format is specific.

```
Classify the following emails.

Example 1:
Email: "You won! Click here!"
Label: SPAM

Example 2:
Email: "Meeting at 3pm tomorrow."
Label: NOT_SPAM

Example 3:
Email: "Congratulations, you have inherited 1M dollars from a prince."
Label: SPAM

Now classify:
Email: {{ new_email }}
Label:
```

**Design choices for few-shot:**
- **How many examples?** Usually 3–8. More helps up to a point; each costs tokens.
- **Order matters.** Recency bias — examples near the end have more influence.
- **Cover the space.** Include examples spanning your class distribution and edge cases.
- **Static or dynamic?** Static = same examples for every request. Dynamic = retrieve examples similar to the current input (via embedding). Dynamic often works better.

**Chain-of-Thought (CoT)** — ask the model to think step-by-step before answering. Massive improvement on reasoning tasks.

```
Q: If a train leaves at 3pm going 60 mph and another leaves at 5pm going 90 mph,
when do they meet?

Let's think step by step.
```

Or, cleaner:

```
Solve this problem. First, list what's given. Then show your reasoning.
Finally, state the answer as: "Answer: <value>".

Problem: {{ problem }}
```

**Structured scratchpads** — force step-by-step output before the final answer:

```
Return your response in this JSON format:
{
  "reasoning": "step-by-step derivation",
  "answer": "final answer only"
}
```

**Self-consistency** — sample $N$ CoT completions at temperature > 0; take the majority vote. Trades cost for accuracy on reasoning benchmarks.

**Tree of Thoughts (ToT)** — a search-based generalization of CoT where the model explores multiple reasoning branches with backtracking. Powerful but expensive; usually overkill.

**When NOT to use CoT.** For simple lookups, classification, or formatting tasks — CoT wastes tokens and can introduce errors. Use it when the task genuinely requires multi-step reasoning.

---

### 2.3 Advanced Techniques: Role, Constraint, Scaffolding

**Role prompting.** "You are a $ROLE" — a persistent bias in behavior. Effects are real but often overstated in blog posts. Use it to:
- Match style ("technical writer", "friendly customer support agent").
- Enforce expertise level ("PhD-level statistician").
- Set safety context ("expert medical educator; always recommend consulting a doctor").

**Not a magic bullet.** "You are the world's smartest scientist" doesn't unlock capabilities the model doesn't have.

**Constraints.** Explicit rules the model must follow. Order them by importance; use imperative voice.

- "MUST use only the information in the provided context."
- "MUST NOT make up statistics."
- "If unsure, respond with 'I don't know'."
- "Response max 200 words."

**Scaffolding — force useful intermediate steps.** Rather than asking for the final answer alone, ask for a sequence:

1. First, extract entities.
2. Then, classify each.
3. Finally, produce the summary.

For long analyses: force an outline first, then flesh it out. Two-pass generation often produces better final outputs.

**Negative examples** — sometimes helpful to show what NOT to do:

```
Good: "The meeting is at 3pm."
Bad: "Regarding the aforementioned matter, please be advised that a scheduled convocation shall commence at approximately 15:00 hours."

Rewrite: {{ input }}
```

**XML tags for structure.** Anthropic-specific advice but works across models. Use tags to delimit sections:

```
<instructions>
Rewrite the clause in plain English.
</instructions>

<examples>
<example>
<input>...</input>
<output>...</output>
</example>
</examples>

<clause>
{{ clause }}
</clause>
```

The model treats these tags as strong structural cues.

**Anti-hallucination tactics:**

- "Only use information from the provided context."
- "If the answer isn't in the context, say 'I don't know'."
- Ask the model to **cite sources** — for each claim, quote the passage.
- Confidence solicitation: "Rate your confidence 1–5. Only include claims with confidence ≥ 4."

---

### 2.4 Prompting Different Model Families

**OpenAI (GPT-4o, GPT-4-turbo, o1/o3-family):**
- Well-behaved with structured system prompts.
- Follows JSON schemas well.
- The o1/o3 reasoning models don't want long CoT prompts — they already "think" internally. Give them the problem; keep the prompt simple.

**Anthropic Claude 3.5+, Claude Opus 4+, Claude Sonnet 4.5+:**
- Loves XML tags for structure.
- Strong at long context; good at "read this and answer with citations."
- Prefers explicit instruction over few-shot in many cases.
- Consider Anthropic's [prompting guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview).

**Google Gemini:**
- Large context (2M tokens on some variants).
- Strong at multi-modal (image, video, audio).
- Similar prompting to OpenAI.

**Open models (LLaMA, Mistral, Qwen, DeepSeek):**
- More sensitive to prompt format — respect their chat templates.
- Smaller models (7B–13B) benefit disproportionately from few-shot examples.
- Fine-tuning is often a better lever than prompt engineering.

**Meta-observation.** Modern frontier models are increasingly robust to prompt-format variation. The difference between "great" and "atrocious" prompts is smaller than 2 years ago — but the difference between "atrocious" and "adequate" is still enormous. Focus on clarity, examples, structure, and iteration.

---

### 2.5 Iteration, Testing, and Prompt Versioning

**Prompts are code.** Version them:

```python
PROMPT_v1 = """..."""
PROMPT_v2 = """..."""

def run(prompt_version, input):
    prompt = {"v1": PROMPT_v1, "v2": PROMPT_v2}[prompt_version]
    return call_llm(prompt.format(input=input))
```

Or use a prompt registry — a folder of `.txt` or `.j2` files, each versioned in git.

**Test them.** Build a **small golden set** (10–100 examples) of representative inputs with expected outputs (or grading criteria). Run every prompt change through it:

```python
def evaluate(prompt, test_cases):
    passes = 0
    for tc in test_cases:
        output = call_llm(prompt.format(input=tc.input))
        if grade(output, tc.expected):
            passes += 1
    return passes / len(test_cases)
```

**Grading methods:**
- **Exact match** — for extractive tasks.
- **Regex / structural checks** — "did the output contain a JSON with these keys?"
- **LLM-as-judge** — a stronger model grades the response against criteria.
- **Human review** — expensive but essential for subtlety.

**A/B two prompts:**
1. Baseline prompt (current production).
2. Candidate prompt (proposed change).
3. Run both on the golden set; compare score.
4. Only promote if the candidate wins on the set AND doesn't regress on any specific subset.

**Failure analysis.** When a prompt underperforms:
- Group failures by input type.
- Look at 20 failing cases; find patterns.
- Add examples covering the failure mode.
- Add explicit rules addressing them.
- Iterate.

**Prompt observability tools** — Langfuse, LangSmith, Helicone, Phoenix. Trace prompts + responses in production; correlate with user feedback.

**Watch for regressions.** Model provider updates (silent auto-upgrades of `gpt-4o` etc.) can subtly shift behavior. Rerun your eval set after any provider update. Consider pinning to specific model dates when stability matters.

---

## 3. Mental Models & Analogies

### 3.1 The "Onboarding a New Employee" Model

Think of the model as a **very smart but context-less new hire** on their first day. They know a lot in general, but not:

- What your company does.
- Who the audience is.
- What tone to use.
- What "success" looks like.
- What edge cases to handle.

**Your job (prompt engineering) is onboarding.** A good onboarding doc has: role description, company context, examples of past excellent work, explicit dos and don'ts, and a specific first task. A bad onboarding doc has: "just do your best."

Every prompt should read like the onboarding you wish you'd received. Vague onboarding → mediocre first-day output. Precise onboarding → immediately-usable output.

Extend the analogy: **few-shot examples are like showing the new hire previous tickets and their resolutions.** They anchor style, format, and quality expectations far better than instructions alone.

### 3.2 The "Steering a Horse" Model

The model is a horse. It has enormous latent capability, its own preferences, and known biases. Your job isn't to *make it move* — it will always move. Your job is **steering** — nudging it toward the destination you want.

- **System prompt** = the harness. Sets basic direction.
- **Task instructions** = current bearing.
- **Examples** = "this is the terrain we prefer to travel."
- **Constraints** = fencing you can't cross.
- **CoT** = making the horse deliberate at each step instead of galloping to a stall.

A stubborn horse can be steered by re-writing the reins (better prompt). But if you're trying to ride the horse across an ocean — that is, the task is fundamentally beyond its capability — no amount of steering helps. You need a boat (a better model, fine-tuning, tool use, retrieval).

The mental model reminds you: **you don't create capability with prompts; you elicit and shape existing capability.** This is why prompt engineering has a ceiling.
![IMG-PROMPT-01](/4%20—%20LLM%20Engineering/images/IMG-PROMPT-01.jpg)

> **Caption:** Anatomy of a good prompt: role, task, rules, format, examples, delimited input.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Model Will Follow My Instructions If I Say 'Must' Enough"

Models are probabilistic; even "MUST" and "NEVER" can be violated occasionally. Robust patterns:

- **Structural constraints** (JSON schema, function calling) instead of "please return JSON."
- **Validation + retry** — check the output; if wrong, retry with corrective feedback.
- **Weakest-link failure modes** — assume the constraint will occasionally break and design defensively.

Never treat prompt rules as guarantees. Treat them as strong biases, and use code to enforce absolutes.

### 4.2 "Longer Prompts Are Always Better"

Beyond a point, longer prompts:
- Cost more tokens.
- Introduce more places for the model to get confused.
- Can dilute the important instructions.

Excellent prompts are *concise and specific*. Every sentence earns its place. Ruthlessly delete padding, hedging, restatements. Test whether removing a section hurts your golden-set score before keeping it.

### 4.3 "My Prompt Is Version-Controlled in a Google Doc"

If prompts aren't in git alongside the code that uses them, they will drift. Symptoms: two engineers using different versions, no history of when a regression started, no way to roll back. Rule: prompts live in the repo, next to the code, with tests. Prompt-management tools (Langfuse, LangSmith, PromptLayer) are a supplement, not a replacement.

---

## 5. Self-Assessment Bank (Prompting)

### Questions

**Q1 (Short answer).** Name at least five elements of a well-structured prompt.

**Q2 (Multiple choice).** For a math word-problem-solving task, adding "Let's think step by step":
- (a) Slows things down but doesn't help.
- (b) Activates chain-of-thought reasoning; typically improves accuracy on multi-step problems.
- (c) Confuses the model.
- (d) Only helps in the o1/o3-family models.

**Q3 (Short answer).** When would you use few-shot prompting over zero-shot?

**Q4 (Multiple choice).** For a strict output format (e.g., JSON), the most reliable technique is:
- (a) Adding "please return JSON" to the prompt.
- (b) Using structured outputs / JSON mode / function calling.
- (c) Higher temperature.
- (d) Longer prompts.

**Q5 (Short answer).** Describe LLM-as-judge for evaluation. Give one caveat.

**Q6 (Multiple choice).** Anthropic's Claude models are particularly responsive to:
- (a) All-caps instructions.
- (b) XML tags to delimit sections.
- (c) Emoji-only prompts.
- (d) Adversarial injections.

**Q7 (Short answer).** In what sense are prompts "code"? Name two practices you'd apply.

**Q8 (Multiple choice).** For a task where the format is unusual (e.g., extracting a specific legal citation style), the most effective technique is:
- (a) Say "extract citations".
- (b) Provide 3–5 examples of input → output pairs (few-shot).
- (c) Use higher temperature.
- (d) Set max_tokens=1.

**Q9 (Short answer).** Why can longer prompts sometimes hurt performance?

**Q10 (Multiple choice).** Self-consistency prompting means:
- (a) Sample $N$ chain-of-thought completions at temperature > 0 and take the majority answer.
- (b) Use the same prompt twice.
- (c) Freeze temperature at 1.0.
- (d) Retry on error.

---

### Answer Key & Detailed Explanations

**A1.** Role/persona, task statement, context/background, explicit rules & constraints, output format specification, few-shot examples, delimited input section. (Any five of these.)

**A2. (b).** CoT prompting works across essentially all models on multi-step reasoning tasks. On o1/o3-family models, however, extended reasoning happens internally — adding CoT prompts can actually be counterproductive; just state the problem.

**A3.** Use few-shot when: (a) the output format is unusual or precise, (b) the task requires specific style, (c) zero-shot performance is inadequate, (d) the domain is niche, (e) you want to steer edge-case handling by providing examples. Zero-shot suffices when the task is common (translation, summarization) and the model is strong.

**A4. (b).** Structural enforcement (JSON mode, structured outputs via schema, function/tool calling) uses grammar-constrained decoding to guarantee valid output. Prompt instructions alone leave a tail of failures — critical if downstream code parses the output.

**A5.** LLM-as-judge: use a strong LLM (usually GPT-4-class or better) to grade another model's outputs against a rubric. Automates evaluation at scale. Caveats: (a) the judge has its own biases (favors verbose outputs, favors first option in pairwise comparisons); (b) judge quality caps evaluation quality; (c) LLM-as-judge for adversarial/safety tasks can be gamed; (d) always validate on a small human-labeled subset.

**A6. (b).** Anthropic explicitly recommends XML tags (`<context>...</context>`, `<examples>...</examples>`) for structuring prompts sent to Claude. The model was trained with such structure in its examples and responds well to it. Works with other providers too.

**A7.** Prompts are code because: (a) they produce deterministic-ish behavior given fixed input, (b) they have bugs, (c) changes can break downstream systems, (d) they need version control. Practices: **git-versioned prompt files**, **test suites (golden sets)** with graded outputs, **A/B testing** before promoting a change to production, **code review**, **semantic versioning** or timestamps for prompts.

**A8. (b).** Few-shot examples pin down the format better than any description. Show 3–5 worked examples covering the format's edge cases; the model will match. Try to include easy AND tricky examples in the shot set.

**A9.** Longer prompts (a) waste tokens (cost), (b) can bury the important instruction in noise, (c) may activate model instruction-following weaknesses ("goldilocks" — too little info bad, too much info also bad), (d) increase risk of internal contradictions in the prompt itself. Excellent prompts are the shortest that reliably work.

**A10. (a).** Self-consistency (Wang et al., 2022): sample multiple chain-of-thought completions at temperature > 0, extract each final answer, take the majority. Improves reasoning accuracy at the cost of $N$× tokens.

---

## 6. Practice Prompts

1. **Rewrite for structure.** Take a "please summarize this" prompt and rewrite it with role/task/rules/format/examples/input sections. A/B against the original.
2. **Few-shot upgrade.** Take a classification task; run zero-shot and 5-shot. Measure accuracy on a labeled dataset.
3. **CoT ablation.** On a math benchmark (GSM8K), compare temperature-0 direct answers to CoT-prompted answers. Report accuracy.
4. **Golden set.** Build a 25-case golden set for a real prompt in your work. Grade with LLM-as-judge. Iterate the prompt until the score plateaus.
5. **Injection defense.** Take a user-content-heavy prompt and add clear delimiter tags. Attempt a prompt-injection attack (ask the user content to override the system). Measure susceptibility before and after.

---

## 7. References

- Wei et al., ["Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903) (2022).
- Wang et al., ["Self-Consistency Improves Chain of Thought Reasoning"](https://arxiv.org/abs/2203.11171) (2022).
- Anthropic prompt engineering guide: [https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview).
- OpenAI prompt engineering guide: [https://platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering).
- Lilian Weng, ["Prompt Engineering"](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) (2023).
- "The Prompt Report" (Schulhoff et al., 2024) — comprehensive survey.
