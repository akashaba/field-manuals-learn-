# Context Engineering — Master Study Guide

> **Track:** Agents + Production · **Module:** 08
> **Prerequisites:** Modules 01–07.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** "Context engineering" is the emerging term of art for **the discipline of designing what the LLM sees each step**. Prompt engineering was the earlier name — but modern agent systems build the LLM's context *dynamically*: retrieved chunks, tool outputs, memory retrieval, planner outputs, prior turns. What ends up in the LLM's context is *engineered*, often at every step.

In an agent, **the context IS the state of the world** as far as the LLM is concerned. Everything you want it to consider must be in view. Everything you don't want cluttering must be out. This isn't a fringe optimization — it's *the* discipline that separates a working agent from a hallucinating one.

**Fundamental principles you must own:**

1. **You are the LLM's world-builder.** It knows only what you show it.
2. **Every token has a cost and a signal-to-noise ratio.** Manage both.
3. **The context is dynamic** — assembled per step from many sources.
4. **Different components need different placements** — system prompt, message history, retrieved content, tool results.
5. **Formatting matters** — XML tags, JSON, delimiters change model behavior.
6. **Context has structure** — a good context has visible sections a model can attend to.

If you retain nothing else: **you are the librarian curating the LLM's reading list for every turn. Curate deliberately.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Anatomy of an Agent Context

At any given turn, the LLM sees a **context** that's assembled from many sources:

```
┌─────────────────────────────────────────────┐
│ System Prompt                                │
│  ├── Role & Identity                         │
│  ├── Capabilities & Rules                    │
│  ├── Tool Descriptions                       │
│  ├── User Profile / Memory                   │
│  ├── Retrieved Context (RAG)                 │
│  └── Guardrails                              │
├─────────────────────────────────────────────┤
│ Conversation History                         │
│  ├── User Turn 1                             │
│  ├── Assistant Turn 1 (with tool calls)      │
│  ├── Tool Results 1                          │
│  ├── ...                                     │
│  └── User Turn N (current)                   │
└─────────────────────────────────────────────┘
```

Each section serves a purpose:

- **Role & identity** — "You are a research assistant..."
- **Capabilities & rules** — "You can search the web, fetch pages, and summarize. You must cite sources..."
- **Tool descriptions** — via structured tool schemas (Module 02).
- **User profile / memory** — long-term facts about the user.
- **Retrieved context** — chunks from RAG, per-turn or per-session.
- **Guardrails** — refusal criteria, escalation triggers.
- **Conversation history** — the turn-by-turn record.
- **Tool results** — observations from tool calls this turn.

**A skilled context engineer designs the assembly.** Not every source belongs in every turn.

**Static vs dynamic context:**

- **Static**: put once in the system prompt (identity, rules, user profile). Same every turn. Prompt-cacheable.
- **Dynamic**: assembled per turn (retrieval, tool results, most-recent-memory queries). Changes each request.

Separating static from dynamic is critical for cost: static portion is **prompt-cacheable** (Module 09 of Month 4); dynamic portion isn't.

---

### 2.2 The Signal-to-Noise Ratio

Every token in context competes for the model's attention. Two failure modes:

**Signal starvation.** Not enough relevant info. Model hallucinates or refuses.

**Noise overwhelm.** Too much irrelevant info. Model gets lost in the middle, over-focuses on distractions, contradicts itself.

**Both are engineered problems.** Solve them by:

**1. Retrieval precision.** Rerank aggressively (Module 16 of Month 4) so only top-5 relevant chunks reach the LLM.

**2. Trimming.** Cut unnecessary parts of tool results before feeding back:
```python
# Bad: dump raw API response
observation = api_response.text   # 50 KB of HTML

# Good: extract just what's useful
observation = {
    "url": ...,
    "title": ...,
    "summary_snippet": first_500_chars,
}
```

**3. Summarization.** For long tool results, run a small LLM to compress: "extract the answer to X from this text."

**4. Selective inclusion.** Don't include every tool result in every subsequent turn. If tool 3's result is stale, drop it.

**5. Structured output.** JSON forces the model to be terse. Prose invites verbose padding.

**6. Explicit boundaries.** Delimit sections with XML tags or clear headers so the model can navigate.

---

### 2.3 Formatting: XML Tags, JSON, Markdown

Format shapes how the model reads the context. Modern best practices:

**XML-style tags for delimiters.**

```
<user_profile>
Brian is a Senior Software Engineer in Helena, Montana.
He prefers concise responses. He works on legmt.gov.
</user_profile>

<retrieved_docs>
<doc id="1" source="handbook.pdf" section="Onboarding">
New employees complete orientation in their first week.
</doc>
<doc id="2" source="benefits.pdf" section="Health">
Health benefits start on the first of the month following start date.
</doc>
</retrieved_docs>

<task>
The user is asking when they'll be covered by health insurance.
Answer using ONLY the retrieved documents. Cite by doc id.
</task>
```

Why XML works: Anthropic (and others) trained on lots of tagged data; models handle these boundaries naturally. Robust to nested content.

**JSON when parseable.** For structured inputs (config, params, data). Not typically for the whole context — mixing prose with JSON in the middle confuses the model.

**Markdown headings for sections.**

```
## Rules
- Answer only from retrieved docs.
- Cite sources.

## Retrieved Content
### Source 1 (handbook.pdf, page 4)
...
```

Works fine; XML tags are cleaner for nested / typed content.

**Placement rules:**

- **Rules & identity at the top.** Sets expectations before content.
- **Retrieved content in the middle** — bulk of the context, delimited.
- **Question at the bottom.** Attention peaks at the end; the model reads through the rules and content, then applies to the question.

**Anti-patterns:**

- **Wall-of-text context** with no boundaries — model can't tell what's rule vs data vs question.
- **Mixed formatting** — some XML, some markdown, some raw text — confuses the model.
- **Instructions scattered** — repeated at random places rather than a coherent rules block.

---

### 2.4 Dynamic Assembly per Turn

The most powerful context-engineering pattern: **decide what to include based on the current turn**.

**Retrieval-per-turn.** In a chatbot, each user query embeds and retrieves relevant chunks from your corpus. Only relevant chunks are in context this turn.

**Memory-per-turn.** Query the user's memory for facts relevant to the current query.

```python
def build_context(user_query, session):
    profile = load_profile(session.user_id)                # static
    relevant_memory = search_memory(user_query, session.user_id, k=5)  # dynamic
    retrieved_docs = rag_retrieve(user_query, k=5)          # dynamic
    recent_history = tail(session.messages, n=5)             # dynamic

    system = f"""You are ...

    <user_profile>
    {profile}
    </user_profile>

    <relevant_memory>
    {format_memory(relevant_memory)}
    </relevant_memory>

    <retrieved_docs>
    {format_docs(retrieved_docs)}
    </retrieved_docs>
    """
    return system, recent_history
```

**Adaptive tool sets.** Different tools for different phases of the agent's task:

```python
def get_tools_for_phase(phase):
    if phase == "planning":
        return [read_tools]         # research only
    if phase == "execution":
        return [read_tools + write_tools]  # can act
    if phase == "review":
        return [read_tools]         # can't modify during review
```

Reduces tool count per turn → better tool selection. Also security-in-depth (can't accidentally write during planning phase).

**Compression on the fly.** If tool result too big, summarize before feeding back:

```python
if len(tool_result) > 5000:
    tool_result = summarize(tool_result, question=user_query)
```

**Pruning.** Drop old tool observations that the LLM already incorporated:

```python
# After the LLM's next message referenced tool_1's result, drop tool_1's full observation
```

---

### 2.5 Context Bloat and How to Fight It

An agent's context grows every turn. Left unchecked:

- Turn 20: 60k tokens in context.
- Turn 40: OOM or context-limit error.

**Diagnostic questions:**
- What's the token budget per turn?
- How does it grow with turn count?
- Which components dominate?
- Which are cacheable (static)?

**Attack strategies:**

**1. Cache the static portion.** System prompt + user profile + tool schemas rarely change; make them the cacheable prefix. See Month 4 Module 09.

**2. Sliding window for messages.** Keep last N turns; older turns go to summary.

**3. Compress tool results as they age.** Fresh results verbatim; older ones a one-line summary.

**4. Retrieval instead of retention.** Instead of keeping every tool result in context, save results to a scratchpad; the LLM can query the scratchpad on demand.

**5. Structured state.** Replace narrative history with a typed state dict; the LLM sees `state.results.count = 12` instead of scrolling 12 tool outputs.

**6. Per-turn budget.** Set a hard cap: "context this turn cannot exceed 20k tokens." Trim / summarize to fit.

**Instrumentation:** log per-turn token counts. Alert when they exceed budget. This is the debugging entry point for context bloat.

**Chunking budget per component:**
- Rules & identity: ~500 tokens.
- Tool schemas: ~1000 tokens.
- User profile / memory: ~500 tokens.
- Retrieved docs: ~4000 tokens (5 chunks × 800 tokens).
- Recent history: ~3000 tokens.
- **Total system context: ~9000 tokens.**
- **Reserve for output**: ~2000 tokens.
- **Buffer**: ~1000 tokens.

Total request: ~12000 tokens per turn. Sustainable.

If your agent exceeds these, revisit. Some pruning will win.

---

## 3. Mental Models & Analogies

### 3.1 The "Chef's Prep Station" Model

Every LLM turn is a chef preparing a dish. The context is the **prep station** — ingredients, tools, recipe, and instructions all laid out.

- A **cluttered station** — ingredients from three different meals piled together — leads to mistakes.
- A **sparse station** — missing the recipe — leads to guessing.
- A **well-laid station** — recipe posted, only relevant ingredients out, tools arranged for the current dish — leads to a good meal.

Your job as context engineer: lay out the station for each dish. Some ingredients stay (staples: salt, oil = static system prompt). Some are dish-specific (produce, protein = retrieved docs). Some tools are always out; some rotate.

Great chefs never work in a cluttered kitchen. Great agents don't operate on cluttered contexts.

### 3.2 The "Investigator's Case Board" Model

An agent solving a task is like a **detective working a case**. The context is the **case board** — pinned evidence, working theories, contacts to interview.

- **Fresh case board** — starts empty; agent adds evidence as it goes.
- **Structured board** — sections for "witnesses," "timeline," "physical evidence."
- **Pruning board** — evidence proven irrelevant gets moved to archive; the visible board stays clear.
- **Summarizing** — after 40 interviews, the detective doesn't keep all 40 transcripts pinned — a summary of interviewees + key quotes suffices.

If the board is a mess, the detective misses connections. If everything's in one blob, patterns don't emerge. Structure + selective inclusion + pruning are the practice.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "More Context = Better Answers"

Not linearly. Past a threshold (varies by model — usually 20–60k tokens for effective use, even in 200k-context models), quality degrades from "lost in the middle." Add context that helps; trim what doesn't. Signal-to-noise beats raw information volume.

### 4.2 "Dumping Every Tool Result Back Into Context"

Common with naive agent loops. After 5 tool calls, the context has 5 verbose tool outputs. The LLM already used them. Old observations pollute; only their conclusions matter for future turns. Compress or drop as they age.

### 4.3 "Format Doesn't Matter — the LLM Handles Anything"

It does, but performance varies. XML tags for delimited sections, structured JSON for data, markdown for prose — clear delimitation improves the model's ability to attend to the right part. Wall-of-text contexts, especially without section markers, degrade quality.

---

## 5. Self-Assessment Bank (Context Engineering)

### Questions

**Q1 (Short answer).** In one paragraph, define "context engineering."

**Q2 (Multiple choice).** In an agent's context, the recommended placement of the user's current question is:
- (a) At the top.
- (b) In the middle.
- (c) At the bottom, after rules and content.
- (d) In the system prompt.

**Q3 (Short answer).** Give three components typically included in an agent's system prompt, and one common source of dynamic per-turn content.

**Q4 (Multiple choice).** For a chatbot with a 20k-token static system prompt including user profile and rules, prompt caching enables:
- (a) Nothing.
- (b) Serving the static portion from cache at ~10% of normal input pricing; 5–10× cost reduction.
- (c) Faster training.
- (d) Longer context.

**Q5 (Short answer).** How do XML tags in the context (e.g., `<retrieved_docs>...</retrieved_docs>`) help the LLM?

**Q6 (Multiple choice).** After turn 10 with many tool observations, an agent's context is bloating. The BEST mitigation is:
- (a) Increase max_tokens.
- (b) Prune old tool observations that the LLM already incorporated + move detailed state to a scratchpad.
- (c) Truncate the messages randomly.
- (d) Switch models.

**Q7 (Short answer).** Explain "adaptive tool sets" and give one motivation.

**Q8 (Multiple choice).** Signal-to-noise ratio in context matters because:
- (a) Tokens cost money regardless of content.
- (b) Beyond an amount, irrelevant tokens degrade the model's attention on relevant ones — quality declines.
- (c) Only for streaming.
- (d) Irrelevant tokens speed up inference.

**Q9 (Short answer).** For per-turn dynamic context, name three things you might retrieve/generate freshly each turn.

**Q10 (Multiple choice).** For a long-running agent, per-turn token instrumentation:
- (a) Is nice-to-have but not important.
- (b) Is essential — it's the diagnostic entry point for context bloat, cost overruns, and quality regressions.
- (c) Only for eval.
- (d) Adds too much overhead.

---

### Answer Key & Detailed Explanations

**A1.** Context engineering is the discipline of designing what the LLM sees at each turn — the system prompt, tool descriptions, retrieved content, memory, and conversation history — assembled dynamically per step to maximize signal-to-noise, respect the context budget, and produce reliable behavior. It generalizes prompt engineering to the agent setting where context is composed from multiple sources.

**A2. (c).** Rules first, content in the middle, question at the end. Question at the end is at a peak-attention position ("end") and matches natural read flow. Empirically better than question-first or question-in-the-middle.

**A3.** **System prompt static components**: role/identity, tool schemas, rules, user profile (semi-static), refusal criteria, formatting instructions. Any three. **Dynamic per-turn**: retrieved RAG chunks, memory query results, tool result observations, user's current message.

**A4. (b).** Prompt caching serves the stable prefix from provider-side KV cache at ~10% of normal input pricing. On a 20k-token static prompt, that's ~5–10× cost reduction on the input portion for every subsequent request that shares that prefix.

**A5.** XML tags delimit sections clearly, letting the model attend to the right part of the context per its role: rules vs docs vs task vs user question. Anthropic (and others) train on tagged content, so models handle nested / typed content naturally. Reduces "which part of this text is the instruction vs data" ambiguity.

**A6. (b).** Prune old tool observations after they've been used; move detailed working state to a scratchpad or structured state store; the LLM sees compact references and can query the scratchpad if it needs details. Increasing max_tokens (a) hides the problem; random truncation (c) loses critical info.

**A7.** **Adaptive tool sets**: expose different tools to the agent based on the current phase or state. E.g., during a "planning" phase, only read/search tools; during "execution," add write/action tools. Motivations: (1) reduces tool count per turn → better tool selection; (2) safety — can't accidentally trigger destructive tools during read-only phases; (3) simpler prompts per phase.

**A8. (b).** Beyond a threshold, extra tokens compete with relevant ones for attention; "lost in the middle" degrades recall. Signal-to-noise, not raw token count, drives quality on retrieval-heavy tasks.

**A9.** (1) **RAG-retrieved documents** relevant to the current query. (2) **Memory query results** — facts about the user relevant to this turn. (3) **Recent conversation history** (sliding window). (4) **Fresh tool results** from the current step. (5) **Dynamic tool set** for the current phase. (6) **Freshness-boosted content** (recent news, latest updates). Any three.

**A10. (b).** Log per-turn `total_tokens`, per-component breakdowns (system, history, retrieved, tool results), latency, cost. This is your diagnostic entry point: when bloat happens, you see it. When cost balloons, you know why. When quality regresses on longer conversations, you can correlate with context size. Absolutely essential in production.

---

## 6. Practice Prompts

1. **Context breakdown.** For your Build #1 agent, log per-turn token counts broken down by component (system, history, retrieved, tool results). Plot over 10-turn conversations.
2. **Prompt cache split.** Reorganize your system prompt so the static parts (rules, tool schemas, user profile) come first as a cacheable prefix; dynamic parts (retrieved docs) come after. Confirm cache hits.
3. **XML-tag test.** For a retrieval-augmented task, compare (a) sources as prose vs (b) sources as `<doc id="N">...</doc>`. Compare citation accuracy.
4. **Compression on the fly.** Add a compressor: if a tool result exceeds 3k tokens, summarize it (small LLM) with focus on the user's question, then feed back.
5. **Adaptive tools.** Build a two-phase agent (planning phase → execution phase) with different tool sets per phase. Confirm the plan phase can't call destructive tools.

---

## 7. References

- Anthropic, ["Prompt Engineering with Claude"](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering) — XML tag guidance.
- Anthropic, ["Long Context Prompting"](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/long-context-tips).
- Liu et al., ["Lost in the Middle"](https://arxiv.org/abs/2307.03172) (2023).
- OpenAI prompt engineering docs.
- Latent Space / other blog posts on "context engineering" as a discipline.
