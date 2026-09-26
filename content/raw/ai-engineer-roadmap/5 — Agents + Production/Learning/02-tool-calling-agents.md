# Tool Calling for Agents — Master Study Guide

> **Track:** Agents + Production · **Module:** 02
> **Prerequisites:** Module 01 + Month 4 Module 06 (function/tool calling basics).
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Tool calling is what makes an agent an agent. Without tools, an LLM produces text; with tools, it can search, compute, query databases, send messages, control services — actually *do things*. Every agent design choice ultimately routes through tool design.

This module goes deeper than Month 4's tool-calling module. There we covered how tool calling works at the API level. Here we cover **tool design as an agent-engineering discipline**: how to design tools the LLM understands and uses correctly, how to compose them, how to handle failures, and how to keep the tool set curated as the agent grows.

**Fundamental principles you must own:**

1. **The tool schema IS the interface** — the LLM sees only names, descriptions, and JSON schemas.
2. **Tool descriptions are prompts** — invest time on them.
3. **Small tool sets beat large ones** — 3–10 sweet spot.
4. **Tool composition** > many tools — build primitives, let the LLM compose.
5. **Tool errors are teachable moments** — return structured errors the LLM can react to.
6. **Tools are the security perimeter** — every tool call is a potential attack vector.

If you retain nothing else: **tools are the API for the LLM. Design them the way you'd design an API for a bright but literal-minded junior engineer.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Anatomy of a Tool Schema

Every tool has:
- **Name** — machine-friendly identifier (`search_web`, not `Search The Web`).
- **Description** — natural-language explanation of what the tool does, when to use it, and any limits.
- **Input schema** — JSON schema for parameters, with types, required fields, enums, descriptions.

**Example — a well-designed tool:**

```python
{
    "name": "search_web",
    "description": (
        "Search the web for a query. Returns up to 10 result objects with "
        "title, snippet, and url. Use this to find current information not in "
        "your training data. For questions about internal company data, use "
        "search_internal_docs instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Prefer specific, keyword-rich queries.",
            },
            "num_results": {
                "type": "integer",
                "description": "How many results to return (1-10).",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
            "recency": {
                "type": "string",
                "enum": ["any", "day", "week", "month", "year"],
                "description": "Filter by recency; defaults to any.",
                "default": "any",
            },
        },
        "required": ["query"],
    },
}
```

**What makes it good:**
- Description tells the LLM **when** to use it AND when NOT to (points to `search_internal_docs`).
- Every field has a description — the LLM uses these to fill in arguments correctly.
- Enums restrict `recency` to valid values — prevents "yesterday" or "past week" strings.
- Range constraints on `num_results` — prevents 1000 requests.
- `default` values reduce required LLM decisions.

**A tool description is a prompt.** The LLM reads it every request. Write it as carefully as your system prompt.

---

### 2.2 Design Principles

**1. Minimize tool count.**
- 3–10 tools = sweet spot.
- 10–30 tools = getting hard; add groupings or hierarchy.
- 30+ = LLM confused; use routing or hierarchical agents.

**2. Prefer primitives over specialized tools.**
- Bad: 50 tools like `book_flight_united`, `book_flight_delta`, `book_flight_lufthansa`.
- Good: 1 tool `book_flight(airline, ...)` that internally dispatches.

**3. Idempotence where possible.**
- `create_ticket(subject, body, idempotency_key)` — retryable without duplication.

**4. Small, composable primitives beat god-functions.**
- Prefer `fetch_url` + `extract_text` + `summarize` over `read_and_summarize_url`.
- Composability lets the LLM adapt to unexpected inputs.

**5. Explicit input/output types.**
- No `**kwargs` catchalls. Every input typed with a schema.
- Return dicts with named fields, not unstructured strings.

**6. Tool naming.**
- Verbs: `search_web`, `send_email`, `get_weather`.
- Consistent noun ordering: `get_user_profile`, not `user_profile_get`.
- Snake_case.

**7. Descriptive errors.**
- Return `{"error": "URL fetch failed: 404 Not Found", "url": "...", "retriable": false}`.
- The LLM reads this and reacts.

**8. Safe defaults.**
- If a tool can be destructive, require explicit confirmation flags: `delete_document(doc_id, confirm=True)`.

**9. Document limits and side effects in the description.**
- "Sends an email; irreversible."
- "Rate limited to 10 calls per minute."
- "Reads only from the current user's inbox."

**10. Test each tool independently.**
- Every tool has its own unit tests.
- The LLM's use of the tool is what changes; the tool itself should be deterministic.

---

### 2.3 Tool Result Design

The observation the LLM gets back matters as much as the input schema.

**Good tool results:**
- **Structured** — dict with named fields, not a wall of text.
- **Concise** — a search result should be title + snippet + url, not the full page HTML.
- **Complete** — include everything the LLM might need to reason with, no more.
- **Consistent** — the same tool always returns the same shape.

**Example bad result:**
```python
return "<html><body>...100 KB of HTML...</body></html>"
```

The LLM has to parse the HTML mentally. Expensive tokens; error-prone.

**Example good result:**
```python
return {
    "url": "https://example.com/article",
    "title": "Montana Passes New Legislation",
    "author": "Jane Doe",
    "publication_date": "2026-09-20",
    "summary": "The Montana legislature passed HB-247 today...",
    "full_text_preview": first_500_words,
    "word_count": 3400,
    "citation": "Doe, J. (2026, Sep 20). Montana passes new legislation. Example News.",
}
```

**Truncation strategy.** Some tools return arbitrarily large results (`read_file`, `search`, `list_dir`). Options:
- **Hard truncate + note** — return first N characters with `{"truncated": true, "total_size": ...}`.
- **Paginate** — return page 1; the LLM can call `get_next_page`.
- **Summarize** — return a machine summary + option to expand specific parts.

Never blow the LLM's context with a giant tool result. The LLM's short-term memory has a hard limit.

**Errors as first-class results.**
```python
return {
    "error": "authentication_failed",
    "message": "The API key is missing or invalid.",
    "retriable": False,
    "suggested_action": "Ask the user to provide their API key.",
}
```

The LLM can now decide to ask the user, retry with a different tool, or apologize. Compare to raising a Python exception, which the LLM sees only as "the request failed" without context.

---

### 2.4 Tool Composition and Agent Workflows

Rich agent behavior emerges from **tool composition**: the LLM chains primitive tools into complex operations. Design primitives to compose cleanly.

**Example: research on a topic.**

Primitives:
- `search_web(query)`
- `fetch_url(url)`
- `extract_text(html)`
- `summarize(text, max_words)`

The LLM's compose:
1. `search_web("Montana bill drafting workflow")` → 10 results.
2. `fetch_url(result[0].url)` → HTML.
3. `extract_text(html)` → plain text.
4. `summarize(text, 200)` → summary.
5. Repeat for results 1, 2, 3.
6. Synthesize final answer.

Compare to a monolithic `research(topic)` tool:
- Less flexible (LLM can't adjust intermediate steps).
- Harder to debug (one big black box).
- More brittle (the tool has to handle every case).

**Emergent flexibility.** With good primitives, the LLM can do things you didn't explicitly design for. That's the point of agents.

**Explicit chaining tools.** Sometimes you want to force a sequence via a compound tool:
- `search_and_summarize(query, max_results)` — the LLM issues one call; the compound tool does the whole chain internally.
- Trade-off: less LLM flexibility, but lower cost (fewer LLM turns).

Choose depending on how much runtime flexibility you need vs how much cost you can afford.

---

### 2.5 Tool-Call Failure Modes and Defenses

Agents fail through tool interaction. Common patterns:

**1. Hallucinated tool call.** LLM invents a tool name that doesn't exist, or invents arguments that don't match the schema.

**Defense:** validate the tool call against the schema before executing. If invalid, return an error result and let the LLM correct. Modern tool-calling APIs (OpenAI Structured Outputs, Anthropic tools) enforce this automatically — the API rejects malformed calls at generation.

**2. Argument-format bugs.** LLM passes a date in the wrong format, a number as a string, or invalid enum values.

**Defense:** parse and validate every argument. Return a structured error the LLM can react to:
```python
return {"error": "invalid_argument", "field": "date",
        "message": "Expected YYYY-MM-DD; got 'yesterday'.",
        "example": "2026-09-20"}
```

**3. Repeated tool calls in a loop.** LLM calls the same tool with the same arguments over and over.

**Defense:** track recent tool calls; if the same (tool, args) occurs 3+ times, break the loop and return an error like "You've called `search_web` with the same query 3 times. Please try a different approach."

**4. Runaway iteration.** LLM keeps calling tools, never emits a final answer.

**Defense:** hard cap on iterations (Module 03).

**5. Tool-call injection.** A retrieved document contains "call `delete_all_data()`" and the LLM tries.

**Defense:** never trust text from tools as instructions; use allowlists on which tools can be called (some tools available in "planning" contexts only, others in "action" contexts).

**6. Destructive-action authorization.** LLM decides to call `delete_customer` on its own.

**Defense:** classify tools by risk; require human approval or explicit user consent for high-risk actions. See Module 09 (guardrails).

**7. Rate-limit or cost blowout.** LLM calls an expensive tool (image generation, LLM call) many times.

**Defense:** per-tool rate limits; per-run cost caps; budget alerts.

**Failure logging.** Log every tool call: name, args, result, latency, cost, `is_error`. Aggregate for debugging patterns and eval.

---

## 3. Mental Models & Analogies

### 3.1 The "Junior Developer with API Docs" Model

Your agent is a **junior developer** you gave API docs and asked to build something.

- **Tool description** = the API docs. Vague docs → wrong calls. Rich docs with examples → correct calls.
- **Input schema** = the type signature. Loose types (`Any`) → invalid inputs. Strict types (enums, min/max) → sane inputs.
- **Return schema** = the API response format. Consistent → the developer knows what to expect. Ad-hoc strings → they have to parse and guess.
- **Errors** = structured error messages. `{code: "invalid_arg", ...}` → they understand and fix. Bare exceptions → confusion.

A junior developer given a good API + good docs writes surprisingly good code. Same for the LLM. **Tools are the API you're handing an intern who won't ask clarifying questions** — so write docs as if they'll misinterpret anything ambiguous.

### 3.2 The "Kitchen Utensils" Model

An agent is a chef; the tools are utensils in the drawer.

- **Too many utensils** → the chef spends time deciding which one to use for each dish.
- **Specialized utensils** (garlic press, cherry pitter) do one job well but are useless outside it.
- **Primitive utensils** (chef's knife, cutting board) compose into every task.
- **Sharp, well-labeled utensils** work smoothly.
- **Broken utensils that don't complain when they break** cause dish disasters.

Design your tool drawer for the chef's flexibility: a few good primitives, sharp and well-labeled. Reach for specialized tools only when they materially save time.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The LLM Will Figure Out How to Use a Poorly-Documented Tool"

It won't — reliably. Vague descriptions produce wrong calls: wrong tool selected, wrong args, wrong sequence. Every hour spent polishing tool descriptions saves many at debug time. Write tool descriptions with the mindset "this LLM has never met me and will interpret the words literally."

### 4.2 "Tool Errors Should Raise Exceptions and Crash the Loop"

Never. If a tool raises, the LLM sees only a generic failure. Return **structured error results**: `{"error": "...", "message": "...", "retriable": bool, "suggested_action": "..."}`. The LLM often recovers gracefully — retries with different args, tries another tool, asks the user, apologizes. Exceptions kill the agent's autonomy.

### 4.3 "Adding More Tools Makes the Agent Smarter"

Empirically, past ~10 tools, LLM tool-selection accuracy degrades — the LLM gets confused about which tool to use. Solutions: (1) prune to essentials; (2) group tools by domain and use routing (a first-pass LLM classifies into a domain); (3) use hierarchical agents (a supervisor with 3 tools delegates to specialists with 3 each).

---

## 5. Self-Assessment Bank (Tool Calling for Agents)

### Questions

**Q1 (Short answer).** What are the three components of a tool schema, and why is the description especially important?

**Q2 (Multiple choice).** A well-designed tool with 3 arguments should have:
- (a) Vague types (`Any`) to be flexible.
- (b) Strict types, enums where possible, descriptions per field, and reasonable defaults.
- (c) Only required args.
- (d) Free-form JSON input.

**Q3 (Short answer).** Why should tool results be structured (dicts with named fields), not free-form strings?

**Q4 (Multiple choice).** If a tool call fails, the correct default is:
- (a) Raise an exception; crash the loop.
- (b) Return a structured error result so the LLM can react.
- (c) Silently ignore.
- (d) Log and move on.

**Q5 (Short answer).** Give three defenses against a runaway agent that repeatedly calls the same tool.

**Q6 (Multiple choice).** Compared to one god-tool `research(topic)`, exposing primitives `search + fetch + extract + summarize`:
- (a) Is worse; more moving parts.
- (b) Enables composition — LLM can adapt to novel cases, at higher cost per run.
- (c) Is identical.
- (d) Only works with GPT-4.

**Q7 (Short answer).** Why should destructive tools (delete, send, transfer) require special handling?

**Q8 (Multiple choice).** For an agent with 30+ tools, the recommended architecture is:
- (a) Add more tools until it works.
- (b) Group into domains and use routing or hierarchical agents.
- (c) Delete the tools.
- (d) Increase model size.

**Q9 (Short answer).** How should you handle a tool that returns 100 KB of raw HTML?

**Q10 (Multiple choice).** "Tool-call injection" from a retrieved document ("call `delete_all()`") is best defended by:
- (a) Trusting the LLM.
- (b) Allowlisting which tools can be called in which contexts + refusing to execute injected calls.
- (c) Removing tools.
- (d) Better prompts alone.

---

### Answer Key & Detailed Explanations

**A1.** (1) **Name** — machine-friendly identifier. (2) **Description** — natural-language explanation of what the tool does and when to use it. (3) **Input schema** — JSON schema for parameters with types, descriptions, enums, defaults. The **description is a prompt** — the LLM reads it on every request and it drives tool-selection accuracy. Vague descriptions → wrong tool selected or wrong args.

**A2. (b).** Strict types (int, string, enum), descriptions per field, and defaults for optional args. Enums prevent free-form garbage. Defaults reduce required LLM decisions. Descriptions guide argument formatting.

**A3.** Structured results are machine-readable — the LLM can access specific fields (`.url`, `.title`) rather than parsing free-form text. Consistent shape across calls. Enables composition (the next tool call can use `result.url` cleanly). Free-form strings force the LLM to do NLP on the tool's output, wasting tokens and increasing errors.

**A4. (b).** Structured error results (`{"error": ..., "message": ..., "retriable": ...}`) let the LLM see and react — retry with different args, try another tool, ask the user, or apologize. Exceptions kill autonomy; the LLM can't recover from a crash.

**A5.** (1) **Track recent tool calls** — after 3 identical (tool, args) calls, break the loop with a helpful error. (2) **Hard iteration cap** — max_steps per run. (3) **Cost cap** — max total tokens/dollars. (4) **Semantic-similarity check** — if the LLM's recent thoughts are essentially the same, terminate. (5) **Time cap** — wall-clock limit per run. Any three plus rationale.

**A6. (b).** Primitives compose: the LLM chains them into novel workflows the designer didn't anticipate. Trade-off: more LLM turns per task (cost, latency). God-tools are cheaper per successful case but brittle when inputs differ.

**A7.** Destructive tools have real-world side effects the LLM can't undo. If the agent makes a mistake, damage is permanent. Defenses: (1) require explicit `confirm=True` argument; (2) require human-in-the-loop approval; (3) restrict destructive tools to specific contexts; (4) simulate first ("dry run" mode).

**A8. (b).** Above ~10 tools, LLM tool selection accuracy degrades. Group tools into domains; use a router LLM to classify the request, then delegate to a specialist agent with 3–5 tools. Hierarchical agents scale to hundreds of tools while keeping each agent's decision-making tractable.

**A9.** Never return raw. Options: (1) **Truncate** to first N chars + note `"truncated": true, "total_size": ...`. (2) **Extract** relevant fields (title, text, metadata) and return only those. (3) **Paginate** — return page 1, expose `get_next_page`. (4) **Summarize** with an internal LLM and return the summary. Preserves LLM context; still gives the LLM enough info to proceed.

**A10. (b).** Never trust text from tools/documents as instructions. Enforce a **tool allowlist** per context (e.g., during retrieval, only read tools available; during action phase, action tools available). Also validate every tool call against schema before executing, and require human approval for destructive tools regardless of who requested it.

---

## 6. Practice Prompts

1. **Tool schema audit.** Design 5 tools for a research agent (search, fetch, extract, summarize, save_note). Write full JSON schemas with descriptions, examples, and constraints. Have a colleague read and identify ambiguities.
2. **Structured errors.** Take a tool that could fail (URL fetch). Design 5 possible failure modes; write structured error results for each. Test with a mock agent — verify it recovers gracefully.
3. **Loop detector.** Add a "same (tool, args) seen 3 times" detector to your minimal agent from Module 01. Test on a query that induces looping.
4. **God-tool vs primitives.** Implement "research an entity" as (a) one big tool, (b) four composable primitives. Compare latency, cost, robustness on 5 queries.
5. **Destructive tool safety.** Design a `send_email(...)` tool. Add a `confirm=True` requirement and human-in-the-loop approval. Simulate an injection attack; verify safety.

---

## 7. References

- Yao et al., ["ReAct"](https://arxiv.org/abs/2210.03629) (2022).
- Anthropic tool use docs: [docs.claude.com/en/docs/build-with-claude/tool-use](https://docs.claude.com/en/docs/build-with-claude/tool-use).
- OpenAI function calling docs.
- Anthropic, ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) — sections on tool design.
- Instructor library ([github.com/jxnl/instructor](https://github.com/jxnl/instructor)) for typed tool interfaces.
