# Query Rewriting - Code Walkthrough

This guide walks through the query rewriting implementation: config, heuristic pipeline, optional LLM, and how to run the examples.

---

## Table of Contents

1. [Overview](#overview)
2. [Config](#config)
3. [Normalize and classify](#normalize-and-classify)
4. [Strip filler and injection](#strip-filler-and-injection)
5. [Expand acronyms](#expand-acronyms)
6. [Heuristic rewrite](#heuristic-rewrite)
7. [LLM rewrite and fallback](#llm-rewrite-and-fallback)
8. [Running the examples](#running-the-examples)

---

## Overview

**Files:**
- `config.js` - Limits, acronym map, strip phrases, LLM toggle.
- `query-rewriter.js` - `normalize`, `classifyIntent`, `stripFiller`, `expandAcronyms`, `stripInjection`, `rewriteHeuristic`, `rewrite`, `parseLLMRewriteResponse`.
- `example.js` - Demos: normalize, strip filler, expand acronyms, strip injection, full heuristic, comparison alternates, troubleshooting filters, async `rewrite()`, optional LLM.

**Output shape:**
```javascript
{
  rewrittenQuery: string,
  alternateQueries: string[],
  filters: Record<string, unknown>,
  diagnostics: { intent, removed, stripped, rationale?, source: 'llm'|'heuristic' }
}
```

---

## Config

`config.js` exposes:
- `maxRewriteLength` - Cap length of rewritten query (default 300).
- `maxAlternates` - Max alternate queries (default 4).
- `allowedFilterKeys` - Keys allowed in `filters` (e.g. category, topic).
- `useLLM` - Whether to try LLM rewrite (default false).
- `llmTimeoutMs` - Timeout for LLM call (0 = no timeout).
- `stripPhrases` - Filler phrases to remove (please, can you, thanks, etc.).
- `acronymMap` - Map of abbreviation → full form (ml → machine learning, k8s → kubernetes).

Env overrides: `QUERY_REWRITE_MAX_LENGTH`, `QUERY_REWRITE_MAX_ALTERNATES`, `QUERY_REWRITE_USE_LLM`, `QUERY_REWRITE_LLM_TIMEOUT_MS`, `LOG_LEVEL`.

---

## Normalize and classify

**normalize(query)**  
Trims, collapses whitespace, strips zero-width and BOM. Use on every raw input.

**classifyIntent(query)**  
Heuristic classification for retrieval strategy:
- `comparison` - "vs", "versus", "difference between", "compare".
- `howto` - "how do", "how to", "steps", "tutorial", "guide".
- `troubleshooting` - "error", "not working", "fix", "debug", "issue".
- `find_doc` - "find", "search", "where is", "locate".
- `factual` - default.

Used to decide alternates (e.g. comparison → emit side A, side B, "A and B comparison") and optional filters.

---

## Strip filler and injection

**stripFiller(query, stripPhrases)**  
Removes phrases like "please", "can you", "thanks" (word-boundary, case-insensitive). Returns `{ cleaned, removed }`.

**stripInjection(query)**  
Removes instruction-like content so it does not affect retrieval:
- "Ignore previous instructions", "You are now…", "System:", "[INST]", etc.
Returns `{ cleaned, stripped }` (stripped = list of matched pattern descriptions for logging).

Use strip *before* acronym expansion so the final query is clean and focused.

---

## Expand acronyms

**expandAcronyms(query, acronymMap)**  
Replaces tokens that match map keys (word-boundary) with their values. Preserves meaning for retrieval (e.g. "ml" → "machine learning") and avoids diluting the embedding with abbreviations the docs may not use.

Config supplies a default `acronymMap`; you can override or extend it per domain.

---

## Heuristic rewrite

**rewriteHeuristic(userQuery, config)**  
Runs the full heuristic pipeline:
1. Normalize.
2. Classify intent.
3. Strip filler → update query.
4. Strip injection → update query.
5. Expand acronyms.
6. Cap length to `maxRewriteLength`.
7. If intent is comparison, build `alternateQueries` (each side + "A and B comparison"), capped by `maxAlternates`.
8. Optionally set `filters` (e.g. troubleshooting → `category: "troubleshooting"`).

Returns `{ rewrittenQuery, alternateQueries, filters, diagnostics }`. Synchronous; no LLM.

---

## LLM rewrite and fallback

**rewrite(userQuery, config, options)**  
- `options.generate(prompt)` - Optional. If present and `config.useLLM` is true, calls the LLM with a strict prompt that asks for JSON: `rewrittenQuery`, `alternateQueries`, `filters`, `rationale`.
- `options.conversation` - Optional. Array of `{ role, content }`; appended to the prompt so the model can resolve "it"/"that".

**parseLLMRewriteResponse(raw)**  
Parses the model output for a single JSON object; extracts and validates `rewrittenQuery`, `alternateQueries`, `filters`, `rationale`. Returns `null` on parse failure.

**Flow:**
1. If `useLLM` and `generate` provided, call `generate(buildRewritePrompt(...))`.
2. Parse with `parseLLMRewriteResponse`; if valid and `rewrittenQuery` non-empty, apply length/alternate caps and return with `diagnostics.source = 'llm'`.
3. On any error or invalid output, return `rewriteHeuristic(userQuery, config)` with `diagnostics.source = 'heuristic'`.

So retrieval always gets at least a heuristic rewrite; LLM is an optional improvement.

---

## Running the examples

From repo root:
```bash
node examples/06_retrieval_strategies/05_query_rewriting/example.js
```

No model is required for examples 1–8 (heuristic only). Example 9 runs LLM rewrite if `QUERY_REWRITE_USE_LLM=1` and an LLM model path is available (e.g. `LLM_MODEL_PATH` or the default Qwen path); otherwise it prints a short message and shows heuristic fallback.

---

## Best practices

- Always normalize and run injection stripping so retrieval never sees instructions.
- Keep acronym expansion domain-specific and configurable.
- Use a single LLM call with a strict JSON contract and validator; do not loop or use tools.
- Log diagnostics (intent, removed, stripped, source) for debugging and tuning.
- Cap rewritten length and number of alternates to avoid runaway cost and latency.
