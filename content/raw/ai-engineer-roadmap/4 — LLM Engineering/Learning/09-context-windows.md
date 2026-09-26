# Context Windows — Master Study Guide

> **Track:** LLM Engineering · **Module:** 09
> **Prerequisites:** Modules 01 (tokens), 03 (LLM APIs), 04 (prompting).
> **Time budget:** ~5–7 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** The **context window** is the ceiling on everything an LLM can pay attention to at once — system prompt, conversation history, retrieved documents, tool outputs, and the model's own generated tokens. It's often the number one operational constraint on real LLM systems:

- Your document doesn't fit → you must chunk, summarize, or retrieve.
- Your conversation grows past the limit → you must trim, summarize, or offload.
- Your prompt is too big → cost and latency skyrocket, and quality often *degrades* (long-context degradation is real).

Larger context isn't a free lunch. A 200k-token context costs 200k tokens' worth of compute and money, and empirically model quality degrades in the middle of very long contexts ("lost in the middle" phenomenon). Engineering context is one of the most impactful skills for building working systems.

**Fundamental principles you must own:**

1. **Every request has a fixed budget.** Prompt + generated output ≤ context limit.
2. **Attention is quadratic in context length** — for the model provider, this cost is priced in.
3. **Long context ≠ good recall.** Beyond some point (varies by model), a needle in a haystack is missed.
4. **Retrieval is often cheaper and better than jamming everything in.** RAG exists because context stuffing doesn't scale.
5. **Prompt caching** dramatically lowers the cost of repeatedly reusing a large system prompt or document.
6. **Compression matters.** Every token you can eliminate saves money, latency, and quality.

If you retain nothing else: **treat context like RAM — a scarce, precious resource. Design your prompts to fit, not to overflow.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 What Fits in the Window

Every provider counts tokens differently, but structurally:

$$\text{tokens}_{\text{used}} = \text{system} + \text{conversation history} + \text{tools} + \text{retrieved} + \text{generated output}$$

You must budget for the **output** too. If the context limit is 128k and you want to generate 4k tokens, your **input** budget is 124k — not 128k.

**Common context limits (as of 2026 — check current provider docs):**

| Model family | Approx. context | Notes |
|--------------|-----------------|-------|
| GPT-4-family | 128k → 200k | Effective quality typically drops past ~60k |
| Claude 3.5 → 4 | 200k → 1M | Long-context recall varies by task |
| Gemini 2.5 Pro | 1M → 2M | Very long; retrieval-friendly |
| Open weights (LLaMA-3.1, Mistral) | 8k → 128k | RoPE + position interpolation |
| Small models (7B–8B) | 8k → 32k | Usually enough for most tasks |

**Token cost math.** With prompt caching disabled, every message in a conversation is billed on every turn as input tokens. A 50k-token document in your system prompt costs $50k tokens every message. Prompt caching (see 2.4) can drop this by 5×–10×.

**Reserved-token overhead.** Each API adds a few tokens for structural formatting (chat markers, tool schemas, `<|im_start|>` in OpenAI, `\n\nHuman:` in Anthropic). Not big, but not zero. Budget ~50 tokens of overhead per message.

---

### 2.2 The "Lost in the Middle" Problem

Empirically, LLMs are much better at recalling information at the **beginning** and **end** of a long context than in the **middle**. Nelson Liu et al. (2023) showed this systematically across many models. Practical implications:

- **Put the most important instructions at the top or bottom** of the prompt.
- **Put retrieved documents right before the question**, not floating in the middle.
- **Very long contexts degrade linearly-to-worse** past model-specific thresholds.

Even at models advertising 200k or 1M context, benchmarks typically report **useful recall** at a smaller effective window (~60–128k for GPT-4/Claude family; higher for Gemini's long context but still with degradation).

**Design implication:** don't assume a model can "just read the whole document." Chunk and retrieve — or at minimum, structure the input so critical info is at natural attention hotspots.

---

### 2.3 Managing Conversation History

For a chatbot, conversation history grows unboundedly. Strategies:

**1. Truncate old messages (sliding window).**
Keep only the last $N$ turns. Simple. Loses old context.

**2. Summarize old messages.**
When history exceeds a threshold, summarize the oldest half via a cheaper model. Preserves gist; loses detail.

**3. Hierarchical summarization.**
Multiple layers of summaries — recent = full, mid-age = paragraph, ancient = single line. Common in long-conversation assistants.

**4. Retrieval over conversation history.**
Embed each turn; retrieve relevant past turns based on current query. Best when history is very long or spans multiple sessions.

**5. Explicit memory extraction.**
Extract facts ("Brian's spouse is X, kids are Y and Z") into structured memory. Load only what's relevant.

**Practical pattern:** Combine truncation for latency + summarization for continuity + retrieval for long-term memory. Most production chatbots use a hybrid.

---

### 2.4 Prompt Caching

Modern APIs support **prompt caching** — you designate a prefix of your prompt as cacheable. On subsequent requests reusing the same prefix, the provider serves it from a KV cache instead of recomputing.

**Cost benefit:**
- Anthropic: cached tokens cost ~10% of normal input tokens.
- OpenAI: cached tokens are ~50% of normal.
- Massive win for: long system prompts, few-shot exemplars, retrieved documents that don't change often.

**Latency benefit:**
- Cached prompts often ~2–5× faster time-to-first-token.

**How to use it (Anthropic example):**

```python
response = client.messages.create(
    model="claude-...",
    system=[
        {"type": "text", "text": "You are ..."},
        {
            "type": "text",
            "text": LONG_DOCUMENT_OR_INSTRUCTIONS,
            "cache_control": {"type": "ephemeral"},  # marks this block as cacheable
        },
    ],
    messages=[{"role": "user", "content": query}],
)
```

**Cache lifetime.** Typically 5 minutes on ephemeral; longer with dedicated caching plans. Cache misses cost normal token pricing plus a small write fee.

**When it pays off:**
- Chatbots with a long system prompt.
- RAG systems where retrieved chunks are frequently reused.
- Agents where the tool schema is large.
- Batch evaluation with the same prompt over many inputs.

**When it doesn't:**
- One-off queries.
- Prompts that change per request (fully dynamic).
- Very short prompts where the fixed cache-write cost dominates.

---

### 2.5 Compression Strategies

When you need more info than fits, or want to reduce cost:

**LLM summarization** — cheap model summarizes chunks into shorter form. Loses detail; useful for coarse memory.

**Extractive compression** — keep only sentences that pass a relevance filter (e.g., embedding-similarity to the query). Preserves exact wording.

**Structured extraction** — turn free text into a compact JSON structure (dates, names, key facts). Often 10× smaller than prose.

**Semantic deduplication** — remove near-duplicate chunks (common in web-scraped corpora).

**Token-level pruning (LLMLingua)** — a small model prunes low-information tokens from the prompt. 2–5× compression with modest quality loss.

**Selective attention prompt engineering.** Refer to sections by name rather than repeating them: "Refer to Section 3 (the compliance guidelines)" — the model already has the context if it's stored elsewhere.

**Truncation-aware prompting.** Design your prompt so that truncating from the middle doesn't destroy it. Put the query at the top and bottom, so if the middle is dropped, the model still knows what to do.

**Prompt-first design.** Ask yourself: what is the *minimum* information the model needs to do this task well? Trim from there. Most prompts have 30–50% waste (verbose examples, redundant instructions, or hedge language) that can be cut without quality loss.

---

## 3. Mental Models & Analogies

### 3.1 The "Working Memory" Model

The context window is the LLM's **short-term working memory**. Everything the model considers when generating its next token must be in the window — nothing more.

Human working memory holds ~7 chunks; LLMs hold tens of thousands of tokens, but they still forget the middle when overloaded and still perform best when the important info is fresh at the top or bottom.

**Design your prompts like a working-memory-friendly workspace:**
- **Bring the essentials into focus.** Put critical instructions where attention peaks — start and end.
- **Externalize the rest.** Store long documents in a vector DB and retrieve only what's needed for the moment.
- **Refresh regularly.** Long conversations need periodic summarization the way humans need coffee.

### 3.2 The "Overhead Bin on an Airplane" Model

Every LLM request is a flight with a fixed cabin space (context). You must fit:
- **Carry-on essentials** (system prompt) — always with you.
- **Personal item** (recent conversation) — accessible during the flight.
- **Documents you might need** (retrieved chunks) — bring only what you'll actually use.
- **Return travel plans** (space reserved for the model's output).

Bring too much and you're refused boarding (context length exceeded). Bring the wrong things and you can't find the boarding pass at the gate (lost in the middle). Pack thoughtfully.

The rise of RAG is basically the airline's answer: instead of stuffing every book you might read into carry-on, put them in a library at the destination (vector DB) and check out only what you need at the moment.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Model Has 200k Context, So I Can Send 200k Tokens"

You can send them; the model won't use them well. Effective context (recall accuracy on middle-position information) typically caps at a fraction of the advertised window. Also cost: 200k tokens at $3/M is $0.60 **per request**. And latency: time-to-first-token grows with context.

Rule: **use RAG or summarization to send only what's needed.** Long context is a fallback, not the default.

### 4.2 "Prompt Caching Is Free Money"

Caching helps when the *cached prefix is stable*. If your system prompt or docs change per request, the cache is invalidated every time. Also, cache-write costs money — for one-off requests, caching is a small loss.

Design for caching: put stable content at the top (system prompt, examples, docs), dynamic content at the bottom (user query, per-request state). Order matters.

### 4.3 "History Truncation Loses Personality/Context"

Naive truncation drops old messages. If those messages contained user preferences ("call me Brian, use metric units, I care about accessibility"), the assistant forgets them mid-conversation.

Fix: **extract preferences and facts into a persistent structured memory** (a separate `user_context` block always included in the system prompt), then truncate raw messages freely.

---

## 5. Self-Assessment Bank (Context Windows)

### Questions

**Q1 (Short answer).** Explain the "lost in the middle" phenomenon.

**Q2 (Multiple choice).** For a model with a 128k context limit, if you want to generate 4k tokens of output, your maximum input size is:
- (a) 128k
- (b) 132k
- (c) 124k
- (d) 128k minus 100 tokens for overhead

**Q3 (Short answer).** Why does prompt caching often reduce cost by 5–10×, and what constraint must the cached prefix satisfy?

**Q4 (Multiple choice).** In a long chat, the most robust way to preserve user preferences while still truncating history is:
- (a) Never truncate.
- (b) Extract preferences into a persistent structured `user_context` block; truncate the raw messages.
- (c) Summarize everything into a single sentence.
- (d) Store history in a vector DB and retrieve per turn.

**Q5 (Short answer).** Give three strategies for managing conversation history that exceeds the context window.

**Q6 (Multiple choice).** Which of the following makes prompt caching *ineffective*?
- (a) A stable system prompt at the top.
- (b) A large retrieved document above the query.
- (c) Content that changes per request (e.g., a fresh timestamp).
- (d) Few-shot examples at the top.

**Q7 (Short answer).** Why is it often better to structure a prompt as `[docs] → [instructions] → [query]` rather than `[query] → [docs]` when the docs are large?

**Q8 (Multiple choice).** For a 50-page PDF that a user wants to chat with, the recommended approach is:
- (a) Paste the whole thing into every request.
- (b) Chunk it into ~500-token segments, embed each, retrieve relevant ones per query.
- (c) Summarize into 200 words and send that.
- (d) Refuse; PDFs can't be processed.

**Q9 (Short answer).** Given a model with 32k context and a 50k-token document, name two ways to enable Q&A against the whole document.

**Q10 (Multiple choice).** LLMLingua-style token pruning:
- (a) Removes stop words.
- (b) Uses a small model to identify and drop low-information tokens from the prompt.
- (c) Compresses tokens with gzip.
- (d) Deletes user turns from history.

---

### Answer Key & Detailed Explanations

**A1.** "Lost in the middle" (Liu et al., 2023): LLMs recall information at the beginning and end of long contexts much better than in the middle. Model performance on retrieval questions drops as target information moves to central positions of very long inputs.

**A2. (c).** Input + output must fit in the context. 128k − 4k = 124k for input. Reserve a bit more for overhead (structural tokens) to be safe.

**A3.** Prompt caching stores the KV cache of a prefix on the provider's side. Reusing the prefix skips the (very expensive) prefill computation; provider passes ~90% of the savings back as reduced token prices. The prefix must be **stable, byte-identical** across requests — any change invalidates the cache.

**A4. (b).** Store distilled preferences and long-term memory in a persistent structured block that's always in the system prompt. Then feel free to truncate raw messages. This is how Claude's memory feature, ChatGPT memory, and every serious chatbot handle it.

**A5.** (1) **Sliding window truncation** — keep last N turns; (2) **Summarization** — periodically summarize old turns; (3) **Hierarchical summaries** — recent full, older compressed; (4) **Retrieval over history** — embed and retrieve relevant past turns; (5) **Structured memory extraction** — factual facts kept in a separate block. Any three from this list.

**A6. (c).** Caching requires byte-identical prefix. A timestamp changing per request means every request looks unique and misses the cache entirely.

**A7.** LLMs perform better when the *query* is at a position of high attention (beginning or end). Placing the query at the bottom after `[docs] → [instructions]` puts it at the "most recent" end of context, which the model attends to most. Also, placing docs first lets prompt caching work: docs (large, stable) cached; query (small, changing) uncached.

**A8. (b).** RAG is the correct pattern. Chunk, embed, retrieve top-k per query. Even if the whole PDF fits, retrieval usually gives better answers than context-stuffing (avoids "lost in the middle" and reduces cost).

**A9.** (1) **RAG** — chunk, embed, retrieve relevant sections per query, generate against retrieved subset. (2) **Map-reduce summarization** — chunk, summarize each chunk, then combine summaries into a final answer. (3) **Iterative refinement** — process one chunk at a time, updating a running summary/answer. (4) **Route to a larger-context model** (Gemini 1M, Claude Sonnet 200k+).

**A10. (b).** LLMLingua (and cousins) train a smaller model to predict which tokens carry the most information for downstream use, and drop the rest. Often 2–5× compression with modest quality loss.

---

## 6. Practice Prompts

1. **Context budget calculator.** Write a function that, given a system prompt, history messages, tool schemas, and target output length, tells you whether the request fits in a given model's context.
2. **Prompt caching test.** Make 3 identical requests with a 20k-token system prompt using Anthropic prompt caching. Measure token prices and time-to-first-token. Compare to 3 uncached requests.
3. **Lost-in-the-middle demo.** Build a needle-in-haystack test: hide a specific fact at position 10%, 50%, 90% of a 30k-token prompt. Ask a model to retrieve it. Compare accuracy.
4. **Conversation compression.** Build a hybrid history manager: last-3-turns full + summary of older turns + retrieval of relevant history. Compare quality to plain truncation.
5. **LLMLingua-style pruning.** On a 5k-token document, use `llmlingua` or the smaller-model heuristic to prune to 2k tokens. Ask the same question against both. Compare answer quality.

---

## 7. References

- Liu et al., ["Lost in the Middle"](https://arxiv.org/abs/2307.03172) (2023).
- Anthropic docs: [Prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching).
- OpenAI docs: [Prompt caching](https://platform.openai.com/docs/guides/prompt-caching).
- Jiang et al., ["LLMLingua"](https://arxiv.org/abs/2310.05736) (2023) — prompt compression.
- Google's "Needle in a Haystack" benchmarks (Gemini team, ongoing).
