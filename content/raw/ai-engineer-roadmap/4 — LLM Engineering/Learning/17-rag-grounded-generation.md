# RAG — Grounded Generation — Master Study Guide

> **Track:** LLM Engineering · **Module:** 17
> **Prerequisites:** Modules 12–16.
> **Time budget:** ~5–7 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** With retrieval and reranking behind you, generation is where the LLM synthesizes an answer from the retrieved context. Do this right and you get **grounded, citable, honest** answers. Do this wrong and the model:

- Hallucinates facts not in the retrieved context.
- Confidently answers when it shouldn't.
- Ignores the retrieved context and pulls from its training data.
- Fails to cite sources.
- Blends contradictory sources incoherently.

Grounded generation is a **prompt engineering + design pattern** problem. It's not a magic model feature.

**Fundamental principles you must own:**

1. **Instruct the model to answer ONLY from the provided context.** Explicit is better than implicit.
2. **Require citations.** Every claim should reference its source.
3. **Let the model say "I don't know."** Refusal to hallucinate is a feature.
4. **Present the context clearly** — delimited, numbered, with source metadata.
5. **Fact-checking / verification loops** post-generation catch hallucinations.
6. **Streaming works with grounded generation** — but citations may arrive after the claim text.

If you retain nothing else: **grounded generation = right prompt + right context format + right refusal behavior + right citations. Not a model choice.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Grounded-Generation Prompt

**A production-quality template:**

```
You are a helpful assistant answering questions using ONLY the provided sources below.

Rules:
1. Base every claim strictly on the sources. Do NOT use outside knowledge.
2. Cite each claim with the source ID in square brackets, like [1] or [2, 3].
3. If the sources do not contain enough information to answer, say:
   "I don't have enough information in the provided sources to answer this."
4. If sources contradict each other, note the disagreement and cite both.
5. Do not invent facts, numbers, dates, names, or quotes.

Sources:
<source id="1" doc="handbook.pdf" section="Onboarding" page="4">
{chunk 1 content}
</source>

<source id="2" doc="handbook.pdf" section="Onboarding" page="5">
{chunk 2 content}
</source>

<source id="3" doc="benefits.pdf" section="Health" page="12">
{chunk 3 content}
</source>

Question: {user_question}

Answer:
```

**Design principles in this template:**

1. **Explicit role and rules** at the top.
2. **Sources delimited with XML-like tags** — models learn to respect these boundaries.
3. **Metadata attached to each source** — enables high-quality citations.
4. **Explicit refusal instruction** — permits and normalizes "I don't know."
5. **Ordering: rules → sources → question**. Model reads through and applies the rules to the sources when answering.

**Model choice matters.** Claude, GPT-4, Gemini all follow these instructions well; smaller models (7B–8B) sometimes drift, ignoring the rules under pressure. If you're on a small model, be even more explicit and consider fine-tuning on grounded-generation exemplars.

**Temperature.** For factual Q&A, use `temperature=0` (deterministic, less inventive) or 0.1–0.3. Higher temperatures encourage more elaborate answers but also more hallucination. **Rule: lower temperature for grounded generation.**

---

### 2.2 Citations and Attribution

Different citation formats have different tradeoffs.

**Inline numeric** (most compact):

> The health benefits start on the first of the month following your start date [3]. New employees complete orientation in their first week [1].

- **Pros:** compact; easy to render as links.
- **Cons:** must maintain a source-ID → source-URL mapping for rendering.

**Inline with metadata** (fully self-contained):

> The health benefits start on the first of the month following your start date (Source: `benefits.pdf`, Section: Health, Page 12).

- **Pros:** self-documenting.
- **Cons:** verbose; costs more tokens.

**Post-hoc citation list**:

> The health benefits start on the first of the month following your start date. New employees complete orientation in their first week.
>
> **Sources:**
> [1] handbook.pdf § Onboarding, page 4
> [3] benefits.pdf § Health, page 12

- **Pros:** clean reading experience.
- **Cons:** the model must accurately match claims to sources — some models drift.

**Best practice — combined inline + post-hoc:**

Model outputs inline `[N]` citations; you post-process to render both inline markers and a source-list footer.

**Structured JSON output with citations:**

```json
{
  "answer": "Health benefits start on the first of the month following your start date.",
  "citations": [
    {"source_id": "3", "quote": "Health benefits start on the first of the month following..."}
  ],
  "confidence": "high"
}
```

Enables programmatic citation validation (see 2.5).

**Cite the quote, not just the source.** Ask the model to include the exact quoted snippet it's citing. This is verifiable — you can regex-check the quote appears in the source. Compare:

- "Coverage starts on the first of the month [3]." → cannot verify.
- "Coverage starts on the first of the month [3, quote: 'Health benefits start on the first of the month following']." → verifiable.

**Anthropic's Citations API** and OpenAI's file-search return structured citation objects. Use them when available.

---

### 2.3 Refusal, Uncertainty, and "I Don't Know"

The most under-appreciated feature of a well-designed RAG system is **the ability to refuse or defer** when the context is insufficient. Without this:

- Model hallucinates facts.
- Users trust wrong answers.
- Business decisions get made on false info.

**Prompt patterns that enable refusal:**

- Explicit permission: "If the sources don't contain enough information, say 'I don't have enough information'."
- Provide an example refusal in the prompt.
- Enforce a schema (JSON with an `"answered": true/false` field).

**Structured refusal:**

```json
{
  "answered": false,
  "reason": "The provided sources describe general onboarding but do not specify the health-benefit start date for contractors.",
  "would_help": ["policies on contractor health benefits"]
}
```

Beyond honesty, structured refusal enables downstream logic — retrieval retry with different queries, escalation to a human, or asking the user for clarification.

**Confidence levels.** Ask the model to grade its own confidence:

```
Rate your confidence in the answer:
- "high" — all key facts are in the sources.
- "medium" — most facts are supported but some inference required.
- "low" — significant uncertainty; answer might be wrong.
```

Not reliable in isolation, but a useful signal when combined with citation validation.

**When to force refusal.** Even if the model tries to answer, you can post-check: if no citations were produced, refuse. If cited quotes don't appear in the sources, refuse. If a claim is unsupported by any retrieved chunk, refuse or flag.

---

### 2.4 Handling Contradiction, Freshness, and Multi-Source Synthesis

**Contradicting sources.** Different docs may disagree. Options:

- **Present both** — "Source [1] says X; Source [3] says Y." Let the user decide.
- **Use freshness metadata** — prefer newer documents.
- **Use source authority** — prefer authoritative sources (policy docs > blog posts).
- **Flag** — annotate the answer with "Sources conflict on this point."

**Freshness-aware prompt:**

```
When sources have creation dates, prefer more recent information for claims about current state.
If the most recent source is older than 30 days, note that the information may be out of date.
```

Metadata in the context enables this: include `date` in each source tag.

**Multi-source synthesis.** The model must combine info from multiple chunks. Tips:

- Group by source or topic in the context (chunks 1–3 from doc A, chunks 4–6 from doc B).
- Ask for a **structured** answer: "First, extract all relevant facts with sources. Then synthesize into a coherent answer."
- For comparison / aggregation questions, decompose (Module 15.2) — retrieve per sub-question, then synthesize.

**"Answer with tabular comparison"** — for questions like "compare X and Y," ask for a table:

```json
{
  "comparison": [
    {"attribute": "Health coverage start", "X": "Day 1", "Y": "1st of next month", "source_X": "3", "source_Y": "5"},
    ...
  ]
}
```

Forces the model to be explicit about differences and sources.

---

### 2.5 Post-Generation Verification

Grounded generation still lets models drift. Add a verification pass to catch hallucinations.

**Citation validation:**

```python
def validate_citations(answer, sources):
    # extract claims and their [N] citations
    claims = parse_cited_claims(answer)
    for claim, cite_ids in claims:
        # for each cited source, verify the claim is supported
        supported = any(entails(claim, sources[cid]) for cid in cite_ids)
        if not supported:
            flag(claim, cite_ids)
    return flags
```

`entails` can be:
- **Exact-quote matching** (if the model output quotes) — simple regex.
- **Cross-encoder NLI** — a pretrained natural language inference model classifies entailment / contradiction / neutral.
- **LLM-as-judge** — cheap model checks each claim against its source. Higher quality; costs a call per claim.

**Attribution scoring (AIS / AttrScore).** Given a claim and a source, score how attributable the claim is. Metrics from research (Rashkin et al., 2023; Bohnet et al., 2022).

**Retry on failure.** If verification flags unsupported claims:

- Regenerate with stronger grounding instructions.
- Retrieve more context.
- Refuse ("Sorry, I can't answer this reliably.").

**Faithfulness metrics** (Module 18 covers evaluation) — periodically sample production traffic and score faithfulness offline.

**Guarded generation frameworks:**
- **Guardrails AI, LMQL, Instructor** — force structured output that can be validated.
- **Anthropic's Citations API, OpenAI Structured Outputs** — provider-level constrained outputs with citations attached.
- **Semantic Scholar's citation-aware models** — for research use cases.

**Never** ship grounded generation without at least basic post-hoc citation checking on a sample of outputs. Users trust cited answers more than uncited ones — an incorrect citation is worse than no citation.

---

## 3. Mental Models & Analogies

### 3.1 The "Open-Book Exam" Model

Grounded generation is asking the model to take an **open-book exam**. Rules:

1. You may only use these books.
2. Cite the page for each claim.
3. If the answer isn't in the books, say "the books don't address this."

Without these rules, the model does what an unconstrained student does: mixes textbook memory with unsupported speculation. **Constrained by the book**, the student:
- Uses the material (better answers on covered topics).
- Refuses on uncovered topics (rather than fabricating).
- Provides page numbers (verifiable, gradable).

Every RAG prompt should feel like giving an open-book exam. The model isn't your friend — it's a test-taker under rules.

### 3.2 The "Auditor's Report" Model

A financial auditor doesn't just write conclusions — they write **conclusions with references to the source documents**, and their claims can be audited independently. If the auditor writes "Company X had $10M revenue in Q3," they cite the exact line in the exact financial report.

Grounded RAG is the auditor. Your model isn't just answering — it's producing an auditable trail. That's why citations, quotes, and refusal-when-unsure are critical: they make the answer **falsifiable** and **verifiable**. If an executive doubts a claim, they can trace it to its source.

Without this, RAG is just a fancy chatbot with confidence problems.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Model Will Just Use the Context I Give It"

Not without instructions. LLMs are trained on the whole internet; their default is to pull answers from wherever. **Explicitly instruct** the model to answer ONLY from the provided sources. Also, don't lure it: don't include instructions like "if the context doesn't cover it, use your general knowledge" — that undoes the grounding.

### 4.2 "Citations Are Automatic"

Modern models can cite, but they don't unless asked and shown the format. A prompt without `[N]` markers, source IDs, or an explicit "cite each claim" instruction typically produces uncited answers that read plausibly but have no verifiable provenance.

### 4.3 "One Answer, One Source"

Many answers require info from multiple chunks. Design your prompt and rendering for `[1, 3]` compound citations. Design your retrieval to bring in the right combination of chunks (Module 15 decomposition helps).

---

## 5. Self-Assessment Bank (Grounded Generation)

### Questions

**Q1 (Short answer).** Give three components of a good grounded-generation prompt.

**Q2 (Multiple choice).** For factual Q&A over retrieved documents, temperature should typically be:
- (a) 1.0
- (b) 0.7
- (c) 0 to 0.3
- (d) 2.0 for creativity

**Q3 (Short answer).** Why should you explicitly allow the model to refuse when context is insufficient?

**Q4 (Multiple choice).** Requiring the model to output a quoted snippet with each citation:
- (a) Adds tokens but enables verification against the source text.
- (b) Is redundant.
- (c) Only works for open-source models.
- (d) Slows response but improves nothing.

**Q5 (Short answer).** Describe two ways to validate citations after generation.

**Q6 (Multiple choice).** If the model produces an answer with no citations even though your prompt asks for them, the best fallback is:
- (a) Trust it anyway.
- (b) Retry with a stronger prompt; if still uncited, refuse or flag.
- (c) Ignore citations.
- (d) Use a different model always.

**Q7 (Short answer).** How can you structure the retrieved-context prompt to help the LLM handle sources that contradict each other?

**Q8 (Multiple choice).** Structured refusal (JSON with `answered: false`) enables:
- (a) Nothing; text refusal is the same.
- (b) Downstream logic (retry, escalate to human, ask for clarification).
- (c) Faster inference.
- (d) Better vector search.

**Q9 (Short answer).** Why is grounded generation NOT solved by "just picking a better model"?

**Q10 (Multiple choice).** The best position for the user's question in a grounded-generation prompt is:
- (a) At the very top.
- (b) In the middle of the sources.
- (c) At the bottom, after the rules and sources.
- (d) Doesn't matter.

---

### Answer Key & Detailed Explanations

**A1.** (1) **Explicit rules** telling the model to answer ONLY from the provided sources. (2) **Delimited, numbered sources** with metadata for citation. (3) **Refusal permission** ("If sources don't contain the answer, say I don't know"). (4) **Citation format** (e.g., `[N]`) required per claim. (5) **User question** placed at the end, after rules and sources. Any three plus rationale.

**A2. (c).** Low temperature (0 to 0.3) discourages inventive completions and produces more faithful, deterministic answers. High temperatures increase creativity and, correspondingly, hallucination.

**A3.** Without explicit permission, the model tends to invent an answer rather than say "I don't know" — a bias from training on helpful behavior. Permitting refusal ("say I don't have enough information") normalizes it and makes the model actually use it. Prevents hallucination on out-of-scope questions; makes the system honest.

**A4. (a).** Quoted snippets add tokens (~50–100 per citation) but let you verify the quote appears verbatim in the cited source with a simple string search or regex. Detects fabricated citations, which are a real failure mode.

**A5.** (1) **Exact-quote matching** — if the model quotes, check the quote appears in the source. (2) **NLI-based entailment** — a cross-encoder or NLI model scores whether the source entails the claim. (3) **LLM-as-judge** — a cheap LLM verifies each claim against its source. (4) **Attribution scoring** (research) — dedicated attribution models. Any two plus mechanism.

**A6. (b).** If the model ignored citation instructions on the first try, a retry with a stronger, more explicit prompt (or an example) often works. If it still doesn't cite, flag the answer or refuse — you can't verify uncited claims. In some architectures, you reject and re-generate; in others, you display "unable to attribute; treat with caution."

**A7.** (1) Include metadata like source date and authority per chunk. (2) Add a rule: "If sources contradict, present both perspectives and note the disagreement, citing both." (3) Group chunks by source in the context so the LLM sees which claims come from which document. (4) Optionally ask for a structured output that separates claims by source.

**A8. (b).** Structured refusal (a JSON with `answered: false`) is machine-actionable — you can code retry-with-different-query, escalate-to-human, or ask-user-for-clarification. Text refusal ("I don't know") requires string parsing and is harder to hook into an automated pipeline.

**A9.** Grounded generation is a **prompt + design pattern** problem, not primarily a model problem. Better models help follow instructions more reliably, but even GPT-5 will hallucinate if the prompt doesn't include grounding rules, source delimiters, refusal permission, and citation requirements. The system design (retrieval quality, prompt template, verification) is what makes RAG reliable — model choice is a smaller lever.

**A10. (c).** Rules first (set expectations), sources second (the context), question third (the task). "Lost in the middle" says content in the middle gets less attention, but the question at the *end* is the most-attended position. Also, placing the question last matches the natural read order of "given rules, given docs, now answer this."

---

## 6. Practice Prompts

1. **Prompt template.** Write a production-quality grounded-generation prompt for your build. Include rules, source formatting, and citation format. Test on 20 sample questions.
2. **Refusal test.** Feed the system 10 questions whose answers are NOT in your retrieved context. Confirm the model refuses each with a clean structured refusal.
3. **Citation validator.** Write a validator that (a) extracts `[N]` markers from the model's output, (b) parses cited quotes if the model provides them, (c) checks each quote against the source content. Flag violations.
4. **Contradiction handling.** Inject two contradicting chunks (different dates for the same event). Ask a question. Observe how the model handles it. Improve the prompt.
5. **Structured refusal.** Force the model into a JSON output with `answered: bool`. Build a retry logic: if `answered=false`, retrieve with an alternative query and try again.

---

## 7. References

- Rashkin et al., ["Measuring Attribution in Natural Language Generation Models"](https://arxiv.org/abs/2112.12870) (2021, updated 2023).
- Bohnet et al., ["Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models"](https://arxiv.org/abs/2212.08037) (2022).
- Menick et al., ["Teaching language models to support answers with verified quotes"](https://arxiv.org/abs/2203.11147) (2022) — GopherCite.
- Anthropic Citations API docs.
- Guardrails AI: [guardrailsai.com](https://www.guardrailsai.com/).
- Instructor library: [github.com/jxnl/instructor](https://github.com/jxnl/instructor).
