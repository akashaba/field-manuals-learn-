# 04.04 — Query Rewriting

> **Module 4: RAG** · Subtopic 4 of 5
> **Prerequisites:** **03.04 §7** (query understanding: identifier routing, HyDE, Query2doc, multi-query — the introduction), 02.03 (structured outputs), 02.01 §4 (decomposition, ReAct), 04.02 (filters, as-of retrieval), 04.03 (reranking against the rewritten question).
> **Outcome:** you can build a **query-planning layer** that turns messy user input into retrieval-ready plans: standalone questions, validated metadata filters, expansions, sub-question graphs, and retrieval/no-retrieval decisions. You can guard it against drift and hallucinated constraints, and keep the added latency close to zero.

---

## 1. The Query Planner

Users write queries for humans ("what about the Senate version?", "same thing but for 2019"), not for retrievers. A **query planner** sits between the user and retrieval:

```
 user turn + chat history + user context (role, jurisdiction, today's date)
                    │
          ┌─────────▼──────────┐
          │   QUERY PLANNER    │  one structured-output LLM call (small, fast model) + deterministic checks
          │ 1 resolve context  │  → standalone question
          │ 2 classify / route │  → no_retrieval | lookup | search | multi_hop | structured(SQL/graph)
          │ 3 extract filters  │  → {session: 2025, status: "enacted", as_of: 2019-06-30}
          │ 4 extract ids      │  → ["HB-45", "2-18-303"]
          │ 5 expand           │  → keywords, paraphrases, hypothetical answer (optional)
          │ 6 decompose        │  → sub-question DAG with dependencies
          └─────────┬──────────┘
                    │ VALIDATE: ids preserved? filters in allowed values? drift check?
                    ▼
    retrieval executors (04.02) ──► rerank against the STANDALONE question (04.03) ──► generator
```

**Design principles:**
- **One planner call, structured output**, not a chain of ad-hoc rewrites. It is cheaper, more consistent, and testable.
- **Keep the original query** in the lexical leg and in logs. Rewriting is additive, not a replacement.
- **Deterministic before generative:** extract identifiers, dates, and known entities with rules first, and let the LLM handle what the rules can't.
- **Validate everything the LLM produces** against allowed values and the original text.

---

## 2. Conversational Reformulation

Multi-turn RAG must resolve **coreference** ("it", "that bill"), **ellipsis** ("and the Senate version?"), and **topic shifts**. The target is a **standalone question** that retrieves correctly without the conversation.

| Turn | Raw | Standalone |
|---|---|---|
| 1 | "What does HB 45 change about county health insurance?" | same |
| 2 | "Did it pass?" | "Did HB 45 (2025 session, county health insurance) pass?" |
| 3 | "What about the Senate companion?" | "What is the Senate companion bill to HB 45 (2025) and what does it change?" |

**Rules for the rewriter prompt:**
- Include only what is needed from the history. Carry over identifiers and constraints, not the whole conversation.
- **Never add constraints the user didn't state** (dates, jurisdictions). If one is ambiguous, keep the question ambiguous, or ask a clarifying question.
- Detect **topic shifts** so that stale constraints from earlier turns aren't carried forward.

---

## 3. Expansion: Bridging the Vocabulary Gap

### 3.1 Multi-query and RAG-Fusion

Generate $n$ paraphrases $q_1, \ldots, q_n$, retrieve for each, and fuse the results (RRF). Probabilistically, this approximates marginalising retrieval over the ways the need could be phrased:

$$
p(d \mid \text{need}) \approx \frac{1}{n}\sum_{i=1}^{n} p(d \mid q_i)
$$

### 3.2 HyDE (hypothetical document embeddings)

Generate one or more hypothetical answers $h_j$, and search with a vector that mixes them with the query:

$$
\mathbf{v} = \mathrm{normalize}\Big(\alpha\, \mathbf{e}(q) + (1-\alpha)\,\frac{1}{m}\sum_{j=1}^{m} \mathbf{e}(h_j)\Big)
$$

The hypothetical answer "looks like" a relevant document in embedding space, even if its facts are wrong. **Use it for the dense leg only.** Hallucinated terms (fake section numbers, invented names) must never enter the lexical leg or the filters.

### 3.3 Keyword and term expansion

Generate domain synonyms and statutory terms of art ("DUI" → "driving under the influence"; "fund the sheriff" → "appropriation", "county budget") for the lexical leg. A curated **synonym dictionary** is often more reliable than LLM expansion for stable domain vocabularies. Use the LLM to *propose* dictionary entries offline.

```python
import numpy as np


def hyde_vector(q_emb: np.ndarray, hyp_embs: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    v = alpha * q_emb / np.linalg.norm(q_emb) + (1 - alpha) * (hyp_embs / np.linalg.norm(hyp_embs, axis=1, keepdims=True)).mean(0)
    return v / np.linalg.norm(v)


def multi_query_rrf(queries: list[str], search, k: int = 60, per_query: int = 50) -> list[tuple[str, float]]:
    """search(q, n) -> ranked doc ids. Fuses results across paraphrases with RRF."""
    scores: dict[str, float] = {}
    for q in queries:
        for rank, d in enumerate(search(q, per_query), start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
```

---

## 4. Decomposition for Multi-Hop Questions

"Which committees heard bills sponsored by the author of HB 45 in 2025?" requires chained lookups:

```
 q1: Who sponsored HB 45 (2025)?                       ─┐
 q2: Which 2025 bills were sponsored by {q1}?            ├─ dependent chain
 q3: Which committees heard each bill in {q2}?          ─┘
```

- **Static decomposition:** the planner emits a DAG of sub-questions with dependencies. The executor answers them in topological order, substituting earlier answers into later questions.
- **Interleaved retrieval and reasoning** (IRCoT, Self-Ask, ReAct): the model decides the next retrieval step from what it has found so far. This is more robust when the path isn't knowable up front, but costs more calls.
- **Step-back prompting:** first retrieve for a more general question ("How are county employee health benefits governed?"), then the specific one. This helps when the answer depends on principles or definitions.
- **Structured routes:** if the question is really a database query ("how many bills did X sponsor?"), route it to text-to-SQL or text-to-Cypher (02.04 Project 2, 04.05) instead of text retrieval.

```python
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SubQ:
    id: str
    text: str                                   # may contain {{other_id}} placeholders
    depends_on: list[str] = field(default_factory=list)


def execute_plan(subqs: list[SubQ], answer_fn, max_steps: int = 8) -> dict[str, str]:
    """answer_fn(question) -> short answer (retrieve + read). Runs in dependency order."""
    done: dict[str, str] = {}
    pending = {s.id: s for s in subqs}
    steps = 0
    while pending:
        ready = [s for s in pending.values() if all(d in done for d in s.depends_on)]
        if not ready:
            raise ValueError(f"cyclic or unsatisfiable dependencies: {list(pending)}")
        for s in ready:
            steps += 1
            if steps > max_steps:
                raise RuntimeError("step budget exceeded")
            q = re.sub(r"\{\{(\w+)\}\}", lambda m: done[m.group(1)], s.text)
            done[s.id] = answer_fn(q)
            del pending[s.id]
    return done


def iterative_retrieval(question: str, retrieve, step: Callable[[str, list[str]], dict],
                        max_hops: int = 4) -> dict:
    """IRCoT-style loop. step(question, evidence) -> {'next_query': str|None, 'answer': str|None}."""
    evidence: list[str] = []
    query = question
    for hop in range(max_hops):
        evidence.extend(retrieve(query))
        out = step(question, evidence)
        if out.get("answer") is not None:
            return {"answer": out["answer"], "hops": hop + 1, "evidence": evidence}
        query = out.get("next_query") or question
    return {"answer": None, "hops": max_hops, "evidence": evidence}
```

---

## 5. Self-Query: Natural Language → Validated Filters

"Enacted bills from the 2023 session about water rights sponsored by Senator Smith" should become:

```json
{"semantic_query": "water rights", "filters": {"session": 2023, "status": "enacted",
 "sponsor": "Smith", "chamber": "senate"}}
```

**Validation is mandatory.** The LLM will invent plausible values:
- Check enums against the index's allowed values (`status ∈ {introduced, enacted, vetoed, …}`).
- Resolve entity names against a directory (fuzzy match with a confidence threshold; ask when ambiguous).
- Check ranges (a session year must exist in the corpus).
- **Unvalidated filters are dropped, not guessed.** A wrong filter silently returns zero or wrong results. That is worse than no filter.

```python
from typing import Callable, Literal

from pydantic import BaseModel, Field


class Filters(BaseModel):
    session: int | None = None
    status: Literal["introduced", "in_committee", "passed", "enacted", "vetoed"] | None = None
    sponsor: str | None = None
    chamber: Literal["house", "senate"] | None = None
    as_of: str | None = Field(None, description="ISO date if the user asked about a point in time")


class QueryPlan(BaseModel):
    reasoning: str                                      # first: think before deciding (02.03 §3.1)
    standalone_question: str
    route: Literal["no_retrieval", "lookup", "search", "multi_hop", "structured"]
    identifiers: list[str] = []
    filters: Filters = Filters()
    keywords: list[str] = []
    paraphrases: list[str] = []
    sub_questions: list[dict] = []


def validate_plan(plan: QueryPlan, original: str, known_sessions: set[int], sponsor_lookup: Callable[[str], str | None],
                  extract_ids: Callable[[str], list[str]]) -> tuple[QueryPlan, list[str]]:
    """Deterministic guards on the LLM plan. Returns (possibly corrected plan, warnings)."""
    warnings = []
    orig_ids = set(extract_ids(original))
    missing = orig_ids - set(extract_ids(plan.standalone_question))
    if missing:                                          # rewrite dropped an identifier: repair it
        plan.standalone_question += " (" + ", ".join(sorted(missing)) + ")"
        warnings.append(f"restored identifiers {sorted(missing)}")
    invented = set(plan.identifiers) - orig_ids - set(extract_ids(plan.standalone_question))
    if invented:
        plan.identifiers = [i for i in plan.identifiers if i not in invented]
        warnings.append(f"dropped invented identifiers {sorted(invented)}")
    f = plan.filters
    if f.session is not None and f.session not in known_sessions:
        warnings.append(f"unknown session {f.session}; filter dropped")
        f.session = None
    if f.sponsor is not None:
        resolved = sponsor_lookup(f.sponsor)
        if resolved is None:
            warnings.append(f"unresolved sponsor '{f.sponsor}'; filter dropped")
        f.sponsor = resolved
    return plan, warnings
```

---

## 6. Adaptive Retrieval: When (Not) to Retrieve

Not every turn needs retrieval ("thanks!", "make that shorter"), and some need several rounds. Routing by estimated complexity (Adaptive-RAG) saves latency and cost. Related research lets the model decide mid-generation:

- **FLARE:** retrieve when the next sentence contains low-confidence tokens.
- **Self-RAG:** trained reflection tokens that decide whether to retrieve and critique relevance.
- **CRAG:** a retrieval evaluator triggers correction or web search when the retrieved documents look poor.

**Production pattern:** a cheap router (rules plus a small model, or a field in the planner output) decides between `no_retrieval | single-shot | multi_hop | structured`. Log its decisions and audit samples: false "no retrieval" decisions produce ungrounded answers.

---

## 7. Guarding Against Rewrite Drift

Rewrites fail in characteristic ways:

| Failure | Example | Guard |
|---|---|---|
| **Dropped constraint** | "penalty for a *second* offense" → "penalty for the offense" | Constraint/entity preservation check; keep the original in the lexical leg |
| **Invented constraint** | Adds "2025" when the user didn't say it | Filters only from spans present in the user text or history; validation (§5) |
| **Identifier corruption** | HB 45 → HB 54 | Deterministic identifier extraction + equality check |
| **Topic drift** | The paraphrase changes the question | Embedding similarity to the original ≥ threshold, else fall back |
| **Stale context** | A constraint from 5 turns ago carried into a new topic | Topic-shift detection; limit the history window |

```python
import numpy as np


def drift_guard(orig_emb: np.ndarray, rewrite_emb: np.ndarray, orig_ids: set[str], rewrite_ids: set[str],
                min_sim: float = 0.75) -> tuple[bool, str]:
    sim = float(orig_emb @ rewrite_emb / (np.linalg.norm(orig_emb) * np.linalg.norm(rewrite_emb)))
    if not orig_ids <= rewrite_ids:
        return False, f"identifiers lost: {sorted(orig_ids - rewrite_ids)}"
    if sim < min_sim:
        return False, f"semantic drift (cos={sim:.2f})"
    return True, "ok"
```

---

## 8. Latency: Making Rewriting Nearly Free

$$
T_{\text{sequential}} = T_{\text{plan}} + T_{\text{retrieve}} \qquad\text{vs.}\qquad
T_{\text{speculative}} = \max\big(T_{\text{plan}},\ T_{\text{retrieve}}(q_{\text{raw}})\big) + T_{\text{retrieve}}(\Delta)
$$

- **Speculative retrieval:** start retrieving with the raw query *while* the planner runs. When the plan arrives, retrieve only for the extra queries ($\Delta$) and fuse. For clear first-turn queries the raw results are often already sufficient.
- **Skip the planner** when rules show the query is standalone (first turn, no pronouns, no ellipsis) and no filters are needed.
- **Small, fast models** for planning. Structured output keeps them reliable (02.03).
- **Cache plans** by (normalised turn, compressed history hash).
- **Parallelise** the retrievals for paraphrases and independent sub-questions.

---

## 9. Evaluating Query Rewriting

- **Retrieval impact:** evidence recall @ budget with and without the planner, **per query class** (first-turn, follow-up, multi-hop, filterable). Gains concentrate in follow-ups and filterable queries. First-turn clear queries may *lose* a little from paraphrase noise.
- **Rewrite quality:** identifier and constraint preservation rate (deterministic), and human or LLM-judged faithfulness on a sample.
- **Filter accuracy:** precision and recall of the extracted filters against labelled gold filters. The **invented-filter rate** must be near 0.
- **Routing accuracy:** a confusion matrix of routes; the false "no retrieval" rate.
- **Cost and latency:** added P50/P95 with and without speculative retrieval.

---

## 10. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Follow-up questions fail** | "Did it pass?" retrieves random bills | Conversational reformulation with identifier carry-over |
| **Filters silently wrong** | Zero or irrelevant results | Validate against allowed values and directories; drop unvalidated filters; log them |
| **HyDE hallucinations leak** | Fake citations boosted in BM25 | HyDE only in the dense leg; lexical leg uses the original + validated keywords |
| **Latency added** | +300–800 ms per turn | Speculative retrieval; skip rules; small model; caching |
| **Multi-hop loops** | The agent keeps searching | Hop and step budgets; stop when the answer is supported; dependency DAG |
| **Over-rewriting clear queries** | First-turn recall drops | Route clear queries around the planner; per-class evaluation |
| **Ambiguity hidden** | The rewriter guesses the meaning | Allow a `clarify` route that asks the user when the confidence is low |

---

## 11. Hands-On Projects

### Project 1 — Conversational Query Planner for Multi-Turn Legislative RAG

**User stories**
- *As an attorney*, I want to ask follow-up questions naturally ("did it pass?", "what about the Senate version?") and still get grounded answers about the right bill.

**Acceptance criteria**
1. Implements `QueryPlan` + `validate_plan` (§5), with one structured-output call per turn to a small model, and deterministic identifier extraction.
2. An evaluation set of ≥ 150 multi-turn conversations (3–6 turns) with a gold standalone question and gold evidence per turn, plus a public conversational QA set (e.g. a QReCC or TREC CAsT sample).
3. Reports, per turn type (first, follow-up, topic shift): evidence recall @ 2k tokens with raw, with the rewrite, and with the rewrite + raw fused. Also the identifier preservation rate (target 100%) and the invented-constraint rate (target < 1%).
4. Speculative retrieval (§8) keeps the added P50 latency < 100 ms compared with no planner.
5. A `clarify` route triggers on ambiguous follow-ups, and its precision is measured on a labelled subset.

**Step-by-step**
1. Write the planner prompt with examples of follow-ups, topic shifts, and ambiguous turns. Enforce `QueryPlan` via structured outputs.
2. Implement `validate_plan`, reusing the identifier regexes from 04.02 §7 / 03.04 §7.
3. Wire it into the retrieval API: run the raw retrieval and the planner in parallel, then fuse.
4. Build the conversation evaluation set (seed with real logs; hand-write the gold rewrites).
5. Evaluate per turn type, and analyse the failures.

---

### Project 2 — Multi-Hop RAG: Static Decomposition vs Interleaved Retrieval

**User stories**
- *As a research analyst*, I want questions that chain several facts answered correctly with evidence for each hop.

**Acceptance criteria**
1. Datasets: a public multi-hop sample (e.g. ≥ 300 MuSiQue or HotpotQA questions with their corpora) plus ≥ 50 domain multi-hop questions (sponsor → bills → committees).
2. Systems:
   - Single-shot RAG.
   - Multi-query RAG-Fusion.
   - Static decomposition (`execute_plan`).
   - IRCoT-style interleaving (`iterative_retrieval`).
   - Step-back + decomposition.
3. Metrics: answer exact match / F1 (or a judged score for the domain set), supporting-evidence recall per hop, the number of LLM and retrieval calls, and latency.
4. An error taxonomy: decomposition errors, retrieval misses per hop, and reasoning errors — with counts.
5. A recommendation on when to route to multi-hop, based on planner signals.

**Step-by-step**
1. Index the corpora with your 04.01/04.02 pipeline.
2. Implement each system behind a common interface with call counting.
3. For static decomposition, have the planner emit the sub-questions with `depends_on`. For IRCoT, implement `step()` as a structured call returning `next_query` or `answer`.
4. Run the evaluation, collect traces, and label 100 failures by type.
5. Train or tune a router that sends only true multi-hop questions to the expensive paths, and re-evaluate the cost/quality curve.

---

### Project 3 — Self-Query Filter Extraction with Validation

**User stories**
- *As a search user*, I want to type "vetoed education bills from the 2021 session" and get exactly those, without learning a filter UI.
- *As the system owner*, I want zero silent wrong filters.

**Acceptance criteria**
1. The filter schema covers session, chamber, status, sponsor, committee, subject, and date ranges, with the allowed values loaded from the database at startup.
2. Extraction via structured outputs + deterministic validation (enums, directory resolution with fuzzy matching and thresholds, range checks).
3. On ≥ 300 labelled queries: per-field precision and recall, the invented-filter rate (< 1%), and the clarification rate for ambiguous names.
4. End-to-end: result-set precision against an oracle filter, and comparison with semantic-only search.
5. Every extracted filter is shown to the user as an editable chip in a small UI (React or a simple page), and edits are logged as feedback.

**Step-by-step**
1. Define the `Filters` model and generate its allowed values from the database.
2. Build the extraction prompt with examples covering negation ("not vetoed") and ranges.
3. Implement the validators and the sponsor/committee directory matching (trigram similarity in Postgres, or rapidfuzz).
4. Label the query set, and evaluate the extraction and the end-to-end results.
5. Build the filter-chip UI and the feedback logging.

---

## 12. Foundational Papers (exact titles)

**Rewriting and expansion**
- Ma et al., 2023 — *Query Rewriting for Retrieval-Augmented Large Language Models*
- Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE)
- Wang, Yang & Wei, 2023 — *Query2doc: Query Expansion with Large Language Models*
- Rackauckas, 2024 — *RAG-Fusion: a New Take on Retrieval-Augmented Generation*
- Zheng et al., 2023 — *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*

**Conversational search**
- Vakulenko et al., 2021 — *Question Rewriting for Conversational Question Answering*
- Anantha et al., 2021 — *Open-Domain Question Answering Goes Conversational via Question Rewriting*
- Dalton, Xiong & Callan, 2020 — *TREC CAsT 2019: The Conversational Assistance Track Overview*
- Mao et al., 2023 — *Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search*

**Multi-hop and adaptive retrieval**
- Press et al., 2022 — *Measuring and Narrowing the Compositionality Gap in Language Models* (Self-Ask)
- Trivedi et al., 2023 — *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*
- Jiang et al., 2023 — *Active Retrieval Augmented Generation* (FLARE)
- Asai et al., 2023 — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
- Yan et al., 2024 — *Corrective Retrieval Augmented Generation*
- Jeong et al., 2024 — *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*

**Datasets**
- Yang et al., 2018 — *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*
- Trivedi et al., 2022 — *MuSiQue: Multihop Questions via Single-hop Question Composition*

## 13. Essential Tooling

| Tool | Role |
|---|---|
| **Provider structured outputs / Instructor / Pydantic** | Reliable `QueryPlan` generation |
| **DSPy** | Optimising planner prompts and multi-hop programs against metrics |
| **rapidfuzz / Postgres `pg_trgm`** | Entity and directory resolution for filters |
| **LangGraph / plain asyncio** | Orchestrating speculative retrieval and multi-hop loops |
| **ranx / RAGAS** | Retrieval and RAG metrics per query class |
| **Langfuse / Phoenix / OpenTelemetry** | Tracing planner decisions and routes |
