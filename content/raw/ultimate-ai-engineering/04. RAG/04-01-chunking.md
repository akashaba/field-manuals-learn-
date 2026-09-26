# 04.01 — Chunking

> **Module 4: RAG** · Subtopic 1 of 5
> **Prerequisites:** 03.01 (embeddings, max sequence length, late chunking), 03.03 (deterministic chunk IDs, CDC re-chunking), 02.02 (context assembly), 01.03 (token counting).
> **Outcome:** you treat chunking as a measurable design decision, not a default parameter. You can parse messy documents faithfully, choose and combine chunking strategies by document type, decouple the *retrieval unit* from the *generation unit*, and evaluate chunkers at a **fixed token budget** so comparisons are fair.

---

## 1. Why Chunking Decides RAG Quality

A RAG system can only answer from what it retrieves, and it can only retrieve **units you created at indexing time**. Chunking sets three things at once:

1. **What an embedding represents.** A chunk that mixes three topics becomes a blurry average (03.01 §5), while a chunk that is too small loses the context needed to match the query.
2. **What the generator sees.** A fragment like "the amount specified in subsection (2)" is useless without subsection (2).
3. **Cost.** More, smaller chunks mean more vectors, more storage, and more rerank calls. Larger chunks mean more prompt tokens per retrieved item.

```
      too small                                   too large
 ┌───────────────────┐                     ┌───────────────────────────┐
 │ precise embedding │                     │ context-complete          │
 │ high ranking prec.│                     │ fewer boundary cuts       │
 │ BUT: missing refs,│    sweet spot is    │ BUT: diluted embedding,   │
 │ broken sentences, │◄── document- and ──►│ wasted prompt tokens,     │
 │ answer split over │    query-dependent  │ lost-in-the-middle inside │
 │ several chunks    │                     │ the chunk itself          │
 └───────────────────┘                     └───────────────────────────┘
```

**The key architectural idea:** the unit you **retrieve** (small, precise, well-embedded) does not have to be the unit you **generate from** (larger, context-complete). Parent–child retrieval, sentence windows, and contextual headers all exploit this split (§4).

---

## 2. Parsing Comes First

Chunking garbage produces well-sized garbage. Most "chunking problems" in production are really **parsing** problems.

| Source | Common failures | Fixes |
|---|---|---|
| **PDF (digital)** | Multi-column reading order; headers, footers, and page numbers inside sentences; hyphenation at line ends; footnotes interleaved | Layout-aware parsers; strip repeated page furniture by frequency; de-hyphenate; keep page numbers as metadata |
| **PDF (scanned)** | OCR errors, especially on numbers and section symbols (§) | OCR with confidence scores; validate numbers and citations with regexes; flag low-confidence pages |
| **Tables** | Rows split from their headers; cells flattened into unreadable text | Serialise each row *with* its column headers ("Fund: General; FY2026: $1.2M"), or keep the table as Markdown and chunk it as a unit |
| **HTML** | Navigation, cookie banners, boilerplate | Main-content extraction; DOM-based sectioning by `<h1>`–`<h6>` |
| **Legislative / legal text** | **Amendatory markup**: new language is <u>underlined</u>, deleted language ~~struck through~~. Plain-text extraction merges both, so the model reads repealed language as law | Convert the formatting into explicit markers (`[NEW]…[/NEW]`, `[DELETED]…[/DELETED]`), or index the "as amended" and "current law" versions separately |
| **Code** | Splits in the middle of functions | AST-based chunking (function/class units + a file-path header) |
| **Email / tickets** | Quoted replies repeated many times | Strip quoted history; thread-level deduplication |

**Keep provenance on every chunk:** source URI, document version, page, section path, and character offsets into the normalised text. Citations, highlighting, and evaluation (§6) all depend on offsets.

---

## 3. Chunking Strategies

### 3.1 Fixed-size token windows with overlap

Windows of $c$ tokens with stride $s$ (overlap $o = c - s$). This is simple and predictable, and structure-blind.

**How much overlap do you need?** Suppose a fact spans $a$ tokens and starts at a uniformly random position. Chunks start at multiples of $s$. The fact is fully contained in the chunk that starts at or just before it exactly when its offset $r = p \bmod s$ satisfies $r + a \le c$. So

$$
P(\text{fully contained}) = \frac{\min\!\big(s,\ \max(0,\ c - a + 1)\big)}{s} = \min\!\left(1,\ \frac{o + s - a + 1}{s}\right)^{+}
$$

**Implication:** an overlap $o \ge a - 1$ guarantees that every span of up to $a$ tokens lies wholly inside some chunk. With $c = 512$ and $s = 448$ ($o = 64$), any fact of ≤ 65 tokens is never cut. Larger overlap buys containment for longer facts, at the cost of $c/s$ times more chunks.

```python
def containment_probability(chunk: int, stride: int, span: int) -> float:
    """P(a random span of `span` tokens lies wholly inside at least one chunk)."""
    return min(stride, max(0, chunk - span + 1)) / stride


def simulate_containment(chunk, stride, span, doc_len=100_000, trials=20_000, seed=0):
    import random
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        p = rng.randrange(0, doc_len - span)
        start = (p // stride) * stride                   # latest chunk starting at or before p
        hits += p + span <= start + chunk
    return hits / trials
```

### 3.2 Recursive splitting

Split on the strongest boundary that fits, recursing to weaker ones: sections → paragraphs → sentences → words. This respects natural boundaries when possible and falls back gracefully. Use a **token counter** (01.03), not character counts, because budgets are measured in tokens.

```python
import re
from typing import Callable

SEPARATORS = ["\n\n\n", "\n\n", "\n", r"(?<=[.!?;:])\s+", " "]


def recursive_split(text: str, max_tokens: int, count: Callable[[str], int],
                    seps: list[str] = SEPARATORS) -> list[str]:
    if count(text) <= max_tokens:
        return [text] if text.strip() else []
    if not seps:                                           # hard cut on words as a last resort
        words, out, cur = text.split(), [], []
        for w in words:
            if cur and count(" ".join(cur + [w])) > max_tokens:
                out.append(" ".join(cur))
                cur = []
            cur.append(w)
        return out + ([" ".join(cur)] if cur else [])
    sep, rest = seps[0], seps[1:]
    parts = [p for p in re.split(sep, text) if p.strip()]
    if len(parts) == 1:
        return recursive_split(text, max_tokens, count, rest)
    joiner = sep if not sep.startswith("(") else " "
    chunks, cur = [], ""
    for part in parts:
        cand = (cur + joiner + part) if cur else part
        if count(cand) <= max_tokens:
            cur = cand
            continue
        if cur:
            chunks.append(cur)
        if count(part) > max_tokens:
            chunks.extend(recursive_split(part, max_tokens, count, rest))
            cur = ""
        else:
            cur = part
    if cur:
        chunks.append(cur)
    return chunks
```

### 3.3 Structure-aware chunking

Use the document's own hierarchy — headings, statute numbering (Title › Chapter › Part › Section › Subsection), Markdown/HTML headings, or code ASTs. Each chunk carries its **section path**, which is prepended as a contextual header (§4.2) and stored as filterable metadata.

```python
import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.*)$|^(\d{1,2}-\d{1,3}-\d{1,4})\.\s+(.*)$", re.M)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_path: list[str]
    start: int                      # char offsets into the normalised document
    end: int
    parent_id: str | None = None
    meta: dict = field(default_factory=dict)


def structure_chunks(doc_id: str, text: str, max_tokens: int, count) -> list[Chunk]:
    """Split on Markdown headings or statute section numbers; sub-split oversized sections."""
    marks = [(m.start(), m) for m in HEADING.finditer(text)] + [(len(text), None)]
    chunks, path = [], []
    if marks[0][0] > 0:                                     # preamble before the first heading
        marks.insert(0, (0, None))
    for (start, m), (end, _) in zip(marks, marks[1:]):
        if m is not None:
            if m.group(1):                                  # markdown: depth = number of '#'
                depth, title = len(m.group(1)), m.group(2).strip()
            else:                                           # statute section: fixed depth
                depth, title = 3, f"{m.group(3)} {m.group(4).strip()}"
            path = path[:depth - 1] + [title]
        body = text[start:end]
        section_id = f"{doc_id}#{len(chunks)}"
        pieces = recursive_split(body, max_tokens, count) or []
        cursor = start
        for i, piece in enumerate(pieces):
            s = text.find(piece, cursor)
            s = s if s >= 0 else cursor
            chunks.append(Chunk(f"{section_id}.{i}", piece, list(path), s, s + len(piece),
                                parent_id=section_id if len(pieces) > 1 else None))
            cursor = s + len(piece)
    return chunks
```

### 3.4 Semantic chunking

Embed consecutive sentences, and cut where the **cosine distance between neighbours** spikes (a topic shift). A cut goes where

$$
\delta_i = 1 - \cos(\mathbf{e}_i, \mathbf{e}_{i+1}) > \mathrm{percentile}_{p}\big(\{\delta_j\}\big)
$$

It works well for unstructured prose (transcripts, essays), and adds little over structure-aware chunking for well-structured documents. It costs one embedding pass per sentence at index time.

```python
import numpy as np


def semantic_chunks(sentences: list[str], embed, pct: float = 90.0, min_sentences: int = 2,
                    max_sentences: int = 30) -> list[str]:
    E = np.asarray(embed(sentences), dtype=float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    dist = 1 - (E[:-1] * E[1:]).sum(1)                        # distance between neighbours
    thresh = np.percentile(dist, pct) if len(dist) else 1.0
    chunks, cur = [], [sentences[0]]
    for i, d in enumerate(dist, start=1):
        if (d > thresh and len(cur) >= min_sentences) or len(cur) >= max_sentences:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(sentences[i])
    return chunks + [" ".join(cur)]
```

### 3.5 Proposition (atomic fact) chunking

An LLM rewrites each passage into **self-contained propositions**: atomic facts with coreferences resolved ("The act takes effect July 1, 2027"). Indexing propositions improved retrieval in Chen et al. (2023, *Dense X Retrieval*). **Costs:** an LLM call per passage, a risk of introducing errors, and many more index entries. Keep a link from each proposition to its source passage and generate from the passage (§4.1).

### 3.6 Hierarchical summaries (RAPTOR)

Cluster the chunks, summarise each cluster with an LLM, then recursively cluster and summarise the summaries into a tree. Index **all levels**. Detailed questions match leaves, and thematic questions match higher-level summaries (Sarthi et al., 2024). This is related to GraphRAG's community summaries (04.05), but organised by semantic clusters rather than entity graphs.

### 3.7 Choosing by document type

| Document type | Default strategy |
|---|---|
| Statutes, contracts, regulations, manuals | **Structure-aware** by section + contextual headers + parent–child |
| Long prose without structure (transcripts, essays) | **Semantic** or recursive; sentence windows |
| FAQs, glossaries, tickets | One Q/A or entry per chunk |
| Tables and spreadsheets | Row-with-headers serialisation, or a table as a unit + a table summary |
| Code | AST units (function/class) + file path + imports header |
| Very long documents needing thematic answers | Add a RAPTOR-style summary tree |

---

## 4. Decoupling Retrieval Units from Generation Units

### 4.1 Parent–child ("small-to-big") and sentence windows

Index small **child** chunks (a sentence group or subsection) for precise matching. At generation time, return the **parent** (the whole section), or a window of ±n sentences around the hit. When several children of one parent are retrieved, merge them into a single parent context, which saves tokens and removes duplicates.

```python
def expand_to_parents(hits: list[str], chunks: dict[str, "Chunk"], parents: dict[str, str],
                      budget_tokens: int, count) -> list[str]:
    """hits: child chunk ids, best first. parents: parent_id -> full parent text.
    Returns the texts to put in context: parents when they fit, else the child itself."""
    out, used, seen = [], 0, set()
    for cid in hits:
        ch = chunks[cid]
        key = ch.parent_id or cid
        if key in seen:
            continue                                        # sibling already covered by its parent
        text = parents.get(ch.parent_id, ch.text) if ch.parent_id else ch.text
        if used + count(text) > budget_tokens:
            text = ch.text                                  # parent too big: fall back to the child
            if used + count(text) > budget_tokens:
                continue
        out.append(text)
        used += count(text)
        seen.add(key)
    return out
```

### 4.2 Contextual chunk headers and contextual retrieval

Prepend a short header to each chunk **before embedding and before BM25 indexing**:

```
[Montana Code Annotated › Title 2 › Chapter 18 › 2-18-303 Procedures to determine health insurance premiums | version: 2025 | status: in force]
<chunk text>
```

**Contextual retrieval** goes further: an LLM writes a 1–2 sentence situating context for each chunk, conditioned on the whole document ("This section describes the employer contribution rules referenced by the 2025 amendment in HB 45"). Anthropic reported large reductions in retrieval failures from contextual embeddings plus contextual BM25, and further gains with reranking. Prompt caching (02.02 §6) makes the per-chunk LLM call affordable, because the document prefix is cached across all its chunks.

### 4.3 Late chunking

Encode the whole document with a long-context embedder, *then* pool token states per chunk span (03.01 §5). Chunk vectors inherit document context without an LLM call. It requires an embedder with a long enough context and access to token-level outputs.

---

## 5. Metadata Is Part of the Chunk

Store, per chunk:
- `doc_id`, `doc_version`, `chunk_id` (deterministic), `ordinal`
- `section_path`, `page`, character offsets
- type, date, status, jurisdiction, ACLs (03.03 §4.1)
- `chunker_version` and `parser_version`

Metadata enables **filters** (04.02, 04.04 self-query), **citations**, **freshness rules**, and **safe re-chunking**. When you change the chunker, bump `chunker_version`, re-chunk in a new index, evaluate, and switch over with an alias (the same pattern as embedding migrations, 03.01 §6.1).

---

## 6. Evaluating Chunkers Fairly

### 6.1 Compare at a fixed token budget, not a fixed k

A chunker with 1,000-token chunks at k = 5 feeds the LLM 5,000 tokens. A 200-token chunker at k = 5 feeds it 1,000. Comparing recall@5 across them is meaningless. Instead, **retrieve until a token budget $B$ is filled**, and measure how much of the gold evidence falls inside.

### 6.2 Token- (or character-) level metrics

Given gold evidence spans $G$ (character offsets) and retrieved spans $R$ within budget $B$:

$$
\text{Recall}_B = \frac{|G \cap R|}{|G|},\qquad
\text{Precision}_B = \frac{|G \cap R|}{|R|},\qquad
\text{IoU}_B = \frac{|G \cap R|}{|G \cup R|}
$$

where $|\cdot|$ counts characters or tokens. Recall is what matters for answer correctness. Precision measures token waste and distraction (02.02 §2).

```python
def span_union(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def span_len(spans):
    return sum(e - s for s, e in span_union(spans))


def span_intersection(a, b):
    a, b, i, j, out = span_union(a), span_union(b), 0, 0, []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def budgeted_span_metrics(ranked_chunks: list["Chunk"], gold: list[tuple[int, int]], budget_chars: int):
    """ranked_chunks: best first, all from ONE document's offset space (extend per doc for corpora)."""
    picked, used = [], 0
    for ch in ranked_chunks:
        size = ch.end - ch.start
        if used + size > budget_chars:
            continue
        picked.append((ch.start, ch.end))
        used += size
    inter = span_len(span_intersection(picked, gold))
    g, r = span_len(gold), span_len(picked)
    union = g + r - inter
    return {"recall": inter / g if g else 0.0, "precision": inter / r if r else 0.0,
            "iou": inter / union if union else 0.0, "used_chars": used}
```

### 6.3 End-to-end checks

Token-level retrieval metrics predict answer quality, but also run the full RAG evaluation (answer correctness and groundedness). Chunking choices affect generation directly — for example, a parent context helps the model resolve cross-references.

---

## 7. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Cross-references unresolved** | "As provided in subsection (3)…" answers are incomplete | Parent–child expansion; resolve and append referenced sections at indexing time (a citation graph, 04.05) |
| **Deleted language treated as law** | Answers cite struck text from bills | Preserve amendatory markup as explicit markers; separate as-amended and current-law indexes (§2) |
| **Tables mangled** | Wrong numbers in answers | Row-with-headers serialisation; table-level chunks + summaries; numeric validation |
| **Header/footer noise** | Page furniture in the middle of chunks and embeddings | Frequency-based furniture stripping; layout-aware parsing |
| **Silent truncation at embed time** | Chunks longer than the embedder's max length | Chunk by the *embedder's* tokenizer; assert length ≤ max (03.01 §5) |
| **Re-chunking cost** | Changing the chunker requires re-embedding everything | Version chunkers; evaluate offline on a sample first; alias cutover; hash-based skipping (03.03 §5) |
| **Unfair chunker comparisons** | "Bigger chunks win" (they just use more tokens) | Fixed-token-budget evaluation (§6) |
| **Duplicate content across documents** | The same boilerplate retrieved repeatedly | Near-duplicate detection (MinHash) at indexing; boilerplate stripping |

---

## 8. Hands-On Projects

### Project 1 — Chunking Strategy Benchmark at a Fixed Token Budget

**User stories**
- *As a RAG engineer*, I want to know which chunking strategy gives the best evidence recall per prompt token on our statutes and bills, so that we pick a chunker on evidence and stop debating chunk sizes.

**Acceptance criteria**
1. A corpus of ≥ 500 documents (statutes, bills, fiscal notes, or another structured domain), plus ≥ 200 questions with **gold evidence spans** (character offsets).
2. Chunkers compared:
   - Fixed windows (256/512/1024 tokens × overlap 0/64/128).
   - Recursive.
   - Structure-aware.
   - Semantic.
   - Structure-aware + contextual headers.
   - Structure-aware + LLM contextual retrieval.
   - Parent–child (child 128–256 tokens, parent = section).
   - Proposition chunking on a 20% subset.
3. For each chunker: token-level recall, precision, and IoU at budgets $B$ ∈ {1k, 2k, 4k} tokens (§6.2), using the same embedder and the same hybrid retriever. Also reports index size and indexing cost.
4. End-to-end answer accuracy for the top 3 chunkers, with paired significance tests.
5. The containment formula (§3.1) is validated against simulation and against the measured split-evidence rate for the fixed chunkers.

**Step-by-step**
1. Normalise the documents and store the text with offsets. Build a span-annotation tool (or use Label Studio) for the gold evidence.
2. Implement the chunkers from §3, with `Chunk` offsets preserved.
3. Index each variant into a separate collection (pgvector or FAISS + BM25) with identical retrieval code.
4. Implement `budgeted_span_metrics` across documents: keep offsets per doc and aggregate.
5. Run the grid; plot recall vs budget per chunker; pick the top 3 for end-to-end QA.
6. Write the recommendation, including indexing cost and re-chunking implications.

---

### Project 2 — Faithful Document Parsing Pipeline (PDF, Tables, Amendatory Markup)

**User stories**
- *As a legislative services developer*, I want bills and statutes parsed with amendment markup, tables, and section structure preserved, so that the RAG system never presents deleted language as current law.

**Acceptance criteria**
1. Handles digital PDFs, scanned PDFs (OCR), HTML, and DOCX. Outputs normalised text plus a structure tree (sections with paths) and offset-preserving provenance.
2. Amendatory formatting (underline/strikethrough in PDF or HTML) is converted to explicit markers with ≥ 98% accuracy on a 50-page hand-checked sample. Produces both an "as amended" view and a "current law" view.
3. Tables are extracted with their headers and serialised row-wise. Numeric values are cross-checked against the source on a sample (≥ 99% exact).
4. Page furniture removal and de-hyphenation, verified by tests on known tricky pages.
5. Parser quality metrics are logged per document (OCR confidence, table count, marker count), and documents below thresholds go to a review queue.

**Step-by-step**
1. Evaluate parsers on 30 representative documents: layout-aware PDF parsers, OCR engines, HTML main-content extraction. Pick one per format.
2. For PDFs, read character-level font and decoration info (underline and strike annotations, or line objects overlapping text) to detect amendments. For HTML, map `<u>`/`<ins>` and `<s>`/`<del>` tags.
3. Build the structure tree from section-number patterns and headings, and validate the numbering sequences.
4. Implement table serialisation and numeric checks.
5. Wire the output into the 04.01 chunkers and re-run Project 1's evaluation to measure the gain from better parsing.

---

### Project 3 — Hierarchical Retrieval for Long Documents (Parent–Child + RAPTOR)

**User stories**
- *As a policy analyst*, I want answers to both detailed questions ("what is the penalty in 45-5-207?") and thematic ones ("how does this title treat juvenile offenders overall?") from the same system.

**Acceptance criteria**
1. Builds a parent–child index and a RAPTOR-style summary tree (≥ 3 levels) over the long documents.
2. Question set: ≥ 100 detail questions and ≥ 50 thematic questions, with graded reference answers.
3. Compares flat chunks, parent–child, RAPTOR (all levels indexed), and parent–child + RAPTOR, on answer quality (LLM judge validated against human labels), tokens, and indexing cost.
4. Shows per question type which strategy wins, and implements a router (04.04) that picks the index level by question type.

**Step-by-step**
1. Reuse the structure-aware chunks; define parents at the section and part levels.
2. Implement the RAPTOR build: embed → cluster (GMM or k-means on reduced dimensions) → summarise each cluster with an LLM (cached prompts) → repeat until one root.
3. Index all levels with a `level` metadata field.
4. Implement `expand_to_parents` for the parent–child variant.
5. Evaluate, then train or configure a simple router, and re-evaluate.

---

## 9. Foundational Papers & Reading (exact titles)

- Chen et al., 2023 — *Dense X Retrieval: What Retrieval Granularity Should We Use?*
- Sarthi et al., 2024 — *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*
- Günther et al., 2024 — *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*
- Liu et al., 2023 — *Lost in the Middle: How Language Models Use Long Contexts*
- Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Gao et al., 2023 — *Retrieval-Augmented Generation for Large Language Models: A Survey*
- Anthropic Engineering, 2024 — *Introducing Contextual Retrieval* (blog)
- Chroma Research, 2024 — *Evaluating Chunking Strategies for Retrieval* (technical report; token-level IoU methodology)
- Barnett et al., 2024 — *Seven Failure Points When Engineering a Retrieval Augmented Generation System*

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **Docling, Unstructured, PyMuPDF, pdfplumber** | Layout-aware parsing, tables, character-level font info |
| **Tesseract / cloud OCR (Azure Document Intelligence, AWS Textract, Google Document AI)** | Scanned documents, forms, tables |
| **trafilatura / readability** | HTML main-content extraction |
| **tree-sitter** | AST-based code chunking |
| **LangChain / LlamaIndex text splitters, chonkie** | Reference splitters (read their code; own your production chunker) |
| **tiktoken / HF tokenizers** | Token-accurate budgeting |
| **datasketch** | MinHash near-duplicate detection |
| **Label Studio** | Gold evidence span annotation |
