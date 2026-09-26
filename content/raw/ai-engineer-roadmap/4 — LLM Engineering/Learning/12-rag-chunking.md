# RAG — Chunking — Master Study Guide

> **Track:** LLM Engineering · **Module:** 12 (RAG track begins)
> **Prerequisites:** Modules 01–11.
> **Time budget:** ~5–7 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** In RAG, **chunking is the step that most often makes or breaks quality.** You can have the best embedding model, the fastest vector DB, and the smartest reranker, but if your chunks are wrong — too big, too small, split mid-sentence, missing context — retrieval quality collapses and generation hallucinates.

Chunking is not a "get it done and move on" step. It's a **first-class design decision** — one you should revisit whenever you hit quality issues.

**Fundamental principles you must own:**

1. **Chunks must be self-contained enough to answer questions on their own.** If a chunk says "he then signed the bill," but doesn't name the person, retrieval won't help — the answer isn't in the chunk.
2. **Chunks must be small enough to embed meaningfully.** Very long chunks average into a semantically mushy vector; retrieval loses precision.
3. **Structure matters.** Splitting on arbitrary character counts loses natural boundaries (headings, sentences, code blocks). Structure-aware chunking preserves meaning.
4. **Metadata is chunk gold.** Filename, page number, section title, timestamp, author — every metadata field enables filtering and citation.
5. **Overlap prevents boundary loss.** A key fact split between two chunks may be lost to both without overlap.
6. **Different content types need different chunkers.** Prose, code, tables, PDFs with figures — one size does not fit all.

If you retain nothing else: **chunk with the retrieval and question types in mind, not just the character count.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Fixed-Size Character/Token Chunking

The simplest strategy: split every $N$ characters or tokens, with optional overlap.

```python
def chunk_by_tokens(text, tokenizer, max_tokens=500, overlap=50):
    ids = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(ids), max_tokens - overlap):
        chunk_ids = ids[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks
```

**Tunables:**
- **`max_tokens`**: typical 200–800. Smaller = more precise retrieval but more chunks. Larger = fewer chunks but blunt semantics.
- **`overlap`**: typical 10–20% of chunk size. Ensures boundary content isn't lost.

**Pros:**
- Trivially simple.
- Predictable size (good for token budgets).
- No dependencies.

**Cons:**
- Splits mid-sentence, mid-paragraph, mid-code-block.
- Semantic incoherence at chunk boundaries.

**When to use:** as a **first baseline**. Also fine for very homogeneous prose corpora where structure is minimal.

---

### 2.2 Recursive Character/Structure-Aware Chunking

The **recursive chunker** (popularized by LangChain's `RecursiveCharacterTextSplitter`) tries a hierarchy of separators, preferring larger structural boundaries:

```python
separators = ["\n\n", "\n", ". ", " ", ""]

def recursive_chunk(text, max_size, separators=separators):
    if len(text) <= max_size:
        return [text]
    for sep in separators:
        if sep in text:
            splits = text.split(sep)
            chunks = []
            current = ""
            for piece in splits:
                if len(current) + len(piece) + len(sep) <= max_size:
                    current += (sep if current else "") + piece
                else:
                    chunks.append(current)
                    current = piece
            if current:
                chunks.append(current)
            return chunks
    # Fallback: fixed-size split
    return [text[i:i+max_size] for i in range(0, len(text), max_size)]
```

**How it works:** try to split on paragraph breaks (`\n\n`) first. If a resulting piece is still too big, split those on line breaks. If still too big, on periods. And so on.

**Advantages over fixed-size:**
- Preserves paragraph and sentence boundaries when possible.
- Fewer mid-sentence splits.
- Still respects the size budget.

**Practical settings:**
- `max_size` = 500–1000 tokens (or 2000–4000 chars).
- `overlap` = 10–20%.

**This is the default recommended chunker for most text.**

---

### 2.3 Document-Structure-Aware Chunking

For structured content (Markdown, HTML, code, PDFs with sections), use the structure to guide chunking.

**Markdown-aware:**

```python
# Split on headers, keep headers as context on each chunk
def markdown_chunk(md_text, max_size=800):
    sections = split_on_headers(md_text)   # returns list of (header, body)
    chunks = []
    for header, body in sections:
        # Ensure each chunk carries its section context
        header_prefix = header + "\n\n"
        body_chunks = recursive_chunk(body, max_size - len(header_prefix))
        for chunk in body_chunks:
            chunks.append(header_prefix + chunk)
    return chunks
```

The trick: **include the section header in every chunk from that section.** A chunk about "Reset Procedure" should say "Reset Procedure" at the top, so retrieval finds it and generation has context.

**HTML-aware.** Use `BeautifulSoup` to walk the DOM: split on `<h1>`, `<h2>`, `<section>`, `<table>` boundaries. Extract table content specially.

**Code-aware.** Split on top-level function/class definitions (`ast.parse` in Python; `tree-sitter` for polyglot). Keep imports at the top of each chunk. Never split a function.

**PDF-aware.** Extract text with layout hints (`pdfplumber`, `unstructured`, `pypdf` + hints). Chunk by pages or by heading detection. Extract tables separately (they're usually the hardest part).

**Preserving hierarchy metadata.** Even if a chunk is a paragraph, store its parent chapter and section names as metadata. Enables filtering and citation.

---

### 2.4 Semantic Chunking

Instead of fixed sizes, split where the **semantic topic changes**.

**Algorithm** (embedding-based, Greg Kamradt's approach):

1. Split text into sentences.
2. Embed each sentence.
3. Compute cosine similarity between consecutive sentences (or sentence groups).
4. Where similarity drops below a threshold (a "semantic break"), start a new chunk.

$$\text{break}_i = 1 \iff \cos(\mathbf{e}_i, \mathbf{e}_{i+1}) < \tau$$

Tune $\tau$ based on your corpus — typically the 5th–10th percentile of consecutive similarities.

**Pros:**
- Chunks are topically coherent.
- Sometimes markedly better retrieval quality on argumentative or narrative text.

**Cons:**
- Computationally expensive (many embedding calls).
- Chunk sizes vary widely; hard to enforce a budget.
- Threshold tuning is corpus-specific.

**When to use:** for high-value corpora where retrieval quality matters more than indexing cost.

---

### 2.5 Special-Purpose Chunking Patterns

**Parent-child chunking** — index small chunks (~200 tokens) for precise retrieval, but retrieve their **parent** larger chunks (~2000 tokens) for the LLM. Trades precision for context.

Concretely:
1. Split into small (child) chunks.
2. Also keep the large (parent) chunks they came from.
3. Embed and index the small chunks.
4. On retrieval, find matching small chunks, then return their parents.

Great for docs where a specific sentence matches the query but the full paragraph is needed for context.

**Multi-vector / summary indexing** — for each chunk, generate a **summary** or **hypothetical questions** it could answer. Embed those too. Index both.

```python
summary = llm(f"Summarize this text in one sentence: {chunk}")
questions = llm(f"List 5 questions this text could answer: {chunk}").split("\n")

# Store: (embedding_of_chunk, embedding_of_summary, embedding_of_each_question) → chunk
```

Improves retrieval on queries that don't lexically or semantically match the chunk's exact wording. Multiplies index size.

**Table extraction.** Tables are especially hard. Options:
- Render each row as text ("Column A = X, Column B = Y") and embed rows individually.
- Store the whole table as one chunk with a header describing what's in it.
- Use vision-language models to extract table content into structured JSON.

**Code chunking.** Special rules:
- Never split a function.
- Include imports at the top of every chunk.
- Include function/class signature as chunk header.
- Consider a docstring-based chunker for well-documented code.

**Chat/log chunking.** Group by conversation turn or a time window. Include user/participant metadata.

**Metadata to always store:**
- `source_id` / `document_id`.
- `filename` / `source_url`.
- `page_number` (for PDFs).
- `section_title` / `heading_path` (e.g., "Chapter 3 > Section 2.4").
- `chunk_index` (position in document).
- `timestamp` (created / updated).
- `author` / `owner`.
- `access_level` (for permission-aware retrieval).

These become filters at retrieval time and citations at generation time.

---

## 3. Mental Models & Analogies

### 3.1 The "Library Card Catalog" Model

Every chunk is a card in a library's catalog. A good card:

- Has enough context on its face to be useful — you shouldn't have to fetch the whole book to know what it's about.
- Points to its shelf location (metadata) — so you can retrieve the full context if needed.
- Fits on a card — not too big, not too small.
- Has a title (section header) that tells you what section of the book it summarizes.

Bad chunks:
- Mid-sentence cards ("...then decided to").
- Cards without shelf info (no metadata).
- Novels condensed onto one card (too long).
- Post-it note fragments (too short).

**The library metaphor makes clear**: chunking is about creating **findable, useful units**, not about mechanical text-splitting.

### 3.2 The "Question-Answer Fit" Model

Ask yourself for every chunk: **what questions could this chunk answer?**

If the answer is "few, if any" — the chunk is too small or content-poor. Enlarge or merge.
If the answer is "many, but poorly" — the chunk is too big or too topic-diverse. Split.
If the answer is "some, precisely" — the chunk is right-sized for retrieval.

This is why **parent-child chunking** works: small chunks are precise for matching the question, but parent chunks give the LLM enough context to answer.

Practical exercise before shipping: sample 20 random chunks and, for each, list the questions it answers. If most chunks answer nothing useful, chunking needs a redesign.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Chunk Size = Optimize This One Number"

Chunk size is one dimension. Others equally matter:
- Structure preservation (respect paragraphs, sections).
- Overlap.
- Metadata attached to each chunk.
- What content type (text, code, table) got which treatment.

Beware anyone claiming "500 tokens is the right chunk size" — depends on content and task.

### 4.2 "Bigger Chunks Are Better Because the LLM Sees More"

The LLM sees whatever you retrieve, in whatever chunk size. But the *retrieval* step ranks by embedding similarity, and **big chunks make embeddings semantically mushy** — averaging multiple topics into one vector. Retrieval precision drops. You end up retrieving the wrong big chunks and the LLM has to work harder.

Sweet spot for prose: 300–800 tokens per chunk with 10–20% overlap. Adjust based on evaluation.

### 4.3 "One Chunker to Rule Them All"

A codebase, a legal contract, a research paper, a customer chat log, a financial table — each has different structural cues. Use different chunkers per content type. Route by MIME type or content class.

---

## 5. Self-Assessment Bank (Chunking)

### Questions

**Q1 (Short answer).** In one paragraph, explain why chunking matters more than embedding-model choice for RAG quality.

**Q2 (Multiple choice).** Which chunking approach preserves paragraph and sentence boundaries when possible?
- (a) Fixed-size character split.
- (b) Recursive character splitter with paragraph/sentence separators.
- (c) Random splits.
- (d) Line-by-line split.

**Q3 (Short answer).** Why is overlap between chunks recommended, and what's a typical size?

**Q4 (Multiple choice).** Parent-child chunking indexes:
- (a) One big chunk per document.
- (b) Small chunks for retrieval, larger parent chunks fetched for the LLM.
- (c) Random subsets.
- (d) Only chapter titles.

**Q5 (Short answer).** For a markdown documentation site with H2 headers, what should each chunk carry to help retrieval and grounding?

**Q6 (Multiple choice).** Semantic chunking (embedding-based splits):
- (a) Splits at semantic topic changes; more coherent chunks but variable size.
- (b) Requires no compute.
- (c) Uses fixed character counts.
- (d) Is always better than recursive chunking.

**Q7 (Short answer).** Describe three metadata fields you'd attach to every chunk and how they help downstream.

**Q8 (Multiple choice).** For a codebase, splitting a chunk in the middle of a function is:
- (a) Fine, the LLM can handle it.
- (b) Bad — you lose syntactic and semantic context; code chunking should never split within a function.
- (c) Required for size limits.
- (d) Better than markdown chunking.

**Q9 (Short answer).** Give one alternative to embedding the raw chunk text (multi-vector or hypothetical-question indexing).

**Q10 (Multiple choice).** A chunk that says "he then signed the bill on Tuesday" (with no antecedent) will:
- (a) Retrieve well because "he" is unambiguous.
- (b) Retrieve poorly because the referent isn't in the chunk — chunking failed to preserve the necessary context.
- (c) Improve LLM grounding.
- (d) Be fine after embedding.

---

### Answer Key & Detailed Explanations

**A1.** Chunking determines what the retriever *can* return. If a chunk doesn't contain the answer, no embedding model or reranker can fix that — the information isn't in the retrieval space. If a chunk is too big and averages multiple topics, embeddings become semantically mushy and the retriever can't distinguish "on topic" from "adjacent." Chunking sets the ceiling on retrieval quality; embedding choice affects the margin within that ceiling.

**A2. (b).** Recursive splitters try higher-order separators first (paragraphs, then sentences, then words), preserving natural boundaries when possible. Fixed-size and random cut wherever the character index lands.

**A3.** Overlap ensures that content near a chunk boundary appears in at least two chunks, so retrieval can still find it if the split happened at an inopportune point (mid-thought, mid-sentence). Typical: 10–20% of chunk size (e.g., 50–100 tokens for a 500-token chunk).

**A4. (b).** Small chunks are precise for query matching (embeddings capture specific ideas); their parent chunks give the LLM sufficient context to answer. Best of both.

**A5.** Each chunk should carry: (1) the section header text at the top of the chunk (so both retrieval and generation see the topic); (2) metadata including `section_title`, `heading_path` (breadcrumb of nested headers), `document_id`, and `source_url` for citation; (3) `chunk_index` for reconstructing document order if needed.

**A6. (a).** Semantic chunking splits where consecutive-sentence similarity drops below a threshold. Chunks are topically coherent but variable in size, and the process needs embedding compute. Whether it beats recursive chunking is corpus-dependent.

**A7.** (1) **`source_id` / `filename`** — for citations and per-source filtering. (2) **`section_title`** or **heading path** — for hierarchical context and topic-based filtering. (3) **`timestamp`** — for time-filtered queries or freshness ranking. (4) **`access_level`** — for permission-scoped retrieval. (5) **`page_number`** — for accurate citation. Any three plus their use.

**A8. (b).** Code has strict semantic and syntactic units (functions, classes). Splitting mid-function loses the function's signature, imports, and scope — the fragment is often uninterpretable. Code chunkers should split at function/class boundaries, keeping each function whole even if it exceeds the size limit.

**A9.** **Multi-vector indexing** — for each chunk, embed also a **summary** or **hypothetical questions** the chunk could answer, and index all vectors pointing to the same chunk. Retrieval on a real user question matches the question-embedding well even if the question doesn't lexically resemble the chunk's text.

**A10. (b).** "He" refers to someone named earlier in the document. Without the referent, the chunk is context-less; retrieval that finds this chunk hands the LLM insufficient information to produce a grounded answer. Fixes: include preceding sentence(s) that name the subject; add document-level summary as metadata; use parent-child chunking.

---

## 6. Practice Prompts

1. **Chunker comparison.** Take a 100-page PDF (or a text file). Chunk it three ways: fixed-size 500 tokens, recursive 500 tokens, and markdown-aware. Compare 20 random chunks — which look coherent?
2. **Overlap ablation.** Chunk the same corpus with 0%, 10%, and 20% overlap. Build a small retrieval index. Test 10 questions; note which questions benefit from overlap.
3. **Semantic chunking.** Implement embedding-based semantic chunking (sentence embeddings, threshold-based break). Compare to recursive chunking on the same corpus.
4. **Parent-child.** Set up parent-child chunking. Test whether it improves answer quality for questions requiring context.
5. **Structure-aware for code.** Write a Python chunker using `ast` that splits a Python file at function/class boundaries. Preserve imports and docstrings.

---

## 7. References

- Greg Kamradt, ["5 Levels of Text Splitting"](https://www.youtube.com/watch?v=8OJC21T2SL4) — practical taxonomy.
- LangChain docs: [Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/).
- LlamaIndex docs: [Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/).
- Unstructured: [github.com/Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) — production-grade document parsing.
- Anthropic, ["Contextual Retrieval"](https://www.anthropic.com/news/contextual-retrieval) (2024) — chunk-context augmentation.
