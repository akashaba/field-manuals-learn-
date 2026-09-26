# 04.05 — GraphRAG

> **Module 4: RAG** · Subtopic 5 of 5
> **Prerequisites:** 04.01–04.04 (chunking, hybrid retrieval, reranking, query planning), 02.03 (structured extraction), graph basics (adjacency, random walks, communities), 04.04 §4 (multi-hop questions).
> **Outcome:** you know **when** graph structure improves RAG and when a strong hybrid baseline is enough. You can build knowledge graphs cheaply and correctly (deterministic edges first, LLM extraction second, entity resolution always), implement the main GraphRAG retrieval patterns (community summaries for global questions, neighbourhood and PPR retrieval for local and multi-hop ones, Text2Cypher for structured ones), and account for their indexing cost.

---

## 1. What Graphs Add, and When They Don't

Chunk-based RAG retrieves **passages similar to the question**. It struggles with:

| Question type | Example | Why chunk RAG struggles | Graph approach |
|---|---|---|---|
| **Global / sensemaking** | "What are the main themes of the 2025 health-care bills?" | No single chunk contains the answer; top-k sees a sliver | Community summaries + map-reduce (Microsoft GraphRAG) |
| **Multi-hop relational** | "Which committees heard bills sponsored by the author of HB 45?" | Each hop needs a different retrieval; chains break | Graph traversal / PPR / Text2Cypher |
| **Aggregation** | "How many bills amended Title 2 this session?" | Counting over retrieved text is unreliable | Structured query over the graph |
| **Connection discovery** | "How is section 2-18-303 related to the state employee benefits act?" | The link runs through intermediate entities | Path finding, neighbourhood expansion |
| **Entity-centric** | "Everything about Senator X's water bills" | Mentions are scattered across documents | Entity node → linked text units |

**When a graph is *not* worth it:**
- Mostly local, factoid questions that hybrid retrieval + reranking already answers (04.02–04.03).
- Rapidly changing corpora where re-extraction cost dominates.
- No reliable way to evaluate the gain.

Systematic comparisons find that GraphRAG helps on multi-hop and global questions but can underperform strong vanilla RAG on single-hop factoid questions (Han et al., 2025). **Always compare against your best hybrid baseline, not a naive one.**

---

## 2. Graph Construction

### 2.1 Deterministic edges first

Many domains contain **explicit structure** that regexes and parsers extract perfectly, at no LLM cost. Legislation is an ideal example:

```
 (:Legislator)──SPONSORED──►(:Bill {id:"HB-45", session:2025, status})──AMENDS──►(:Section {id:"2-18-303"})
                                  │                                                   │
                            REFERRED_TO                                    REFERENCES / PART_OF
                                  ▼                                                   ▼
                            (:Committee)                               (:Section)   (:Chapter)──PART_OF──►(:Title)
 (:Chunk {text, embedding})──MENTIONS──►(:Entity)          (:Chunk)──FROM──►(:Bill | :Section)
```

Citations ("as defined in 2-18-101"), amendment clauses ("Section 2-18-303, MCA, is amended to read:"), sponsor and committee records, and the hierarchy (Title › Chapter › Part › Section) come from the source systems or from regex parsing, with near-perfect precision.

```python
import re

SECTION = r"\d{1,2}-\d{1,3}-\d{1,4}"
AMENDS = re.compile(rf"Section\s+({SECTION}),\s+MCA,\s+is\s+amended", re.I)
REFERENCES = re.compile(rf"(?:as\s+(?:defined|provided)\s+in|pursuant\s+to|under)\s+(?:section\s+)?({SECTION})", re.I)


def extract_citation_edges(doc_id: str, text: str) -> list[tuple[str, str, str]]:
    """Deterministic (source, relation, target) triples from statutory text."""
    edges = [(doc_id, "AMENDS", m.group(1)) for m in AMENDS.finditer(text)]
    edges += [(doc_id, "REFERENCES", m.group(1)) for m in REFERENCES.finditer(text)]
    return sorted(set(edges))
```

### 2.2 LLM extraction for semantic relations

Use an LLM for what rules cannot capture: entities in free text (agencies, programmes, concepts) and typed relations ("establishes", "funds", "exempts"). Do it with an **explicit ontology** and structured outputs (02.03):

```python
from typing import Literal

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(description="Canonical surface form as written")
    type: Literal["agency", "program", "fund", "person", "organization", "concept", "place"]
    description: str = Field(description="One sentence grounded in the text")


class Relation(BaseModel):
    evidence: str = Field(description="Verbatim quote supporting the relation")
    source: str
    relation: Literal["ESTABLISHES", "FUNDS", "EXEMPTS", "REQUIRES", "ADMINISTERS", "REPEALS", "RELATED_TO"]
    target: str


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
```

**Rules:**
- **Evidence-first** relations (02.03 §3.1), with verbatim quotes verified against the chunk.
- A closed set of relation types (an open-ended "RELATED_TO" is a last resort).
- **Gleaning:** optional follow-up passes asking "did you miss any entities?" raise recall at extra cost.
- Store `chunk_id` provenance on every node and edge.

### 2.3 Entity resolution (the step most GraphRAG demos skip)

"Dept. of Public Health and Human Services", "DPHHS", and "the department" must become **one node**. Otherwise the graph fragments, and communities and paths become meaningless.

1. **Normalise:** case, punctuation, abbreviations dictionary, legal suffixes.
2. **Block:** group candidates cheaply (same type + a shared token or a similar embedding) to avoid $O(n^2)$ comparisons.
3. **Score pairs:** string similarity + embedding similarity + type agreement + context overlap.
4. **Merge** above a high threshold (union–find). Send the middle band to an **LLM adjudicator** or human review.
5. **Keep aliases** on the canonical node, for search and display.

```python
import re
from collections import defaultdict
from difflib import SequenceMatcher

ABBREV = {"dept": "department", "dept.": "department", "&": "and", "svcs": "services"}


def normalize(name: str) -> str:
    toks = [ABBREV.get(t, t) for t in re.sub(r"[^\w&.\s-]", " ", name.lower()).split()]
    return " ".join(t.strip(".") for t in toks if t not in {"the", "of", "montana", "state"})


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def resolve_entities(entities: list[tuple[str, str]], aliases: dict[str, str] | None = None,
                     threshold: float = 0.88) -> dict[str, str]:
    """entities: [(name, type)]. aliases: known acronym -> full name. Returns name -> canonical name."""
    aliases = {k.lower(): v for k, v in (aliases or {}).items()}
    uf, norm = UnionFind(), {}
    for name, _ in entities:
        norm[name] = normalize(aliases.get(name.lower(), name))
    blocks = defaultdict(list)                                  # block by (type, any content token)
    for name, typ in entities:
        for tok in set(norm[name].split()):
            if len(tok) > 3:
                blocks[(typ, tok)].append(name)
    for members in blocks.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if norm[a] == norm[b] or SequenceMatcher(None, norm[a], norm[b]).ratio() >= threshold:
                    uf.union(a, b)
    groups = defaultdict(list)
    for name, _ in entities:
        groups[uf.find(name)].append(name)
    canon = lambda g: max(g, key=lambda n: (not n.lower().startswith("the "), len(n)))  # longest, no leading article
    return {n: canon(g) for g in groups.values() for n in g}
```

---

## 3. Microsoft GraphRAG: Communities for Global Questions

### 3.1 Indexing pipeline

```
 chunks ─► LLM entity/relation extraction (+ gleaning) ─► entity resolution ─► graph
       ─► hierarchical community detection (Leiden) ─► LLM community reports (per level)
       ─► embeddings for entities / reports / chunks
```

**Community detection** partitions the graph to maximise **modularity**:

$$
Q = \frac{1}{2m}\sum_{i,j}\left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i, c_j)
$$

Here $A$ is the adjacency matrix, $k_i$ the node degrees, $m$ the number of edges, and $\delta$ is 1 when two nodes are in the same community. **Leiden** (Traag et al., 2019) improves on Louvain by guaranteeing well-connected communities. Applied recursively, it yields a **hierarchy** (level 0 = broad themes, deeper levels = specific clusters).

### 3.2 Query modes

- **Global search (map-reduce):** for a chosen hierarchy level, each community report produces a partial answer with a helpfulness score (map). The best partial answers are combined into the final answer (reduce). It answers "what are the themes" questions over the whole corpus, but costs many LLM calls per query.
- **Local search:** find the entities matching the query (embedding + name match), expand to their neighbourhood, and assemble the linked text units, relations, and community reports under a token budget.
- **DRIFT search:** starts from community-level context, generates follow-up local queries, and refines. It is a hybrid of global and local.

```python
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity


def detect_communities(G: nx.Graph, resolution: float = 1.0, seed: int = 0):
    comms = louvain_communities(G, resolution=resolution, seed=seed)   # Leiden: use graspologic / leidenalg
    return comms, modularity(G, comms, resolution=resolution)


def global_search(question: str, reports: list[str], map_fn, reduce_fn, top_n: int = 8) -> str:
    """map_fn(question, report) -> (partial_answer, helpfulness 0-100); reduce_fn(question, partials) -> answer."""
    partials = [(score, ans) for ans, score in (map_fn(question, r) for r in reports) if score > 0]
    partials.sort(reverse=True)
    return reduce_fn(question, [a for _, a in partials[:top_n]])
```

### 3.3 Cost reality

$$
C_{\text{index}} \approx N_{\text{chunks}} \cdot (1 + g) \cdot (T_{\text{in}} + T_{\text{out}}) \;+\; N_{\text{comm}} \cdot T_{\text{report}}
$$

where $g$ is the number of gleaning passes. For large corpora this is **orders of magnitude** above embedding-only indexing, and updates can force re-extraction and re-summarisation.

**LazyGraphRAG** (Microsoft Research, Nov 2024) defers LLM work to query time. Indexing uses cheap noun-phrase extraction plus communities, at about the cost of vector RAG. At query time it runs a budgeted, iterative-deepening relevance search. Microsoft reported quality comparable to GraphRAG global search on global questions at a small fraction of the query cost. The lesson generalises: **pay for LLM structure only where queries need it.**

---

## 4. Graph Retrieval for Local and Multi-Hop Questions

### 4.1 Personalised PageRank (HippoRAG)

HippoRAG (Gutiérrez et al., 2024) builds an entity graph from the chunks, links query entities to graph nodes, and runs **Personalised PageRank** from those seed nodes. Chunks are then scored by the PPR mass of the entities they contain. PPR spreads relevance along relations, so passages that connect *through* intermediate entities surface in one step, without iterative LLM calls:

$$
\boldsymbol{\pi} = (1 - \alpha)\,\mathbf{e}_S + \alpha\, \mathbf{P}^\top \boldsymbol{\pi}
$$

where $\mathbf{e}_S$ is the seed distribution (uniform over the query entities, or weighted by match score), $\mathbf{P}$ is the row-normalised transition matrix, and $\alpha$ (≈ 0.5–0.85) is the damping factor. The restart probability is $1 - \alpha$.

```python
import numpy as np


def personalized_pagerank(G: nx.Graph, seeds: dict, alpha: float = 0.85, iters: int = 100,
                          tol: float = 1e-10) -> dict:
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
    e = np.zeros(len(nodes))
    for n, w in seeds.items():
        e[idx[n]] = w
    e /= e.sum()
    pi = e.copy()
    for _ in range(iters):
        spread = A.T @ (pi * inv)                              # P^T pi with P = D^-1 A
        dangling = pi[deg == 0].sum()
        new = (1 - alpha) * e + alpha * (spread + dangling * e)
        if np.abs(new - pi).sum() < tol:
            pi = new
            break
        pi = new
    return dict(zip(nodes, pi))


def ppr_chunk_ranking(G, seeds, chunk_entities: dict[str, set], alpha=0.5, top_k=10):
    """Score chunks by the summed PPR mass of the entities they mention."""
    pr = personalized_pagerank(G, seeds, alpha=alpha)
    scores = {c: sum(pr.get(e, 0.0) for e in ents) for c, ents in chunk_entities.items()}
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
```

### 4.2 Neighbourhood expansion under a token budget

For entity-centric questions, take the k-hop neighbourhood of the matched entities, rank the nodes (PPR or edge weights), and assemble the context in this order: entity descriptions → relations with evidence quotes → linked chunks. Stop when the token budget is reached, and always include **provenance** so the answer can cite the source chunks, not the graph.

```python
def local_context(G: nx.Graph, seeds: list[str], chunk_text: dict[str, str], node_chunks: dict[str, list[str]],
                  budget_tokens: int, count, hops: int = 2) -> list[str]:
    sub_nodes = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        frontier = {n for f in frontier for n in G.neighbors(f)} - sub_nodes
        sub_nodes |= frontier
    pr = personalized_pagerank(G.subgraph(sub_nodes).copy(), {s: 1.0 for s in seeds if s in sub_nodes})
    out, used, seen = [], 0, set()
    for node, _ in sorted(pr.items(), key=lambda kv: -kv[1]):
        for cid in node_chunks.get(node, []):
            if cid in seen:
                continue
            t = chunk_text[cid]
            if used + count(t) > budget_tokens:
                return out
            out.append(f"[{cid}] {t}")
            used += count(t)
            seen.add(cid)
    return out
```

### 4.3 Text2Cypher for structured questions

Aggregations and exact relational questions ("how many bills sponsored by X were referred to the Judiciary committee in 2025?") should be answered by **querying the graph**, not by reading text:

```cypher
MATCH (l:Legislator {name: $sponsor})-[:SPONSORED]->(b:Bill {session: 2025})-[:REFERRED_TO]->(c:Committee {name: "Judiciary"})
RETURN count(DISTINCT b) AS bills
```

Generate Cypher with an LLM that is given the **schema** (labels, relationship types, properties) and examples, then **guard** the query before execution: read-only, allowed labels, a bounded result size, and a timeout. Parameterise the literal values, and validate them against the database.

```python
import re

FORBIDDEN = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+dbms|CALL\s+apoc)\b", re.I)
LABEL = re.compile(r":\s*`?([A-Za-z_][A-Za-z0-9_]*)")


def guard_cypher(q: str, allowed_labels: set[str], max_limit: int = 200) -> str:
    if FORBIDDEN.search(q):
        raise ValueError("write or admin operations are not allowed")
    unknown = {l for l in LABEL.findall(q) if l not in allowed_labels}
    if unknown:
        raise ValueError(f"unknown labels/relationship types: {sorted(unknown)}")
    if not re.search(r"\bLIMIT\s+\d+\b", q, re.I) and "count(" not in q.lower():
        q = q.rstrip().rstrip(";") + f" LIMIT {max_limit}"
    return q
```

Also run generated queries under a **read-only database role**. The guard is defence in depth, not the only control.

---

## 5. Choosing an Approach

| Need | Approach | Indexing cost | Query cost |
|---|---|---|---|
| Local factoid questions | Hybrid RAG + reranking (04.02–04.03) | Low | Low |
| Multi-hop over entities | HippoRAG-style PPR, or iterative retrieval (04.04) | Medium (entity extraction) | Low (PPR) / Medium (iterative) |
| Global / thematic over the whole corpus | GraphRAG global search, LazyGraphRAG, or RAPTOR summaries (04.01 §3.6) | High / Low / Medium | High / Budgeted / Low |
| Exact relational and aggregate questions | Knowledge graph + Text2Cypher (or SQL) | Deterministic ETL | Low |
| Mixed workloads | Router (04.04 §6) over all of the above | Sum of the parts | Per route |

**For structured domains (legislation, contracts, product catalogues), the highest-ROI graph is usually the deterministic one** — citations, hierarchy, sponsors, amendments — with LLM extraction added only for the semantic relations that questions actually need.

---

## 6. Evaluating GraphRAG

- **Local questions:** the standard RAG metrics (evidence recall @ budget, answer correctness). The baseline must be your best hybrid + reranker pipeline.
- **Multi-hop:** supporting-evidence recall per hop, and answer EM/F1 (04.04 Project 2).
- **Global questions:** there is no single gold answer. Use **pairwise LLM-judge comparisons** on comprehensiveness, diversity, empowerment, and directness (the GraphRAG paper's protocol). Control for judge bias: randomise the answer order, use multiple judges, blind the system names, and validate against human preferences on a sample. Longer answers win judges unfairly, so control for length.
- **Structured questions:** execution accuracy against gold Cypher/SQL results.
- **Cost accounting:** indexing tokens and dollars, query tokens, latency, and **update cost** (what a single changed document triggers).

---

## 7. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Fragmented graph** | Duplicate entities; tiny communities | Entity resolution with blocking + adjudication; alias tables |
| **Hallucinated relations** | Edges with no textual support | Evidence-first extraction; verify quotes; drop unsupported edges |
| **Indexing cost explosion** | Budget blown before launch | Deterministic edges first; LLM extraction only for needed relation types; LazyGraphRAG-style deferral; smaller extraction models |
| **Updates are expensive** | One edited document triggers large recomputation | Incremental extraction per changed chunk; periodic (not per-change) community recomputation; versioned graph |
| **Global answers are vague** | Generic themes, no citations | Ground community reports in claims with chunk IDs; require citations in the reduce step |
| **Text2Cypher errors or risks** | Wrong counts; attempted writes | Schema-grounded prompts, examples, guards, a read-only role, execution-accuracy evals |
| **No gain over baseline** | Extra complexity without accuracy | Route only graph-suited questions to graph paths; measure per query type |

---

## 8. Hands-On Projects

### Project 1 — Legislative Knowledge Graph with Graph-Augmented RAG and Text2Cypher

**User stories**
- *As a legislative researcher*, I want to ask relational questions ("which sections amended by 2025 bills from the Health committee reference 2-18-101?") and get exact, cited answers.
- *As the platform owner*, I want the graph kept in sync with the bill-drafting system automatically.

**Acceptance criteria**
1. Neo4j (or Postgres with a graph extension) populated with deterministic nodes and edges (bills, sections, legislators, committees, AMENDS, REFERENCES, PART_OF, SPONSORED, REFERRED_TO) from source data plus `extract_citation_edges`. Edge extraction precision ≥ 99% on a 200-edge audit.
2. LLM extraction adds entities and relations (agencies, programmes, funds) with evidence quotes, plus entity resolution (`resolve_entities` + LLM adjudication). The duplicate rate after resolution is < 2% on an audit.
3. Query router: text RAG (04.02) for passage questions; Text2Cypher with `guard_cypher` and a read-only role for relational and aggregate questions; graph-augmented RAG (`local_context`) for entity-centric questions.
4. On ≥ 200 questions (≥ 60 relational/aggregate): execution accuracy for the Cypher route, answer accuracy for the others, compared with hybrid RAG alone.
5. CDC-driven incremental updates (03.03 §5): a changed bill updates its edges within minutes, without a full rebuild.

**Step-by-step**
1. Design the graph schema; load the bills, sections, and legislators from the source systems.
2. Run `extract_citation_edges` over all statutory text; audit a sample.
3. Run LLM extraction (batch API) on chunks mentioning agencies and programmes; verify the evidence quotes; resolve the entities.
4. Build Text2Cypher prompts with the schema and 15–20 examples; add the guard and the read-only role; evaluate execution accuracy.
5. Implement the router and graph-augmented context; run the evaluation against the hybrid baseline.
6. Add the incremental update consumer on the Kafka topic.

---

### Project 2 — GraphRAG vs Strong Hybrid RAG: Global and Local Questions, with Cost Accounting

**User stories**
- *As an architect*, I need evidence on whether Microsoft-style GraphRAG (or LazyGraphRAG) justifies its cost for our corpus, compared with our best hybrid pipeline and RAPTOR summaries.

**Acceptance criteria**
1. The corpus has ≥ 1M tokens. The question set has ≥ 50 global/thematic questions and ≥ 150 local questions.
2. Systems: hybrid + reranker (the baseline); RAPTOR (04.01 Project 3); the GraphRAG library (global, local, DRIFT); and a LazyGraphRAG-style or budgeted variant if available.
3. Global questions: pairwise LLM-judge comparisons (randomised order, 2+ judges, length-controlled) with a human-preference validation subset. Local questions: accuracy and evidence recall.
4. Full cost accounting: indexing tokens and dollars, per-query tokens, latency, and the update cost for 1% corpus churn.
5. A recommendation on which question types justify which approach, with a router design.

**Step-by-step**
1. Prepare the corpus and questions; label the local questions with gold evidence.
2. Build each index; log all token usage.
3. Run all systems on all questions; cache the outputs.
4. Run the pairwise judging protocol with bias controls, and validate on 30 human comparisons.
5. Compute the metrics and costs; write the decision memo.

---

### Project 3 — HippoRAG-Style PPR Retrieval for Multi-Hop QA

**User stories**
- *As a retrieval engineer*, I want single-step multi-hop retrieval that is cheaper than iterative LLM loops, so that complex questions stay within our latency budget.

**Acceptance criteria**
1. Builds an entity graph from chunks (open information extraction or a typed schema), with synonym edges between similar entities (embedding similarity ≥ τ) and chunk ↔ entity links.
2. Retrieval: link query entities to nodes → `personalized_pagerank` → `ppr_chunk_ranking`. Tunes the damping and seed weighting.
3. On a multi-hop benchmark sample (MuSiQue / 2WikiMultiHopQA, ≥ 300 questions) plus domain multi-hop questions: Recall@2/@5 of the supporting passages and answer EM/F1, against dense retrieval, hybrid, and IRCoT (04.04).
4. Latency and LLM-call comparison: PPR retrieval needs ≤ 1 LLM call for query entity extraction versus several for iterative methods.
5. An ablation: without synonym edges, without entity resolution, and with different damping values.

**Step-by-step**
1. Extract the entities and triples per chunk (structured outputs) and build the graph in networkx (or igraph for speed).
2. Add synonym edges via entity-embedding kNN, and entity resolution (§2.3).
3. Implement query-entity linking (exact + embedding match) and PPR ranking.
4. Evaluate against the baselines; run the ablations.
5. Write up where PPR wins and where it fails (e.g. questions whose bridge entity isn't mentioned in the query).

---

## 9. Foundational Papers & Reading (exact titles)

**GraphRAG and graph retrieval**
- Edge et al., 2024 — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*
- Gutiérrez et al., 2024 — *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*
- Gutiérrez et al., 2025 — *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models* (HippoRAG 2)
- Guo et al., 2024 — *LightRAG: Simple and Fast Retrieval-Augmented Generation*
- He et al., 2024 — *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering*
- Sun et al., 2023 — *Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph*
- Peng et al., 2024 — *Graph Retrieval-Augmented Generation: A Survey*
- Pan et al., 2023 — *Unifying Large Language Models and Knowledge Graphs: A Roadmap*
- Han et al., 2025 — *RAG vs. GraphRAG: A Systematic Evaluation and Key Insights*
- Ozsoy et al., 2024 — *Text2Cypher: Bridging Natural Language and Graph Databases*
- Microsoft Research blog, Oct 2024 — *Introducing DRIFT Search: Combining global and local search methods to improve quality and efficiency*
- Microsoft Research blog, Nov 2024 — *LazyGraphRAG: Setting a new standard for quality and cost*

**Graph algorithms**
- Newman, 2006 — *Modularity and community structure in networks*
- Blondel et al., 2008 — *Fast unfolding of communities in large networks* (Louvain)
- Traag, Waltman & van Eck, 2019 — *From Louvain to Leiden: guaranteeing well-connected communities*
- Page et al., 1999 — *The PageRank Citation Ranking: Bringing Order to the Web*
- Haveliwala, 2002 — *Topic-Sensitive PageRank*

**Entity resolution**
- Papadakis et al., 2020 — *Blocking and Filtering Techniques for Entity Resolution: A Survey*
- Christen, 2012 — *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection* (book)

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **Microsoft GraphRAG** (library) | Reference implementation: extraction, communities, global/local/DRIFT search |
| **LightRAG, HippoRAG** repos | Alternative graph-RAG designs to study |
| **Neo4j** (+ vector indexes, GDS library) / **Memgraph** / **Apache AGE (Postgres)** | Graph storage, Cypher, graph algorithms |
| **networkx / igraph / graspologic / leidenalg** | Community detection, PageRank, prototyping |
| **Splink / dedupe / rapidfuzz** | Entity resolution |
| **Provider structured outputs / Instructor** | Evidence-first entity and relation extraction |
| **LangChain / LlamaIndex graph modules** | Text2Cypher and graph-RAG integrations (use with guards) |
