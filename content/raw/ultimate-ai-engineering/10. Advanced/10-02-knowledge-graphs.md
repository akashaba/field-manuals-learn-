# 10.02 — Knowledge Graphs (Modelling, Building, and Querying Explicit Knowledge)

> **Module 10: Advanced** · Subtopic 2 of 4
> **Prerequisites:**
> - 04.05 (GraphRAG: community summaries, graph-augmented retrieval);
> - 02.03 (structured outputs);
> - 03.03 (pgvector, Postgres operations);
> - 07.02 §4 (the SQL guard pattern);
> - 10.01 §3 (canonical document representation).
>
> **Outcome:** you can model a domain as a graph (property graph or RDF) with an enforceable schema, and build it from structured sources, rules and LLM extraction, with provenance on every edge. You can resolve entities, represent law that changes over time bitemporally, query safely (including LLM-generated Cypher), and use the graph for impact analysis and grounded answers.

> **Relationship to 04.05.** GraphRAG builds an *approximate* graph from text in order to improve retrieval. This subtopic is about a **curated, schema-governed knowledge graph** that acts as a **system of record** for relationships: bills ↔ sections ↔ sponsors ↔ committees ↔ votes ↔ versions. Here correctness, time, and provenance matter more than recall. The two combine in §8.

---

## 1. When a Knowledge Graph Earns Its Keep

Use a KG when **relationships are the product**. Legislative work is a strong fit:
- *Which sections cite 7-6-4005, transitively?* This is impact analysis for a drafter.
- *Which bills this session amend Title 15, chapter 30, and which committees heard them?*
- *What did 7-6-4005 say on 1 January 2024, and which bill changed it?*
- *Which of Representative X's bills were amended in the Senate?*

Each of these is a multi-hop join over typed relationships with temporal qualifiers. Vector search can't answer them reliably, and relational SQL can answer them but becomes painful with variable-length paths.

**Don't build a KG when** the questions are "find text about X" (use RAG) or the relationships are few and fixed (use relational tables). A KG has real costs: an ontology, entity resolution, and upkeep.

---

## 2. Data Models and Standards

| | **Labelled Property Graph (LPG)** | **RDF** |
|---|---|---|
| Unit | Nodes and relationships, each with labels/types and key-value properties | Triples (subject, predicate, object) with global IRIs |
| Schema | Constraints and indexes (Neo4j); application-level validation | **RDFS/OWL** (semantics, inference), **SHACL** (validation) |
| Query | **Cypher / openCypher**, **GQL (ISO/IEC 39075:2024)**, **SQL/PGQ** (ISO/IEC 9075-16:2023) | **SPARQL 1.1** |
| Edge properties | Native (`valid_from`, `source`, `confidence`) | RDF-star / RDF 1.2 triple terms, or reification |
| Strength | Developer ergonomics, path queries, analytics | Interoperability, linked open data, formal semantics (e.g. ELI, Akoma Ntoso-aligned legal vocabularies) |
| Stores | Neo4j, Memgraph, **Apache AGE (Postgres)**, Amazon Neptune, Spanner Graph | GraphDB, Stardog, Apache Jena Fuseki, Amazon Neptune |

**A pragmatic choice for a Postgres + pgvector shop.** **Apache AGE** adds openCypher to PostgreSQL, and **Azure Database for PostgreSQL flexible server supports it** (enable it in `azure.extensions` and `shared_preload_libraries`). You get:
- graph, vectors (pgvector), and relational data **in one transactional database**;
- one backup and operations story;
- joins between Cypher results and SQL tables.

Move to a dedicated graph database (Neo4j) when graph analytics (GDS), very deep traversals at scale, or graph-native tooling dominate. Use RDF when you must publish linked data or align with legal-information standards.

```sql
-- Apache AGE on Postgres: Cypher inside SQL, joinable with relational + pgvector tables
LOAD 'age';                       -- (skip on Azure when AGE is in shared_preload_libraries)
SET search_path = ag_catalog, "$user", public;
SELECT * FROM create_graph('mca');

SELECT * FROM cypher('mca', $$
  MATCH (s:Section {mca: '7-6-4005'})<-[:CITES*1..3]-(x:Section)
  RETURN DISTINCT x.mca
$$) AS (mca agtype);
```

---

## 3. Ontology and Schema: Make the Graph Enforceable

Start from **competency questions** (the §1 list), derive the entity and relationship types, and **enforce them at write time**. An unenforced graph decays into an unqueryable mess within a session.

```
 (Legislator)<-[:SPONSORED_BY]-(Bill)-[:AMENDS|REPEALS|CREATES {bill_section, effective}]->(Section)-[:PART_OF]->(Part)->(Chapter)->(Title)
                                 │  └─[:REFERRED_TO {date}]->(Committee)<-[:MEMBER_OF {session, role}]-(Legislator)
                                 └─[:HAS_VERSION]->(BillVersion {stage: introduced|amended|enrolled})-[:VOTE {date, yeas, nays}]->(Chamber)
 (Section)-[:CITES {valid_from, valid_to, recorded_at, source, span, extractor, confidence}]->(Section)
```

```python
import re

SCHEMA = {
    "nodes": {
        "Bill":       {"required": {"id": r"^(HB|SB|HJ|SJ|HR|SR) \d{1,4}$", "session": r"^\d{4}$", "title": r".+"}},
        "Section":    {"required": {"mca": r"^\d{1,2}-\d{1,2}-\d{3,4}$"}},
        "Legislator": {"required": {"name": r".+", "chamber": r"^(House|Senate)$"}},
        "Committee":  {"required": {"code": r"^[A-Z]{2,6}$"}},
    },
    "edges": {  # type: (source label, target label, max outgoing per source or None)
        "AMENDS":       ("Bill", "Section", None),
        "REPEALS":      ("Bill", "Section", None),
        "CITES":        ("Section", "Section", None),
        "SPONSORED_BY": ("Bill", "Legislator", 1),            # exactly one primary sponsor
        "REFERRED_TO":  ("Bill", "Committee", None),
    },
}

def validate(nodes, edges, schema=SCHEMA):
    """SHACL-style validation. nodes: {id: (label, props)}; edges: [(src, type, dst)]. Returns violations."""
    errs = []
    for nid, (label, props) in nodes.items():
        spec = schema["nodes"].get(label)
        if spec is None:
            errs.append(f"{nid}: unknown label {label}"); continue
        for k, pat in spec["required"].items():
            if k not in props:
                errs.append(f"{nid}: missing {label}.{k}")
            elif not re.fullmatch(pat, str(props[k])):
                errs.append(f"{nid}: {label}.{k}={props[k]!r} fails {pat}")
    out_count = {}
    for s, t, d in edges:
        if t not in schema["edges"]:
            errs.append(f"({s})-[{t}]->({d}): unknown relationship type"); continue
        src_l, dst_l, card = schema["edges"][t]
        if s not in nodes or d not in nodes:
            errs.append(f"({s})-[{t}]->({d}): dangling endpoint"); continue
        if nodes[s][0] != src_l or nodes[d][0] != dst_l:
            errs.append(f"({s})-[{t}]->({d}): {nodes[s][0]}->{nodes[d][0]} not allowed (expects {src_l}->{dst_l})")
        out_count[(s, t)] = out_count.get((s, t), 0) + 1
        if card is not None and out_count[(s, t)] > card:
            errs.append(f"{s}: more than {card} {t} edge(s)")
    return errs

nodes = {
    "hb12": ("Bill", {"id": "HB 12", "session": "2025", "title": "Revise county budget laws"}),
    "s1":   ("Section", {"mca": "7-6-4005"}),
    "s2":   ("Section", {"mca": "7-6-40005"}),                                 # malformed MCA number
    "l1":   ("Legislator", {"name": "J. Smith", "chamber": "House"}),
    "l2":   ("Legislator", {"name": "A. Jones", "chamber": "Senate"}),
    "c1":   ("Committee", {"code": "LOCG"}),
}
edges = [("hb12", "AMENDS", "s1"), ("hb12", "SPONSORED_BY", "l1"), ("hb12", "SPONSORED_BY", "l2"),
         ("s1", "AMENDS", "s2"), ("hb12", "REFERRED_TO", "c9")]
errs = validate(nodes, edges)
for e in errs:
    print(e)
assert len(errs) == 4
```

In production:
- Enforce uniqueness and existence constraints in the database: Neo4j `CREATE CONSTRAINT … IS UNIQUE` / `IS NOT NULL`, or unique indexes on AGE label tables.
- Run this validator in the ingestion pipeline, and **reject** or **quarantine** invalid writes rather than silently fixing them.
- Version the schema like code (08.05 §2 bundle).

---

## 4. Building the Graph: Structured First, Rules Second, LLMs Last

```
 SYSTEM-OF-RECORD DATA (highest trust) ── bill-tracking DB: bills, versions, sponsors, referrals, votes
      │   (Kafka CDC → upsert, idempotent by natural keys; 10.03 §3)
 DETERMINISTIC EXTRACTION ─────────────── regexes/parsers over canonical docs (10.01 §3): MCA cross-references,
      │                                     "amends section …" clauses, effective-date sections
 LLM EXTRACTION (lowest trust) ────────── relations that need language understanding: "notwithstanding …",
                                            delegations of authority, defined-term usage; schema-constrained
                                            output (02.03/02.04), confidence + span, human review below a threshold
 Every edge carries PROVENANCE: source document + version, character span, extractor name@version, confidence, recorded_at
```

**Why this order?** Your own bill system already knows sponsors, referrals, and votes exactly. Re-extracting them from PDFs with an LLM adds cost and error. Reserve LLMs for relations that exist only in prose.

```python
import re
from dataclasses import dataclass

MCA = r"\d{1,2}-\d{1,2}-\d{3,4}"
RANGE = re.compile(rf"sections?\s+({MCA})\s+through\s+({MCA})", re.I)
LIST = re.compile(rf"sections?\s+((?:{MCA}(?:,\s*|,?\s+and\s+|,?\s+or\s+))*{MCA})", re.I)
ONE = re.compile(MCA)

@dataclass(frozen=True)
class Edge:
    src: str; rel: str; dst: str
    start: int; end: int; extractor: str = "xref-regex@1.2"; confidence: float = 1.0

def _key(mca):
    return tuple(int(x) for x in mca.split("-"))

def extract_xrefs(src_section, text, known_sections):
    """Cross-references with provenance. Ranges expand against the KNOWN section index (never invent sections);
    citations of unknown sections are kept at lower confidence for review."""
    edges, covered = [], []
    for m in RANGE.finditer(text):
        lo, hi = _key(m.group(1)), _key(m.group(2))
        for s in sorted(known_sections, key=_key):
            if lo <= _key(s) <= hi and s != src_section:
                edges.append(Edge(src_section, "CITES", s, m.start(), m.end()))
        covered.append((m.start(), m.end()))
    for m in LIST.finditer(text):
        if any(a <= m.start() < b for a, b in covered):
            continue
        for c in ONE.finditer(m.group(1)):
            s = c.group(0)
            if s != src_section:
                edges.append(Edge(src_section, "CITES", s, m.start(1) + c.start(), m.start(1) + c.end(),
                                  confidence=1.0 if s in known_sections else 0.5))
    return sorted(set(edges), key=lambda e: (e.dst, e.start))

known = {"7-6-4005", "7-6-4006", "7-6-4007", "7-6-4010", "7-6-4012", "15-30-2101", "2-2-102"}
text = ("(1) Except as provided in sections 7-6-4006 through 7-6-4010, the governing body shall adopt a budget. "
        "(2) The hearing required under section 2-2-102 and 15-30-2101 must be noticed as provided in 7-6-4012. "
        "(3) A reference to section 7-6-4099 is void.")
edges = extract_xrefs("7-6-4005", text, known)
for e in edges:
    print(f"{e.src} -CITES-> {e.dst}  conf={e.confidence}  span=({e.start},{e.end}) {text[e.start:e.end][:40]!r}")
assert [e.dst for e in edges] == ["15-30-2101", "2-2-102", "7-6-4006", "7-6-4007", "7-6-4010", "7-6-4099"]
assert next(e for e in edges if e.dst == "7-6-4099").confidence == 0.5
assert all(e.dst != "7-6-4012" for e in edges)     # a KNOWN MISS: "as provided in 7-6-4012" lacks the word "section"
```

What the test deliberately shows:
- The range "7-6-4006 **through** 7-6-4010" expands to the **three known sections** in range, not to 5 invented numbers.
- A citation of the non-existent **7-6-4099** is kept at confidence 0.5 for a human to check. It may be a drafting error worth flagging.
- The **known miss** "as provided in 7-6-4012" is a recall gap. Measure recall on a labelled sample, then either add patterns or send low-recall section types to an **LLM extractor** with structured output. The LLM pass then goes through the same schema validation and provenance.

---

## 5. Entity Resolution

The same person appears as "Rep. John A. Smith (R-HD 12)", "J. Smith", "Representative Smith", and "John Smith". Two *different* Smiths may serve at the same time. The standard pipeline is **blocking → pairwise scoring → clustering → evaluation** (Fellegi–Sunter in spirit; learned matchers such as Ditto at scale):

```python
import re
from collections import defaultdict
from itertools import combinations

TITLES = {"rep", "representative", "sen", "senator", "mr", "ms", "mrs", "dr", "the", "honorable", "chair", "chairman"}

def parse_mention(s):
    """'Rep. John A. Smith (R-HD 12)' -> first, last, initials, party, district."""
    m = re.search(r"\((?P<party>[RDIL])-(?P<ch>HD|SD)\s*(?P<num>\d+)\)", s)
    toks = [t for t in re.findall(r"[a-z]+", re.sub(r"\(.*?\)", "", s).lower()) if t not in TITLES]
    return dict(raw=s, first=toks[0] if len(toks) > 1 else "", last=toks[-1] if toks else "",
                initials={t[0] for t in toks[:-1]}, district=(m.group("ch") + m.group("num")) if m else None)

def match_score(a, b):
    """Hard conflicts first (surname, seat, full first name), then accumulated evidence."""
    if a["last"] != b["last"]:
        return 0.0
    if a["district"] and b["district"] and a["district"] != b["district"]:
        return 0.0
    if len(a["first"]) > 1 and len(b["first"]) > 1 and a["first"] != b["first"]:
        return 0.0
    score = 0.5
    if a["district"] and a["district"] == b["district"]: score += 0.4
    if a["initials"] & b["initials"]: score += 0.2
    if a["first"] and a["first"] == b["first"]: score += 0.2
    return min(score, 1.0)

def resolve(mentions, threshold=0.7):
    """Blocking by surname -> pairwise scoring -> union-find clusters."""
    parsed = [parse_mention(m) for m in mentions]
    parent = list(range(len(parsed)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    blocks = defaultdict(list)
    for i, p in enumerate(parsed):
        blocks[p["last"]].append(i)
    compared = 0
    for ids in blocks.values():
        for i, j in combinations(ids, 2):
            compared += 1
            if match_score(parsed[i], parsed[j]) >= threshold:
                parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i in range(len(parsed)):
        clusters[find(i)].append(mentions[i])
    return list(clusters.values()), compared

def pairwise_prf(pred_clusters, gold_of):
    pairs = lambda cl: {frozenset(p) for c in cl for p in combinations(c, 2)}
    groups = defaultdict(list)
    for m, g in gold_of.items(): groups[g].append(m)
    pred, gold = pairs(pred_clusters), pairs(groups.values())
    tp = len(pred & gold)
    return tp / max(1, len(pred)), tp / max(1, len(gold))

mentions = ["Rep. John A. Smith (R-HD 12)", "J. Smith (R-HD 12)", "Representative Smith (R-HD 12)", "John Smith",
            "Rep. James Smith (D-HD 40)", "Sen. Maria Lopez (D-SD 7)", "Senator M. Lopez", "Lopez (D-SD 7)",
            "Rep. Smith (D-HD 40)"]
gold = dict(zip(mentions, ["smith12"] * 4 + ["smith40", "lopez7", "lopez7", "lopez7", "smith40"]))
clusters, compared = resolve(mentions)
for c in clusters:
    print(c)
p, r = pairwise_prf(clusters, gold)
print(f"pairwise precision={p:.2f} recall={r:.2f}; compared {compared} pairs vs {len(mentions) * (len(mentions) - 1) // 2} without blocking")
assert p == 1.0 and r >= 0.8
```

**Production notes:**
- **Anchor to authoritative IDs.** The legislature's own legislator IDs are the canonical key; ER maps free-text mentions onto them.
- **Hard conflicts beat evidence.** The rules for different seats and different full first names prevent the classic false merge of two Smiths.
- **Union-find is transitive.** One bad pairwise link merges two whole clusters. Guard with conflict checks at cluster level, not only pair level. Review merges above a cluster size, and keep merges **reversible** (store the mention→entity mapping, never destructively merge nodes).
- **Blocking** cut the comparisons in half here. At scale it is the difference between $O(n^2)$ and feasible. Measure **blocking recall** (true pairs that share a block) separately.

---

## 6. Time: Bitemporal Edges for Law That Changes

Law has two independent timelines:
- **valid time:** when a provision was in force (effective dates);
- **transaction time:** when *your system* recorded it, including late corrections.

Questions like "what did the law say on date X, as we understood it on date Y?" need both. They matter for audits, for reproducing a past AI answer, and for litigation support.

```python
from dataclasses import dataclass
from datetime import date

INF = date.max

@dataclass(frozen=True)
class TEdge:
    src: str; rel: str; dst: str
    valid_from: date; valid_to: date          # legal effectiveness
    recorded_at: date                         # when the graph learned it
    source: str                               # provenance (chapter law / enrolled bill)

def as_of(edges, valid_on: date, known_on: date = INF):
    """Facts in force on `valid_on`, as known on `known_on`: for each (src, rel, dst, valid_from) keep the
    latest record written on or before `known_on`, then filter by the valid-time interval."""
    latest = {}
    for e in edges:
        if e.recorded_at > known_on:
            continue
        k = (e.src, e.rel, e.dst, e.valid_from)
        if k not in latest or e.recorded_at > latest[k].recorded_at:
            latest[k] = e
    return sorted((e.src, e.rel, e.dst) for e in latest.values() if e.valid_from <= valid_on < e.valid_to)

E = [
    TEdge("7-6-4005", "CITES", "7-6-4006",   date(2019, 10, 1), INF,               date(2019, 6, 1),  "Ch. 102, L. 2019"),
    TEdge("7-6-4005", "CITES", "2-2-102",    date(2019, 10, 1), INF,               date(2019, 6, 1),  "Ch. 102, L. 2019"),
    # HB 12 (2025) closes the 2-2-102 reference and adds 15-30-2101, effective Oct 1, 2025:
    TEdge("7-6-4005", "CITES", "2-2-102",    date(2019, 10, 1), date(2025, 10, 1), date(2025, 5, 20), "Ch. 311, L. 2025"),
    TEdge("7-6-4005", "CITES", "15-30-2101", date(2025, 10, 1), INF,               date(2025, 5, 20), "Ch. 311, L. 2025"),
    # late correction: the 2019 law also cited 7-6-4007 (data-entry miss found in 2026)
    TEdge("7-6-4005", "CITES", "7-6-4007",   date(2019, 10, 1), INF,               date(2026, 2, 3),  "Ch. 102, L. 2019 (correction)"),
]
print("law on 2024-01-01, as known today:  ", as_of(E, date(2024, 1, 1)))
print("law on 2024-01-01, as known 2025-12:", as_of(E, date(2024, 1, 1), known_on=date(2025, 12, 31)))
print("law on 2026-01-01, as known today:  ", as_of(E, date(2026, 1, 1)))
print("law on 2026-01-01, as known 2025-01:", as_of(E, date(2026, 1, 1), known_on=date(2025, 1, 31)))
assert ("7-6-4005", "CITES", "7-6-4007") in as_of(E, date(2024, 1, 1))
assert ("7-6-4005", "CITES", "7-6-4007") not in as_of(E, date(2024, 1, 1), known_on=date(2025, 12, 31))
assert ("7-6-4005", "CITES", "2-2-102") not in as_of(E, date(2026, 1, 1))
assert ("7-6-4005", "CITES", "2-2-102") in as_of(E, date(2026, 1, 1), known_on=date(2025, 1, 31))   # pre-HB 12 view
```

The four queries answer four different, legitimate questions:
- the law in 2024 *with* the 2026 correction;
- the law in 2024 as we believed it *before* the correction, which is what an AI answer given in 2025 would have been grounded in;
- the law after HB 12;
- the future as seen before HB 12 passed.

**Log `known_on` (the graph snapshot) with every AI answer** (06.04), so any answer can be reproduced exactly.

---

## 7. Querying Safely, Including LLM-Generated Cypher

**Text-to-Cypher** lets staff ask graph questions in English. It inherits every risk of text-to-SQL (07.02 §4). Apply layered defences:
1. **Read-only database credentials** (a role without write privileges; for AGE, a Postgres role limited to SELECT on the graph schema).
2. **Parameterised queries.** Put values in `$params`, never string-concatenate.
3. **Static guard:** no write clauses, no procedure calls outside an allowlist, labels and relationship types limited to the schema, and an enforced `LIMIT`.
4. **Timeouts and result-size caps.** Variable-length paths `*` without an upper bound can explode, so require bounded ranges.
5. **Evaluation:** execution accuracy on a labelled question→answer set, not string match on the query.

```python
import re

WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|ALTER|GRANT|DENY|REVOKE)\b", re.I)
CALL = re.compile(r"\bCALL\s+([\w.]+)", re.I)

def guard_cypher(q, allowed_labels, allowed_rels, allowed_procs=frozenset({"db.index.fulltext.queryNodes"}),
                 max_limit=100):
    """Static guard for LLM-generated Cypher; defence in depth on top of a read-only DB role.
    Returns (ok, query_with_limit_or_reason)."""
    body = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "''", q)             # blank out string literals
    body = re.sub(r"//.*?$|/\*.*?\*/", " ", body, flags=re.S | re.M)             # strip comments
    if ";" in body.strip().rstrip(";"):
        return False, "multiple statements"
    if WRITE.search(body):
        return False, f"write clause: {WRITE.search(body).group(1)}"
    for proc in CALL.findall(body):
        if proc not in allowed_procs:
            return False, f"procedure not allowed: {proc}"
    if re.search(r"\*\s*(\.\.)?\s*\]", body) or re.search(r"\*\s*\d+\s*\.\.\s*\]", body):
        return False, "unbounded variable-length path"
    labels = set(re.findall(r"\(\s*\w*\s*:\s*`?(\w+)`?", body)) | set(re.findall(r"\)\s*:\s*`?(\w+)`?", body))
    rels = set(re.findall(r"\[\s*\w*\s*:\s*`?(\w+)`?", body))
    for bad in sorted(labels - allowed_labels):
        return False, f"label not allowed: {bad}"
    for bad in sorted(rels - allowed_rels):
        return False, f"relationship not allowed: {bad}"
    m = list(re.finditer(r"\bLIMIT\s+(\d+)\b", body, re.I))
    if not m:
        return True, q.rstrip().rstrip(";") + f" LIMIT {max_limit}"
    if int(m[-1].group(1)) > max_limit:
        return False, f"LIMIT {m[-1].group(1)} > {max_limit}"
    return True, q

L = {"Bill", "Section", "Legislator", "Committee"}
R = {"AMENDS", "REPEALS", "CITES", "SPONSORED_BY", "REFERRED_TO"}
tests = {
    "MATCH (b:Bill {session:'2025'})-[:AMENDS]->(s:Section) WHERE s.mca STARTS WITH '7-6-' RETURN b.id, s.mca": True,
    "MATCH (s:Section)<-[:CITES*1..3]-(x:Section) WHERE s.mca = $mca RETURN DISTINCT x.mca LIMIT 50": True,
    "MATCH (b:Bill) WHERE b.title = 'DELETE me; SET x' RETURN b LIMIT 5": True,         # keywords inside strings are fine
    "MATCH (s:Section)<-[:CITES*]-(x) RETURN x": False,                                  # unbounded path
    "MATCH (b:Bill) DETACH DELETE b": False,
    "MATCH (u:User) RETURN u.email": False,                                             # label not in schema
    "CALL apoc.load.json('http://evil/x') YIELD value RETURN value": False,
    "MATCH (b:Bill) RETURN b LIMIT 100000": False,
    "MATCH (b:Bill) RETURN b; MATCH (n) DELETE n": False,
    "LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row": False,
}
for q, expect in tests.items():
    ok, out = guard_cypher(q, L, R)
    print(("ALLOW" if ok else "BLOCK"), "|", out[:95])
    assert ok == expect, q
```

**Better than free-form generation: template-first.** For the 20 most common question shapes (impact of amending X; bills by sponsor; referrals by committee), expose **parameterised query tools** through MCP (05.04). The LLM picks a tool and fills parameters, which is safer, faster, and testable. Free-form text-to-Cypher behind the guard is reserved for the long tail.

---

## 8. Using the Graph: Impact Analysis, Centrality, Grounded Answers

**Impact analysis** is the drafter's killer feature. Before amending a section, list every section that references it, directly or transitively, with the citation path as an explanation:

```python
import networkx as nx

def impact_of_amending(G, section, max_depth=3):
    """Sections citing `section` directly or transitively (reverse CITES), each with one shortest path."""
    R = G.reverse(copy=False)
    dist = nx.single_source_shortest_path_length(R, section, cutoff=max_depth)
    paths = nx.single_source_shortest_path(R, section, cutoff=max_depth)
    return sorted(((d, n, " <- ".join(paths[n])) for n, d in dist.items() if n != section), key=lambda x: (x[0], x[1]))

G = nx.DiGraph()   # u -> v means "u CITES v"
G.add_edges_from([("7-6-4005", "7-6-4006"), ("7-6-4005", "15-30-2101"), ("7-6-4020", "7-6-4005"),
                  ("7-6-4030", "7-6-4020"), ("7-6-4101", "7-6-4005"), ("20-9-141", "7-6-4030"),
                  ("15-30-2101", "15-1-101"), ("2-2-102", "15-1-101"), ("7-6-4101", "15-1-101")])
for depth, sec, path in impact_of_amending(G, "7-6-4005"):
    print(f"depth {depth}: {sec:10s} via {path}")
assert {s for _, s, _ in impact_of_amending(G, "7-6-4005")} == {"7-6-4020", "7-6-4101", "7-6-4030", "20-9-141"}
pr = nx.pagerank(G)
print("most-cited (PageRank):", [s for s, _ in sorted(pr.items(), key=lambda x: -x[1])[:3]])
assert max(pr, key=pr.get) == "15-1-101"
```

- **Centrality** (PageRank on CITES) surfaces the *load-bearing* sections, such as definitions sections. Changes to them deserve extra review. It is also a useful prior for retrieval ranking.
- **Grounded answers with the graph** combine the pieces. RAG retrieves text chunks (04.x), and the graph supplies relationships: impact paths, sponsors, and effective dates **as of** the question's date (§6). The LLM composes an answer citing both chunk IDs and graph facts, each with provenance. The router (05.02) picks graph tools for relational questions and retrieval for "what does it say" questions.
- **GraphRAG (04.05)** community summaries can be built *on top of this curated graph* instead of an LLM-extracted one. You get better communities and fewer spurious edges.

---

## 9. Operating a Knowledge Graph

| Concern | Practice |
|---|---|
| **Ingestion** | Kafka CDC from the bill system → idempotent upserts keyed by natural IDs (`bill_id`, `mca`, `legislator_id`); document pipeline events (10.01) → extraction jobs (10.03 §3) |
| **Quality metrics** | Constraint violations (should be 0), orphan nodes, edges with confidence < τ awaiting review, extraction precision/recall on a labelled sample, ER merge-review backlog |
| **Provenance and audit** | Every edge: source, span, extractor@version, recorded_at; graph snapshots referenced by AI answers |
| **Schema evolution** | Versioned migrations; backfill jobs; compatibility with queries and tools (contract tests) |
| **Access control** | Read-only roles for AI tools; row/property-level restrictions for non-public data (draft requests are confidential until introduced!) |
| **Performance** | Indexes on lookup keys (`Section.mca`, `Bill.id`), bounded traversals, precomputed impact closures for hot sections |

**Confidentiality matters here.** Bill *drafting requests* are typically confidential until introduction. A graph that joins requests to sponsors must enforce access at query time, and so must every AI tool built on it (07.02 §5, 03.03 §6).

---

## 10. Production Challenges and Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **Schema drift** | Ad-hoc labels and properties; queries break | Enforced schema + validator in ingestion; versioned migrations |
| **LLM-extracted edges wrong** | Spurious relationships | Structured sources and rules first; confidence + review; precision sampling |
| **Over-merged entities** | Two legislators become one | Hard-conflict rules, cluster-level checks, reversible mappings, authoritative IDs |
| **Temporal confusion** | Answers mix old and new law | Bitemporal edges; `as_of` in every query tool; log the snapshot per answer |
| **Unsafe text-to-Cypher** | Writes, exfiltration, runaway queries | Read-only role + guard + bounded paths + timeouts; templated tools first |
| **Recall gaps in extraction** | Missing cross-references | Labelled recall eval; LLM fallback for gap patterns |
| **Two sources of truth** | Graph disagrees with the bill system | The bill system is the system of record; the graph is derived and rebuildable from events |
| **Confidential data leakage** | Draft requests visible through graph tools | Query-time authorisation; separate subgraphs or properties by classification |

---

## 11. Hands-On Projects

### Project 1 — MCA Cross-Reference Graph and Drafter Impact Tool

**User stories**
- *As a bill drafter*, before I amend a section I want to see every section that references it (with paths and effective dates), so that I can draft conforming amendments and avoid broken references.

**Acceptance criteria**
- The graph covers all MCA sections, with CITES edges extracted by rules + LLM fallback. Precision ≥ 0.98 and recall ≥ 0.95 on a 500-reference labelled sample.
- An impact tool (API + MCP tool) returns depth-bounded dependents with paths, as of a given date, in < 300 ms p95.
- It flags citations to non-existent or repealed sections, and a report of such references goes to code commissioner staff.
- It is deployed on Postgres + Apache AGE (or Neo4j), with a rebuild-from-source pipeline.

**Step-by-step**
1. Parse the MCA into canonical sections (10.01 §3), then run `extract_xrefs`. Label a sample and measure precision and recall.
2. Add LLM extraction for missed patterns (structured output), then validate and review.
3. Load into AGE with bitemporal edge properties, index the keys, and build the impact query (bounded `*1..k`).
4. Expose the tool through MCP and the drafting UI (10.04), with an `as_of` parameter.
5. Build the broken-reference report and schedule nightly rebuilds.

### Project 2 — Legislative Knowledge Graph with Entity Resolution and Time

**User stories**
- *As a research analyst*, I want to ask relational questions across sessions (sponsors, committees, versions, votes, sections amended), answered correctly for any date.

**Acceptance criteria**
- Ingestion from the bill-tracking DB via Kafka CDC (idempotent), plus historical backfill for ≥ 3 sessions.
- ER maps free-text names (hearing transcripts from 10.01 Project 2, and press mentions) to legislator IDs with pairwise precision ≥ 0.99 and recall ≥ 0.9 on a labelled set. All merges are reversible.
- Bitemporal queries reproduce historical answers, verified on 20 known historical facts and corrections.
- Constraint violations are 0 in production, and a data-quality dashboard is live.

**Step-by-step**
1. Design the ontology from competency questions and review it with legislative staff.
2. Build CDC consumers to upsert into the graph, with a schema validator gate.
3. Build the ER pipeline (blocking, scoring, clustering, review UI).
4. Implement valid/transaction time on edges and an `as_of` query layer.
5. Run the historical-fact test suite and ship the dashboards.

### Project 3 — Graph-Grounded Q&A with Safe Text-to-Cypher

**User stories**
- *As legislative staff*, I want to ask questions in English that combine "what does it say" with "who/what/when is related", and get answers citing both text and graph facts.

**Acceptance criteria**
- The router chooses between templated graph tools, guarded text-to-Cypher, and RAG. Routing accuracy is ≥ 95% on a 300-question labelled set.
- Execution accuracy for text-to-Cypher is ≥ 85%, with **0** guard bypasses in a red-team suite of 100 adversarial prompts (write attempts, label smuggling, unbounded paths, injected instructions in retrieved text).
- Answers cite chunk IDs and graph edges (with provenance and an `as_of` snapshot). The groundedness guard (07.01) passes ≥ 98%.
- Latency is p95 ≤ 4 s end-to-end. Traces show the tool calls and queries (06.04).

**Step-by-step**
1. Build templated tools for the top question shapes, then add the guarded text-to-Cypher with schema-aware prompting (labels, relationship types, examples).
2. Build a labelled question set with gold answers, and measure execution accuracy.
3. Assemble a red-team suite (07.03) and harden the guard and DB role.
4. Write the answer composer with dual citations, and add the groundedness guard.
5. Evaluate end-to-end and ship through the 08.05 gates.

---

## 12. Foundational Papers and Standards (exact titles)

- Hogan et al., 2021 — *Knowledge Graphs* (ACM Computing Surveys)
- Pan et al., 2023 — *Unifying Large Language Models and Knowledge Graphs: A Roadmap*
- Edge et al., 2024 — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* (see 04.05)
- Bordes et al., 2013 — *Translating Embeddings for Modeling Multi-relational Data* (TransE)
- Fellegi & Sunter, 1969 — *A Theory for Record Linkage*
- Li et al., 2020 — *Deep Entity Matching with Pre-Trained Language Models* (Ditto)
- Papadakis et al., 2020 — *Blocking and Filtering Techniques for Entity Resolution: A Survey*
- Snodgrass & Ahn, 1985 — *A Taxonomy of Time in Databases*
- Francis et al., 2018 — *Cypher: An Evolving Query Language for Property Graphs*
- Ozsoy et al., 2024 — *Text2Cypher: Bridging Natural Language and Graph Databases*
- ISO/IEC 39075:2024 — *Information technology — Database languages — GQL*
- ISO/IEC 9075-16:2023 — *SQL/PGQ (Property Graph Queries)*
- W3C — *SHACL: Shapes Constraint Language*; *SPARQL 1.1 Query Language*; *OWL 2 Web Ontology Language*
- OASIS — *Akoma Ntoso Version 1.0* (legal document XML standard)

## 13. Essential Tooling

| Tool | Use |
|---|---|
| **Apache AGE** (Postgres; supported on Azure Database for PostgreSQL flexible server) | openCypher next to pgvector and relational data |
| **Neo4j** (+ GDS, APOC with restricted procedures) | Dedicated graph DB, algorithms, tooling |
| **networkx / igraph** | Offline analysis, prototyping algorithms |
| **Apache Jena / GraphDB / rdflib, pySHACL** | RDF, SPARQL, SHACL validation |
| **Splink, Dedupe, Zingg** | Scalable entity resolution |
| **LLM extraction with structured outputs** (02.03), Neo4j LLM Graph Builder | Relation extraction with schemas |
| **Kafka CDC (Debezium)** | Keeping the graph in sync with the systems of record |

**Image prompt (legislative KG):** *"Knowledge-graph visualisation on white: coloured nodes — Bill (blue), Section (green), Legislator (orange), Committee (purple), BillVersion (light blue) — with labelled arrows AMENDS, CITES, SPONSORED_BY, REFERRED_TO, HAS_VERSION. One Section node highlighted with a red halo and its transitive CITES dependents highlighted, labelled 'impact analysis'. A small timeline strip below showing valid_from/valid_to bars on two CITES edges. Clean, flat, legible labels."*
