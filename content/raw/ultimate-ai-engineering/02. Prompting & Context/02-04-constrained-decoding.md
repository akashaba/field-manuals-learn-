# 02.04 — Constrained Decoding

> **Module 2: Prompting & Context** · Subtopic 4 of 4
> **Prerequisites:** automata theory basics (regular expressions, DFAs, context-free grammars, pushdown automata), 01.02 (sampling pipeline, speculative decoding), 01.03 (byte-level BPE, token boundaries), 02.03 (schemas).
> **Outcome:** you can build a constrained-decoding engine from first principles (regex → DFA → token index, and CFG → incremental parser → token mask), reason about its performance and its **distortion of the model's distribution**, and choose and operate a production backend with confidence.

> **Relationship to earlier material:** 01.02 §6 introduced token masking as a serving feature, and 02.03 uses it through provider APIs. This subtopic opens the engine up.

---

## 1. The Problem, Formally

Let $\mathcal{L} \subseteq \Sigma^*$ be the target language over characters (or bytes): a regex, JSON Schema, SQL grammar, etc. The tokenizer vocabulary is $\mathcal{V}$, where each token $v$ decodes to a string $\mathrm{str}(v) \in \Sigma^*$. After generating a string $s$, the **allowed set** is

$$
\mathcal{A}(s) = \Big\{\, v \in \mathcal{V} \;:\; s \cdot \mathrm{str}(v) \in \mathrm{Pref}(\mathcal{L}) \,\Big\} \;\cup\; \big\{\texttt{EOS} \;:\; s \in \mathcal{L}\big\}
$$

where $\mathrm{Pref}(\mathcal{L})$ is the set of prefixes of strings in $\mathcal{L}$. For a live prefix (one that can still be completed), the masked sampling step is

$$
\tilde p(v \mid s) = \frac{p(v \mid s)\,\mathbb{1}[v \in \mathcal{A}(s)]}{\sum_{u \in \mathcal{A}(s)} p(u \mid s)}
$$

Two engineering problems fall out of this definition:

1. **Speed.** Computing $\mathcal{A}(s)$ naively means checking $|\mathcal{V}| \approx 10^5$ tokens per step. At 50 tokens/s per sequence and batch 64, that's hundreds of millions of prefix checks per second. The mask must be ready *before* the GPU finishes the forward pass (≈ 10–30 ms), or it becomes the bottleneck.
2. **Faithfulness.** The locally renormalised $\tilde p$ is **not** the model's distribution conditioned on $\mathcal{L}$ (§6).


![LLM-02-2](/02.%20Prompting%20&%20Context/images/LLM-02-2.jpg)


---

## 2. Baseline: Check Every Token

The simplest correct engine checks each token with a **partial-match** oracle. The `regex` module supports `partial=True`, which returns a match if the string is a prefix of some match.

```python
import regex


def allowed_tokens_naive(pattern: str, prefix: str, vocab: dict[int, str], eos_id: int) -> set[int]:
    """O(|V| * len) per step. Correct, far too slow for serving; use as a test oracle."""
    rx = regex.compile(pattern)
    allowed = {tid for tid, tok in vocab.items()
               if tok and rx.fullmatch(prefix + tok, partial=True) is not None}
    if rx.fullmatch(prefix) is not None:
        allowed.add(eos_id)
    return allowed
```

Keep this as the **reference oracle** for property-based tests of any faster engine.

---

## 3. Regular Languages: Regex → DFA → Token Index

### 3.1 The key idea (Willard & Louf, 2023)

Compile the regex to a **DFA** $M = (Q, \Sigma, \delta, q_0, F)$ and remove dead states (states from which no accepting state is reachable). For every state $q$ and every token $v$, run the DFA over the token's characters:

$$
\delta^*(q, v) = \delta(\ldots\delta(\delta(q, c_1), c_2)\ldots, c_n), \qquad \mathrm{str}(v) = c_1 c_2 \ldots c_n
$$

Precompute the **token index** $I[q] = \{\, v \mapsto \delta^*(q,v) \;:\; \delta^*(q, v) \text{ defined and live} \}$. At decode time, the mask is a **lookup** $I[q_t]$, and the state update is $q_{t+1} = I[q_t][v_t]$: $O(1)$ per step after an $O(|Q|\cdot|\mathcal{V}|\cdot \bar\ell)$ precompute, where $\bar\ell$ is the mean token length.

```
 regex  ──(Thompson/Brzozowski)──►  NFA ──(subset construction)──► DFA ──(minimise, prune dead)──► M
                                                                                                    │
                                          for q in Q: for v in V: walk δ over str(v)                ▼
                                                                                     token index I[q] = {v: q'}
```

```python
import interegular


def compile_regex_index(pattern: str, vocab: dict[int, str]):
    """Build {state: {token_id: next_state}} plus accepting states. Educational, not optimized."""
    fsm = interegular.parse_pattern(pattern).to_fsm()

    # Prune dead states: keep states that can reach an accepting state.
    rev: dict[int, set[int]] = {q: set() for q in fsm.states}
    for q, edges in fsm.map.items():
        for nxt in edges.values():
            rev.setdefault(nxt, set()).add(q)
    live, frontier = set(fsm.finals), list(fsm.finals)
    while frontier:
        for p in rev.get(frontier.pop(), ()):
            if p not in live:
                live.add(p)
                frontier.append(p)

    def walk(q, text):
        for ch in text:
            q = fsm.map.get(q, {}).get(fsm.alphabet[ch])
            if q is None or q not in live:
                return None
        return q

    index = {q: {tid: nq for tid, tok in vocab.items()
                 if tok and (nq := walk(q, tok)) is not None}
             for q in fsm.states if q in live}
    return fsm.initial, index, set(fsm.finals)


class RegexConstraint:
    def __init__(self, pattern: str, vocab: dict[int, str], eos_id: int):
        self.state, self.index, self.finals = compile_regex_index(pattern, vocab)
        self.eos_id = eos_id

    def allowed(self) -> set[int]:
        ids = set(self.index.get(self.state, {}))
        return ids | {self.eos_id} if self.state in self.finals else ids

    def advance(self, token_id: int) -> None:
        if token_id != self.eos_id:
            self.state = self.index[self.state][token_id]
```

### 3.2 JSON Schema → regex

Bounded JSON Schemas (no recursion, no unbounded nesting) compile to regular expressions. For example:

```
{"type":"object","properties":{"id":{"type":"string","pattern":"HB-\\d+"},"n":{"type":"integer"}},
 "required":["id","n"],"additionalProperties":false}

  ⟶  \{[ ]?"id"[ ]?:[ ]?"HB-\d+"[ ]?,[ ]?"n"[ ]?:[ ]?(-)?(0|[1-9][0-9]*)[ ]?\}
```

**Design decisions hidden in this compilation:**

- **Whitespace policy.** Allowing arbitrary whitespace (`[ \n\t]*`) lets the model emit whitespace forever. Some models do exactly that when the constraint is fighting their preferences, which is known as **whitespace runaway**. Fix it by allowing at most one space, or compact JSON only.
- **Property order.** Fixed order yields a small DFA. Allowing any order is factorial in the number of properties, so engines fix the order (typically schema order).
- **String contents.** `"([^"\\\x00-\x1f]|\\["\\/bfnrtu])*"` — unbounded strings are fine in a DFA, but **bounded lengths** (`{1,256}`) blow up the state count. Enforce long bounds in post-validation instead (02.03 §4).
- **Numbers:** integers vs floats vs exponent forms. Numeric *ranges* are expressible as regexes, but produce large automata.

### 3.3 Where regular languages stop

Arbitrarily nested JSON, recursive schemas, balanced parentheses, and programming-language syntax are **not regular**. A DFA would need unbounded memory. You can unroll nesting to a fixed depth (the DFA grows exponentially), or move to context-free grammars.

---

## 4. Context-Free Grammars: Incremental Parsing + Token Masks

### 4.1 From DFA state to parser state

For a CFG, the decoder state is a **parser configuration**: an LR stack, a set of Earley items, or a pushdown automaton's (state, stack). "Is $s\cdot\mathrm{str}(v)$ a viable prefix?" becomes "can the parser consume these characters without error?"

The central performance problem: **the parser state includes a stack, so you cannot precompute a finite index over all states.** Production engines attack this in several ways:

| Engine | Key idea |
|---|---|
| **XGrammar** (Dong et al., 2024) | Split tokens into **context-independent** ones (validity determined by the top of the stack alone; precomputed per grammar position) and **context-dependent** ones (checked at runtime against the full stack). The first group is the vast majority. Uses a persistent stack for cheap branching, and overlaps mask computation with the GPU forward pass |
| **llguidance** | A lexer (regex-derived, token-indexed like §3) plus an Earley parser over lexemes. Masks are computed lazily by walking a **token trie** with the parser, sharing work across tokens with common prefixes |
| **SynCode** (Ugare et al., 2024) | Precomputed "DFA mask store" for terminals, plus LR(1) parser acceptance sequences |
| **Outlines (CFG mode)**, **llama.cpp GBNF** | Incremental parsers with per-step checks; simpler, slower on large vocabularies |

### 4.2 Token-trie mask computation

Tokens share prefixes (`"`, `",`, `"}`, `"],`…). Organise $\mathcal{V}$ as a **trie** of token strings and do a DFS that carries the parser state. When a character is rejected, prune the whole subtree. Each node is visited once, and shared prefixes are parsed once.

```
 trie:  (root)
         ├─ "[" ──► accept? ── "[[" ── …
         ├─ "]" ──► …
         ├─ "," ──► …
         └─ "1" ── "12" ── "123"          one parser step per edge, not per token
```

```python
from dataclasses import dataclass, field

State = tuple[int, str]   # (depth, mode) — a counter suffices for one bracket type


def step(state: State, ch: str) -> State | None:
    """Incremental recognizer for nested integer lists, e.g. [1,[2,3],[]]. Grammar:
       L -> '[' (E (',' E)*)? ']' ;  E -> INT | L ;  INT -> [0-9]+"""
    depth, mode = state
    if mode == "int":
        if ch.isdigit():
            return state
        mode = "after"                       # integer ended; process ch in 'after' mode
    if mode == "start":
        return (1, "open") if ch == "[" else None
    if mode in ("open", "elem"):
        if ch == "[":
            return (depth + 1, "open")
        if ch.isdigit():
            return (depth, "int")
        if ch == "]" and mode == "open":
            return (depth - 1, "done") if depth == 1 else (depth - 1, "after")
        return None
    if mode == "after":
        if ch == ",":
            return (depth, "elem")
        if ch == "]":
            return (depth - 1, "done") if depth == 1 else (depth - 1, "after")
        return None
    return None                              # 'done': nothing may follow


def accepting(state: State) -> bool:
    return state[1] == "done"


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    token_ids: list[int] = field(default_factory=list)


def build_trie(vocab: dict[int, str]) -> TrieNode:
    root = TrieNode()
    for tid, tok in vocab.items():
        node = root
        for ch in tok:
            node = node.children.setdefault(ch, TrieNode())
        node.token_ids.append(tid)
    return root


def trie_mask(root: TrieNode, state: State, eos_id: int) -> set[int]:
    allowed, stack = set(), [(root, state)]
    while stack:
        node, st = stack.pop()
        for ch, child in node.children.items():
            nst = step(st, ch)
            if nst is None:
                continue                      # prune entire subtree
            allowed.update(child.token_ids)
            stack.append((child, nst))
    if accepting(state):
        allowed.add(eos_id)
    return allowed
```

---

## 5. Token-Boundary Problems (Where Naive Engines Go Wrong)

Grammars are defined over characters, but models emit **tokens**, and the two don't align.

1. **Tokens spanning grammar symbols.** `",\n  "` crosses a comma, whitespace, and a string opener. The engine must walk *all* the characters of each token (as in §3–4), not map tokens to grammar terminals one-to-one.
2. **Non-canonical tokenisations.** The mask admits any token sequence whose *string* is valid, including tokenisations the model rarely saw in training (e.g. `"na" + "me"` instead of `"name"`). Once the model is forced onto such a path, its quality degrades. This **token misalignment** was characterised and addressed with subword-aligned constraints in DOMINO (Beurer-Kellner et al., 2024).
3. **Prompt/output boundary (token healing).** If the prompt ends in the middle of what would normally be one token (e.g. a template ends with `{"name": "`), the model's natural continuation token may start with characters already in the prompt. Engines "heal" by backing up one token and constraining the regeneration (01.03 §8).
4. **Byte-level tokens and UTF-8.** Byte-level BPE tokens can be *partial UTF-8 characters*. Engines working over bytes must handle incomplete code points, and must reject byte sequences that can never complete to valid UTF-8 inside a string.
5. **EOS and special tokens.** EOS is allowed only in accepting states. Other special tokens (tool-call markers, end-of-turn) must be explicitly allowed or blocked according to the chat template.

---

## 6. Distribution Distortion: The Part Most Engineers Miss

### 6.1 Local masking ≠ global conditioning

What we usually *want* is the model's distribution **conditioned** on validity:

$$
\pi(y) = \frac{p(y)\,\mathbb{1}[y \in \mathcal{L}]}{Z}, \qquad Z = \sum_{y' \in \mathcal{L}} p(y')
$$

What token masking actually samples from is the **locally renormalised** product:

$$
q(y) = \prod_{t=1}^{|y|} \frac{p(y_t \mid y_{<t})\,\mathbb{1}[y_t \in \mathcal{A}_t]}{Z_t},
\qquad Z_t = \sum_{v \in \mathcal{A}_t} p(v \mid y_{<t})
$$

These two coincide only when every allowed prefix carries the same fraction of its probability mass into $\mathcal{L}$. In general, masking **over-weights prefixes whose continuations the model considered unlikely**, because the renormalisation at later steps "rescues" them. Their ratio is exactly:

$$
\frac{\pi(y)}{q(y)} = \frac{\prod_t Z_t}{Z}
\quad\Longrightarrow\quad
w(y) = \prod_{t} Z_t \ \text{ is an unnormalised importance weight}
$$

### 6.2 Worked example (runnable)

A toy LM over $\{a, b\}$ generates 2 tokens. The model strongly prefers $a$ first, but after $a$ it almost always wants another $a$ — and $aa$ is **forbidden**:

```python
from itertools import product

P1 = {"a": 0.9, "b": 0.1}                          # p(y1)
P2 = {"a": {"a": 0.99, "b": 0.01},                 # p(y2 | y1)
      "b": {"a": 0.5, "b": 0.5}}
VALID = {"ab", "ba", "bb"}                         # 'aa' forbidden

p = {x + y: P1[x] * P2[x][y] for x, y in product("ab", "ab")}
Z = sum(p[s] for s in VALID)
target = {s: p[s] / Z for s in VALID}              # π: true conditional

def allowed1():                                    # first tokens that can still complete
    return {x for x in "ab" if any(s[0] == x for s in VALID)}

def allowed2(x):
    return {y for y in "ab" if x + y in VALID}

q, w = {}, {}
for x in allowed1():
    z1 = sum(P1[u] for u in allowed1())
    for y in allowed2(x):
        z2 = sum(P2[x][u] for u in allowed2(x))
        q[x + y] = (P1[x] / z1) * (P2[x][y] / z2)  # masked sampler
        w[x + y] = z1 * z2                          # importance weight Π Z_t

print({s: round(target[s], 3) for s in sorted(VALID)})  # {'ab': 0.083, 'ba': 0.459, 'bb': 0.459}
print({s: round(q[s], 3) for s in sorted(VALID)})       # {'ab': 0.9,   'ba': 0.05,  'bb': 0.05}
```

The masked sampler outputs `ab` 90% of the time, although the model assigns it only 8% of the valid mass. The model "wanted" `a`, then was forced into a continuation it considered very unlikely. **In practice** this shows up as outputs that are syntactically valid but semantically odd: truncated strings, default-looking values, or fields filled with whatever tokens happened to survive the mask.

### 6.3 Mitigations

| Approach | Idea | Cost |
|---|---|---|
| **Align the prompt with the constraint** | Describe the schema and give examples, so $p$ already puts most of its mass on $\mathcal{L}$ and the $Z_t \approx 1$ | Free; the most important fix in practice |
| **Rejection sampling** | Sample unconstrained; keep samples in $\mathcal{L}$ | Exact $\pi$; wasteful when $Z$ is small |
| **Importance resampling / SMC** | Run $N$ masked particles, weight by $\prod_t Z_t$, resample (Loula et al., 2025) | Asymptotically exact; $N$× compute |
| **Grammar-Aligned Decoding (ASAp)** (Park et al., 2024) | Iteratively learn corrections to $Z_t$ from past samples so that sampling converges to $\pi$ | Multiple passes; research-grade |
| **Two-step: free generation → constrained extraction** | Reason freely, then convert to structure | Extra call; preserves reasoning quality (02.03 §3.1) |

A useful diagnostic: **log $\sum_t \log Z_t$ per request**. Large negative values mean the constraint fought the model hard. Those outputs deserve extra validation, and they often point to a prompt or schema problem.

---

## 7. Performance Engineering

### 7.1 Budget

```
  GPU forward pass (decode step, batch B) ────────────────────────────  ~10–30 ms
  CPU: compute masks for B sequences      ────────────                   must finish inside this window
  apply mask (bitmask → logits, on GPU)   ─                              ~µs with a fused kernel
```

- **Bitmasks.** Pack $|\mathcal{V}|$ booleans into $\lceil |\mathcal{V}|/32 \rceil$ int32 words (≈ 4 KB for 128k tokens). Copy to the GPU, then apply with a fused kernel that sets masked logits to $-\infty$.
- **Overlap.** Compute the masks for step $t+1$ on the CPU while the GPU runs step $t$'s forward pass (XGrammar/llguidance integrations in vLLM and SGLang do this).
- **Cache by state.** For DFAs, masks are per-state and fully precomputable. For CFGs, cache by (grammar position, top-of-stack) for context-independent tokens.
- **Compile caching.** Grammar compilation (seconds for large schemas) is keyed by the schema hash; cache it across requests and replicas. This is why providers report first-request latency (02.03 §2.1).

```python
import numpy as np


def pack_bitmask(allowed_ids, vocab_size: int) -> np.ndarray:
    words = np.zeros((vocab_size + 31) // 32, dtype=np.uint32)
    ids = np.fromiter(allowed_ids, dtype=np.int64)
    np.bitwise_or.at(words, ids // 32, (np.uint32(1) << (ids % 32).astype(np.uint32)))
    return words.view(np.int32)


def apply_bitmask(logits: np.ndarray, words: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")[: logits.shape[-1]].astype(bool)
    return np.where(bits, logits, -np.inf)
```

### 7.2 Jump-forward decoding

When the constraint leaves exactly **one** valid continuation for several characters (e.g. after `{"bill_id": ` in a fixed-order schema the next characters must be `"`), the engine can append them *without calling the model* (SGLang's compressed FSM). This saves decode steps on boilerplate-heavy schemas.

**Caveat:** the forced *string* must be retokenized so that the model sees a canonical tokenisation (§5.2), and prefix caching must handle the inserted tokens.

### 7.3 Interaction with other serving features

- **Speculative decoding (01.02 §5):** draft tokens must be validated against the constraint too. Verification applies masks at each drafted position, and grammar state must roll back on rejection.
- **Batching:** each sequence carries its own grammar state. Heterogeneous schemas in one batch are normal; mask generation must be per-sequence and parallel (thread pool).
- **Beam search and parallel sampling:** fork the grammar state cheaply with persistent (immutable) stacks.

---

## 8. Beyond Syntax: Semantic Constraints

The same masking mechanism can enforce **semantic** validity when you can compute allowed continuations:

- **Database-grounded values:** restrict a string field to existing IDs, via a trie of valid values → regex alternation, or a dynamic mask.
- **Schema-aware SQL:** allow only real table and column names in identifier positions, with joins validated against foreign keys (PICARD-style incremental parsing, Scholak et al., 2021).
- **Type-aware code:** monitor-guided decoding queries a static analyser, such as a language server, for valid member names at `.` positions (Agrawal et al., 2023).
- **Tool arguments:** enum values loaded from live configuration.

These constraints must be **cheap** (sub-millisecond per step) or precomputed. Otherwise, validate after generation and repair (02.03 §4).

---

## 9. A Hugging Face Integration Sketch

```python
import torch
from transformers import LogitsProcessor


class DFALogitsProcessor(LogitsProcessor):
    """Batch-aware regex constraint for transformers.generate (educational).

    `vocab` must map token_id -> the exact decoded string contribution of that token.
    Build it with care for byte-level tokenizers, e.g.
      {i: tok.convert_tokens_to_string([t]) for t, i in tok.get_vocab().items()}
    and drop special tokens except EOS.
    """

    def __init__(self, pattern: str, vocab: dict[int, str], eos_id: int, prompt_len: int):
        self.q0, self.index, self.finals = compile_regex_index(pattern, vocab)
        self.eos_id, self.prompt_len = eos_id, prompt_len
        self.states: list[int] | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B = input_ids.shape[0]
        if self.states is None:
            self.states = [self.q0] * B
        elif input_ids.shape[1] > self.prompt_len:            # advance on the last sampled token
            for b in range(B):
                last = int(input_ids[b, -1])
                if last != self.eos_id and self.states[b] is not None:
                    self.states[b] = self.index[self.states[b]].get(last)
        mask = torch.full_like(scores, float("-inf"))
        for b, q in enumerate(self.states):
            if q is None:                                      # should not happen if masks are applied
                mask[b, self.eos_id] = 0.0
                continue
            ids = list(self.index.get(q, {}))
            if q in self.finals:
                ids.append(self.eos_id)
            mask[b, ids] = 0.0
        return scores + mask
```

Use this to *learn* the mechanism. In production, use your engine's native structured-output support (XGrammar or llguidance in vLLM/SGLang), which handles byte-level tokens, caching, overlap, and batching correctly.

---

## 10. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Mask computation bottleneck** | Throughput drops when structured output is on | Use XGrammar/llguidance; overlap with the forward pass; cache compiled grammars; simplify schemas |
| **Grammar compile latency** | First request with a new schema is slow | Precompile on deploy; cache by schema hash across replicas; avoid per-request dynamic schemas |
| **Whitespace/runaway loops** | Output fills `max_tokens` with spaces or newlines | Compact-JSON whitespace policy; bounded whitespace; `max_tokens` guard |
| **Semantically odd but valid output** | Default-looking values, truncated strings | Align the prompt with the schema; log $\sum \log Z_t$; two-step reason→extract; semantic validation |
| **Unsupported schema keywords** | Silently ignored constraints | Test the backend with a conformance suite (02.03 Project 2); enforce in post-validation |
| **Token misalignment** | Degraded quality inside constrained strings | Engines with canonical/subword-aligned handling; token healing at prompt boundaries |
| **Speculative decoding conflicts** | Acceptance rate drops or crashes | Use engine versions that support spec-decode + grammar together; validate drafts under the mask |
| **Recursion or deep nesting** | Compile failures, exponential DFAs | CFG backend; cap depth in the schema; flatten |

---

## 11. Hands-On Projects

### Project 1 — Build a Regex-Constrained Decoding Engine

**User stories**
- *As an inference engineer*, I want to implement regex → DFA → token-index constrained decoding myself, so that I understand and can debug production engines.
- *As a performance engineer*, I want measured mask latency vs vocabulary size and DFA size, so that I know when this approach breaks down.

**Acceptance criteria**
1. `compile_regex_index` works with a real tokenizer (e.g. a ~150k-vocabulary open model), correctly handling byte-level tokens and partial UTF-8. Special tokens are excluded except EOS.
2. **Property test** (Hypothesis): for 20 regexes × 200 random prefixes, the index-based allowed set equals `allowed_tokens_naive` (§2).
3. Integrated into `transformers.generate` via a `LogitsProcessor`: 100% of 500 generations fully match the regex (e.g. dates, IDs, emails, the bounded JSON schemas from §3.2).
4. Benchmarks index build time and memory vs regex complexity, per-step mask time (P50/P99), and throughput impact vs unconstrained decoding.
5. Implements jump-forward for single-path states and measures the saved decode steps on a boilerplate-heavy JSON schema.

**Step-by-step**
1. Build the vocabulary-string map carefully. Test that concatenating the decoded strings of a token sequence equals `tokenizer.decode` of the sequence.
2. Implement §3.1 and optimise the build: walk a token trie per DFA state instead of each token separately (shared prefixes).
3. Write the Hypothesis tests against the naive oracle.
4. Implement the `LogitsProcessor` (§9) with a bitmask path (§7.1).
5. Benchmark with `time.perf_counter_ns`, and profile with py-spy.
6. Add jump-forward: if `len(allowed) == 1` and it isn't EOS, append the token without a forward pass. Then compare outputs and speed.

---

### Project 2 — Grammar-Constrained DSL/SQL Generation with Semantic Constraints

**User stories**
- *As a data-platform engineer*, I want natural-language questions over a legislative database converted into **always-parseable** SQL (or an internal query DSL) that only references real tables and columns, so that the query layer never receives garbage.

**Acceptance criteria**
1. A CFG (GBNF, Lark, or the llguidance/XGrammar format) for a safe SQL subset: `SELECT` only, whitelisted functions, `LIMIT` required.
2. Identifier positions are constrained to the live schema's table and column names (generated from `information_schema` at startup).
3. Served through vLLM or SGLang with a grammar backend. Latency overhead vs unconstrained generation is reported.
4. On ≥ 150 labelled NL→SQL questions: parse rate (target 100%), execution rate, execution accuracy (result-set match), and comparison against unconstrained generation + post-hoc parse + repair.
5. The distortion diagnostic ($\sum \log Z_t$ or a proxy) is logged, and low-score cases are analysed.

**Step-by-step**
1. Load a sample database (e.g. bills, sponsors, committees, votes) into Postgres or SQLite.
2. Write the grammar. Generate the identifier alternations from the schema at startup.
3. Configure the engine's structured-output grammar option; verify with a few hand-written prompts.
4. Build the NL→SQL prompt with schema descriptions and 3–5 retrieved examples (02.01 §3.1).
5. Evaluate constrained vs unconstrained+repair on parse, execution, and accuracy.
6. Write up where constraints helped, where they hurt, and why.

---

### Project 3 — Measuring and Correcting Constraint-Induced Distortion

**User stories**
- *As an applied scientist*, I want to quantify how much masked decoding distorts a model's distribution on realistic constraints, and whether importance-weighted sampling fixes it, so that we know when structured outputs can bias results (e.g. in LLM-based classification or survey simulation).

**Acceptance criteria**
1. Reproduces §6.2 exactly. Then builds a small real-model case: a ~0.5–1B model, a constraint with enumerable outputs (e.g. an enum of 10 multi-token labels, or dates within one month), and exact $\pi$ computed by scoring every valid string's log-probability.
2. Compares against $\pi$: masked sampling, rejection sampling, and importance resampling with $w = \prod_t Z_t$ ($N \in \{4, 16, 64\}$ particles), measuring KL divergence and total-variation distance from 5,000 samples each.
3. Shows how prompt alignment (describing the enum in the prompt) changes the distortion.
4. Reports compute cost per method and a recommendation of when correction is worth it.

**Step-by-step**
1. Implement exact enumeration: for each valid string, tokenise it canonically and sum the token log-probs under the prompt.
2. Implement masked sampling with a trie over the valid strings (a regex alternation compiled to an index works too).
3. Record $Z_t$ at each step and compute the weights.
4. Implement importance resampling: draw $N$ masked samples, then resample one in proportion to $w$.
5. Compute the metrics and plot the distributions side by side.
6. Write the analysis, including the effect of non-canonical tokenisations admitted by the mask.

---

## 12. Foundational Papers (exact titles)

**Core algorithms**
- Willard & Louf, 2023 — *Efficient Guided Generation for Large Language Models*
- Dong et al., 2024 — *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models*
- Koo, Liu & He, 2024 — *Automata-based constraints for language model decoding*
- Ugare et al., 2024 — *SynCode: LLM Generation with Grammar Augmentation*
- Geng et al., 2023 — *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*
- Beurer-Kellner, Fischer & Vechev, 2024 — *Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation*
- Zheng et al., 2024 — *SGLang: Efficient Execution of Structured Language Model Programs* (compressed FSM / jump-forward)

**Distribution faithfulness**
- Park et al., 2024 — *Grammar-Aligned Decoding*
- Loula et al., 2025 — *Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo*
- Tam et al., 2024 — *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models*

**Semantic constraints**
- Scholak, Schucher & Bahdanau, 2021 — *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models*
- Poesia et al., 2022 — *Synchromesh: Reliable code generation from pre-trained language models*
- Agrawal et al., 2023 — *Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context*

**Benchmarks**
- Geng et al., 2025 — *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models*

**Theory background**
- Hopcroft, Motwani & Ullman — *Introduction to Automata Theory, Languages, and Computation* (textbook)
- Earley, 1970 — *An Efficient Context-Free Parsing Algorithm*

## 13. Essential Tooling

| Tool | Role |
|---|---|
| **XGrammar** | High-performance CFG/JSON constrained decoding; default backend in several engines |
| **llguidance** / **Guidance** | Earley-based grammars with lazy trie masks; token healing |
| **Outlines** | Regex/JSON/CFG structured generation; readable reference implementation |
| **vLLM / SGLang structured outputs** | Production integration (batching, overlap, spec-decode interplay) |
| **llama.cpp GBNF** | Grammar-constrained local inference |
| **lm-format-enforcer**, **SynCode** | Alternative engines to study and compare |
| **interegular**, **regex** (PyPI), **Lark** | Regex→FSM, partial matching, grammar authoring and parsing |
| **Hypothesis** | Property tests against a naive oracle |
