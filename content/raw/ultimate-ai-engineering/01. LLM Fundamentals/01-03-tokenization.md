# 01.03 — Tokenization

> **Module 1: LLM Fundamentals** · Subtopic 3 of 4
> **Prerequisites:** Unicode and UTF-8, regular expressions, basic probability (maximum likelihood, EM), 01.01 (embeddings, LM head).
> **Outcome:** you can train, audit, and extend tokenizers. You can predict how tokenization affects cost, context usage, multilingual fairness, and arithmetic. You can also prevent the production bugs tokenizers cause: streaming corruption, special-token injection, and template drift.

---

## 1. Why Tokenization Is an Engineering Problem, Not a Preprocessing Detail

The tokenizer decides:
- **Cost and latency.** You pay per token, and prefill time grows with token count.
- **Effective context.** A 128k-token window holds very different amounts of text depending on the language and domain.
- **What the model can "see".** Digit grouping affects arithmetic. Leading spaces create distinct tokens (`"Hello"` vs `" Hello"`). Rare strings become long, fragmented sequences.
- **Security surface.** Special tokens that appear in user text can hijack chat structure.
- **Reproducibility.** A mismatched tokenizer or chat template silently degrades a model that is otherwise correct.

### 1.1 The pipeline

```
raw text
   │
   ▼  1. NORMALIZATION     NFC/NFKC, lowercasing (rare in LLMs), control-char handling
   ▼  2. PRE-TOKENIZATION  regex split into "words"/chunks (merges never cross these boundaries)
   ▼  3. MODEL             BPE | Unigram | WordPiece  →  subword units
   ▼  4. POST-PROCESSING   add BOS/EOS, apply chat template, special tokens
   ▼
token ids  ──►  model  ──►  token ids  ──►  DECODE (bytes → UTF-8, strip markers)  ──► text
```
![LLM-05](/01.%20LLM%20Fundamentals/images/LLM-05.jpg)

---

## 2. Byte-Pair Encoding (BPE)

### 2.1 Training

Starting from a base alphabet, repeatedly merge the **most frequent adjacent pair** into a new symbol until the vocabulary reaches the target size $V$. The output is an ordered **merge list**, and that order *is* the tokenizer.

**Byte-level BPE** (GPT-2 onward) uses the 256 byte values as the base alphabet, so *every* string is encodable. There is no `<unk>` token and no out-of-vocabulary failure, at the cost of possibly splitting multi-byte characters across tokens.

**Pre-tokenization regex** keeps merges from crossing semantic boundaries (letters/digits/punctuation/whitespace). The cl100k-style pattern (requires the `regex` module for `\p{…}` and possessive quantifiers):

```python
GPT4_SPLIT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

Note `\p{N}{1,3}`: numbers are chunked into groups of at most 3 digits, left to right. This choice measurably affects arithmetic (§6.4).

### 2.2 Reference implementation

```python
import regex as re
from collections import Counter

GPT4_SPLIT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def train_bpe(text: str, vocab_size: int, pattern: str = GPT4_SPLIT):
    assert vocab_size >= 256
    # Count unique chunks once: training cost scales with distinct words, not corpus size.
    words = Counter(tuple(c.encode("utf-8")) for c in re.findall(pattern, text))
    merges: dict[tuple[int, int], int] = {}
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        pairs = Counter()
        for w, f in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges[best] = new_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        words = Counter({merge(w, best, new_id): f for w, f in words.items()})
    return merges, vocab


def encode(text: str, merges, pattern: str = GPT4_SPLIT) -> list[int]:
    out = []
    for chunk in re.findall(pattern, text):
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # apply the EARLIEST-learned merge available (lowest new id), exactly as in training
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            ids = list(merge(ids, pair, merges[pair]))
        out.extend(ids)
    return out


def decode(ids: list[int], vocab) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

**Complexity.** Naive training is $O(V \cdot W)$ over unique words $W$. Production trainers (HF `tokenizers`, SentencePiece) maintain **incremental pair counts** with an index from pairs to the words that contain them, plus a heap, so each merge touches only the affected words. Encoding in `tiktoken` uses ranks (merge priority) and a Rust implementation. Cache encoded chunks: real text repeats words heavily.

---

## 3. Unigram Language Model Tokenization

Unigram (Kudo, 2018; the default in SentencePiece) works in the opposite direction from BPE. It starts with a **large** seed vocabulary and **prunes** it. It assumes each segmentation $\mathbf{x} = (x_1,\dots,x_M)$ of text $X$ is a product of independent piece probabilities:

$$
P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i),
\qquad
\mathbf{x}^* = \arg\max_{\mathbf{x}\in\mathcal{S}(X)} P(\mathbf{x}) \quad (\text{Viterbi})
$$

Training maximizes the marginal likelihood over all segmentations $\mathcal{S}(X_s)$ of each sentence:

$$
\mathcal{L} = \sum_{s=1}^{|D|} \log \sum_{\mathbf{x}\in\mathcal{S}(X_s)} P(\mathbf{x})
$$

This is optimized with **EM** (forward-backward over the segmentation lattice). After each EM round, compute for every piece how much $\mathcal{L}$ drops if it is removed. Keep the top η% (e.g. 80%) and repeat until reaching $V$. Single characters are always kept, which guarantees coverage.

**Subword regularization:** at training time, *sample* segmentations $\mathbf{x}\sim P(\mathbf{x}\mid X)^{\alpha}$ instead of taking the Viterbi path. This makes the model robust to segmentation noise. **BPE-dropout** does the same thing for BPE by randomly skipping merges.

## 4. WordPiece

WordPiece (BERT) is BPE-like, but merges the pair that most increases training-data likelihood:

$$
\mathrm{score}(a,b) = \frac{\mathrm{freq}(ab)}{\mathrm{freq}(a)\cdot \mathrm{freq}(b)}
$$

Inference uses **greedy longest-match-first**, with `##` marking word-internal pieces. It is mostly seen in encoder models (embeddings, rerankers). Know it because retrieval stacks still use it.

## 5. SentencePiece

SentencePiece is a *library and framework*, not an algorithm (it implements BPE and Unigram). It treats input as a raw Unicode stream with **no language-specific pre-tokenization**, and encodes whitespace as `▁` (U+2581), which makes it lossless and reversible. `byte_fallback=True` maps unknown characters to `<0xNN>` byte tokens, giving the same coverage guarantee as byte-level BPE. Llama 1/2, T5, Gemma, and many multilingual models use it.

---

## 6. Design Trade-offs: Vocabulary Size, Compression, Fairness

### 6.1 Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Compression** | $\dfrac{\text{UTF-8 bytes}}{\text{tokens}}$ | Higher is better for cost and context |
| **Fertility** | $\dfrac{\text{tokens}}{\text{words}}$ | Subwords per word; ≈1.2–1.5 for English on modern tokenizers |
| **Premium** (vs English) | $\dfrac{\text{tokens}(X_{\text{lang}})}{\text{tokens}(X_{\text{en}})}$ on **parallel** text | Cost and context penalty for a language |
| **Bits-per-byte (BPB)** | $\dfrac{\sum_t -\log p(t)}{\ln 2 \cdot n_{\text{bytes}}}$ | **The only fair way to compare LMs with different tokenizers** |

> ⚠️ Per-token perplexity is **not comparable** across tokenizers. A tokenizer with longer tokens has fewer, harder predictions, so its per-token loss is higher even when the model is better. Always normalize by bytes or characters.

### 6.2 Vocabulary size

The embedding and LM-head parameters total $V \cdot d$ each (untied). With $V = 128\text{k}$ and $d=4096$, that is ≈ 524M parameters per matrix. Larger $V$ means:

- Better compression: shorter sequences, cheaper attention, more text per context window.
- A larger softmax per decode step, and an LM head that becomes a meaningful share of decode time for small models.
- **More rare, under-trained tokens**, which are a source of glitch behaviour (§8).

Larger models tolerate larger vocabularies. Recent frontier open models use roughly 100k–260k tokens.

### 6.3 Multilingual fairness

Tokenizers trained mostly on English and code produce large premiums for other scripts and low-resource languages. The same content can cost several times more tokens in, for example, Burmese, Amharic, or many African languages. This multiplies API cost, consumes context, raises latency, and *also* hurts model quality, because each token carries less meaning. Mitigations include training on a balanced corpus with language-wise sampling temperature, a larger vocabulary, and vocabulary extension for target languages (Project 3).

### 6.4 Numbers and arithmetic

- Single-digit tokenization (Llama 1/2, via `split_digits`) gives consistent digit alignment.
- 3-digit left-to-right chunks (cl100k) cause misalignment: `12345` → `123|45`, while `1234` → `123|4`. Place values no longer line up between operands.
- Research shows **right-to-left** grouping of digits (commas are one way to force this) improves arithmetic accuracy in frontier models. For numeric workloads, format numbers deliberately in prompts, or offload the math to tools.

---

## 7. Chat Templates & Special Tokens

Chat models are trained on a specific serialization, for example:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n
```

**Rules:**
1. **Never hand-build templates.** Use `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` with the template shipped with the checkpoint (it is Jinja stored in `tokenizer_config.json` or `chat_template.jinja`).
2. **Double BOS** is a classic bug: the template adds BOS, and then `tokenizer(..., add_special_tokens=True)` adds another. Encode template output with `add_special_tokens=False`.
3. **Stop tokens:** configure all end-of-turn IDs (e.g. `<|eot_id|>` *and* `<|end_of_text|>`), not only `eos_token_id`. Otherwise the model keeps generating past the end of its turn.
4. **Tool-call formats** are template-specific. Parse with the model family's parser in your serving engine.

### 7.1 Special-token injection (security)

If user-supplied text containing the literal string `<|eot_id|><|start_header_id|>system<|end_header_id|>` is tokenized *as special tokens*, the user can forge a system turn.

```python
# tiktoken: raises on special tokens in user text by default (disallowed_special="all")
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
safe_ids = enc.encode(user_text, disallowed_special=())   # treat them as ordinary text

# Hugging Face: split special-token strings into ordinary pieces for untrusted input
ids = tokenizer(user_text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
```

Build the chat structure from **token IDs you control**, and encode untrusted segments with special-token parsing disabled. Add a regression test that feeds template control strings through every untrusted input path.

---

## 8. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Streaming mojibake** | `�` characters mid-stream, broken emoji or CJK | Incremental UTF-8 decoding (code below). For SentencePiece tokenizers, decode with a prefix/read-offset window, because `▁` handling makes per-token decoding context-dependent |
| **Token counting drift** | Context overflow errors, wrong billing | Count with the *exact* model tokenizer **after** applying the chat template; add a margin for template and tool overhead |
| **Leading-space tokens** | Few-shot or completion prompts behave oddly; `"Answer:"` + `" Yes"` vs `"Yes"` | Keep spacing consistent with training; do not strip trailing whitespace inconsistently |
| **Token healing** | Prompt ends mid-token (`"http:"`), so the model cannot produce the natural merged token (`"://"`) | Back up the last token and constrain the first generated token to start with the removed bytes (as in Guidance) |
| **Glitch / under-trained tokens** | Bizarre outputs on specific strings | Detect under-trained tokens (small embedding norm, low training frequency) and filter or avoid them in inputs |
| **Tokenizer/model version mismatch** | Silent quality loss after an upgrade | Pin the tokenizer with the model revision; checksum `tokenizer.json`; test golden encodings in CI |
| **Throughput bottleneck** | CPU-bound tokenization at high QPS | Rust tokenizers (`tokenizers`, `tiktoken`), batch encoding, a separate tokenizer worker pool, caching of repeated prefixes |
| **Normalization surprises** | Identical-looking strings produce different tokens | Canonicalize to NFC at ingestion; beware of full-width and compatibility characters |

**Incremental UTF-8-safe detokenizer (byte-level BPE):**

```python
import codecs


class StreamingDetokenizer:
    """Emits only complete UTF-8 characters; buffers partial multi-byte sequences."""

    def __init__(self, id_to_bytes: dict[int, bytes]):
        self.id_to_bytes = id_to_bytes
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        return self.decoder.decode(self.id_to_bytes[token_id], final=False)

    def flush(self) -> str:
        return self.decoder.decode(b"", final=True)


# With tiktoken: id_to_bytes = {i: enc.decode_single_token_bytes(i) for i in range(enc.n_vocab)}
```

---

## 9. Beyond Subwords: Tokenizer-Free Models

- **ByT5:** operates on raw bytes. Robust to noise and spelling, but sequences are about 4× longer.
- **MEGABYTE:** a multiscale architecture with fixed-size byte patches, a global model over patches, and a local model within each patch.
- **Byte Latent Transformer (BLT):** dynamic, **entropy-based patching**. It spends compute where the next byte is hard to predict, and matches subword models at scale with better robustness.

These remove the tokenizer failure modes above but demand new kernels and serving assumptions. It is an active research frontier worth tracking.

---

## 10. Hands-On Projects

### Project 1 — Byte-Level BPE from Scratch, Bit-Compatible with `tiktoken`

**User stories**
- *As an AI engineer*, I want to implement BPE training and encoding myself, so that I can reason precisely about any tokenizer bug in production.
- *As a platform engineer*, I want my encoder to reproduce `cl100k_base` exactly, so that I can trust it for offline token accounting.

**Acceptance criteria**
1. `train_bpe` on a ≥ 50 MB corpus produces a vocabulary of a configurable size (e.g. 8k, 32k). Training time is logged, and an incremental-count version is ≥ 10× faster than the naive one.
2. **Round-trip property test** (Hypothesis): `decode(encode(s)) == s` for 10k random Unicode strings, including emoji, CJK, RTL scripts, combining marks, and control characters.
3. Loading `cl100k_base` mergeable ranks into your encoder produces **identical ids** to `tiktoken` on a 1 MB mixed corpus (code, prose, multilingual text).
4. Special-token handling: encoding untrusted text never yields special ids unless explicitly allowed (a test covers this).
5. A benchmark reports MB/s against `tiktoken`, with a profile explaining the gap.

**Step-by-step**
1. Implement §2.2 and add tests.
2. Optimize training: maintain `pair → count` and `pair → set(word_ids)`. After each merge, update only the affected words and adjust counts. Use a lazy-deletion max-heap.
3. For tiktoken compatibility: `ranks = enc._mergeable_ranks` (bytes → rank). Encode each regex chunk by repeatedly merging the adjacent pair whose **concatenated bytes** has the lowest rank. (Operate on byte strings, not the merge-id pairs from your own trainer.)
4. Diff against `enc.encode(text, disallowed_special=())` over the corpus and print the first mismatch with its context.
5. Add an LRU cache on chunk → ids, and optionally port the hot loop to Rust via PyO3.

---

### Project 2 — Multilingual Tokenizer Cost & Fairness Audit

**User stories**
- *As a product lead* launching in multiple regions, I need to know how much more each language costs per equivalent content, and how much context it consumes, so that I can price fairly and choose a model.

**Acceptance criteria**
1. Uses a **parallel corpus** (e.g. FLORES-200 dev/devtest) for ≥ 20 languages across ≥ 5 scripts, including ≥ 5 low-resource languages (e.g. Swahili, Yoruba, Luganda, Amharic, Khmer).
2. Evaluates ≥ 5 tokenizers (e.g. cl100k/o200k via `tiktoken`, plus Llama, Qwen, Gemma, and Mistral families via HF).
3. Computes compression (bytes/token), fertility, and **premium vs English** per (language, tokenizer), with bootstrap 95% confidence intervals.
4. Produces a heatmap and a "cost per 1M characters" table using configurable price inputs.
5. A written finding identifies the best tokenizer per language group and quantifies the effective-context loss (the English-equivalent characters that fit in a 32k window).

**Step-by-step**
1. Load FLORES-200 through `datasets` and align sentences by id.
2. Build a common `Tokenizer` interface wrapping `tiktoken` and `AutoTokenizer` (with `add_special_tokens=False`).
3. Compute the metrics per sentence, then aggregate with bootstrap resampling.
4. Visualize with seaborn or Plotly: languages × tokenizers, colored by premium.
5. Write a two-page report with recommendations (model choice, per-language pricing, whether to extend the vocabulary).

---

### Project 3 — Domain Vocabulary Extension with Honest Evaluation

**User stories**
- *As an ML engineer* adapting an open model to a specialised corpus (e.g. legal or legislative text, clinical notes, a codebase), I want to add domain tokens, so that sequences get shorter and domain modelling improves without harming general ability.

**Acceptance criteria**
1. Trains a domain BPE on the target corpus and selects the top-N (e.g. 2k–8k) new tokens that are **not** in the base vocabulary and that yield the largest compression gain.
2. Extends the base tokenizer, resizes embeddings and the LM head (`resize_token_embeddings`), and initializes each new embedding as the **mean of its old sub-token embeddings** (compare against random initialization).
3. Continued pretraining (optionally LoRA plus trainable embeddings) on the domain corpus.
4. Reports domain tokens-per-character reduced by ≥ 10%.
5. Reports **bits-per-byte** (not per-token perplexity) on held-out domain and general text, before and after. Domain BPB improves and general BPB regresses by ≤ 2%.
6. Throughput gain is measured end-to-end (tokens/s and characters/s at generation).

**Step-by-step**
1. Collect and clean the domain corpus; hold out 5% for evaluation, plus a general-text evaluation set.
2. Train a domain tokenizer with HF `tokenizers` and diff its vocabulary against the base; score candidates by frequency × (base tokens − 1).
3. Add tokens with `tokenizer.add_tokens` (as non-special tokens), then for each new token compute `old_ids = base_tok.encode(token_str)` and set `E[new] = E[old_ids].mean(0)` (same for the untied LM head).
4. Train with a higher learning rate on embeddings than on the body (or freeze the body first, then unfreeze).
5. Evaluate BPB (formula in §6.1), and run a small general-capability eval to catch regressions.

---

## 11. Foundational Papers (exact titles)

**Subword algorithms**
- Sennrich, Haddow & Birch, 2016 — *Neural Machine Translation of Rare Words with Subword Units*
- Schuster & Nakajima, 2012 — *Japanese and Korean Voice Search* (WordPiece origin)
- Wu et al., 2016 — *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*
- Kudo, 2018 — *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*
- Kudo & Richardson, 2018 — *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*
- Provilkov, Emelianenko & Voita, 2020 — *BPE-Dropout: Simple and Effective Subword Regularization*
- Radford et al., 2019 — *Language Models are Unsupervised Multitask Learners* (byte-level BPE)

**Analysis and behaviour**
- Petrov et al., 2023 — *Language Model Tokenizers Introduce Unfairness Between Languages*
- Rust et al., 2021 — *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models*
- Schmidt et al., 2024 — *Tokenization Is More Than Compression*
- Singh & Strouse, 2024 — *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs*
- Land & Bartolo, 2024 — *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models*
- Dagan, Synnaeve & Rozière, 2024 — *Getting the most out of your tokenizer for pre-training and domain adaptation*

**Tokenizer-free**
- Xue et al., 2022 — *ByT5: Towards a token-free future with pre-trained byte-to-byte models*
- Yu et al., 2023 — *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers*
- Pagnoni et al., 2024 — *Byte Latent Transformer: Patches Scale Better Than Tokens*

## 12. Essential Tooling

| Tool | Role |
|---|---|
| **Hugging Face `tokenizers`** | Fast Rust training and encoding for BPE, Unigram, and WordPiece; `tokenizer.json` format |
| **`transformers` `AutoTokenizer` + `apply_chat_template`** | Model-exact encoding and chat serialization |
| **`tiktoken`** | OpenAI encodings; fastest BPE encoder for token accounting |
| **`sentencepiece`** | Training and using SentencePiece models |
| **`minbpe`** (Karpathy) | A readable reference to study |
| **`regex`** (PyPI) | Unicode property classes for pre-tokenizer patterns |
| **Hypothesis** | Property-based round-trip testing |
| **FLORES-200** | A parallel corpus for multilingual audits |
