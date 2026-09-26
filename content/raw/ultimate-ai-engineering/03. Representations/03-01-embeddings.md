# 03.01 — Embeddings

> **Module 3: Representations** · Subtopic 1 of 4
> **Prerequisites:** 01.01 (transformer encoders/decoders, attention masks), linear algebra (inner products, norms), softmax and cross-entropy, basic IR vocabulary (query, document, relevance).
> **Outcome:** you can explain how modern text embedding models are built and trained, choose one on *your* data rather than a leaderboard, fine-tune it with hard negatives, compress it (Matryoshka truncation, int8, binary) with measured recall loss, and operate embeddings in production: versioning, migrations, drift, and privacy.

---

## 1. What an Embedding Model Computes

An embedding model maps text to a fixed-size vector $f_\theta: \Sigma^* \to \mathbb{R}^d$ so that **geometric proximity approximates semantic relevance** for a target task. In retrieval, the score is usually

$$
s(q, d) = \langle f_\theta(q),\, f_\theta(d) \rangle \quad\text{or}\quad \cos(f_\theta(q), f_\theta(d))
$$

### 1.1 Bi-encoder architecture

```
   query text                           document text
       │                                      │
 ┌─────▼──────┐   shared (or twin) weights ┌──▼─────────┐
 │  Encoder   │◄──────────────────────────►│  Encoder   │   BERT-style (bidirectional) or
 │ (L layers) │                            │ (L layers) │   LLM-based (causal / bi-directionalised)
 └─────┬──────┘                            └──┬─────────┘
   token states H ∈ ℝ^{T×h}              token states
       │ POOL: mean | CLS | last-token        │
       │ (optional) projection → d dims       │
       │ L2-normalise                         │
       ▼                                      ▼
      q ∈ ℝ^d  ───────── ⟨q, d⟩ ──────────►  d ∈ ℝ^d     (documents embedded OFFLINE, indexed)
```

**Why bi-encoders dominate first-stage retrieval:** documents are encoded **once, offline**. A query costs one encoder pass plus a nearest-neighbour search (03.02). **Cross-encoders** (query and document jointly through one transformer) are far more accurate but cost one forward pass *per candidate*, so they are used for reranking (03.04).

### 1.2 Pooling strategies

| Pooling | Formula | Used by |
|---|---|---|
| **Mean** | $\mathbf{e} = \frac{\sum_t m_t \mathbf{h}_t}{\sum_t m_t}$ ($m$ = attention mask) | Sentence-BERT, E5, GTE, most BERT-based embedders |
| **CLS** | $\mathbf{e} = \mathbf{h}_{\texttt{[CLS]}}$ | BGE, some BERT-derived models |
| **Last-token (EOS)** | $\mathbf{e} = \mathbf{h}_{t_{\text{last}}}$ | Decoder-LLM embedders (causal attention: only the last token has seen everything) |
| **Latent attention** | Learned attention pooling over $\mathbf{H}$ | NV-Embed |

> ⚠️ **Pooling, normalisation, and prefixes are part of the model.** Using mean pooling on a CLS-trained model, or forgetting the `"query: "` / `"passage: "` prefixes (E5) or the task instruction on the query side (instruction-tuned models), silently loses several points of recall. Read the model card, and write a unit test that reproduces its reference embeddings.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def last_token_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Works for right padding: index of the last non-pad token per row."""
    idx = mask.sum(1) - 1
    return last_hidden[torch.arange(last_hidden.size(0)), idx]


@torch.inference_mode()
def embed(texts: list[str], model, tokenizer, pooling="mean", max_len=512, batch_size=64):
    # Sort by length to minimise padding; restore the original order at the end.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(model.device)
        h = model(**enc).last_hidden_state
        e = mean_pool(h, enc["attention_mask"]) if pooling == "mean" else last_token_pool(h, enc["attention_mask"])
        e = F.normalize(e.float(), dim=-1)
        for j, i in enumerate(idx):
            out[i] = e[j].cpu()
    return torch.stack(out)
```

### 1.3 Geometry you must know

For unit vectors, cosine similarity, inner product, and Euclidean distance induce **the same ranking**:

$$
\|\mathbf{a}-\mathbf{b}\|_2^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\langle \mathbf{a},\mathbf{b}\rangle = 2 - 2\cos(\mathbf{a},\mathbf{b})
$$

So **normalise once, at write time**, and use the cheapest metric your index supports (inner product). Two further effects matter in practice:

- **Anisotropy:** raw transformer representations occupy a narrow cone. All cosines are high (e.g. 0.6–0.9), so absolute thresholds are meaningless across models. Contrastive training spreads them out, but **never hard-code a similarity threshold without calibrating it per model**.
- **Hubness:** in high dimensions, some points become nearest neighbours of disproportionately many queries. The same irrelevant chunk then appears in many results. Detect it by counting how often each document appears in top-k across a query sample, and mitigate with reranking and diversity.

---

## 2. Training Embedding Models: Contrastive Learning

### 2.1 InfoNCE with in-batch negatives

Given a batch of $B$ positive pairs $(q_i, d_i^+)$, treat every other document in the batch as a negative:

$$
\mathcal{L}_{q\to d} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp\!\big(s(q_i, d_i^+)/\tau\big)}{\sum_{j=1}^{B} \exp\!\big(s(q_i, d_j^+)/\tau\big) + \sum_{k} \exp\!\big(s(q_i, d_{i,k}^-)/\tau\big)}
$$

- $\tau$ is the **temperature** (typically 0.01–0.05 with cosine scores). A lower $\tau$ sharpens the distribution and emphasises hard negatives.
- $d_{i,k}^-$ are **mined hard negatives** (§2.2).
- A symmetric loss $\tfrac12(\mathcal{L}_{q\to d} + \mathcal{L}_{d\to q})$ often helps for symmetric tasks.
- **Batch size matters:** more negatives give a tighter bound on mutual information. GradCache decouples the contrastive batch size from GPU memory by caching representations and back-propagating in chunks.

```python
import torch
import torch.nn.functional as F


def info_nce(q: torch.Tensor, d_pos: torch.Tensor, d_neg: torch.Tensor | None = None,
             tau: float = 0.02, symmetric: bool = False) -> torch.Tensor:
    """q, d_pos: (B, D) L2-normalised. d_neg: (B, K, D) hard negatives (optional)."""
    B = q.size(0)
    logits = q @ d_pos.T / tau                                  # (B, B): in-batch negatives
    if d_neg is not None:
        hard = torch.einsum("bd,bkd->bk", q, d_neg) / tau       # (B, K): each query's own negatives
        logits = torch.cat([logits, hard], dim=1)
    labels = torch.arange(B, device=q.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(d_pos @ q.T / tau, labels))
    return loss
```

### 2.2 Hard negatives — and the false-negative trap

Random in-batch negatives are easy. The model learns much more from **hard negatives**: documents that score high but are not relevant (mined with BM25 or with the current model, as in ANCE).

**The trap:** in real corpora, many mined "negatives" are actually **unlabelled positives** (false negatives). Training pushes them away and teaches the model to ignore true matches. Mitigations:

- **Margin filtering:** discard negative candidates whose score exceeds a fraction of the positive's score, e.g. $s(q, d^-) > 0.95\, s(q, d^+)$.
- **Rank window:** sample negatives from ranks 10–100, not 1–5.
- **Cross-encoder denoising:** label the candidate negatives with a strong reranker, and drop those it scores as relevant.
- **Deduplicate** near-identical documents before mining.

```python
import numpy as np


def mine_hard_negatives(q_emb, doc_emb, positives: list[set[int]], k=5,
                        rank_window=(10, 100), max_ratio=0.95, seed=0):
    """Return k negative doc ids per query, skipping likely false negatives."""
    rng = np.random.default_rng(seed)
    scores = q_emb @ doc_emb.T
    out = []
    for i, pos in enumerate(positives):
        order = np.argsort(-scores[i])
        pos_score = max(scores[i, p] for p in pos)
        window = [j for j in order[rank_window[0]:rank_window[1]]
                  if j not in pos and scores[i, j] < max_ratio * pos_score]
        out.append(rng.choice(window, size=min(k, len(window)), replace=False).tolist())
    return out
```

### 2.3 The modern training recipe

```
Stage 1: weakly-supervised contrastive pre-training
         hundreds of millions of (title, body), (question, answer), (query, click) pairs; huge batches
Stage 2: supervised fine-tuning
         curated labelled pairs + mined hard negatives (MS MARCO, NQ, NLI, domain data)
Stage 3 (optional): task instructions  "Instruct: retrieve statutes relevant to the question\nQuery: …"
Stage 4 (optional): distillation from a cross-encoder (soft labels) and Matryoshka / quantization-aware losses
```

**LLM-based embedders** (E5-Mistral, NV-Embed, LLM2Vec, and many current MTEB leaders) start from a decoder LLM. They use last-token or latent-attention pooling, sometimes switch on **bidirectional attention** for embedding, and train with instructions plus synthetic data generated by LLMs. They are stronger on hard, instruction-dependent tasks, but 10–100× more expensive per token than a ~100–500M-parameter BERT-class encoder. That cost matters when re-embedding a large corpus.

### 2.4 Synthetic training data for your domain

When you have documents but no queries, generate queries per chunk with an LLM ("Write 3 questions a legislative attorney would ask that this passage answers"). Filter them with **round-trip consistency**: keep a pair only if the source chunk ranks in the top-k for its generated query under a baseline retriever, or a cross-encoder judges it relevant. This is how you fine-tune for a domain in days rather than months (GPL, InPars, Promptagator).

---

## 3. Matryoshka Representations and Compression

### 3.1 Matryoshka Representation Learning (MRL)

MRL trains the model so that the **prefixes** of the embedding are themselves good embeddings. The loss is summed over nested dimensions $\mathcal{M} = \{64, 128, 256, 512, 1024, \ldots\}$:

$$
\mathcal{L}_{\text{MRL}} = \sum_{m \in \mathcal{M}} c_m\; \mathcal{L}\big(\mathbf{q}_{1:m}/\|\mathbf{q}_{1:m}\|,\ \mathbf{d}_{1:m}/\|\mathbf{d}_{1:m}\|\big)
$$

At serving time you can truncate to $m$ dimensions (then **re-normalise**) and trade recall for memory and speed. A common pattern is to search with short vectors and **rescore** the top candidates with full vectors.

```python
def matryoshka_info_nce(q, d_pos, dims=(64, 128, 256, 512, 768), weights=None, tau=0.02):
    weights = weights or [1.0] * len(dims)
    total = 0.0
    for w, m in zip(weights, dims):
        total = total + w * info_nce(F.normalize(q[:, :m], dim=-1),
                                     F.normalize(d_pos[:, :m], dim=-1), tau=tau)
    return total / sum(weights)
```

> Truncating a model that was **not** trained with MRL degrades recall much faster. Check the model card before truncating.

### 3.2 Scalar and binary quantization

Memory for $N$ vectors of dimension $d$:

$$
\text{bytes} = N \cdot d \cdot b, \qquad b = 4\ (\text{fp32}),\ 2\ (\text{fp16}),\ 1\ (\text{int8}),\ \tfrac18\ (\text{binary})
$$

Example: 100M chunks × 1024 dims in fp32 is **410 GB**. The same collection is 102 GB in int8, and **12.8 GB in binary**.

- **int8 (scalar) quantization:** per-dimension min/max (or percentile) calibration from a sample, mapping each value to [−128, 127]. Recall is typically near-lossless.
- **Binary quantization:** $b_i = \mathbb{1}[x_i > 0]$, compared with **Hamming distance** (XOR + popcount: extremely fast). Recall drops noticeably on its own, but recovers most of the way with **rescoring**: retrieve $k \cdot r$ candidates by Hamming distance, then rescore them with int8 or float vectors.

```python
import numpy as np


def int8_calibrate(sample: np.ndarray, lo_pct=0.5, hi_pct=99.5):
    lo = np.percentile(sample, lo_pct, axis=0)
    hi = np.percentile(sample, hi_pct, axis=0)
    return lo, hi


def int8_quantize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    scale = (hi - lo) / 255.0
    return np.clip(np.round((x - lo) / np.where(scale == 0, 1, scale)) - 128, -128, 127).astype(np.int8)


def binarize(x: np.ndarray) -> np.ndarray:
    return np.packbits(x > 0, axis=-1)                          # (N, d/8) uint8


def hamming_topk(q_bits: np.ndarray, db_bits: np.ndarray, k: int) -> np.ndarray:
    dist = np.unpackbits(np.bitwise_xor(db_bits, q_bits[None, :]), axis=-1).sum(-1)
    return np.argpartition(dist, min(k, len(dist) - 1))[:k]


def binary_search_with_rescore(q: np.ndarray, db_float: np.ndarray, db_bits: np.ndarray,
                               k=10, oversample=10) -> np.ndarray:
    cand = hamming_topk(binarize(q[None])[0], db_bits, k * oversample)
    exact = db_float[cand] @ q
    return cand[np.argsort(-exact)[:k]]
```

---

## 4. Evaluating Embeddings — On Your Data

Leaderboards such as **MTEB/MMTEB** and **BEIR** are useful for shortlisting models, but they are not decisive. Domain shift (legal text, code, clinical notes, a non-English language) frequently reorders them. Build a domain eval set of 200–1,000 queries with graded relevance labels (**qrels**).

**Metrics** (for a ranked list for query $q$, and relevance $\text{rel}_i \in \{0,1,2,3\}$):

$$
\text{Recall@}k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|},\qquad
\text{MRR} = \frac{1}{|Q|}\sum_{q} \frac{1}{\text{rank of first relevant}}
$$

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i}-1}{\log_2(i+1)},\qquad
\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

Use **Recall@50–100** for a first-stage retriever that feeds a reranker. Use **nDCG@10** for the final ranking the user or LLM sees.

```python
import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    return next((1.0 / (i + 1) for i, d in enumerate(ranked) if d in relevant), 0.0)


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    dcg = sum((2 ** qrels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
```

**Also measure:** encode throughput (documents/s on your hardware), query latency, vector memory, maximum sequence length versus your chunk lengths, and multilingual coverage if relevant.

---

## 5. Long Documents and Chunking (the Embedding View)

An embedding compresses a whole passage into one point. Long, multi-topic passages turn into a blurry average.

- **Chunk to the unit of answer:** a section, clause, or paragraph group of 200–500 tokens, with small overlaps. Respect structure (headings, statute sections) rather than fixed character counts.
- **Never exceed the model's max length.** Tokenizers truncate *silently*. Log the token counts at indexing time and alert when truncation happens.
- **Contextual chunk headers** (02.02 §3.2): prepend the document title and section path to each chunk before embedding.
- **Late chunking:** run a long-context embedder over the whole document, *then* mean-pool token states per chunk span. Each chunk's vector then carries document context.
- **Multi-vector per document** (per chunk, plus a summary vector) and **late interaction** (ColBERT, 03.04) are the alternatives to single-vector compression.

---

## 6. Operating Embeddings in Production

### 6.1 Versioning and migrations

Vectors from different models, or even different preprocessing, live in **incompatible spaces**. Mixing them in one index produces garbage rankings with no errors.

```
            ┌───────────── write path ─────────────┐
 source ─► CDC/events ─► embed(model=v1) ─► index_v1   (serving)
                    └──► embed(model=v2) ─► index_v2   (dual-write + backfill)

 cutover: shadow queries on v2 → offline + online eval → flip alias → keep v1 for rollback → delete
```

- Store `embedding_model_id` and a `preprocess_version` with every vector.
- Budget for re-embedding: $\text{time} = N_{\text{tokens}} / \text{throughput}$. For example, 50M chunks × 400 tokens = 20B tokens, which is days on a single GPU for an LLM-based embedder. Hosted APIs charge per token.
- Keep queries and documents on the **same** model version. The most common migration bug is updating the query encoder before the backfill has finished.

### 6.2 Drift and monitoring

- **Query drift:** track the distribution of top-1 similarity scores and of cluster assignments over time. A shift suggests new topics or vocabulary your model handles poorly.
- **Retrieval quality canaries:** a fixed set of queries with known answers, run on a schedule, alerting on recall drops.
- **Embedding throughput and truncation-rate dashboards.**

### 6.3 Privacy

Embeddings are **not anonymisation**. Text can be substantially reconstructed from embeddings (vec2text; Morris et al., 2023). Treat vector stores holding personal data with the same access controls, encryption, retention rules, and deletion guarantees as the source text.

### 6.4 Throughput engineering

Sort by length before batching, use fp16/bf16, and export to ONNX Runtime or TensorRT for BERT-class models. Serve LLM-based embedders with inference engines that support embedding endpoints. Cache query embeddings for frequent queries.

---

## 7. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Missing prefixes / instructions** | Recall several points below the model card | Wrap the model in a client that enforces the query/document formats; test against reference vectors |
| **Silent truncation** | Long chunks retrieved poorly | Token-count logging; chunk below max length; alerts |
| **Mixed model versions** | Random-looking results after a deploy | `model_id` on every vector; dual index + alias cutover |
| **Hard-coded similarity thresholds** | "No results" or junk after a model swap | Calibrate thresholds per model on labelled data; prefer top-k + reranker |
| **Domain mismatch** | Good MTEB score, poor production recall | Domain eval set; fine-tune with synthetic queries + hard negatives |
| **Memory cost** | Vector RAM dominates the bill | MRL truncation, int8, binary + rescoring; disk-based indexes (03.02) |
| **Exact identifiers missed** | "HB-123" or "2-18-303" queries fail | Hybrid retrieval with lexical search (03.04) |
| **Hubs in results** | The same irrelevant chunk everywhere | Hubness audit; reranking; deduplication; diversity (MMR) |
| **Embedding inversion risk** | PII exposure via the vector store | Access control, encryption, deletion propagation |

---

## 8. Hands-On Projects

### Project 1 — Domain Fine-Tuning of an Embedding Model with Hard Negatives

**User stories**
- *As a search engineer* at a legislative office, I want an embedding model fine-tuned on statutes and bills, so that attorneys' natural-language questions retrieve the right code sections.
- *As a tech lead*, I want the gain proven on a held-out labelled set, so that the extra training pipeline is justified.

**Acceptance criteria**
1. A corpus of ≥ 20k chunks (statutes, bills, fiscal notes, or another specialised domain) chunked by section, with contextual headers.
2. ≥ 20k synthetic training queries generated by an LLM and filtered by round-trip consistency (§2.4). A **human-labelled** test set of ≥ 300 queries is kept separate and never used in training.
3. Fine-tuning with `info_nce` (§2.1) plus ≥ 3 mined hard negatives per query, with false-negative filtering (§2.2). Also trains an MRL variant (§3.1).
4. Reports Recall@10/@100, nDCG@10, and MRR for: the base model, fine-tuned (random negatives), fine-tuned (hard negatives), and fine-tuned + MRL at {128, 256, full} dims. The best model beats the base by ≥ 5 nDCG@10 points, with bootstrap confidence intervals.
5. A general-domain regression check (a BEIR subset) is reported, to catch catastrophic forgetting.

**Step-by-step**
1. Pick a base model (e.g. a ~100–600M-parameter open embedder with a permissive license). Reproduce its model-card embeddings in a unit test.
2. Build the query generator and the filter; store the triples `(query, pos_id, [neg_ids])`.
3. Mine the hard negatives with the base model using `mine_hard_negatives`, then denoise them with a cross-encoder.
4. Train with sentence-transformers (`MultipleNegativesRankingLoss`, `MatryoshkaLoss`) or your own loop using `info_nce`: bf16, large batch (GradCache if needed), lr ≈ 2e-5, 1–3 epochs.
5. Evaluate with FAISS flat search and the metrics in §4.
6. Write up the gains per query type, failure analysis, and whether MRL@256 is good enough to halve memory.

---

### Project 2 — Embedding Model Selection & Compression Pareto Study

**User stories**
- *As a platform architect*, I need to choose an embedding model and storage format for 50M chunks under a fixed memory and latency budget, so that the infrastructure cost is justified by measured recall.

**Acceptance criteria**
1. Evaluates ≥ 5 models (BERT-class and LLM-based, open and hosted) on your domain eval set, plus one public BEIR dataset.
2. For each model, measures full-precision, MRL truncations (where supported), int8, and binary + rescoring (oversample ∈ {4, 10, 20}).
3. Records Recall@10, nDCG@10, bytes per vector, encode throughput (documents/s on a fixed GPU), query latency, and projected cost to embed and store 50M chunks.
4. Produces a Pareto chart (recall vs memory per vector) and a recommendation with a sensitivity analysis.

**Step-by-step**
1. Write a model wrapper enforcing each model's pooling, prefixes, and normalisation (§1.2).
2. Embed the corpus once per model, and cache the vectors to disk (`.npy`, fp16).
3. Implement `int8_quantize`, `binarize`, and `binary_search_with_rescore` (§3.2). Validate them against FAISS flat search.
4. Run the metric grid and compute the costs (API price per token, or GPU-hours × hourly rate).
5. Plot and write the decision memo.

---

### Project 3 — Zero-Downtime Embedding Migration Service (Kafka + Spring Boot)

**User stories**
- *As a backend engineer*, I want to switch the production embedding model without downtime or quality regressions, so that we can adopt better models safely.
- *As an operator*, I want the cutover to be reversible within minutes.

**Acceptance criteria**
1. Document change events flow through Kafka topics (`docs.changed`). A Spring Boot (or Python) embedding worker writes to a versioned index (`chunks_v1`, `chunks_v2`) with `model_id` stored on each vector.
2. Dual-write mode plus a backfill job, with progress metrics (percentage complete, tokens/s, ETA) and idempotent upserts keyed by `(chunk_id, content_hash, model_id)`.
3. Shadow mode: a sample of live queries runs against v2, and the overlap@10 with v1 plus quality on the labelled set are logged.
4. The cutover flips an alias or config flag atomically; rollback is tested; v1 is deleted only after a soak period.
5. A chaos test: kill the worker mid-backfill; after restart there are no duplicates or gaps (verified by counts and content hashes).

**Step-by-step**
1. Model the chunk table with `chunk_id, doc_id, content_hash, text, model_id, embedding, updated_at`.
2. Build the consumer with at-least-once semantics and idempotent upserts. Commit Kafka offsets after the database write.
3. Build the backfill job as a paginated scan by primary key, publishing to a backfill topic with rate limiting.
4. Implement the shadow-query sampler and a metrics exporter (Micrometer/Prometheus).
5. Implement the alias switch (e.g. a view or a config service) and a runbook for cutover and rollback.
6. Run the chaos test and document the results.

---

## 9. Foundational Papers (exact titles)

**Foundations**
- Mikolov et al., 2013 — *Efficient Estimation of Word Representations in Vector Space*
- Reimers & Gurevych, 2019 — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- Karpukhin et al., 2020 — *Dense Passage Retrieval for Open-Domain Question Answering*
- van den Oord, Li & Vinyals, 2018 — *Representation Learning with Contrastive Predictive Coding*
- Gao, Yao & Chen, 2021 — *SimCSE: Simple Contrastive Learning of Sentence Embeddings*
- Ethayarajh, 2019 — *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings*
- Radovanović, Nanopoulos & Ivanović, 2010 — *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data*

**Training recipes**
- Xiong et al., 2020 — *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval*
- Gao et al., 2021 — *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup*
- Wang et al., 2022 — *Text Embeddings by Weakly-Supervised Contrastive Pre-training*
- Li et al., 2023 — *Towards General Text Embeddings with Multi-stage Contrastive Learning*
- Su et al., 2022 — *One Embedder, Any Task: Instruction-Finetuned Text Embeddings*
- Wang et al., 2023 — *Improving Text Embeddings with Large Language Models*
- BehnamGhader et al., 2024 — *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders*
- Lee et al., 2024 — *NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models*
- Wang et al., 2021 — *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval*
- Bonifacio et al., 2022 — *InPars: Data Augmentation for Information Retrieval using Large Language Models*
- Dai et al., 2022 — *Promptagator: Few-shot Dense Retrieval From 8 Examples*

**Compression and long documents**
- Kusupati et al., 2022 — *Matryoshka Representation Learning*
- Günther et al., 2024 — *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*

**Evaluation and privacy**
- Thakur et al., 2021 — *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*
- Muennighoff et al., 2022 — *MTEB: Massive Text Embedding Benchmark*
- Enevoldsen et al., 2025 — *MMTEB: Massive Multilingual Text Embedding Benchmark*
- Morris et al., 2023 — *Text Embeddings Reveal (Almost) As Much As Text*

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **sentence-transformers** | Training (MNRL, Matryoshka losses), inference, evaluation |
| **Hugging Face `transformers`** | Raw model access for custom pooling |
| **MTEB** (package + leaderboard) | Standardised evaluation and shortlisting |
| **FAISS** (`IndexFlatIP`) | Exact search for evaluation; ANN later (03.02) |
| **ONNX Runtime / TensorRT / Text Embeddings Inference (TEI)** | High-throughput embedding serving |
| **vLLM / SGLang embedding endpoints** | Serving LLM-based embedders |
| **Hosted embedding APIs** (OpenAI, Voyage, Cohere, Google, …) | Managed models; check dimensions, MRL support, pricing, data policy |
| **ranx / pytrec_eval / ir_measures** | IR metrics with significance tests |
