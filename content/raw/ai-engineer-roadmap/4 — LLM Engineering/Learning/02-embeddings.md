# Embeddings — Master Study Guide

> **Track:** LLM Engineering · **Module:** 02
> **Prerequisites:** Basic linear algebra (dot product, cosine similarity), Module 01.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Embeddings are dense numerical vectors that represent **meaning** in a form machines can compare, cluster, and index. They are the foundation of:

- **Semantic search** — "find documents like this one" without keyword overlap.
- **RAG (Retrieval-Augmented Generation)** — retrieving the right context before generating.
- **Clustering / topic discovery** on unlabeled data.
- **Recommender systems** — user embeddings, item embeddings, matched via dot product.
- **Classification** — embed once, train a small classifier on top.
- **Deduplication** — near-duplicate detection.

An embedding model turns text (or images, or audio) into a vector where **semantic similarity ≈ geometric proximity**. Once you internalize this, half the modern AI stack demystifies.

**Fundamental principles you must own:**

1. **An embedding is a dense vector** (typically 384 – 3072 dimensions) that represents a piece of content.
2. **Cosine similarity** (or dot product on unit-normalized vectors) measures semantic closeness.
3. **The embedding model is not the LLM.** They're specialized — usually smaller, encoder-only, trained with contrastive objectives.
4. **Embedding quality depends on training data and objective.** A model trained for "sentences similar in meaning" ≠ one trained for "queries retrieve relevant docs".
5. **Higher dimensions cost more** (storage, retrieval time) but often plateau on quality past ~768.
6. **Normalization matters.** Always L2-normalize before cosine similarity; some models embed unit-normalized by default.

If you retain nothing else: **an embedding is a vector where nearness = semantic similarity, and it's how machines "understand" without generating.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 How Embedding Models Work

Modern text embeddings come from **encoder-only transformers** (BERT-family), typically fine-tuned with a **contrastive objective**:

- Given a batch of (query, positive_doc) pairs, embed each.
- Loss: pull each pair's embeddings together; push the query away from all *other* docs in the batch (as negatives).

This is **InfoNCE** loss (Module 04, Month 3) — the same objective used in SimCLR for images.

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\sum_{d} \exp(\text{sim}(q, d) / \tau)}$$

Where $\text{sim}$ is cosine similarity and $\tau$ is a temperature.

**Pooling strategy** — how do you turn a transformer's per-token outputs into a single vector?

- **CLS token** — take the `[CLS]` token's output. Common in BERT-derived models.
- **Mean pooling** — average of all token outputs (masked by attention mask to skip padding). Standard for `sentence-transformers`.
- **Last-token pooling** — for decoder-only encoders (e.g., some LLM-derived embedding models). Take the last non-padding token's output.

**Training data sources:**
- Question-answer pairs (MS MARCO, Natural Questions).
- Paraphrase pairs (Quora, StackExchange duplicates).
- Web-scale weak supervision (title-text pairs).

**Popular embedding model families in 2026:**

| Family | Examples | Dim | Use case |
|--------|----------|-----|----------|
| OpenAI | `text-embedding-3-small`, `text-embedding-3-large` | 1536 / 3072 (variable) | General-purpose, API |
| Cohere | `embed-english-v3.0`, `embed-multilingual-v3.0` | 1024 | Multilingual, API |
| sentence-transformers | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` | 384 / 768 | Small, self-hosted |
| BGE (BAAI) | `bge-large-en-v1.5`, `bge-m3` | 1024 | Strong open, multi-vector |
| Nomic | `nomic-embed-text-v1.5` | 768 (matryoshka) | Fully open, long context |
| Voyage | `voyage-3` | 1024 | Retrieval-optimized |
| E5 (Microsoft) | `intfloat/multilingual-e5-large` | 1024 | Multilingual open |
| jina.ai | `jina-embeddings-v3` | 1024 | Late chunking, long context |

**Which to pick?** See Module 13 for retrieval-specific selection. Rule of thumb: `bge-large-en-v1.5` for English open-source; `text-embedding-3-small` for cheap API; `voyage-3` or `text-embedding-3-large` for higher quality.

---

### 2.2 Similarity Metrics: Cosine, Dot Product, Euclidean

**Cosine similarity** — the default:

$$\text{cos\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

- Range $[-1, 1]$. Invariant to vector magnitude.
- Interpretation: measures the **angle** between vectors.
- Most embedding models are trained to maximize this metric.

**Dot product** — cheaper, equivalent to cosine when vectors are unit-normalized:

$$\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i$$

- Cheaper (no norm computation).
- Some models (OpenAI's `text-embedding-3-*`) return normalized vectors, so dot product = cosine.
- **Always L2-normalize before dot-product search** if using it as a stand-in for cosine.

**Euclidean (L2) distance:**

$$d_2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_i (u_i - v_i)^2}$$

- Range $[0, \infty)$. Smaller = more similar.
- For unit-normalized vectors, monotonically related to cosine: $\|u - v\|^2 = 2 - 2\cos(u, v)$, so ranking by Euclidean = ranking by cosine.

**Which to use:**
- **Cosine** (or dot product on normalized vectors) — nearly always. It's what the model was trained for.
- Euclidean only if the model was explicitly trained with it (rare).

**Score interpretation.** Cosine 0.85 is not "85% match" in any calibrated sense. Different models have different score distributions. A score of 0.85 with one embedding model might be excellent; with another, mediocre. **Always calibrate by looking at your own data.**

---

### 2.3 Dimensionality: Cost, Quality, and Matryoshka

**Standard embedding dims:** 384 (small), 768 (mid), 1024 or 1536 (large), 3072 (very large).

**Storage cost.** Each float32 dimension = 4 bytes. A 1024-dim vector is 4 KB. 10 million vectors × 4 KB = **40 GB** — the memory-index cost that determines vector-DB architecture.

**Quality vs dim.** More dims usually helps, but with diminishing returns past ~768 for most tasks. Task-specific — for hard retrieval on specialized domains, 3072 may help.

**Half precision (fp16) or int8 quantization** — halves or quarters memory with minimal quality loss for many workloads. Widely supported in vector DBs (pgvector's `halfvec`, Qdrant's scalar quantization).

**Matryoshka Representation Learning (MRL)** — train an embedding such that its first $k$ dimensions form a *valid* smaller embedding on their own. Truncate at any dim (say, from 3072 → 512) and quality degrades gracefully.

- OpenAI's `text-embedding-3-*` supports Matryoshka: you can request a `dimensions=512` embedding by truncating.
- Nomic's `nomic-embed-text-v1.5` explicitly supports this.
- Advantage: **run cheap retrieval on truncated dims, then rerank with full dims**.

**Best-practice pipeline:**
1. Store full-dim (1024) embeddings in the DB.
2. For fast candidate retrieval, use int8-quantized or truncated (256) versions.
3. Rerank top candidates with the full float embedding.

Storage-conscious teams see 4–10× cost reduction using these tricks.

---

### 2.4 Embedding Non-Text: Images, Audio, Multimodal

**Image embeddings:**
- **CLIP** (Radford et al., 2021) — a vision-language embedding model. Text and image project into a **shared space**. Enables zero-shot image classification and text-to-image search.
- **SigLIP, DINO-v2** — successors with different pretraining objectives.
- **Meta's ImageBind** — six modalities in one space (image, text, audio, video, depth, thermal).

**Audio embeddings:**
- **Whisper's encoder** — can be extracted for embedding-based audio search.
- **CLAP** — contrastive language-audio pretraining, analogous to CLIP for audio.

**Code embeddings:**
- **OpenAI's `text-embedding-3-*`** trained on code too.
- **CodeBERT, GraphCodeBERT** — code-specific.
- **Voyage-code** — code retrieval.

**Cross-modal search.** With CLIP, you can search images with text queries — the query text embedding lies in the same space as image embeddings, so cosine similarity works across modalities.

**Late interaction models (ColBERT).** Instead of one embedding per document, embed each **token** individually and score by max-similarity per query token:

$$\text{score}(q, d) = \sum_{t \in q} \max_{t' \in d} \text{sim}(v_t, v_{t'})$$

- Much higher quality than single-vector retrieval, but more compute + storage (one vector per token).
- Used in ColBERTv2, BGE-M3 (in its multi-vector mode).

**Reranker vs embedding.** Embedding models produce vectors for cheap ANN retrieval. **Rerankers** (Module 16) are typically cross-encoders that score (query, doc) pairs jointly — slower, more accurate. Standard pipeline: embed → retrieve top-100 → rerank → top-10.

---

### 2.5 Practical Embedding Use in Production

**Batching.** Embedding APIs handle batches efficiently:

```python
import openai
client = openai.OpenAI()

texts = ["doc 1", "doc 2", "doc 3", ...]  # up to 2048 for OpenAI
resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
vectors = [d.embedding for d in resp.data]
```

**With `sentence-transformers` (self-hosted):**

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
vectors = model.encode(texts, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=True)
```

Batch size is a memory/throughput tradeoff. On a single GPU, batch size 64–256 is typical.

**Instruction prefixes.** Many modern embedding models expect a **prefix** to distinguish queries from documents:

- BGE: `query: {q}` and `passage: {d}`.
- E5: `query: {q}` and `passage: {d}`.
- Nomic: `search_query: {q}` and `search_document: {d}`.

Skipping this hurts quality substantially — the model was trained to expect it.

**Caching.** Embeddings are deterministic (given the same model). Cache them:

```python
import hashlib
def cache_key(text, model):
    return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()
```

Store in Redis, disk, or your vector DB itself. Re-embedding costs money — only redo when the text or model changes.

**Chunking before embedding.** Embedding models have a max input length (e.g., 512 tokens for many, 8192 for some newer ones). Long docs must be chunked first (Module 12). One embedding per chunk; retrieve chunks, not docs.

**Model migrations.** If you upgrade your embedding model, **all** existing embeddings must be re-computed — vectors from different models aren't comparable. Plan migrations carefully:
- Store both old and new embeddings during a transition.
- Re-embed content in the background.
- Cutover queries once coverage is complete.

**Cost estimation.** OpenAI's `text-embedding-3-small` is ~$0.02 per 1M tokens (2026 pricing). Embedding a 10M-token corpus costs ~$200. Cheap.

---

## 3. Mental Models & Analogies

### 3.1 The "Meaning Map" Model

An embedding model draws a **map of meanings** in a high-dimensional space. Every sentence, every paragraph, every image gets placed on this map. **Similar meanings sit close together; dissimilar meanings sit far apart** — regardless of the specific words used.

- "The cat sat on the mat" and "A feline rested on the rug" land in the same neighborhood, even though they share almost no words.
- "Java coffee" and "Java the programming language" land in *different* neighborhoods because the model learned context from training.
- CLIP places a photo of a dog *and* the string "a dog" close together in a **shared** map spanning images and text.

Once you have the map, machine operations become geometric operations. Semantic search = nearest neighbors. Clustering = finding dense regions. Deduplication = collapsing very-close pairs. Recommendation = finding items near a user's average position.

The map's quality is set by training. A general-purpose embedding maps concepts uniformly; a domain-specific one (legal, medical) distinguishes concepts your general map treats as identical. **Choosing an embedding model = choosing the map's cartographer.**

### 3.2 The "GPS Coordinates for Ideas" Model

An embedding is like a GPS coordinate — but with 384 or 1024 or 3072 axes instead of latitude/longitude. Two coordinates near each other represent nearby *ideas* rather than nearby *places*.

- Storage: your database is a **giant world map** with coordinates for every document.
- Querying: your question also has a coordinate. Find nearby documents.
- The distance metric (cosine) is the "GPS to distance" formula on this weird map.

This model gives correct intuitions:
- **Bigger dimensionality** = a more detailed map (finer distinctions), but more storage and slower search.
- **Different embedding models** = different mapping projections. A map projection that expands the poles vs one that expands the equator — same globe, different distortions.
- **Approximate nearest neighbors (Module 14)** = using a road network to find nearby places quickly instead of measuring every pairwise distance.
- **Fine-tuning an embedding on your data** = redrawing regions of the map to match your specific application's meaning distinctions.

![IMG-EMB-01](/4%20—%20LLM%20Engineering/images/IMG-EMB-01.jpg)
> **Caption:** Embeddings map meaning into a geometric space; semantically similar items land near each other.
> **Placement:** Section 2.1 or Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "One Embedding Model Fits All Tasks"

Different training objectives produce different geometries. A model trained for **paraphrase similarity** (STS) may perform poorly on **retrieval** (find relevant docs given a query). Symmetric similarity ≠ asymmetric retrieval — a query "how do transformers work" and a document "The transformer architecture uses attention..." are not paraphrases, but they should retrieve each other. Use retrieval-trained models (E5, BGE, MS-MARCO fine-tunes) for retrieval, not raw STS models.

### 4.2 "Cosine Similarity Score Is Absolute"

A 0.85 cosine is not "85% relevant" in any calibrated sense — it's a model-specific score. Different models have wildly different distributions (some cluster around 0.3–0.7; others 0.5–0.95). Set thresholds **empirically per model and per dataset**. Never hard-code "similarity > 0.8" as a decision rule without validation.

### 4.3 "Forgetting Instruction Prefixes"

Many modern embedding models (BGE, E5, Nomic, some Voyage models) expect a prefix like `query: {q}` for queries and `passage: {d}` for documents. Skipping this can drop retrieval quality by 5–15%. Read the model card. Include the prefix. If you're mixing multiple embedding models, keep a per-model prefix registry.

---

## 5. Self-Assessment Bank (Embeddings)

### Questions

**Q1 (Short answer).** In one sentence, what does an embedding model do?

**Q2 (Multiple choice).** Which similarity metric is the standard default for text embeddings?
- (a) Manhattan (L1)
- (b) Cosine similarity
- (c) Chebyshev
- (d) Hamming

**Q3 (Short answer).** Explain the InfoNCE loss used to train modern embedding models — what pulls together, what pushes apart?

**Q4 (Multiple choice).** L2-normalizing vectors before using dot product for search:
- (a) Slows things down.
- (b) Makes dot product equivalent to cosine similarity.
- (c) Is unnecessary.
- (d) Breaks the search.

**Q5 (Short answer).** What is Matryoshka Representation Learning, and why is it useful in production?

**Q6 (Multiple choice).** For a fast candidate-retrieval / accurate-reranker pipeline, the standard division is:
- (a) Both stages use the same embedding model.
- (b) Retrieval: fast embedding model + ANN; reranking: cross-encoder scoring (query, doc) pairs.
- (c) Retrieval: LLM; reranker: BM25.
- (d) Retrieval: hashing; reranker: SQL.

**Q7 (Short answer).** Name three popular embedding models and one situation where you'd choose each.

**Q8 (Multiple choice).** Instruction prefixes on models like BGE and E5 (e.g., `query: `) are:
- (a) Optional formatting.
- (b) Required for optimal quality — the model was trained to expect them.
- (c) Only for languages other than English.
- (d) Only for images.

**Q9 (Short answer).** Why can't you compare cosine similarity scores across two different embedding models?

**Q10 (Multiple choice).** CLIP is unique because it:
- (a) Embeds images only.
- (b) Embeds text only.
- (c) Embeds images and text into a shared space, enabling text↔image search.
- (d) Requires labeled data for every task.

---

### Answer Key & Detailed Explanations

**A1.** An embedding model maps text (or other content) to a dense vector in a high-dimensional space where semantic similarity corresponds to geometric closeness (typically cosine similarity).

**A2. (b).** Cosine similarity — measures the angle between vectors, invariant to magnitude. Nearly all embedding models are trained to maximize cosine similarity between semantically related pairs.

**A3.** InfoNCE (contrastive) loss: given a batch of (query, positive_doc) pairs, embed each. For each query, softmax over similarities to *all* candidate docs in the batch (its positive and the others as negatives). Cross-entropy against a target that says "the positive is the correct one." Effect: pulls positive pairs together and pushes negatives apart in embedding space.

**A4. (b).** For unit-normalized vectors, dot product equals cosine similarity — same ranking, faster compute (no norm calculation at query time). Many vector DBs assume normalized inputs when configured for dot-product distance.

**A5.** Matryoshka Representation Learning trains an embedding model such that its first $k$ dimensions form a valid smaller embedding. You can truncate a 3072-dim vector to 512 dims with graceful quality degradation. Production use: store full vectors, but do fast approximate retrieval on truncated versions; rerank with full vectors. Saves storage and query time with minimal quality cost.

**A6. (b).** Standard two-stage pipeline: (1) **Retrieval** — a fast bi-encoder embedding model + approximate nearest neighbor index returns top-100 candidates. (2) **Reranking** — a cross-encoder or a larger LLM scores each (query, doc) pair jointly, producing top-10. Bi-encoders are fast (embed independently), cross-encoders are more accurate (attend across query and doc).

**A7.** Any three with valid reasoning, for example: **`text-embedding-3-small`** — cheap API, general-purpose, 1536-dim. **`bge-large-en-v1.5`** — best open English retrieval-tuned. **`all-MiniLM-L6-v2`** — very small (384 dim), fast, self-hosted, when speed and cost dominate. **`multilingual-e5-large`** — solid multilingual retrieval. **`voyage-3`** — retrieval-optimized API. **CLIP** — image-text shared space.

**A8. (b).** Required. Models like BGE and E5 were trained with these prefixes; using the model without them produces significantly worse similarity scores because the model has learned to treat prefixed inputs differently. See each model's card for the exact prefix format.

**A9.** Different models were trained with different loss functions, temperatures, and data. Their embedding spaces have different score distributions — 0.7 may be "excellent" in one model and "mediocre" in another. You can only calibrate thresholds within a single model on labeled data.

**A10. (c).** CLIP was trained on 400M (image, text-caption) pairs with a contrastive objective, producing a shared embedding space. This enables zero-shot image classification (embed the image, embed candidate class names, pick nearest) and cross-modal search (text query → image results, or vice versa).

---

## 6. Practice Prompts

1. **Compare models.** Embed 100 sentence pairs (a public STS benchmark) using OpenAI, `bge-base-en-v1.5`, and `all-MiniLM-L6-v2`. Correlate their similarity scores with human labels. Which wins?
2. **Instruction prefix ablation.** Take BGE. Embed 100 (query, doc) pairs with and without the prefix. Compute Recall@10 both ways. Confirm the drop.
3. **Matryoshka.** With OpenAI's `text-embedding-3-large` (3072 dim), truncate to 3072, 1024, 512, 256, 128 dims. Compute retrieval quality (Recall@10) on a small labeled dataset for each dim.
4. **CLIP cross-modal.** Use `sentence-transformers`' `clip-ViT-B-32`. Encode a folder of images and a set of text queries. Retrieve the top image for each query.
5. **Deduplication.** Embed a corpus of documents; find pairs with cosine > 0.95; inspect them — how often are they true near-duplicates? Calibrate a threshold.

---

## 7. References

- Reimers & Gurevych, ["Sentence-BERT"](https://arxiv.org/abs/1908.10084) (2019).
- Radford et al., ["Learning Transferable Visual Models from Natural Language Supervision"](https://arxiv.org/abs/2103.00020) (2021) — CLIP.
- Wang et al., ["Text Embeddings by Weakly-Supervised Contrastive Pre-training"](https://arxiv.org/abs/2212.03533) (2022) — E5.
- Kusupati et al., ["Matryoshka Representation Learning"](https://arxiv.org/abs/2205.13147) (2022).
- Xiao et al., ["C-Pack: Packaged Resources To Advance General Chinese Embedding"](https://arxiv.org/abs/2309.07597) (2023) — BGE.
- Massive Text Embedding Benchmark (MTEB): [leaderboard on HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) — current SOTA per task.
