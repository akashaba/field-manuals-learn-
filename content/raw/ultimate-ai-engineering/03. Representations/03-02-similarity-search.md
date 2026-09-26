# 03.02 — Similarity Search

> **Module 3: Representations** · Subtopic 2 of 4
> **Prerequisites:** 03.01 (normalised embeddings, recall metrics), k-means, heaps and graph traversal, memory hierarchy (RAM vs SSD, cache lines, SIMD).
> **Outcome:** you can implement the core approximate nearest neighbour (ANN) families from scratch (LSH, IVF, PQ, HNSW) and reason quantitatively about their recall, latency, memory, and build-time trade-offs. You can handle filtered search correctly, and choose and tune an index for a given scale and SLO.

---

## 1. The Problem

Given a database $X = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\} \subset \mathbb{R}^d$ and a query $\mathbf{q}$, return

$$
\mathrm{kNN}(\mathbf{q}) = \mathrm*{arg\,top\text{-}k}_{i \in [N]} \; -\mathrm{dist}(\mathbf{q}, \mathbf{x}_i)
$$

**Exact search** costs $O(Nd)$ per query. With $N = 10^8$ and $d = 1024$ that is $10^{11}$ multiply-adds (~100 ms on a big GPU, and far more on a CPU), and 400 GB of fp32 vectors streamed from memory per query. ANN trades a little recall for orders of magnitude in speed:

$$
\text{Recall@}k = \frac{|\mathrm{ANN}_k(\mathbf{q}) \cap \mathrm{kNN}_k(\mathbf{q})|}{k}
$$

Note that this is recall **against exact search**, which is different from relevance recall in 03.01. A retrieval system's end-to-end quality is bounded by both.

### 1.1 Why high dimensions are hard

- **Distance concentration** (Beyer et al., 1999): as $d$ grows, for many distributions $\frac{\max_i \mathrm{dist} - \min_i \mathrm{dist}}{\min_i \mathrm{dist}} \to 0$. Nearest and farthest neighbours become nearly equidistant.
- **Space partitioning fails:** KD-trees must visit a number of cells that grows exponentially with $d$ and degrade to brute force beyond ~20 dimensions.
- **What saves us:** real embeddings have a low **intrinsic dimension** (they lie near a manifold), and we only need *approximate* answers.

### 1.2 Metrics and the MIPS reduction

- Use **L2** or **inner product (IP)**. Cosine = IP on normalised vectors (03.01 §1.3).
- **Maximum inner-product search** (MIPS, for unnormalised vectors such as recommendation embeddings) is not a metric search: $\langle \mathbf{q},\mathbf{x}\rangle$ does not satisfy the triangle inequality. Reduce it to L2 by augmenting each vector with one extra dimension (Bachrach et al., 2014), where $M = \max_i \|\mathbf{x}_i\|$:

$$
\tilde{\mathbf{x}} = \big[\mathbf{x};\ \sqrt{M^2 - \|\mathbf{x}\|^2}\big],\quad \tilde{\mathbf{q}} = [\mathbf{q};\ 0]
\;\Rightarrow\;
\|\tilde{\mathbf{q}} - \tilde{\mathbf{x}}\|^2 = \|\mathbf{q}\|^2 + M^2 - 2\langle \mathbf{q}, \mathbf{x}\rangle
$$

Minimising L2 distance over the augmented vectors therefore maximises the inner product.

### 1.3 Exact baseline (always keep one)

```python
import numpy as np


def exact_topk(Q: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Inner-product top-k for unit vectors. Q: (nq, d), X: (N, d). Returns (nq, k) ids, best first."""
    S = Q @ X.T
    idx = np.argpartition(-S, kth=min(k, S.shape[1] - 1), axis=1)[:, :k]
    part = np.take_along_axis(S, idx, axis=1)
    return np.take_along_axis(idx, np.argsort(-part, axis=1), axis=1)


def recall_vs_exact(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)]))
```

Exact search is the **right answer** for ≲ 1M vectors on a GPU, and for heavily filtered subsets (§6). It is also mandatory as ground truth in every benchmark.

---

## 2. Locality-Sensitive Hashing (LSH)

A hash family is **locality-sensitive** if similar items collide with higher probability. For cosine similarity, **SimHash** (Charikar, 2002) uses random hyperplanes $\mathbf{r} \sim \mathcal{N}(0, I)$ and $h(\mathbf{x}) = \mathrm{sign}(\langle \mathbf{r}, \mathbf{x}\rangle)$:

$$
\Pr[h(\mathbf{x}) = h(\mathbf{y})] = 1 - \frac{\theta(\mathbf{x},\mathbf{y})}{\pi}
$$

Concatenate $b$ bits per table ("AND" — raises precision) and use $L$ tables ("OR" — raises recall). A pair with single-bit collision probability $p$ becomes a candidate with probability

$$
P_{\text{cand}} = 1 - (1 - p^{b})^{L}
$$

This is the S-curve that you tune with $b$ and $L$.

```python
from collections import defaultdict


class SimHashLSH:
    def __init__(self, dim: int, bits: int = 16, tables: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((tables, bits, dim))
        self.buckets = [defaultdict(list) for _ in range(tables)]
        self.X = None

    def _keys(self, x: np.ndarray):
        bits = (np.einsum("tbd,d->tb", self.planes, x) > 0)
        return [row.tobytes() for row in np.packbits(bits, axis=1)]

    def fit(self, X: np.ndarray):
        self.X = X
        for i, x in enumerate(X):
            for t, key in enumerate(self._keys(x)):
                self.buckets[t][key].append(i)
        return self

    def search(self, q: np.ndarray, k: int = 10):
        cand = {i for t, key in enumerate(self._keys(q)) for i in self.buckets[t].get(key, ())}
        if not cand:
            return []
        cand = np.fromiter(cand, dtype=np.int64)
        return cand[np.argsort(-(self.X[cand] @ q))[:k]].tolist()
```

**In practice:** LSH has clean theoretical guarantees but needs many tables (memory) to reach high recall on dense text embeddings. Graph and quantization methods dominate. LSH is still useful for **near-duplicate detection** (MinHash/SimHash over shingles or fingerprints) and streaming settings.

---

## 3. Inverted File Index (IVF)

Partition the space with k-means into $n_{\text{list}}$ cells (the **coarse quantizer**), and store each vector in the list of its nearest centroid. At query time, scan only the $n_{\text{probe}}$ closest lists.

```
 centroids c_1..c_nlist                         query q
        │                                          │
  ┌─────┴──────────────────────┐          nearest nprobe centroids
  │ list 1: [ids, vectors]     │◄─────────────────┤
  │ list 2: [ids, vectors]     │                  │
  │ ...                        │◄─────────────────┘
  │ list nlist                 │       scan ≈ N·nprobe/nlist vectors
  └────────────────────────────┘
```

- **Cost per query:** $\approx n_{\text{list}}\cdot d + \frac{n_{\text{probe}}}{n_{\text{list}}} N d$. Choose $n_{\text{list}}$ on the order of $\sqrt{N}$ (FAISS guidance: roughly $4\sqrt{N}$–$16\sqrt{N}$) and tune $n_{\text{probe}}$ for recall.
- **Failure mode:** true neighbours that sit just across a cell boundary. Raising $n_{\text{probe}}$ fixes it at a linear cost. List sizes can be **imbalanced**: skewed data yields giant lists and unpredictable latency.
- **Updates:** inserts go to the nearest list and are cheap. After heavy distribution drift, the centroids go stale and you need a retrain.

```python
def kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        d2 = (X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1)
        assign = d2.argmin(1)
        for j in range(k):
            members = X[assign == j]
            C[j] = members.mean(0) if len(members) else X[rng.integers(len(X))]
    return C


class IVFFlat:
    def __init__(self, nlist: int):
        self.nlist = nlist

    def fit(self, X: np.ndarray):
        self.C = kmeans(X, self.nlist)
        assign = (X @ self.C.T - 0.5 * (self.C ** 2).sum(1)).argmax(1)   # nearest centroid (L2)
        self.lists = [np.where(assign == j)[0] for j in range(self.nlist)]
        self.X = X
        return self

    def search(self, q: np.ndarray, k: int = 10, nprobe: int = 8) -> np.ndarray:
        probe = np.argsort(-(self.C @ q - 0.5 * (self.C ** 2).sum(1)))[:nprobe]
        cand = np.concatenate([self.lists[j] for j in probe])
        return cand[np.argsort(-(self.X[cand] @ q))[:k]]
```

---

## 4. Product Quantization (PQ)

PQ compresses vectors so that the whole index fits in RAM (or GPU memory), and computes distances *without decompressing them* (Jégou et al., 2011).

### 4.1 Encoding

Split $\mathbf{x} \in \mathbb{R}^d$ into $m$ subvectors of $d/m$ dimensions. Learn a k-means codebook with $k^* = 256$ centroids **per subspace**. Encode each subvector by its centroid index (1 byte):

$$
\mathbf{x} \approx [\,\mathbf{c}^{(1)}_{i_1};\ \mathbf{c}^{(2)}_{i_2};\ \ldots;\ \mathbf{c}^{(m)}_{i_m}\,], \qquad \text{code} = (i_1,\ldots,i_m) \in \{0..255\}^m
$$

**Memory:** $m$ bytes per vector (e.g. $d = 1024$ fp32 = 4096 B → $m = 64$ gives **64 B, a 64× reduction**), plus the codebooks, $m \cdot 256 \cdot (d/m) \cdot 4$ bytes = $1024\,d$ bytes total.

### 4.2 Asymmetric distance computation (ADC)

Keep the query in full precision. Precompute one lookup table per subspace, $T_j[c] = \|\mathbf{q}^{(j)} - \mathbf{c}^{(j)}_c\|^2$, which costs $256\,d$ flops per query. Each database distance is then just $m$ table lookups plus additions:

$$
\|\mathbf{q} - \mathbf{x}\|^2 \approx \sum_{j=1}^{m} T_j\big[\,i_j(\mathbf{x})\,\big]
$$

This is SIMD-friendly. FAISS "fast scan" packs 4-bit codes into registers, reaching billions of distance evaluations per second per core.

```python
class ProductQuantizer:
    def __init__(self, d: int, m: int, ks: int = 256):
        assert d % m == 0
        self.d, self.m, self.ks, self.ds = d, m, ks, d // m

    def fit(self, X: np.ndarray, iters: int = 20):
        self.codebooks = np.stack([kmeans(X[:, j * self.ds:(j + 1) * self.ds], self.ks, iters, seed=j)
                                   for j in range(self.m)])              # (m, ks, ds)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        codes = np.empty((len(X), self.m), dtype=np.uint8)
        for j in range(self.m):
            sub, C = X[:, j * self.ds:(j + 1) * self.ds], self.codebooks[j]
            codes[:, j] = ((sub ** 2).sum(1, keepdims=True) - 2 * sub @ C.T + (C ** 2).sum(1)).argmin(1)
        return codes

    def adc_table(self, q: np.ndarray) -> np.ndarray:
        qs = q.reshape(self.m, 1, self.ds)
        return ((self.codebooks - qs) ** 2).sum(-1)                        # (m, ks)

    def search(self, q: np.ndarray, codes: np.ndarray, k: int = 10) -> np.ndarray:
        T = self.adc_table(q)
        dist = T[np.arange(self.m), codes].sum(1)                          # (N,)
        return np.argsort(dist)[:k]
```

### 4.3 Variants you'll meet

- **IVF-PQ:** an IVF coarse quantizer, with PQ encoding the **residual** $\mathbf{x} - \mathbf{c}_{\text{list}}$ (smaller residuals → more accurate codes). This is the workhorse of billion-scale in-RAM search.
- **OPQ:** learn a rotation $R$ before PQ so that subspaces are decorrelated and balanced.
- **Anisotropic quantization (ScaNN):** penalise quantization error *parallel* to $\mathbf{x}$ more heavily than orthogonal error, because parallel error distorts inner products more.
- **RaBitQ:** randomised binary quantization with theoretical error bounds; strong recall per bit.
- **Rescoring:** PQ distances are approximate, so retrieve $k \cdot r$ candidates and re-rank them with full-precision vectors (from RAM, SSD, or object storage).

---

## 5. Graph-Based Search: HNSW and DiskANN

### 5.1 HNSW (Hierarchical Navigable Small World)

HNSW builds a multi-layer proximity graph (Malkov & Yashunin, 2018):

```
 layer 2:   A ─────────────────────── F                  few nodes, long links (express lanes)
            │                         │
 layer 1:   A ────── C ────── E ───── F ──── H           more nodes
            │        │        │       │      │
 layer 0:   A─B──C──D──E──F──G──H──I──J──K──L            all nodes, ≤ M0 = 2M links each
```

- **Level assignment:** each node gets level $\ell = \lfloor -\ln(U)\cdot m_L \rfloor$ with $U \sim \text{Uniform}(0,1)$ and $m_L = 1/\ln M$, so $\Pr[\ell \geq l] = M^{-l}$. The layer sizes shrink geometrically, like a skip list.
- **Search:** greedy descent from the top-layer entry point to layer 1 (beam width 1). Then, on layer 0, a best-first search with a candidate list of size `efSearch` returns the top-k. The typical complexity is $O(\log N)$ hops.
- **Insertion:** search with `efConstruction` to find candidates on each layer. Connect to $M$ neighbours chosen by the **diversity heuristic**: keep a candidate only if it is closer to the new node than to any already-selected neighbour. This keeps links spread in different directions, which preserves navigability in clustered data.
- **Parameters:**
  - `M` (16–64) sets memory and recall.
  - `efConstruction` (100–500) sets build quality and build time.
  - `efSearch` (≥ k; 50–500) is the per-query recall/latency knob.
- **Memory:** vectors plus about $N \cdot M_0 \cdot 4$ bytes of links at layer 0, plus smaller upper layers. With $M = 32$ that is roughly 256–300 B per vector *on top of* the vectors themselves.
- **Weak points:** HNSW is RAM-resident, has random memory access (cache misses), builds slowly at scale, and handles **deletions** poorly (tombstones degrade the graph; periodic rebuilds or repair are needed).

```python
import heapq
import math
import random


class HNSW:
    """Compact educational HNSW for unit vectors (cosine distance = 1 - dot)."""

    def __init__(self, M: int = 16, ef_construction: int = 100, seed: int = 0):
        self.M, self.M0, self.efc = M, 2 * M, ef_construction
        self.mL, self.rng = 1 / math.log(M), random.Random(seed)
        self.vecs: list[np.ndarray] = []
        self.graph: list[dict[int, list[int]]] = []    # graph[level][node] -> neighbours
        self.entry, self.max_level = None, -1

    def _d(self, a: np.ndarray, i: int) -> float:
        return 1.0 - float(self.vecs[i] @ a)

    def _search_layer(self, q, entry_points, ef, level):
        visited = set(entry_points)
        cand = [(self._d(q, e), e) for e in entry_points]
        heapq.heapify(cand)
        best = [(-d, e) for d, e in cand]                  # max-heap of the ef closest so far
        heapq.heapify(best)
        while cand:
            d, c = heapq.heappop(cand)
            if d > -best[0][0]:
                break                                       # closest candidate worse than worst kept
            for n in self.graph[level].get(c, ()):
                if n in visited:
                    continue
                visited.add(n)
                dn = self._d(q, n)
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(cand, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-nd, e) for nd, e in best)           # ascending distance

    def _select(self, base: np.ndarray, cands, M):
        """Diversity heuristic, then fill up with the closest remaining candidates."""
        chosen = []
        for d, c in cands:
            if all(self._d(self.vecs[c], s) > d for s in chosen):
                chosen.append(c)
            if len(chosen) >= M:
                return chosen
        rest = [c for _, c in cands if c not in chosen]
        return chosen + rest[:M - len(chosen)]

    def add(self, x: np.ndarray) -> int:
        x = x / np.linalg.norm(x)
        i = len(self.vecs)
        self.vecs.append(x)
        level = int(-math.log(1.0 - self.rng.random()) * self.mL)
        while len(self.graph) <= level:
            self.graph.append({})
        for l in range(level + 1):
            self.graph[l][i] = []
        if self.entry is None:
            self.entry, self.max_level = i, level
            return i
        ep = [self.entry]
        for l in range(self.max_level, level, -1):          # greedy descent above the node's level
            ep = [self._search_layer(x, ep, 1, l)[0][1]]
        for l in range(min(level, self.max_level), -1, -1):
            cands = self._search_layer(x, ep, self.efc, l)
            m_max = self.M0 if l == 0 else self.M
            self.graph[l][i] = self._select(x, cands, self.M)
            for n in self.graph[l][i]:
                self.graph[l][n].append(i)
                if len(self.graph[l][n]) > m_max:            # shrink the neighbour's list
                    vn = self.vecs[n]
                    nc = sorted((self._d(vn, j), j) for j in self.graph[l][n])
                    self.graph[l][n] = self._select(vn, nc, m_max)
            ep = [e for _, e in cands]
        if level > self.max_level:
            self.entry, self.max_level = i, level
        return i

    def search(self, q: np.ndarray, k: int = 10, ef: int = 50) -> list[int]:
        q = q / np.linalg.norm(q)
        ep = [self.entry]
        for l in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        return [e for _, e in self._search_layer(q, ep, max(ef, k), 0)[:k]]
```

![LLM-03-1](/03.%20Representations/images/LLM-03-1.jpg)

### 5.2 DiskANN / Vamana: billion scale on one node

DiskANN (Subramanya et al., 2019) keeps a **single-layer graph plus full vectors on SSD**, and only **PQ codes in RAM**:

- **Vamana graph:** built with **α-robust pruning**. A candidate is dropped if an already-selected neighbour $p^*$ satisfies $\alpha \cdot d(p^*, c) \le d(p, c)$, with $\alpha > 1$ keeping longer edges. The result is a small diameter, so few hops are needed.
- **Search:** a beam search guided by in-RAM PQ distances. Full vectors and neighbour lists are read from SSD in 4 KB sectors, and the candidates are reranked exactly.
- **Result:** billion-point search with ~5 ms latency and > 95% recall on a single machine with ~64 GB RAM plus NVMe. **Fresh-DiskANN** adds streaming inserts and deletes. **Filtered-DiskANN** adds label-aware graphs.

### 5.3 GPU ANN

GPU-native graphs (CAGRA in NVIDIA cuVS) and GPU IVF-PQ reach extreme throughput for **batched** queries and fast index builds. The trade-offs are GPU memory limits and cost. A common pattern is to build on the GPU and serve on CPUs (HNSW converted from CAGRA).

---

## 6. Filtered Vector Search

Real queries carry predicates: `tenant_id = 42 AND status = 'enacted' AND session >= 2023`. Let the **selectivity** $\sigma$ be the fraction of vectors that pass the filter.

| Strategy | How | Good when | Failure mode |
|---|---|---|---|
| **Pre-filter + exact** | Filter via a metadata index, then brute-force the survivors | $\sigma N$ small (≲ 10–50k) | Slow for large $\sigma N$ |
| **Post-filter** | ANN for $k' = k/\sigma \cdot$ slack, then filter | $\sigma$ large (≳ 0.1) | Returns < k results when $\sigma$ is small; latency spikes |
| **In-graph filtering** | Traverse the graph but only *return* passing nodes (still traversing others) | Medium $\sigma$ | At low $\sigma$ the search explores huge regions with no passing nodes |
| **Filter-aware graphs** (ACORN, Filtered-DiskANN) | Denser graphs, or label-specific edges, so that the passing subgraph stays connected | Low $\sigma$, arbitrary predicates | More memory and build time |
| **Partitioning** (per tenant or category) | Separate index per high-cardinality key | Hard multi-tenant isolation | Many small indexes; operational overhead |
| **Iterative scan** (e.g. pgvector `hnsw.iterative_scan`) | Keep scanning the index until k passing results are found | Postgres-native filtering | Latency is unbounded without a scan limit |

A **query planner** chooses the strategy by estimating $\sigma$:

```python
def plan_filtered_search(n_total: int, est_selectivity: float, k: int,
                         exact_limit: int = 50_000, post_filter_min_sel: float = 0.1) -> dict:
    n_pass = int(n_total * est_selectivity)
    if n_pass <= exact_limit:
        return {"strategy": "prefilter_exact", "scan": n_pass}
    if est_selectivity >= post_filter_min_sel:
        return {"strategy": "postfilter_ann", "k_prime": int(k / est_selectivity * 1.5) + k}
    return {"strategy": "filtered_graph", "ef_search": max(4 * k, int(k / est_selectivity))}
```

---

## 7. Choosing and Tuning an Index

### 7.1 Decision guide

| Scale (vectors) | Memory budget | Typical choice |
|---|---|---|
| ≤ 1M | Anything | **Exact (flat)** on GPU/CPU, or HNSW for sub-ms CPU latency |
| 1M–100M | Vectors fit in RAM | **HNSW** (int8/fp16 vectors) — best recall/latency on CPU |
| 1M–1B | Vectors don't fit | **IVF-PQ + rescoring** (RAM), or **DiskANN** (SSD) |
| ≥ 1B | Distributed | Sharded IVF-PQ / DiskANN / SPANN; partition by tenant or time |
| High batch QPS, GPU available | GPU memory | **CAGRA / GPU IVF-PQ** |

### 7.2 Benchmark protocol

1. Compute exact ground truth for ≥ 1,000 held-out queries drawn from the **real query distribution**. Don't use database vectors as queries — they find themselves.
2. Sweep the search-time knob (`efSearch`, `nprobe`) and plot **recall@10 vs QPS** on a log axis. Report the build time, index size, and P99 latency at a target recall (e.g. 0.95).
3. Test with **filters** at several selectivities, **concurrent inserts and deletes**, and a **cold cache** (first query after restart, or SSD page cache dropped).
4. Measure **end-to-end retrieval quality** (nDCG) as well. A 0.99 vs 0.95 ANN recall often makes no measurable difference after reranking — so don't overpay.

---

## 8. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Recall silently lower than assumed** | Relevance issues nobody can reproduce | Recall canary against exact search on sampled queries; alert on drops |
| **Filtered queries return < k** | Empty or short result lists | Selectivity-aware planning (§6); iterative scans; filter-aware indexes |
| **Deletes degrade HNSW** | Recall declines over weeks | Periodic rebuild/compaction; graph repair; delete-friendly indexes |
| **Imbalanced IVF lists** | P99 latency spikes | Balanced k-means; more lists; cap list sizes; periodic retraining |
| **Index doesn't fit in RAM** | OOM, swapping | int8/PQ compression; DiskANN; shard; MRL truncation (03.01) |
| **Slow builds block releases** | Hours to rebuild | GPU builds; incremental inserts; blue/green indexes |
| **Cold start latency** | Slow first queries after deploy | Warm up by preloading pages; replay recent queries |
| **Benchmarks don't match production** | Great offline numbers, poor live ones | Benchmark with real queries, real filters, concurrent writes, and production hardware |

---

## 9. Hands-On Projects

### Project 1 — Build IVF, PQ, and HNSW from Scratch; Benchmark Against FAISS and hnswlib

**User stories**
- *As an ML infrastructure engineer*, I want to implement the core ANN algorithms myself, so that I can tune and debug production vector indexes from first principles.

**Acceptance criteria**
1. Implements `exact_topk`, `SimHashLSH`, `IVFFlat`, `ProductQuantizer` (with IVF-PQ residual encoding added), and `HNSW` (§2–5), with unit tests that check recall against exact search on synthetic data.
2. Dataset: ≥ 1M real embeddings (your corpus from 03.01, or a public set such as a 1M sample of an ANN-benchmarks dataset), with 1,000 held-out queries and exact ground truth.
3. Produces recall@10-vs-QPS curves for each implementation *and* for FAISS (`IndexIVFPQ`, `IndexHNSWFlat`) and hnswlib, using the same threads and hardware.
4. Reports the build time, memory per vector (measured), and the effect of M, efConstruction, nlist/nprobe, and m (PQ sub-quantizers).
5. The write-up explains the gap between your implementation and the libraries (SIMD, memory layout, language), with a profile.

**Step-by-step**
1. Prepare the data: fp32, L2-normalised, with ground truth computed by `exact_topk` in batches.
2. Implement and unit-test each index on 10k synthetic points first.
3. Add IVF-PQ: coarse k-means, residuals, PQ on residuals, and ADC with per-list tables.
4. Wire in FAISS and hnswlib with the equivalent parameters.
5. Write a sweep runner that saves JSON results and a plotting script (log-scale QPS).
6. Profile your HNSW (`py-spy`), then optionally port the distance loop to numba or Rust.

---

### Project 2 — Filtered Vector Search Under Varying Selectivity

**User stories**
- *As a search engineer* for a multi-tenant document platform, I need filtered queries (tenant, document type, date range) to return k correct results within the latency SLO at any filter selectivity.

**Acceptance criteria**
1. A dataset of ≥ 2M vectors with realistic metadata (tenant with a Zipf-distributed size, document type, date).
2. A query set whose filter selectivities span 0.0001 to 0.5.
3. Strategies compared: pre-filter + exact, post-filter ANN with adaptive k′, in-graph filtering (hnswlib filter function or FAISS `IDSelector`), pgvector with iterative scan, and one engine with native filtered HNSW (e.g. Qdrant).
4. For each strategy × selectivity bucket: recall@10 against filtered exact search, the fraction of queries returning < k results, and P50/P99 latency.
5. Implements the `plan_filtered_search` planner (§6) with selectivity estimates from metadata statistics, and shows that it achieves the best-of-breed curve.

**Step-by-step**
1. Generate the metadata and compute filtered ground truth per query (exact search over the passing subset).
2. Implement each strategy behind a common interface.
3. Estimate selectivity with per-field histograms, assuming independence; measure the estimation error.
4. Run the sweep and plot recall and P99 latency against selectivity for each strategy.
5. Tune the planner's thresholds, and document the results.

---

### Project 3 — Capacity Plan and Prototype for 500M Vectors on a Budget

**User stories**
- *As a platform architect*, I need to serve 500M document chunks at 200 QPS with P99 < 50 ms and recall@10 ≥ 0.95 (against exact search), on the cheapest hardware, so that we can budget the project.

**Acceptance criteria**
1. A written capacity model (a spreadsheet or a Python notebook) for three designs:
   - (a) HNSW + int8 in RAM, sharded.
   - (b) IVF-PQ in RAM + SSD rescoring.
   - (c) DiskANN on NVMe.
   The model computes memory, SSD, node count, and monthly cost.
2. A prototype at 10–50M vectors for at least designs (b) and (c), measuring recall/QPS/P99 and extrapolating with a stated scaling assumption.
3. Validates the model's predictions against the measurements within ±25%, or explains the deviation.
4. Recommendation plus a failure-mode analysis (node loss, reindexing time, cold start).

**Step-by-step**
1. Write the formulas: vector bytes, graph bytes, PQ bytes, replication factor, and headroom.
2. Build the prototypes with FAISS (IVF-PQ + a memory-mapped rescoring store) and DiskANN (Microsoft's `diskannpy`) or an engine that implements it.
3. Load test with the 01.02 §9 load-tester pattern (Poisson arrivals).
4. Compare measured vs modelled numbers, and adjust the model.
5. Write the recommendation, including the operational runbook items.

---

## 10. Foundational Papers (exact titles)

**Theory and hashing**
- Beyer et al., 1999 — *When Is "Nearest Neighbor" Meaningful?*
- Indyk & Motwani, 1998 — *Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality*
- Charikar, 2002 — *Similarity Estimation Techniques from Rounding Algorithms*
- Shrivastava & Li, 2014 — *Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)*
- Bachrach et al., 2014 — *Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product Spaces*

**Quantization**
- Jégou, Douze & Schmid, 2011 — *Product Quantization for Nearest Neighbor Search*
- Ge et al., 2013 — *Optimized Product Quantization for Approximate Nearest Neighbor Search*
- Guo et al., 2020 — *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*
- Gao & Long, 2024 — *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search*

**Graphs**
- Malkov & Yashunin, 2018 — *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*
- Fu et al., 2019 — *Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph*
- Subramanya et al., 2019 — *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*
- Singh et al., 2021 — *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search*
- Chen et al., 2021 — *SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search*
- Ootomo et al., 2023 — *CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs*

**Filtered search**
- Gollapudi et al., 2023 — *Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters*
- Patel et al., 2024 — *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data*

**Systems and benchmarks**
- Johnson, Douze & Jégou, 2017 — *Billion-scale similarity search with GPUs*
- Douze et al., 2024 — *The Faiss library*
- Aumüller, Bernhardsson & Faithfull, 2018 — *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms*
- Simhadri et al., 2022 — *Results of the NeurIPS'21 Challenge on Billion-Scale Approximate Nearest Neighbor Search*

## 11. Essential Tooling

| Tool | Role |
|---|---|
| **FAISS** (CPU/GPU) | Reference implementations: Flat, IVF, PQ, OPQ, HNSW, fast-scan, index factory |
| **hnswlib** | Fast, minimal HNSW with filtering callbacks |
| **ScaNN** | Anisotropic quantization + tree search |
| **DiskANN / diskannpy** | SSD-resident graph indexes |
| **NVIDIA cuVS** (CAGRA, IVF-PQ) | GPU ANN |
| **ann-benchmarks**, **big-ann-benchmarks**, **VectorDBBench** | Standard datasets and harnesses |
| **numba / Rust (PyO3)** | Speeding up your own implementations |
| **perf / py-spy / VTune** | Profiling cache misses and hot loops |
