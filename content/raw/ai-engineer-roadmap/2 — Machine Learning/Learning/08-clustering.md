# Clustering (Unsupervised Learning) — Master Study Guide

> **Track:** Machine Learning · **Module:** 08
> **Prerequisites:** KNN (Module 07) helps with distance intuition.
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Clustering is your **exploratory workhorse** for unsupervised problems: finding structure in data without labels. Practical uses:

- **Customer segmentation** for marketing.
- **Anomaly detection** — points that don't fit any cluster.
- **Feature engineering** — cluster IDs as new features.
- **Data compression / quantization** — represent points by their cluster centroid.
- **Semi-supervised learning** — cluster first, then label a few representatives per cluster.

Unlike classification, there is **no ground truth** — no answer key. This makes clustering both freeing (you can discover structure) and treacherous (you can find "structure" that isn't real). Interpretation, domain knowledge, and honest evaluation are essential.

**Fundamental principles you must own:**

1. **Clustering finds groups such that within-group similarity is high and between-group similarity is low.**
2. **What "similar" means depends on the distance metric and feature scaling** — bad scaling ruins clustering.
3. **Different algorithms have different biases** — K-means assumes convex spherical clusters; DBSCAN doesn't; hierarchical clustering builds a whole tree of possibilities.
4. **Choosing $k$** (number of clusters) is often the hardest problem, and there's no universally right answer.
5. **Evaluation is tricky without labels** — internal metrics (silhouette, Davies-Bouldin) are heuristics; external metrics require some ground truth.
6. **Clustering results are highly sensitive to preprocessing** — scaling, dimensionality reduction, distance choice.

If you retain nothing else: **clustering is opinionated — your algorithm and preprocessing choices *are* the clusters. Never present clusters as "objective structure in the data."**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 K-Means and Its Variants

**K-means** partitions $n$ points into $k$ clusters by minimizing **within-cluster sum of squares (WCSS)**:

$$\min_{C_1, \ldots, C_k, \boldsymbol\mu_1, \ldots, \boldsymbol\mu_k} \sum_{j=1}^k \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \boldsymbol\mu_j\|_2^2$$

Where:
- **$C_j$** = the set of points assigned to cluster $j$.
- **$\boldsymbol\mu_j$** = the centroid (mean) of cluster $j$.

**Lloyd's algorithm** (the standard K-means solver):

1. **Initialize** $k$ centroids (randomly, or via K-means++).
2. **Assign** each point to the nearest centroid: $C_j = \{\mathbf{x}_i : j = \arg\min_l \|\mathbf{x}_i - \boldsymbol\mu_l\|\}$.
3. **Update** each centroid to the mean of its assigned points: $\boldsymbol\mu_j = \frac{1}{|C_j|}\sum_{\mathbf{x} \in C_j} \mathbf{x}$.
4. Repeat 2–3 until assignments don't change (or a max-iter is hit).

**Guarantees:** WCSS decreases monotonically. K-means converges to a local minimum in finite steps. But the *global* minimum is not guaranteed — different initializations give different results.

**K-means++** is a smarter initialization: pick the first centroid uniformly at random; pick each subsequent centroid with probability proportional to squared distance from the nearest already-picked centroid. Reduces the chance of a bad local minimum. Default in scikit-learn.

**Complexity.** $O(n \cdot k \cdot p \cdot I)$ per run, where $I$ is number of iterations. Fast for millions of points.

**Assumptions and failure modes:**
- **Convex, roughly spherical clusters.** Fails on elongated, curved, or nested clusters.
- **Similar-sized clusters.** Fails when one cluster is 10× bigger than another (K-means splits big clusters).
- **Similar-density clusters.** Fails when clusters have very different variances.
- **Numeric features.** Categorical features need embedding or K-modes/K-prototypes instead.

**Variants:**
- **Mini-batch K-means** — stochastic updates on random batches; scales to huge datasets.
- **Bisecting K-means** — recursively split largest cluster.
- **K-medoids (PAM)** — centroids must be actual points (robust to outliers).
- **K-modes / K-prototypes** — for categorical or mixed data.

---

### 2.2 Hierarchical Clustering

**Agglomerative** (bottom-up): start with every point as its own cluster; repeatedly merge the two most similar clusters until only one remains.

**Divisive** (top-down): start with all points in one cluster; recursively split. Less common in practice.

**Result:** a **dendrogram** — a tree showing all the merges (or splits). You cut the tree at a chosen height to get a specific number of clusters.

**Linkage functions** define the distance between two clusters $A$ and $B$:

- **Single linkage:** $d(A, B) = \min_{a \in A, b \in B} d(a, b)$. Sensitive to chaining (elongated clusters).
- **Complete linkage:** $d(A, B) = \max_{a \in A, b \in B} d(a, b)$. Compact clusters; sensitive to outliers.
- **Average linkage:** $d(A, B) = \frac{1}{|A||B|}\sum_{a, b} d(a, b)$. Balanced.
- **Ward linkage:** merge the pair that minimizes the increase in total WCSS. Produces balanced, K-means-like clusters. Common default.

**Complexity.** Naïvely $O(n^3)$; efficient implementations reach $O(n^2 \log n)$. Doesn't scale to millions.

**Advantages:**
- Doesn't require pre-choosing $k$ — inspect the dendrogram.
- Handles non-spherical clusters better than K-means (with single/average linkage).
- Deterministic (no random initialization).

**Disadvantages:**
- Slow.
- No global objective is optimized (except Ward, which optimizes a K-means-like criterion locally).
- Merges are greedy and irreversible.

---

### 2.3 DBSCAN and Density-Based Methods

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) groups points that are densely packed together and marks isolated points as noise. Two parameters:

- **`eps`** — the radius of the neighborhood.
- **`min_samples`** — minimum points to form a "dense" region.

**Point types:**
- **Core point** — has ≥ `min_samples` points within `eps`.
- **Border point** — has < `min_samples` in its neighborhood but is within `eps` of a core point.
- **Noise** — neither core nor border.

**Algorithm:**
1. For each unvisited point, mark it visited.
2. If it's a core point, start a new cluster; add all its density-reachable points to the cluster.
3. If it's a border, join the neighboring core's cluster.
4. Otherwise, mark it as noise.

**Advantages:**
- Discovers **arbitrarily shaped** clusters.
- Doesn't require $k$ in advance.
- **Robust to outliers** (they become noise).

**Disadvantages:**
- Sensitive to `eps` and `min_samples`.
- Struggles with **variable density** — a single `eps` can't handle both dense and sparse clusters.
- Curse of dimensionality — distance-based, same issues as KNN.

**HDBSCAN** (Hierarchical DBSCAN) — chooses `eps` adaptively per cluster. Modern favorite; more robust to hyperparameter choice.

**Choosing `eps`.** Plot the sorted $k$-distance graph (distance to $k$-th nearest neighbor for each point); look for the "elbow." Or use HDBSCAN.

---

### 2.4 Choosing k and Evaluating Clusters

**Elbow method:**
- Fit K-means for $k = 1, 2, \ldots, K_{\max}$.
- Plot WCSS (inertia) vs $k$.
- Look for a "kink" where the marginal drop levels off.
- Subjective — different eyes see different elbows.

**Silhouette score** for each point $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- **$a(i)$** = mean distance from $i$ to points in its own cluster (cohesion).
- **$b(i)$** = mean distance from $i$ to points in the *nearest other* cluster (separation).
- **$s(i) \in [-1, 1]$**. Near 1: well-clustered. Near 0: on the boundary. Negative: possibly in the wrong cluster.

The **average silhouette** over all points is an overall score. Choose $k$ that maximizes it.

**Other internal metrics:**
- **Davies-Bouldin index** — lower is better. Ratio of within-cluster to between-cluster distances.
- **Calinski-Harabasz** — higher is better. Ratio of between-cluster to within-cluster variance.
- **Gap statistic** — compares WCSS to that under a null (uniform) reference distribution.

**External metrics** (require ground-truth labels — usually for validating an algorithm on labeled benchmarks):
- **Adjusted Rand Index (ARI)** — similarity to true labels, adjusted for chance. In [-1, 1]; 0 = random; 1 = perfect.
- **Normalized Mutual Information (NMI)** — information overlap between clusterings.

**Reality check.** No metric replaces **domain interpretation**. Look at the clusters. Do they make sense? Can you name them? Are the differences between them actionable?

---

### 2.5 Preprocessing & Practical Wisdom

**Scaling is essential.** Distance-based clusterings + unscaled features = the largest-range feature dictates the clusters. Always scale unless you have a *specific* reason not to.

**Dimensionality reduction before clustering:**
- **PCA** — linear; use to strip noise, preserve variance.
- **UMAP / t-SNE** — nonlinear; often produces visually clean clusters in 2D, but distances in the reduced space are distorted. Great for visualization; be careful about clustering directly on t-SNE outputs.
- **Autoencoders / neural embeddings** — task-specific low-dim representations.

**Categorical features:**
- **K-modes** for pure categorical.
- **K-prototypes** for mixed numeric + categorical.
- Or **embed** with target encoding / model-based embeddings and use K-means.

**Handling imbalance and rare structure:**
- K-means will merge tiny clusters into big ones. Use DBSCAN/HDBSCAN if rare clusters matter.
- For anomaly detection, treat DBSCAN's noise points as anomalies.

**Reproducibility:**
- K-means is randomized (initialization). Fix `random_state` and run multiple `n_init` (scikit-learn does 10 by default).
- Hierarchical is deterministic given inputs.

**Interpretation:**
- Compute cluster **centroids** in original (unscaled) feature space for readability.
- Compute **feature distributions per cluster** to characterize what each cluster "is."
- Name clusters based on interpretation, not just numbers.

**Reporting:**
- Sample sizes per cluster.
- Silhouette score.
- Feature summary per cluster (mean/median/mode).
- A one-line "identity" of each cluster.

---

## 3. Mental Models & Analogies

### 3.1 The "Party Grouping" Model

Imagine a big party. People naturally cluster by conversation topic:

- **K-means** is the host walking around who assigns each guest to the nearest of $k$ pre-set "topic centroids" (say: sports, work, food, family). Guests near a topic centroid get sorted into that group. If someone is between two topics, they get pulled toward the closest one — even if they're not really "in" either.

- **Hierarchical** is the reverse: initially every guest is their own tiny group; then you repeatedly merge the two most similar people (or groups) — first close friends, then acquaintances, up to eventually one giant blob. You can cut the merges at any level to get $k$ groups.

- **DBSCAN** is a wanderer looking at density: any dense knot of guests is a cluster. People standing alone in the corner are noise, no matter how "close" they might be to something in a metric sense.

Each gives a different answer to "what are the groups?" — and that's the point. Groups aren't objective properties of a party; they're a perspective the algorithm imposes.

### 3.2 The "Map of Neighborhoods" Model

Think of feature space as a city map. Each data point is a household. Clustering is drawing **neighborhood boundaries**:

- **K-means** = drawing Voronoi lines from $k$ "town halls." Every house belongs to the nearest hall. Boundaries are straight lines. Every neighborhood is convex.
- **Hierarchical** = ZIP codes nested inside cities nested inside metros. A hierarchy of ever-larger neighborhoods.
- **DBSCAN** = following the density of houses along the map. Where houses cluster tightly, that's a neighborhood. Isolated houses aren't in any neighborhood (they're rural — "noise").

Just as a real map's neighborhoods depend on the mapmaker's purpose (school districts, delivery routes, real estate markets), your clustering depends on your algorithm and preprocessing.


![IMG-RF-01](/2%20—%20Machine%20Learning/images/IMG-CL-01.jpg)
> **Caption:** Different algorithms have different biases; the "right" clustering depends on your assumptions about cluster shape and density.
> **Placement:** Section 2.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "K-Means Found the True Clusters"

There are no "true" clusters — only useful ones. K-means will *always* return $k$ clusters, no matter what the data actually contains. Uniform noise in $[0, 1]^2$ with $k=5$ yields 5 clusters that look plausible until you realize they're arbitrary. Always sanity-check with:
- Try multiple $k$ and multiple algorithms.
- Look at silhouette per-point, not just mean.
- Interpret in domain terms.

### 4.2 "Silhouette Score Tells Me the Right k"

Silhouette rewards well-separated, compact clusters. It **strongly favors** K-means's biases (spherical, similar-size clusters). It **fails** on:
- Nested / non-convex clusters (silhouette will call the "wrong" $k$).
- Hierarchical structure (there may be no single right $k$).
- Very-high-dimensional data (all distances concentrate).

Use silhouette as *one* signal, not the answer.

### 4.3 "Clustering Doesn't Have Leakage Because There's No Target"

You can absolutely leak information into a clustering. Examples:
- **Scaling on the whole dataset** including future test data — the mean and std you'd normally compute at train time bleed into test.
- Using clustering **as a feature** in a downstream supervised model without wrapping in a Pipeline that refits per fold.
- Selecting $k$ or preprocessing choices based on how "reasonable" a downstream supervised model's performance is — that's implicit label peeking.

Treat clustering like any other model — fit inside the appropriate CV loop when downstream evaluation matters.

---

## 5. Self-Assessment Bank (Clustering)

### Questions

**Q1 (Short answer).** What quantity does K-means minimize? Under what assumptions does the algorithm work well?

**Q2 (Multiple choice).** K-means++ is:
A. A different loss function.
B. A smarter initialization strategy for centroids.
C. A version that handles categorical features.
D. A hierarchical variant.

**Q3 (Short answer).** Describe the elbow method for choosing $k$ and one of its weaknesses.

**Q4 (Multiple choice).** Which method does NOT require you to specify $k$ in advance?
A. K-means
B. K-medoids
C. DBSCAN
D. Mini-batch K-means

**Q5 (Short answer).** Explain the three point types (core, border, noise) in DBSCAN.

**Q6 (Multiple choice).** For non-convex "moon-shaped" clusters, the best-performing method is likely:
A. K-means.
B. Agglomerative with average linkage or DBSCAN.
C. Bisecting K-means.
D. Any method — they all handle non-convex clusters.

**Q7 (Short answer).** Define the silhouette score of a point $i$ and interpret values close to +1, 0, and -1.

**Q8 (Multiple choice).** Failing to scale features before K-means results in:
A. Slower convergence but same clusters.
B. Clusters dominated by the feature with the largest range.
C. Perfectly balanced clusters.
D. Higher silhouette scores.

**Q9 (Short answer).** How would you evaluate a clustering when you have no ground-truth labels? Name two techniques.

**Q10 (Multiple choice).** A dendrogram is:
A. A graph of loss vs iteration in K-means.
B. A tree diagram of all agglomerative merges (or divisive splits).
C. A synonym for a silhouette plot.
D. A visualization of DBSCAN's density.

---

### Answer Key & Detailed Explanations

**A1.** K-means minimizes the **within-cluster sum of squares** (WCSS), also called inertia: $\sum_j \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol\mu_j\|^2$. It works well when clusters are **convex, roughly spherical, similar in size, and similar in density**, and when features are on comparable scales.

**A2. B.** K-means++ picks initial centroids probabilistically to spread them out — reducing the chance of a bad local minimum. It doesn't change the loss or algorithm structure otherwise.

**A3.** Fit K-means for a range of $k$ and plot WCSS vs $k$. Look for the "kink" where the marginal decrease in WCSS levels off — that's the elbow. **Weakness:** the elbow is subjective. On smooth curves without a clear kink, different observers see different $k$. Silhouette or gap statistic offer more objective signals.

**A4. C.** DBSCAN discovers clusters based on density; the number is a function of `eps` and `min_samples`, not specified directly. K-means / K-medoids / Mini-batch all require $k$.

**A5.** A **core** point has ≥ `min_samples` points within `eps` of it. A **border** point is within `eps` of a core but has < `min_samples` within its own `eps`-neighborhood. A **noise** point is neither — no core is within `eps` of it. Clusters are formed by connecting cores through their neighborhoods; borders join whichever adjacent core they're near; noise stays unassigned.

**A6. B.** K-means enforces convex boundaries and will misassign moon-shape crescents. DBSCAN or agglomerative with single/average linkage can trace along the crescents' local density / distance chains.

**A7.** $s(i) = (b(i) - a(i)) / \max(a(i), b(i))$, where $a(i)$ is mean distance within its cluster and $b(i)$ is mean distance to points in the nearest other cluster. **+1**: much closer to own cluster than to others — well clustered. **0**: on the boundary between two clusters. **-1**: closer to another cluster than to its own — possibly misassigned.

**A8. B.** With unscaled features, distances are dominated by the largest-range feature. K-means minimizes WCSS in that space, so the clusters end up aligned with the dominant feature — often meaningless.

**A9.** **Internal metrics** — silhouette score, Davies-Bouldin, Calinski-Harabasz, gap statistic. **Stability** — refit on bootstrap samples and check how consistent the clustering is. **Downstream utility** — use the cluster IDs as features in a supervised task and see if they help. **Visual inspection** — project via PCA/UMAP and see if clusters look meaningful. **Domain interpretation** — can you name each cluster and would a domain expert agree?

**A10. B.** A dendrogram shows the merging (or splitting) history in hierarchical clustering. The y-axis is the linkage distance at which each merge occurred; the x-axis is the individual data points. Cutting the dendrogram horizontally at a chosen height gives you a specific number of clusters.

---

## 6. Practice Prompts

1. **K-means from scratch.** Implement Lloyd's algorithm from scratch. Verify convergence on a synthetic 2D dataset. Compare to `sklearn.cluster.KMeans`.
2. **Elbow + silhouette.** On a customer-purchases dataset (or any real dataset), fit K-means for $k = 2, \ldots, 15$. Plot WCSS and silhouette. Compare their suggested $k$s.
3. **Curved clusters.** Generate the "two moons" dataset. Cluster it with K-means, agglomerative (Ward, single, average), and DBSCAN. Compare cluster assignments visually.
4. **Cluster-as-feature.** Fit K-means on a supervised dataset (using only features, not labels). Add cluster ID as a categorical feature. Does it help a downstream classifier?
5. **Dimensionality first.** On a high-dim dataset (e.g., 1000+ features), cluster raw vs after PCA to 20 dimensions. Compare quality (silhouette, downstream utility, and interpretability).

---

## 7. References

- Lloyd, "Least Squares Quantization in PCM" (1957, published 1982) — original K-means.
- Arthur & Vassilvitskii, ["k-means++: The Advantages of Careful Seeding"](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) (2007).
- Ester et al., ["A Density-Based Algorithm for Discovering Clusters"](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) (1996) — DBSCAN.
- Campello et al., ["Density-Based Clustering Based on Hierarchical Density Estimates"](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14) (2013) — HDBSCAN.
- ESL chapter 14.
- scikit-learn docs: [Clustering](https://scikit-learn.org/stable/modules/clustering.html).
