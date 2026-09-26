# Linear Algebra — Master Study Guide

> **Track:** Foundations · **Module:** 07
> **Prerequisites:** Comfortable with algebra; some calculus helps.
> **Time budget:** ~25–35 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every serious ML technique is at heart a linear-algebra operation. Neural networks are chained matrix multiplications with non-linear pinches in between. PCA, embeddings, regression, transformers' attention, SVD-based recommender systems — all speak the same language of vectors, matrices, and their decompositions. If you can't visualize what a matrix does to a vector, you'll experience ML as magic; if you can, ML becomes engineering.

Learning linear algebra "properly" for ML/data work means you can:

1. **Compute** — dot products, matrix products, norms, inverses (via NumPy/SciPy).
2. **Interpret** — a matrix as a linear map, an eigenvector as an "unchanging direction," a rank as an intrinsic dimensionality.
3. **Decompose** — understand what eigen-decomposition, SVD, and QR give you and when to reach for each.
4. **Debug** — a shape error, a singular matrix, or a convergence failure through a linear-algebra lens.

**Fundamental principles you must own:**

1. **Matrices are functions.** Multiplying by an $m \times n$ matrix maps a vector in $\mathbb{R}^n$ to one in $\mathbb{R}^m$.
2. **Vectors have geometry.** Length is a norm, angle is a dot product, distance is a norm of a difference.
3. **Basis and coordinates are separable.** The same geometric vector has different coordinates in different bases; changing basis is $B^{-1} A B$.
4. **Rank is what matters most.** Rank tells you the true dimensionality of the mapping, the size of the column space, and how much information the matrix loses.
5. **Everything reduces to solving $Ax = b$ or a least-squares approximation to it.**

If you retain nothing else: **do the operations by hand on small matrices before typing them into NumPy.** Manual computation once carves the intuition; NumPy after that becomes verification.

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Vectors, Norms, and the Dot Product

**Vectors.** An element of $\mathbb{R}^n$ is a column of $n$ real numbers.

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

Where:
- **$v_i$** = the $i$-th coordinate.
- $\mathbf{v}$ (bold) is a vector; $v_i$ (scalar) is a coordinate.

**Norms.** A norm measures "size." The most common:

- **L2 (Euclidean) norm:**

  $$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$$

- **L1 (Manhattan) norm:**

  $$\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$$

- **L∞ (max) norm:**

  $$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

**Dot product / inner product.**

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\|_2 \, \|\mathbf{v}\|_2 \cos\theta$$

Where **$\theta$** is the angle between them. Geometric meaning: how much of $\mathbf{u}$ points along $\mathbf{v}$.

**Orthogonality.** $\mathbf{u} \perp \mathbf{v} \iff \mathbf{u} \cdot \mathbf{v} = 0$.

**Cosine similarity** (used everywhere in NLP embeddings):

$$\text{cos\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \, \|\mathbf{v}\|_2}$$

Bounded in $[-1, 1]$, invariant to vector magnitude.

**Real-world example.** In a search system with word embeddings, ranking documents by cosine similarity to a query vector is dot-product on unit-normalized vectors — a single line of NumPy.

---

### 2.2 Matrices as Linear Maps

**Matrix-vector product.** For $A \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$:

$$(A\mathbf{x})_i = \sum_{j=1}^n A_{ij} x_j$$

Two equivalent interpretations:

- **Row picture:** each output entry is the dot product of a row of $A$ with $\mathbf{x}$.
- **Column picture:** the output is a **linear combination of the columns of $A$** with coefficients from $\mathbf{x}$:

  $$A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$

The column picture is the more useful mental model — it explains why the **column space** of $A$ is the set of all possible $A\mathbf{x}$.

**Matrix-matrix product.** For $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$:

$$(AB)_{ij} = \sum_{p=1}^k A_{ip} B_{pj}$$

**$AB$ is a composition of linear maps**: first apply $B$, then $A$. Not commutative: $AB \neq BA$ in general.

**Special matrices:**

- **Identity** $I$: $I \mathbf{x} = \mathbf{x}$.
- **Diagonal** $D$: entries only on the diagonal; scales each coordinate independently.
- **Orthogonal** $Q$: $Q^\top Q = I$; rotation/reflection; preserves lengths.
- **Symmetric** $S = S^\top$: real eigenvalues, orthonormal eigenvectors.
- **Positive definite (PD)** $A$: $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq 0$; all eigenvalues > 0. Covariance matrices of full-rank data are PD.

**Transpose.** $(A^\top)_{ij} = A_{ji}$. $(AB)^\top = B^\top A^\top$.

**Inverse.** $A A^{-1} = A^{-1} A = I$; exists iff $A$ is square and full rank ($\det A \neq 0$). Computationally you rarely form $A^{-1}$; you solve $A\mathbf{x} = \mathbf{b}$ directly.

---

### 2.3 Rank, Nullspace, and the Fundamental Subspaces

Given $A \in \mathbb{R}^{m \times n}$, there are **four fundamental subspaces**:

- **Column space** $\mathcal{C}(A) \subset \mathbb{R}^m$ — spanned by columns; the image of the map.
- **Nullspace** $\mathcal{N}(A) \subset \mathbb{R}^n$ — $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$; the kernel.
- **Row space** $\mathcal{C}(A^\top) \subset \mathbb{R}^n$.
- **Left nullspace** $\mathcal{N}(A^\top) \subset \mathbb{R}^m$.

**Rank** = dimension of the column space = dimension of the row space.

**Rank-nullity theorem:**

$$\text{rank}(A) + \dim \mathcal{N}(A) = n$$

Where **$n$** is the number of columns.

**Why rank matters:**

- $A\mathbf{x} = \mathbf{b}$ has a solution iff $\mathbf{b} \in \mathcal{C}(A)$.
- The solution is unique iff $\mathcal{N}(A) = \{\mathbf{0}\}$ (i.e., $A$ is one-to-one; columns are linearly independent).
- **Low-rank matrices** compress well — SVD lets you replace $A$ with $\tilde A$ of rank $r$ minimizing $\|A - \tilde A\|_F$.

**Real-world example.** A dataset $X \in \mathbb{R}^{n \times d}$ with rank $r < d$ has multicollinearity — some feature is a linear combination of others, and $X^\top X$ is singular. Ridge regression adds $\lambda I$ to make it invertible.

---

### 2.4 Eigenvalues, Eigenvectors, and Diagonalization

**Definition.** $\lambda \in \mathbb{C}$ is an **eigenvalue** and $\mathbf{v} \neq \mathbf{0}$ an **eigenvector** of $A$ if:

$$A\mathbf{v} = \lambda \mathbf{v}$$

Where:
- **$A$** = a square matrix.
- **$\mathbf{v}$** = a direction that $A$ merely scales.
- **$\lambda$** = the scaling factor.

**Characteristic polynomial:** $\det(A - \lambda I) = 0$ yields eigenvalues.

**Diagonalization.** If $A$ has $n$ linearly independent eigenvectors, then

$$A = P D P^{-1}$$

where **$P$** has eigenvectors as columns and **$D$** is diagonal with eigenvalues. This lets you compute $A^k = P D^k P^{-1}$ in $O(n^2)$ once $P, D$ are known — the "closed form" for matrix powers.

**Symmetric case (spectral theorem).** If $A = A^\top$, then $A = Q \Lambda Q^\top$ with orthogonal $Q$. Every symmetric matrix has real eigenvalues and orthonormal eigenvectors — an unusually clean world.

**Applications in ML:**

- **PCA** — eigen-decompose the covariance matrix; top eigenvectors are principal directions.
- **PageRank** — the dominant eigenvector of a link matrix.
- **Spectral clustering** — top eigenvectors of a Laplacian.
- **Stability analysis** — eigenvalues of a Jacobian tell you about local behavior of a dynamical system.

**Power iteration** (how NumPy actually computes the top eigenvector):

$$\mathbf{v}_{k+1} = \frac{A \mathbf{v}_k}{\|A \mathbf{v}_k\|_2}$$

Repeat; $\mathbf{v}_k$ converges to the dominant eigenvector.

---

### 2.5 SVD, Least Squares, and PCA

**Singular Value Decomposition (SVD).** For any $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^\top$$

Where:
- **$U \in \mathbb{R}^{m \times m}$** — orthogonal (left singular vectors).
- **$\Sigma \in \mathbb{R}^{m \times n}$** — diagonal (singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$).
- **$V \in \mathbb{R}^{n \times n}$** — orthogonal (right singular vectors).

Every matrix has an SVD, even non-square and singular ones. This is the Swiss army knife of numerical linear algebra.

**Geometric interpretation.** Any linear map decomposes as: rotate (via $V^\top$), scale each axis (via $\Sigma$), rotate (via $U$).

**Least squares.** For overdetermined $A\mathbf{x} \approx \mathbf{b}$ (more equations than unknowns), the least-squares solution minimizes $\|A\mathbf{x} - \mathbf{b}\|_2^2$:

$$\mathbf{x}^* = (A^\top A)^{-1} A^\top \mathbf{b}$$

The **pseudo-inverse** $A^+ = V \Sigma^+ U^\top$ (where $\Sigma^+$ is the reciprocal of nonzero singular values) gives the same solution in a numerically stable way:

$$\mathbf{x}^* = A^+ \mathbf{b}$$

**PCA via SVD.** For a centered data matrix $X \in \mathbb{R}^{n \times d}$ (rows = samples), the principal components are the right singular vectors:

$$X = U \Sigma V^\top, \quad \text{PC}_k = V_{:,k}$$

The **explained variance** of the $k$-th component is:

$$\text{Var}(\text{PC}_k) = \frac{\sigma_k^2}{n - 1}$$

Reducing to top-$k$ components:

$$X_k = U_{:,1:k} \Sigma_{1:k, 1:k} V_{:,1:k}^\top$$

This is the **rank-$k$ best approximation** of $X$ (Eckart–Young theorem).

**Real-world example.** Compress a 512×512 grayscale image ($262{,}144$ values) with SVD to rank 50: keep only $512 \times 50 + 50 + 50 \times 512 = 51{,}250$ values — a 5× reduction with visually decent reconstruction.

---

## 3. Mental Models & Analogies

### 3.1 The "Transformer of Space" Model (Matrices as Maps)

Imagine a rubber sheet of graph paper representing $\mathbb{R}^2$. A $2 \times 2$ matrix $A$ is a **transformation** you apply to that sheet:

- **Diagonal $\begin{bmatrix}2&0\\0&3\end{bmatrix}$** stretches by 2 horizontally, 3 vertically.
- **Rotation $\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$** spins the sheet by angle $\theta$.
- **Shear $\begin{bmatrix}1&k\\0&1\end{bmatrix}$** tilts vertical lines by slope $k$.
- **Singular matrix** $\begin{bmatrix}1&2\\2&4\end{bmatrix}$ **collapses the plane onto a line** — rank 1, everything perpendicular to the "collapse direction" is lost. The nullspace is that lost direction.

Eigenvectors are the **directions on the sheet that don't rotate** — they only stretch (by $\lambda$). If $\lambda < 0$, they flip. If $\lambda = 0$, they collapse to zero (nullspace direction).

**SVD**, then, is the statement that *every* transformation is a rotation + axis-scaling + rotation. Even the ugliest matrix is three clean operations in a trench coat.

### 3.2 The "Recipe Composition" Model (Basis Change and Coordinates)

A vector is a *physical* thing — an arrow in space. Its **coordinates** are just how you describe that arrow given a specific basis (a coordinate system).

Think of a recipe:

- The **dish** is the physical vector.
- The **measurement system** (metric, imperial, "handfuls") is the basis.
- The **numbers on the recipe card** are the coordinates.

The dish doesn't change if you translate the recipe from cups to milliliters — you just multiply by a **conversion matrix** ($B^{-1}$). Similarly, a **change of basis** transforms coordinates via $\mathbf{x}' = B^{-1} \mathbf{x}$.

The best basis is often the one that makes your problem diagonal. **Eigen-decomposition finds a basis in which $A$ becomes a diagonal scaling.** PCA finds the basis in which the data's covariance is diagonal (uncorrelated components).

![IMG-NP-01](/1%20—%20Foundations/images/IMG-LA-01.png)
> **Caption:** A matrix moves the basis vectors; the transformed vector is the same linear combination in the new basis.
> **Placement:** Section 2.2.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Matrix Multiplication Is Commutative"

$AB \neq BA$ in general. Concretely: rotating then scaling ≠ scaling then rotating. Sometimes even the shapes forbid one side. Also: $AB = 0$ does **not** imply $A = 0$ or $B = 0$ — a singular matrix can send another matrix to zero (e.g., project onto the kernel of the other).

### 4.2 "You Should Invert $A$ to Solve $A\mathbf{x} = \mathbf{b}$"

Never compute `np.linalg.inv(A) @ b`. Instead, `np.linalg.solve(A, b)`. `solve` uses LU factorization (or QR/Cholesky for special structure) which is:

- **Numerically stable** — inverses amplify rounding errors.
- **Faster** — no wasted work computing the full inverse.
- **Handles rank-deficient cases** more gracefully (via `lstsq` or the pseudo-inverse).

Corollary: if you find yourself writing `X.T @ X` and inverting it for regression, use `np.linalg.lstsq(X, y)` instead.

### 4.3 "PCA on Un-standardized Data Is Fine"

PCA maximizes variance along its principal axes. If one feature ranges 0–1 and another ranges 0–1,000,000, the second **dominates** the principal components even if it's meaningless noise. Standardize (subtract mean, divide by std) unless you have a specific reason not to. Ditto for anything that uses Euclidean distance — k-means, KNN, kernel methods.

---

## 5. Self-Assessment Bank (Linear Algebra)

### Questions

**Q1 (Short answer).** State the dot product formula and its geometric meaning.

**Q2 (Multiple choice).** For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $AB$ has shape:
A. $m \times n$
B. $n \times p$
C. $m \times p$
D. $p \times m$

**Q3 (Short answer).** What does "rank" of a matrix mean? What is the rank of $\begin{bmatrix}1&2\\2&4\end{bmatrix}$?

**Q4 (Multiple choice).** Which of the following matrices is **not** invertible?
A. Identity $I$.
B. Any orthogonal matrix.
C. A matrix with a zero row.
D. A diagonal matrix with all nonzero entries.

**Q5 (Short answer).** Explain what an eigenvector is in one sentence and why symmetric matrices have particularly nice ones.

**Q6 (Multiple choice).** For the least-squares problem $A\mathbf{x} \approx \mathbf{b}$ (overdetermined), the normal equation is:
A. $A\mathbf{x} = \mathbf{b}$
B. $A^\top A \mathbf{x} = A^\top \mathbf{b}$
C. $A A^\top \mathbf{x} = A A^\top \mathbf{b}$
D. $\mathbf{x} = A^{-1} \mathbf{b}$

**Q7 (Short answer).** Given the SVD $A = U \Sigma V^\top$, how do you construct the rank-$k$ best approximation of $A$?

**Q8 (Multiple choice).** Why should you use `np.linalg.solve(A, b)` instead of `np.linalg.inv(A) @ b`?
A. It's shorter to type.
B. It's more numerically stable and faster.
C. `inv` doesn't exist.
D. `solve` handles complex numbers automatically.

**Q9 (Short answer).** In PCA, why should features usually be standardized (zero mean, unit variance) before applying the decomposition?

**Q10 (Multiple choice).** The nullspace of $A$ is:
A. The set of vectors $\mathbf{b}$ such that $A\mathbf{x} = \mathbf{b}$ has a solution.
B. The set of vectors $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{0}$.
C. The set of eigenvectors of $A$.
D. The set of orthogonal columns of $A$.

---

### Answer Key & Detailed Explanations

**A1.** $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$. Geometrically, it measures how much of $\mathbf{u}$ points along $\mathbf{v}$, scaled by their lengths.

**A2. C.** Inner dimensions must match ($n$), and the result is (outer × outer) = $m \times p$.

**A3.** The rank of a matrix is the dimension of the space spanned by its columns (equivalently, its rows) — the true dimensionality of the linear map. For $\begin{bmatrix}1&2\\2&4\end{bmatrix}$, the second row is 2× the first — the rows are linearly dependent, so rank = 1.

**A4. C.** A zero row means one row of $A$ is $\mathbf{0}^\top$, so the columns must fail to span $\mathbb{R}^n$ — the matrix is singular. Identity, orthogonal, and diagonal-with-nonzero-entries matrices are all invertible.

**A5.** An eigenvector is a nonzero vector that a matrix merely scales without rotating: $A\mathbf{v} = \lambda \mathbf{v}$. Symmetric matrices have real eigenvalues and a **full set of orthonormal eigenvectors** (the spectral theorem), which lets us diagonalize with $A = Q \Lambda Q^\top$ (orthogonal $Q$).

**A6. B.** $A^\top A \mathbf{x} = A^\top \mathbf{b}$ is the normal equation. Deriving: minimize $\|A\mathbf{x} - \mathbf{b}\|^2$; setting the gradient with respect to $\mathbf{x}$ to zero yields $2A^\top(A\mathbf{x} - \mathbf{b}) = 0$.

**A7.** Keep the top $k$ singular values and their corresponding columns:

$$A_k = U_{:,1:k} \, \Sigma_{1:k, 1:k} \, V_{:,1:k}^\top$$

By the Eckart–Young theorem, this is the best rank-$k$ approximation in both Frobenius and spectral norms.

**A8. B.** `solve` uses LU factorization (or Cholesky/QR when applicable), which is $O(n^3)$ but *much* better conditioned than forming $A^{-1}$ explicitly. Forming an inverse costs the same asymptotically but amplifies rounding errors, especially for ill-conditioned matrices.

**A9.** PCA maximizes variance along orthogonal axes. If features have wildly different scales (e.g., "salary" in thousands vs "years of education" 0–20), the high-variance feature dominates every principal component regardless of its actual informativeness. Standardization forces every feature to contribute on equal footing.

**A10. B.** The nullspace is the set of inputs mapped to zero — the kernel of the map. Its dimension plus the rank equals $n$ (rank–nullity).

---

## 6. Practice Prompts

1. By hand: multiply $\begin{bmatrix}1&2\\3&4\end{bmatrix} \begin{bmatrix}5\\6\end{bmatrix}$. Then verify with NumPy. Then swap operands and note the shape error.
2. In NumPy, take a 100×3 dataset (make one up), center it, compute its covariance matrix, and find its eigenvalues/eigenvectors. Compare to `np.linalg.svd` on the centered data.
3. Compress a grayscale image via SVD. Plot reconstruction error vs number of retained singular values on a log-log plot.
4. Solve a least-squares regression by (a) `np.linalg.solve(X.T @ X, X.T @ y)`, (b) `np.linalg.lstsq(X, y)`, and (c) `np.linalg.pinv(X) @ y`. Compare numerically for a moderately ill-conditioned matrix.
5. Verify the Eckart–Young theorem empirically: take a random $100 \times 100$ matrix, compute its rank-$k$ SVD approximation for $k = 1, \ldots, 100$, and plot $\|A - A_k\|_F$ vs $k$.

---

## 7. References

- Gilbert Strang, *Introduction to Linear Algebra* — the friendliest textbook.
- Gilbert Strang's MIT 18.06 lectures (free on OCW/YouTube).
- 3Blue1Brown, *Essence of Linear Algebra* — the animated intuition source.
- Trefethen & Bau, *Numerical Linear Algebra* — for the numerical side (SVD, QR, conditioning).
- Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* (free PDF).
