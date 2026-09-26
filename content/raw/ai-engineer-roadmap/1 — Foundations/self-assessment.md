# Foundations — Consolidated Self-Assessment

> **Scope:** All 8 learning modules **and** both builds.
> **Format:** Timed (2 hours), no reference material, notebook off.
> **Passing bar:** ≥ 80% overall **and** ≥ 60% in every section.

This is the exit exam for the Foundations track. Take it once when you *think* you're ready. Score honestly. Retake it 2 weeks later after fixing weak spots.

---

## Section A — Python (10 points)

**A1 (1 pt).** In one sentence: what does it mean that "variables in Python are names, not boxes"?

**A2 (1 pt, MC).** Which is a lazy iterable expression?
- (a) `[x**2 for x in range(10)]`
- (b) `{x**2 for x in range(10)}`
- (c) `(x**2 for x in range(10))`
- (d) `list(range(10))`

**A3 (2 pt).** Explain the bug and fix it:
```python
def add_tag(tag, tags=[]):
    tags.append(tag)
    return tags
```

**A4 (2 pt).** Write a decorator `@retry(times=3)` that retries the wrapped function up to `times` times on any `Exception`, sleeping 0.5 s between attempts, preserving the function's metadata.

**A5 (2 pt).** Explain the difference between a shallow and a deep copy, giving one example where the difference matters.

**A6 (2 pt).** Give one scenario where you'd reach for `asyncio`, one for threads, and one for `multiprocessing`.

---

## Section B — NumPy & Pandas (12 points)

**B1 (1 pt, MC).** Which shape pair broadcasts?
- (a) `(3, 4)` and `(2, 4)`
- (b) `(3, 1)` and `(1, 5)`
- (c) `(4,)` and `(3, 5)`
- (d) `(1, 3)` and `(4,)`

**B2 (2 pt).** Given `X = np.random.randn(1000, 5)`, write a one-liner that standardizes each column to zero mean and unit variance.

**B3 (2 pt).** Explain the difference between `.loc`, `.iloc`, and plain `[]` indexing on a DataFrame.

**B4 (2 pt).** What is a `SettingWithCopyWarning`, why does it happen, and how do you fix it correctly?

**B5 (3 pt).** Given `orders(user_id, amount, ts)`, write Pandas code that returns for each user their **7-day trailing sum of `amount`** at each `ts` — using `groupby` + `rolling`.

**B6 (2 pt).** You have 1 GB of CSV that won't fit in memory. Describe two Pandas techniques (naming APIs, not implementations) that let you still process it.

---

## Section C — SQL (12 points)

**C1 (1 pt, MC).** In which order does SQL logically evaluate `WHERE`, `GROUP BY`, `HAVING`, `SELECT`, `ORDER BY`?

**C2 (2 pt).** Write a query that returns the top-3 highest-paid employees **per department**, breaking ties alphabetically by name.

**C3 (2 pt).** Explain the difference between `INNER JOIN`, `LEFT JOIN`, and an "anti-join". Give a `WHERE` predicate that turns a `LEFT JOIN` into an effective `INNER JOIN`.

**C4 (2 pt).** You have a composite index on `(country, city, created_at)`. Which of these queries can use it?
- `WHERE country = 'US'`
- `WHERE city = 'Helena'`
- `WHERE country = 'US' AND created_at > NOW() - INTERVAL '7 days'`
- `WHERE country = 'US' AND city = 'Helena' AND created_at > NOW() - INTERVAL '7 days'`

**C5 (2 pt).** In one paragraph, contrast `SELECT COUNT(*) FROM t` with `SELECT COUNT(col) FROM t`.

**C6 (3 pt).** Write a recursive CTE that returns every ancestor of employee id 42 in an `employees(id, manager_id, name)` table.

---

## Section D — Git & GitHub (10 points)

**D1 (1 pt).** Name the four Git object types.

**D2 (2 pt).** Contrast `git reset --soft HEAD~1`, `git reset --mixed HEAD~1`, and `git reset --hard HEAD~1`.

**D3 (2 pt).** You accidentally committed a secret and pushed. Describe the correct response in three concrete steps.

**D4 (2 pt).** Explain what a fast-forward merge is, and when Git can perform one.

**D5 (3 pt).** Write a `.github/workflows/ci.yml` skeleton that on push checks out code, sets up Python 3.11, installs deps, runs `ruff check`, and runs `pytest`.

---

## Section E — APIs & JSON (10 points)

**E1 (1 pt).** What are the six JSON primitive/composite types?

**E2 (2 pt).** Explain the difference between `401 Unauthorized` and `403 Forbidden`.

**E3 (2 pt).** Give the formula for exponential backoff with full jitter and explain why jitter is critical.

**E4 (2 pt).** A `POST` request's response is dropped by the network. Should you retry? Under what condition is it safe?

**E5 (3 pt).** Design a REST-ish endpoint for "list a user's paid orders, sorted by creation time descending, paginated by cursor, returning only `id` and `amount`". Give the exact URL and query params.

---

## Section F — Linux & CLI (10 points)

**F1 (1 pt).** What does `set -euo pipefail` do?

**F2 (2 pt).** Write a Bash one-liner that prints the top-5 most-frequent HTTP status codes in `access.log` (status is column 9 in the default Nginx format).

**F3 (2 pt).** Explain the difference between a hard link and a symbolic link.

**F4 (2 pt).** What happens if you unquote a variable that contains spaces in a Bash command? Give a concrete example.

**F5 (3 pt).** Explain the difference between a Docker container and a virtual machine. What kernel does each use?

---

## Section G — Linear Algebra (10 points)

**G1 (1 pt).** For $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$, what is the shape of $AB$?

**G2 (2 pt).** State the rank-nullity theorem.

**G3 (2 pt).** Define an eigenvector and eigenvalue. Why is a symmetric matrix's spectrum especially clean?

**G4 (2 pt).** Give the least-squares normal equation and explain when it gives the same answer as `np.linalg.lstsq`.

**G5 (3 pt).** State the SVD ($A = U \Sigma V^\top$) and construct the rank-$k$ best approximation. Which theorem guarantees it's optimal in Frobenius norm?

---

## Section H — Probability & Statistics (12 points)

**H1 (1 pt).** State the Central Limit Theorem in one sentence.

**H2 (2 pt).** In the disease-test scenario (prevalence 0.1%, sensitivity 99%, specificity 99%), compute $P(\text{sick} \mid +)$ to 2 decimal places. Show the setup.

**H3 (2 pt).** Explain what a 95% confidence interval means. What does it **not** mean?

**H4 (2 pt).** You run 20 A/B tests at $\alpha = 0.05$, all with no true effect. What's the expected number of false positives, and name one correction.

**H5 (2 pt).** Describe the nonparametric bootstrap algorithm in ≤ 4 bullet points.

**H6 (3 pt).** Give the bias–variance decomposition of MSE and explain how it justifies regularization.

---

## Section I — Build Discipline (14 points)

**I1 (2 pt).** In one paragraph, explain why a data pipeline should be **idempotent** and give one anti-example.

**I2 (2 pt).** What is a data dictionary and what belongs in it?

**I3 (2 pt).** In a FastAPI service, where should model loading happen and why?

**I4 (2 pt).** Explain why a Docker image should have a non-root runtime user and how you'd add one to a Python image.

**I5 (3 pt).** Draft a `pyproject.toml` skeleton for an installable package that pins Python 3.11+, declares `fastapi`, `uvicorn`, and `pydantic` as runtime deps, and `pytest`, `ruff`, `mypy` as dev deps.

**I6 (3 pt).** Design a small **error envelope** JSON shape for your API and explain the fields.

---

# Answer Key & Detailed Explanations

## Section A

**A1.** In Python, an assignment binds a **name** to an existing object. Multiple names can point to the same object; mutating the object is visible through all names. There are no "boxes" holding values — only labels pointing at heap objects.

**A2.** **(c)** — a generator expression, evaluated on demand.

**A3.** The bug is the mutable default `tags=[]`, evaluated **once** at function definition time and shared across calls. Fix:
```python
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
```

**A4.** Full solution:
```python
import functools, time

def retry(times: int = 3, delay: float = 0.5):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < times:
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return deco
```

**A5.** **Shallow** copies the outer container; nested references are shared. **Deep** recursively copies. Matters for nested lists/dicts:
```python
a = [[1, 2]]
b = a.copy()      # shallow
b[0].append(9)
print(a)          # [[1, 2, 9]] — surprise, a mutated
```

**A6.** `asyncio` — many concurrent I/O-bound operations (network calls, DB queries) with low overhead. **Threads** — I/O-bound work where a synchronous library or blocking C extension is in play. **`multiprocessing`** — CPU-bound pure-Python work that must run truly in parallel, escaping the GIL.

---

## Section B

**B1.** **(b)** — `(3, 1)` and `(1, 5)` broadcast to `(3, 5)`. (a) fails (3 ≠ 2). (c) `(4,)` left-pads to `(1, 4)` — mismatch with `(3, 5)`. (d) `(4,)` left-pads to `(1, 4)` — mismatch with `(1, 3)`.

**B2.** `(X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)`.

**B3.** `.loc[i]` uses **row labels** from the Index. `.iloc[i]` uses **integer position**. `df[i]` with an integer looks in **column names** (not rows); with a string, returns that column; with a boolean mask, filters rows.

**B4.** `SettingWithCopyWarning` triggers when you assign to a chained-indexed view that Pandas can't tell was intended to modify the original. Fix: perform the operation in a single `.loc` call: `df.loc[mask, "col"] = value`.

**B5.**
```python
(
    orders
    .sort_values(["user_id", "ts"])
    .assign(seven_day=lambda d: d.groupby("user_id", group_keys=False)
                                 .rolling("7D", on="ts")["amount"].sum())
)
```
Or equivalently set `ts` as an index per group and use `rolling("7D")`.

**B6.** (i) `pd.read_csv(..., chunksize=N)` yields DataFrames of $N$ rows; process and aggregate per chunk. (ii) Use `dtype=` to shrink columns (e.g., `float32`, `category`). Bonus: switch to **Parquet** or **DuckDB** to avoid loading everything at all.

---

## Section C

**C1.** `FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`.

**C2.**
```sql
WITH ranked AS (
    SELECT dept_id, name, salary,
           ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC, name ASC) AS rn
    FROM employees
)
SELECT dept_id, name, salary FROM ranked WHERE rn <= 3;
```

**C3.** `INNER JOIN` returns only matched rows. `LEFT JOIN` returns all left rows, unmatched right side is NULL. An **anti-join** returns left rows with **no** match on the right (`WHERE NOT EXISTS (...)`). Placing a predicate on the right side in `WHERE` (e.g., `WHERE r.status = 'x'`) turns `LEFT JOIN` into effectively `INNER JOIN` because `NULL = 'x'` is `NULL` (drops the row).

**C4.** Leftmost-prefix rule: uses (a) `country`; (b) `country + created_at` (skipping middle); (c) `country + city + created_at` — but (b) *not* efficient in some engines. **`WHERE city = 'Helena'`** alone does **not** use the index.

**C5.** `COUNT(*)` counts **all rows** including NULL-containing ones. `COUNT(col)` counts **rows where `col IS NOT NULL`**. If `col` has no NULLs they're equal; otherwise `COUNT(col) ≤ COUNT(*)`.

**C6.**
```sql
WITH RECURSIVE ancestors AS (
    SELECT id, manager_id, name, 0 AS depth FROM employees WHERE id = 42
    UNION ALL
    SELECT e.id, e.manager_id, e.name, a.depth + 1
    FROM employees e JOIN ancestors a ON e.id = a.manager_id
)
SELECT * FROM ancestors WHERE id <> 42 ORDER BY depth;
```

---

## Section D

**D1.** Blob, tree, commit, tag.

**D2.** All three move `HEAD` to a target commit. `--soft` keeps changes staged in the index; `--mixed` (default) unstages changes but preserves them in the working tree; `--hard` throws away everything — commit, index, and working tree changes.

**D3.** (1) Rotate the compromised credential immediately (it's public — treat it as fully leaked). (2) Rewrite history to purge the secret using `git filter-repo` or BFG, then force-push. (3) Notify all collaborators to re-clone or perform the same rewrite so their local copies don't reintroduce it.

**D4.** A fast-forward merge is possible when the target branch has no new commits since your branch diverged. Git just moves the branch pointer forward; no merge commit is created. If both sides have new commits, Git must create a merge commit or you must rebase first.

**D5.**
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest -q
```

---

## Section E

**E1.** `null`, `boolean`, `number`, `string`, `array`, `object`.

**E2.** `401` — you're not authenticated (credentials missing or invalid). `403` — you're authenticated but not authorized for this resource. Mnemonic: 401 = "who?", 403 = "no."

**E3.** Full jitter: $t_n = \text{Uniform}(0, c \cdot b^{n-1})$ for base delay $c$, growth $b > 1$, attempt $n$. Jitter is critical because without it, thousands of clients whose requests failed in a wave will all retry at the same instant — DoS-ing the recovering server. Jitter spreads retries.

**E4.** Do **not** retry a naïve `POST` — it can produce duplicate side effects. Retry is safe when the endpoint supports **idempotency keys** (client sends an `Idempotency-Key` header; server dedupes by it) or when the request is inherently idempotent.

**E5.** `GET /v1/users/42/orders?status=paid&sort=-created_at&cursor=<token>&limit=20&fields=id,amount`.

---

## Section F

**F1.** `-e` exits on error, `-u` errors on undefined variables, `-o pipefail` propagates failures through pipelines. Together: fail loud, fail fast.

**F2.** `awk '{print $9}' access.log | sort | uniq -c | sort -rn | head -5`.

**F3.** A **hard link** is a second directory entry for the same inode (same file, two names); can't cross filesystems or link directories. A **symbolic link** is a file whose content is a path; can cross filesystems and point to directories; breaks when the target is deleted.

**F4.** Unquoted variables undergo word splitting.
```bash
F="my file.txt"
rm $F         # runs: rm my file.txt  → treats it as two filenames
rm "$F"       # correct
```

**F5.** A **VM** runs its own kernel on virtual hardware. A **container** shares the host's kernel and uses namespaces + cgroups to isolate the process's view of files, PIDs, network, and users. Containers are cheaper (no second kernel) but less isolated (a kernel bug crosses the boundary).

---

## Section G

**G1.** $m \times n$.

**G2.** $\text{rank}(A) + \dim \mathcal{N}(A) = n$, where $n$ is the number of columns.

**G3.** An **eigenvector** $\mathbf{v} \neq \mathbf{0}$ of $A$ is a direction the matrix merely scales: $A\mathbf{v} = \lambda\mathbf{v}$. The scalar $\lambda$ is the **eigenvalue**. For symmetric $A$, the spectral theorem gives **real eigenvalues** and **orthonormal eigenvectors** ($A = Q\Lambda Q^\top$).

**G4.** Normal equation: $A^\top A \mathbf{x} = A^\top \mathbf{b}$. Gives the same answer as `np.linalg.lstsq(A, b)` when $A$ has full column rank (the solution is unique). For rank-deficient $A$, `lstsq` returns the minimum-norm solution via SVD; the normal equations become singular.

**G5.** For any $A$, $A = U \Sigma V^\top$ with orthogonal $U, V$ and diagonal $\Sigma$ of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$. The rank-$k$ best approximation is:

$$A_k = U_{:,1:k} \, \Sigma_{1:k,1:k} \, V_{:,1:k}^\top$$

The **Eckart–Young theorem** guarantees $A_k$ minimizes $\|A - \tilde A\|_F$ over all matrices of rank $\leq k$ (also holds in the spectral norm).

---

## Section H

**H1.** For i.i.d. samples with finite variance $\sigma^2$, the standardized sample mean $\sqrt{n}(\bar{X}_n - \mu) / \sigma$ converges in distribution to $\mathcal{N}(0, 1)$ as $n \to \infty$.

**H2.** With $P(\text{sick}) = 0.001$, sensitivity 0.99, specificity 0.99:

$$P(+) = 0.99 \cdot 0.001 + 0.01 \cdot 0.999 = 0.00099 + 0.00999 = 0.01098$$

$$P(\text{sick} \mid +) = \frac{0.99 \cdot 0.001}{0.01098} \approx 0.090 \approx 9\%$$

**H3.** A 95% CI is a procedure that, applied to many independent experiments, produces intervals that contain the true parameter 95% of the time. It does **not** mean this specific realized interval contains the truth with probability 0.95 (frequentist) — for that you need a Bayesian credible interval.

**H4.** Expected false positives: $20 \times 0.05 = 1$. Corrections: **Bonferroni** (reject only if $p < \alpha/m$) or **Benjamini–Hochberg** (FDR control).

**H5.** Bootstrap:
- Draw many resamples of size $n$ **with replacement** from the sample.
- Compute the statistic $\hat\theta^*$ on each resample.
- The distribution of $\hat\theta^*$ approximates the sampling distribution of $\hat\theta$.
- Report percentile or bias-corrected CIs from the resample distribution.

**H6.** $\text{MSE}(\hat\theta) = \text{Bias}(\hat\theta)^2 + \text{Var}(\hat\theta)$. Regularization deliberately introduces bias to reduce variance sharply — trading a small bias for a much bigger variance reduction, thereby lowering total MSE.

---

## Section I

**I1.** An idempotent pipeline produces the same output regardless of how many times you re-run it. Anti-example: a script that appends today's data to a rolling CSV without deduplicating — running it twice on the same day double-counts.

**I2.** A data dictionary is a per-column reference: name, dtype, meaning, source, null policy, and value constraints. It's the shared vocabulary between analysts and engineers.

**I3.** In FastAPI's **lifespan** context manager (or `@app.on_event("startup")` in older versions). Loading once at startup ensures the model isn't re-loaded per request and that the service refuses to start if the model is missing.

**I4.** Running as root inside a container gives an attacker who exits the app process the same privileges as root on some namespaces, and mishandling volume mounts can write root-owned files onto the host. Add:
```dockerfile
RUN useradd -u 1000 -m app
USER 1000:1000
```

**I5.**
```toml
[project]
name = "ml_api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "pydantic>=2.7",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "ruff>=0.5",
  "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**I6.**
```json
{
  "error": {
    "code": "invalid_features",
    "message": "sepal_length must be non-negative",
    "request_id": "6a1e..."
  }
}
```
- **`code`**: machine-readable, stable identifier (drives client logic).
- **`message`**: human-readable explanation (never expose PII or stack traces).
- **`request_id`**: correlation id — clients quote it in tickets so you can find the log line.

---

## Scoring

| Section | Points | Yours |
|---------|--------|-------|
| A. Python | 10 | |
| B. NumPy & Pandas | 12 | |
| C. SQL | 12 | |
| D. Git & GitHub | 10 | |
| E. APIs & JSON | 10 | |
| F. Linux & CLI | 10 | |
| G. Linear Algebra | 10 | |
| H. Probability & Statistics | 12 | |
| I. Build Discipline | 14 | |
| **Total** | **100** | |

**Retake conditions.** Anything under 8/12 in a section, or under 80 overall — spend a week on the weak spots, then re-take a shuffled version of these questions.

**Passing.** Once you clear the bar, you're ready to leave Foundations and move to the next track. Ship the builds first if you haven't.
