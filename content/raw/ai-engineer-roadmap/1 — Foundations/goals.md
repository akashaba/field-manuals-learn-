# Foundations — Goals

> **Track:** Foundations (Module 1 of the broader learning journey)
> **Meta-goal:** *Become comfortable writing production-ish Python, not just notebooks.*

---

## 1. The North Star

The single sentence that governs every decision in this track:

> **"When I finish Foundations, I can pick up any ML/data problem, set up a clean repo, model the data, prototype in a notebook, factor it into a proper Python package with tests, expose it behind a FastAPI endpoint, and deploy it — without needing to Google the mechanics."**

Every module, every exercise, and every review below is scored against that sentence. If a topic doesn't move you toward it, deprioritize it.

---

## 2. Terminal Learning Objectives (Level: mastery, not exposure)

By the end of this track you will be able to do **all** of the following without a reference open:

### 2.1 Python Craftsmanship
- **LO-1.1** Write idiomatic Python 3.11+ using type hints, dataclasses, generators, context managers, and comprehensions correctly.
- **LO-1.2** Structure a project as an installable package (`pyproject.toml`, `src/` layout) with `pip install -e .`.
- **LO-1.3** Write and run **pytest** unit tests, fixtures, parametrization, and coverage.
- **LO-1.4** Use `ruff`, `black`, `mypy` in a pre-commit hook.
- **LO-1.5** Manage environments with `venv`, `pip-tools`, or `uv` and pin dependencies deterministically.

### 2.2 Data Wrangling
- **LO-2.1** Vectorize computations in **NumPy** (no `for` loops over arrays).
- **LO-2.2** Fluently `merge`, `groupby`, `pivot`, `melt`, `apply`, and window-function in **Pandas**.
- **LO-2.3** Read/write CSV, Parquet, JSON, and SQL tables into DataFrames.
- **LO-2.4** Diagnose memory blow-ups and use chunking or Polars/DuckDB when Pandas won't fit.

### 2.3 SQL Fluency
- **LO-3.1** Write joins (inner, left, right, full, self, anti, semi), CTEs, window functions, and subqueries.
- **LO-3.2** Read a `EXPLAIN`/`EXPLAIN ANALYZE` plan and reason about index use.
- **LO-3.3** Model a schema in **3NF**, then denormalize where reads dominate.
- **LO-3.4** Write reproducible migrations (Alembic or Flyway) rather than raw DDL sprayed at the DB.

### 2.4 Version Control
- **LO-4.1** Use Git as a graph, not a checkpoint tool: branch, rebase, cherry-pick, bisect, reflog.
- **LO-4.2** Open a PR that is small, single-purpose, and reviewable.
- **LO-4.3** Configure GitHub Actions for CI (lint + test + type-check on push).

### 2.5 HTTP & JSON
- **LO-5.1** Explain HTTP verbs, status codes, headers, and idempotency.
- **LO-5.2** Design a REST-ish JSON API (resources, pagination, filtering, error envelopes).
- **LO-5.3** Consume APIs with `httpx`/`requests` including retries, timeouts, and pagination.

### 2.6 The Command Line & Linux
- **LO-6.1** Traverse a filesystem, chain pipes, redirect I/O, and use `grep`/`sed`/`awk`/`jq` fluently.
- **LO-6.2** Manage processes, users, permissions, and services in Linux.
- **LO-6.3** Write Bash scripts that fail loudly (`set -euo pipefail`).
- **LO-6.4** Run and debug Docker containers.

### 2.7 Linear Algebra (working knowledge for ML)
- **LO-7.1** Compute dot products, matrix-vector and matrix-matrix products by hand and in NumPy.
- **LO-7.2** Explain eigen-decomposition, SVD, and their role in PCA.
- **LO-7.3** Reason about linear systems, rank, nullspace, and least squares.

### 2.8 Probability & Statistics
- **LO-8.1** Compute expectations, variances, and covariances.
- **LO-8.2** Apply the Law of Large Numbers, Central Limit Theorem, and Bayes' rule correctly.
- **LO-8.3** Perform hypothesis tests (t-test, chi-square, permutation) and interpret p-values honestly.
- **LO-8.4** Estimate confidence intervals via analytical formulas and bootstrap.

---

## 3. Deliverables (the "proof of learning")

You will produce **two** portfolio-grade deliverables, each with a README, tests, and a public GitHub repo:

1. **Data Analysis Project** — a full exploratory-to-report pipeline on a real dataset. See `builds/01-data-analysis-project.md`.
2. **ML API with FastAPI** — a trained model exposed as a versioned HTTP service, dockerized, tested, and CI'd. See `builds/02-ml-api-fastapi.md`.

If a hiring manager or a future teammate reads these two repos and can (a) reproduce them locally in under 10 minutes and (b) understand every design decision, the goal is met.

---

## 4. Definition of Done (per topic)

A topic is **done** when you can:

- [ ] Teach it back for 5 minutes without notes (**Feynman test**).
- [ ] Score ≥ 8/10 on that module's Self-Assessment Bank.
- [ ] Ship a small artifact that uses it (a script, a notebook cell, or a repo).
- [ ] Explain **one** common failure mode from lived experience (not from a book).

---

## 5. Anti-goals (what you are *not* doing here)

Naming these upfront prevents scope creep:

- **You are not learning "web dev".** FastAPI here is a means to serve a model, not to build a SaaS.
- **You are not becoming a DBA.** SQL to the level of reading a query plan is enough — index internals can wait.
- **You are not proving math theorems.** Linear algebra & stats are tools; work computationally first, prove second.
- **You are not chasing the newest library.** `pandas` and `numpy` are load-bearing. Polars, Ibis, DuckDB come later.
- **You are not writing tutorials or blog posts about this track** until the builds are shipped.

---

## 6. Cadence & Time Budget (suggested)

| Phase | Weeks | Focus |
|-------|-------|-------|
| 1 | 1–2 | Python + Git + Linux/CLI (the "hands") |
| 2 | 3–4 | NumPy + Pandas + SQL (the "data") |
| 3 | 5 | APIs + JSON (the "wire") |
| 4 | 6–7 | Linear Algebra + Probability/Stats (the "math") |
| 5 | 8 | **Build #1**: Data Analysis Project |
| 6 | 9–10 | **Build #2**: ML API with FastAPI |
| 7 | 11 | Consolidated Self-Assessment + retros |

Adjust to your reality — but keep the **two builds** immovable. They are the deliverables.

---

## 7. Success Signals

You will know Foundations is behind you when the following happen naturally, without conscious effort:

- You reach for a `.py` file and `pytest` before opening a notebook.
- You feel physical discomfort when you see a `for` loop over a NumPy array.
- You write `git rebase -i` more often than `git commit --amend`.
- You read someone else's SQL query and mentally trace the join order.
- Your `~/.bashrc` and `~/.gitconfig` are personal — not the defaults.
- When someone asks "can you spin up an endpoint that returns predictions?", you say **"give me an hour"** and mean it.

---

## 8. Related Files

- `learning/01-python.md`
- `learning/02-numpy-pandas.md`
- `learning/03-sql.md`
- `learning/04-git-github.md`
- `learning/05-apis-json.md`
- `learning/06-linux-cli.md`
- `learning/07-linear-algebra.md`
- `learning/08-probability-statistics.md`
- `builds/01-data-analysis-project.md`
- `builds/02-ml-api-fastapi.md`
- `self-assessment.md`
