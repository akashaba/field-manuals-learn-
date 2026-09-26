# Build #1 — Data Analysis Project

> **Track:** Foundations · **Build:** 01
> **Prerequisites:** Modules 01–04, 06, 08 minimum (Python, NumPy/Pandas, SQL, Git, CLI, Stats).
> **Time budget:** ~20–30 hours of focused work spread over ~1 week.
> **Deliverable:** A public GitHub repo with a reproducible pipeline, tests, a written report, and one polished chart set.

---

## 1. Executive Summary

You will pick a **real, medium-sized dataset** (100k–10M rows), form a **specific, answerable question**, and produce a **clean, reproducible analysis** that ends in a short written report with evidence. This is not a Kaggle-style leaderboard chase — it's a *proper analysis project* that mirrors what a professional data analyst or engineer actually ships:

- Data ingested from a source (CSV, API, or public database).
- Loaded into a structured store (Parquet + SQLite or DuckDB).
- Cleaned, joined, and transformed via a documented pipeline.
- Explored with Pandas/SQL, visualized with matplotlib or Plotly.
- Written up with methodology, findings, caveats, and next steps.
- All code is versioned, tested, linted, and reproducible on a fresh machine in **< 10 minutes**.

This is the **first artifact** you'll link on a résumé from this track. Treat it accordingly.

---

## 2. Learning Outcomes

By the end you will have practiced:

1. Structuring a Python project as an installable package (not a folder of notebooks).
2. Loading messy real-world data and expressing cleaning steps as **idempotent** operations.
3. Writing SQL queries against a local warehouse (SQLite or DuckDB) instead of everything in Pandas.
4. Visualizing with **intentionality** — chart type matched to question.
5. Testing data pipelines with `pytest` (data assertions, unit tests for transforms).
6. Writing a technical report that a non-author can read and act on.
7. Reproducibility: a fresh clone + `make setup && make run` completes end-to-end.

---

## 3. Scope: Pick One

Choose **one** of the following prompts (or a comparable one you propose). The dataset should be **large enough to punish sloppy code** but **small enough to fit on a laptop**.

### 3.1 Option A — Legislative Analytics (aligned with Brian's day job)
> Analyze 5+ years of bill submissions at the Montana Legislature (or another open legislative dataset). Answer questions like:
> - Which sponsors have the highest bill-passage rate, controlling for chamber and topic?
> - Have bill-length distributions changed over time?
> - Are cross-party co-sponsored bills more likely to pass than single-party ones?

### 3.2 Option B — Public Transit / Traffic
> Pull a year of a city's transit-ride or bikeshare data. Answer:
> - How does weekday-vs-weekend ridership differ by station?
> - Is there a weather effect (join with a public weather API)? Quantify it.

### 3.3 Option C — Global Football Fixtures (nod to soccer analytics interest)
> Analyze 3–5 seasons of top-league match data. Answer:
> - Does home advantage exist in every league to the same degree?
> - How do goal-scoring rates vary by minute of match? Is stoppage-time inflation real?

### 3.4 Option D — Financial Time Series
> Analyze a decade of daily OHLC data for a basket of assets. Answer:
> - How stable are correlations between assets over 30-day rolling windows?
> - Is realized volatility mean-reverting?

**Whichever you pick, the question must be:**

- **Specific** — not "explore this dataset".
- **Answerable** — the data must actually contain the signal.
- **Falsifiable** — the answer could go either way.

---

## 4. Repository Structure

```
your-project/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/             # untouched inputs (gitignored except a .gitkeep + README)
│   ├── interim/         # partially processed (gitignored)
│   └── processed/       # final analysis-ready tables (may be committed if small)
├── notebooks/
│   ├── 01_explore.ipynb
│   └── 02_deep_dive.ipynb
├── src/
│   └── your_project/
│       ├── __init__.py
│       ├── config.py
│       ├── ingest.py
│       ├── clean.py
│       ├── transform.py
│       ├── analyze.py
│       ├── viz.py
│       └── cli.py
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_load.sql
│   └── analysis/
│       ├── passage_rate_by_sponsor.sql
│       └── ...
├── tests/
│   ├── test_clean.py
│   ├── test_transform.py
│   └── fixtures/
├── reports/
│   ├── figures/
│   └── report.md
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock  (or requirements.txt / poetry.lock)
```

**Rules:**

- Raw data is never mutated in place. If it comes in, it stays. Every step writes new files.
- Notebooks are for **exploration and narrative**, never the source of truth. When a piece of logic proves out in a notebook, **move it into `src/your_project/`** and import it back into the notebook.
- SQL files are versioned and re-runnable.
- `data/raw/` is `.gitignore`'d. `README.md` documents where to download it (or a script does so idempotently).

---

## 5. Milestones & Acceptance Criteria

Break the work into **7 milestones**. Each has a checklist and, where relevant, a "definition of done".

### M1 — Kickoff & Scaffolding (2h)
- [ ] Repo initialized on GitHub with MIT license and a proper README skeleton.
- [ ] `pyproject.toml` with dependencies pinned. `uv` (or `pip-tools`) used to lock.
- [ ] Pre-commit hooks: `ruff`, `black`, `end-of-file-fixer`, `check-yaml`.
- [ ] GitHub Actions workflow that runs `ruff`, `pytest`, and (optionally) `mypy`.
- [ ] `Makefile` targets: `setup`, `data`, `test`, `run`, `report`, `clean`.

### M2 — Data Ingestion (3–5h)
- [ ] `src/your_project/ingest.py` downloads or reads raw data, writes to `data/raw/`.
- [ ] Idempotent: rerunning does not duplicate.
- [ ] Failure modes handled: network errors, malformed rows, timeouts.
- [ ] Ingestion logs its inputs, outputs, and row counts to stdout.

### M3 — Cleaning & Structuring (3–5h)
- [ ] Cleaning steps expressed as **pure functions** (input → output DataFrame).
- [ ] Rows dropped, columns renamed, dtypes converted, null policies documented.
- [ ] Cleaned data written as **Parquet** (columnar, typed, efficient) to `data/processed/`.
- [ ] A data dictionary (`docs/data_dictionary.md`) documents every final column: name, type, meaning, source column, null policy.

### M4 — Warehouse & SQL (3–5h)
- [ ] Load processed Parquet into **SQLite** or **DuckDB**.
- [ ] Analysis queries live in `sql/analysis/*.sql`, versioned.
- [ ] At least **three** analysis queries answer distinct sub-questions of the main question.
- [ ] Each query has a comment explaining the business question it answers.

### M5 — Analysis & Visualization (5–8h)
- [ ] For each sub-question, produce a **chart + prose paragraph**.
- [ ] Charts follow good practice: labeled axes, units, sensible ranges, no chartjunk, colorblind-safe palette.
- [ ] At least one chart uses a **statistical technique** covered in Module 08: bootstrap CI, permutation test, or hypothesis test.
- [ ] `src/your_project/viz.py` centralizes styling — no chart is styled ad hoc.

### M6 — Tests & Data Assertions (2–3h)
- [ ] `pytest` covers cleaning and transform functions with realistic fixtures.
- [ ] Data assertions in code (`assert df["col"].notna().all()`, or use `pandera` / `great_expectations`).
- [ ] CI runs tests on every push to main.
- [ ] Coverage ≥ 70% on `src/your_project/` (measure with `coverage`, don't obsess).

### M7 — Report & Reproducibility (2–3h)
- [ ] `reports/report.md` (~1500–2500 words) with sections: Executive Summary, Data, Method, Findings, Caveats, Next Steps.
- [ ] All charts embedded from `reports/figures/` (produced by `make report`).
- [ ] README shows a **fresh-clone-to-report** command sequence that completes in < 10 minutes.
- [ ] A friend clones the repo and reproduces it, following only the README. You fix whatever they get stuck on.

---

## 6. Non-negotiables (do not skip)

1. **The pipeline is idempotent.** Running `make run` twice does not produce different results.
2. **No secrets in the repo.** API keys via env vars only; `.env.example` documents required vars.
3. **Every commit is small and well-named.** `feat: add bill passage rate query`, not `stuff`.
4. **Charts have units and clear axis labels.** No unlabeled y-axes. No two-sig-fig percentages hiding a base of 3.
5. **Statistical claims are qualified.** "Sponsors A and B pass bills at 62% vs 48% (bootstrap 95% CI [58, 66] vs [44, 52])" — not "A is way better than B".

---

## 7. Nice-to-Haves (stretch)

- Push processed data to a public S3/R2 bucket so cloners don't have to re-ingest.
- Use `DuckDB` directly on Parquet without loading into memory (`duckdb.query("SELECT ... FROM 'data/processed/*.parquet'")`).
- Build a small Streamlit or Observable Framework dashboard.
- Add a `dvc.yaml` or `snakemake` pipeline for stage-level caching.
- Publish the report as a GitHub Pages site.

---

## 8. Report Template

Use this as `reports/report.md` scaffold:

```markdown
# <Title of the Analysis>

**Author:** Brian  ·  **Date:** YYYY-MM-DD  ·  **Repo:** <link>

## Executive Summary
- Question:
- Answer (one sentence):
- Confidence & caveats (one line):

## Data
- Source:
- Time range:
- Row count / grain:
- Known issues:

## Methodology
- Cleaning:
- Aggregations:
- Statistical tests / CIs used:
- Assumptions:

## Findings
### Sub-question 1
[chart]
Paragraph.

### Sub-question 2
[chart]
Paragraph.

## Caveats & Limitations
-
-

## Next Steps
-
-

## Appendix
- Data dictionary link
- SQL queries link
- Environment / reproduction steps
```

---

## 9. Evaluation Rubric (grade yourself brutally)

| Area | 1 (poor) | 3 (ok) | 5 (excellent) |
|------|----------|--------|---------------|
| Question specificity | vague | answerable | specific + falsifiable |
| Code structure | script in one file | package layout | tested, linted, typed |
| Data hygiene | in-place edits | separate raw/processed | idempotent + Parquet + assertions |
| SQL usage | pandas only | SQL for some queries | SQL for core analysis, versioned |
| Visualization | default charts | labeled + intentional | polished + colorblind-safe + captioned |
| Statistical rigor | point estimates | CI or test | CI + effect size + caveat |
| Reproducibility | "works on my machine" | one-command run | fresh clone < 10 min |
| Documentation | terse README | README + inline docstrings | README + data dict + report |

Aim for a **4 or 5 in every row** before publishing.

---

## 10. Related Files

- `../goals.md`
- `../learning/01-python.md` (project structure, testing)
- `../learning/02-numpy-pandas.md` (dataframes)
- `../learning/03-sql.md` (warehouse queries)
- `../learning/08-probability-statistics.md` (bootstrap, CIs, tests)
- `../builds/02-ml-api-fastapi.md`
