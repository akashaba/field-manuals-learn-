# NumPy + Pandas — Master Study Guide

> **Track:** Foundations · **Module:** 02
> **Prerequisites:** Module 01 (Python), basic arithmetic and set thinking.
> **Time budget:** ~25–30 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** NumPy and Pandas are the twin pillars of Python data work. NumPy provides the **n-dimensional array**, a contiguous block of memory with a fixed dtype that C, Fortran, and CUDA code can operate on at native speed. Pandas builds a **labeled tabular** abstraction on top of NumPy so you can slice, group, merge, and reshape data the way an analyst thinks.

Every popular ML/data library — scikit-learn, PyTorch (via `.numpy()`), TensorFlow, XGBoost, statsmodels, Matplotlib, scipy — either **is** NumPy underneath or **speaks** NumPy at its I/O boundary. So fluency here is not optional.

**Fundamental principles you must own:**

1. **Vectorization > loops.** Python-level `for` loops over arrays are 50–500× slower than the vectorized equivalent because the loop overhead dominates. Learn to *think in whole arrays.*
2. **Broadcasting.** NumPy silently expands shapes so that a `(3,1)` array can be added to a `(1,4)` array to produce a `(3,4)` result — with no memory copy. Broadcasting is the syntax of tensor code.
3. **Views vs copies.** Most slicing returns a **view** on the same buffer. Mutating a view mutates the original. `.copy()` makes a real copy. Understanding this is the difference between subtle bugs and correct code.
4. **Pandas is NumPy plus labels.** A `DataFrame` is a dict of `Series`; a `Series` is a `numpy.ndarray` plus an `Index`. Every operation is a combination of alignment (by label) and computation (by NumPy).
5. **The split-apply-combine pattern.** Nearly all analytics reduces to *split* rows into groups, *apply* a function, *combine* the results. `groupby` is the canonical implementation and worth mastering.

If you retain nothing else: **when you feel a `for` loop coming on, stop and ask "can this be vectorized?"** The answer is *almost always yes.*

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 NumPy Arrays: Shape, Dtype, and Memory

**Definition.** An `ndarray` is a homogeneous, fixed-size, N-dimensional array with a `shape` tuple, a `dtype`, and `strides` describing how to walk it.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
a.shape      # (2, 3)
a.dtype      # dtype('float32')
a.strides    # (12, 4) — bytes to step per axis (2 rows of 3 float32s)
a.nbytes     # 24 = 2 * 3 * 4
```

**Technical nuances:**

- **Dtype matters for memory and speed.** `float64` is the default; using `float32` halves memory and often doubles throughput for large arrays. `int8` for categorical labels can be 8× smaller.
- **`np.array` copies data** from the input; `np.asarray` doesn't copy if input is already an array with the right dtype.
- **`reshape` returns a view** when memory is contiguous; **`resize` mutates in place** (rarely what you want).
- **`ravel()`** returns a flattened view when possible; **`flatten()`** always copies.
- **`np.newaxis`** (aka `None`) adds a size-1 axis: `a[:, None]` promotes a `(3,)` vector to a `(3,1)` column.

**Real-world example.** Loading 100M float64s from disk is 800 MB. If you know the values fit in float32 or int32, changing the dtype saves 400 MB — the difference between OOM and fitting in memory.

---

### 2.2 Broadcasting

**Definition.** When operating on two arrays of different shapes, NumPy aligns them from the trailing axes and *virtually* stretches dimensions of size 1 to match. No data is copied — the stride is set to 0.

**The rules (memorize):**

1. If arrays have different numbers of dimensions, **left-pad** the smaller shape with 1s.
2. Two dimensions are **compatible** if they are equal, or one of them is 1.
3. The output shape takes the max on each axis.

**Concrete example.** Add a per-column bias vector to a batch of samples:

$$X \in \mathbb{R}^{n \times d}, \quad b \in \mathbb{R}^{d}, \quad Y_{ij} = X_{ij} + b_j$$

```python
X = np.random.randn(1000, 5)     # 1000 samples, 5 features
b = np.array([1, 2, 3, 4, 5])    # (5,)
Y = X + b                        # (1000, 5) + (5,) → broadcast → (1000, 5)
```

Another: pairwise distances between two sets of points.

$$D_{ij} = \|x_i - y_j\|_2 = \sqrt{\sum_{k=1}^{d} (x_{ik} - y_{jk})^2}$$

```python
X = np.random.randn(100, 3)      # (100, 3)
Y = np.random.randn(50, 3)       # (50, 3)
# introduce axes so shapes are (100, 1, 3) and (1, 50, 3)
diff = X[:, None, :] - Y[None, :, :]   # (100, 50, 3)
D = np.sqrt((diff**2).sum(axis=-1))    # (100, 50)
```

That two-line trick replaces a double `for` loop and runs ~200× faster on large inputs.

**Technical nuances:**

- Broadcasting **never copies**; it uses stride tricks. But a subsequent operation (`+`, `*`) *does* allocate the result.
- Common bug: broadcasting silently succeeds where you expected a shape error. If your model output has an unexpected shape, print `.shape` at every step.
- `np.broadcast_shapes(a.shape, b.shape)` tells you the result shape without doing the op.

---

### 2.3 Vectorization & ufuncs

**Definition.** A **universal function** (ufunc) is a NumPy function (`np.add`, `np.exp`, `np.sin`, ...) that operates element-wise, supports broadcasting, and executes in optimized C. Vectorization = expressing computation as ufunc calls over whole arrays.

**Example — the sigmoid function:**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
```

That one line handles any shape, dtype, and broadcast partner.

**Benchmarking the difference:**

```python
xs = np.random.randn(10_000_000)

%timeit [1/(1+math.exp(-x)) for x in xs]   # ~5–10 s (Python loop)
%timeit sigmoid(xs)                        # ~30–60 ms (vectorized)
```

**Technical nuances:**

- **`np.where(cond, a, b)`** is a vectorized `if`. Prefer it to boolean masks + iteration.
- **`np.select`** handles multiple conditions.
- **Aggregations** (`.sum`, `.mean`, `.max`, `.argmax`, `.std`) take an `axis=` argument: 0 collapses rows, 1 collapses columns.
- **Reductions with `keepdims=True`** preserve the reduced axis as size 1 — extremely useful for broadcasting the result back:

  ```python
  X = np.random.randn(1000, 5)
  X_centered = X - X.mean(axis=0, keepdims=True)   # broadcast-safe
  ```

- **Fused ops** are cheaper: `np.multiply(a, b, out=a)` writes in place, avoiding an allocation.

---

### 2.4 Pandas: Series, DataFrame, and Indexing

**Definition.** A `Series` is a 1-D labeled array (values + Index). A `DataFrame` is a 2-D labeled table where each column is a `Series` and columns share the same row Index.

```python
import pandas as pd

df = pd.DataFrame({
    "user_id":  [1, 2, 3, 4, 5],
    "country":  ["US", "US", "UG", "UG", "KE"],
    "revenue":  [10.0, 15.0, 8.0, np.nan, 12.0],
    "signup":   pd.to_datetime(["2025-01-01","2025-02-14","2025-03-05","2025-04-20","2025-05-01"]),
})
df.set_index("user_id", inplace=True)
```

**Selecting rows and columns (memorize the distinction):**

- **`df.loc[label]`** — label-based (uses the Index). Inclusive on both ends of a slice.
- **`df.iloc[position]`** — integer-position-based. Exclusive on the right, like Python slices.
- **`df["col"]`** — a Series (single column).
- **`df[["col1", "col2"]]`** — a DataFrame (multiple columns).
- **`df[mask]`** — boolean-mask row filter.

```python
df.loc[3, "revenue"]        # scalar for user 3
df.loc[df["country"] == "UG", ["revenue", "signup"]]  # rows for UG
df.iloc[0:2, 0:2]           # first two rows, first two columns
```

**Technical nuances:**

- **`SettingWithCopyWarning`** appears when you chain indexing (`df[df.a > 0]["b"] = 5`) — Pandas isn't sure whether you meant the copy or the original. Fix with `.loc`:
  ```python
  df.loc[df.a > 0, "b"] = 5
  ```
- **`dtype='category'`** for low-cardinality strings saves huge memory.
- **Missing data:** `NaN` for floats, `pd.NA` (nullable) for the new nullable dtypes (`Int64`, `Float64`, `boolean`, `string`).
- **`.apply` is slow** — it drops back into Python. Try vectorized ops first, then `.map`, and only then `.apply`.
- **`pd.eval` / `df.query`** can accelerate large expressions by parsing them once.

**Real-world example.** In the LEG-MT bill drafting system, you might load a CSV of bill submissions, filter by session year, group by sponsor, and sum bill counts — five lines of Pandas replacing a hundred lines of hand-written parsing.

---

### 2.5 GroupBy, Merge, Reshape (split-apply-combine)

**GroupBy (split-apply-combine):**

```python
(
    df
    .groupby("country", as_index=False)
    .agg(revenue_total=("revenue", "sum"),
         revenue_mean =("revenue", "mean"),
         n_users      =("revenue", "size"))
)
```

You can chain multiple aggregations, custom functions, and transformations:

- **`.agg`** — reduces each group to one row per group.
- **`.transform`** — returns a same-shaped Series aligned to the original (great for group-normalized values).
- **`.filter`** — keeps or drops whole groups based on a predicate.
- **`.apply`** — general escape hatch; slow.

**Example (group-standardize):**

$$z_{i} = \frac{x_i - \mu_{g(i)}}{\sigma_{g(i)}}$$

```python
df["revenue_z"] = (
    df.groupby("country")["revenue"]
      .transform(lambda s: (s - s.mean()) / s.std(ddof=0))
)
```

**Merge / Join.** Pandas supports SQL-style joins:

```python
orders = pd.DataFrame({"user_id":[1,2,2,3], "amt":[10,5,7,3]})
users  = pd.DataFrame({"user_id":[1,2,3,4], "name":["A","B","C","D"]})

pd.merge(orders, users, on="user_id", how="left")     # inner|left|right|outer
```

**Reshape — long ↔ wide:**

```python
wide  = df.pivot_table(index="country", columns="signup_month",
                      values="revenue", aggfunc="sum")
long  = wide.reset_index().melt(id_vars="country",
                                var_name="signup_month",
                                value_name="revenue")
```

**Technical nuances:**

- `groupby(..., sort=False)` skips sorting groups — measurable speedup on many groups.
- `merge(..., validate="one_to_one")` (or `"one_to_many"`, etc.) fails loudly if your keys aren't unique as expected — cheap insurance against silent duplication.
- For very wide reshapes, `pivot_table` supports `aggfunc="first"` or list of aggfuncs.
- Time-series has its own operators: `df.resample("D").mean()`, `df.rolling(7).mean()`, `df.ewm(alpha=0.1).mean()`.

---

## 3. Mental Models & Analogies

### 3.1 The "Spreadsheet with Superpowers" Model (DataFrames)

Think of a `DataFrame` as an Excel sheet where **every column has a type** (no accidental string-in-numeric-column), **every row has a name** (the Index), and **every operation can be expressed once and applied to a whole column or the whole table**. When you would drag a formula down 100,000 rows in Excel, in Pandas you write a single expression on the whole column.

The Index is the row-name column that's always visible on the left of Excel — except in Pandas it can be **multi-level** (MultiIndex) so you can group by year *and* country in the same axis, or **datetime-aware** so `.loc["2025-Q1"]` picks the right slice automatically.

### 3.2 The "Elastic Sheets on a Grid" Model (Broadcasting)

Imagine each array is a rectangular rubber sheet of numbers, and their shapes are (rows, columns). Broadcasting is a rule that says: *if a side of one sheet is size-1, that sheet is elastic on that side — it stretches to match the other sheet.* If neither is size-1 and they differ, they can't stretch and NumPy raises `ValueError: operands could not be broadcast together`.

- A `(3,1)` sheet and a `(1,4)` sheet both stretch to `(3,4)`.
- A `(3,4)` sheet and a `(4,)` vector: the vector is left-padded to `(1,4)`, then stretches to `(3,4)`.
- A `(3,4)` sheet and a `(3,)` vector: fails, because `(3,)` becomes `(1,3)` which doesn't match `(3,4)` on the trailing axis. Fix by making the vector explicit: `v[:, None]` → `(3,1)`.

![IMG-NP-01](/1%20—%20Foundations/images/IMG-NP-01.png)
> **Caption:** Size-1 axes stretch (virtually) to match; the output takes the max shape on each axis.
> **Placement:** Section 2.2 (Broadcasting).

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "I'll Just Loop Over the DataFrame Rows with `iterrows`"

`for i, row in df.iterrows(): ...` is idiomatic-looking but **extremely slow** — it constructs a `Series` per row. For 1M rows it can take minutes what vectorized ops do in milliseconds. The next resort after that is `.apply`, then `.itertuples` (much faster than `iterrows`), and *only when unavoidable* an explicit loop. Rule: try vectorized first; benchmark; only then reach for iteration.

### 4.2 "Views and Copies Are Interchangeable"

```python
df2 = df[df["country"] == "UG"]
df2["revenue"] = 0
```

That second line will (a) mutate `df` if `df2` was a view, (b) mutate only `df2` if it was a copy, and (c) usually raise a `SettingWithCopyWarning` because Pandas can't tell what you intended. The right pattern is `df.loc[df.country == "UG", "revenue"] = 0`. Same story in NumPy: `a[::2]` is a view; modifying it modifies `a`.

### 4.3 "`NaN` Behaves Like Any Other Value in Comparisons"

`np.nan == np.nan` is **`False`**. So is `np.nan < 5` and `np.nan > 5`. Filtering `df[df.x != np.nan]` **keeps every row**, silently. Use `df.x.isna()` / `df.x.notna()`. And be aware that many aggregations skip NaNs by default (`.mean`) while others don't (`.dot`). Read the docstring for the ones you rely on.

---

## 5. Self-Assessment Bank (NumPy + Pandas)

### Questions

**Q1 (Short answer).** Explain broadcasting in ≤3 sentences. Give one shape pair that broadcasts and one that fails.

**Q2 (Multiple choice).** Which is the fastest way to compute the row-wise mean of a `(1_000_000, 100)` NumPy array?
A. A Python `for` loop over rows, taking `sum(row)/len(row)`
B. `np.mean(a, axis=1)`
C. `pd.DataFrame(a).apply(np.mean, axis=1)`
D. `[a[i].sum()/100 for i in range(a.shape[0])]`

**Q3 (Short answer).** What is the difference between `df.loc[i]` and `df.iloc[i]`? What is `df[i]` (with an integer)?

**Q4 (Multiple choice).** After the following code, what does `a` look like?
```python
a = np.arange(6).reshape(2, 3)
b = a[:, 1:]
b[0, 0] = 999
```
A. `[[0,1,2],[3,4,5]]`
B. `[[0,999,2],[3,4,5]]`
C. `[[0,999,999],[3,4,5]]`
D. `ValueError`

**Q5 (Short answer).** Write a one-liner that returns the top-3 revenue-generating countries from `df` (columns: `country`, `revenue`).

**Q6 (Multiple choice).** Which is the correct dtype hint for a Pandas column of small integers with missing values?
A. `int64`
B. `Int64` (nullable)
C. `float64`
D. `category`

**Q7 (Short answer).** Given `x = np.random.randn(1000, 5)`, standardize each column (zero mean, unit variance) in one line.

**Q8 (Multiple choice).** `df.merge(other, on="id", how="left", validate="one_to_one")` will:
A. Silently accept duplicate ids on the right side.
B. Raise if any id appears more than once on either side.
C. Only join if both sides have the same length.
D. Perform a full outer join.

**Q9 (Short answer).** What is the semantic difference between `.agg`, `.transform`, and `.apply` on a `GroupBy` object?

**Q10 (Multiple choice).** Given a Series `s` with 1M float64 values, which uses the LEAST memory?
A. `s.astype("float32")`
B. `s.astype("category")` (values are truly continuous)
C. `s.astype("int64")`
D. `s.astype("string")`

---

### Answer Key & Detailed Explanations

**A1.** Broadcasting is NumPy's rule for operating on arrays of different shapes: align the shapes from the right, treat missing left-axes as 1, and virtually stretch any size-1 axis to match. `(3,1)` and `(1,4)` broadcast to `(3,4)`. `(3,4)` and `(2,4)` fail because 3 ≠ 2 and neither is 1.

**A2. B.** `np.mean(a, axis=1)` calls into optimized C, releases the GIL, and computes 100M values in ~50 ms. Options A and D pay Python-loop overhead per row. Option C wraps the array in a DataFrame and applies at Python speed per row.

**A3.** `df.loc[i]` looks `i` up in the row **Index** (labels). `df.iloc[i]` uses **integer position**. `df[i]` with an integer tries to look `i` up in the **column names** (not rows!) and raises `KeyError` if the column doesn't exist — a common source of confusion.

**A4. B.** `a[:, 1:]` is a **view** onto columns 1–2 of `a`. Mutating `b[0, 0]` mutates `a[0, 1]`.

**A5.** `df.groupby("country")["revenue"].sum().nlargest(3)`.

**A6. B.** `Int64` (capital I) is Pandas' nullable integer type that supports `pd.NA`. Regular `int64` cannot hold NaN, so Pandas normally upgrades to float — losing precision on large ints and wasting memory. `category` is for repeated string labels.

**A7.** `(x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)`. (`keepdims=True` isn't strictly required here because 1-D broadcasts, but it's a good habit that generalizes to higher-D reductions.)

**A8. B.** `validate="one_to_one"` asserts that the merge key is unique on **both** sides; the merge fails loudly if not. This is a cheap way to catch data-integrity bugs.

**A9.** `.agg` reduces each group to fewer rows (typically 1 per group). `.transform` returns the same shape as the input, broadcast back to each original row (useful for group-normalized columns). `.apply` is the general escape hatch — it can return anything, but is slow and its return shape depends on what the function returns per group.

**A10. A.** `float32` halves memory (4 GB → 2 GB for a 1G-value series, or 8 MB → 4 MB for 1M values). `category` requires distinct-value lookup; for truly continuous data every value is unique, so the category codes buy nothing. `int64` doesn't help. `string` is variable-length and larger.

---

## 6. Practice Prompts

1. Load a 1 GB CSV in chunks (`chunksize=100_000`), compute a per-group sum, and produce the final aggregation without ever loading the whole thing.
2. Implement pairwise Euclidean distances between two `(N, d)` arrays in one line using broadcasting. Verify against `scipy.spatial.distance.cdist`.
3. Convert a "long" DataFrame of `(user, date, event, value)` rows into a "wide" one of `(user, date)` rows with one column per event. Then convert back.
4. Given a DataFrame of stock prices with a `DatetimeIndex`, compute a 20-day rolling mean and 20-day rolling z-score of returns.
5. Use `df.query()` to filter a DataFrame with a compound condition and benchmark it against boolean-mask indexing.

---

## 7. References

- Wes McKinney, *Python for Data Analysis*, 3rd ed.
- NumPy docs: [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), [Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- Pandas docs: [10 Minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html), [Enhancing Performance](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- Jake VanderPlas, *Python Data Science Handbook* (free online)
