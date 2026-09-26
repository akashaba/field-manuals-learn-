# SQL — Master Study Guide

> **Track:** Foundations · **Module:** 03
> **Prerequisites:** Basic understanding of tables (rows/columns). Module 02 helps but isn't required.
> **Time budget:** ~20–30 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Structured Query Language is the *durable* skill of data work. Frameworks come and go — Rails, Django, Spark, dbt, Snowflake — but the SQL you write today is the SQL your grandkid will still write. Every serious data source (production databases, warehouses, lakes with query engines, cloud analytics services) speaks it. Every ML system's training data starts as a query. Every dashboard is a query. Every backend that persists state runs queries.

Learning SQL "properly" means moving past `SELECT * FROM users` toward:

- Modeling data in **normalized** shapes (and knowing when to break the rules).
- Composing queries with **CTEs and subqueries** that read like a paragraph.
- Using **window functions** for anything that needs "per-row context of neighbors".
- Reading a **query plan** and reasoning about indexes.
- Writing SQL that a code reviewer will merge without wincing.

**Fundamental principles you must own:**

1. **SQL is declarative.** You describe *what* you want; the planner picks *how*. Your job is to give it a query it can optimize — not to force a specific execution order.
2. **Sets, not rows.** SQL operates on **sets** of rows. If you're mentally iterating row-by-row, you're writing procedural code inside a set-based language, and it will be slow and buggy.
3. **The relational model.** Tables + foreign keys + constraints. Data integrity lives in the database, not in your application code.
4. **Nulls are three-valued.** `NULL` is not a value; comparisons involving `NULL` yield `NULL` (neither true nor false). This changes how you write predicates.
5. **Order of evaluation** (memorize): `FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT`. This explains why you can't reference a `SELECT` alias in `WHERE`.

If you retain nothing else: **think in sets, and read the query plan.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Joins (all of them)

**Definition.** A join combines rows from two tables based on a related column. The join *type* decides how unmatched rows are treated.

Given:
```
users(id, name)                orders(id, user_id, amount)
1  Alice                       10  1  100
2  Bob                         11  1   50
3  Carol                       12  2   75
                               13  99  20  ← no matching user
```

- **INNER JOIN** — rows that match on both sides.
- **LEFT (OUTER) JOIN** — every row from the left; unmatched right side is `NULL`.
- **RIGHT (OUTER) JOIN** — mirror of LEFT (rare in practice; just flip the tables).
- **FULL (OUTER) JOIN** — every row from both sides; unmatched fills with `NULL`.
- **CROSS JOIN** — Cartesian product (every left row × every right row).
- **SELF JOIN** — a table joined to itself (aliased).
- **SEMI JOIN** (`WHERE EXISTS`) — rows on the left where a match exists on the right, without duplicating rows.
- **ANTI JOIN** (`WHERE NOT EXISTS`) — rows on the left where no match exists.

**Example — semi-join for "users who have at least one order":**

```sql
SELECT u.id, u.name
FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);
```

**Technical nuances:**

- **`ON` vs `WHERE`** differ for outer joins: filter conditions on the outer side go in `WHERE`; conditions on the joined side usually belong in `ON`, or you accidentally turn a `LEFT JOIN` into an `INNER JOIN`.
- **Duplicate rows from joins** are a classic bug: if `orders` has 3 rows for user 1, an inner join to `users` produces 3 rows for user 1, and any `SUM(amount)` grouped at the wrong grain will lie. Aggregate first, then join, or use window functions.
- **`USING(col)`** is a shorthand when the join column is named identically in both tables.

---

### 2.2 CTEs, Subqueries, and Query Composition

**Definition.** A **Common Table Expression** (CTE) is a temporary named result set, defined with `WITH`, that you can reference downstream in the same query.

```sql
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount)                     AS revenue
    FROM orders
    GROUP BY 1
),
rolling AS (
    SELECT
        month,
        revenue,
        AVG(revenue) OVER (
            ORDER BY month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS revenue_ma3
    FROM monthly_revenue
)
SELECT * FROM rolling ORDER BY month;
```

**Why CTEs matter:**
- They read top-to-bottom, like a script.
- Each step gets a name — self-documenting.
- Easier to test in isolation (`SELECT * FROM cte_name`).
- **Recursive CTEs** enable graph/tree traversal (org charts, thread trees, transitive closures).

**Recursive example — build the ancestor chain of an employee:**

```sql
WITH RECURSIVE chain AS (
    SELECT id, manager_id, name, 0 AS depth
    FROM employees WHERE id = 42
    UNION ALL
    SELECT e.id, e.manager_id, e.name, c.depth + 1
    FROM employees e
    JOIN chain c ON e.id = c.manager_id
)
SELECT * FROM chain;
```

**Subquery vs CTE.** Both work; CTEs are usually more readable. In some engines (older Postgres < 12) CTEs acted as **optimization fences** — no longer true in modern versions.

---

### 2.3 Window Functions

**Definition.** A window function computes a value **for each row** using a *window* of related rows, without collapsing rows the way `GROUP BY` does.

Skeleton:
```sql
<function>() OVER (
    PARTITION BY <cols>              -- optional: split rows into groups
    ORDER BY     <cols>              -- optional: order within each partition
    <frame>                          -- optional: rows/range window
)
```

**Common functions:**

- **Ranking:** `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `NTILE(n)`, `PERCENT_RANK()`.
- **Analytical:** `LAG(col, k)`, `LEAD(col, k)`, `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE`.
- **Aggregate-as-window:** `SUM`, `AVG`, `COUNT`, `MIN`, `MAX` with an `OVER` clause.

**Example — most recent order per user:**

```sql
SELECT *
FROM (
    SELECT o.*,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date DESC) AS rn
    FROM orders o
) t
WHERE rn = 1;
```

**Example — running total and 7-day moving average:**

```sql
SELECT
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date)                              AS running_total,
    AVG(amount) OVER (ORDER BY order_date
                      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)         AS avg_7d
FROM orders;
```

**Frame formulas.** Given a running sum over ordered rows $x_1, x_2, \ldots$:

$$S_i = \sum_{j=1}^{i} x_j$$

That's `SUM(x) OVER (ORDER BY t)`. A moving average of window $w$:

$$\bar{x}_i^{(w)} = \frac{1}{w}\sum_{j=i-w+1}^{i} x_j$$

That's `AVG(x) OVER (ORDER BY t ROWS BETWEEN w-1 PRECEDING AND CURRENT ROW)`.

**Technical nuances:**

- **`ROWS` vs `RANGE`** — `ROWS` counts physical rows; `RANGE` uses value distance. Different results when your `ORDER BY` column has duplicates.
- Window functions are evaluated **after** `WHERE` and `GROUP BY` but **before** `ORDER BY`. So you can't filter by their result directly — wrap in a subquery/CTE.
- Multiple window functions can share a window with `WINDOW w AS (...)`.

---

### 2.4 Indexes, EXPLAIN, and Performance

**Definition.** An **index** is a supplementary data structure (usually a B-tree) that maps values in one or more columns to row locations, so the database can find matching rows without scanning the whole table.

**Read a plan (Postgres syntax):**

```sql
EXPLAIN ANALYZE
SELECT * FROM orders WHERE user_id = 42;
```

Two extremes you'll see:

- `Seq Scan on orders (cost=... rows=1000000)` — the DB is reading every row. Fine for tiny tables; disastrous for large ones on selective queries.
- `Index Scan using orders_user_id_idx on orders (cost=... rows=17)` — the DB used your index.

**Rules of thumb for indexing:**

- Index **columns used in `WHERE`, `JOIN`, and `ORDER BY`**, especially high-selectivity ones.
- **Composite indexes** on `(a, b)` are usable for queries on `a` alone and `a AND b`, but not `b` alone (leftmost-prefix rule).
- Indexes **cost** on writes: every insert/update/delete has to update every relevant index.
- **Covering indexes** (`INCLUDE` clause in Postgres/SQL Server) let index-only scans return columns without touching the table.
- **Partial indexes** (`WHERE status = 'active'`) shrink the index and speed up hot queries.
- **B-tree** is the default; **hash** is fast for exact equality only; **GIN/GiST** for JSONB, arrays, full-text, geospatial.

**Cardinality intuition — what indexes buy:** For a B-tree of depth $d$ and $n$ rows:

$$d \approx \log_b(n), \quad \text{where } b \approx 100\text{–}200$$

For $n = 10^9$, $d \approx 4\text{–}5$, meaning a lookup touches ~5 pages instead of $10^9$ rows. That's the "orders of magnitude" difference.

---

### 2.5 Data Modeling: Normalization, Constraints, Transactions

**Normalization forms (short version):**

- **1NF** — atomic values (no lists in cells).
- **2NF** — every non-key column depends on the *whole* primary key (matters for composite keys).
- **3NF** — every non-key column depends on the primary key and *nothing else*. No transitive dependencies (`user_id → zip → state` should not put `state` in `users`).
- **BCNF** — stricter 3NF; rare to need explicitly.

**When to denormalize:** OLAP/warehouse contexts often use **star schemas** (fact + dimension tables) that intentionally denormalize dimensions for read-side joins to be cheap.

**Constraints that live in the DB (not the app):**

- `PRIMARY KEY`, `UNIQUE`, `NOT NULL`
- `FOREIGN KEY ... REFERENCES ... ON DELETE {CASCADE | RESTRICT | SET NULL}`
- `CHECK (amount >= 0)`
- Deferrable constraints if needed for cyclic bulk loads.

**Transactions (ACID):**

- **Atomicity** — all or nothing.
- **Consistency** — constraints hold at commit.
- **Isolation** — concurrent transactions don't step on each other (see levels below).
- **Durability** — committed data survives crashes.

**Isolation levels (weakest → strongest):**

1. **Read uncommitted** — can read another tx's uncommitted changes ("dirty reads"). Rare/never used.
2. **Read committed** — default in Postgres. No dirty reads, but non-repeatable reads are possible.
3. **Repeatable read** — same read returns same result within a tx. In Postgres this is actually **snapshot isolation**.
4. **Serializable** — behaves as if transactions ran one at a time. Costs concurrency; may abort with serialization errors that your app must retry.

**Migrations, not raw DDL.** In production, schema changes go through a **migration tool** (Alembic, Flyway, Liquibase). Every migration is a small, reviewed, versioned SQL file. Rollbacks are planned.

---

## 3. Mental Models & Analogies

### 3.1 The "Library Catalog" Model (Indexes)

Imagine a physical library with millions of books. Without a catalog you must walk every shelf to find *"books by author = Achebe"* — that's a sequential scan. The card catalog is an **index**: an alphabetized structure that points to the shelf location. You look up "Achebe" in $O(\log n)$ card flips and walk directly to the book.

- **Composite index** `(author, year)` = catalog sorted first by author, then by year. Great for "Achebe, published 1988"; useless for "any author, published 1988."
- **Covering index with `INCLUDE(title)`** = the card also shows the title, so you don't even need to walk to the shelf.
- **The cost of the catalog** is real: every time you add a book, you must file new cards in every relevant catalog. Too many catalogs and your acquisitions department drowns.

### 3.2 The "Kitchen Order Ticket" Model (Set-based Thinking)

A short-order cook receives a **ticket**: *"table 7 wants 3 burgers, 2 fries, 1 shake."* The cook doesn't process one burger, then walk back to the pass, then process one fry, etc. — he groups the work, cooks all three burgers together, drops all fries together, blends the shake. That's **set-based**.

Procedural row-by-row SQL is like walking back to the ticket for each item, one at a time. It "works" but it's wasteful. Whenever you feel like writing a cursor or a loop over query results, ask: *"can I express this as one operation on the whole set?"* The answer is often a `GROUP BY` or a **window function**.

> 🖼️ **Image Prompt [IMG-SQL-01]:** *"Two side-by-side kitchen scenes. LEFT: a chaotic cook running back and forth to a single ticket, holding one plate at a time — labeled 'Row by row'. RIGHT: a calm cook with a full ticket, all burgers grilling together, all fries in one basket — labeled 'Set based'. Cartoon flat vector style, warm colors, clear labels."*
> **Caption:** SQL is a language for the right-hand cook. Group the work; do it once.
> **Placement:** Section 2.1 / Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "`NULL` Behaves Like Any Other Value"

`WHERE x = NULL` is **always false** (well, `NULL`), and so is `WHERE x != NULL`. Rows with `x IS NULL` slip through both filters. The right predicates are `IS NULL` and `IS NOT NULL`. Similarly `NULL + 5 = NULL`, `NULL OR TRUE = TRUE`, `NULL AND FALSE = FALSE`, but `NULL OR FALSE = NULL`. Reading three-valued logic tables once will save you a decade of confusion.

### 4.2 "`SELECT *` Is Fine in Production"

`SELECT *` bakes the current column list into your query's shape. Add a column and every downstream reader who wrote `INSERT INTO t2 (a, b, c) SELECT * FROM t1` breaks. `*` also inflates network transfer and prevents the planner from using **index-only scans**. Rule: enumerate columns in anything that runs more than once.

### 4.3 "Joining and Then Aggregating Preserves the Grain"

If a user has 3 orders and each order has 4 line items, joining `orders JOIN line_items` on `order_id` produces 12 rows for that user. Then `SUM(orders.amount)` per user *multiplies* the amount by 4 because each order row is now duplicated. The fix: aggregate at the correct grain first (a CTE per grain), then join. Alternatively, use `SUM(DISTINCT ...)` carefully — but distinct-sum is a code smell that usually means the join is wrong.

---

## 5. Self-Assessment Bank (SQL)

### Questions

**Q1 (Short answer).** In what order does SQL logically evaluate `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, and `LIMIT`? Why can't you reference a SELECT alias in `WHERE`?

**Q2 (Multiple choice).** Given:
```sql
SELECT u.id, COUNT(o.id) AS n_orders
FROM users u LEFT JOIN orders o ON o.user_id = u.id AND o.status = 'paid'
GROUP BY u.id;
```
What is the effect of moving `o.status = 'paid'` from the `ON` clause to a `WHERE` clause?
A. No difference.
B. It turns the `LEFT JOIN` into effectively an `INNER JOIN` — users with no paid orders are dropped.
C. It causes a syntax error.
D. It filters both sides of the join.

**Q3 (Short answer).** Explain the difference between `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()`.

**Q4 (Multiple choice).** What does `SUM(amount) OVER (PARTITION BY user_id ORDER BY t)` return?
A. Total amount per user (one row per user).
B. Grand total (one row).
C. Running total of `amount` per user, ordered by `t`, on each row.
D. Same as `SUM(amount) OVER (PARTITION BY user_id)`.

**Q5 (Short answer).** You have a composite index on `(country, city)`. Which of these queries can use it: (a) `WHERE country='US'`, (b) `WHERE city='Helena'`, (c) `WHERE country='US' AND city='Helena'`, (d) `WHERE city='Helena' AND country='US'`?

**Q6 (Multiple choice).** Which is the correct anti-join pattern for "users with no orders"?
A. `SELECT u.* FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE o.id IS NULL`
B. `SELECT u.* FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)`
C. `SELECT u.* FROM users u WHERE u.id NOT IN (SELECT user_id FROM orders)` (with the caveat that this behaves surprisingly if `user_id` can be `NULL`)
D. All of the above (each has trade-offs).

**Q7 (Short answer).** What is a `SETTING` isolation level of `REPEATABLE READ` in Postgres actually implementing under the hood? Give one anomaly it does *not* prevent.

**Q8 (Multiple choice).** You have `orders(id, order_date, amount)` (100M rows), and you frequently run `WHERE order_date BETWEEN ? AND ?`. Which index is most likely optimal?
A. Hash index on `order_date`.
B. B-tree index on `order_date`.
C. B-tree index on `id`.
D. B-tree index on `(amount, order_date)`.

**Q9 (Short answer).** Write a query that returns, for each department, the top-3 highest-paid employees (ties broken by name).

**Q10 (Multiple choice).** Which of the following is *not* an ACID property?
A. Atomicity
B. Consistency
C. Idempotency
D. Durability

---

### Answer Key & Detailed Explanations

**A1.** Logical order: `FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT`. The `SELECT` list is evaluated after `WHERE`/`GROUP BY`/`HAVING`, so an alias created in `SELECT` doesn't exist yet when `WHERE` is evaluated — hence you can use aliases in `ORDER BY` (evaluated later) but not in `WHERE`.

**A2. B.** In `WHERE`, `o.status = 'paid'` filters *after* the join. For users with no orders, `o.status` is `NULL`, and `NULL = 'paid'` is `NULL` → the row is dropped. Keeping the predicate in `ON` filters only the right side during the join, preserving unmatched left rows.

**A3.** Given tied values:
- `ROW_NUMBER()` — assigns a unique sequential integer even to ties (deterministic only with a tiebreaker).
- `RANK()` — ties share a rank, and the next non-tie skips ranks: 1, 2, 2, 4.
- `DENSE_RANK()` — ties share a rank, but the next non-tie does not skip: 1, 2, 2, 3.

**A4. C.** With both `PARTITION BY` and `ORDER BY`, the default frame is `RANGE UNBOUNDED PRECEDING AND CURRENT ROW`, producing a **running total** per partition.

**A5.** The **leftmost-prefix rule** applies. (a) yes — leading column. (b) no — no leading column. (c) yes. (d) yes — the query-planner is smart enough to rewrite `city AND country` to use `(country, city)` since `AND` is commutative. Some engines also allow "index skip scan" for (b) but it is expensive.

**A6. D.** All three work if the underlying columns aren't nullable. The `LEFT JOIN … IS NULL` and `NOT EXISTS` patterns are both idiomatic and often optimize to the same anti-join plan. `NOT IN (SELECT ...)` is dangerous if the subquery ever yields a `NULL`: `x NOT IN (…, NULL, …)` becomes `NULL`, and the row is dropped — even if `x` is not in the list.

**A7.** In Postgres, `REPEATABLE READ` is implemented as **snapshot isolation**: the transaction sees a consistent snapshot at its start. It prevents dirty reads and non-repeatable reads, and even (mostly) phantom reads. It does **not** prevent **write skew**, where two concurrent transactions read overlapping data and each write based on that read, producing an inconsistency neither would produce alone. That's why real applications sometimes need `SERIALIZABLE`.

**A8. B.** Range predicates match B-trees well. A hash index doesn't support ranges. An index on `id` doesn't help this filter. An index on `(amount, order_date)` starts with `amount`, so range on `order_date` alone won't use it via leftmost-prefix.

**A9.**
```sql
WITH ranked AS (
    SELECT emp_id, dept_id, name, salary,
           ROW_NUMBER() OVER (
               PARTITION BY dept_id
               ORDER BY salary DESC, name ASC
           ) AS rn
    FROM employees
)
SELECT emp_id, dept_id, name, salary
FROM ranked
WHERE rn <= 3
ORDER BY dept_id, rn;
```
(Use `RANK()` instead if you want ties to *all* be included even beyond 3.)

**A10. C.** Idempotency is a property of an operation ("running it twice = running it once"), not of transactions. ACID = Atomicity, Consistency, Isolation, Durability.

---

## 6. Practice Prompts

1. On a Postgres sandbox with `pg_stat_statements` on, run a slow query, `EXPLAIN ANALYZE` it, add an appropriate index, and measure the speedup.
2. Design a schema for a simple bill-drafting workflow (bills, versions, sponsors, sessions, comments). Enumerate constraints. Load 100k synthetic rows and write 5 useful queries.
3. Write a recursive CTE that turns an employee `manager_id` chain into a materialized ancestor path.
4. Rewrite a 60-line stored procedure that uses cursors into one set-based CTE query. Measure both.
5. Set up Alembic for a Python project, define two tables, and evolve the schema across three migrations. Practice `alembic downgrade`.

---

## 7. References

- Markus Winand, *SQL Performance Explained* (or its free web version, *Use the Index, Luke*)
- Joe Celko, *SQL for Smarties*
- PostgreSQL documentation, particularly [Performance Tips](https://www.postgresql.org/docs/current/performance-tips.html) and [MVCC](https://www.postgresql.org/docs/current/mvcc.html)
- Mode Analytics SQL Tutorial (free, hands-on)
