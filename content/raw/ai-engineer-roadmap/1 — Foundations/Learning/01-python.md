# Python (Properly) — Master Study Guide

> **Track:** Foundations · **Module:** 01
> **Prerequisites:** Basic programming familiarity in any language.
> **Time budget:** ~25–35 hours of focused practice.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Python is the *lingua franca* of data, ML, scripting, and increasingly backend services. But the gap between "I can write a notebook that trains a model" and "I can ship a Python service that a team relies on" is enormous. That gap is filled by understanding Python **as a language and a tooling ecosystem**, not just as a syntax to copy-paste from Stack Overflow.

Learning Python "properly" means internalizing five ideas:

1. **Everything is an object.** Functions, classes, modules, and even *types* are first-class values. Once this clicks, decorators, metaclasses, and `functools.partial` stop feeling magical.
2. **Names bind, they don't hold.** Variables in Python are **references** to objects, not boxes. This one distinction resolves 80% of "why did this list mutate?" bugs.
3. **Iteration is a protocol.** `for x in y` works because `y` implements `__iter__`. Generators, iterators, and lazy pipelines are the same idea under different masks.
4. **Duck typing plus type hints.** Python at runtime asks *"can you quack?"*, but its type-checker asks *"were you declared a duck?"* Idiomatic Python 3.11+ code writes for both audiences.
5. **The Python packaging story is a real thing.** `pyproject.toml`, virtual environments, entry points, and `pip install -e .` are not optional for production work — they are the difference between a script and a **project**.

**Fundamental principles you must own:**

- **PEP 8** — style. Read once, then let `ruff`/`black` enforce it.
- **PEP 20 (The Zen of Python)** — philosophy. `import this`. Especially: *"Explicit is better than implicit"* and *"Errors should never pass silently."*
- **PEP 484 / PEP 604** — type hints. `def add(a: int, b: int) -> int:` and `x: int | None = None`.
- **The reference model** — mutable vs immutable, identity (`is`) vs equality (`==`).
- **The GIL (Global Interpreter Lock)** — a single Python thread executes bytecode at a time in CPython. Concurrency uses `asyncio` / multiprocessing for CPU-bound work.

If you retain nothing else: **Python is small at its core (about 35 keywords), and every "advanced" feature is a composition of the small core.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Object & Reference Model

**Definition.** Every value in Python is an object with an `id`, a `type`, and a `value`. Names (variables) are *labels* attached to those objects.

```python
a = [1, 2, 3]
b = a              # b now labels the same list object
b.append(4)
print(a)           # [1, 2, 3, 4]  ← surprising if you think a is a "box"
print(a is b)      # True  ← same object
print(id(a) == id(b))  # True
```

**Technical nuances:**

- **Mutable** types (`list`, `dict`, `set`, most user classes) can be changed in place → aliasing bugs.
- **Immutable** types (`int`, `float`, `str`, `tuple`, `frozenset`) can never be changed; "modification" always produces a new object.
- **`is` checks identity**, **`==` checks equality**. `None`, `True`, `False` are singletons; always use `is None`.
- **Default arguments are evaluated once** at function-definition time:

  ```python
  def bad(items=[]):        # ← the SAME list every call
      items.append(1)
      return items

  bad(); bad(); bad()       # returns [1, 1, 1] on the third call — a landmine
  ```

  Fix with the sentinel pattern:

  ```python
  def good(items: list[int] | None = None) -> list[int]:
      items = [] if items is None else items
      items.append(1)
      return items
  ```

- **Copying**: `list(x)`, `x.copy()`, `copy.copy(x)` are *shallow*. `copy.deepcopy(x)` recurses.

**Real-world example.** In a Flask/FastAPI handler, if you use a mutable default for a query-parameter list, every request will see the previous request's data. This is a *real* production incident many teams have hit.

---

### 2.2 Functions, Closures, Decorators

**Definition.** Functions are objects. They can be assigned to variables, passed as arguments, returned, and *close over* enclosing-scope variables.

```python
def make_multiplier(k: int):
    def multiply(x: int) -> int:
        return x * k          # k is closed over from the enclosing scope
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
double(10)   # 20
triple(10)   # 30
```

**Decorators** are just functions that take a function and return a function:

```python
import functools, time

def timeit(fn):
    @functools.wraps(fn)             # preserves __name__, __doc__, __wrapped__
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            print(f"{fn.__name__} took {time.perf_counter() - t0:.4f}s")
    return wrapper

@timeit
def slow_add(a, b):
    time.sleep(0.1)
    return a + b
```

**Technical nuances:**

- `*args` and `**kwargs` capture positional and keyword arguments respectively.
- `functools.wraps` is **not optional**; without it, decorated functions lose metadata and break in Sphinx, `inspect`, IDE tooltips, and `pytest`.
- Class-based decorators (`class Timer: def __call__(self, fn): ...`) let you carry state.
- Decorators with arguments require *three* levels of nesting (factory → decorator → wrapper).

**Real-world example.** FastAPI's `@app.get("/users/{id}")` is a decorator factory. Understanding decorators is how you stop treating web frameworks as magic.

---

### 2.3 Iterators, Generators, and Lazy Pipelines

**Definition.** An **iterator** is any object with `__iter__` (returning itself) and `__next__` (returning the next value or raising `StopIteration`). A **generator** is a compact way to write one, using `yield`.

```python
def read_large_file(path: str):
    with open(path, "r") as f:
        for line in f:        # file is itself an iterator, one line at a time
            yield line.rstrip("\n")

for row in read_large_file("huge.csv"):
    process(row)
```

**Generators are lazy** — no memory is used until you pull.

**Technical nuances:**

- Generator expressions use `(x*x for x in nums)` — parentheses, not brackets — and are lazy.
- `itertools` is Python's most under-loved standard-library module. Learn `chain`, `islice`, `groupby`, `product`, `combinations`, `accumulate`, `tee`.
- Generators can be *pipelined*:

  ```python
  lines   = read_large_file("log.txt")
  errors  = (l for l in lines if "ERROR" in l)
  parsed  = (json.loads(l.split(" ", 1)[1]) for l in errors)
  first10 = itertools.islice(parsed, 10)
  ```

  Nothing is read from disk until you iterate `first10`.
- **`yield from`** delegates to a sub-iterator: `yield from other_gen()`.
- Coroutines (`async def`) build on the generator machinery.

**Real-world example.** Streaming a 50 GB dataset row-by-row through a `pandas.read_csv(..., chunksize=100_000)` iterator is generator-thinking applied to Pandas. Without it, you OOM.

---

### 2.4 Type Hints, Dataclasses, and Protocols

**Definition.** Type hints (PEP 484+) annotate names with expected types. They are *not enforced at runtime* by the interpreter — they are checked by tools like `mypy` and `pyright`.

```python
from dataclasses import dataclass, field
from typing import Protocol

@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float
    tags: list[str] = field(default_factory=list)

class SupportsArea(Protocol):
    def area(self) -> float: ...

def total_area(shapes: list[SupportsArea]) -> float:
    return sum(s.area() for s in shapes)
```

**Technical nuances:**

- **`frozen=True`** makes the dataclass immutable (like a `namedtuple` but with methods and typing).
- **`slots=True`** replaces the per-instance `__dict__` with a fixed slot layout → smaller memory, faster attribute access.
- **`Protocol`** enables **structural** typing: any class with an `area()` method matches, no inheritance required. Duck typing meets the type checker.
- **`TypedDict`** annotates dictionary shapes (great for JSON payloads).
- **`Literal["a", "b"]`**, **`Final`**, **`TypeAlias`**, **`ParamSpec`** are power tools worth knowing.

**Real-world example.** A FastAPI `pydantic.BaseModel` is essentially a dataclass with runtime validation added. Knowing dataclass semantics makes you fluent in Pydantic immediately.

---

### 2.5 Errors, Context Managers, and Resource Discipline

**Definition.** Python uses **exceptions** for error signaling and **context managers** (`with` statement) for guaranteed resource cleanup.

```python
class RetryableError(Exception):
    pass

def fetch(url: str) -> bytes:
    try:
        return http.get(url).content
    except TimeoutError as e:
        raise RetryableError(f"timeout on {url}") from e  # chain, don't swallow
    except Exception:
        # unknown → let it propagate
        raise
```

Context managers:

```python
from contextlib import contextmanager

@contextmanager
def db_transaction(conn):
    tx = conn.begin()
    try:
        yield tx
        tx.commit()
    except Exception:
        tx.rollback()
        raise

with db_transaction(conn) as tx:
    tx.execute("INSERT ...")
```

**Technical nuances:**

- **Never `except:` bare** — catches `KeyboardInterrupt` and `SystemExit`. Use `except Exception:`.
- **Chain exceptions with `raise X from e`** so the original traceback survives.
- Prefer specific exceptions over broad ones. Handle only what you know how to handle.
- `contextlib.ExitStack` lets you enter a dynamic number of context managers.
- `finally` runs whether or not an exception occurred; use it for cleanup that must always happen.

**Real-world example.** File handles, DB connections, sockets, GPU memory, distributed locks — all use context managers. Learning to write your own is the difference between "leaks in prod after 4 days" and "never leaks."

---

## 3. Mental Models & Analogies

### 3.1 The "Name Tag on a Box" Model (Reference Semantics)

Imagine every Python object lives in a giant warehouse. When you write `a = [1, 2, 3]`, Python creates a *crate* labeled `[1, 2, 3]` and puts a **sticky note** with your name `a` on it. When you write `b = a`, Python peels off a second sticky note and puts it on the *same crate*. There's still one crate, now with two notes.

- **Mutating** (`b.append(4)`) rearranges what's inside the crate. Both notes still point to it — so `a` "sees" the change.
- **Rebinding** (`b = [7, 8]`) moves only the `b` sticky note to a different crate. `a`'s crate is unchanged.

The `is` operator asks *"same crate?"*; the `==` operator asks *"same contents?"*.

> 🖼️ **Image Prompt [IMG-PY-01]:** *"A clean isometric illustration of a warehouse with three labeled wooden crates. One crate holds the numbers 1, 2, 3. Two sticky notes labeled 'a' and 'b' are attached to the same crate with arrows pointing to it. A third crate holds 7, 8 and is unlabeled. Textbook illustration style, soft pastel colors, white background, minimal shadow."*
> **Caption:** Names are sticky notes, not boxes — assignment binds a new note; mutation edits the crate.
> **Placement:** Section 2.1 (Object & Reference Model).

### 3.2 The "Restaurant Kitchen" Model (Generators & Lazy Evaluation)

A **list** is a buffet: the cook prepares every dish before the doors open. If you only wanted the first salad, you still paid for all 100 dishes and the fridge is stuffed.

A **generator** is à-la-carte: the cook does nothing until you order. Each `yield` is one plate carried out. When you leave, the kitchen shuts down — no waste, no fridge blowout.

- `nums = [x*x for x in range(10**8)]` — buffet. Kills your RAM.
- `nums = (x*x for x in range(10**8))` — à-la-carte. Uses ~200 bytes.
- `sum(x*x for x in range(10**8))` — you order every plate, but only one is in your hand at a time.

The mental shift: **think in flows, not in containers.** Whenever you find yourself writing `list(...)` around a generator, ask "do I actually need the whole list, or am I about to iterate it once anyway?"

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Mutable Default Arguments Are Fresh Each Call"

Beginners write:

```python
def append_to(x, target=[]):
    target.append(x)
    return target
```

...expecting a fresh list every call. They get a **shared, growing list across calls** because default arguments are evaluated **once**, when the `def` statement runs. This is one of Python's most infamous gotchas and it is on every senior interview.

**Fix:** Use `None` as a sentinel; construct fresh inside the body.

### 4.2 "`is` and `==` Are the Same Thing"

They aren't. `is` is identity (same object). `==` is equality (equivalent value). Beginners write `x is None` correctly by accident, then write `if name is "Brian":` and get burned when Python's string interning quietly stops folding their string and the branch never runs.

**Rule of thumb:** Use `is` only for `None`, `True`, `False`, and *sentinels you defined*. Use `==` for values.

### 4.3 "The GIL Means Python Is Slow"

The GIL means CPython executes one thread of *Python bytecode* at a time. But:

- **I/O-bound code** (network, disk) releases the GIL during the wait — threads and `asyncio` scale fine.
- **CPU-bound numeric code** in NumPy, TensorFlow, PyTorch releases the GIL inside C extensions.
- For CPU-bound *pure Python*, use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`.

Beginners either (a) reach for threads and are surprised nothing speeds up, or (b) conclude "Python can't do concurrency" and switch languages prematurely. Understand *which kind of workload you have* before choosing a concurrency tool.

---

## 5. Self-Assessment Bank (Python)

### Questions

**Q1 (Short answer).** Explain the difference between `is` and `==`. Give one case where they return different answers.

**Q2 (Multiple choice).** What does this print?
```python
def f(x, xs=[]):
    xs.append(x)
    return xs

print(f(1)); print(f(2)); print(f(3))
```
A. `[1] [2] [3]`
B. `[1] [1, 2] [1, 2, 3]`
C. `[1] [2] [1, 2, 3]`
D. `TypeError`

**Q3 (Short answer).** Why do we use `functools.wraps` when writing a decorator? What breaks without it?

**Q4 (Multiple choice).** Which of the following is a *lazy* expression?
A. `[x*x for x in range(10)]`
B. `{x*x for x in range(10)}`
C. `(x*x for x in range(10))`
D. `list(map(lambda x: x*x, range(10)))`

**Q5 (Short answer).** What is the difference between a *shallow* copy and a *deep* copy? Give an example where the difference matters.

**Q6 (Multiple choice).** In `try/except/else/finally`, the `else` clause runs when:
A. Any exception occurred.
B. No exception occurred in the `try` block.
C. The `try` block returned early.
D. The interpreter is shutting down.

**Q7 (Short answer).** Explain what the GIL is and one scenario where it doesn't hurt performance.

**Q8 (Multiple choice).** Which is the correct way to define an immutable dataclass with fixed attribute slots?
A. `@dataclass(frozen=True)` alone
B. `@dataclass(slots=True)` alone
C. `@dataclass(frozen=True, slots=True)`
D. `@dataclass(readonly=True)`

**Q9 (Short answer).** Write, in one line, an expression that returns the sum of squares of the first 1,000,000 integers **without materializing them into a list**.

**Q10 (Multiple choice).** What is a `Protocol` in typing terms?
A. A network protocol wrapped as a class.
B. A structural type — anything with the right methods satisfies it, no inheritance required.
C. A nominal type — you must explicitly subclass to satisfy it.
D. A runtime interface check enforced by the interpreter.

---

### Answer Key & Detailed Explanations

**A1.** `is` compares object **identity** (do two names refer to the same object in memory?). `==` compares **equality** (do two objects have equivalent values, as defined by `__eq__`?). Divergence example: `a = [1, 2]; b = [1, 2]` — here `a == b` is `True` but `a is b` is `False`. Another: small ints are cached in CPython so `256 is 256` is `True` but `257 is 257` is *implementation-defined* — never rely on it.

**A2. B.** The default list `xs=[]` is created **once** when `def` executes. Every call that doesn't pass `xs` mutates the same list. Output: `[1]`, then `[1, 2]`, then `[1, 2, 3]`.

**A3.** `functools.wraps` copies the wrapped function's `__name__`, `__doc__`, `__qualname__`, `__module__`, `__wrapped__`, and `__dict__` onto the wrapper. Without it, decorated functions all appear as `wrapper` — breaking IDE help, Sphinx docs, `inspect.signature`, `pytest` collection, and stack traces.

**A4. C.** Parentheses build a **generator expression**, which is lazy. `[]` builds a list eagerly; `{}` a set eagerly; `list(map(...))` forces materialization. A bare `map(...)` object is also lazy, but option D wraps it in `list()`.

**A5.** A **shallow copy** duplicates the outer container but references to nested objects are shared. A **deep copy** recursively duplicates everything. Example:
```python
a = [[1, 2], [3, 4]]
b = a.copy()          # shallow
b[0].append(99)
print(a)              # [[1, 2, 99], [3, 4]] — a was mutated!
import copy
c = copy.deepcopy(a)
c[0].append(0)
print(a)              # unchanged
```

**A6. B.** `else` runs only when the `try` block completed **without** raising an exception (and without returning). It's useful for code that should run only in the success path but shouldn't be inside `try` (to avoid catching its exceptions accidentally).

**A7.** The **Global Interpreter Lock** is a mutex in CPython that ensures only one thread executes Python bytecode at a time. It does not hurt performance when the workload is **I/O-bound** (network calls, file reads) because the GIL is released during the wait — threads can overlap I/O. It also doesn't hurt performance when heavy numerical work happens in C extensions (NumPy, PyTorch) that release the GIL during their C code.

**A8. C.** `frozen=True` prevents attribute assignment; `slots=True` replaces the `__dict__` with fixed slots. Combine them for a compact, immutable value type. (Note: `slots=True` was added in Python 3.10.)

**A9.** `sum(i*i for i in range(1_000_000))`. The generator expression means only one squared int exists at a time; peak memory is O(1).

**A10. B.** `typing.Protocol` (PEP 544) is **structural**: a class satisfies a `Protocol` if it has methods with matching names and signatures, regardless of inheritance. This formalizes duck typing for static type-checkers. Options A/C/D describe the wrong model.

---

## 6. Practice Prompts (do these, don't just read them)

1. Take a Jupyter notebook you wrote in the past. Convert it to an installable package with `src/` layout, `pyproject.toml`, and one CLI entry point via `console_scripts`.
2. Write a decorator `@retry(tries=3, delay=0.5, backoff=2.0)` with proper `functools.wraps` and type hints.
3. Write a generator pipeline that reads a large log file, filters lines containing `"ERROR"`, parses their JSON tail, and yields dicts — without ever holding the whole file in memory.
4. Write a `Point3D` frozen dataclass with a method `distance_to(other) -> float`. Add `__slots__`. Benchmark it against a plain class with `%timeit`.
5. Run `mypy --strict` over a small module of yours and fix every error without using `# type: ignore`.

---

## 7. References

- Fluent Python, 2nd ed. — Luciano Ramalho
- Python docs: [The Language Reference](https://docs.python.org/3/reference/)
- PEP 8, PEP 20, PEP 484, PEP 517/518/621
- Real Python — decorators, context managers, typing tutorials
- Talks: Raymond Hettinger's "Transforming Code into Beautiful, Idiomatic Python"
