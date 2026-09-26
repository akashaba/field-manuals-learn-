# Git & GitHub — Master Study Guide

> **Track:** Foundations · **Module:** 04
> **Prerequisites:** Comfortable in a terminal.
> **Time budget:** ~15–20 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every professional engineer's day is scaffolded by version control. Git is not "the tool that saves my code" — Git is a **content-addressable filesystem with a directed acyclic graph of snapshots** on top. Once you internalize that, the surface command-line becomes a set of graph operations rather than a set of memorized incantations.

GitHub, then, is *one hosted implementation* of the collaborative rituals — pull requests, code review, CI, issue tracking — that scale Git across teams. The two are separable: Git is the version-control system; GitHub is the (currently dominant) platform.

**Fundamental principles you must own:**

1. **Git tracks content, not files.** Every unique blob is stored once (deduplicated by SHA). Renames are inferred, not tracked.
2. **A commit is a snapshot, not a diff.** Diffs are computed on demand between two snapshots.
3. **Everything is local first.** `git commit` records to your local repo; `git push` copies commits to a remote. This is why Git is fast and works offline.
4. **Branches are pointers.** A branch is a movable label pointing to a commit. Merging and rebasing are graph operations on those pointers.
5. **History is (mostly) rewritable — before you push.** Once shared, history should be treated as immutable to avoid confusing collaborators.

If you retain nothing else: **draw the DAG.** Any Git problem becomes tractable the moment you sketch the commit graph on paper.

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Git Object Model

Git stores four kinds of objects, each identified by a **SHA-1 (or SHA-256)** hash of its content:

- **Blob** — a file's contents (no name, no metadata).
- **Tree** — a directory: a list of `(mode, name, sha)` entries pointing to blobs or subtrees.
- **Commit** — a snapshot: `tree` + `parent(s)` + `author` + `committer` + `message`.
- **Tag** — a named, optionally-signed pointer to another object (usually a commit).

```
   ┌──────────┐
   │  commit  │─── parent ──▶ (older commit)
   │  msg,    │
   │  author  │─── tree ────▶ ┌──────────┐
   └──────────┘                │   tree    │
                               │  src/     │─▶ ┌──────────┐
                               │  README   │   │  blob    │
                               └──────────┘   └──────────┘
```

**Practical implications:**

- Because content is hashed, **identical files across branches share storage**. Copying a 100 MB file into 5 branches costs 100 MB, not 500 MB.
- **`git gc`** repacks loose objects into pack files with delta compression to save space.
- **The staging area** ("index") is *itself* a tree object under construction between working directory and next commit.

**Under the hood, plumbing commands:** `git hash-object`, `git cat-file`, `git ls-tree`, `git rev-parse`. Poke at `.git/objects/` once and everything clicks.

---

### 2.2 Branches, Merges, and Rebases

**A branch is a movable pointer.** `HEAD` is a pointer *to* the current branch (or directly to a commit in "detached HEAD" state).

**Fast-forward merge** — when the target branch has no new commits, Git just moves the pointer forward:

```
before:                 after fast-forward merge of feature into main:
main ────A                     main ────A────B────C
         └───B────C ← feature                    ▲
                                                 └ feature
```

**Three-way merge** — when both branches have new commits, Git computes the common ancestor and produces a **merge commit** with two parents:

```
main ────A────D────E        →     main ────A────D────E────M
              └───B────C                          └───B────C ─┘
                        ↑                                    (M has parents E and C)
                     feature
```

**Rebase** — replays your commits *on top* of another branch, producing a linear history:

```
before:                        after `git rebase main` on feature:
main ────A────D────E           main ────A────D────E
              └───B────C                          └───B'────C'
```

`B'` and `C'` are **new commits** with the same content but new parent pointers → new SHAs. This is why rebasing shared branches is dangerous: collaborators still have `B` and `C`.

**Common workflows:**

- **Feature branch + merge PR** — cheap and simple; history has merge bubbles.
- **Trunk-based with rebase** — linear history; each PR is squashed or rebased before merging.
- **GitFlow** — heavy, with `develop`, `release/*`, `hotfix/*` branches; rare in modern setups.

**Merge conflicts:** They're a *feature*, not a bug. Git only asks you when both sides changed the same hunk. Resolve carefully, remove conflict markers, `git add` the resolved files, `git commit` (or `git rebase --continue`).

---

### 2.3 The Distributed Model: Remotes, Fetch, Pull, Push

**Definitions:**

- A **remote** is a named URL of another copy of the repo (`origin`, `upstream`, ...).
- **`git fetch`** downloads objects and updates remote-tracking branches (`origin/main`), but **does not** touch your working tree.
- **`git pull`** = `git fetch` + `git merge` (or `git rebase` if configured).
- **`git push`** uploads your local commits to a remote branch.

**Tracking branches.** Your local `main` can be set to *track* `origin/main`. Then `git status` tells you "ahead 2, behind 3", and `git pull` / `git push` know where to go.

**Forks and upstreams.** On GitHub, a **fork** is a server-side clone. Your workflow:

1. Clone your fork as `origin`.
2. Add the original repo as `upstream` (`git remote add upstream ...`).
3. Sync frequently: `git fetch upstream && git rebase upstream/main`.

**Pull requests** are GitHub's UI on top of "please review my branch and merge it into yours." A good PR:

- Solves one problem.
- Is small enough to review in <30 minutes.
- Has a description explaining *why*, not just *what*.
- Passes CI (linters, tests, type-checks) before requesting review.
- Uses draft PRs for work-in-progress feedback.

---

### 2.4 Rewriting History Safely

Git offers powerful, controlled history-editing. Use these on **your own unpushed branch**, or on shared branches only with team coordination.

**Amend the last commit** — fix a typo or add a forgotten file:
```bash
git add path/to/fixed_file
git commit --amend --no-edit    # keeps old message
```

**Interactive rebase** — reorder, squash, edit, drop commits:
```bash
git rebase -i HEAD~5
```
Options per commit: `pick`, `reword`, `edit`, `squash`, `fixup`, `drop`.

**Cherry-pick** — apply one commit from another branch:
```bash
git cherry-pick abc1234
```

**Reset** — move `HEAD` (and optionally the index/working tree) backward or forward:
- `git reset --soft HEAD~1` — undo the last commit; keep changes staged.
- `git reset --mixed HEAD~1` (default) — undo the commit; unstage changes.
- `git reset --hard HEAD~1` — **destructive**. Discards commit and changes.

**Revert** — create a new commit that undoes the changes of another. Preferred on shared history because it doesn't rewrite.

**Reflog** — Git remembers every position of `HEAD` for ~90 days. Your safety net:
```bash
git reflog
git reset --hard HEAD@{5}     # jump back
```

**Bisect** — binary search for the commit that introduced a bug:
```bash
git bisect start
git bisect bad
git bisect good v1.2.0
# Git checks out a mid commit; you test; you say `bisect good` or `bad`.
# Repeated log2(n) times until it names the culprit.
```

---

### 2.5 Collaboration on GitHub: PRs, Reviews, CI, Releases

**Pull requests** are conversations attached to a diff. Master these:

- **Title/description conventions** — imperative mood ("Add rate limiting"), link the issue, describe rationale.
- **Draft PRs** — early feedback without pinging reviewers.
- **Requested reviewers, CODEOWNERS** — enforce ownership.
- **Suggestions in review** — `Ctrl+G` gives you inline "suggested change" blocks the author can commit with one click.
- **Squash vs merge vs rebase merge** — pick per repo policy; squash is common for small teams.

**GitHub Actions (CI).** A minimal starter:

```yaml
# .github/workflows/ci.yml
name: ci
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy src
      - run: pytest -q
```

**Branch protection rules:**

- Require PR review.
- Require passing status checks.
- Disallow force-push to `main`.
- Require linear history (if you want to enforce rebase workflow).

**Releases and tags:**

- **Annotated tag** — `git tag -a v1.0.0 -m "First release"`.
- **Release** on GitHub — a UI object attached to a tag, with notes and downloadable assets.
- **Semantic versioning** — `MAJOR.MINOR.PATCH`; breaking, feature, fix.

---

## 3. Mental Models & Analogies

### 3.1 The "Notebook of Photographs" Model (Commits as Snapshots)

Imagine a photographer who, every so often, takes a full photograph of a scene and pastes it into a notebook. Each page is dated, captioned (the commit message), and refers back (an arrow) to the previous page. The photographer never crosses out old pages — they just add new ones.

- **Branches** are colored ribbons hanging from a specific page. Move the ribbon to a newer page — nothing else changes.
- **`git checkout`** is turning the notebook to the page that ribbon points to.
- **Merging** clips two ribbons together with a new page glued in between, whose arrow points to *both* previous pages.
- **Rebasing** photocopies pages onto a different part of the notebook with the arrows redirected — the originals are still there for a while (reflog), then eventually recycled by `gc`.

This model demolishes the "diffs are stored" misconception. Diffs are just derived by comparing photographs.

### 3.2 The "Airport Baggage Carousel" Model (Staging Area)

Three areas, one bag:

- **Working directory** — the bag sitting on your hotel bed. You're throwing shirts in and out.
- **Index / staging area** — the bag on the carousel about to be loaded. You've decided what's going.
- **Repository** — the plane. Once loaded (`commit`), the bag flies.

The commands map naturally:

- `git add` — put the item into the carousel bag.
- `git restore --staged <file>` — take it back out of the carousel bag.
- `git commit` — load the plane.
- `git checkout -- <file>` (or `git restore <file>`) — throw away hotel-bed changes.
- `git diff` — what's on the bed vs the carousel.
- `git diff --staged` — what's in the carousel vs the last flight.

Once you see the three-stage separation, `git status`'s prose stops feeling like riddles.

> 🖼️ **Image Prompt [IMG-GIT-01]:** *"A clear illustrated diagram of Git's three areas: Working Directory (a laptop with unstaged edits), Staging Area (a suitcase being packed), and Repository (a filing cabinet with a history of snapshots). Arrows labeled 'git add', 'git commit', 'git restore', 'git checkout' between them. Flat modern illustration, teal and orange palette, white background."*
> **Caption:** Files move rightward through three areas; each Git command targets a specific arrow.
> **Placement:** Section 2.1 / Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Rebasing a Shared Branch"

If you rebase a branch that other people have already based work on, their history now diverges. `git pull --rebase` on their side will produce duplicate commits and mystery conflicts. **Rule:** rebase only your local, unshared branches. Once pushed and someone else pulls, prefer merge.

### 4.2 "`git pull` Is Always Safe"

Default `git pull` runs `merge`, which produces messy merge commits on every trivial sync. Worse, if you have uncommitted changes, the merge can fail in ambiguous ways. Configure `pull.rebase = true` (or `pull.ff = only`) globally, keep your working tree clean before pulling, and commit or stash any WIP first.

### 4.3 "`git commit` Saves My Work to the Server"

No. Commits are **local**. Until you `git push`, your work exists only on your machine. If your laptop dies, so does the commit. This surprises newcomers who think Git is Dropbox. Corollary: `git clone` copies the *whole history* from the remote to your machine — that's why Git works offline.

---

## 5. Self-Assessment Bank (Git & GitHub)

### Questions

**Q1 (Short answer).** What are the four object types in Git and how do they relate?

**Q2 (Multiple choice).** After `git checkout feature; git rebase main`, which is true?
A. `feature` now has new commits with new SHAs but the same changes.
B. `main` has been merged into `feature` with a merge commit.
C. `main` has moved to `feature`'s tip.
D. Nothing changes.

**Q3 (Short answer).** What is the difference between `git reset --soft`, `--mixed`, and `--hard`?

**Q4 (Multiple choice).** You committed a secret API key to `main` and pushed. Which of the following is the CORRECT response?
A. `git commit --amend` to remove it and force-push.
B. Rotate the key, then use `git filter-repo` (or BFG) to purge it from history, then force-push, then coordinate with all clone holders.
C. Delete the file in a new commit; the secret is safe.
D. Ignore it — it's already public and can't be un-published, but past history hides it.

**Q5 (Short answer).** Explain what "detached HEAD" state means and one legitimate reason to be in it.

**Q6 (Multiple choice).** Which command finds the commit that introduced a bug via binary search over history?
A. `git blame`
B. `git log --grep`
C. `git bisect`
D. `git diff-tree`

**Q7 (Short answer).** What is the difference between `git fetch` and `git pull`?

**Q8 (Multiple choice).** You force-pushed and destroyed a commit locally. Can you get it back?
A. Yes, from `git reflog` within ~90 days.
B. Yes, always — Git never deletes anything.
C. No — force-push is permanent.
D. Only if someone else had cloned the branch.

**Q9 (Short answer).** What does `git cherry-pick` do, and when is it the right tool?

**Q10 (Multiple choice).** A "fast-forward merge" happens when:
A. The target branch has no new commits since your branch diverged.
B. Both branches have new commits and Git creates a merge commit.
C. You rebase before merging.
D. You use `--squash`.

---

### Answer Key & Detailed Explanations

**A1.** **Blob** (file contents), **tree** (directory listing pointing to blobs and subtrees), **commit** (snapshot pointing to a tree, with parent(s) and metadata), **tag** (a named pointer, often signed). A commit's `tree` gives the root directory snapshot; that tree references sub-trees and blobs; commits chain via their `parent` field. Tags don't participate in the chain; they just name a specific object.

**A2. A.** Rebase replays your commits on top of the target, producing new commits with new SHAs but identical content. The old commits still exist temporarily in the reflog.

**A3.** All three move `HEAD` (and the current branch pointer) to a target commit. They differ in what they do to the **index** and **working tree**:
- `--soft` — moves the branch pointer only. Index and working tree unchanged (changes are staged).
- `--mixed` — moves the pointer and resets the index. Working tree unchanged (changes are unstaged).
- `--hard` — moves everything. **Uncommitted changes are lost.**

**A4. B.** Once a secret is pushed, treat it as compromised — **rotate it immediately** (that is the highest-priority step). Purging from history requires rewriting all objects that contain the file (`git filter-repo` or BFG), force-pushing, and asking every clone holder to re-clone or run the same rewrite. GitHub also caches, and the secret may already be scraped — hence rotation is non-negotiable.

**A5.** "Detached HEAD" means `HEAD` points directly to a commit rather than to a branch. Any commits you make aren't attached to any branch, so if you check out something else, they're orphaned (reachable only via reflog until gc'd). Legitimate uses: inspecting an old commit read-only, cutting a temporary release build, running tests at a specific point in history.

**A6. C.** `git bisect` runs a binary search over commits, asking you at each step whether the current one is good or bad. It converges in ~log₂(n) steps.

**A7.** `git fetch` downloads objects and updates remote-tracking refs (`origin/main`) — but does **not** modify your working branch. `git pull` is `git fetch` followed by a `merge` (or `rebase`, if configured) into your current branch. Fetch is safe; pull mutates.

**A8. A.** `git reflog` records local `HEAD` movements for about 90 days by default (`gc.reflogExpire`). Even after a force-push, the commit is still in your local objects and reflog. You can `git reset --hard <sha>` back to it. But if you deleted the local clone, the commit may only survive on someone else's copy.

**A9.** `cherry-pick` applies the changes of a specific commit onto your current branch as a new commit. It's the right tool when you want *this specific change* from another branch without merging in everything else — e.g., backporting a bug fix from `main` to a `release/1.4` branch. Warning: if the commit depends on prior commits in its branch, cherry-picking alone may not compile.

**A10. A.** A fast-forward is only possible when your commits are strictly ahead of the target. Git just slides the pointer forward; no merge commit is created.

---

## 6. Practice Prompts

1. Initialize an empty repo. Use plumbing commands (`hash-object`, `update-index`, `write-tree`, `commit-tree`) to create a commit **without ever running `git add` or `git commit`**. You will never fear Git again.
2. Create a feature branch, make 5 commits, then interactively rebase them into 2 well-worded commits with `git rebase -i`.
3. Introduce a bug 20 commits back. Use `git bisect` with a shell script to find it automatically.
4. Set up a GitHub repo with branch protection, a CI workflow, CODEOWNERS, and a template PR. Open a PR against yourself and iterate.
5. Simulate a merge conflict, resolve it, then simulate the *same* conflict during a rebase, and resolve that. Notice the difference in mental model.

---

## 7. References

- Scott Chacon & Ben Straub, *Pro Git* (free online at git-scm.com/book)
- Julia Evans, *"Oh Shit, Git!"* zines / articles
- GitHub Docs on Actions, PRs, and branch protection
- Talk: Linus Torvalds, "Google Tech Talk on Git" (2007) — old but foundational
