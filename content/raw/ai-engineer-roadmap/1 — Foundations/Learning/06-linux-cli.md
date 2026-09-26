# Linux & the Command Line — Master Study Guide

> **Track:** Foundations · **Module:** 06
> **Prerequisites:** A terminal and willingness to break things safely.
> **Time budget:** ~15–20 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every server you'll ever deploy to, every container you'll run, and every dev tool worth using talks Unix. The command line is where you glue systems together, debug production, orchestrate pipelines, and script the boring parts of your life. Notebooks and IDEs are conveniences on top of this substrate; when they break — and they will — you need to see under the hood.

Learning Linux "properly" is not memorizing 500 commands. It's internalizing:

- **The Unix philosophy** — small, composable, text-in-text-out programs.
- **The filesystem as a namespace** — everything is a file (or acts like one).
- **Processes, users, permissions** — the security and execution model.
- **Streams and pipes** — data flows from `stdout` to `stdin`, with `stderr` for out-of-band diagnostics.
- **Shell scripting** — automating repeatable work with strong error handling.
- **Docker/containers** — reproducible, portable execution environments.

**Fundamental principles you must own:**

1. **Everything is a file.** Regular files, directories, devices (`/dev/null`, `/dev/tty`), sockets, pipes, and even process info (`/proc`) are exposed through the filesystem interface.
2. **Programs read `stdin`, write `stdout` and `stderr`, and return an exit code.** That's the contract. Pipes chain `stdout` of one to `stdin` of the next.
3. **Text is the universal interface.** Line-oriented text lets tools like `grep`, `awk`, `sed`, `jq` compose beautifully.
4. **Every action has a user, a group, and a permission bitmask.** `rwx` for owner/group/others.
5. **The shell is a programming language.** It has variables, control flow, functions, and error semantics — treat it that way.

If you retain nothing else: **`man` and `--help` are your first friends. Read them before Googling.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Filesystem, Paths, and Navigation

**The filesystem is a tree rooted at `/`.** Key directories to know:

| Path       | Purpose                                     |
|------------|---------------------------------------------|
| `/`        | Root of the tree                            |
| `/home`    | User home directories (`~` is your home)    |
| `/etc`     | System-wide configuration                   |
| `/var`     | Variable state (logs, caches)               |
| `/tmp`     | Temporary files (wiped on reboot)           |
| `/usr`     | User-space programs and libraries           |
| `/opt`     | Optional/third-party software               |
| `/bin`, `/sbin`, `/usr/bin`, `/usr/local/bin` | Executables |
| `/dev`     | Device files                                |
| `/proc`    | Kernel + per-process info (virtual FS)      |
| `/sys`     | Kernel/device tree (virtual FS)             |

**Navigation:**

- `pwd` — where am I?
- `cd <dir>` — go there. `cd -` returns to the previous. `cd` alone goes home.
- `ls -lah` — long listing, hidden files, human sizes.
- `tree -L 2` — recursive directory tree, 2 levels deep.
- `find <dir> -name '*.py' -mtime -1` — powerful and worth mastering.
- `stat <file>` — inode metadata.

**Absolute vs relative paths:** absolute start with `/`; relative don't. `..` is parent; `.` is current; `~` expands to `$HOME`.

**Globs vs regex:** `*.py` is a **glob** (shell expansion), not a regex. Globs use `*`, `?`, `[abc]`, `[!abc]`. Regex is for tools like `grep`.

---

### 2.2 Streams, Redirection, Pipes

Every process has three standard streams:

- **stdin** (0) — input.
- **stdout** (1) — normal output.
- **stderr** (2) — diagnostic output.

**Redirection:**

- `cmd > out.txt` — redirect stdout to a file (truncates).
- `cmd >> out.txt` — append.
- `cmd 2> err.txt` — redirect stderr.
- `cmd > out.txt 2>&1` — stderr to same place as stdout.
- `cmd &> all.txt` — both (bash shortcut).
- `cmd < in.txt` — read stdin from file.
- `cmd <<< "text"` — here-string; feed "text" as stdin.
- `cmd <<EOF ... EOF` — here-doc.

**Pipes** connect stdout → stdin:

```bash
cat access.log | grep ' 500 ' | awk '{print $7}' | sort | uniq -c | sort -rn | head
```

That one-liner reads a log, keeps 500-status lines, extracts the URL column, and prints the top 10 URLs by frequency. Pure Unix composition.

**Related tools:**

- **`tee`** — duplicate stdin to a file *and* forward it (`... | tee out.txt | ...`).
- **`xargs`** — build commands from stdin words: `find . -name '*.log' | xargs gzip`.
- **Process substitution** — `diff <(sort a.txt) <(sort b.txt)` treats command output as a temporary file.

**Exit codes:**

- `0` = success; anything else = failure.
- `$?` = the last command's exit code.
- Chain with `&&` (only run if previous succeeded) and `||` (only run if previous failed):
  ```bash
  make test && git push || echo "tests failed; not pushing"
  ```

---

### 2.3 The Text-Processing Toolkit: grep, sed, awk, jq

**`grep` — pattern matching:**

```bash
grep -R --include='*.py' 'TODO' src/
grep -Eo '[0-9]{3}-[0-9]{4}' file.txt         # extract matches only
grep -Pv '^\s*#' config.ini                   # exclude comment lines
```

Common flags: `-i` (case-insensitive), `-v` (invert), `-n` (line numbers), `-A/-B/-C` (context), `-E` (extended regex), `-P` (PCRE).

**`sed` — stream editor:**

```bash
sed 's/foo/bar/g' file.txt                    # substitute all
sed -i 's/foo/bar/g' file.txt                 # in place
sed -n '10,20p' file.txt                      # print lines 10–20
```

**`awk` — line + field processor. A tiny language.**

```bash
awk -F, '$3 > 100 {print $1, $3}' data.csv    # rows where col3 > 100
awk 'NR==1 {next} {sum+=$2} END {print sum/NR}' data.csv   # avg of col2, skip header
```

Key ideas: `-F` sets the field separator; `$1..$NF` are fields; `NR` is the current record number; `BEGIN {}` and `END {}` blocks run before/after all lines.

**`jq` — JSON on the CLI.**

```bash
curl -s https://api.github.com/users/torvalds | jq '.public_repos'
cat data.json | jq '.users[] | select(.premium) | {id, name}'
```

Filters compose: `.` (identity), `.foo` (field), `.[]` (iterate), `select(cond)`, `map(...)`, `group_by`, `sort_by`.

**Others worth knowing:** `cut`, `sort`, `uniq`, `tr`, `paste`, `head`, `tail`, `wc`, `column`, `xargs`, `find`.

---

### 2.4 Processes, Users, Permissions

**Processes:**

- `ps aux` — snapshot of all processes.
- `top` / `htop` — live view (htop is much friendlier).
- `pgrep -f python` — find PIDs by command.
- `kill <pid>` — send `SIGTERM` (default 15). `kill -9 <pid>` sends `SIGKILL` (immediate).
- `nohup cmd &` — run detached from the terminal.
- `disown` — remove a job from the shell's control.
- `wait`, `jobs`, `fg`, `bg` — shell job control.

**Signals worth knowing:** `SIGTERM` (polite stop), `SIGKILL` (force), `SIGINT` (Ctrl-C), `SIGHUP` (terminal closed; often used to reload configs), `SIGSTOP`/`SIGCONT`.

**Users and groups:**

- `id` — your uid/gid/groups.
- `whoami`, `groups`.
- `sudo` — run as another user (usually root). Configured in `/etc/sudoers`.
- `useradd`, `usermod`, `passwd` — user management.

**Permissions (`rwx` triplets):**

$$\text{mode} = \underbrace{\text{owner}}_{rwx}\underbrace{\text{group}}_{rwx}\underbrace{\text{others}}_{rwx}$$

- **Read (r)** = list dir / read file.
- **Write (w)** = modify dir contents / write file.
- **Execute (x)** = enter dir / run file.

**Octal:** `rwx` = 7, `rw-` = 6, `r--` = 4, `r-x` = 5, etc. `chmod 755 file` = owner rwx, group r-x, other r-x.

**Special bits:**

- **setuid** (`chmod u+s`) — runs as file owner.
- **setgid** (`chmod g+s`) — new files in dir inherit group.
- **sticky** (`chmod +t`, seen on `/tmp`) — only owner can delete their files.

**Ownership:** `chown user:group file`.

---

### 2.5 Shell Scripting & Docker Basics

**Safe Bash script header:**

```bash
#!/usr/bin/env bash
set -euo pipefail   # -e: exit on error; -u: undefined var is error; -o pipefail: fail on any pipeline stage
IFS=$'\n\t'         # sane default IFS

TARGET="${1:?usage: deploy.sh <target>}"   # required arg
LOG_DIR="${LOG_DIR:-/var/log/deploy}"       # default value

trap 'echo "❌ failed at line $LINENO" >&2' ERR

deploy() {
    local target="$1"
    echo "🚀 Deploying to ${target}..."
    # ... work ...
}

deploy "$TARGET"
```

**Common patterns:**

- **Quote your variables**: `"$VAR"` not `$VAR` — spaces in paths destroy scripts.
- **Prefer `[[ ]]`** over `[ ]` for tests in Bash (`[[` handles empty vars, regex).
- **Use `$(...)`** not backticks for command substitution.
- **Use functions** for anything that repeats.
- **Log to stderr for diagnostics**: `echo "..." >&2`.

**Docker basics:**

```bash
# Build an image
docker build -t myapp:1.0 .

# Run a container
docker run --rm -it -p 8000:8000 -v $PWD/data:/data myapp:1.0

# List running containers
docker ps
docker ps -a           # including stopped

# Enter a running container
docker exec -it <name> bash

# Logs
docker logs -f <name>
```

**Minimal `Dockerfile` for a Python app:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen
COPY src/ ./src/
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key concepts:

- **Layers** — each instruction creates a layer, cached by Docker. Order instructions from least-often-changing to most-often-changing.
- **`.dockerignore`** — like `.gitignore`, keeps unnecessary files out of the build context.
- **Multi-stage builds** — build in one image, copy artifacts to a lean runtime image.
- **Compose** — `docker compose up` orchestrates multiple containers via `docker-compose.yml`.

---

## 3. Mental Models & Analogies

### 3.1 The "Assembly Line" Model (Pipes and Unix Composition)

Imagine a factory floor. Each station does **one thing** and hands its output to the next. `grep` is the station that lets in only the widgets matching a pattern. `sort` is the sorter. `uniq -c` is the counter. `head` is the "top-10" clerk.

The philosophy is:

- **Each station is stupid on purpose** — it doesn't know or care about the whole pipeline.
- **Text is the universal shape** — like standardized parts. Any station can accept any predecessor's output.
- **Diagnostic messages don't go on the conveyor** (`stderr` is the intercom, separate from `stdout`).

This is why a one-liner Unix pipeline routinely replaces a 200-line Python script. Not because it's faster to type (though it often is), but because *composition* means you don't have to invent the plumbing every time.

### 3.2 The "Nested Sandboxes" Model (Containers)

Think of a computer as a set of nested sandboxes:

- The **kernel** is the ground floor — the hardware interface.
- Regular **processes** are sandboxes that share the kernel: they see the same filesystem, users, and network by default.
- **Containers** are processes with **namespaced views** — their own filesystem (thanks to a mount namespace + overlay), their own PID space, their own network stack — while still sharing the same kernel.
- **VMs** are the next level: they carry their own kernel too.

Container images are the **snapshot of the sandbox** (files, environment, entrypoint). Running a container = booting a process inside that snapshot with its narrowed view of the world. That's why "works on my machine" stops being an argument: **the sandbox travels with the app.**

> 🖼️ **Image Prompt [IMG-CLI-01]:** *"A conceptual illustration of Linux processes as boxes on top of a kernel base. Regular processes share the same filesystem and network view. A container is drawn as a translucent bubble around one process, showing its own mini filesystem tree, own PID list, own network — but the arrows underneath still connect to the same kernel base. Clean, isometric, modern flat design, label 'shared kernel, isolated user-space'."*
> **Caption:** Containers namespace what a process sees; they don't duplicate the kernel.
> **Placement:** Section 2.5 / Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "`rm -rf /` Is a Meme, Not a Real Risk"

Beginners run `rm -rf $VAR/subdir` without checking that `$VAR` is set. If `$VAR` is empty, it becomes `rm -rf /subdir` — or worse. Rules:

- Always **quote** variables: `rm -rf "$VAR/subdir"`.
- Set `set -u` in scripts to fail on unset variables.
- Prefer `trash` or `mv` over `rm` for interactive work.
- Never `sudo rm -rf` anything you didn't just create.

### 4.2 "`sudo` Fixes Everything"

`sudo` grants root privileges. It's the right answer when a system-wide config or a package install truly needs it, and the wrong answer when your script only needs write access to your own project. Overusing `sudo` teaches you nothing about the real permission problem, and it can chown your files to root — which then requires more `sudo` to undo. Rule: pause and ask "what specifically doesn't have permission?" before `sudo`-ing.

### 4.3 "Docker Containers Are VMs"

They aren't. Containers share the host kernel. Practical consequences:

- **You cannot run Windows containers on a Linux host** (or vice versa) without a VM under the hood. On macOS/Windows, "Docker" is actually a Linux VM running Docker.
- **Kernel-level features** (kernel modules, non-namespaced syscalls) can leak.
- **Containers are process-level isolation, not security boundaries.** Don't run untrusted code as root inside a container and think it's safe.

---

## 5. Self-Assessment Bank (Linux & CLI)

### Questions

**Q1 (Short answer).** What are the three standard streams, and how do you redirect stderr to the same place as stdout?

**Q2 (Multiple choice).** What does `set -euo pipefail` do in a Bash script?
A. Enables error, undefined-var, and pipe-fail handling — the script exits on any of them.
B. Turns on verbose logging.
C. Sets the exit code to 0 regardless of errors.
D. Disables all safety features.

**Q3 (Short answer).** Explain the difference between a hard link and a symbolic link.

**Q4 (Multiple choice).** Which of the following is idiomatic Unix?
A. `grep foo file.txt | awk '{print $2}' | sort | uniq -c`
B. Writing one Python script that does grep, awk, sort, uniq internally.
C. Copying a file into memory and processing with `sed -e ...`.
D. Using `find` with a Python one-liner.

**Q5 (Short answer).** What is the difference between `SIGTERM` and `SIGKILL`?

**Q6 (Multiple choice).** In octal permissions, what does `chmod 640 file.txt` grant?
A. Owner rw, group r, other none.
B. Owner rwx, group rwx, other r.
C. Owner r, group rw, other rwx.
D. Owner rw, group rwx, other rw.

**Q7 (Short answer).** Why is quoting shell variables important? Give a concrete example of a bug caused by not quoting.

**Q8 (Multiple choice).** `docker build` caches layers. Which of the following is the BEST `Dockerfile` layer order for a Python app?
A. Copy source, then install dependencies, then set WORKDIR.
B. Set WORKDIR, copy pyproject.toml, install dependencies, copy source, CMD.
C. Copy everything, then install and run in one big RUN.
D. Order doesn't matter.

**Q9 (Short answer).** Explain what `find . -type f -mtime -1 -name '*.log'` finds.

**Q10 (Multiple choice).** You need the top-5 most-frequent IPs from `access.log` (IP is column 1). Which one-liner works?
A. `awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -5`
B. `grep -c '^' access.log | head -5`
C. `wc -l access.log | head -5`
D. `sort access.log | head -5`

---

### Answer Key & Detailed Explanations

**A1.** `stdin` (fd 0), `stdout` (fd 1), `stderr` (fd 2). Redirect stderr to stdout with `2>&1`, and typically you send both to a file with `cmd > out.txt 2>&1` (order matters — the `2>&1` must come *after* the stdout redirection). Bash shorthand: `cmd &> out.txt`.

**A2.** **A.** `-e` exits on any command failure; `-u` fails on undefined variables; `-o pipefail` makes a pipeline fail if *any* stage fails (default is only the last stage's exit code). Together they turn Bash from "silently continue past errors" into "fail loud and fast" — table stakes for any script beyond 10 lines.

**A3.** A **hard link** is a second directory entry pointing to the same inode as the original file. The file exists as long as any hard link exists; deleting one leaves the others valid. Hard links can't cross filesystems and generally can't point to directories. A **symbolic (soft) link** is a small special file whose contents are a path to another file. It can cross filesystems and point to directories. If the target is deleted, the symlink becomes a **dangling** pointer.

**A4. A.** Composed one-liner with each tool doing one thing. Options B/C/D are perfectly *possible* but violate the Unix philosophy.

**A5.** `SIGTERM` (15) is a polite request — the program can catch it, clean up, and exit. `SIGKILL` (9) is uncatchable and unblockable; the kernel immediately terminates the process. Use `SIGTERM` first; if the process doesn't respond, escalate to `SIGKILL`.

**A6. A.** `640` = `6-4-0` = `rw- r-- ---` = owner rw, group r, others none. This is a common mode for readable-but-not-executable files with restricted access.

**A7.** Unquoted variables undergo **word splitting** and **glob expansion**. Example:
```bash
FILE="my document.txt"
rm $FILE            # runs: rm my document.txt  → tries to delete two files: "my" and "document.txt"
rm "$FILE"          # runs: rm "my document.txt"  → correct
```

**A8. B.** Docker caches each layer; a layer's cache is invalidated when its input changes. Copying `pyproject.toml` (rarely changes) before `pip install` (expensive) then copying source (changes every build) means dependency installs are cached across most rebuilds.

**A9.** Files (`-type f`), modified in the last 24 hours (`-mtime -1`), whose name matches `*.log`, starting at the current directory recursively.

**A10. A.** Extract IPs (`awk '{print $1}'`), sort (needed by `uniq`), count with `uniq -c`, sort by count numerically reversed (`sort -rn`), take top 5. This is one of the most useful Unix pipelines to memorize.

---

## 6. Practice Prompts

1. Write a Bash script that takes a directory and prints the top-10 largest files under it, human-readable. Use `find`, `du`, `sort`.
2. Write a `pipe` that parses `nginx` access logs and prints requests/second per status code family (`2xx`, `4xx`, `5xx`) over the last hour.
3. Package your Python data-analysis project (from Build #1) into a Docker image with a multi-stage build. Aim for < 200 MB.
4. Create a systemd service (or a launchd on macOS) that runs your FastAPI app on boot, with restart-on-failure.
5. Write a `.bashrc` alias section for the 10 commands you actually run each day. Read `bash-it` or `oh-my-zsh` for inspiration but write your own.

---

## 7. References

- Michael Kerrisk, *The Linux Programming Interface* (canonical)
- *The Art of Unix Programming*, Eric S. Raymond (free online)
- Brian Kernighan & Rob Pike, *The Unix Programming Environment* (classic)
- Julia Evans, zines on `strace`, `ss`, `perf`, and networking
- `man bash`, `man 7 signal`, `man 5 systemd.service`
