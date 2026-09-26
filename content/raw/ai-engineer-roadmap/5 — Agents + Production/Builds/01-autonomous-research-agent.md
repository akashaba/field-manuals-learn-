# Build #1 — Autonomous Research / Automation Agent

> **Track:** Agents + Production · **Build:** 01
> **Prerequisites:** All 13 Agents+Production learning modules + prior builds.
> **Time budget:** ~60–80 hours over ~3–4 weeks.
> **Deliverable:** A public GitHub repo containing an **autonomous agent** that takes a user request, plans, uses tools, reasons across steps, and delivers a final grounded answer — with all the production-grade concerns implemented.
> **Example flow (user's own):** `User → Agent → Search → Tools → Data → Reason → Final answer.`

---

## 1. Executive Summary

The user's own words for this build:

> "🔥 An autonomous research/automation agent. Example: User → Agent → Search → Tools → Data → Reason → Final answer. The important skill is not 'using LangChain.' It's understanding the architecture underneath it."

Anyone can call LangChain and hand off. This build is the **agent a senior engineer would ship**:

- ReAct-style loop with tool calling, iteration budgets, and error recovery.
- Multi-strategy planning: plan-and-execute for known tasks, ReAct for exploration.
- Memory: short-term messages + long-term facts + session scratchpad.
- Guardrails: prompt-injection defense, tool-tier authorization, human-in-the-loop for destructive actions.
- MCP integration: expose select tools via MCP; consume 1+ external MCP server.
- Graph orchestration via LangGraph (or bare-Python for demonstration).
- Full evaluation suite in CI with regression gate.
- Production deployment: async FastAPI, streaming SSE, durable state, cost caps, observability, docker + CI/CD.

At the end, you hand a hiring manager the repo and say: *"This is what a modern autonomous agent looks like when someone who actually understands the architecture builds it."*

---

## 2. Learning Outcomes

By the end you will have practiced:

1. Building an agent from scratch (minimal loop) and then layering a framework.
2. Designing tool schemas the LLM understands and uses correctly.
3. Implementing planning (Plan-and-Execute + Reflection).
4. Engineering short-term + long-term memory with retrieval.
5. Multi-step orchestration via LangGraph (or equivalent).
6. Context engineering — dynamic assembly per turn.
7. Guardrails — injection defense, tool-tier auth, HITL.
8. Building and consuming MCP servers.
9. Agent evaluation — task-level, trace-level, step-level; RAGAS-style eval in CI.
10. Production deployment — async FastAPI, streaming, durability, cost caps, observability.

---

## 3. Choose a Task

Pick something with **real utility** to you, so you'll use the agent afterwards.

### 3.1 Option A — Legislative Research Agent (aligned with the day job)
> Given a bill number or topic, produce a briefing: history, sponsors, current status, key provisions, notable amendments, fiscal impact, media coverage, related bills. Tools: legmt.gov API/scraper, web search, WestLaw-like citation lookup.

### 3.2 Option B — Job Market Intelligence Agent (aligned with remote-job-search)
> Given a role and target companies/geographies, produce a briefing: job openings, salary ranges, recent hires, company financial health, tech stack (from job posts and GitHub), interview process reports. Tools: LinkedIn API (or scraper w/ ToS respect), Glassdoor, GitHub API, company websites, Levels.fyi.

### 3.3 Option C — Football / Soccer Fixture Analysis Agent (aligned with soccer analytics interest)
> Given a fixture, produce a preview: team form, injuries, tactical matchups, betting-market shifts, historical head-to-head, weather, referee tendencies. Tools: football-data.org, understat.com, weather API, betting-odds APIs, news search.

### 3.4 Option D — Personal Research Concierge
> Given any research question, retrieve, cross-reference, and synthesize a cited briefing. Tools: web search, Wikipedia, arXiv, YouTube transcripts, PDF fetch/parse.

### 3.5 Option E — Codebase Understanding Agent
> Given a repo and a question ("how does auth work?" "where is X handled?"), the agent explores the codebase and answers with citations. Tools: file listing, code search, symbol grep, file read.

**Requirements:**
- Real-world tools with real APIs (not just LLM calls).
- Multi-step task: 3–10 tool calls per typical query.
- Value beyond a single API call — synthesis matters.

---

## 4. Repository Structure

```
autonomous-agent/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── eval.yml
│       └── docker.yml
├── configs/
│   ├── prod.yaml
│   ├── dev.yaml
│   └── eval.yaml
├── data/
│   └── eval/
│       └── tasks.jsonl              # 50+ curated eval tasks
├── docs/
│   ├── architecture.md              # diagram + rationale
│   ├── model_card.md
│   ├── decisions.md                 # ADRs
│   └── operations.md                # runbook, cost, monitoring
├── src/
│   └── research_agent/
│       ├── __init__.py
│       ├── config.py                # pydantic-settings
│       ├── llm/
│       │   ├── client.py            # provider-agnostic wrapper
│       │   ├── streaming.py
│       │   └── router.py            # model tier routing
│       ├── tools/
│       │   ├── base.py              # Tool + schema base
│       │   ├── search.py
│       │   ├── fetch.py
│       │   ├── extract.py
│       │   ├── db.py
│       │   ├── notes.py
│       │   └── send_email.py        # example destructive tool with HITL
│       ├── memory/
│       │   ├── short_term.py        # conversation state
│       │   ├── long_term.py         # extracted facts
│       │   └── scratchpad.py        # session-scoped notes
│       ├── planning/
│       │   ├── planner.py           # plan-and-execute
│       │   └── reflection.py
│       ├── graph/
│       │   ├── state.py             # LangGraph typed state
│       │   ├── nodes.py             # node functions
│       │   └── edges.py             # conditional edges
│       ├── context/
│       │   ├── builder.py           # dynamic context assembly
│       │   └── formatters.py
│       ├── guardrails/
│       │   ├── input.py             # PII, prompt injection
│       │   ├── output.py            # PII, URL exfiltration
│       │   ├── tool_auth.py         # per-tier authorization
│       │   └── hitl.py              # human-in-the-loop
│       ├── mcp/
│       │   ├── server.py            # expose 2-3 tools as MCP
│       │   └── client.py            # consume an external MCP server
│       ├── observability/
│       │   ├── tracing.py           # OpenTelemetry / LangSmith
│       │   ├── metrics.py           # Prometheus
│       │   └── logging.py           # structured JSON
│       ├── eval/
│       │   ├── runner.py            # runs the eval suite
│       │   ├── graders.py           # programmatic + LLM-as-judge
│       │   └── report.py
│       └── api/
│           ├── main.py              # FastAPI app
│           ├── schemas.py           # Pydantic
│           ├── auth.py              # api-key
│           └── routes/
│               ├── chat.py          # POST /v1/agent/run + SSE stream
│               ├── jobs.py          # long-running background jobs
│               └── admin.py         # health, metrics, evals
├── tests/
│   ├── test_tools.py
│   ├── test_memory.py
│   ├── test_planning.py
│   ├── test_guardrails.py
│   ├── test_graph.py
│   ├── test_api.py
│   └── test_eval_regression.py      # CI regression on eval
├── artifacts/
│   ├── model_versions.json          # pinned model IDs
│   └── graph.png                    # visualized state graph
├── Dockerfile
├── docker-compose.yml               # includes Postgres, Redis, LangSmith proxy
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 5. Milestones & Acceptance Criteria

### M1 — Kickoff & Scaffolding (3–4h)
- [ ] Repo on GitHub; MIT license; README skeleton.
- [ ] `pyproject.toml` pinned deps (anthropic, openai, fastapi, langgraph, mcp, ragas, structlog, ...).
- [ ] `docker-compose` for local dev (Postgres, Redis).
- [ ] Pre-commit + CI + Makefile targets: `setup`, `run`, `test`, `eval`, `docker`, `mcp-serve`.

### M2 — Minimal Agent (3–4h)
- [ ] Working ReAct agent in ~150 lines, no framework (Module 01).
- [ ] 3 tools: `search_web`, `fetch_url`, `extract_text`.
- [ ] Termination conditions: max_steps, max_tokens, max_wall_time, loop-detector.
- [ ] Full trace logged to JSONL.
- [ ] End-to-end: runs a research query; returns a cited answer.

### M3 — LangGraph Version (4–5h)
- [ ] Rewrite the agent as a LangGraph state graph.
- [ ] Typed state (TypedDict): messages, plan, results, errors, iteration.
- [ ] Nodes: planner, executor, reflector, finalizer.
- [ ] Conditional edges: continue-execution vs finalize; retry on error.
- [ ] Same functionality as M2; more structured; graph visualization in `artifacts/graph.png`.

### M4 — Tool Suite (4–5h)
- [ ] 6–10 tools relevant to your task choice.
- [ ] Every tool: type-safe schema (Pydantic), docstring, structured error return, unit tests.
- [ ] Tool tiers: read-only, write-safe, destructive.
- [ ] Tool allowlist per graph node (planning phase = read-only; execution phase = writes allowed).

### M5 — Planning (4–5h)
- [ ] Plan-and-Execute path for complex queries.
- [ ] Planner emits structured JSON plan (steps + dependencies).
- [ ] Executor runs steps; supports parallel execution of independent steps.
- [ ] Reflection loop for final output: agent critiques its answer; revises if issues found.

### M6 — Memory (4–5h)
- [ ] Short-term memory: sliding window + periodic summarization.
- [ ] Long-term memory: extracted user facts stored in pgvector.
- [ ] Session scratchpad: intermediate tool results not needed in final answer, but retrievable via `read_scratchpad(query)` tool.
- [ ] Memory-per-turn: retrieve relevant facts based on current query.

### M7 — Context Engineering (3–4h)
- [ ] `context/builder.py` dynamically assembles per-turn context.
- [ ] Static part (system prompt, tool schemas, user profile) marked cacheable.
- [ ] Dynamic part (retrieved memory, tool results, current query) refreshed each turn.
- [ ] Token budget enforced per turn; log breakdown by component.

### M8 — Guardrails (5–6h)
- [ ] **Input filters**: PII detection/redaction; prompt-injection heuristics; length caps.
- [ ] **Output filters**: PII scrubbing; URL-exfiltration flag; content moderation.
- [ ] **Structural defenses**: retrieved content wrapped in `<untrusted_data>` tags; system prompt treats those as data.
- [ ] **Tool-tier authorization**: destructive tools (email, delete) require `confirm=True` + HITL.
- [ ] **Rate limits + cost caps**: per-user, per-run, global.
- [ ] Injection test suite: 20 attack prompts; verify defenses hold.

### M9 — MCP Integration (4–5h)
- [ ] Expose 2–3 of your tools as an MCP server (`mcp/server.py`). Test with Claude Desktop.
- [ ] Consume an external MCP server (e.g., filesystem, GitHub) via `mcp/client.py`; add its tools to your agent's toolset.
- [ ] Document server setup + config in README.

### M10 — Evaluation (5–6h)
- [ ] 50+ eval tasks in `data/eval/tasks.jsonl` with expected outcomes and grading criteria.
- [ ] **Task-level** grading: programmatic checks + LLM-as-judge for open-ended.
- [ ] **Trace-level**: step count, tool-selection accuracy (LLM-as-judge on golden traces).
- [ ] **Meta-metrics**: cost, latency, refusal rate.
- [ ] Baseline metrics saved; CI regression test blocks merges on regression > 2%.

### M11 — Observability (3–4h)
- [ ] Structured JSON logs with `request_id`, `user_id`, per-step details.
- [ ] Traces to LangSmith (or self-hosted Phoenix).
- [ ] Prometheus metrics endpoint: success rate, cost, latency, step count, tool errors.
- [ ] Small Grafana dashboard (JSON file committed).
- [ ] Alerts wired for cost anomalies, success rate drops.

### M12 — API + Streaming (4–5h)
- [ ] FastAPI service with:
  - `POST /v1/agent/run` — synchronous for short runs.
  - `POST /v1/agent/jobs` — background job for long runs; returns `job_id`.
  - `GET /v1/agent/jobs/{id}/stream` — SSE stream of trace events.
  - `GET /v1/agent/jobs/{id}` — final result.
  - `POST /v1/agent/jobs/{id}/cancel` — cancel a run.
  - `/healthz`, `/readyz`, `/metrics`.
- [ ] API-key auth + per-user rate limiting.
- [ ] Async throughout; verified with load test.

### M13 — Durability (3–4h)
- [ ] LangGraph checkpointing enabled — agent state persists to Postgres.
- [ ] Server restart doesn't lose in-flight runs; resumes from last checkpoint.
- [ ] Client can disconnect and reconnect to a running job; sees prior trace + live progress.

### M14 — HITL + Approval Flow (3–4h)
- [ ] For destructive tool calls, agent pauses at graph node awaiting approval.
- [ ] Simple approval UI (or CLI): show pending action + trace context; approve / edit / reject.
- [ ] Timeout on pending approvals (e.g., 24h).

### M15 — Docker + CI/CD (2–3h)
- [ ] Multi-stage Dockerfile (< 2 GB); non-root user.
- [ ] docker-compose runs everything locally (agent + Postgres + Redis).
- [ ] CI: lint (ruff), type-check (mypy), tests, eval regression, docker build.
- [ ] CD: push to GHCR on `main`; optional deploy to Fly.io / Render.

### M16 — Docs & Demo (3–4h)
- [ ] README quickstart (< 15 min from clone to running a query).
- [ ] Architecture diagram (`docs/architecture.md`).
- [ ] Model card with capabilities, limits, evaluation numbers.
- [ ] Decisions log (ADRs) for the top 5 design choices.
- [ ] Operations runbook: known issues, incident response, cost monitoring.
- [ ] Optional live demo URL.

---

## 6. Non-negotiables

1. **From-scratch minimal agent (M2) exists** — proves you understand the architecture without a framework.
2. **Iteration/cost/latency caps enforced**.
3. **Prompt-injection defenses tested with an attack suite**.
4. **Destructive tools gated behind HITL approval**.
5. **All tool errors returned structurally to the LLM** — no unhandled exceptions crashing the loop.
6. **CI regression test runs the eval suite** and blocks merges on regressions.
7. **Structured tracing** on every run — LangSmith / Phoenix or equivalent.
8. **Async I/O throughout** the FastAPI service.
9. **Cost + latency + step-count logged** per run.
10. **No secrets in the repo** — env vars only.
11. **Multi-tenant safe** — user data isolated at the DB layer.

---

## 7. Nice-to-Haves (stretch)

- **Multi-agent variant** — supervisor + specialized workers for different sub-tasks.
- **DSPy-compiled prompts** for the planner / grader for auto-optimization.
- **Fine-tuned small model** (LoRA) for the planner step — cheaper than frontier model.
- **Rich UI** — Next.js chat + trace visualizer + HITL approval panel.
- **Slack / Discord bot** wrapper.
- **A/B experiment framework** — split traffic between two agent versions; per-cohort metrics.
- **Model-provider fallback** (Anthropic ↔ OpenAI) on 5xx.
- **Cost dashboard** with weekly spend by user + endpoint.
- **Semantic response cache** for FAQ-style questions.

---

## 8. Evaluation Rubric

| Area | 1 (poor) | 3 (ok) | 5 (excellent) |
|------|----------|--------|---------------|
| Architecture understanding | framework-glued | LangGraph used, understood | from-scratch minimal + framework + reasoned design |
| Tool design | dict-based, ad-hoc | Pydantic schemas | typed + tiered + structured errors + unit tests |
| Planning | pure ReAct | Plan-and-Execute added | + reflection + adaptive replanning |
| Memory | none | short-term messages | short + long + scratchpad + retrieval |
| Context engineering | dumps everything | some structure | dynamic assembly + XML tags + token budgeting |
| Guardrails | none | rate limits | injection defense + tool auth + HITL + attack suite |
| MCP | not used | consumes one | consumes + publishes own server |
| Orchestration | ad-hoc | LangGraph | LangGraph + checkpointing + streaming |
| Evaluation | anecdotal | task-level | task + trace + step + CI regression |
| Observability | logs only | + metrics | + tracing + alerts + dashboards |
| API + deployment | script | FastAPI endpoint | async + SSE + background jobs + durability |
| Docs | short README | + model card | + architecture + ADRs + runbook |

Aim for 4+ across the board before publishing.

---

## 9. Related Files

- `../goals.md`
- `../learning/01-agent-fundamentals.md` through `../learning/13-production-deployment.md`
- `../../llm-engineering/builds/01-chat-with-documents.md` — RAG foundation reused
- `../../foundations/builds/02-ml-api-fastapi.md` — API patterns

---

## 10. Final Advice

The user's directive:

> "The important skill is not 'using LangChain.' It's understanding the architecture underneath it."

Every design decision on this build should be **defensible in first principles**. When someone reads your code:

- They should see WHY you used a state graph (or didn't).
- They should see WHY certain tools are tiered.
- They should see WHY memory is split into three stores.
- They should see WHY the eval suite has these specific tasks.
- They should see WHY the guardrails layer this way.

Frameworks help you build faster. Architecture skill lets you build *well*. Ship the architecture — the framework is just the pen.

That's the whole track.
