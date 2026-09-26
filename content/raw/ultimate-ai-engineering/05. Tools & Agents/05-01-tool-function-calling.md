# 05.01 — Tool / Function Calling

> **Module 5: Tools & Agents** · Subtopic 1 of 6
> **Prerequisites:** 02.03 (schemas, strict outputs, validation and repair), 02.04 (constrained decoding), 02.01 §6 (prompt injection), 02.02 §4 (tool context), async programming.
> **Outcome:** you can design tools that models use correctly, build a production tool runtime (validation, timeouts, idempotency, parallelism, sandboxing, observability), use provider tool features (strict schemas, tool choice, parallel calls, tool search, programmatic calling) deliberately, and evaluate tool use as its own capability.

---

## 1. The Mechanics

A model cannot execute anything. **Tool calling** is a protocol: the model emits a *structured request*, your code executes it, and the result goes back into the context.

```
 ┌────────────┐  1. messages + tool definitions (name, description, JSON Schema)
 │ your app   │──────────────────────────────────────────────────────────────►┌───────┐
 │ (runtime)  │  2. assistant turn: text + tool_use{id, name, input}           │ model │
 │            │◄──────────────────────────────────────────────────────────────└───────┘
 │ 3. validate input → authorise → execute (timeout, retries) → shape result
 │ 4. user turn: tool_result{tool_use_id, content, is_error}
 │            │──────────────────────────────────────────────────────────────►  model continues
 └────────────┘  … loop until stop_reason == end_turn (or budget exhausted)
```

Under the hood, tool definitions are rendered into the prompt, and the model was trained to emit a special structured block. With **strict** tool use, the arguments are grammar-constrained to your schema (02.04), so they are guaranteed to parse and match their types. They are still not guaranteed to be *correct*.

### 1.1 Provider features (checked September 2026 — re-verify before building)

| Feature | Anthropic Claude API | Purpose |
|---|---|---|
| Tool definition | `name`, `description`, `input_schema`, optional `strict: true`, optional `input_examples` | Schema + documentation |
| `tool_choice` | `auto` (default), `any`, `tool` (a specific tool), `none`; `disable_parallel_tool_use` | Force or forbid calls; single-call turns |
| Response | `stop_reason: "tool_use"` + `tool_use` blocks `{id, name, input}` | Execute each, reply with `tool_result` |
| Result | `tool_result{tool_use_id, content, is_error}` | Errors are information the model can act on |
| Parallel tool use | Several `tool_use` blocks in one turn | Execute them concurrently; return all results in one user turn |
| **Tool search** | Server tool (`tool_search_tool_regex_…` / `tool_search_tool_bm25_…`) + `defer_loading: true` on tools | Keep thousands of tools out of context; discovered tools are loaded on demand |
| **Programmatic tool calling** | Code execution tool + `allowed_callers` on your tools | The model writes code that calls your tools in a sandbox; intermediate results don't enter the context |
| Server tools | Web search, web fetch, code execution, MCP connector, … | Executed on the provider's infrastructure |

OpenAI's Responses API offers the equivalent concepts (function tools with `strict`, `function_call` / `function_call_output` items, parallel calls, and hosted tools). Keep your runtime **provider-agnostic** (§4) and write thin adapters per provider.

---

## 2. Designing Tools the Model Uses Correctly

A tool definition is a prompt (02.02 §4). Most tool-use failures are **interface design failures**.

### 2.1 Principles

| Principle | Bad | Good |
|---|---|---|
| **Task-level, not API-level** | `get_bill`, `get_versions`, `diff_text` (3 calls to answer one question) | `compare_bill_versions(bill_id, from_version, to_version)` |
| **Name by intent, namespace by domain** | `query`, `run` | `legis_search_bills`, `legis_get_section` |
| **Descriptions say when, when not, and what returns** | "Searches bills." | "Search bill text and metadata. Use for questions about proposed or enacted bills. Not for current statute text (use legis_get_section). Returns ≤ 20 hits with bill_id, title, status, and a snippet." |
| **Constrained arguments** | `status: string` | `status: enum[introduced, in_committee, passed, enacted, vetoed]` |
| **Semantic identifiers** | `id: 83721` | `bill_id: "HB-45"` with a documented format |
| **Model-friendly results** | Raw 40-field JSON, 30k tokens | Only the needed fields, paginated, with a handle for more |
| **Actionable errors** | `500 Internal Server Error` | `"bill_id 'HB45' not found. Format is 'HB-45'. Did you mean HB-45 (2025)?"` |
| **Idempotency where possible** | `create_draft()` duplicates on retry | `create_draft(request_id)` returns the existing draft on repeat |
| **Safety annotations** | none | `read_only`, `destructive`, `idempotent`, `requires_approval` (drives 05.06 gating) |

### 2.2 How many tools?

Every tool definition costs context tokens on *every* call, and selection accuracy drops as the number of similar tools grows. Options, in order of preference:
1. **Fewer, better tools**: consolidate thin wrappers.
2. **Namespacing** plus crisp "use / don't use" descriptions.
3. **Tool retrieval**: your own (embed the descriptions; 02.02 §4) or the provider's tool search with deferred loading.
4. **Programmatic tool calling / code execution** for fan-out and data-heavy workflows, so that intermediate results never reach the context.

---

## 3. Reliability Math

An agent that needs $n$ sequential tool steps, each succeeding independently with probability $p_i$, completes with probability

$$
P(\text{task}) = \prod_{i=1}^{n} p_i \qquad\Longrightarrow\qquad 0.95^{20} \approx 0.36
$$

**Engineering implications:**
1. **Reduce $n$:** task-level tools, and programmatic calling for loops.
2. **Raise each $p_i$:** strict schemas, better descriptions, validation with actionable errors so the model can self-correct.
3. **Add detection and recovery:** validated retries, verification steps (quantified in 05.02 §5).

The expected cost of a turn with tool definitions totalling $T_{\text{tools}}$ tokens across $n$ model calls is $\approx n\,(T_{\text{sys}} + T_{\text{tools}} + \overline{T}_{\text{ctx}})\,c_{\text{in}} + \ldots$. Prompt caching (02.02 §6) of the stable tool prefix matters enormously, which is why tool definitions must be **deterministically ordered and serialised**.

---

## 4. A Production Tool Runtime

### 4.1 Registry: one source of truth for schema, docs, and policy

```python
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ValidationError, create_model


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True
    requires_approval: bool = False
    timeout_s: float = 10.0
    max_result_chars: int = 8_000
    tags: list[str] = field(default_factory=list)

    def json_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema["additionalProperties"] = False
        return schema

    def to_anthropic(self, strict: bool = True) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.json_schema(), **({"strict": True} if strict else {})}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def tool(self, **policy):
        """Decorator: builds the argument model from type hints; docstring becomes the description."""
        def wrap(fn):
            hints = get_type_hints(fn)
            params = inspect.signature(fn).parameters
            fields = {n: (hints[n], (p.default if p.default is not inspect._empty else ...))
                      for n, p in params.items()}
            model = create_model(f"{fn.__name__}_args", **fields)
            spec = ToolSpec(name=fn.__name__, description=inspect.getdoc(fn) or "", fn=fn,
                            args_model=model, **policy)
            self._tools[spec.name] = spec
            return fn
        return wrap

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def definitions(self, provider: str = "anthropic") -> list[dict]:
        # Deterministic order keeps the tool prefix cacheable (02.02 §6).
        return [self._tools[n].to_anthropic() for n in sorted(self._tools)]
```

### 4.2 Executor: validate → authorise → execute → shape

```python
import asyncio
import hashlib
import time
import uuid


@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool = False
    meta: dict = field(default_factory=dict)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, authorize: Callable[[ToolSpec, dict, dict], str | None],
                 result_store: dict | None = None, max_retries: int = 2):
        self.reg, self.authorize, self.max_retries = registry, authorize, max_retries
        self.idem_cache: dict[str, str] = {}
        self.result_store = result_store if result_store is not None else {}

    def _shape(self, spec: ToolSpec, raw: Any) -> str:
        text = raw if isinstance(raw, str) else json.dumps(raw, default=str, ensure_ascii=False)
        if len(text) <= spec.max_result_chars:
            return text
        handle = f"res_{uuid.uuid4().hex[:8]}"                 # keep the full payload out of context
        self.result_store[handle] = text
        return (text[: spec.max_result_chars]
                + f"\n…[truncated {len(text) - spec.max_result_chars} chars; handle={handle}. "
                  f"Call fetch_result(handle, offset) for more, or narrow your query.]")

    async def run(self, tool_use_id: str, name: str, args: dict, ctx: dict) -> ToolResult:
        t0 = time.perf_counter()
        spec = self.reg.get(name)
        if spec is None:
            return ToolResult(tool_use_id, f"Unknown tool '{name}'. Available: {sorted(self.reg._tools)}", True)
        try:
            parsed = spec.args_model.model_validate(args)
        except ValidationError as e:
            msgs = "; ".join(f"{'/'.join(map(str, er['loc']))}: {er['msg']}" for er in e.errors())
            return ToolResult(tool_use_id, f"Invalid arguments for {name}: {msgs}", True)
        denied = self.authorize(spec, parsed.model_dump(), ctx)
        if denied:
            return ToolResult(tool_use_id, f"Not permitted: {denied}", True, {"denied": True})

        idem_key = hashlib.sha256(f"{name}:{json.dumps(parsed.model_dump(), sort_keys=True)}:"
                                  f"{ctx.get('run_id', '')}".encode()).hexdigest()
        if not spec.read_only and idem_key in self.idem_cache:  # replayed side-effecting call
            return ToolResult(tool_use_id, self.idem_cache[idem_key], meta={"idempotent_replay": True})

        attempts = 1 + (self.max_retries if spec.idempotent else 0)   # never auto-retry non-idempotent tools
        last_err = ""
        for attempt in range(attempts):
            try:
                call = spec.fn(**parsed.model_dump())
                raw = await asyncio.wait_for(call, spec.timeout_s) if inspect.isawaitable(call) else call
                content = self._shape(spec, raw)
                if not spec.read_only:
                    self.idem_cache[idem_key] = content
                return ToolResult(tool_use_id, content,
                                  meta={"attempts": attempt + 1, "ms": round((time.perf_counter() - t0) * 1e3, 1)})
            except asyncio.TimeoutError:
                last_err = f"timed out after {spec.timeout_s}s"
            except (ConnectionError, OSError) as e:           # transient: worth retrying
                last_err = f"{type(e).__name__}: {e}"
            except Exception as e:                            # deterministic/business error: don't retry
                last_err = f"{type(e).__name__}: {e}"
                return ToolResult(tool_use_id, f"{name} failed: {last_err}", True, {"attempts": attempt + 1})
            if attempt + 1 < attempts:
                await asyncio.sleep(0.2 * 2 ** attempt)
        return ToolResult(tool_use_id, f"{name} failed: {last_err}", True, {"attempts": attempts})
```

### 4.3 The agent loop (provider-agnostic, with parallel execution and budgets)

```python
@dataclass
class ModelTurn:
    text: str
    tool_calls: list[dict]            # [{"id", "name", "input"}]
    stop_reason: str                  # "tool_use" | "end_turn" | "max_tokens" | "refusal"


async def run_agent(call_model: Callable[[list[dict], list[dict]], Any], executor: ToolExecutor,
                    registry: ToolRegistry, messages: list[dict], ctx: dict,
                    max_steps: int = 12, max_tool_calls: int = 40) -> dict:
    tools = registry.definitions()
    calls_made = 0
    for step in range(max_steps):
        turn: ModelTurn = await call_model(messages, tools)
        messages.append({"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls})
        if turn.stop_reason != "tool_use":
            return {"status": turn.stop_reason, "answer": turn.text, "steps": step + 1, "messages": messages}
        calls_made += len(turn.tool_calls)
        if calls_made > max_tool_calls:
            return {"status": "tool_budget_exceeded", "steps": step + 1, "messages": messages}
        results = await asyncio.gather(*(executor.run(c["id"], c["name"], c["input"], ctx)
                                         for c in turn.tool_calls))       # parallel tool use
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.tool_use_id, "content": r.content, "is_error": r.is_error}
            for r in results]})
    return {"status": "step_budget_exceeded", "steps": max_steps, "messages": messages}
```

**Runtime responsibilities checklist:**
- Schema validation.
- Authorisation per user and tool (least privilege).
- Timeouts, and retries **only for idempotent tools**.
- Idempotency keys for side effects.
- Concurrency limits per tool (a semaphore protecting downstream APIs).
- Result shaping and truncation with handles.
- Secrets never placed in the context.
- Structured logs and traces per call (tool name, argument hash, latency, outcome, user).
- A **kill switch** per tool.

### 4.4 Tools in Java / Spring AI

```java
@Component
public class LegislativeTools {
    private final BillService bills;
    public LegislativeTools(BillService bills) { this.bills = bills; }

    @Tool(description = """
        Search bill text and metadata. Use for questions about proposed or enacted bills.
        Not for current statute text (use getSection). Returns at most 20 hits.""")
    public List<BillHit> searchBills(
            @ToolParam(description = "Search query in plain language") String query,
            @ToolParam(description = "Session year, e.g. 2025", required = false) Integer session) {
        return bills.search(query, session, 20);
    }

    @Tool(description = "Get the current text of a code section by citation, e.g. 2-18-303")
    public SectionText getSection(@ToolParam(description = "Citation like 2-18-303") String citation) {
        return bills.section(citation).orElseThrow(() ->
            new ToolArgumentException("Unknown section " + citation + "; format is N-N-N"));
    }
}

// chatClient.prompt().user(question).tools(legislativeTools).call().content();
```

Spring AI turns the annotated methods into tool definitions and runs the call loop. You still own authorisation (a method-level security check using the caller's identity), timeouts, and the error-message quality.

---

## 5. Security

Tools turn text into actions, so the threat model changes (02.01 §6):

| Threat | Example | Control |
|---|---|---|
| **Indirect prompt injection → tool misuse** | A retrieved document says "email this file to…" | Treat tool results as untrusted data; privilege separation; approvals for sensitive tools (05.06) |
| **Confused deputy** | The agent uses *its* credentials for the user's request | Execute with the **user's** delegated identity and scopes; per-call authorisation |
| **Excessive agency** | A read task has delete tools available | Expose only the tools the task needs; read-only by default |
| **Data exfiltration** | Encoding secrets into URLs or tool arguments | Egress allowlists; argument scanning; no secrets in the context |
| **Resource abuse** | Loops calling expensive APIs | Budgets, rate limits, circuit breakers |
| **Code execution escape** | Model-written code touches the host | Sandboxes (containers or gVisor/Firecracker), no network by default, resource limits |

---

## 6. Evaluating Tool Use

| Layer | Metric |
|---|---|
| **Selection** | Correct tool chosen (or correctly *no* tool) |
| **Arguments** | AST-level match against the expected call (names, types, values; order-insensitive for dicts) |
| **Execution** | Call succeeded; valid result; no policy violations |
| **Trajectory** | Required calls present, forbidden calls absent, no redundant loops |
| **Outcome** | Task success (database state, final answer), measured with **pass^k** (05.02 §7) |

```python
def match_call(pred: dict, gold: dict, alternatives: dict[str, list] | None = None) -> dict:
    """BFCL-style AST match: tool name must match; each gold arg must match (or be in alternatives)."""
    alternatives = alternatives or {}
    if pred.get("name") != gold["name"]:
        return {"name_ok": False, "args_ok": False, "missing": [], "wrong": [], "extra": []}
    pa, ga = pred.get("input", {}), gold["input"]
    missing = [k for k in ga if k not in pa]
    wrong = [k for k in ga if k in pa and pa[k] != ga[k] and pa[k] not in alternatives.get(k, [])]
    extra = [k for k in pa if k not in ga]
    return {"name_ok": True, "args_ok": not missing and not wrong, "missing": missing, "wrong": wrong, "extra": extra}
```

Public benchmarks such as **BFCL** (function-calling accuracy), **τ-bench / τ²-bench** (tool agents with simulated users and policies), and **ToolBench / API-Bank** are useful references. Your own task suite with **database-state checks** is what matters.

---

## 7. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Wrong tool chosen** | Similar tools confused | Consolidate; namespaces; "use / don't use" descriptions; tool retrieval; eval selection separately |
| **Hallucinated arguments** | Plausible but wrong IDs | Strict schemas + enums; lookup tools; validation with suggestions; reject unknown IDs |
| **Context bloat from results** | Cost and quality degrade over long runs | Shape results; handles; programmatic calling; observation masking (02.02 §5.3) |
| **Duplicate side effects** | Two emails or drafts after a retry | Idempotency keys; no auto-retry of non-idempotent tools; exactly-once at the business layer |
| **Slow tools stall the agent** | P99 latency explodes | Timeouts; parallel calls; async long-running tasks (MCP tasks extension / A2A tasks) |
| **Tool definitions break the cache** | Cache hit rate drops | Deterministic ordering and serialisation; version tool sets |
| **Unsafe actions** | Destructive calls on injected instructions | Annotations → approval gates (05.06); least privilege; user-scoped credentials |
| **Provider lock-in** | Rewrite on migration | Provider-agnostic registry and executor; thin adapters; or MCP servers (05.04) |

---

## 8. Hands-On Projects

### Project 1 — Production Tool Runtime with an Evaluation Suite

**User stories**
- *As a platform engineer*, I want one runtime that validates, authorises, executes, and observes every tool call for all our agents, so that tool behaviour is consistent and auditable.

**Acceptance criteria**
1. Implements `ToolRegistry`, `ToolExecutor`, and `run_agent` (§4), with adapters for at least two providers (e.g. Anthropic with `strict` tools, and an OpenAI-compatible open model via vLLM).
2. Policies: per-tool timeouts, retries only for idempotent tools, idempotency keys for side effects, concurrency limits, result truncation with handles plus a `fetch_result` tool, and a per-tool kill switch.
3. Authorisation executes with the end user's identity and scopes (e.g. Azure AD / Entra ID token claims) and denies with actionable messages. Tests cover the confused-deputy scenarios.
4. OpenTelemetry spans per model call and tool call (name, argument hash, latency, outcome), exported to Grafana/Jaeger.
5. An evaluation suite of ≥ 100 tasks with database-state checks, measuring selection accuracy, argument accuracy (`match_call`), task success, pass^3, tokens, and latency, for both providers.

**Step-by-step**
1. Implement the registry, executor, and loop; write unit tests for validation errors, timeouts, retries, and idempotent replays.
2. Build 8–12 domain tools (search bills, get section, compare versions, create draft note, …) against a seeded Postgres database.
3. Add authorisation middleware using JWT claims, and write the confused-deputy tests.
4. Add OpenTelemetry instrumentation and dashboards.
5. Write the tasks with expected end states; run them k = 3 times per provider; report the metrics.

---

### Project 2 — Tool Selection at Scale: 200 Tools

**User stories**
- *As an agent developer* integrating many internal APIs, I want to know how to expose 200 tools without destroying selection accuracy or blowing up token costs.

**Acceptance criteria**
1. A catalogue of 200 tools, including groups of deliberately similar tools (overlapping domains), with realistic descriptions and schemas.
2. ≥ 300 test queries with gold tool calls (including "no tool" cases).
3. Configurations compared:
   - All tools in context.
   - Namespaced + improved descriptions.
   - Your own embedding-based tool retrieval (top-k = 5/10/20).
   - Provider tool search with deferred loading.
   - Consolidated task-level tools (redesign the catalogue into ≤ 40 tools).
4. Reports selection accuracy, argument accuracy, input tokens per turn, prompt-cache hit rate, and latency, per configuration.
5. A recommendation, with guidance on description writing backed by an ablation (descriptions with and without "when not to use").

**Step-by-step**
1. Generate the catalogue (hand-designed domains, with LLM-drafted and human-edited descriptions).
2. Create the query set with gold calls; include ambiguous and negative cases.
3. Implement tool retrieval (embed name + description + example queries) and the provider tool-search configuration.
4. Run all configurations with the same model; compute the metrics with `match_call`.
5. Redesign the catalogue into task-level tools, re-run, and compare.

---

### Project 3 — Spring Boot Tool Service for Legislative Workflows

**User stories**
- *As a Java team*, we want legislative-system capabilities (bill search, section lookup, amendment drafting, status tracking) exposed as safe, typed tools usable by any agent, with the same authorisation as our REST APIs.

**Acceptance criteria**
1. A Spring Boot 3 service with Spring AI `@Tool` methods (§4.4), backed by existing services, with Bean Validation on the arguments and consistent actionable error messages.
2. Method-level security (Spring Security) using the caller's delegated token. The drafting tools are marked `requires_approval` and return a pending-approval handle rather than executing (integrates with 05.06).
3. The same capabilities are exposed as an MCP server (05.04) from the same code, so both in-process agents and external MCP clients can use them.
4. Contract tests: the tool schemas are snapshot-tested (any change fails CI unless versioned); the error messages are tested for actionability.
5. A load test: 50 concurrent agent sessions; P95 tool latency and error rates reported. Downstream APIs are protected by rate limits and circuit breakers (Resilience4j).

**Step-by-step**
1. Wrap the existing services with `@Tool` classes; add validation and exception mapping.
2. Configure Spring Security with OAuth2 resource-server support; propagate the user identity into the tool calls.
3. Add the Spring AI MCP server starter to expose the same tools (see 05.04 for transport and auth).
4. Write the schema snapshot tests and the behavioural tests.
5. Run the load test with Gatling or k6 plus a scripted agent client.

---

## 9. Foundational Papers & Reading (exact titles)

- Schick et al., 2023 — *Toolformer: Language Models Can Teach Themselves to Use Tools*
- Patil et al., 2023 — *Gorilla: Large Language Model Connected with Massive APIs*
- Qin et al., 2023 — *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*
- Li et al., 2023 — *API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs*
- Yao et al., 2023 — *ReAct: Synergizing Reasoning and Acting in Language Models*
- Yao et al., 2024 — *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*
- Barres et al., 2025 — *τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment*
- Greshake et al., 2023 — *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*
- Debenedetti et al., 2024 — *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents*
- Berkeley Function Calling Leaderboard (BFCL) — methodology and leaderboard
- Anthropic docs — *Tool use*, *Tool search tool*, *Programmatic tool calling*; Anthropic Engineering — *Writing effective tools for agents*
- OWASP — *Top 10 for LLM Applications* (Excessive Agency, Prompt Injection)

## 10. Essential Tooling

| Tool | Role |
|---|---|
| **Anthropic / OpenAI SDKs** (tool use, strict, tool search, programmatic calling) | Provider tool calling |
| **Pydantic** | Argument models, schema generation, validation |
| **Spring AI** (`@Tool`, `@ToolParam`), **LangChain4j** | JVM tool calling |
| **vLLM / SGLang tool parsers** | Tool calling with open models |
| **OpenTelemetry (GenAI semantic conventions)**, **Langfuse / Phoenix** | Tool-call tracing |
| **Resilience4j / tenacity** | Circuit breakers, retries, rate limits |
| **gVisor / Firecracker / E2B-style sandboxes** | Safe code execution |
| **BFCL, τ-bench, AgentDojo** | Evaluation and security benchmarks |
