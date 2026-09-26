# 05.05 — A2A & Agent Interoperability

> **Module 5: Tools & Agents** · Subtopic 5 of 6
> **Prerequisites:** 05.02 (agent workflows, multi-agent), 05.03 (task state, durability), 05.04 (MCP, OAuth for agents), HTTP/gRPC, webhooks, PKI basics.
> **Outcome:** you understand where agent-to-agent protocols fit relative to MCP, can implement and operate agents that interoperate over **A2A v1.0** (agent cards, tasks, messages, artifacts, streaming, push notifications), and can design the trust, identity, and reliability layer that cross-organisation agent collaboration requires.

> **Version note (checked September 2026).**
> - **A2A v1.0**, the first stable specification, was released in **March 2026**. It added multi-protocol bindings with version negotiation, multi-tenancy, and **signed agent cards**.
> - Google launched A2A in April 2025 and donated it to the Linux Foundation. IBM's ACP merged into it in August 2025.
> - In **August 2026**, A2A joined the **Agentic AI Foundation (AAIF)**, alongside MCP, AGENTS.md, goose, and agentgateway.
> - Method and field names below follow the v1.0 spec. Verify against the spec version your SDK implements.

---

## 1. Tools vs Agents: Why a Second Protocol?

| | **MCP** (05.04) | **A2A** |
|---|---|---|
| Connects | An AI application ↔ **tools, data, prompts** | An agent ↔ **another agent** |
| Counterpart is | A capability with a schema (transparent, deterministic-ish) | An **opaque, autonomous peer** with its own reasoning, tools, and memory |
| Interaction | Call → result (plus input requests, tasks extension) | **Task** lifecycle: multi-turn, long-running, may ask for input, streams progress, produces artifacts |
| Typical boundary | Inside an application or organisation | Across teams, vendors, and organisations |
| Discovery | `server/discover`, `tools/list` | **Agent Card** (skills, interfaces, security schemes) |

They are **complementary**. An agent uses MCP to reach its tools, and A2A to delegate work to other agents that have *their own* tools.

```
  ┌──────────── Org A ────────────┐                 ┌────────────── Org B ───────────────┐
  │ Legislative Analyst Agent      │  A2A (task)     │ Fiscal Analysis Agent              │
  │  ├─ MCP → bill DB server       │ ──────────────► │  ├─ MCP → budget warehouse server  │
  │  └─ MCP → statute server       │ ◄────────────── │  └─ MCP → revenue-model server      │
  └───────────────────────────────┘  artifacts,      └────────────────────────────────────┘
                                      status stream
```

---

## 2. A2A Core Concepts (v1.0)

### 2.1 Agent Card: discovery and contract

A JSON document published by the remote agent, conventionally at a well-known URI (`/.well-known/agent-card.json`; confirm the path for your spec version). It declares:
- **Identity:** name, description, provider, version.
- **Interfaces:** the protocol bindings supported (JSON-RPC 2.0, gRPC, HTTP+JSON/REST) and their endpoints.
- **Capabilities:** streaming, push notifications, extended card.
- **Skills:** id, name, description, tags, examples, input and output modes.
- **Security schemes:** API key, HTTP auth, OAuth2, OpenID Connect, mutual TLS.
- An optional **signature**.

An authenticated **extended card** (`GetExtendedAgentCard`) can reveal additional skills to authorised clients.

```json
{
  "name": "Fiscal Analysis Agent",
  "description": "Estimates fiscal impact of proposed legislation using the state budget warehouse.",
  "version": "2.3.0",
  "provider": { "organization": "Legislative Fiscal Division" },
  "supportedInterfaces": [
    { "protocolBinding": "JSONRPC", "url": "https://fiscal-agent.example.gov/a2a/v1" },
    { "protocolBinding": "GRPC",    "url": "fiscal-agent.example.gov:443" }
  ],
  "capabilities": { "streaming": true, "pushNotifications": true, "extendedAgentCard": true },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["application/json", "text/markdown"],
  "skills": [
    { "id": "fiscal-note-estimate",
      "name": "Fiscal note estimate",
      "description": "Given bill text or bill_id, estimate 4-year fiscal impact by fund with assumptions.",
      "tags": ["fiscal", "budget", "legislation"],
      "examples": ["Estimate the fiscal impact of HB-45 (2025)"] }
  ],
  "securitySchemes": { "oauth": { "oauth2SecurityScheme": { "flows": { "clientCredentials": {
      "tokenUrl": "https://login.example.gov/oauth2/token", "scopes": { "fiscal.read": "Run estimates" } } } } } },
  "security": [ { "oauth": ["fiscal.read"] } ]
}
```

(The field names are illustrative of the v1.0 structure. Generate cards with your SDK rather than by hand.)

### 2.2 Tasks, messages, parts, artifacts

- **Message:** a turn with `role` (`user` or `agent`) and a list of **Parts** (text, file references or bytes, structured data). Messages carry a `messageId` (use it for idempotency), plus `contextId` (the conversation) and `taskId` (continuation).
- **Task:** the stateful unit of work, with an ID, `status` (state + message + timestamp), history, and **artifacts**.
- **Artifact:** a deliverable produced by the agent (e.g. a fiscal note JSON, a report), composed of parts. Artifacts can be streamed incrementally.

**Task states** (v1.0 enum style): `SUBMITTED` → `WORKING` → one of:
- **Interrupted:** `INPUT_REQUIRED`, `AUTH_REQUIRED`.
- **Terminal:** `COMPLETED`, `FAILED`, `CANCELED`, `REJECTED`.

```
 SUBMITTED ──► WORKING ──► COMPLETED
                 │  ▲ ─► FAILED
                 │  │ ─► CANCELED (client CancelTask)
                 ▼  │ ─► REJECTED (agent declines)
          INPUT_REQUIRED / AUTH_REQUIRED ── client sends message with taskId ──┘
```

### 2.3 Operations

| Operation | Purpose |
|---|---|
| `SendMessage` | Start or continue a task; returns a Task (or a direct Message for trivial replies) |
| `SendStreamingMessage` | Same, but streams `Task`, `Message`, `TaskStatusUpdateEvent`, and `TaskArtifactUpdateEvent` |
| `GetTask` / `ListTasks` | Poll state; query tasks with filters and pagination |
| `CancelTask` | Request cancellation |
| `SubscribeToTask` | Re-attach a stream to an existing task |
| `Create/Get/List/DeleteTaskPushNotificationConfig` | Register webhooks for long-running tasks |
| `GetExtendedAgentCard` | Authenticated card with additional details |

Clients send the **`A2A-Version`** service parameter (a header in HTTP bindings) on each request. **Extensions** are declared in the card and negotiated through the `A2A-Extensions` parameter.

### 2.4 Three interaction styles

| Style | Mechanism | Use when |
|---|---|---|
| **Request/response + polling** | `SendMessage` → `GetTask` until terminal | Simple clients; short tasks |
| **Streaming** | `SendStreamingMessage` / `SubscribeToTask` | Interactive UIs; progress and partial artifacts |
| **Push notifications** | Webhook configured per task; the agent POSTs updates | Long-running tasks (minutes to days); clients that can't hold connections |

---

## 3. Implementing the Server Side

The essence of an A2A server is a **durable task state machine** (05.03) behind a protocol binding. Use an official SDK for the wire format. The logic below is what you own.

```python
import time
import uuid
from dataclasses import dataclass, field

TERMINAL = {"COMPLETED", "FAILED", "CANCELED", "REJECTED"}
INTERRUPTED = {"INPUT_REQUIRED", "AUTH_REQUIRED"}
ALLOWED = {
    "SUBMITTED": {"WORKING", "REJECTED", "CANCELED", "FAILED"},
    "WORKING": {"COMPLETED", "FAILED", "CANCELED", "INPUT_REQUIRED", "AUTH_REQUIRED"},
    "INPUT_REQUIRED": {"WORKING", "CANCELED", "FAILED"},
    "AUTH_REQUIRED": {"WORKING", "CANCELED", "FAILED"},
}


class InvalidTransition(Exception):
    pass


@dataclass
class Task:
    id: str
    context_id: str
    state: str = "SUBMITTED"
    history: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    status_message: str = ""
    updated: float = field(default_factory=time.time)

    def transition(self, new: str, message: str = "") -> None:
        if new not in ALLOWED.get(self.state, set()):
            raise InvalidTransition(f"{self.state} -> {new} not allowed")
        self.state, self.status_message, self.updated = new, message, time.time()


class A2AServer:
    """Binding-agnostic core: handlers for SendMessage / GetTask / CancelTask with idempotency."""

    def __init__(self, skill_handler):
        self.tasks: dict[str, Task] = {}
        self.seen_messages: dict[str, str] = {}              # messageId -> taskId (idempotency)
        self.handle = skill_handler                          # (task, message) -> dict(state, artifact?, message?)

    def send_message(self, message: dict, principal: str) -> dict:
        mid = message["messageId"]
        if mid in self.seen_messages:                          # retried delivery: return the same task
            return self._view(self.tasks[self.seen_messages[mid]])
        if message.get("taskId"):
            task = self.tasks.get(message["taskId"])
            if task is None:
                return {"error": {"code": "TaskNotFound"}}
            if task.state in TERMINAL:
                return {"error": {"code": "TaskNotContinuable", "state": task.state}}
            if task.state in INTERRUPTED:
                task.transition("WORKING", "resumed with client input")
        else:
            task = Task(id=str(uuid.uuid4()), context_id=message.get("contextId") or str(uuid.uuid4()))
            self.tasks[task.id] = task
            task.transition("WORKING", "accepted")
        self.seen_messages[mid] = task.id
        task.history.append({"role": "user", "principal": principal, **message})
        out = self.handle(task, message)                        # in production: enqueue durable work (05.03)
        if out.get("artifact"):
            task.artifacts.append(out["artifact"])
        task.transition(out["state"], out.get("message", ""))
        return self._view(task)

    def get_task(self, task_id: str) -> dict:
        t = self.tasks.get(task_id)
        return self._view(t) if t else {"error": {"code": "TaskNotFound"}}

    def cancel_task(self, task_id: str) -> dict:
        t = self.tasks.get(task_id)
        if t is None:
            return {"error": {"code": "TaskNotFound"}}
        if t.state in TERMINAL:
            return {"error": {"code": "TaskNotCancelable", "state": t.state}}
        t.transition("CANCELED", "canceled by client")
        return self._view(t)

    @staticmethod
    def _view(t: Task) -> dict:
        return {"id": t.id, "contextId": t.context_id, "status": {"state": t.state, "message": t.status_message},
                "artifacts": t.artifacts}
```

**Server responsibilities beyond the protocol:**
- Durable task storage and event history (05.03).
- **Idempotency** on `messageId`.
- Per-tenant isolation (v1.0 multi-tenancy).
- Authorisation per skill.
- Budgets and timeouts per task.
- Cancellation that actually stops the underlying work.
- Artifact retention policies.

---

## 4. Implementing the Client Side

```python
import time
import uuid

TERMINAL = {"COMPLETED", "FAILED", "CANCELED", "REJECTED"}


def run_remote_task(send, get, message: dict, answer_input=None, poll_s: float = 0.0,
                    max_polls: int = 100) -> dict:
    """send(message)->task view; get(task_id)->task view. answer_input(task)->message for INPUT_REQUIRED."""
    task = send(message)
    for _ in range(max_polls):
        state = task.get("status", {}).get("state")
        if state in TERMINAL or "error" in task:
            return task
        if state == "INPUT_REQUIRED":
            if answer_input is None:
                return task                                        # surface to a human (05.06)
            reply = answer_input(task)
            task = send({**reply, "taskId": task["id"], "contextId": task["contextId"],
                         "messageId": str(uuid.uuid4())})
            continue
        if poll_s:
            time.sleep(poll_s)                                     # production: exponential backoff + jitter
        task = get(task["id"])
    return {**task, "client_error": "poll budget exhausted"}
```

**Client rules:**
- Choose skills by reading the **card** (and verifying its signature, §5).
- Send a unique `messageId` per logical message, and reuse it on network retries.
- Carry `contextId` for multi-turn collaboration.
- **Treat remote agents' outputs as untrusted input** (02.01 §6): validate artifacts against expected schemas (02.03), and never let a remote agent's text become instructions to privileged tools.
- Bound everything: timeouts, poll budgets, cost, and the depth of delegation chains (to prevent agent-to-agent loops).

---

## 5. Trust, Identity, and Security

Cross-organisation agents need **authentication of both sides**, **authorisation per skill**, and **integrity of discovery data**.

| Concern | Mechanism |
|---|---|
| **Is this card really from the provider?** | **Signed agent cards** (v1.0): a signature over the canonicalised card, verified against the provider's published keys (e.g. JWKS). Pin the keys for known partners |
| **Is the caller who it claims to be?** | Declared security schemes: OAuth2 client credentials, OIDC, mTLS (SPIFFE/SPIRE identities within a mesh) |
| **What may the caller do?** | Scopes per skill; tenant isolation; rate limits |
| **Webhooks** | Authenticate push notifications (e.g. a token or signature the client registered); verify timestamps to block replays; allowlist callback URLs (SSRF protection) |
| **Content risk** | Artifacts and messages are untrusted: schema validation, injection defences, and a DLP policy on what your agent sends out |
| **Delegation chains** | Propagate the *user's* delegated identity (token exchange) where appropriate; record the chain in audit logs; cap the hop count |

```python
import hashlib
import hmac
import json
import time


def canonical_json(obj) -> bytes:
    """Deterministic serialisation (approximation of RFC 8785 JCS for plain JSON types)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def card_digest(card: dict) -> str:
    body = {k: v for k, v in card.items() if k != "signatures"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def verify_webhook(body: bytes, headers: dict, secret: bytes, tolerance_s: int = 300) -> bool:
    """One common pattern for authenticating push notifications you registered for: HMAC over timestamp.body."""
    ts, sig = headers.get("X-Timestamp", ""), headers.get("X-Signature", "")
    if not ts.isdigit() or abs(time.time() - int(ts)) > tolerance_s:
        return False
    expected = hmac.new(secret, ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)
```

In production, card signatures use **asymmetric JWS** (the verifier needs only public keys), not HMAC. The `card_digest` above shows why canonicalisation matters: a whitespace or key-order difference must not break verification.

---

## 6. The Broader Interoperability Landscape

| Layer | Standards / projects | Role |
|---|---|---|
| Agent ↔ tools/data | **MCP** | Capabilities and context |
| Agent ↔ agent | **A2A** (ACP merged in) | Delegation, tasks, artifacts |
| Agent ↔ user interface | **AG-UI** (agent–user interaction protocol), vendor UI protocols | Streaming agent events to front-ends |
| Agent instructions for repos | **AGENTS.md** | Context conventions for coding agents |
| Traffic control | **agentgateway**, API gateways | Auth, policy, observability for MCP/A2A traffic |
| Observability | **OpenTelemetry GenAI semantic conventions** | Cross-vendor traces for LLM, tool, and agent spans |
| Framework-level | OpenAI Agents SDK handoffs, Google ADK, Claude Agent SDK, LangGraph, Spring AI | In-process multi-agent composition |

**Design guidance:**
- Use **in-process composition** (05.02) when you control all the agents and they share a runtime.
- Use **A2A** at *organisational or deployment boundaries*: different teams, languages, vendors, trust domains, or release cycles.
- Use **MCP** for tools. Don't wrap a deterministic API as an "agent" just to use A2A.

---

## 7. Reliability Patterns for Remote Agents

- **Timeouts and deadlines** propagated in messages (metadata) and enforced on both sides.
- **Retries with idempotency** (`messageId`), and exponential backoff with jitter.
- **Circuit breakers** per remote agent; fall back to a degraded local path or a human queue.
- **Result validation:** artifact schemas plus semantic checks (02.03 §4). Reject and re-ask with precise errors.
- **Durable client state:** the client's own workflow checkpoints the remote task ID, so it can resume after a crash (05.03).
- **Contract testing:** consumer-driven contract tests against the remote card and skills. Alert on card changes, much like MCP definition pinning (05.04 §7).
- **SLAs:** latency and success-rate objectives per skill, published in the card or its documentation, and monitored.

---

## 8. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Version skew** | Calls fail after a partner upgrade | `A2A-Version` negotiation; support N-1; contract tests; card change alerts |
| **Lost tasks** | The client crashed mid-task; the work is orphaned | Persist task IDs in durable workflows; `ListTasks` for reconciliation; push notifications |
| **Duplicate work** | Network retries create duplicate tasks | `messageId` idempotency on the server |
| **Delegation loops** | Agents keep delegating to each other | Hop-count metadata; call-graph audit; budgets |
| **Untrusted outputs** | Injection via partner artifacts | Schema validation; spotlighting; privilege separation |
| **Spoofed cards / endpoints** | Traffic sent to an impostor | Signed cards; key pinning; mTLS |
| **Webhook abuse** | Forged or replayed updates; SSRF | Authenticated webhooks with timestamps; callback allowlists |
| **Observability gaps** | Cross-org failures are hard to diagnose | Trace-context propagation; correlation IDs (`taskId`, `contextId`) in logs |

---

## 9. Hands-On Projects

### Project 1 — Cross-Stack Agent Collaboration over A2A (Python ↔ Java)

**User stories**
- *As a legislative analyst*, I want my analysis agent to delegate fiscal estimates to the fiscal division's agent (built by another team, on another stack) and include its results, with provenance, in my bill impact report.

**Acceptance criteria**
1. Agent A (Python) is the bill analysis agent with MCP tools (05.04). Agent B (Java, e.g. Spring Boot + the A2A Java SDK) is the fiscal agent with a `fiscal-note-estimate` skill that returns a JSON artifact matching a published schema.
2. B publishes an agent card (JSON-RPC binding at least; gRPC optional), with OAuth2 client credentials in its security schemes. A verifies the card's signature against pinned keys.
3. Flows supported:
   - Synchronous polling.
   - Streaming progress to A's UI.
   - `INPUT_REQUIRED` when the bill has no fiscal note data (A asks the user, then continues the task).
   - `CancelTask`.
4. Reliability: A retries with the same `messageId` after injected network failures (no duplicate tasks on B). B's tasks are durable across a restart.
5. End-to-end traces (OpenTelemetry) span both services, correlated by `taskId`.

**Step-by-step**
1. Define B's skill contract (input and output schemas) and build its agent card.
2. Implement B with the Java SDK, a durable task store (Postgres), and a fiscal estimation pipeline (tools via MCP or direct).
3. Implement A's client using `run_remote_task` semantics, plus streaming and cancellation.
4. Add OAuth2 (Keycloak or Entra ID), card signing (JWS) and verification, and key pinning.
5. Run the fault-injection and restart tests; verify the traces.

---

### Project 2 — Agent Registry and A2A Gateway

**User stories**
- *As an enterprise architect*, I want a catalogue of approved internal and partner agents with verified cards, access policies, and usage analytics, so that teams can safely discover and use each other's agents.

**Acceptance criteria**
1. The registry stores agent cards, verified signatures, owners, data classifications, allowed consumers, and SLA targets. It supports skill search (keyword + embedding) and card-change detection (`card_digest` diffs).
2. The gateway proxies A2A traffic with authentication (token validation, mTLS), per-skill authorisation, rate limits, hop-count enforcement, audit logs, and payload size limits.
3. A reconciliation job lists stale tasks per consumer and alerts on orphans.
4. Dashboards per agent and skill: requests, success rate, latency, cost, and terminal-state distribution.
5. Security tests: a spoofed card, an unauthorised skill call, a delegation loop, and a replayed webhook — all blocked.

**Step-by-step**
1. Design the registry schema and API; implement signature verification (JWKS) and digest tracking.
2. Build the gateway (e.g. Envoy with ext_authz plus a policy service, or Spring Cloud Gateway filters).
3. Add hop-count metadata propagation and enforcement.
4. Instrument with OpenTelemetry and build the dashboards.
5. Write the adversarial tests.

---

### Project 3 — A2A Conformance and Chaos Suite

**User stories**
- *As a QA lead*, I want an automated suite that verifies any A2A agent our teams ship behaves correctly under normal and adverse conditions, before it's added to the registry.

**Acceptance criteria**
1. The protocol conformance tests cover: card validity; each declared binding; `SendMessage` / `GetTask` / `CancelTask` semantics; the state-machine transitions (no illegal transitions); `messageId` idempotency; `INPUT_REQUIRED` continuation; streaming event order; and push-notification delivery and authentication.
2. Chaos tests: network drops mid-stream (the client re-subscribes), server restarts during `WORKING` (the task survives), slow responses (client deadlines), and malformed artifacts (client rejection).
3. The suite runs as a CLI and in CI against any agent URL, producing a JUnit/HTML report and a pass/fail gate for registry admission.
4. It is validated against at least two independent implementations (e.g. your Python and Java agents).

**Step-by-step**
1. Encode the state machine (`ALLOWED`) and the protocol expectations as test cases.
2. Build a test client with fault-injection hooks (a proxy that drops or delays).
3. Implement the push-notification receiver with `verify_webhook` and assertions.
4. Package the suite with reporting, and integrate it into CI.
5. Run it against both agents; file and fix the issues found.

---

## 10. Specifications & Reading

- **Agent2Agent (A2A) Protocol Specification v1.0** (a2a-protocol.org) — Agent Card, Task lifecycle, Protocol bindings (JSON-RPC, gRPC, HTTP+JSON), Push notifications, Security, Extensions
- Agentic AI Foundation — *A2A joins AAIF's open agentic stack* (2026)
- Google Developers Blog, 2025 — *Announcing the Agent2Agent Protocol (A2A)*
- MCP Specification 2026-07-28 (for the tool layer; 05.04)
- RFC 8785 — *JSON Canonicalization Scheme (JCS)*; RFC 7515 — *JSON Web Signature (JWS)*; RFC 8693 — *OAuth 2.0 Token Exchange*; RFC 8615 — *Well-Known Uniform Resource Identifiers*
- OpenTelemetry — *Semantic conventions for generative AI systems*
- Cemri et al., 2025 — *Why Do Multi-Agent LLM Systems Fail?* (coordination failure modes that also apply across protocols)
- Ehtesham et al., 2025 — *A survey of agent interoperability protocols: Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent-to-Agent Protocol (A2A), and Agent Network Protocol (ANP)*

## 11. Essential Tooling

| Tool | Role |
|---|---|
| **A2A SDKs** (Python, JS/TS, Java, Go, .NET — check the current list and v1.0 support) | Wire format, bindings, streaming, push |
| **A2A Inspector / sample agents** | Debugging and reference implementations |
| **Spring Boot / Quarkus** | JVM agent services |
| **Keycloak / Entra ID, SPIFFE/SPIRE** | Agent identity (OAuth2 client credentials, workload identity, mTLS) |
| **agentgateway / Envoy / Spring Cloud Gateway** | Policy enforcement for A2A traffic |
| **Temporal / Postgres** | Durable task state on both sides |
| **OpenTelemetry** | Cross-agent tracing |
| **Pact** (consumer-driven contracts) | Contract tests for skills and artifacts |
