# Model Context Protocol (MCP) — Master Study Guide

> **Track:** Agents + Production · **Module:** 10
> **Prerequisites:** Modules 01–09.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** **MCP (Model Context Protocol)** is a standardized protocol for connecting LLMs to tools, resources, and prompts — introduced by Anthropic in late 2024 and widely adopted through 2025–2026. Think of MCP as the "**USB-C for AI**": before it, every tool/agent framework had its own way of exposing tools; each LLM app had to integrate each one. With MCP, a tool exposes an MCP server once, and any MCP-aware LLM app can use it.

By 2026, MCP is table stakes: Claude Desktop, Cursor, Zed, VS Code (Copilot), Cline, and many other coding agents all speak MCP. Enterprise integrations (Google Drive, GitHub, Slack, Postgres, and thousands of servers) publish MCP endpoints. Building your own MCP server is straightforward and immediately gives your tool a portable audience.

**Fundamental principles you must own:**

1. **MCP standardizes** three things: **tools** (functions), **resources** (data), and **prompts** (templates).
2. **Client-server architecture.** The **client** is the LLM app; the **server** exposes tools/resources; they communicate over JSON-RPC.
3. **Transport-agnostic.** Servers can run over stdio (local processes), HTTP+SSE, or WebSockets.
4. **Discoverability.** Clients can list capabilities of a server before deciding to use them.
5. **Security is a first-class concern.** Servers run with the client's permissions; tool calls need user consent.
6. **You can build both sides.** Publish a server (my_data → MCP) or embed a client (my_agent → MCP).

If you retain nothing else: **MCP is the USB-C of AI tool integration. Learn to be both a server publisher and a client consumer.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Architecture and Entities

MCP is a **client-server protocol** using JSON-RPC 2.0 over a chosen transport.

**Roles:**

- **Host** — the user-facing application (Claude Desktop, Cursor, your custom app).
- **Client** — the connection manager inside the host that talks to one or more servers.
- **Server** — a process (local or remote) that exposes tools, resources, or prompts.

One host can connect to many servers (each with its own capabilities). Each host-client-server relationship is bidirectional over JSON-RPC.

**What servers expose:**

**1. Tools** — callable functions (like OpenAI function calling but decoupled from any specific model). Each tool has a name, description, and input schema.

**2. Resources** — read-only data the LLM can access on demand. Examples: files, database rows, config values. Identified by URI (e.g., `file:///docs/handbook.md`, `postgres://db/customers/42`).

**3. Prompts** — pre-defined prompt templates the server exposes. The client can invoke them with arguments.

**4. (Emerging) Sampling & Roots** — advanced capabilities: server-initiated LLM calls, filesystem root scoping. Growing surface area.

**Transports:**

- **stdio** — server runs as a subprocess; client writes to stdin, reads from stdout. Great for local tools (Claude Desktop's default).
- **HTTP + SSE** — server is a remote HTTP endpoint with SSE for streaming. Great for cloud services.
- **WebSocket** — bidirectional streaming. Less common in practice.

**Message flow:**

1. **Initialize** — client and server negotiate protocol version and capabilities.
2. **List** — client asks server for available tools/resources/prompts.
3. **Call** — client invokes a tool or reads a resource.
4. **Notifications** — server can push updates (resource changed, progress).

---

### 2.2 Writing an MCP Server (Python Example)

Use the official Python SDK (`mcp`):

```python
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("legmt-bill-tools")

@mcp.tool()
def get_bill(bill_id: str) -> dict:
    """Fetch a bill by ID.

    Args:
        bill_id: The unique identifier of the bill, e.g. 'HB-247'.

    Returns:
        A dict with bill metadata: title, sponsor, status, session, text.
    """
    # ... lookup from your database
    return {"bill_id": bill_id, "title": "...", "sponsor": "...", ...}

@mcp.tool()
def list_bills_by_sponsor(sponsor_name: str, session: str = "current") -> list[dict]:
    """List all bills by a given sponsor in a session."""
    return [...]

@mcp.resource("bill://{bill_id}/text")
def get_bill_text(bill_id: str) -> str:
    """Return the full text of a bill as a resource."""
    return fetch_bill_text_from_db(bill_id)

@mcp.prompt()
def summarize_bill(bill_id: str) -> list[dict]:
    """A prompt template for summarizing a bill."""
    return [{
        "role": "user",
        "content": {"type": "text",
                    "text": f"Summarize bill {bill_id} in three bullet points, "
                            f"focusing on: (1) intent, (2) key provisions, (3) fiscal impact."},
    }]

if __name__ == "__main__":
    mcp.run()   # runs over stdio by default
```

That's it. `mcp.run()` handles the JSON-RPC protocol; your tool functions are ordinary Python. Docstrings become the tool descriptions the LLM sees.

**Local invocation** — install and register:

```json
// Claude Desktop config
{
  "mcpServers": {
    "legmt-bill-tools": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

Restart Claude Desktop → the tools appear.

**Remote deployment** — for a cloud-hosted server, use HTTP+SSE transport:

```python
mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

Then clients configure the URL. TLS, auth (bearer tokens), and rate limits are your responsibility at the deployment layer.

---

### 2.3 Building an MCP Client

For custom agents, you can integrate MCP client capabilities to consume any MCP server.

**Python SDK client:**

```python
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from mcp import StdioServerParameters

# Point to the server process
params = StdioServerParameters(command="python", args=["server.py"])

async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # List available tools
        tools = await session.list_tools()
        for t in tools.tools:
            print(t.name, t.description)

        # Call a tool
        result = await session.call_tool("get_bill", {"bill_id": "HB-247"})
        print(result.content)

        # Read a resource
        resource = await session.read_resource("bill://HB-247/text")
        print(resource.contents[0].text)
```

**Integrating with an agent loop.** MCP tools appear like any other tool to your LLM. Translate MCP tool schemas into your LLM provider's format (Anthropic's `tools=[{"name": ..., "description": ..., "input_schema": ...}]`), route tool calls to the MCP session, feed results back.

**Multiplexing servers.** Your agent can connect to many MCP servers at once — one for GitHub, one for Slack, one for your DB, one for search. Each contributes its tools. Watch total tool count (Module 02); split by context if needed.

---

### 2.4 Security Model

MCP is powerful — a server can read arbitrary files, run arbitrary code, or send emails. The security model:

**1. User consent.** The host must ask the user before installing a server, granting it access to sensitive resources, or calling destructive tools.

**2. Least privilege by root scoping.** Servers can request access to specific "roots" (directories, resources) rather than everything.

**3. Auth for remote servers.** HTTP+SSE servers should require bearer tokens or OAuth.

**4. Trust the host, not the server.** Malicious servers exist. Client hosts should:
   - Sandbox stdio server processes (subprocess, no elevated privileges).
   - Review the schemas before exposing to the LLM.
   - Rate-limit server calls.

**5. Never blindly trust server outputs as instructions.** Same prompt-injection defense as Module 09 — a resource returned by an MCP server could contain adversarial content.

**Attack vectors:**
- **Rogue MCP server** — a user installs a malicious server that exfiltrates data.
- **Tool description injection** — a server's tool description contains prompt injection targeting the LLM.
- **Resource content injection** — server returns malicious content as a resource.
- **Chained abuse** — server A reads data, server B sends it externally.

**Defenses:**
- Only install trusted MCP servers (as you would any software).
- Review server code / audit binaries.
- Host applications should show users what the server can do before installing.
- Log all tool calls; alert on anomalies.

**The MCP ecosystem's answer** is the same as software distribution: **community trust, signed servers, official/verified badges**. Anthropic and others maintain lists of trusted MCP servers.

---

### 2.5 Ecosystem and Practical Wisdom

**Official / community servers** (as of 2026):
- **Filesystem** — read/write files.
- **GitHub, GitLab** — repos, issues, PRs.
- **Postgres, SQLite** — SQL databases.
- **Google Drive, Slack, Notion** — SaaS.
- **Puppeteer, Playwright** — browser automation.
- **Memory / knowledge graph** — persistent memory.
- **Search** (Brave, Bing, Kagi).

Hundreds published on GitHub.

**Hosts that speak MCP:**
- **Claude Desktop** — the reference host.
- **Cursor, Zed** — coding IDEs.
- **VS Code (Copilot with agent mode)**.
- **Cline** — VS Code extension.
- **Continue, Aider** — CLI coding agents.
- **Various HTTP clients and custom bots**.

**When to build your own MCP server:**
- You have a proprietary data source or system.
- You want your tools available to any agent/host without integrations per host.
- You want to publish tools as reusable components across your org's projects.

**When to consume an existing one:**
- The tools you need already exist as MCP servers.
- You want plug-and-play integration.

**Design tips:**
- **Keep the tool set small** per server; separate concerns.
- **Clear tool descriptions** — the LLM sees these; write them for the LLM.
- **Idempotency and error handling** — same as any tool (Module 02).
- **Version your server** — schemas will evolve.
- **Log usage** — see how the LLM actually uses your tools.

**When NOT to use MCP:**
- **Simple in-app tools** — if your tool is a single function called from your own agent, wire it directly. MCP overhead isn't needed.
- **Very high-throughput** — MCP's JSON-RPC over stdio/HTTP adds latency. For internal microservice-to-agent calls, direct gRPC may be faster.

---

## 3. Mental Models & Analogies

### 3.1 The "USB-C" Model

Before USB-C: every device had its own connector — micro-USB, mini-USB, Lightning, proprietary. Every peripheral had to make many adapters. Every host had to speak every protocol.

USB-C: one connector for everything. Devices ship with USB-C ports; peripherals with USB-C connectors. Interoperability by default.

MCP does the same for AI tools:
- Before MCP: every LLM app had its own way to expose tools. Every tool provider integrated with each app.
- With MCP: expose your tool once as an MCP server; any MCP-speaking app (Claude Desktop, Cursor, custom agents) uses it.

The "USB-C" framing is Anthropic's own — apt because it captures both the standardization win and the fact that MCP is agnostic to what's on either end.

### 3.2 The "Plugin System" Model

Think of a modern IDE like VS Code. It has:
- **Core** — the editor.
- **Extensions** — installable plugins that add capabilities.
- **Marketplace** — where extensions are discovered.
- **Manifest** — extensions declare their capabilities.
- **Sandbox** — extensions run with defined permissions.

MCP is the plugin system for AI apps:
- **Core** — the host (Claude Desktop, Cursor).
- **Extensions** — MCP servers, each adding tools/resources/prompts.
- **Marketplace** — GitHub, community catalogs.
- **Manifest** — the server's tool/resource list, discovered at initialization.
- **Sandbox** — subprocess isolation, user consent for permissions.

Once you see it as "plugin architecture for LLM apps," the design decisions make sense: capabilities are declared, discovered, and negotiated; security is layered; users decide what to install; hosts don't need to know about specific tools ahead of time.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "MCP Is a New Model or API"

It isn't. MCP is a **protocol** — a specification for how a host and a server exchange tool calls, resource reads, and prompts. Any LLM can be behind the host. Any language / process can serve. It's plumbing, not a model.

### 4.2 "I Need MCP to Give My Agent Tools"

No. Your agent can call tools directly (functions in your codebase). MCP is for making tools **portable across agents/hosts**. If your tool is only used inside your own agent, direct integration is simpler and lower-latency.

### 4.3 "Installing an MCP Server Is Safe"

An MCP server runs code on your machine (or in your infra) with substantial permissions. Read-only servers can still leak data; write-enabled servers can modify your systems. **Treat MCP servers like software installs** — from trusted sources, reviewed, sandboxed when possible.

---

## 5. Self-Assessment Bank (MCP)

### Questions

**Q1 (Short answer).** In one paragraph, describe what MCP is and why it matters.

**Q2 (Multiple choice).** MCP servers can expose:
- (a) Only tools.
- (b) Only resources.
- (c) Tools, resources, and prompt templates.
- (d) Only models.

**Q3 (Short answer).** What is the "USB-C" analogy Anthropic uses for MCP, and why is it apt?

**Q4 (Multiple choice).** The transport used for local Claude Desktop MCP integrations is typically:
- (a) HTTP.
- (b) stdio (server runs as a subprocess).
- (c) WebSocket.
- (d) gRPC.

**Q5 (Short answer).** Describe how a client discovers what a server can do.

**Q6 (Multiple choice).** For a proprietary company database, exposing an MCP server:
- (a) Is impossible.
- (b) Lets any MCP-aware host (Claude Desktop, Cursor, custom agents) query the data uniformly.
- (c) Requires OpenAI compatibility.
- (d) Only works for Postgres.

**Q7 (Short answer).** Name two security concerns with MCP and one mitigation for each.

**Q8 (Multiple choice).** When you shouldn't reach for MCP:
- (a) When you want tools portable across many hosts.
- (b) When the tools are only used inside your own agent and you want minimal latency; direct function calls are simpler.
- (c) When you need to expose files.
- (d) When you're using Claude Desktop.

**Q9 (Short answer).** Give two design tips for writing a good MCP server.

**Q10 (Multiple choice).** MCP servers can expose "resources" identified by URIs like `bill://HB-247/text`. Resources are:
- (a) Same as tools.
- (b) Read-only data the LLM can access on demand.
- (c) Only for image data.
- (d) Deprecated.

---

### Answer Key & Detailed Explanations

**A1.** MCP (Model Context Protocol) is an open protocol from Anthropic (2024) standardizing how LLM applications ("hosts") talk to external tools, data sources, and prompt providers ("servers"). It uses JSON-RPC over stdio/HTTP/WebSocket. It matters because before MCP, every LLM app had to integrate every tool separately; with MCP, a tool exposes itself once and any MCP-aware host can use it — dramatically reducing integration effort and enabling interoperable tool ecosystems.

**A2. (c).** MCP servers expose three primary primitives: **tools** (callable functions), **resources** (read-only data with URIs), and **prompts** (templates the client can invoke). Additional emerging capabilities include sampling and roots.

**A3.** USB-C is a universal connector: before, every device had its own port; after, one connector fits everything. MCP is universal for LLM tool integration: before, every LLM app had a proprietary way to expose tools; after, any MCP server works with any MCP host. Reduces N×M integrations to N+M.

**A4. (b).** stdio (standard input/output) — the server runs as a subprocess of the host; JSON-RPC messages flow over the subprocess's stdin/stdout. Standard for local tools. HTTP+SSE is used for remote servers.

**A5.** During the JSON-RPC `initialize` handshake, the server declares its capabilities (tools, resources, prompts). The client then can call `list_tools`, `list_resources`, `list_prompts` to enumerate specific items with their schemas / URIs / descriptions. Everything a client uses is discovered dynamically — no hard-coded knowledge required.

**A6. (b).** Wrap your database access in a small MCP server (Python SDK's ~50 lines). Any MCP-aware host — Claude Desktop, Cursor, your custom agent — can now query the data. Central place to enforce access control; portable across hosts without per-host integration.

**A7.** Any two: (1) **Rogue server** — installing a malicious server that exfiltrates data. Mitigation: install only trusted servers; review code; sandbox subprocess. (2) **Prompt injection via server content** — server returns a resource with adversarial text. Mitigation: mark tool/resource content as untrusted data (as with any indirect prompt injection); validate high-stakes tool calls against user intent. (3) **Excessive permissions** — server has more access than needed. Mitigation: least-privilege / root scoping; per-tool authorization checks in the host.

**A8. (b).** If your tools are only used inside your own agent (not exposed to other hosts), direct function calls are simpler, lower-latency, and don't add JSON-RPC overhead. MCP is worth it when the tools become **portable / reusable** across multiple agents or hosts.

**A9.** (1) **Clear docstrings** — the LLM sees them as tool descriptions; write for the LLM. (2) **Small tool set per server** — avoid dumping 30 tools into one server; split by concern. (3) **Idempotency + structured errors**. (4) **Versioning** — schemas evolve. (5) **Logging** — observe how the LLM actually uses your tools. Any two plus rationale.

**A10. (b).** Resources in MCP are read-only data endpoints identified by URIs. The LLM can request a resource by URI and receive its contents. Distinct from tools (which take actions); resources are for exposing data (files, DB rows, config) in a discoverable way.

---

## 6. Practice Prompts

1. **Build a server.** Write an MCP server exposing 2 tools related to your work (e.g., "get_legislative_bill", "list_bills_by_sponsor"). Test with Claude Desktop.
2. **Build a client.** Write a Python agent that connects to an existing MCP server (e.g., the community "filesystem" server) and calls its tools in a ReAct loop.
3. **Resources.** Extend your server from prompt 1 with a resource endpoint (e.g., `bill://{id}/text` returns full bill text). Test.
4. **Deploy remote.** Convert your stdio server to HTTP+SSE, deploy to a small VPS with TLS + bearer-token auth.
5. **Multi-server agent.** Build an agent that simultaneously uses two MCP servers (e.g., filesystem + your custom bill server). Handle merged tool listing.

---

## 7. References

- Model Context Protocol official spec: [modelcontextprotocol.io](https://modelcontextprotocol.io/).
- Anthropic, ["Introducing the Model Context Protocol"](https://www.anthropic.com/news/model-context-protocol) (2024).
- MCP Python SDK: [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk).
- Awesome MCP servers list on GitHub — community catalog.
- Cursor, Zed, VS Code Copilot docs on MCP integration.
