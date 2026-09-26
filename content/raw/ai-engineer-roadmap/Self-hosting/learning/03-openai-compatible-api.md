# 03 — The OpenAI-Compatible API

> **Module goal:** Understand why "OpenAI-compatible" has become the universal API shape for LLM serving in 2026, what the endpoints and payloads actually look like, and how to build a thin wrapper that lets you swap between Anthropic, OpenAI, and your self-hosted vLLM with a single config change.

---

## 1. Executive Summary & Core Concepts

**"OpenAI-compatible" API** = HTTP endpoints and request/response payloads matching the shape of OpenAI's Chat Completions, Completions, and Embeddings APIs. Introduced by OpenAI in 2022–2023, this shape has become the *de facto* standard for LLM serving. The reasons:

1. **The `openai-python` SDK is the most-installed LLM client.** If a server speaks its protocol, existing code works with a `base_url` change.
2. **The payload is a reasonable, minimal REST design.** Chat as `messages: [{role, content}]`, streaming via SSE, tool calls, JSON mode.
3. **Fast follower dynamics.** Once a few competitors adopted it (Together, Anyscale, Groq), the compatibility became a market expectation.

**As of 2026**, essentially every LLM inference stack supports the OpenAI API shape:

| Provider | Speaks OpenAI Chat Completions |
|----------|-------------------------------|
| OpenAI | ✓ (original) |
| Anthropic | ✓ (compatibility layer since 2024; native is different) |
| vLLM | ✓ (built-in) |
| TGI | ✓ |
| Ollama | ✓ |
| llama.cpp server | ✓ |
| Together AI, Fireworks, Anyscale, Groq, Perplexity | ✓ |
| Mistral | ✓ |
| Google Gemini | ✓ (compatibility layer; native is different) |
| Azure OpenAI | ✓ (with minor URL/auth differences) |

**Anthropic's native API** has a slightly different shape (system prompt as a top-level field, different tool-use structure, `stop_reason` vs `finish_reason`). Anthropic *also* offers an OpenAI-compatible endpoint for drop-in migration. Real production code typically supports both natively; the OpenAI shape is used as the *portability* layer.

**Practical implication:** you can write your service against `openai-python`, point at any of these providers, and switch by changing `base_url` and `api_key`. This is the biggest single lever for avoiding vendor lock-in.

---

## 2. Deep-Dive Breakdown

### 2.1 The Chat Completions Endpoint

The main event. Request:

```
POST /v1/chat/completions
Authorization: Bearer <api-key>
Content-Type: application/json

{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "temperature": 0.2,
  "max_tokens": 100,
  "stream": false,
  "tools": [...],
  "response_format": {"type": "json_object"},
  "user": "user-123"
}
```

Response:

```json
{
  "id": "chatcmpl-xxxx",
  "object": "chat.completion",
  "created": 1727280000,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "4"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 1,
    "total_tokens": 25
  }
}
```

Key fields the compatibility layer must respect:
- `messages` — the conversation, in order
- `model` — server-side model identifier
- `temperature`, `top_p`, `max_tokens`, `stop`, `n`, `seed` — sampling parameters
- `stream` — SSE mode (see 2.3)
- `tools` — function-calling schemas (see 2.4)
- `response_format` — structured output (`json_object` or `json_schema`)
- `usage` — prompt/completion token counts (many clients depend on this for cost accounting)

Fields Anthropic-native uses differently (worth memorizing for API translation):
- Anthropic native: `system` is a top-level field. OpenAI: it's a `role: system` message.
- Anthropic: `stop_reason` (e.g., `"end_turn"`, `"max_tokens"`, `"tool_use"`). OpenAI: `finish_reason` (`"stop"`, `"length"`, `"tool_calls"`).
- Anthropic: `content` can be a list of blocks (`[{"type":"text",...},{"type":"tool_use",...}]`). OpenAI: string content + separate `tool_calls` field.

### 2.2 Using It From `openai-python`

The killer feature is that every provider "just works" with the OpenAI SDK by setting `base_url` and `api_key`:

```python
from openai import OpenAI

# Against vLLM
vllm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-vllm-api-key",
)

# Against OpenAI
openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

# Against Together AI
together_client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
)

# Against Anthropic's OpenAI-compatible endpoint
anthropic_openai_shim = OpenAI(
    base_url="https://api.anthropic.com/v1",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

# All four clients accept the same call:
resp = vllm_client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hi"}],
)
```

This is what "portable across providers" means concretely. Your service takes a `PROVIDER` config value; loads the right client; identical business logic follows.

### 2.3 Streaming (SSE)

For streaming responses, the server returns Server-Sent Events (SSE) — a sequence of `data:` chunks over a persistent HTTP connection:

```
data: {"choices":[{"delta":{"role":"assistant","content":""},"index":0}]}

data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"choices":[{"delta":{"content":" world"},"index":0}]}

data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

Client-side, the `openai-python` SDK handles this transparently:

```python
stream = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[...],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Why SSE over WebSockets?** Simpler protocol (unidirectional server → client); works through HTTP proxies and firewalls; doesn't require special client libraries; matches the "one response chunked" mental model. WebSockets are overkill for the streaming-generation use case.

Any real service exposes a streaming endpoint — first-token latency (TTFT) is what users perceive as "the bot is responding."

### 2.4 Tool Calls (Function Calling)

The tool-calling API standard in OpenAI's shape:

Request specifies tools:
```json
{
  "model": "...",
  "messages": [{"role": "user", "content": "What's the weather in Helena?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

Response contains tool_calls:
```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Helena\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Client runs the tool, then continues the conversation with a `tool` message referencing the call ID:
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc123", ...}]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "72°F, sunny"}
  ]
}
```

vLLM supports tool calling for Llama 3.1+, Qwen 2.5, Mistral, and other tool-tuned open models via `--enable-auto-tool-choice --tool-call-parser llama3_json` (or the parser matching the model family). The quality of the tool-call parsing depends on the underlying model's training.

### 2.5 Structured Output

Two flavors:

**Simple JSON mode:**
```json
{
  "response_format": {"type": "json_object"}
}
```
Server tries to constrain the output to valid JSON. Not schema-enforced.

**JSON schema (OpenAI's structured output feature since 2024):**
```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "Person",
      "schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"]
      }
    }
  }
}
```
Server constrains generation to satisfy the schema (100% enforcement via constrained decoding). vLLM implements this via the `--guided-decoding-backend outlines` or `xgrammar` integration.

### 2.6 Building a Provider-Router Layer

The single most useful abstraction for a production service using multiple LLM providers:

```python
# provider_router.py
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import os

class LLMProvider:
    """Unified interface across providers, speaking OpenAI shape internally."""

    def __init__(self, provider: str):
        self.provider = provider
        if provider == "openai":
            self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.default_model = "gpt-4o-mini"
        elif provider == "anthropic":
            # Native Anthropic; adapter converts to OpenAI shape internally
            self.native = AsyncAnthropic()
            self.default_model = "claude-sonnet-5"
        elif provider == "vllm-selfhost":
            self.client = AsyncOpenAI(
                base_url=os.environ["VLLM_BASE_URL"],
                api_key=os.environ.get("VLLM_API_KEY", "none"),
            )
            self.default_model = os.environ.get("VLLM_MODEL", "llama-3.1-8b")
        elif provider == "together":
            self.client = AsyncOpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=os.environ["TOGETHER_API_KEY"],
            )
            self.default_model = "meta-llama/Llama-3.1-8B-Instruct-Turbo"
        else:
            raise ValueError(f"unknown provider {provider}")

    async def chat(self, messages, *, model=None, temperature=0.2, max_tokens=1024, stream=False):
        model = model or self.default_model
        if self.provider == "anthropic":
            return await self._anthropic_chat(messages, model, temperature, max_tokens, stream)
        # OpenAI-compatible path
        return await self.client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, stream=stream,
        )

    async def _anthropic_chat(self, messages, model, temperature, max_tokens, stream):
        """Translate OpenAI-shape to Anthropic-shape and back."""
        # Extract system message (Anthropic has it as a top-level field)
        system = None
        conversation = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conversation.append(m)
        if stream:
            # Return an async iterator that yields OpenAI-shape chunks
            return self._anthropic_stream_shim(system, conversation, model, temperature, max_tokens)
        resp = await self.native.messages.create(
            model=model, system=system, messages=conversation,
            temperature=temperature, max_tokens=max_tokens,
        )
        # Translate to OpenAI-shape response object
        return _to_openai_response(resp)

    async def _anthropic_stream_shim(self, system, conversation, model, temperature, max_tokens):
        async with self.native.messages.stream(
            model=model, system=system, messages=conversation,
            temperature=temperature, max_tokens=max_tokens,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    yield _to_openai_chunk(event.delta.text)
```

The pattern: use OpenAI shape as the internal lingua franca; adapters translate at the boundaries. Now your business logic doesn't care whether the model is Anthropic-native, vLLM-selfhosted, or a Together-hosted Llama.

Application config:
```yaml
# In prod
llm_provider: "anthropic"
llm_model: "claude-sonnet-5"

# For cost testing
llm_provider: "vllm-selfhost"
llm_model: "llama-3.1-8b"

# For a spike test
llm_provider: "together"
llm_model: "meta-llama/Llama-3.1-70B-Instruct-Turbo"
```

Config change → new provider. No code change. **This is the point of OpenAI compatibility.**

`[IMG-SH03-01]` — *Prompt: Architecture diagram of a provider-router pattern. Center: an application service labeled "Business logic" with an arrow to a "LLMProvider" abstraction layer. From the abstraction, four outbound arrows to different providers: OpenAI (fp16 icon), Anthropic (with a translation-adapter box shown mid-arrow), self-hosted vLLM (a GPU icon), Together AI (cloud icon). All four accept the same OpenAI-shape request format. Config file on the side showing `llm_provider: "..."` as the single switch. Clean architecture-diagram style, muted palette, arrows labeled with "OpenAI-compatible chat/completions" everywhere.*

---

## 3. Mental Models & Analogies

### Model 1: SQL, But for LLM Inference

SQL is a standardized query language that every relational DB implements (with dialects). You write your query once, and it runs against Postgres, MySQL, SQLite, or SQL Server. You may adjust dialect-specific syntax at the edges, but the core is the same.

The OpenAI Chat Completions API has become that for LLM inference. Same request shape, same response shape, works everywhere. Providers implement dialects (Anthropic's tool-use is slightly different; Google's function args slightly different) but the compatibility layer is broad enough that most application code is portable.

This is a **strategic gift.** Providers who fought against it (early Anthropic, early Google) eventually offered compatibility layers because the market demanded portability. Any provider that doesn't will lose share to those that do.

### Model 2: The Universal Charger

Once upon a time every phone had its own charging port; now everything's USB-C (mostly). The USB-C shape isn't the "best" possible charger — it's the one everyone agreed to. Once agreed, the ecosystem benefits enormously: hotels stock one cable, airlines put USB-C in every seat, adapters proliferate.

OpenAI Chat Completions is the USB-C of LLM APIs. It isn't perfect — Anthropic's native structure is arguably cleaner (system as top-level, content blocks); Gemini has features not expressible in the OpenAI shape (grounding). But adoption wins.

**Your service should speak OpenAI-compatible as its primary shape, and use provider-native APIs only for features the compatibility layer doesn't reach yet.**

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Assuming perfect drop-in compatibility.**
"Just change the base_url" is 95% true. The 5% that isn't:
- Some providers don't support all OpenAI parameters (`logit_bias`, `frequency_penalty`, `presence_penalty`, `logprobs` — coverage varies)
- Tool-calling parsing quality varies significantly — Llama's `llama3_json` parser is less reliable than GPT-4's built-in
- Usage/token accounting fields sometimes missing or different (Anthropic's cache_creation and cache_read; some providers report only totals)
- Rate-limit response shapes differ (429 body varies)

Always test the specific features you use against the specific provider you use. Don't assume.

**Pitfall #2 — Streaming edge cases.**
When translating SSE across providers, a lot goes wrong:
- Different providers may emit `[DONE]` differently, or not at all
- Some intermediate keepalive lines with `event: ping` need filtering
- Anthropic native streaming includes non-content events (message_start, message_delta) that don't map to OpenAI chunks
- Different providers handle mid-stream errors differently

Test your streaming path with all providers you support. Streaming bugs are user-visible (broken UI) — the highest priority to catch.

**Pitfall #3 — Trusting `usage` for cost accounting universally.**
Some providers under-report or omit token counts. Some report input_tokens *including* cache_read tokens; some report only new tokens. Cache accounting differs. Trusting `usage` field naively can produce cost dashboards that miss 20-40% of real cost.

Fix: verify your cost accounting against the provider's billing dashboard for a period. If they diverge, patch the accounting to match reality. Don't blindly trust the response's `usage` — verify.

---

## 5. Self-Assessment Bank

**Q1 (MC):** The primary value of OpenAI-compatible APIs is:
A) Better model quality
B) Portability across providers via a shared request/response shape
C) Lower cost
D) Faster inference

**Q2 (short):** What are the three main differences between Anthropic's native API shape and the OpenAI Chat Completions shape?

**Q3 (MC):** For a self-hosted vLLM at `http://vllm.internal:8000`, the correct `openai-python` client construction is:
A) `OpenAI(api_key="...")` with no other config
B) `OpenAI(base_url="http://vllm.internal:8000/v1", api_key="...")`
C) `OpenAI(host="vllm.internal", port=8000)`
D) `OpenAI(provider="vllm")`

**Q4 (short):** Sketch how you'd support both Anthropic-native and OpenAI-compatible providers behind a single application interface.

**Q5 (MC):** In OpenAI-compatible streaming, the "done" signal is:
A) HTTP 200 status
B) A `data: [DONE]\n\n` line
C) The socket closes
D) A `finish_reason: complete` message

**Q6 (short):** Give a concrete example of a scenario where "just change the base_url" from Anthropic to a self-hosted Llama would break something.

**Q7 (MC):** For JSON schema-enforced structured output on a self-hosted Llama via vLLM, which flag activates it?
A) `--response-format json`
B) `--guided-decoding-backend outlines` (or xgrammar)
C) `--json-schema-enforce`
D) Not supported

**Q8 (short):** Why is streaming (SSE) preferred over polling for LLM responses?

**Q9 (MC):** OpenAI-compatible tool calls use which finish_reason to signal that the model wants to call a tool?
A) `"stop"`
B) `"length"`
C) `"tool_calls"`
D) `"function_call"` (deprecated form)

**Q10 (short):** Your service is Anthropic-native today. A senior engineer suggests adopting OpenAI-compatible shape internally so you can add a self-hosted-vLLM route for cost sensitivity. Write a 3-sentence pro and a 3-sentence con for the discussion.

---

### Answer Key

**A1: B.** OpenAI-compatible is a portability standard — the same request/response shape means the same client code works across dozens of providers (managed, self-hosted, on-prem). Not a quality, cost, or speed win in itself; portability is the value.

**A2:** Three differences worth memorizing:
1. **System prompt location** — Anthropic: top-level `system` field. OpenAI: `role: system` message inside `messages`.
2. **Response content structure** — Anthropic: `content` is a list of blocks (`text`, `tool_use`, `image`). OpenAI: string `content` + separate `tool_calls` field.
3. **Stop reason naming** — Anthropic: `stop_reason` (`"end_turn"`, `"max_tokens"`, `"tool_use"`, `"stop_sequence"`). OpenAI: `finish_reason` (`"stop"`, `"length"`, `"tool_calls"`).

Others: streaming event names differ; tool-use blocks in Anthropic native use nested structure; Anthropic exposes cache_read_input_tokens / cache_creation_input_tokens separately.

**A3: B.** `OpenAI(base_url="http://vllm.internal:8000/v1", api_key="...")`. The `/v1` suffix is important because vLLM (and OpenAI) mount their endpoints at `/v1/chat/completions` etc. The `api_key` can be any string (or the value you set with `--api-key` when launching vLLM); it's required by the SDK even if the server doesn't check it.

**A4:** Provider-router pattern: define an internal `LLMProvider` class that accepts OpenAI-shape method signatures (`chat(messages, model, temperature, max_tokens, stream)`). For providers speaking OpenAI (self-hosted vLLM, Together, OpenAI itself, Anthropic's OpenAI-compatible endpoint), use `openai-python` with the right `base_url`. For Anthropic-native, wrap the Anthropic SDK inside adapter methods that translate: extract system message; map roles; translate the response's content-blocks and stop_reason into OpenAI shape. Configure the choice per-service via env var or config file. Your business logic imports `LLMProvider` and never touches native SDKs. When you need a native-only feature (Anthropic prompt caching), add a method that gracefully no-ops on non-Anthropic providers.

**A5: B.** OpenAI-compatible streaming ends with a literal `data: [DONE]\n\n` line. The SDK detects this and stops iteration. HTTP 200 is the initial response status; sockets don't close mid-stream in normal operation; there is no `"complete"` finish_reason value.

**A6:** Many possible answers; sample:
- **Tool calling reliability**: Anthropic's tool-use is highly reliable and returns structured `tool_use` blocks. Self-hosted Llama with vLLM's tool parser sometimes produces malformed JSON in tool arguments, or misses tool calls entirely on ambiguous prompts. Your service parses tool calls; the Llama version fails silently or crashes.
- **Prompt caching**: Anthropic caches 4-8k system prompts at ~10% cost; vLLM has prefix caching that caches per-request but not identically. Cost model changes.
- **Long-context quality**: Anthropic Sonnet is strong at 100k+ contexts; a self-hosted 8B model degrades significantly beyond 8-16k. If your service passes long context, quality collapses.
- **Structured output**: OpenAI/Anthropic have well-tuned JSON schema enforcement; open models via vLLM's constrained decoding are usually fine but sometimes reject valid outputs or slow generation noticeably.
- **`logprobs` field**: some services depend on it; not all providers expose it.

The failure is silent — the API contract is the same but the *quality* of what fills that contract varies.

**A7: B.** `--guided-decoding-backend outlines` (or `xgrammar` in newer vLLM) enables JSON-schema-enforced generation. Clients then pass `response_format={"type": "json_schema", "json_schema": {...}}` and the server constrains generation to match. The constraint is enforced at token generation time via a state machine that disallows tokens that would violate the schema — 100% valid JSON output guaranteed.

**A8:** Streaming (SSE) is preferred over polling because:
1. **Perceived latency (TTFT).** Users see the first token in ~500ms even if full generation takes 10s. Polling can't beat this — polls are point-in-time snapshots.
2. **Efficiency.** One persistent HTTP connection vs many short polls (each adds TCP setup + response overhead + auth checks).
3. **Simplicity of protocol.** SSE is unidirectional server → client, works through proxies, is text-based, and every major HTTP framework has support.
4. **Natural fit for the generation model** — LLMs generate token-by-token; streaming lets that reach the user in near real time rather than being buffered.

Polling makes sense for long-running batch jobs (LLM batch API) where results take minutes and streaming would tie up connections uselessly.

**A9: C.** `finish_reason: "tool_calls"` (plural). Older OpenAI API used `finish_reason: "function_call"` (D) — deprecated as of the transition from `functions` to `tools`. Some providers still return the deprecated form; robust code handles both.

**A10:** **Pro:** Cost sensitivity is a real strategic lever. If our workload has portions well-suited to a smaller open model (classification, extraction, summarization of well-defined domains), self-hosting an 8B Llama can be ~5-15× cheaper per token than Sonnet with comparable quality on those tasks. Building the abstraction now costs weeks; it pays back the first time we route a portion of traffic and cuts $10k+/month off the bill. Also: it's a hedge — provider outages, price changes, or geopolitical restrictions on API access don't stop our service.

**Con:** Adopting OpenAI-compatible internally means writing (and maintaining) an Anthropic-native → OpenAI-shape adapter, which is real code with real bugs — streaming translation especially. We give up direct access to Anthropic-native features (prompt caching, cache_creation_input_tokens for cost accounting, native tool-use blocks) until we adapter-map each of them, which will slip on our roadmap. And we take on operational responsibility: running vLLM on GPU infrastructure is a new expertise, on-call, GPU capacity planning — non-trivial for a team not already there.

Balance: **worth doing if we have the traffic to justify the operational overhead, and time to build the adapter carefully.** Premature at low volumes — pay the Anthropic bill, focus on product.

---

**Related modules:**
- `learning/01-vllm-serving.md` — server-side that speaks this API
- `learning/02-quantization.md` — deployment considerations
- `builds/01-self-host-llama-api.md` — capstone builds all three
- `../capstones/04-work-with-llm-apis.md` — the client-side companion

**Practice prompts:**
1. Take your existing Anthropic-native code from Capstone 04 (`llmcore`). Add an OpenAI-compatible provider option. Switch by config; ensure behavior is identical.
2. Deploy vLLM with `--api-key` set. Verify the openai-python SDK works with the right base_url.
3. Compare tool-calling quality on 20 test prompts between (a) Anthropic Sonnet, (b) OpenAI GPT-4o-mini, (c) self-hosted Llama 3.1 8B via vLLM. Report which fails what.

**References:**
- OpenAI Chat Completions API reference — https://platform.openai.com/docs/api-reference/chat
- vLLM OpenAI Compatible Server docs — https://docs.vllm.ai
- Anthropic OpenAI-compatible endpoint docs
- Server-Sent Events spec — https://html.spec.whatwg.org/multipage/server-sent-events.html
- LiteLLM library (a production-ready provider router) — https://docs.litellm.ai
