# Capstone Project 04 — Work with LLM APIs (Production Client)

> **Deliverable:** A robust, reusable Python client wrapper around the Anthropic API (portable to OpenAI/others) that handles streaming, retries with backoff, structured output validation, function/tool calling, cost accounting, and prompt caching — packaged as a small library you can drop into any project.
>
> **Time:** 4–6 hours.
>
> **What you'll be able to say:** "I built a production-grade LLM client wrapper. It streams first-token responses, retries with jitter, validates structured JSON with Pydantic, dispatches tool calls, tracks per-request cost, and uses Anthropic's prompt cache to cut token costs by ~85% on repeated system prompts."

---

## 1. Project Overview

An LLM client that survives production has to do a lot more than "call the API with a prompt." This project builds a wrapper library `llmcore` that any FastAPI or agent code can import to get:

1. **Async streaming** — first-token latency matters for UX
2. **Automatic retry with jitter** — handle 429/5xx politely
3. **Structured outputs** — return validated Pydantic models, not raw strings
4. **Tool calling** — a clean dispatcher for function tools
5. **Cost tracking** — per-request USD, cumulative counters
6. **Prompt caching** — Anthropic `cache_control` breakpoints
7. **Observability** — trace attributes with `gen_ai.*` semantic conventions
8. **Configurable model routing** — Haiku for cheap, Sonnet for balanced, Opus for hard

### Architecture
![IMG-CAP04-01](/7%20-%20Projects/images/IMG-CAP04-01.jpg)


### Repo scaffold

```
llmcore/
├── pyproject.toml
├── llmcore/
│   ├── __init__.py
│   ├── client.py       # main LLMClient class
│   ├── retry.py        # backoff + jitter
│   ├── streaming.py    # SSE handling
│   ├── structured.py   # Pydantic JSON validation
│   ├── tools.py        # tool dispatcher
│   ├── cost.py         # price table + metrics
│   ├── cache.py        # prompt cache helpers
│   └── tracing.py      # OTel wrappers
├── tests/
│   ├── test_retry.py
│   ├── test_structured.py
│   └── test_tools.py
└── examples/
    ├── streaming_chat.py
    ├── extract_person.py
    └── weather_agent.py
```

---

## 2. Step-by-Step Implementation

### Step 1 — Core client (`llmcore/client.py`)

```python
"""client.py — the main entry point."""
from __future__ import annotations
import os
from typing import AsyncIterator
from anthropic import AsyncAnthropic
from .cost import compute_cost, record_cost
from .retry import with_backoff
from .tracing import span_llm_call

DEFAULT_MODEL = "claude-sonnet-5"

class LLMClient:
    def __init__(self, api_key: str | None = None, default_model: str = DEFAULT_MODEL):
        self._client = AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.default_model = default_model

    @with_backoff(max_retries=5)
    async def complete(
        self,
        messages: list[dict],
        *,
        system: str | list | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        cache_control: bool = False,
    ) -> dict:
        model = model or self.default_model
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = _wrap_system(system, cache_control)
        if tools:
            kwargs["tools"] = tools

        with span_llm_call(model=model) as span:
            resp = await self._client.messages.create(**kwargs)
            cost_usd = compute_cost(model, resp.usage)
            record_cost(model=model, cost_usd=cost_usd)
            span.set_attributes({
                "gen_ai.usage.input_tokens": resp.usage.input_tokens,
                "gen_ai.usage.output_tokens": resp.usage.output_tokens,
                "gen_ai.usage.cache_read_input_tokens":
                    getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                "cost_usd": cost_usd,
                "gen_ai.response.stop_reason": resp.stop_reason,
            })
        return {
            "content": resp.content,
            "stop_reason": resp.stop_reason,
            "usage": resp.usage,
            "cost_usd": cost_usd,
        }

def _wrap_system(system, cache_control: bool):
    """Wrap system prompt into cache-controlled blocks if requested."""
    if isinstance(system, str):
        if cache_control:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system
    return system  # already a list of blocks
```

**Why an async client?** LLM calls take seconds; blocking the event loop would kill your throughput. Async lets a single Python process juggle 100+ concurrent LLM calls.

**Why wrap system prompt with `cache_control`?** Anthropic caches the prompt prefix. Cached input tokens cost ~10% of normal — huge savings when a big system prompt is reused.

### Step 2 — Retry with jitter (`llmcore/retry.py`)

```python
"""retry.py — exponential backoff with jitter."""
import asyncio, random, functools, logging
from anthropic import APIError, APIStatusError, RateLimitError

log = logging.getLogger("llmcore.retry")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

def with_backoff(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return await fn(*args, **kwargs)
                except RateLimitError as e:
                    retry_after = _parse_retry_after(e)
                    if attempt >= max_retries:
                        raise
                    delay = min(max_delay, retry_after or base_delay * (2 ** attempt))
                    delay = random.uniform(0, delay)  # full jitter
                    log.warning("rate_limited attempt=%d sleep=%.2fs", attempt, delay)
                    await asyncio.sleep(delay)
                except APIStatusError as e:
                    if e.status_code not in RETRYABLE_STATUS or attempt >= max_retries:
                        raise
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    delay = random.uniform(0, delay)
                    log.warning("api_error status=%d attempt=%d sleep=%.2fs",
                                e.status_code, attempt, delay)
                    await asyncio.sleep(delay)
                attempt += 1
        return wrapper
    return deco

def _parse_retry_after(err):
    try:
        headers = err.response.headers
        val = headers.get("retry-after") or headers.get("anthropic-ratelimit-tokens-reset")
        return float(val) if val else None
    except Exception:
        return None
```

**Why "full jitter"?** If 100 clients hit 429 at once and back off deterministically (say, 2s), they all retry at t+2s — the thundering herd is preserved. Full jitter (random between 0 and the cap) breaks the herd.

**Why cap max_delay?** Beyond ~60s, the request is likely dead-in-water; better to fail fast and let the caller decide (queue for later, return partial, error to user) than to block a worker.

### Step 3 — Streaming (`llmcore/streaming.py`)

```python
"""streaming.py — async iterator over content chunks."""
from typing import AsyncIterator
from anthropic import AsyncAnthropic

async def stream_complete(
    client: AsyncAnthropic,
    messages: list[dict],
    *,
    system: str | None = None,
    model: str = "claude-sonnet-5",
    max_tokens: int = 1024,
) -> AsyncIterator[str]:
    """Yields text chunks as the model streams."""
    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
    ) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
        # Emit metadata as a sentinel dict (caller can filter strings)
        yield {"__final__": True, "usage": final.usage.model_dump()}
```

Usage in FastAPI:

```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def gen():
        async for chunk in stream_complete(client, [{"role":"user","content":req.q}]):
            if isinstance(chunk, str):
                yield f"data: {chunk}\n\n"
            else:
                yield f"event: done\ndata: {json.dumps(chunk)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

**Why stream?** For a 1500-token response at typical output speed, first-token latency is ~500ms and full response is ~10s. Streaming means the user sees text at 500ms; non-streaming, they wait 10s. UX-critical for chat.

### Step 4 — Structured output (`llmcore/structured.py`)

```python
"""structured.py — validated Pydantic responses."""
import json
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError
from .client import LLMClient

T = TypeVar("T", bound=BaseModel)

async def extract(
    llm: LLMClient,
    schema: Type[T],
    text: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_retries: int = 2,
) -> T:
    """Ask the LLM to extract data into `schema`, retrying on validation errors."""
    system_prompt = (system or "") + f"""
You extract information into JSON matching this schema:
{json.dumps(schema.model_json_schema(), indent=2)}

Respond with ONLY valid JSON — no prose, no markdown fences.
""".strip()

    messages = [{"role": "user", "content": text}]
    for attempt in range(max_retries + 1):
        resp = await llm.complete(
            messages=messages,
            system=system_prompt,
            model=model,
            temperature=0,  # deterministic for extraction
        )
        raw = resp["content"][0].text
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries:
                raise
            # Feed the error back to the model
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Validation failed: {e}. Respond with corrected JSON only.",
            })
    raise RuntimeError("unreachable")
```

Usage:
```python
class Person(BaseModel):
    name: str
    age: int | None = None
    occupation: str | None = None

person = await extract(llm, Person, "Alice is 34 and works as a nurse.")
# Person(name='Alice', age=34, occupation='nurse')
```

**Why validate?** LLMs occasionally emit almost-JSON (a trailing comma, `null` where a string is expected, an extra explanation line). Validation forces the model to comply; the retry loop closes the loop by feeding the error back.

**Why temperature=0?** Extraction is deterministic; sampling variance is noise here.

### Step 5 — Tool calling (`llmcore/tools.py`)

```python
"""tools.py — clean tool dispatch."""
from typing import Callable, Any
from pydantic import BaseModel

class Tool:
    def __init__(self, name: str, description: str, input_schema: dict, fn: Callable):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.fn = fn

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    def to_anthropic(self) -> list[dict]:
        return [t.to_anthropic() for t in self._tools.values()]
    async def dispatch(self, name: str, inputs: dict) -> Any:
        tool = self._tools[name]
        result = tool.fn(**inputs)
        if hasattr(result, "__await__"):
            result = await result
        return result

async def run_tool_loop(llm, messages, registry: ToolRegistry, max_iters: int = 10):
    """Standard agent loop: model calls tool, we run it, feed back, repeat."""
    for _ in range(max_iters):
        resp = await llm.complete(
            messages=messages,
            tools=registry.to_anthropic(),
        )
        blocks = resp["content"]
        if resp["stop_reason"] != "tool_use":
            return blocks  # done — return final assistant message
        # Append the assistant turn (with tool_use blocks)
        messages.append({"role": "assistant", "content": blocks})
        # Run each tool call, append tool_result blocks
        results = []
        for b in blocks:
            if getattr(b, "type", None) == "tool_use":
                out = await registry.dispatch(b.name, b.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": str(out),
                })
        messages.append({"role": "user", "content": results})
    raise RuntimeError("tool loop exceeded max_iters")
```

Usage:
```python
registry = ToolRegistry()
registry.register(Tool(
    name="get_weather",
    description="Get current weather for a city",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    fn=lambda city: f"Weather in {city}: 72°F, sunny",
))

msgs = [{"role": "user", "content": "What's the weather in Helena?"}]
result = await run_tool_loop(llm, msgs, registry)
```

**Why a registry pattern?** Decouples tool declaration from the call site. The same registry is used for the model's schema and for dispatch. Add auth, per-tool rate limiting, and audit logging in one place.

### Step 6 — Cost accounting (`llmcore/cost.py`)

```python
"""cost.py — price table + counters."""
from prometheus_client import Counter

PRICE_TABLE = {
    # Prices per 1M tokens; check Anthropic's current pricing page
    "claude-opus-5":    {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    "claude-sonnet-5":  {"input": 3.0,  "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 0.8,  "output": 4.0,  "cache_read": 0.08, "cache_write": 1.0},
}

COST_TOTAL = Counter("llm_cost_usd_total", "Cumulative LLM cost", ["model"])
TOKENS_IN  = Counter("llm_tokens_input_total", "Input tokens", ["model"])
TOKENS_OUT = Counter("llm_tokens_output_total", "Output tokens", ["model"])

def compute_cost(model: str, usage) -> float:
    p = PRICE_TABLE.get(model)
    if p is None:
        return 0.0
    in_tok = usage.input_tokens
    out_tok = usage.output_tokens
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    # "input_tokens" from the API already excludes cache_read/cache_write in Anthropic's response
    return (
        in_tok * p["input"] +
        out_tok * p["output"] +
        cache_read * p["cache_read"] +
        cache_write * p["cache_write"]
    ) / 1_000_000

def record_cost(model: str, cost_usd: float):
    COST_TOTAL.labels(model=model).inc(cost_usd)
```

**Why per-request cost?** So you can group by model, route, user, prompt version — the dimensions that matter for FinOps (Module 14 in Month 6).

### Step 7 — Prompt caching helpers (`llmcore/cache.py`)

```python
"""cache.py — cache_control block helpers."""

def cached_system(text: str) -> list[dict]:
    """Wrap a system string as a single ephemeral-cached block."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

def cached_system_with_tools(system: str, tools_text: str) -> list[dict]:
    """Two cache breakpoints: the main system prompt, then tool definitions."""
    return [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": tools_text, "cache_control": {"type": "ephemeral"}},
    ]
```

Usage: pass `cache_control=True` to `LLMClient.complete()` when your system prompt is stable and > 1024 tokens. Measure `cache_read_input_tokens / total_input_tokens` in traces — you want that ratio > 80% for hot-path prompts.

### Step 8 — Tracing (`llmcore/tracing.py`)

```python
"""tracing.py — OpenTelemetry semantic conventions for gen-ai."""
from contextlib import contextmanager
from opentelemetry import trace

_tracer = trace.get_tracer("llmcore")

@contextmanager
def span_llm_call(model: str, system: str = "anthropic"):
    with _tracer.start_as_current_span("gen_ai.chat") as span:
        span.set_attribute("gen_ai.system", system)
        span.set_attribute("gen_ai.request.model", model)
        yield span
```

Config your exporter (LangFuse, LangSmith, Phoenix, Datadog) once at app startup; every LLM call carries semantic-convention attributes automatically.

### Step 9 — Tests

```python
# tests/test_retry.py
import pytest
from unittest.mock import AsyncMock
from anthropic import RateLimitError, APIStatusError
from llmcore.retry import with_backoff

class MockResp:
    def __init__(self, status):
        self.status_code = status
        self.headers = {"retry-after": "0.01"}

@pytest.mark.asyncio
async def test_rate_limit_retries():
    call_count = 0
    @with_backoff(max_retries=3, base_delay=0.01)
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError("rl", response=MockResp(429), body={})
        return "ok"
    result = await flaky()
    assert result == "ok"
    assert call_count == 3

@pytest.mark.asyncio
async def test_non_retryable_raises():
    @with_backoff(max_retries=3, base_delay=0.01)
    async def bad():
        raise APIStatusError("400", response=MockResp(400), body={})
    with pytest.raises(APIStatusError):
        await bad()
```

---

## 3. Example — Putting it all together

```python
# examples/extract_person.py
import asyncio
from pydantic import BaseModel
from llmcore.client import LLMClient
from llmcore.structured import extract

class Person(BaseModel):
    name: str
    age: int | None = None
    occupation: str | None = None
    location: str | None = None

async def main():
    llm = LLMClient()
    p = await extract(
        llm, Person,
        "Meet Priya Shah, a 29-year-old data scientist based in Helena, Montana."
    )
    print(p)
    # Person(name='Priya Shah', age=29, occupation='data scientist', location='Helena, Montana')

asyncio.run(main())
```

---

## 4. Why Each Piece Matters

| Piece | Failure mode without it |
|-------|-------------------------|
| Async | One request blocks the whole event loop; throughput collapses |
| Retry with jitter | 429s cascade into user-visible errors; thundering herds on retry |
| Streaming | Users stare at a blank screen for 10s waiting for response |
| Structured output | Downstream parsing crashes on almost-JSON; fragile chain |
| Tool dispatch | Coupled tool implementations; no auditing; no shared auth |
| Cost accounting | End-of-month bill surprise; no per-user attribution |
| Prompt caching | Paying 10× for the same system prompt on every request |
| Tracing | Bug reports come in with "the bot said X"; you can't reproduce |

---

## 5. Extensions

- **Fallback across providers.** If Anthropic 500s, retry against OpenAI with the same interface.
- **Semantic cache layer.** In-memory or Redis; skip the LLM for near-duplicate queries.
- **Batch API adapter.** For non-realtime workloads, submit through Anthropic's message batches at 50% cost.
- **Rate-limit budget.** A local token-bucket that caps outbound RPM/TPM so you never trigger provider 429s.
- **Model router.** Cheap classifier that picks Haiku/Sonnet/Opus per request based on complexity.

`[IMG-CAP04-02]` — *Prompt: A code walkthrough diagram — a UML-style sequence with time flowing downward. Actors: Client (FastAPI handler), LLMCore, Anthropic API, Prometheus. Steps: 1) FastAPI → LLMCore.complete(); 2) LLMCore starts span; 3) LLMCore → Anthropic streaming call; 4) Anthropic returns first-token chunk (highlighted "500ms first token"); 5) chunks continue until final; 6) LLMCore computes cost from usage; 7) LLMCore records to Prometheus and closes span with cost + tokens attributes; 8) LLMCore returns to FastAPI. Show a parallel "on 429" branch with backoff-and-retry loop. Clean sequence-diagram style with UML notation.*

---

## 6. Interview Talking Points

- **"How would you make an LLM API call production-safe?"** Walk through retry/backoff, streaming for UX, structured validation, cost tracking, tracing, caching. This project *is* the answer.
- **"How do you handle 429s?"** Full-jitter exponential backoff, capped at 60s, honor `Retry-After`; on cap, propagate 429 to your caller with your own `Retry-After` header.
- **"Why validate LLM output?"** LLMs occasionally emit malformed JSON, extra prose, or type-invalid fields. Pydantic + retry with the error fed back is the standard pattern.
- **"How do you keep LLM costs under control?"** Prompt caching for stable prefixes, model routing, per-request cost dashboards, budget alerts, per-user caps.
- **"How is tool-calling implemented under the hood?"** The model returns a `tool_use` block; your code runs the tool and returns a `tool_result` block; loop until `stop_reason != "tool_use"`. Show the code.

---

## 7. References

- Anthropic API docs — https://docs.anthropic.com
- OpenTelemetry GenAI semantic conventions
- Instructor library (structured output pattern in Python) — https://python.useinstructor.com/
- Simon Willison's `llm` CLI — clean example of a portable LLM client design
