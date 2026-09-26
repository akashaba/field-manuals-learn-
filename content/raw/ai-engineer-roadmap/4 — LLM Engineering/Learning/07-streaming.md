# Streaming — Master Study Guide

> **Track:** LLM Engineering · **Module:** 07
> **Prerequisites:** Modules 01–06.
> **Time budget:** ~5–7 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Streaming — receiving the model's response **token by token as it generates** — is what makes ChatGPT feel responsive rather than glacial. It transforms the perceived latency from "10-second wait" to "instant, watch it type." Any user-facing LLM app that doesn't stream feels broken.

Beyond UX, streaming also:

- Enables **early termination** — stop generation when you have enough.
- Supports **UI/API composition** — feed streamed chunks into other systems.
- Reduces **time-to-first-token (TTFT)** as your primary latency metric.

**Fundamental principles you must own:**

1. **Streaming = Server-Sent Events (SSE) over HTTP.** Chunks flow one at a time.
2. **Latency has two components**: TTFT (time to first token) and TPOT (time per output token, or tokens/sec).
3. **The client is responsible for reassembly.** Each provider streams its own chunk shape.
4. **Structured outputs and tool calls can stream too**, with partial JSON coming through.
5. **Aggregate carefully.** Bugs here produce weird truncations or duplications.
6. **Test cancellation.** Users close tabs; streams must clean up.

If you retain nothing else: **stream anything user-visible; measure TTFT, not just total latency.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Server-Sent Events (SSE) and the Protocol

LLM streaming uses **Server-Sent Events**: an HTTP response that keeps the connection open and sends chunks of text over time, each preceded by `data: ` and terminated by two newlines.

```
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: [DONE]
```

**OpenAI streaming:**

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True,
    stream_options={"include_usage": True},   # include usage in final chunk
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
    if chunk.usage:
        print(f"\nusage: {chunk.usage}")
```

**Anthropic streaming:**

```python
with client.messages.stream(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[...],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final_message = stream.get_final_message()
    print(f"\nusage: {final_message.usage}")
```

**Google Gemini streaming:**

```python
resp = model.generate_content(prompt, stream=True)
for chunk in resp:
    print(chunk.text, end="", flush=True)
```

The wire protocol differs slightly per provider (event types, message boundaries), but SSE is the underlying transport.

**Bytes vs text.** Streams are UTF-8 byte streams. Some tokens are multi-byte characters (emoji, non-Latin). Naive `.decode('utf-8')` on partial chunks can raise `UnicodeDecodeError`. Modern client libraries handle buffering, but be aware if you implement streaming yourself.

---

### 2.2 Async Streaming in Python

For a real service, you almost always want async streaming.

**OpenAI async:**

```python
from openai import AsyncOpenAI
client = AsyncOpenAI()

async def generate(prompt):
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

**Anthropic async:**

```python
from anthropic import AsyncAnthropic
client = AsyncAnthropic()

async def generate(prompt):
    async with client.messages.stream(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
```

**Serving streaming from FastAPI:**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(prompt: str):
    async def event_stream():
        async for chunk in generate(prompt):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**Client-side (browser, JavaScript):**

```javascript
const resp = await fetch("/chat/stream", { method: "POST", body: JSON.stringify({prompt}) });
const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
let buffer = "";
while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += value;
    // parse SSE lines from buffer, extract "data: {...}", update UI
}
```

---

### 2.3 Streaming with Structured Outputs and Tool Calls

**Structured outputs can stream, too.** OpenAI supports it:

```python
with client.beta.chat.completions.stream(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=Person,
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)
    final = stream.get_final_completion()
    person = final.choices[0].message.parsed
```

Partial JSON is delivered as it generates. Use `event.type` to distinguish between text deltas and other events.

**Tool calls in streaming.** OpenAI streams tool call arguments incrementally:

```python
current_tool_call = {"name": "", "arguments": ""}
for chunk in stream:
    for tc in chunk.choices[0].delta.tool_calls or []:
        if tc.function.name:
            current_tool_call["name"] += tc.function.name
        if tc.function.arguments:
            current_tool_call["arguments"] += tc.function.arguments

# After stream ends, parse:
args = json.loads(current_tool_call["arguments"])
```

Anthropic exposes rich event types via `stream.text_stream`, `stream.tool_use_stream`, etc.

**Why stream tool arguments?** For a tool that takes a long time to generate arguments (a complex SQL query, a big JSON extraction), streaming the arguments lets you display progress in a UI ("Model is composing a query...").

**Partial parsing.** For structured outputs, `partial-json` and similar libraries can parse **incomplete JSON** — useful for progressive rendering.

---

### 2.4 Latency Metrics: TTFT, TPOT, and End-to-End

**Time to First Token (TTFT)** — time from request start to first token emitted. Dominated by:
- Network latency to the provider.
- Model warmup, batching queue.
- Prompt processing (proportional to input tokens).

**Time Per Output Token (TPOT)** — average time between subsequent tokens. Dominated by:
- Model size and hardware.
- Batch load on the provider's infrastructure.

**End-to-end latency:**

$$T_{\text{total}} = T_{\text{TTFT}} + N_{\text{output}} \cdot T_{\text{TPOT}}$$

**Typical numbers (2026 ballpark):**

| Model | TTFT | TPOT |
|-------|------|------|
| GPT-4o | ~500ms | ~30 tokens/s |
| Claude 3.5 Sonnet | ~600ms | ~80 tokens/s |
| GPT-4o-mini | ~300ms | ~100 tokens/s |
| Groq's LLaMA-3.1-70B | ~200ms | ~250 tokens/s |

Latency varies significantly by time of day, provider load, and prompt length.

**UX implication:** even a 3-second total generation feels fast if TTFT is 400ms (user sees words appear immediately). Same 3-second generation delivered as one blob at t=3s feels sluggish.

**Measure both metrics in production.** Log first-token-time and total-completion-time per request. Alert on regressions.

**When streaming isn't better:**
- **Structured JSON where downstream code can't use partial results.** Might as well await the full response.
- **Very short responses** (< 20 tokens) — streaming overhead not worth it.
- **Bulk backend jobs** — user isn't waiting, so TTFT doesn't matter.

---

### 2.5 Cancellation, Errors, and Robustness

**Cancellation.** Users close tabs. Your streaming should propagate cancellation upstream:

```python
async def event_stream(request):
    async for chunk in llm.generate(prompt):
        if await request.is_disconnected():
            break               # stops iterating; cleanup happens automatically
        yield f"data: {chunk}\n\n"
```

For OpenAI-style clients:

```python
try:
    async for chunk in stream:
        if cancelled_by_client():
            await stream.close()   # or the equivalent method
            break
        ...
finally:
    # Ensure any resources are released
    pass
```

**Errors mid-stream.** A stream can fail after some tokens have been sent. Handle:

- If your server has already sent partial content to the client, you can't retract it. Send an error-event to signal the client.
- If nothing has been sent yet (still in TTFT window), you can safely retry with the same input and start over.

```python
async def robust_stream():
    started = False
    try:
        async for chunk in generate():
            started = True
            yield {"type": "text", "value": chunk}
    except Exception as e:
        if started:
            yield {"type": "error", "value": str(e)}
        else:
            raise  # let caller retry
```

**Timeouts on streams.** Set a total-time cap and a per-chunk cap:

```python
async def with_timeouts(stream, per_chunk_timeout=30, total_timeout=180):
    start = time.time()
    async for chunk in async_timeout_generator(stream, per_chunk_timeout):
        yield chunk
        if time.time() - start > total_timeout:
            raise TimeoutError("Stream total time exceeded")
```

**Health metrics to alert on:**
- TTFT p50, p95, p99.
- Time per token p50, p95.
- Cancellation rate.
- Error rate mid-stream.

**Buffering considerations.** Reverse proxies (nginx, CloudFront) can buffer SSE unless configured to pass through — a common gotcha. In nginx: `proxy_buffering off;`. Also disable compression on SSE streams (`Content-Encoding: identity`).

---

## 3. Mental Models & Analogies

### 3.1 The "Waiter Bringing Courses" Model

Non-streaming = ordering dinner, waiter disappears for 20 minutes, then brings the whole meal at once. Streaming = waiter starts bringing dishes as they're ready — appetizer at 5 minutes, entrée at 12, dessert at 20. Same total time, radically different experience.

Consequences:
- **Time to first dish (TTFT) is the perceived latency.** Even if the total meal takes just as long, you feel served, not ignored.
- **You can decide you're full and leave** (cancellation). Non-streaming, you're committed to the whole thing.
- **Course reordering is possible for smart kitchens.** If a dish is failing, replace it. In streaming, once bytes go out, they can't come back.
- **You need a plate** — the client has to buffer and assemble chunks correctly. Drop a chunk and the meal's ruined.

This model captures why streaming is a UX transformation, not just a latency optimization.

### 3.2 The "Text Ticker Tape" Model

Streaming is a ticker tape: characters (or tokens) come off one at a time in order, and you're building your display by appending. Some events on the tape are content; some are metadata (usage counts, finish reasons, tool calls). Your job is to distinguish, buffer, and reassemble.

This model emphasizes the **ordered, incremental** nature of streaming. Unlike a request/response API where you get a coherent JSON blob, streaming forces you to think about state: "so far, I have this partial message; the next chunk will extend it."

If your rendering logic doesn't handle "half a token" gracefully — say, you split UTF-8 bytes mid-character — you get gibberish. If your JSON parser doesn't handle "half an object," you get parse errors. Streaming teaches robustness to partial state.
![IMG-STREAM-01](/4%20—%20LLM%20Engineering/images/IMG-STREAM-01.jpg)

> **Caption:** Streaming doesn't reduce total time; it improves perceived latency by delivering the first token quickly.
> **Placement:** Section 2.4.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Streaming Doesn't Matter for Backend Services"

Sometimes true (batch jobs), often false. Any user-facing UI benefits from streaming. Any pipeline that consumes partial output (progressive JSON parsers, live UI updates, real-time transcription) benefits. Even backend logs are more useful when they capture the generation as it happens rather than at the end.

### 4.2 "Just Concatenate All the Deltas"

Watch for edge cases:
- **Empty deltas** — some chunks are metadata only (usage stats, finish reason).
- **Reasoning content** (o1/o3 models, Anthropic extended thinking) — streamed on a separate channel; don't concatenate into visible output.
- **Tool call arguments** — arrive as separate JSON-string fragments; concatenate then parse.
- **Multiple choices** — when `n > 1`, each choice streams separately.

Test with realistic streams; don't assume the shape.

### 4.3 "SSE Streams Just Work Through Any Proxy"

Reverse proxies love to buffer. Nginx, CloudFront, some load balancers will wait for a full response before forwarding — defeating streaming. Fixes:
- `proxy_buffering off;` (nginx).
- Chunked transfer encoding on your server.
- Disable response compression for SSE routes.
- Test end-to-end from your production edge, not just localhost.

---

## 5. Self-Assessment Bank (Streaming)

### Questions

**Q1 (Short answer).** What is the HTTP-level protocol used for LLM streaming?

**Q2 (Multiple choice).** TTFT stands for:
- (a) Time to Follow Task.
- (b) Time to First Token — the elapsed time from request send to first token received.
- (c) Total Time For Tokens.
- (d) Tokens To First Turn.

**Q3 (Short answer).** In one paragraph, explain how TTFT and TPOT together determine the user-perceived latency of a streaming response.

**Q4 (Multiple choice).** Which of these is a legitimate reason NOT to stream?
- (a) User-facing chatbot.
- (b) Backend batch job with no live consumer.
- (c) Any long generation.
- (d) A translation service.

**Q5 (Short answer).** When streaming a structured JSON output, how do you handle partial JSON?

**Q6 (Multiple choice).** Nginx by default:
- (a) Passes streams through unchanged.
- (b) Buffers responses — you must set `proxy_buffering off` for SSE.
- (c) Doesn't support SSE.
- (d) Adds compression to streams automatically.

**Q7 (Short answer).** How do you propagate a client-side cancellation (user closes tab) back to the upstream LLM call?

**Q8 (Multiple choice).** Streamed tool call arguments:
- (a) Arrive all at once as a JSON string.
- (b) Arrive as incremental fragments that must be concatenated then parsed.
- (c) Cannot be streamed.
- (d) Are always delivered separately from text.

**Q9 (Short answer).** Name three metrics you should log for every streaming LLM request in production.

**Q10 (Multiple choice).** When a stream fails mid-generation after some tokens were emitted:
- (a) Retry from the beginning silently.
- (b) You can't retract emitted content; send an error-event to signal the client.
- (c) Ignore.
- (d) Force reload.

---

### Answer Key & Detailed Explanations

**A1.** **Server-Sent Events (SSE)**, an HTTP-based protocol where the server keeps the connection open and sends messages prefixed with `data: ` and terminated with `\n\n`. The response's `Content-Type` is `text/event-stream`.

**A2. (b).** Time to First Token — the time from your request being sent to the first token being received by your client. It's the dominant contributor to *perceived* latency in a streaming UX.

**A3.** Perceived latency = TTFT + generation time. Streaming lets the user see the first word around TTFT ms after asking (typically 200–800ms); subsequent tokens arrive every 1/TPOT seconds. For a 500-token response at 50 tokens/sec (TPOT = 20ms), total time is TTFT (say 500ms) + 500 × 20ms = 10.5s, but the user feels it's fast because words appear immediately and continuously. The same 10.5-second wait for a non-streaming response would feel painfully long.

**A4. (b).** Backend batch jobs consumed by other services (or written to storage) don't benefit from streaming's UX property. Everything user-facing does.

**A5.** Use a **partial JSON parser** (e.g., the `partial-json` npm package or Python equivalents like the `jsonstreams`/`streaming-json-py` libs). These parse best-guess valid JSON at any point in the stream, letting you progressively render fields as they arrive. Alternatively, some providers (OpenAI's structured output streams) emit `content.delta` events keyed to schema paths, so you can render field-by-field.

**A6. (b).** Nginx buffers responses by default. For SSE endpoints you must set `proxy_buffering off;`. Also often need to disable gzip on streaming responses.

**A7.** In FastAPI: check `await request.is_disconnected()` periodically inside the streaming loop and break when true. In other frameworks: watch the underlying HTTP connection's cancel signal. On break, ensure you call the provider client's `close()` (or equivalent) so the upstream connection is released — otherwise you'll leak connections and be billed for tokens the user won't see.

**A8. (b).** Tool calls stream as incremental fragments. OpenAI emits `tool_calls[i].function.arguments` deltas that you concatenate; only after the stream ends (or the tool_call is marked complete) is the full arguments string valid JSON.

**A9.** Any three of: TTFT (time to first token), TPOT (time per output token) or total generation time, input tokens, output tokens, total cost, cancellation rate, error rate, provider model, request ID, per-chunk timing distribution.

**A10. (b).** Once tokens are sent to the client, they can't be retracted (the client has already rendered them). Best practice: send an error event on the stream (`{"type": "error", "message": ...}`), let the client display or handle it. If the error occurs before *any* tokens were sent, you can safely retry silently or return an error status code.

---

## 6. Practice Prompts

1. **Basic stream.** Build a FastAPI endpoint that streams tokens from OpenAI, and a simple HTML page with JS that consumes it via `EventSource`.
2. **Cancellation.** Extend the above to detect `is_disconnected` and close the upstream stream. Verify with a curl call that ends mid-generation.
3. **Structured stream.** Stream a Pydantic-typed output; progressively parse and render fields as they arrive.
4. **Tool streaming.** Stream a tool call. Accumulate the tool call's argument fragments, parse the final JSON, execute the tool, and continue.
5. **Latency dashboard.** Track TTFT and TPOT across 1000 real requests. Plot histograms. Alert if p95 TTFT exceeds a threshold.

---

## 7. References

- MDN Web Docs, ["Using server-sent events"](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).
- OpenAI streaming: [https://platform.openai.com/docs/api-reference/chat/streaming](https://platform.openai.com/docs/api-reference/chat/streaming).
- Anthropic streaming: [https://docs.anthropic.com/en/api/messages-streaming](https://docs.anthropic.com/en/api/messages-streaming).
- FastAPI streaming: [https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse).
- Vercel AI SDK: [https://sdk.vercel.ai/docs](https://sdk.vercel.ai/docs) — canonical UI streaming toolkit for JS.
