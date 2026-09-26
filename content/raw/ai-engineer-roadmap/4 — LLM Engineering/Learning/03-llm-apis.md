# LLM APIs — Master Study Guide

> **Track:** LLM Engineering · **Module:** 03
> **Prerequisites:** Foundations Module 05 (APIs & JSON), Module 01 (Tokens).
> **Time budget:** ~8–10 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Nearly all production LLM engineering happens through **APIs** — OpenAI, Anthropic, Google, together.ai, Groq, and dozens more. The APIs share a family resemblance (messages, roles, streaming, structured outputs) but differ in every detail that matters when writing robust code.

Working confidently with LLM APIs means:

- Calling them correctly with proper error handling.
- Choosing between providers based on capability, not folklore.
- Handling rate limits, retries, timeouts, and streaming.
- Understanding cost implications of every parameter.
- Being able to swap providers without rewriting your app.

**Fundamental principles you must own:**

1. **The Chat Completions API is the modern standard.** All frontier providers speak a `messages: [{role, content}]` shape.
2. **API calls are stateless.** You resend the full conversation each time.
3. **Every knob affects cost, latency, and quality** — max_tokens, temperature, top_p, stream, tools.
4. **Errors are common and specific.** 429 (rate limit), 400 (bad request), 500 (provider issue) — you must handle each.
5. **Never let a single call take forever.** Set timeouts. Use streaming. Use retries with backoff.
6. **Cost accounting starts on day one.** Log token counts per request — otherwise you'll get surprise bills.

If you retain nothing else: **treat LLM APIs like any other external service — with timeouts, retries, structured errors, observability, and abstraction over vendors.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Chat Completions API Shape

Modern LLM APIs are variations on a common theme: **send a list of messages, get a message back.**

**OpenAI:**

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.2,
    max_tokens=200,
)
print(resp.choices[0].message.content)
print(resp.usage)   # prompt_tokens, completion_tokens, total_tokens
```

**Anthropic:**

```python
from anthropic import Anthropic
client = Anthropic()

resp = client.messages.create(
    model="claude-3-5-sonnet-latest",
    system="You are a helpful assistant.",   # top-level, not a message
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=200,
    temperature=0.2,
)
print(resp.content[0].text)
print(resp.usage)   # input_tokens, output_tokens, cache_read_input_tokens, ...
```

**Google Gemini:**

```python
import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-pro",
                              system_instruction="You are a helpful assistant.")
resp = model.generate_content("What is the capital of France?",
                              generation_config={"temperature": 0.2, "max_output_tokens": 200})
print(resp.text)
```

**Open models via together.ai / Groq / Fireworks / DeepInfra:**

Most speak the OpenAI-compatible shape. Just change `base_url` and `model`:

```python
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=os.environ["TOGETHER_API_KEY"])
resp = client.chat.completions.create(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", messages=[...])
```

**Key parameters (universal or near-universal):**

- **`model`** — the specific model version.
- **`messages`** — the conversation, with `role` in `{system, user, assistant, tool}`.
- **`temperature`** ∈ [0, 2] — sampling randomness. 0 = deterministic-ish; 1 = default; higher = wilder.
- **`top_p`** ∈ (0, 1] — nucleus sampling. Sample from the smallest set of tokens whose cumulative probability ≥ top_p.
- **`max_tokens`** — hard cap on output length.
- **`stop`** — sequences that halt generation.
- **`stream`** — bool, return SSE stream.
- **`tools`** / **`functions`** — for function calling (Module 06).
- **`response_format`** — JSON mode / structured outputs (Module 05).
- **`seed`** — best-effort reproducibility (not fully deterministic).

**Provider-specific:**

- OpenAI: `logprobs`, `logit_bias`, `parallel_tool_calls`, `reasoning_effort` (o1/o3 series).
- Anthropic: `system` (top-level), `cache_control` (prompt caching), `thinking` (extended thinking).
- Google: `safety_settings`, `system_instruction`, `tools` (function calling).

---

### 2.2 Roles and the Conversation Model

**System role** — instructions that shape the model's behavior throughout the conversation.

- OpenAI: `{"role": "system", "content": "..."}` as the first message.
- Anthropic: `system="..."` as a top-level argument (not in messages).
- Google: `system_instruction="..."` on the model.

**User role** — the human's input.

**Assistant role** — the model's prior responses. In multi-turn conversations, previous assistant turns are stored here.

**Tool role** — the result of a tool call (Module 06). Injected after an assistant turn that requested the tool.

**A typical multi-turn conversation:**

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Helena?"},
    {"role": "assistant", "content": "It's currently 20°F and snowing."},
    {"role": "user", "content": "And in Kampala?"},
]
```

Every request sends the **entire history**. The API is stateless — the server doesn't remember prior calls; you're responsible for maintaining and re-sending state.

**Multi-modal content.** Frontier models accept mixed content:

```python
# OpenAI (Anthropic is similar with type: "image")
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://..."}},
    ],
}]
```

Also: audio (Whisper for transcription, GPT-4o for audio input), PDFs (Claude 3.5 Sonnet+, Gemini), files (via file APIs).

---

### 2.3 Error Handling: Rate Limits, Timeouts, Retries

**Errors you WILL hit:**

- **429 Too Many Requests** — rate limit. Wait and retry.
- **408 / timeout** — the request took too long. Retry (with a longer timeout).
- **500 / 502 / 503** — provider issue. Retry with backoff.
- **529 (Anthropic)** — overloaded. Retry.
- **400** — your bad. Don't retry.
- **401** — bad API key. Don't retry.
- **413** — payload too large. Truncate.
- **context_length_exceeded** — prompt too long. Fix your context management.

**Retry with exponential backoff and jitter:**

```python
import time, random
from anthropic import Anthropic, RateLimitError, APIError

def call_with_retry(client, messages, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.messages.create(messages=messages, **kwargs)
        except (RateLimitError, APIError) as e:
            if attempt == max_retries - 1:
                raise
            # exponential backoff with full jitter
            delay = min(60, (2 ** attempt) + random.random())
            time.sleep(delay)
```

Or use **`tenacity`** for a cleaner pattern:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
def call(client, **kwargs):
    return client.messages.create(**kwargs)
```

**Timeouts.** Set them explicitly. Default timeouts are usually too long (or non-existent):

```python
client = Anthropic(timeout=30.0, max_retries=0)   # do retries yourself
```

**Rate limits.** Providers rate-limit by **RPM (requests per minute)** and **TPM (tokens per minute)**. To scale beyond one API key's limits:
- Request tier upgrades (usage-based).
- Batch parallel calls with a semaphore to stay under RPM.
- Multiple accounts / provider fallback for redundancy.

**Provider fallback pattern:**

```python
try:
    resp = call(primary_client, ...)
except (APIError, TimeoutError):
    resp = call(fallback_client, ...)
```

Fall back from OpenAI to Anthropic (or vice versa), or from a big model to a smaller one. Requires abstracting the message shape — see Section 2.5.

---

### 2.4 Sampling Parameters and Determinism

**Temperature** — controls sampling randomness by dividing logits before softmax.

- `temperature = 0` — nearly deterministic; picks the most likely token. Best for code, math, extraction.
- `temperature = 0.7` — moderately creative. Default for chat.
- `temperature = 1.0` — the model's "raw" distribution.
- `temperature > 1.0` — wild. Rare in production.

**`top_p` (nucleus sampling)** — sample from the smallest set of tokens whose cumulative probability ≥ p.

- `top_p = 0.9` is a common default.
- Combines with temperature — typically fix one at 1.0 and vary the other.

**Reproducibility.**
- OpenAI: `seed=42` — best-effort deterministic. Not truly deterministic due to floating-point non-associativity and inference-batch effects, but usually gets you the same output for the same input across close-in-time calls.
- Anthropic: no seed parameter yet, but `temperature=0` is close.
- Confirm you always get the same fingerprint (`system_fingerprint` in OpenAI response) between "identical" calls.

**When to use `temperature=0`:**
- Code generation (want the "best guess").
- Structured extraction.
- Classification tasks.
- Anything eval'd against a ground truth.

**When to use `temperature > 0`:**
- Brainstorming, creative writing.
- Diverse generation (self-consistency, majority vote).
- User-facing chat where a bit of personality helps.

**`stop` sequences** — halt generation when a specific string appears. Useful for controlled generation:

```python
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stop=["\n\nHuman:", "<END>"],
)
```

---

### 2.5 Provider Abstraction and Portable Code

If your app hits multiple providers (fallback, cost optimization, A/B testing), abstract the differences.

**LiteLLM** — a popular unifying wrapper:

```python
from litellm import completion

resp = completion(
    model="openai/gpt-4o",   # or "anthropic/claude-3-5-sonnet-latest"
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

LiteLLM handles the message-shape translation, error normalization, and cost accounting.

**Or write your own thin abstraction:**

```python
from typing import Literal, TypedDict, Protocol

class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class LLMClient(Protocol):
    def chat(self, messages: list[Message], **kwargs) -> str: ...

class OpenAIClient:
    def __init__(self, model: str, **defaults):
        self.model = model
        self.defaults = defaults
        self._client = OpenAI()
    def chat(self, messages, **kwargs):
        params = {**self.defaults, **kwargs}
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        return resp.choices[0].message.content

class AnthropicClient:
    def __init__(self, model: str, **defaults):
        self.model = model
        self.defaults = defaults
        self._client = Anthropic()
    def chat(self, messages, **kwargs):
        system = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)
        resp = self._client.messages.create(
            model=self.model, system=system, messages=chat_messages,
            **{**self.defaults, **kwargs},
        )
        return resp.content[0].text
```

**Observability. Log every call:**

```python
def log_call(model, messages, resp, latency_ms):
    logger.info({
        "event": "llm_call",
        "model": model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "latency_ms": latency_ms,
        "estimated_cost": price_estimate(model, resp.usage),
        "request_id": ...,
    })
```

Consider open-source observability tools: **Langfuse**, **Helicone**, **Phoenix (Arize)**, **LangSmith**. They give you traces, latency breakdowns, cost per session, and prompt-diff comparisons.

**Cost accounting** should be a first-class concern from the start. Add it to your logs, dashboards, and alerts. Nothing spooks a CFO like a $50k line item nobody predicted.

---

## 3. Mental Models & Analogies

### 3.1 The "Rest-Style Server, But Stateless" Model

Every LLM API is essentially a REST-ish JSON service that takes a conversation history + parameters and returns a completion. **Same discipline you'd apply to any external service applies here:**

- Timeouts (don't hang forever).
- Retries with exponential backoff on transient errors (429, 500, timeouts).
- Circuit breakers (stop calling a broken provider for a while).
- Logging and metrics (RPS, latency, error rate, cost).
- Idempotency where applicable (write IDs, use them if retries happen).

The one twist: **billed usage per call**. That adds cost as a first-class metric alongside latency and error rate.

Treat LLM providers like you'd treat Stripe or Twilio — external, occasionally flaky, always billable. No "special sauce" — the same engineering discipline you'd apply anywhere.

### 3.2 The "Function That Takes a Conversation and Returns a Completion" Model

Think of the LLM API as a **pure function**:

$$\text{response} = f(\text{model}, \text{messages}, \text{params})$$

The function is expensive (money and latency), sometimes non-deterministic (temperature > 0), sometimes unreliable (transient errors), and always billed. But it's a function — same input tends to produce similar output; the state lives entirely in the input.

Consequences of the pure-function frame:
- **You own the state.** Store it. Version it. Reproduce bugs by re-sending the same input.
- **Caching works.** Same messages → cache the response (mind temperature).
- **Testing is possible.** Snapshot a canonical set of inputs and check outputs stay reasonable across model updates.
- **Provider swaps become mechanical.** Same input; different function.

Anti-pattern: hiding LLM calls behind stateful frameworks that make debugging opaque. Every LLM call should be traceable back to a specific `messages` list and parameters. If you can't reproduce a bug by resending the same input, your abstraction is too thick.
![IMG-API-01](/4%20—%20LLM%20Engineering/images/IMG-API-01.jpg)
> **Caption:** LLM APIs are stateless request/response; every reliability pattern from ordinary REST applies here plus cost accounting.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "The Provider Remembers My Conversation"

They don't. LLM APIs (chat completions) are **stateless**. Every request must include the entire conversation history. If you send only the latest message, the model has no memory of what came before. Corollary: on every user turn, you re-bill the entire history. Also: don't confuse the OpenAI **Assistants API** (which does maintain server-side thread state) with the Chat Completions API — they're different products with different pricing and behavior.

### 4.2 "429 Means I Need a Bigger Plan"

Sometimes. Usually it means your **request pattern** is burstier than the RPM/TPM limit. Solutions in order of preference:
1. Add exponential backoff with jitter to your retry logic.
2. Add a semaphore to cap concurrent in-flight requests.
3. Batch multiple prompts into a single request where the API supports it.
4. Only then, request a rate-limit increase.

Just increasing limits without fixing your bursty client pattern often masks the real problem.

### 4.3 "I'll Handle Errors Later"

You won't. Write the retry + timeout + fallback wrapper on day one. LLM APIs fail regularly; downstream code that assumes success turns into midnight pager calls. Also: log every failed call and its parameters, so you can diagnose whether it's your code, the provider, or a specific input pattern. A logs-less LLM app is opaque and unfixable.

---

## 5. Self-Assessment Bank (LLM APIs)

### Questions

**Q1 (Short answer).** Describe the standard shape of a chat-completions request (fields you always send).

**Q2 (Multiple choice).** Are LLM chat completions APIs stateful or stateless?
- (a) Stateful — the provider remembers your conversation.
- (b) Stateless — you resend the full conversation each request.
- (c) Depends on the model.
- (d) Depends on the auth scheme.

**Q3 (Short answer).** Explain the difference between temperature and top_p sampling.

**Q4 (Multiple choice).** For code generation and structured extraction, the recommended temperature is:
- (a) 1.0
- (b) 0.7
- (c) 0.0 (or close to it)
- (d) 2.0

**Q5 (Short answer).** Write, in pseudocode, a retry-with-backoff wrapper for an LLM call that retries on rate-limit and 5xx errors but not on 400.

**Q6 (Multiple choice).** Prompt caching typically:
- (a) Costs the same as regular tokens.
- (b) Reduces the cost of stable-prefix input tokens on subsequent requests (~10–50% of normal).
- (c) Only works for output tokens.
- (d) Requires a special model.

**Q7 (Short answer).** Why do you almost always need to set explicit timeouts on LLM API calls?

**Q8 (Multiple choice).** In OpenAI's chat API, the `system` message is:
- (a) Ignored.
- (b) The first message in the `messages` array with role="system".
- (c) A top-level argument.
- (d) Encoded in the model URL.

**Q9 (Short answer).** Name three signals you should log for every LLM API call in production.

**Q10 (Multiple choice).** LiteLLM's main value proposition is:
- (a) It's the fastest client.
- (b) It normalizes the request/response shape across providers so you can swap models with one line change.
- (c) It replaces the model.
- (d) It embeds vectors.

---

### Answer Key & Detailed Explanations

**A1.** At minimum: `model`, `messages` (list of `{role, content}`), and optionally `max_tokens`, `temperature`, `stream`. Additional common params: `top_p`, `stop`, `tools`, `response_format`, `seed`.

**A2. (b).** Chat completions APIs are stateless. You must send the entire conversation history on every request. Note: OpenAI's separate Assistants API does have server-side thread state — but the vanilla chat completions API doesn't.

**A3.** **Temperature** divides logits by $T$ before softmax — controls the overall "sharpness" of the distribution. $T = 0$ is nearly deterministic; higher $T$ = more uniform, more varied outputs. **Top_p (nucleus)** truncates the distribution to the smallest cumulative-probability set exceeding $p$; then samples from that. Top_p is a hard cutoff on which tokens can even be considered; temperature adjusts probabilities across all tokens. In practice, fix one at 1.0 and vary the other.

**A4. (c).** For deterministic, correctness-sensitive tasks like code generation, extraction, classification, and math, use temperature 0 (or near-0). This selects the highest-probability token at each step — the model's most confident guess.

**A5.**
```python
def call_with_backoff(fn, is_retryable, max_retries=5, base=1.0, cap=60.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if not is_retryable(e) or attempt == max_retries - 1:
                raise
            delay = min(cap, base * (2 ** attempt)) + random.random()
            time.sleep(delay)

# is_retryable: True for RateLimitError, 500/502/503/529/timeout; False for 400/401/403/404.
```

**A6. (b).** Prompt caching lets a stable prefix be reused across requests at reduced cost. Anthropic charges ~10% of normal for cached reads (after a slightly higher first-write cost); OpenAI charges ~50% for cached reads (automatic on prefixes ≥ 1024 tokens). Only the prefix matches — any change invalidates the cache.

**A7.** Without timeouts, a stuck HTTP request can hang your process indefinitely — a single dropped connection or hung provider means your workers pile up and never recover. Set 15–60s timeouts depending on task; also implement retries so transient timeouts don't fail the whole operation.

**A8. (b).** In OpenAI's Chat Completions, the system message is the first entry of the `messages` array with `role="system"`. In Anthropic's Messages API, it's a top-level `system=` argument. In Google's Gemini API, it's `system_instruction` on the model object. This is one of the most annoying cross-provider differences.

**A9.** Any three of: model name, input tokens, output tokens, latency, cost estimate, request ID, prompt (or a hash), response (or a hash), error class if failed, retry count, user/session ID, request timestamp. Cost + tokens + latency + error rate are the "big four" for dashboards.

**A10. (b).** LiteLLM provides a unified client interface. You can call `openai/gpt-4o` or `anthropic/claude-3-5-sonnet-latest` or `groq/llama-3.1-70b` through the same function signature. Useful for provider abstraction, fallback, and A/B testing without vendor lock-in.

---

## 6. Practice Prompts

1. **Provider abstraction.** Write an `LLM` class that accepts `provider="openai"` or `"anthropic"` and normalizes the message shape. Call the same code with each provider.
2. **Retry + fallback.** Wrap an LLM call with (a) exponential backoff + jitter retry, (b) fallback to a secondary provider if primary fails 3 times.
3. **Cost accountant.** Build a decorator `@track_cost` that logs input tokens, output tokens, latency, and dollar cost per call.
4. **Batched RPM control.** Write a semaphore-based dispatcher that keeps concurrent in-flight calls ≤ 20 and RPM under a limit.
5. **Reproducibility check.** With `temperature=0` and `seed=42`, call the same prompt 10 times to OpenAI. Compare outputs and the `system_fingerprint`. Vary the seed; observe changes.

---

## 7. References

- OpenAI API: [https://platform.openai.com/docs](https://platform.openai.com/docs).
- Anthropic API: [https://docs.anthropic.com/en/api](https://docs.anthropic.com/en/api).
- Google Gemini API: [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs).
- LiteLLM: [https://docs.litellm.ai](https://docs.litellm.ai).
- Simon Willison's LLM blog series: [https://simonwillison.net/tags/llm/](https://simonwillison.net/tags/llm/).
- Anthropic engineering blog: [https://www.anthropic.com/news/prompt-engineering-for-business-performance](https://www.anthropic.com/news/prompt-engineering-for-business-performance) (2024+).
