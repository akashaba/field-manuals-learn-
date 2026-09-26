# 10 — Rate Limiting

> **Module goal:** Prevent a single misbehaving client (or your own runaway agent) from taking down your service or bankrupting your LLM bill. Master the algorithms, the storage options, the 429/Retry-After protocol, and the LLM-specific twist: rate limiting by tokens and by dollars, not just by requests.

---

## 1. Executive Summary & Core Concepts

**Rate limiting** enforces an upper bound on how much of a resource a client can consume in a time window. Its purposes:

1. **Fairness** — one noisy user doesn't starve everyone else
2. **Cost control** — LLM APIs charge per token; unbounded traffic means unbounded bills
3. **Abuse mitigation** — brute-force, scraping, credential stuffing, DoS-adjacent attacks
4. **Backpressure to upstream** — your service can't pass more than X RPS to Anthropic; you must reject at your edge, not queue infinitely
5. **SLO protection** — degradation is graceful (429 to *some* users) instead of catastrophic (5xx to *all* users)

**Four canonical algorithms:**

| Algorithm | Behavior | When to use |
|-----------|----------|-------------|
| **Fixed window** | Count requests per calendar minute | Simple; suffers "double-burst" at window boundary |
| **Sliding window (log)** | Store every request timestamp | Accurate; memory-heavy |
| **Sliding window (counter)** | Weighted approximation of two adjacent windows | Cheap + accurate — most production choices |
| **Token bucket** | Refill at rate R, capacity C; each request costs tokens | Bursty-friendly; standard for API SDKs |
| **Leaky bucket** | Fixed drain rate, queue at bucket | Smoothing traffic, not bursty-friendly |

**LLM-specific:** you rate-limit by **three dimensions** simultaneously:
- **RPM** (requests per minute)
- **TPM** (tokens per minute — input + output)
- **$ per period** (per-user budget cap)

An LLM request can be 1× or 100× the cost of the next. Requesting 100 tokens/min is very different from 1000 tokens/min. Every major LLM API (OpenAI, Anthropic, Bedrock) publishes RPM *and* TPM limits.

**Where to enforce:**
- **Edge (nginx, Envoy, API Gateway)** — cheap, coarse, DDoS-friendly, but doesn't know request costs
- **Application (middleware)** — per-user, per-route, cost-aware — the most flexible
- **Upstream client (SDK)** — protects *outbound* calls to LLM providers (retry with backoff on 429)

**Storage for counters:**
- **In-process** — fast, no shared state; only works for single-replica
- **Redis** — the standard: `INCR` + `EXPIRE`, or Lua script for atomic multi-op
- **Cloud-native** — Cloudflare Workers, DynamoDB, Memorystore

**The client contract** (RFC 6585, IETF `RateLimit-*` draft):
- Status `429 Too Many Requests`
- `Retry-After: N` (seconds) header
- `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset` (draft-standard)
- A JSON body explaining the limit hit

If you don't return `Retry-After`, well-behaved clients spin.

---

## 2. Deep-Dive Breakdown

### 2.1 Token Bucket — The Default Choice

The classic algorithm: a bucket holds tokens (not LLM tokens — abstract permits). Refills at rate $R$ tokens/second, up to capacity $C$. Each request consumes $k$ tokens (usually 1 for RPS, or a computed value for weighted requests). If the bucket has < $k$ tokens, the request is rejected (or queued).

```python
import time
from dataclasses import dataclass, field

@dataclass
class TokenBucket:
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def take(self, k: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= k:
            self.tokens -= k
            return True
        return False
```

Properties: allows short bursts up to `capacity`, then rate-limits to `refill_rate`. Tune `capacity` for bursty legitimate traffic (e.g., a browser tab opens and fires 8 requests in 100ms — you don't want to reject those).

### 2.2 Redis-Backed Sliding Window Counter

For a multi-replica service, in-process buckets fragment (each replica has its own bucket → user gets N× the intended limit). Use Redis with an atomic Lua script:

```python
# rate_limit.py
import redis, time
r = redis.Redis()

LUA_SLIDING = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, now .. '-' .. math.random())
    redis.call('EXPIRE', key, window)
    return {1, limit - count - 1}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = oldest[2] and (tonumber(oldest[2]) + window - now) or window
    return {0, retry_after}
end
"""
rl = r.register_script(LUA_SLIDING)

def allow(user_id: str, limit: int, window_sec: int) -> tuple[bool, float]:
    now = time.time()
    allowed, meta = rl(keys=[f"rl:{user_id}"], args=[now, window_sec, limit])
    return bool(allowed), float(meta)
```

Trade-off: the "log" approach stores every timestamp — memory grows with rate. For very high traffic, use the *approximate* sliding-window counter:

$$
\text{count}_{\text{effective}} = \text{count}_{\text{current}} + \text{count}_{\text{prev}} \cdot \frac{\text{window} - \text{elapsed\_in\_current}}{\text{window}}
$$

O(1) memory, ~5% error at window boundary.

### 2.3 FastAPI Middleware Integration

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    user_id = request.headers.get("x-user-id") or request.client.host
    tier = get_user_tier(user_id)  # free/pro/enterprise
    limits = TIER_LIMITS[tier]     # e.g., {"rpm": 60, "tpm": 100_000, "usd_per_day": 5.0}

    allowed, retry_after = allow(user_id, limits["rpm"], 60)
    if not allowed:
        return JSONResponse(
            status_code=429,
            headers={
                "Retry-After": str(int(retry_after) + 1),
                "RateLimit-Limit": str(limits["rpm"]),
                "RateLimit-Remaining": "0",
                "RateLimit-Reset": str(int(time.time() + retry_after)),
            },
            content={"error": "rate_limited", "detail": "RPM limit exceeded"},
        )
    return await call_next(request)
```

For the **TPM** and **$-per-day** dimensions, you can't decide before the request — you don't know the token cost yet. Two patterns:
1. **Pre-authorize with an estimate**, then reconcile after the response (subtract actual tokens from bucket at the end).
2. **Post-check** — always allow, but if the user is already over TPM/$ cap for the window, reject *next* request.

Anthropic and OpenAI both do variants of this on their end.

### 2.4 Client-Side: Handling 429 from Providers

Your service calls Anthropic; Anthropic 429s. **Exponential backoff with jitter** is the standard:

```python
import httpx, asyncio, random

async def call_with_backoff(client, request_kwargs, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = await client.post(**request_kwargs)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()
            retry_after = float(resp.headers.get("retry-after", "1"))
        except httpx.TimeoutException:
            retry_after = 2 ** attempt
        # Full jitter — random in [0, cap]
        cap = min(60, retry_after * (2 ** attempt))
        await asyncio.sleep(random.uniform(0, cap))
    raise Exception("max retries exceeded")
```

Rules:
- **Always jitter.** If 1000 clients back off deterministically to the same second, they thundering-herd the retry.
- **Cap the wait** (~60s). Beyond that, fail the request; do not tie up a worker for minutes.
- **Distinguish retryable errors**: 429 → yes, 503 → yes, 400 → no (retrying makes no difference).

**Anthropic-specific:** the `x-ratelimit-*` response headers tell you exactly when the bucket refills — you can proactively slow yourself instead of hammering until 429s appear.

### 2.5 Cost-Based Rate Limiting (The AI Twist)

Traditional RPS misses the point when a request can cost 10× another. The right dimension is **$ per user per period.**

```python
class CostBucket:
    """Redis-backed spending cap."""
    def try_charge(self, user_id: str, cost_usd: float, cap_usd: float, period: str) -> bool:
        key = f"spend:{user_id}:{period}"  # e.g., spend:u123:2026-09-23
        # Atomic incrbyfloat and check
        new_total = r.incrbyfloat(key, cost_usd)
        r.expire(key, 86400 * 2)
        if new_total > cap_usd:
            # Refund and reject
            r.incrbyfloat(key, -cost_usd)
            return False
        return True

# In your handler:
estimated_cost = estimate_llm_cost(request)  # from token estimate × price
if not cost_bucket.try_charge(user_id, estimated_cost, cap=5.00, period=today()):
    raise HTTPException(429, "Daily spending cap reached")
resp = await llm_call(...)
actual_cost = compute_cost(resp)
delta = actual_cost - estimated_cost
if delta != 0:
    r.incrbyfloat(f"spend:{user_id}:{today()}", delta)  # reconcile
```

Combined with tiers (free = $0.50/day, pro = $10/day, enterprise = unlimited), this is the single most effective LLM cost control.

`[IMG-10-01]` — *Prompt: A diagram illustrating three rate-limiting dimensions for an LLM API service, drawn as three separate bucket icons flowing into a request-processing pipeline. Bucket 1 labeled "RPM: 60 req/min" filled with request icons. Bucket 2 labeled "TPM: 100k tokens/min" filled with token icons. Bucket 3 labeled "USD: $5/day" filled with dollar signs. Arrows show a request must pass through all three checks. Below, a red "429 Too Many Requests" response is emitted when any bucket is empty. Clean technical diagram style, blue/green/red palette.*

---

## 3. Mental Models & Analogies

### Model 1: The Amusement Park Line

Every ride has a maximum riders-per-hour. There's a queue with a fixed length (bucket capacity). Riders arriving faster than the ride can process either wait (queued) or get turned away (rejected — the sign says "return in 30 minutes"). VIP guests get a shorter line (tier-based limits). The whole park has a total-attendance cap that's independent of per-ride caps — analogous to global rate limiting on top of per-user.

The park explicitly rejects overloading: if it let everyone into every line, every ride breaks. The rejection is the *feature*. Same for rate limiting — the 429 is the design goal, not a bug. Without it, your service would eat *every* request until it collapsed for everyone.

### Model 2: The Credit Card

Rate limiting is not just RPS; it's a **credit facility**. Your issuer gives you:
- **Credit limit** (max outstanding balance = burst capacity)
- **Refill on payment cycle** (refill rate)
- **Per-transaction limits** (max cost per request)
- **Merchant category codes** (tier / route sensitivity)
- **Fraud triggers** (unusual patterns → block)

The card issuer's model matches production rate limiting almost exactly, and both exist for the same reason: the party holding the credit (the bank / your service) needs to bound liability from any single account. When a card issuer rejects a $10,000 charge from an account with a $5,000 limit, that's not "denial of service" — that's the mechanism working. Same for your 429.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — In-process rate limiting in a multi-replica service.**
You put a token bucket in FastAPI middleware. In dev, it works. In prod you scale to 5 pods; each pod has its own bucket; the user's effective limit is 5× the intended limit; the bucket resets independently per pod. Users hit uneven limits based on which pod they land on. **Fix:** back the bucket with Redis (shared state). In-process limiting is a *coordination optimization* — a small local pre-filter to save a Redis round-trip on the hot path — never the source of truth.

**Pitfall #2 — 429 without `Retry-After`.**
Client code sees 429, retries immediately, gets 429, retries immediately, DDoSing you and itself. Legitimate clients (browsers, well-written SDKs) obey `Retry-After`; without it they degrade to fixed backoff or worse. **Always** emit `Retry-After: <seconds>` and the `RateLimit-*` headers. And document them so client authors don't ignore them. Bonus: SDKs like httpx's built-in retry transport can be configured to honor these headers automatically.

**Pitfall #3 — Only rate-limiting requests, not cost.**
"I set 60 RPM per user." A user sends 60 requests/min, each with a 200k-token context and asking for 8k tokens output. That's ~12.5M tokens/min from one user; if you're on Anthropic's Sonnet at ~$3/M in + $15/M out, that's roughly $150/min ≈ $216k/day per user. Your RPM limit did nothing to stop it. **Fix:** always add TPM and $-per-day dimensions. Estimate token cost before dispatch; reconcile after. Cap on the tightest dimension. Enforce token-length limits on input (max_prompt_tokens); reject > cap before ever calling the LLM.

---

## 5. Self-Assessment Bank

**Q1 (MC):** In the token bucket algorithm, `capacity` is:
A) The rate at which tokens refill
B) The maximum number of tokens that can accumulate — i.e., the maximum burst
C) The maximum requests per second
D) The window duration

**Q2 (short):** Why is Redis a common backing store for rate limits in a multi-replica service, and what's the single-replica alternative?

**Q3 (MC):** A well-behaved 429 response should include:
A) `Retry-After` seconds header
B) `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset` headers
C) A JSON body describing the limit that was hit
D) All of the above

**Q4 (short):** Your service rate-limits at 60 RPM per user. Explain why this alone is insufficient for an LLM-serving API and what dimensions you should add.

**Q5 (MC):** Which retry strategy is safest when 1000 clients simultaneously hit a 429?
A) Retry immediately
B) Fixed 1-second backoff
C) Exponential backoff with jitter, capped at 60s
D) Retry every second forever

**Q6 (MC):** Rate-limit "double-burst at window boundary" is a problem specific to:
A) Token bucket
B) Fixed-window counter
C) Sliding window counter (approximate)
D) Leaky bucket

**Q7 (short):** Describe how you'd enforce a $5/day-per-user spend cap on an LLM API. What data structures, when to charge, what to do on cap-hit.

**Q8 (MC):** You call Anthropic and receive a 429 with `Retry-After: 12`. Your service is behind a caller-facing rate limit of your own. What should you return to your caller?
A) 500 Internal Server Error
B) 429 Too Many Requests with your own `Retry-After` header
C) 503 Service Unavailable
D) Wait 12s and retry, then respond

**Q9 (short):** Explain why you should validate max input length (max prompt tokens) as its own gate *before* the rate limiter, even though the rate limiter caps TPM.

**Q10 (MC):** Global rate limiting (across all users) is complementary to per-user rate limiting because:
A) It protects the upstream provider's aggregate limits (you can only push so many RPM to Anthropic)
B) It smooths thundering herds when many users hit at once
C) It bounds your infra capacity from any single cause
D) All of the above

---

### Answer Key

**A1: B.** `capacity` is the max stored tokens = the largest burst you'll allow. `refill_rate` is (A) — how fast tokens replenish. A bucket configured `capacity=10, refill_rate=1/s` allows a burst of 10 followed by a steady 1 req/s.

**A2:** Redis provides shared, atomic state across your replicas. Every pod hits the same Redis keys, so a single user's counter is truly one counter regardless of which pod serves the request. Single-replica alternative: in-process (a Python dict of buckets). It's faster (no network hop) but only correct if you'll never scale beyond one replica. In hybrid deployments, some services use in-process as a *first-pass* cheap check (reject clear over-limits without Redis), falling through to Redis for the authoritative decision.

**A3: D — all of the above.** `Retry-After` tells the client when to try again; `RateLimit-*` headers tell them their remaining budget so they can slow down proactively; JSON body allows the client to distinguish which limit fired (RPM vs TPM vs $) and adjust. All three are cheap to include and make your API dramatically nicer to integrate against.

**A4:** RPM alone treats every request as equivalent, but LLM requests vary in cost by orders of magnitude. A 200k-token prompt with a 4k-token output is not the same load as "hi." Add: **TPM** (total input+output tokens per minute per user) and **$ per day per user** (converts variable request costs into a single fungible budget). Optionally also **concurrent requests per user** (protects against pathological parallel abuse). Rate-limit on the *tightest* dimension.

**A5: C.** Exponential backoff with jitter is the standard. Immediate retry (A) makes 429s worse. Fixed backoff (B) causes all 1000 clients to retry at t+1s simultaneously — a thundering herd. Retry forever (D) is unbounded resource use. The cap (60s) prevents any single request from tying up a worker indefinitely; the jitter spreads retries; the exponential shape ensures faster recovery than pure random.

**A6: B — fixed-window counter.** Because the counter resets at a fixed boundary (e.g., top of the minute), a user can send 60 requests at 12:00:59 and another 60 at 12:01:00 — effectively 120 in ~1 second. Sliding window and token bucket smooth this. Fixed window is chosen only when the simpler math is worth the accuracy loss (rare in production).

**A7:** Use a Redis key `spend:<user_id>:<YYYY-MM-DD>` with `INCRBYFLOAT`. Before dispatching an LLM call, **estimate** the cost from `(input_tokens_estimate + max_output_tokens) × price_table[model]` and `INCRBYFLOAT` by that amount. If the new balance exceeds $5, `INCRBYFLOAT` by the negative delta (refund) and return `429 {"error": "daily_spend_cap"}` with `Retry-After: <seconds-until-tomorrow-UTC>`. After the LLM responds, compute the **actual** cost and `INCRBYFLOAT` the delta (positive or negative) so the counter converges to reality. Set TTL to 48h to auto-expire yesterday's key. Roll over at UTC midnight (or user's timezone if you're doing per-user local billing).

**A8: B.** Rate-limit exhaustion upstream is a form of rate-limit exhaustion your caller should know about. Returning 500 (A) suggests a server bug they can't help with. 503 (C) implies you're broken. 429 with `Retry-After` (perhaps 12s + a small buffer) is honest and instructs the client correctly. Waiting 12s in-request (D) ties up a worker slot and increases user-visible latency; only makes sense if the wait is very short (< 1s) and you can guarantee it.

**A9:** Because a single 500k-token prompt can, by itself, exceed your TPM cap for the minute — you'd reject the request, but only *after* the client uploaded and you parsed 500k tokens (network, compute). By validating max prompt length as a cheap upfront gate (e.g., "reject > 32k tokens"), you fail fast, save compute and bandwidth, and give a clearer error ("input too long") than "rate-limited." It also protects against a class of DoS where the attacker sends huge prompts hoping to exhaust your token budget instantly. Layered defense: input-length cap → per-request cost estimate → rate-limit check → LLM call.

**A10: D.** All apply. (A) You can't push more than your provider quota; hitting it results in cascading 429s. Enforce a global RPM/TPM ceiling below your provider's limit and shed load early. (B) Prevents user-independent traffic spikes (e.g., viral moment) from taking down the whole service. (C) Bounds infra saturation regardless of which user is the cause. Global limits are set at 80–90% of upstream capacity; per-user is set for fairness and business tiering.

---

**Related modules:**
- `learning/01-fastapi.md` — where the middleware attaches
- `learning/11-caching.md` — caching relieves rate-limit pressure by avoiding duplicate calls
- `learning/12-authentication.md` — you need identity to rate-limit per user
- `learning/14-cost-optimization.md` — the $-per-day cap is a cost tool

**Practice prompts:**
1. Implement the sliding-window Lua script above against local Redis. Test 100 requests at various rates.
2. Add a FastAPI middleware that reads `x-user-id`, looks up their tier, and enforces RPM + TPM + $/day. Emit `RateLimit-*` headers.
3. Write client-side backoff for calls to Anthropic; handle 429s with `Retry-After`, jitter, and a 60s cap.
4. Compare rate-limited service throughput vs. unbounded under 10× the intended load — plot latency curves.

**References:**
- Stripe engineering blog — "Scaling your API with rate limiters"
- Cloudflare — "How we built rate limiting capable of scaling to millions of domains"
- RFC 6585 — 429 Too Many Requests
- IETF draft — "RateLimit Fields for HTTP" (`RateLimit-Limit`, etc.)
- Anthropic docs — rate limits, `x-ratelimit-*` headers
