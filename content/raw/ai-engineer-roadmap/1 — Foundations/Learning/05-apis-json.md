# APIs & JSON — Master Study Guide

> **Track:** Foundations · **Module:** 05
> **Prerequisites:** Modules 01 (Python). Module 06 (Linux/CLI) helps for `curl` and `jq`.
> **Time budget:** ~15–20 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Modern software is a **network of services** speaking JSON over HTTP. Whether you're consuming a public data API, integrating with an internal microservice, or *serving* a machine-learning model as an endpoint (which is exactly what you'll build in Build #2), you must be fluent in:

- **HTTP** — the transport (methods, status codes, headers, bodies).
- **JSON** — the payload format (types, encoding, schemas).
- **REST-ish conventions** — the vocabulary teams use for resource-oriented APIs.
- **Client-side ergonomics** — timeouts, retries, pagination, auth, idempotency.

Master these and you can (a) integrate any modern system, (b) design an API a teammate can consume without asking questions, and (c) reason about failures instead of guessing.

**Fundamental principles you must own:**

1. **HTTP is stateless.** Each request stands alone. State lives on the server (database) or is carried in the request (auth token, cookie).
2. **A URL identifies a resource; a method describes the action.** `GET /users/42` reads user 42; `DELETE /users/42` deletes it. Don't smuggle actions into URLs.
3. **Status codes matter.** `2xx` success, `3xx` redirect, `4xx` client error, `5xx` server error. The specific code is contract, not decoration.
4. **JSON has six types.** `null, boolean, number, string, array, object`. That's it. There is no date type, no int-vs-float distinction, no comment syntax, and no tuple.
5. **Idempotency is a design choice.** `GET`, `PUT`, `DELETE` should be idempotent (safe to retry); `POST` is not. This choice interacts with retries in critical ways.

If you retain nothing else: **read the docs, use `curl` or `httpx` to exercise the API, and log the raw request/response when things fail.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 HTTP: Methods, Status Codes, Headers

**Methods (verbs):**

| Method  | Purpose                       | Safe? | Idempotent? | Body?         |
|---------|-------------------------------|:-----:|:-----------:|:-------------:|
| GET     | Read a resource               |   ✓   |     ✓       | no            |
| HEAD    | Metadata only (no body)       |   ✓   |     ✓       | no            |
| POST    | Create / non-idempotent action|       |             | yes           |
| PUT     | Full replacement              |       |     ✓       | yes           |
| PATCH   | Partial update                |       |    (usually)| yes           |
| DELETE  | Delete a resource             |       |     ✓       | opt           |
| OPTIONS | Discover allowed methods (CORS)|  ✓   |     ✓       | no            |

*Safe* = does not modify state.
*Idempotent* = repeating the request has the same effect as making it once.

**Status codes worth memorizing:**

- **200 OK** — request succeeded, body has the result.
- **201 Created** — created a resource; `Location` header points to it.
- **202 Accepted** — accepted for async processing.
- **204 No Content** — succeeded, no body (common for `DELETE`).
- **301 / 308 Moved Permanently** — resource relocated (308 preserves method/body).
- **302 / 307 Found / Temporary Redirect** — temporary.
- **304 Not Modified** — conditional GET; use your cached copy.
- **400 Bad Request** — malformed request.
- **401 Unauthorized** — missing/invalid credentials.
- **403 Forbidden** — authenticated but not allowed.
- **404 Not Found** — no such resource.
- **405 Method Not Allowed** — route exists but not for this method.
- **409 Conflict** — state conflict (duplicate key, version mismatch).
- **422 Unprocessable Entity** — validation failed (common in modern APIs).
- **429 Too Many Requests** — rate limited; `Retry-After` header.
- **500 Internal Server Error** — server bug.
- **502 Bad Gateway / 503 Service Unavailable / 504 Gateway Timeout** — upstream issues, often transient.

**Headers you'll use every day:** `Content-Type`, `Accept`, `Authorization`, `User-Agent`, `Cache-Control`, `ETag`, `If-None-Match`, `Retry-After`, `Location`, `Content-Length`.

---

### 2.2 JSON: Types, Encoding, and Schemas

**JSON types:** `null`, `boolean`, `number`, `string`, `array`, `object`. Objects are unordered maps of `string → value`.

**Wire example:**

```json
{
  "user_id": 42,
  "name": "Brian",
  "premium": true,
  "roles": ["admin", "editor"],
  "created_at": "2025-01-15T09:30:00Z",
  "address": null
}
```

**Gotchas you must know:**

- **JSON numbers have no integer/float distinction.** A large integer (e.g., a 64-bit ID) may lose precision when parsed as a JS `Number` (double-precision float, only 53 significant bits). Common fix: send large IDs as strings.
- **JSON has no date type.** Convention is ISO-8601 strings (`"2025-01-15T09:30:00Z"`). Parsing is your responsibility.
- **JSON has no comment syntax.** Configuration files that need comments should use JSON5, YAML, TOML, or JSONC (an extension).
- **UTF-8** is the standard encoding.
- **Trailing commas are invalid** in strict JSON.

**Python translation (`json` stdlib):**

```python
import json

s = json.dumps({"a": 1, "b": [2, 3]})   # → '{"a": 1, "b": [2, 3]}'
d = json.loads(s)                        # → {'a': 1, 'b': [2, 3]}

# Custom types (datetime, Decimal, ...) need custom encoders:
from datetime import datetime
class DTEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

json.dumps({"now": datetime.utcnow()}, cls=DTEncoder)
```

**Schema tools:**

- **JSON Schema** — the interoperable standard. Verbose but portable.
- **Pydantic** (Python) — Pythonic model classes with validation and JSON I/O.
- **OpenAPI 3.x** — describes an entire HTTP API (paths, methods, schemas, security). FastAPI generates it automatically.

**Command-line tool: `jq`.** Learn it.

```bash
curl -s https://api.example.com/users | jq '.[] | select(.premium) | {id, name}'
```

---

### 2.3 REST-ish API Design

REST as originally described by Roy Fielding is stricter than what most teams practice. In industry, **"REST-ish"** or **"resource-oriented JSON over HTTP"** is what you'll build and consume.

**Resource design:**

- Nouns, not verbs, in URLs. `/users`, not `/getUsers`.
- Nesting reflects containment: `/users/42/orders` for orders belonging to user 42.
- **Collections are plural**, **items are singular via ID**.
- **Filters, sorts, pagination** live in query params: `/orders?status=paid&sort=-created_at&limit=20`.
- **Versioning** — put the version in the path (`/v1/users`) or in a header. Path is more discoverable.

**Common patterns:**

- **Pagination** — three main styles:
  - Offset/limit (`?offset=100&limit=25`) — simple; drifts on inserts.
  - Cursor-based (`?cursor=abc123`) — stable across inserts; harder to jump around.
  - Page number (`?page=5&per_page=25`) — friendly UI; same drift issues.
- **Filtering** — `?status=paid` or richer `?filter[status]=paid`.
- **Sparse fieldsets** — `?fields=id,name` to shrink payloads.
- **Rate limiting** — `X-RateLimit-Remaining` / `X-RateLimit-Reset` headers.
- **Error envelope** — standard shape, e.g.:

  ```json
  {
    "error": {
      "code": "insufficient_funds",
      "message": "Balance is 5.00, needed 12.00",
      "request_id": "req_9f8c2..."
    }
  }
  ```

- **Idempotency keys** — pass a client-generated UUID on `POST` requests that must be safely retried; the server dedupes by key.

**Alternatives worth knowing (but not the default):**

- **GraphQL** — one endpoint, clients declare the shape they want. Solves over/underfetching.
- **gRPC** — binary protocol over HTTP/2 with Protobuf. High-throughput inter-service.
- **Server-Sent Events / WebSockets** — for push/streaming.

---

### 2.4 Consuming APIs from Python

**`httpx` is the modern choice** (sync + async, HTTP/2, timeouts, connection pooling). `requests` is still fine for simple scripts.

**Basic pattern with a client (reuses connections):**

```python
import httpx

BASE = "https://api.example.com/v1"

with httpx.Client(
    base_url=BASE,
    timeout=httpx.Timeout(5.0, connect=2.0),
    headers={"Authorization": f"Bearer {TOKEN}",
             "User-Agent": "my-service/1.0"},
) as client:
    r = client.get("/users", params={"limit": 100})
    r.raise_for_status()
    users = r.json()
```

**Timeouts (always set them).** Default `None` means "wait forever" — a great way to hang your program.

**Retries with exponential backoff.** Use `tenacity`:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)
def get_user(client, uid):
    r = client.get(f"/users/{uid}")
    r.raise_for_status()
    return r.json()
```

**Backoff math.** Exponential backoff waits $t_n = c \cdot b^{n-1}$ seconds on attempt $n$ (base $b > 1$). Add **jitter** — a random factor — to spread retries across clients so you don't stampede a recovering server:

$$t_n = \text{Uniform}(0, c \cdot b^{n-1})$$

**Pagination in a generator:**

```python
def iter_all_users(client):
    cursor = None
    while True:
        r = client.get("/users", params={"cursor": cursor, "limit": 100})
        r.raise_for_status()
        page = r.json()
        yield from page["data"]
        cursor = page.get("next_cursor")
        if not cursor:
            return
```

**Async concurrency** for lots of independent calls:

```python
import asyncio, httpx

async def fetch(client, url):
    r = await client.get(url)
    r.raise_for_status()
    return r.json()

async def main(urls):
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*(fetch(client, u) for u in urls))
```

Do **not** open one client per call. Reuse.

---

### 2.5 Authentication & Security Basics

**Common auth models:**

- **API key** — `Authorization: Bearer <token>` header, or a query param (less secure).
- **HTTP Basic** — `Authorization: Basic <b64(user:pass)>`. Only over HTTPS.
- **OAuth 2.0 / OIDC** — token-based, with flows for humans (authorization code) and machines (client credentials).
- **JWT (JSON Web Token)** — self-contained, signed token. Verify signature and expiry; treat unsigned JWTs as untrusted.
- **mTLS** — mutual TLS with client certs, common in service-to-service in strict environments.

**Security must-dos:**

- Always use HTTPS.
- Never log request bodies or `Authorization` headers.
- Rotate keys; support key expiry.
- Validate all input on the server; don't trust the client.
- Set a **short default timeout** on outbound calls.
- Follow least privilege: scope tokens to the minimum needed.

**CORS (browser-side).** If your API is called from a browser on a different origin, the server must send `Access-Control-Allow-*` headers. `curl` and Python clients don't care; browsers do.

---

## 3. Mental Models & Analogies

### 3.1 The "Restaurant Order Ticket" Model (HTTP Request/Response)

An HTTP request is a **ticket** you slip through the kitchen window:

- **Method** — the verb ("bring", "replace", "remove").
- **URL** — which dish / which table.
- **Headers** — special instructions ("no dairy", "for the birthday").
- **Body** — the actual details of the dish (only when the verb needs it).

The kitchen returns a **response ticket** back:

- **Status code** — did the order succeed, or did something go wrong?
- **Headers** — meta info (time it took, receipt number).
- **Body** — the plate (or an explanation of what went wrong).

The kitchen has no memory of you between orders — each ticket must include your table (auth). If the waiter drops a ticket, you slip in another one; because most orders are idempotent ("bring me table 7's status"), duplicates don't matter. But "order me another latte" (`POST`) needs an idempotency key or you get two lattes.

### 3.2 The "Postal System" Model (JSON as Envelope)

JSON is the standard **envelope** for shipping data across the wire. It defines only the shape (rectangle, dimensions, addressing). It says nothing about *what's inside* — that's the schema, agreed upon by sender and receiver.

- **No native date** — like postal mail: you write dates in your own format inside the letter. Everyone has to agree on ISO-8601.
- **No native binary** — you base64-encode blobs, like sending a photo as its printed hex codes.
- **No comments** — the envelope is machine-processed; annotations go elsewhere.
- **Schemas (Pydantic, OpenAPI, JSON Schema)** are the shipping labels and customs declarations — they let both parties agree on what's inside without opening every envelope.

> 🖼️ **Image Prompt [IMG-API-01]:** *"A clean split illustration: LEFT shows an HTTP request as a paper ticket with fields 'GET /users/42', 'Authorization: Bearer ...', 'Accept: application/json'; RIGHT shows a JSON response envelope popping out of a mailbox, its contents rendered as key-value pairs. Arrow between them labeled 'HTTPS'. Modern flat vector, teal + orange, textbook style."*
> **Caption:** Request as a ticket; response as an envelope of JSON.
> **Placement:** Section 2.1 / Mental Models.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "I'll Just Retry Every Failure"

Retrying `POST /charge_customer` naively can charge the customer twice. Rules:

- **Retry only idempotent verbs** (`GET`, `PUT`, `DELETE`) or requests carrying an **idempotency key**.
- **Retry only on transient errors** — network timeouts, `502`, `503`, `504`, sometimes `429`. Never retry `400`, `401`, `403`, `404`.
- Use **exponential backoff with jitter**, not fixed delays.
- Cap total retries; a runaway retry loop DDoSes the recovering service.

### 4.2 "Status Codes Are Optional"

Some APIs return `HTTP 200` with `{"error": "not found"}` inside the body. This forces every client to double-check both the status code *and* the body. It also makes `raise_for_status()` useless and hides errors from HTTP-level monitoring. The right pattern is: **status code carries the error class; body carries the details**. `200` implies success.

### 4.3 "JSON Numbers Are Safe for IDs"

`{"id": 9007199254740993}` may become `9007199254740992` after a round-trip through a JS/JSON parser using 64-bit floats. If your IDs might exceed $2^{53} \approx 9.007 \times 10^{15}$, send them as **strings** (`"id": "9007199254740993"`). This applies to Twitter/X snowflake IDs, some database sequences, and many UUID-encoded numbers.

---

## 5. Self-Assessment Bank (APIs & JSON)

### Questions

**Q1 (Short answer).** Which HTTP methods are considered idempotent, and why does idempotency matter?

**Q2 (Multiple choice).** A `POST /orders` succeeds but the response never reaches your client (network dropped). If you retry blindly, what could go wrong, and what is the standard fix?
A. Nothing; POSTs are safe to retry.
B. You may create duplicate orders; fix with an **idempotency key** header the server dedupes on.
C. You will get a `409 Conflict`; ignore it.
D. Retry with `PUT` instead.

**Q3 (Short answer).** Explain the difference between `401 Unauthorized` and `403 Forbidden`.

**Q4 (Multiple choice).** JSON has how many primitive/composite types?
A. 4
B. 6
C. 8
D. 10

**Q5 (Short answer).** Why can a 64-bit integer ID lose precision when passed through JavaScript?

**Q6 (Multiple choice).** What is the correct pagination approach for a feed where new items are inserted at the top constantly?
A. Offset/limit — simplest and stable.
B. Cursor-based — stable across inserts.
C. Page numbers — friendly UI, no drift.
D. Random sampling.

**Q7 (Short answer).** What is the purpose of the `Retry-After` header? Which status codes commonly include it?

**Q8 (Multiple choice).** When designing a REST-ish endpoint to "list a user's paid orders", the best URL is:
A. `GET /getUserPaidOrders?id=42`
B. `POST /users/42/orders/paid`
C. `GET /users/42/orders?status=paid`
D. `GET /orders/user/42/paid`

**Q9 (Short answer).** Give the formula for exponential backoff with full jitter and explain why jitter is important.

**Q10 (Multiple choice).** Why is `Content-Type: application/json; charset=utf-8` a good default header?
A. It's required by RFC 8259.
B. It tells the server how to parse the body and avoids encoding ambiguity.
C. It makes requests faster.
D. It enables gzip compression.

---

### Answer Key & Detailed Explanations

**A1.** Idempotent methods: `GET`, `HEAD`, `PUT`, `DELETE`, `OPTIONS`, `TRACE`. `PATCH` is sometimes idempotent depending on semantics. `POST` is **not** idempotent by default. Idempotency matters because networks drop responses, retries are inevitable, and idempotent endpoints let you retry without side effects.

**A2. B.** Blind retry can produce duplicate orders because `POST` is not idempotent. The standard fix is a client-generated **idempotency key** (`Idempotency-Key: <uuid>`) that the server stores; retries with the same key return the original result rather than creating a new order.

**A3.** `401` means the request lacks valid credentials (you haven't proven who you are). `403` means the server knows who you are but refuses (you're authenticated but not authorized for this resource/action). Mnemonic: 401 = "who?", 403 = "no."

**A4. B.** Six: `null`, `boolean`, `number`, `string`, `array`, `object`.

**A5.** JavaScript uses IEEE-754 double-precision floats for all numbers. Doubles have a 53-bit mantissa, so integers beyond $2^{53} \approx 9.007 \times 10^{15}$ can no longer be represented exactly. JSON parsers in JS convert numbers to `Number`, so `9007199254740993` becomes `9007199254740992`. The workaround is to send large IDs as strings.

**A6. B.** Cursor-based pagination survives concurrent inserts because the cursor names a specific position (e.g., "the item right after this timestamp+id"). Offset/limit shifts as new rows appear at the top — you get duplicates on page 2. Page numbers have the same issue.

**A7.** `Retry-After` tells the client how long to wait before retrying. It appears in `429 Too Many Requests` and `503 Service Unavailable`. It can be a delay in seconds (`Retry-After: 30`) or an HTTP-date. Well-behaved clients honor it.

**A8. C.** Nouns, not verbs; resources nested by containment; filters as query params. This is the idiomatic REST-ish shape.

**A9.** Formula (full jitter): $t_n = \text{Uniform}(0, c \cdot b^{n-1})$, where $c$ is the base delay and $b$ is the growth factor (commonly 2). Jitter is critical because without it, thousands of clients whose requests all failed at the same time will *all* retry at the same time — hammering the recovering service in perfectly synchronized waves.

**A10. B.** `Content-Type` tells the receiver how to interpret the body. `application/json` signals JSON, and `charset=utf-8` avoids ambiguity for older parsers. Some receivers reject requests without `Content-Type`.

---

## 6. Practice Prompts

1. Pick a public API (GitHub, USGS, OpenWeatherMap). Read the docs; use `curl` and `jq` to fetch and transform a dataset; then rewrite the same script in `httpx`.
2. Add retry-with-jitter to a script that hits a rate-limited API. Verify it doesn't stampede under simulated failures.
3. Design a REST-ish spec for a "bill drafting" service: routes, methods, query params, error shape, pagination. Write it as OpenAPI YAML.
4. Read the raw HTTP request/response with `curl -v` for a `GET` and a `POST`. Explain each header line.
5. Write a Pydantic model that validates a JSON payload with a datetime, a nullable field, a discriminated union, and a strict integer.

---

## 7. References

- MDN Web Docs — HTTP overview, methods, status codes
- Roy Fielding, *Architectural Styles* (dissertation) — REST's original definition
- Zalando's *RESTful API Guidelines* (opinionated but excellent)
- Stripe/GitHub API docs — canonical examples of well-designed REST APIs
- Michael Kerrisk's HTTP chapters in *The Linux Programming Interface* (for the network layer)
