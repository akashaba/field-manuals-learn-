# 12 — Authentication & Authorization

> **Module goal:** Move past `x-api-key` header hacks to real production auth. Master API keys, JWTs, OAuth 2.0 / OIDC, session cookies, multi-tenant scoping, and the AuthN vs AuthZ distinction. Understand what breaks each, how to store secrets, and how to layer identity through your AI service so every LLM call carries a verifiable actor.

---

## 1. Executive Summary & Core Concepts

Two distinct concerns:
- **Authentication (AuthN)** — *who is this?* You verify a claim of identity.
- **Authorization (AuthZ)** — *what may they do?* You check permissions on a resource.

Confusing them is one of the most common security bugs in AI services: "the JWT is valid, so allow anything" skips AuthZ entirely.

**Four common AuthN patterns for AI services:**

| Pattern | Used for | Pros | Cons |
|---------|----------|------|------|
| **API keys** | Machine-to-machine, developer keys | Simple; long-lived; easy to revoke | Long-lived = high blast radius if leaked; no user identity |
| **JWT (bearer token)** | Frontend↔backend, service↔service | Stateless verification; carries claims | Revocation is hard; leaked = valid until expiry |
| **OAuth 2.0 / OIDC** | Third-party sign-in, delegated access | Industry standard; refresh tokens; scopes | Complex; needs an IdP (Auth0, Okta, Cognito, Azure AD B2C) |
| **Session cookies** | Web apps served from your domain | HttpOnly + Secure protects from XSS theft; revocable on server | Only fits same-origin web apps |

**AuthZ patterns:**
- **RBAC** (Role-Based) — user→roles→permissions. Simple, coarse.
- **ABAC** (Attribute-Based) — decision from attributes of user, resource, action, context. Fine-grained, complex.
- **Policy engines** — OPA (Open Policy Agent), Cedar (AWS), Casbin. Externalize AuthZ decisions.

**Multi-tenancy:** every request must be scoped to a tenant. The pattern:
1. Extract identity from AuthN (JWT, API key)
2. Resolve tenant from identity (`tenant_id` claim or DB lookup)
3. Attach `tenant_id` to request context
4. Every downstream query/action includes `WHERE tenant_id = $1` or its equivalent

Skipping step 4 anywhere = tenant leakage. This is a *systemic property* — one missed query in a 200-endpoint service is a breach.

**Secrets — the substrate of AuthN.** Everything above depends on: signing keys (JWT), API keys (upstream providers), OAuth client secrets, DB passwords. Storage: secret manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault, 1Password), NEVER git, NEVER `.env` in a Docker image. Environment variables at runtime OK; injected from a secret manager at deploy.

---

## 2. Deep-Dive Breakdown

### 2.1 API Keys — Simple but Dangerous

The simplest AuthN: client sends `Authorization: Bearer sk_live_<opaque>`; server looks it up in a table. Rules to make it safe:

1. **Generate cryptographically random**, at least 32 bytes: `secrets.token_urlsafe(32)`.
2. **Store hashed, not plaintext** (bcrypt or SHA-256 with per-key salt). If the DB leaks, keys aren't usable.
3. **Prefix by environment** (`sk_live_`, `sk_test_`) so leaks are obvious in log grepping.
4. **Emit a short prefix as `key_id`** for logs; never log the full key.
5. **Rotate on schedule and on suspicion.** Support multi-key per user during transition.
6. **Rate-limit per key.** (Module 10.)
7. **Detect leaked keys** — scan git commits, Slack, error reports. GitHub secret scanning is a partner ecosystem for exactly this.

```python
import secrets, hashlib
from sqlalchemy.orm import Session

def create_api_key(db: Session, user_id: str) -> tuple[str, str]:
    """Returns (plaintext_key_shown_once, key_id_for_ref)."""
    key = "sk_live_" + secrets.token_urlsafe(32)
    key_id = key[:12]  # short prefix for logs / user UI
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    db.execute("INSERT INTO api_keys (user_id, key_id, key_hash, created_at) VALUES (?,?,?,now())",
               (user_id, key_id, key_hash))
    return key, key_id

def verify_api_key(db: Session, presented: str) -> str | None:
    """Returns user_id if valid, else None."""
    key_hash = hashlib.sha256(presented.encode()).hexdigest()
    row = db.execute("SELECT user_id FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
                     (key_hash,)).fetchone()
    return row.user_id if row else None
```

### 2.2 JWT — Stateless Bearer Tokens

A JWT (JSON Web Token) has three base64url-encoded parts: `header.payload.signature`. The signature is HMAC-SHA256 (`HS256`) or RSA/ECDSA (`RS256`/`ES256`) over `header.payload`.

**Payload (claims):**
```json
{
  "iss": "auth.example.com",
  "sub": "user-abc123",
  "aud": "api.example.com",
  "exp": 1758934800,
  "iat": 1758931200,
  "tenant_id": "t-xyz",
  "roles": ["editor"],
  "scopes": ["read:docs", "write:chat"]
}
```

**Verification in FastAPI:**

```python
from fastapi import Depends, HTTPException, Header
from jose import jwt, JWTError

JWKS_URL = "https://auth.example.com/.well-known/jwks.json"
JWKS = fetch_jwks(JWKS_URL)  # cache, refresh periodically

def get_current_user(authorization: str = Header(...)) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer")
    token = authorization[7:]
    try:
        header = jwt.get_unverified_header(token)
        key = next(k for k in JWKS["keys"] if k["kid"] == header["kid"])
        claims = jwt.decode(
            token, key,
            algorithms=["RS256"],
            audience="api.example.com",
            issuer="https://auth.example.com",
        )
    except (JWTError, StopIteration):
        raise HTTPException(401, "invalid token")
    return claims

@app.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    if "write:chat" not in user["scopes"]:
        raise HTTPException(403, "missing scope")
    # user["tenant_id"] is now trusted; use it in every query
    ...
```

**JWT pitfalls to memorize:**
- **`alg: none` attack** — accept only allow-listed algorithms; never let the header dictate.
- **HS256 vs RS256 confusion** — if your verify code accepts both and uses the public key as an HMAC secret, an attacker signs an HS256 token with the public key. Always pin the algorithm.
- **No revocation** — a signed JWT is valid until expiry. Keep expiries **short (5–15 min)** and use **refresh tokens** stored server-side (revocable).
- **`aud` and `iss` checks** are mandatory — otherwise a token issued for service A is accepted by service B.
- **Clock skew** — allow 30–60s leeway on `exp` / `nbf`; enforce anything wider is a bug.

### 2.3 OAuth 2.0 / OIDC — The Standard for User Sign-in

**OAuth 2.0** is a delegation protocol: user grants your app access to their data on a resource server (Google, GitHub, Microsoft). **OIDC (OpenID Connect)** layers identity on top: instead of just delegated API access, you also get an `id_token` (a JWT) with authenticated user claims.

**Authorization Code Flow with PKCE** (modern, for SPAs and mobile):

1. Client generates `code_verifier` (random) and `code_challenge` = SHA-256(verifier)
2. Redirect user to IdP's `/authorize` with `client_id`, `redirect_uri`, `scope`, `state`, `code_challenge`
3. User signs in; IdP redirects back with `?code=...&state=...`
4. Client POSTs `code` + `code_verifier` to IdP's `/token`
5. Receives `access_token`, `id_token`, `refresh_token`
6. Access token → call your API; ID token → know the user's identity

Brian's own product (legmt.gov) uses Azure AD B2C via MSAL — same OIDC pattern. In your AI service backend, you validate the ID token as a JWT (Section 2.2) and use claims for AuthZ.

**Scopes** are your first line of authorization: `read:docs write:chat admin:users`. Emit only tokens with scopes the user actually needs; check scope on every endpoint.

**Refresh tokens** are long-lived credentials to obtain new access tokens without re-authenticating. Store in httpOnly Secure SameSite=Strict cookies (web) or Keychain/Keystore (mobile). NEVER localStorage — one XSS = full account takeover.

### 2.4 FastAPI Dependency Injection for AuthZ

Layer AuthN → AuthZ cleanly using `Depends`:

```python
from fastapi import Depends, HTTPException

def get_user(authorization: str = Header(...)) -> dict:
    return validate_jwt(authorization)  # from 2.2

def require_scope(*required: str):
    def check(user: dict = Depends(get_user)) -> dict:
        if not set(required).issubset(set(user["scopes"])):
            raise HTTPException(403, f"missing scope: {required}")
        return user
    return check

def get_tenant_context(user: dict = Depends(get_user)) -> str:
    tid = user.get("tenant_id")
    if not tid:
        raise HTTPException(403, "missing tenant")
    return tid

@app.post("/chat")
async def chat(req: ChatRequest,
               user: dict = Depends(require_scope("write:chat")),
               tenant: str = Depends(get_tenant_context)):
    # Every db call parameterized by `tenant`
    ...
```

The pattern gives you composable auth. Adding a new scope requirement is one dependency; the boilerplate is centralized.

### 2.5 Multi-Tenant Data Isolation

Beyond passing `tenant_id` through function calls, use **defense in depth**:

- **Row-level security (RLS)** in Postgres — the DB enforces `tenant_id = current_setting('app.tenant')` on every query. Even a buggy service can't cross tenants.
- **Separate schemas or databases per tenant** — highest isolation, hardest to operate.
- **Per-tenant encryption keys** — data at rest encrypted per-tenant; a breach of one tenant's key doesn't expose others.

For LLM apps specifically:
- **Prompt cache MUST be tenant-keyed** (Module 11)
- **Vector index MUST filter by tenant** — the RAG retrieval step is a common leak (`WHERE tenant_id = $1` on your vector store's metadata filter)
- **User content in system prompts must be sanitized** — a prompt injection can attempt to exfiltrate other tenants' data if the model has cross-tenant context

`[IMG-12-01]` — *Prompt: A layered authentication and authorization diagram for an AI service. Left column: three actors (browser SPA, native mobile app, ServerToServer client). Middle: an identity provider labeled "Auth0 / Cognito / Azure AD B2C" issuing JWTs. Right: the AI backend showing middleware layers — "1. JWT verification (issuer, audience, signature)", "2. Scope check (write:chat)", "3. Tenant resolution", "4. RBAC/ABAC policy", "5. Data-layer row-level security". Arrows showing a request traverses all five. Clean architecture diagram, neutral colors.*

---

## 3. Mental Models & Analogies

### Model 1: The Passport, Visa, and Boarding Pass

At an airport:
- **Passport** = your identity — issued by an authority (a government / an IdP). It says who you are. Verifiable via cryptographic features. That's authentication.
- **Visa** = authorization for a destination — granted by the destination country based on your passport. Says "you may enter for tourism, 90 days." That's authorization scoped to a resource.
- **Boarding pass** = ephemeral session credential for a specific flight — issued once you've cleared check-in, valid only for that flight. That's your access token.
- **Passport revocation** = kill switch at the government level; **visa cancellation** = at the destination level; **boarding pass invalidation** = at the airline level. Each has different revocation cost and speed.

A JWT is a boarding pass: self-contained, verifiable at the gate without radioing HQ, but useless to invalidate mid-flight — which is why they're short-lived. A refresh token is closer to your visa: presented periodically to get new boarding passes.

### Model 2: The Building's Access Card

A modern office building has:
- **Turnstile at the lobby** — reads your card, checks you're a valid employee (AuthN)
- **Elevator authorization** — your card unlocks certain floors, not others (RBAC)
- **Room keycard reader** — some rooms need extra badge scan (ABAC — you're an employee AND on the approved list for that server room)
- **Time-of-day rules** — accountants can access their floor 24/7; contractors only 9-5 M-F (ABAC with context)
- **Audit log** — every scan recorded; investigations can trace who was where when
- **Revocation** — deprovisioning fires the card immediately for all future scans

Notice the layers are independent. Passing the turnstile doesn't grant floor access. Getting off the elevator doesn't grant server-room entry. This is defense in depth. In your AI service, every hop (edge → API → LLM → DB) checks its own layer of authorization; no single failure grants full access.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Trusting client-supplied `tenant_id` or `user_id`.**
Endpoint reads `body.tenant_id` or `?user_id=` from the request. Attacker changes it, sees another user's data. **Fix:** identity claims come from the *verified* token (JWT `sub`, `tenant_id` claims that the IdP signed), never from a query param or body field. If you need to reference a target user_id in a path (e.g., `/users/{user_id}/settings`), verify the authenticated caller is authorized for that target (self, admin, or same-tenant).

**Pitfall #2 — JWT `alg: none` or algorithm confusion.**
Older JWT libraries accept `alg: "none"` (no signature) or accept whatever algorithm the token's header says. Attacker crafts `{"alg":"none"}.payload.` with no signature; naive verifier accepts. Or: your app is configured with an RS256 public key, but the library also accepts HS256, so attacker signs an HS256 token using the public key bytes as the HMAC secret — and the verifier "checks" it correctly. **Fix:** pin `algorithms=["RS256"]` (or your specific choice) in every `jwt.decode` call. Use a maintained library (`python-jose`, `PyJWT`) that has these attacks patched. Test negatively: try to authenticate with `alg:none` and confirm it's rejected.

**Pitfall #3 — Logging tokens, keys, or full authorization headers.**
Structured logs contain `authorization: Bearer eyJhbG...`; the logs go to a SaaS aggregator; anyone with logs access has valid tokens. Also common: pasting a curl command in a Slack ticket. **Fix:** logging middleware redacts `Authorization`, `Cookie`, `x-api-key`, and any known secret headers *before* the log line is written (Module 6). Never `print(request.headers)`. In error handlers, ensure exception messages don't include the token from parse errors. And treat any leaked-token report as a P0 rotation event.

---

## 5. Self-Assessment Bank

**Q1 (MC):** Authentication answers "who are you?"; authorization answers:
A) When are you?
B) Where are you?
C) What may you do?
D) Why are you?

**Q2 (short):** Explain three JWT-specific attacks and one mitigation for each.

**Q3 (MC):** In OAuth 2.0 Authorization Code Flow with PKCE, the purpose of `code_verifier` and `code_challenge` is to:
A) Encrypt the authorization code in transit
B) Bind the authorization code to the specific client that started the flow, mitigating code interception
C) Add a second factor for the user
D) Replace the need for client_secret entirely

**Q4 (short):** Your API accepts a JWT and reads `tenant_id` from the token claims. Elsewhere, a POST endpoint accepts `body.tenant_id` and uses it in a query. Explain the vulnerability.

**Q5 (MC):** Which of the following should be stored HASHED, not plaintext, in your DB?
A) User passwords
B) API keys
C) OAuth client secrets
D) Both A and B (client secrets you hold as the OAuth client are used as-is)

**Q6 (MC):** A JWT expiry (`exp`) should typically be:
A) Never — tokens should be permanent
B) 5–15 minutes for access tokens; refresh tokens for long-term
C) Exactly 1 hour by IETF mandate
D) The user's session length

**Q7 (short):** Design the auth flow for a multi-tenant AI SaaS where enterprises sign in via their own SSO (SAML/OIDC IdP) and their users hit your API. Where does `tenant_id` come from?

**Q8 (MC):** RBAC vs ABAC — which is more appropriate for "editors of a document can share it only during business hours in the user's timezone"?
A) RBAC
B) ABAC
C) Neither; use ACLs
D) Both work equally

**Q9 (MC):** You detect a leaked API key posted in a public GitHub gist. Your response order should be:
A) Log a ticket, discuss in next sprint
B) Rotate the key immediately, revoke the old one, audit its usage
C) Rate-limit the key
D) Ask the user to delete the gist

**Q10 (short):** Explain row-level security (RLS) in Postgres and why it's valuable for a multi-tenant AI service.

---

### Answer Key

**A1: C.** AuthN = identity ("who"); AuthZ = permission ("what allowed"). They are separate and independently reviewable: fixing AuthN doesn't fix AuthZ (a correctly-identified user still needs their scope checked).

**A2:** Three attacks + mitigation each:
1. **`alg: none` / algorithm-confusion** — attacker specifies `alg` in the header that your library naively honors. Mitigation: **pin allowed algorithms** in the decode call; use a modern library; test negatively.
2. **Token replay / stolen bearer** — attacker steals a JWT (XSS, exposed logs, MITM) and uses it. Mitigation: **short expiries + refresh tokens**; HttpOnly Secure cookies for browser tokens; bind tokens to client fingerprint (device DPoP) for high-value ops; monitor for anomalous use.
3. **Missing `aud`/`iss` check** — token from another service accepted by yours. Mitigation: **enforce audience and issuer** in every `decode` call; give each service a distinct audience string.
4. **JWKS trust** — attacker publishes their own JWKS at a URL your service naïvely fetches. Mitigation: **pin JWKS URL and issuer**; only trust JWKS from your IdP's `.well-known/jwks.json` over TLS.
5. **Long-lived tokens with sensitive claims** — a leaked long-life token exposes admin permissions until expiry. Mitigation: **least privilege in claims**, and **short expiries** for scope-heavy tokens.

**A3: B.** PKCE mitigates the "authorization code interception" attack: on native/mobile, the redirect URI may be intercepted by another app registered for the same scheme; the code alone isn't enough because exchanging it for tokens requires `code_verifier`, which only the original client holds. It doesn't encrypt the code (A), isn't a 2FA mechanism (C), and doesn't inherently replace client_secret (D), though it does enable public clients to safely omit client_secret.

**A4:** The endpoint that reads `body.tenant_id` is trusting **client-supplied identity data.** Attacker with a valid JWT for tenant A calls the endpoint with `body.tenant_id = "tenant-B"`; the query runs against tenant B's data; attacker sees another organization's records. Fix: never accept `tenant_id` from the request body/params — always use the value from the verified JWT claims. If the endpoint must operate on a different tenant (admin flows), verify the authenticated caller has an explicit permission to do so, and log it.

**A5: D.** Store user passwords and API keys hashed. For passwords use a slow, salt-inclusive algorithm (Argon2id / bcrypt / scrypt). For API keys, SHA-256 is acceptable (they're already high-entropy random). OAuth client secrets are typically stored plaintext (or encrypted at rest with envelope encryption) because *you use them to sign requests*; you can't hash something you need to send. But you protect them in a secret manager and never log them.

**A6: B.** Access tokens: 5–15 minutes; refresh tokens: hours to weeks depending on sensitivity. Short-lived access tokens minimize damage from theft. Refresh tokens live longer but are revocable server-side (they're stored per-user in your DB and consulted on every refresh). Some sensitive ops (funds transfer, permission changes) even require *re-authentication* rather than just a valid access token — "step-up auth."

**A7:** Design:
1. Enterprise admin configures their IdP's OIDC issuer URL and their org's `tenant_id` in your product's admin console.
2. When users sign in, your app's login page routes them to their enterprise IdP (based on email domain or explicit workspace selection).
3. IdP authenticates and returns an OIDC `id_token` to your app.
4. Your backend validates the token (issuer matches configured enterprise IdP, signature verifies against IdP's JWKS, audience is your app).
5. Your backend either uses the `tenant_id` claim (if IdP sends it) or looks up `tenant_id` by matching the token's `iss` to your DB of enterprise integrations.
6. Your backend mints its **own** short-lived JWT (5-min access) with `sub`, `tenant_id`, and `scopes` — this is what the frontend uses for all subsequent API calls. Enterprise IdP tokens are never sent to your API endpoints (loose coupling; you control claim structure).
7. Refresh via your own refresh token (revocable).
Key point: **`tenant_id` comes from your side of the trust boundary** (either you set it based on the enterprise integration record, or the enterprise IdP claims it but you validate `iss ↔ tenant_id` mapping).

**A8: B.** RBAC gives coarse role→permission mapping (editor role has share permission). But "only during business hours in the user's timezone" is a **contextual attribute** — RBAC can't express it. ABAC decides based on user attributes, resource attributes, action, and *environment context* (time, location, device). Use a policy engine (OPA, Cedar) or a rule DSL. Note that RBAC + ABAC is often the sweet spot: roles for the coarse permission ("editor"), ABAC layered on top for contextual constraints ("editor AND now ∈ business_hours(user.tz)").

**A9: B.** Priority order for a leaked key:
1. **Rotate** — issue a new key.
2. **Revoke** the old one — kill it in your DB right now.
3. **Audit** — query logs for all uses of the old key: source IPs, endpoints, timing. Determine whether unauthorized use occurred.
4. **Notify** the affected customer and, depending on data accessed, internal security / legal / regulator per your incident response plan.
5. **Post-mortem** — how did it leak? (Committed to git? Pasted in Slack? Screenshotted?) Add controls: pre-commit hooks that detect key prefixes, GitHub secret scanning, DLP on messaging.
Asking the user to delete the gist (D) doesn't help — GitHub's mirrors, Google Cache, Wayback Machine, and any bot that already scraped it still have the key. Rotation is the only remediation.

**A10:** **Row-level security (RLS)** is a Postgres feature where you attach policies to a table so that SELECT/INSERT/UPDATE/DELETE only see rows matching a predicate — evaluated against session variables. Typical pattern:
```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents
  USING (tenant_id = current_setting('app.tenant_id')::uuid);
```
Your app sets `SET app.tenant_id = '<from JWT>'` at the start of each request/transaction. Every subsequent query, no matter how it's written, automatically filters by tenant. **Value for AI services:** a bug in a service (missing `WHERE tenant_id = ?`) can no longer produce cross-tenant leakage — the DB refuses to return the wrong rows. It's defense in depth, cheap once configured, and dramatically reduces the surface area of AuthZ errors in complex systems. Similar features: MySQL security definer procs; MongoDB $expr with $$ROOT; Firestore rules.

---

**Related modules:**
- `learning/13-security.md` — the broader security picture
- `learning/10-rate-limiting.md` — auth-tied rate-limits
- `learning/09-llm-observability.md` — attach hashed user_id to traces
- `learning/06-logging.md` — redact tokens from logs

**Practice prompts:**
1. Wire an OIDC IdP (Auth0 / Cognito / Keycloak) with your Month 4 or 5 build. Frontend gets a JWT; backend verifies via JWKS.
2. Add scope checks on your endpoints. Test with a token missing the required scope; confirm 403.
3. Enable Postgres RLS on your main table. Attempt a cross-tenant query; confirm zero rows.
4. Write a `create_api_key` + `verify_api_key` pair with hashed storage. Test end-to-end.
5. Deliberately introduce a JWT `alg: none` attempt against your service; confirm rejection.

**References:**
- OWASP JWT Cheat Sheet — https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html
- Auth0 blog — OIDC, PKCE, flows
- Aaron Parecki — "OAuth 2.0 Simplified" (free online book)
- Postgres docs — Row Security Policies
- OWASP Authentication Cheat Sheet
