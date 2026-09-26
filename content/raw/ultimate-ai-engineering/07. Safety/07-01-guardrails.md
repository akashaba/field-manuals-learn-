# 07.01 — Guardrails

> **Module 7: Safety** · Subtopic 1 of 3
> **Prerequisites:** 02.03 (validation layers), 04.03 §5 (calibration), 05.01 (tool runtime), 05.06 (approvals, escalation economics), 06.01–06.02 (evals, datasets).
> **Outcome:** you can design, implement, evaluate, and operate **runtime guardrails**: layered controls on inputs, retrieval, actions, and outputs, expressed as versioned policy. You can choose classifiers and thresholds by cost, guard streaming outputs without destroying latency, verify groundedness, and measure guardrails like any other model: precision, recall, over-refusal, and latency.

> **Scope:** this subtopic covers *policy enforcement around model behaviour*. 07.02 covers system security (threat modelling, supply chain, output handling). 07.03 goes deep on prompt injection. Guardrails are one control among several — they are **not** a security boundary by themselves (see 07.03 on why classifiers alone fail against adaptive attackers).

---

## 1. What Guardrails Are (and Aren't)

**Guardrails** are runtime checks that enforce a deployment's policy on what goes into the model, what it can do, and what comes out. They complement, not replace:
- the model's own trained safety behaviour;
- a secure architecture (least privilege, isolation — 07.02, 07.03);
- human oversight (05.06).

```
 user / upstream ──►┌──────────────┐   ┌───────────────┐   ┌───────────────┐   ┌──────────────┐──► user
                    │ INPUT RAILS  │──►│ CONTEXT RAILS │──►│ ACTION RAILS  │──►│ OUTPUT RAILS │
                    │ authN/quota  │   │ source trust  │   │ tool policy   │   │ PII leakage  │
                    │ PII redact   │   │ ACL filtering │   │ arg validation│   │ policy class.│
                    │ topic/scope  │   │ injection scan│   │ approvals     │   │ groundedness │
                    │ injection    │   │ (07.03)       │   │ egress rules  │   │ format/schema│
                    │ detection    │   │               │   │ budgets       │   │ disclaimers  │
                    └──────┬───────┘   └──────┬────────┘   └──────┬────────┘   └──────┬───────┘
                           └──────────────────┴── audit log, metrics, human review queue ─┘
```

**Design principles:**
1. **Policy first, tooling second.** Write down what the system must never do, must always do, and must escalate — per deployment.
2. **Deterministic where possible.** Regexes, schemas, allowlists, and permission checks are cheap and unbypassable. Use classifiers only for what rules can't express.
3. **Layered, not monolithic.** Independent layers catch different failures (§4).
4. **Measured like a model.** Every guardrail has precision, recall, and latency, measured on an eval set (§6).
5. **Actions, not just verdicts:** block, redact, rewrite, warn, escalate, or log — chosen per category and severity.

---

## 2. Policy as Code

### 2.1 From policy document to enforceable rules

For a legislative-services assistant, an illustrative policy might read:

| Category | Rule | Stage | Detector | Action | Severity |
|---|---|---|---|---|---|
| Constituent PII | Never reveal personal contact or ID data from records | Output | Regex + NER | Redact | High |
| Confidential drafting requests | Don't disclose unreleased draft requests to unauthorised users | Context + output | ACL filter + classifier | Block + log | High |
| Legal advice to the public | Don't give individualised legal advice; provide information with a referral | Output | Classifier | Rewrite with disclaimer | Medium |
| Political neutrality | Bill summaries must not advocate for or against | Output | Rubric judge (sampled) + prompt rules | Warn / regenerate | Medium |
| Out of scope | Unrelated tasks (e.g. personal errands) | Input | Topic classifier | Polite refusal | Low |
| Ungrounded legal claims | Every legal claim must cite a source in context | Output | Groundedness check (§5) | Remove claim / regenerate | High |
| Destructive actions | Filing, sharing, publishing | Action | Tool annotations | Approval token (05.06) | High |

**Version the policy** like code: rule IDs, owners, changelog, and tests. The policy file is the contract between legal/compliance and engineering.

### 2.2 A minimal policy engine

```python
import re
import time
from dataclasses import dataclass, field
from typing import Callable

ACTION_ORDER = {"allow": 0, "log": 1, "warn": 2, "redact": 3, "rewrite": 4, "escalate": 5, "block": 6}


@dataclass
class Rule:
    id: str
    stage: str                                      # input | context | action | output
    check: Callable[[str, dict], tuple[bool, str]]  # (violated?, detail)
    action: str
    severity: str = "medium"
    fail_closed: bool = False                       # if the detector errors/times out: block (True) or allow (False)
    timeout_s: float = 0.2


@dataclass
class Decision:
    action: str
    hits: list[dict] = field(default_factory=list)
    text: str | None = None


class PolicyEngine:
    def __init__(self, rules: list[Rule], redactor: Callable[[str], str] | None = None):
        self.rules, self.redactor = rules, redactor

    def evaluate(self, stage: str, text: str, ctx: dict) -> Decision:
        hits, worst = [], "allow"
        for r in (r for r in self.rules if r.stage == stage):
            t0 = time.perf_counter()
            try:
                violated, detail = r.check(text, ctx)
                if time.perf_counter() - t0 > r.timeout_s:
                    raise TimeoutError("detector exceeded budget")
            except Exception as e:                  # detector failure: apply the fail mode explicitly
                violated, detail = r.fail_closed, f"detector error: {type(e).__name__}"
            if violated:
                hits.append({"rule": r.id, "action": r.action, "severity": r.severity, "detail": detail})
                if ACTION_ORDER[r.action] > ACTION_ORDER[worst]:
                    worst = r.action
        out = text
        if worst == "block":
            out = None                                   # never hand blocked content to the caller
        elif worst == "redact" and self.redactor:
            out = self.redactor(text)
        return Decision(worst, hits, out)


SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def pii_check(text: str, ctx: dict) -> tuple[bool, str]:
    found = [n for n, p in (("ssn", SSN), ("email", EMAIL)) if p.search(text)]
    return bool(found), ",".join(found)


def redact_pii(text: str) -> str:
    return EMAIL.sub("[EMAIL]", SSN.sub("[SSN]", text))
```

**Fail-open vs fail-closed** is a *per-rule* decision. PII leakage and destructive-action checks fail **closed**, meaning a detector outage blocks. Low-severity topic checks fail **open**, so the product keeps working. Log every detector failure, because an unmonitored fail-open rule is a silent hole.

---

## 3. Detectors

| Detector | Use for | Notes |
|---|---|---|
| **Rules** (regex, allowlists, schemas, permission checks) | PII formats, identifiers, URLs/domains, schema validity, tool permissions | Deterministic, fast, unbypassable for what they cover |
| **NER / PII models** (e.g. Presidio + spaCy/transformers) | Names, addresses, free-form PII | Tune per domain; public officials' names aren't PII in legislative text |
| **Safety classifiers** (Llama Guard family, provider moderation / content-safety APIs, fine-tuned small models) | Harm categories, jailbreak and injection attempts | Map their taxonomy to *your* policy; calibrate thresholds on your data |
| **LLM-as-guard** (a prompted rubric judge with structured output) | Nuanced, deployment-specific policies (neutrality, legal-advice boundary) | Slow and costly; validate like any judge (06.01 §4); use on sampled traffic or high-risk paths |
| **Groundedness / NLI checkers** | Unsupported claims (§5) | Claim-level; needs the context manifest (02.02 §8) |

### 3.1 Thresholds by cost

Each classifier outputs a score $s$. Choose the blocking threshold $\tau$ per category, from the costs of the two error types. Missing a violation costs $C_{\text{fn}}$ (harm, legal exposure). Blocking a legitimate request costs $C_{\text{fp}}$ (lost utility, user frustration, extra escalations). With a calibrated $p = P(\text{violation}\mid s)$, block when

$$
p\,C_{\text{fn}} > (1-p)\,C_{\text{fp}} \iff p > \frac{C_{\text{fp}}}{C_{\text{fp}} + C_{\text{fn}}}
$$

High-harm categories get low thresholds (block aggressively). Low-harm categories get high thresholds, to avoid over-refusal. **Calibrate first** (04.03 §5): vendor classifier scores are rarely calibrated to your traffic.

---

## 4. Layering: Why Multiple Imperfect Guards Help — and Their Limits

If layers fail **independently**, with miss rates $m_i$ and false-positive rates $f_i$:

$$
P(\text{miss all}) = \prod_i m_i, \qquad P(\text{at least one false block}) = 1 - \prod_i (1 - f_i)
$$

Three layers each missing 20% catch 99.2% jointly, but three layers each with a 2% false-positive rate block ~5.9% of benign traffic. **Stacking guards trades recall for over-refusal** — measure both.

The independence assumption is **optimistic against adversaries.** An attacker who adapts to your guards produces inputs that evade all of them together (07.03 §2). Layering reduces *accidental* failures dramatically, and makes attacks harder. It does not make them impossible. Pair it with architectural containment.

```python
def cascade_rates(miss_rates: list[float], fp_rates: list[float]) -> dict:
    p_miss, p_pass_benign = 1.0, 1.0
    for m in miss_rates:
        p_miss *= m
    for f in fp_rates:
        p_pass_benign *= (1 - f)
    return {"joint_recall": round(1 - p_miss, 4), "joint_false_block": round(1 - p_pass_benign, 4)}
```

---

## 5. Groundedness Guards

For RAG and legal content, the highest-value output guard is **claim verification**:
1. Split the answer into atomic claims (sentences, or LLM-extracted propositions, as in FActScore).
2. For each claim, check whether it is **entailed** by the cited or retrieved sources: an NLI model, or a validated LLM judge prompted with the claim plus its evidence.
3. Act on the unsupported claims: remove them, or flag them inline ("⚠ not found in the provided sources"), or regenerate with feedback, or abstain.

$$
\text{Groundedness} = \frac{\#\{\text{claims entailed by the context}\}}{\#\{\text{claims}\}}
$$

Track it per response and per release (06.03).

```python
import re
from typing import Callable


def split_claims(answer: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 3]


def groundedness_guard(answer: str, sources: list[str], entails: Callable[[str, list[str]], float],
                       threshold: float = 0.7, mode: str = "flag") -> dict:
    """entails(claim, sources) -> probability the claim is supported (NLI model or validated judge)."""
    claims = split_claims(answer)
    scored = [(c, entails(c, sources)) for c in claims]
    unsupported = [c for c, p in scored if p < threshold]
    if mode == "remove":
        text = " ".join(c for c, p in scored if p >= threshold)
    elif mode == "flag":
        text = " ".join(c if p >= threshold else f"{c} [⚠ not supported by the provided sources]" for c, p in scored)
    else:
        text = answer
    ratio = 1 - len(unsupported) / len(claims) if claims else 1.0
    return {"groundedness": round(ratio, 3), "unsupported": unsupported, "text": text,
            "action": "regenerate" if ratio < 0.5 else "ok"}
```

---

## 6. Streaming Output Guards

Users expect streaming (01.02 §1). Waiting for the full response before checking it destroys the time to first token, and checking only per chunk misses violations that span chunk boundaries. Use a **holdback buffer**: emit text only once it is at least $H$ characters (or tokens) behind the model's frontier, and scan *emitted + buffer* on every chunk.

$$
\text{added latency per token} \approx \frac{H}{\text{tokens/s}}, \qquad H \ge \text{longest pattern you must catch}
$$

For semantic checks (a classifier on the full text), run them on sentence boundaries, asynchronously. If a late check fails, **retract**: stop the stream, replace the rendered text with a safe message in the UI, and log the event. Design the UI protocol so that retraction is possible (a message ID plus a "replace" event).

```python
import re


class StreamingGuard:
    def __init__(self, patterns: list[re.Pattern], holdback: int = 32):
        self.patterns, self.h = patterns, holdback
        self.emitted, self.buffer, self.stopped = "", "", False

    def _violates(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)

    def push(self, chunk: str) -> str:
        """Returns text safe to emit now ('' if holding back or stopped)."""
        if self.stopped:
            return ""
        self.buffer += chunk
        if self._violates(self.emitted + self.buffer):
            self.stopped = True
            return ""
        safe_len = max(0, len(self.buffer) - self.h)
        out, self.buffer = self.buffer[:safe_len], self.buffer[safe_len:]
        self.emitted += out
        return out

    def finish(self) -> str:
        if self.stopped or self._violates(self.emitted + self.buffer):
            self.stopped = True
            return ""
        out, self.buffer = self.buffer, ""
        self.emitted += out
        return out
```

---

## 7. Evaluating and Operating Guardrails

### 7.1 Evaluation sets

For each policy category:
- **Positives:** real violations, red-team examples, and paraphrases.
- **Hard negatives:** benign requests that look similar (e.g. "what is the statute on stalking?" is not a request to stalk; public officials' names are not PII).
- **Benign traffic sample:** to measure over-refusal at production rates.

Report per-category precision, recall, **over-refusal rate**, and **latency overhead** (P50/P95), with CIs (06.01). Re-run the evaluation whenever a detector, a threshold, or the underlying model changes.

```python
def guard_metrics(y_true: list[int], y_pred: list[int], benign_flags: list[int] | None = None) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    res = {"precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn) if tp + fn else 1.0}
    if benign_flags is not None:
        res["over_refusal"] = sum(benign_flags) / len(benign_flags)
    return res
```

### 7.2 Operations

- **Audit log** of every guard decision (rule ID, action, scores, redacted excerpt) → monitoring dashboards (06.03).
- **Human review queues** for escalations and a sample of blocks, to find false positives (05.06).
- **Shadow mode** for new rules: log the decisions without enforcing, measure, then enforce.
- **Kill switches and per-tenant configuration**: different deployments (public site vs internal staff) get different policies.
- **User experience:** refusals explain *what* the system can do instead, without revealing detector internals that help attackers tune their evasion.

---

## 8. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Over-refusal** | Legitimate legal-research questions blocked | Hard-negative eval sets; cost-based thresholds; category-specific policies; shadow-mode tuning |
| **Latency overhead** | TTFT doubles | Rules first; parallel detectors; small classifiers; streaming holdback instead of full buffering; async semantic checks with retraction |
| **Vendor taxonomy mismatch** | The classifier blocks the wrong things | Map categories to your policy; recalibrate on your data; fine-tune a small model |
| **Detector outages** | Silent fail-open | Explicit per-rule fail modes; alert on detector errors |
| **Chunk-boundary leaks** | PII split across chunks slips through | Holdback buffer scanning emitted + buffer |
| **Hallucinated legal claims** | Confident but unsupported statements | Claim-level groundedness guard; citation validation; abstention (04.03 §5) |
| **Adversarial evasion** | Attackers rephrase around the classifiers | Don't rely on classifiers as a boundary; architectural containment (07.03); approvals for consequential actions |
| **Policy drift** | Rules and compliance expectations diverge | Policy as code with owners, reviews, tests, and changelogs |

---

## 9. Hands-On Projects

### Project 1 — Guardrail Service for a Legislative Assistant

**User stories**
- *As a compliance officer*, I want the assistant's policies (PII, confidential drafts, legal-advice boundary, neutrality, grounding) enforced consistently and auditable, without relying on prompt wording alone.

**Acceptance criteria**
1. A policy file (YAML) with ≥ 8 rules across the input, context, action, and output stages (IDs, owners, actions, severities, fail modes), loaded by a `PolicyEngine` service (FastAPI or Spring Boot) and exposed to the assistant as middleware.
2. Detectors: regex/PII (Presidio), a safety or injection classifier, an LLM rubric guard for the legal-advice boundary (validated on ≥ 150 labelled items), and the groundedness guard (§5).
3. Streaming: `StreamingGuard` with holdback, plus asynchronous sentence-level checks with UI retraction events. Measures the added TTFT and inter-token latency.
4. An audit log of every decision, a dashboard of block/redact/escalate rates per rule, and a shadow mode for new rules.
5. Tests: unit tests per rule; an end-to-end suite of violations and hard negatives; a detector-outage test that verifies the fail-closed/fail-open behaviour.

**Step-by-step**
1. Draft the policy with stakeholders; encode the rules; write the tests first.
2. Implement the engine and the detectors; integrate them as pre- and post-processing around the model and tool calls (05.01 runtime hooks).
3. Implement the streaming guard and the retraction protocol in the React client.
4. Add audit logging (OpenTelemetry events, 06.04) and the dashboards.
5. Run in shadow mode on replayed traffic, tune, then enforce.

---

### Project 2 — Guardrail Evaluation and Threshold Tuning

**User stories**
- *As the safety lead*, I want evidence that each guardrail catches what it should and doesn't block legitimate use, with thresholds set by explicit cost trade-offs.

**Acceptance criteria**
1. A per-category evaluation set: ≥ 100 positives, ≥ 100 hard negatives, and a 2,000-item benign traffic sample (redacted), with labelling guidelines and κ ≥ 0.7 (06.02).
2. Compares ≥ 3 detector options per category (e.g. a vendor moderation API, an open guard model, an LLM rubric judge, a small fine-tuned classifier): precision, recall, over-refusal, P95 latency, and cost per 1k requests.
3. Calibrates the scores; sets per-category thresholds with the §3.1 cost rule; reports the operating points.
4. Measures the cascade behaviour empirically, compares it against `cascade_rates` (the independence assumption), and explains the gap.
5. Recommends a configuration, documented in the policy file.

**Step-by-step**
1. Build the datasets (red-team prompts, paraphrases, hard negatives, benign samples).
2. Run all the detectors; store the scores.
3. Calibrate (Platt/isotonic); compute the thresholds; compute the metrics with CIs.
4. Evaluate the stacked configurations; analyse the correlated failures.
5. Write the recommendation.

---

### Project 3 — Claim-Level Groundedness Guard for Legal Answers

**User stories**
- *As an attorney*, I want every legal statement in the assistant's answers either supported by a cited source or clearly flagged, so that I never rely on an unsupported claim.

**Acceptance criteria**
1. Claim extraction (sentence split, plus LLM proposition extraction for compound sentences) and entailment scoring (an NLI cross-encoder and an LLM judge), both validated on ≥ 300 labelled claim–evidence pairs.
2. Modes: flag, remove, and regenerate with feedback (listing the unsupported claims). Measures their effect on answer correctness, groundedness, and user-perceived helpfulness (a small user study or expert ratings).
3. On ≥ 200 questions: reduces the unsupported-claim rate by ≥ 50% relative to no guard, with ≤ 5 points loss in answer completeness.
4. The added latency is measured; the guard runs asynchronously with streaming, using retraction or inline flags.
5. Groundedness is exported as an online metric with EWMA alerting (06.03).

**Step-by-step**
1. Label the claims from sampled answers against their retrieved context.
2. Implement the extraction and the two entailment backends; validate them (κ, sensitivity/specificity).
3. Implement `groundedness_guard` with the three modes; run the evaluation.
4. Integrate it into the streaming pipeline and the UI.
5. Deploy the monitoring and alerting.

---

## 10. Foundational Papers & Reading (exact titles)

- Inan et al., 2023 — *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations*
- Rebedea et al., 2023 — *NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails*
- Markov et al., 2023 — *A Holistic Approach to Undesired Content Detection in the Real World*
- Bai et al., 2022 — *Constitutional AI: Harmlessness from AI Feedback*
- Sharma et al., 2025 — *Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming*
- Röttger et al., 2023 — *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*
- Min et al., 2023 — *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*
- Honovich et al., 2022 — *TRUE: Re-evaluating Factual Consistency Evaluation*
- OWASP — *Top 10 for LLM Applications 2025*; NIST — *AI 600-1: Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile* (2024)

## 11. Essential Tooling

| Tool | Role |
|---|---|
| **NeMo Guardrails**, **Guardrails AI** | Programmable rails and validators |
| **Llama Guard / open guard models**, provider moderation and content-safety APIs (e.g. Azure AI Content Safety, Bedrock Guardrails) | Safety classification |
| **Microsoft Presidio** | PII detection and anonymisation |
| **NLI cross-encoders** (sentence-transformers), validated LLM judges | Groundedness checks |
| **OpenTelemetry + Grafana** | Guard decision auditing and dashboards |
| **promptfoo / Inspect AI / garak** | Guardrail regression and red-team testing |
