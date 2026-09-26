# 05.06 — Human-in-the-Loop

> **Module 5: Tools & Agents** · Subtopic 6 of 6
> **Prerequisites:** 05.01 (tool annotations, runtime), 05.02 (interrupts, durable execution), 05.03 (event-sourced state), 04.03 §5 (calibration), basic queueing and decision theory.
> **Outcome:** you can decide *where* humans belong in an agentic system using explicit risk economics, implement tamper-proof approval flows that survive restarts and can't be bypassed, design review experiences that resist rubber-stamping, staff review queues against SLAs, and turn human feedback into measurable system improvement.

---

## 1. Why, Where, and How Much Human Oversight

Human oversight is not a UI checkbox. It is a **control** in a risk-management system. Three questions determine the design:
1. **What can go wrong, and can it be undone?** (reversibility, blast radius)
2. **How likely is the agent to be wrong on *this* item?** (calibrated confidence)
3. **What does a human review cost, and how good are reviewers at catching errors?** (cost, and reviewer recall)

### 1.1 Autonomy levels

| Level | Agent does | Human does | Example |
|---|---|---|---|
| **L0 Assist** | Suggests | Decides and acts | Autocomplete in a drafting editor |
| **L1 Draft** | Produces a complete draft | Reviews and edits every output before use | Bill summary drafts for attorneys |
| **L2 Approve-to-act** | Proposes a concrete action | Approves or rejects each action | "Send this amendment to the committee folder?" |
| **L3 Exception-based** | Acts automatically when confident and low-risk | Handles escalations, plus audits samples | Auto-tagging bill subjects; escalating low-confidence ones |
| **L4 Autonomous + audit** | Acts | Monitors metrics, audits, and can halt | Low-risk internal data hygiene tasks |

Different **actions within one agent** sit at different levels. Reading is L4, drafting is L1, and publishing is L2. Drive this from tool annotations (`read_only`, `destructive`, `requires_approval`; 05.01 §2.1), not from ad-hoc prompt text.

**Public-sector and regulatory context:** high-risk AI uses face explicit human-oversight requirements (e.g. **EU AI Act, Article 14**, for high-risk systems), and risk frameworks such as **NIST AI RMF** expect documented oversight and accountability. Legislative drafting is an archetypal L1/L2 domain: the output has legal force, so a qualified human must own every published word.

---

## 2. The Economics of Gating

For an item the agent would handle with calibrated probability of being correct $p$, let $C_e$ be the cost of an uncaught error, $C_r$ the cost of a human review, and $\rho$ the reviewer's recall (the probability that the reviewer catches an error). Then:

$$
\mathbb{E}[\text{cost} \mid \text{auto}] = (1-p)\,C_e,
\qquad
\mathbb{E}[\text{cost} \mid \text{review}] = C_r + (1-p)(1-\rho)\,C_e
$$

Review is worth it when $(1-p)\,\rho\,C_e > C_r$, i.e. auto-execute only when

$$
p \;\ge\; \tau^* = 1 - \frac{C_r}{\rho\, C_e}
$$

**Implications:**
- For **irreversible or high-impact** actions, $C_e$ is huge, so $\tau^* \to 1$: always review. Hard-code these as `requires_approval`.
- Cheap reviews and moderate error costs justify exception-based designs (L3), *if and only if* $p$ is **calibrated** (04.03 §5). Uncalibrated confidence makes the threshold meaningless.
- **Reviewer recall $\rho$ is not 1.** Rubber-stamping lowers $\rho$ and silently raises your risk (§5).

```python
def auto_threshold(review_cost: float, error_cost: float, reviewer_recall: float = 0.9) -> float:
    return max(0.0, 1.0 - review_cost / (reviewer_recall * error_cost))


def choose_threshold(probs, correct, review_cost, error_cost, reviewer_recall=0.9, grid=None):
    """Empirically pick tau minimising total expected cost on a validation set (uses calibrated probs)."""
    grid = grid or [i / 100 for i in range(50, 100)] + [0.995, 0.999, 1.0]
    best = None
    for tau in grid:
        cost = 0.0
        for p, ok in zip(probs, correct):
            if p >= tau:
                cost += 0 if ok else error_cost                                        # auto
            else:
                cost += review_cost + (0 if ok else (1 - reviewer_recall) * error_cost)  # reviewed
        auto_rate = sum(p >= tau for p in probs) / len(probs)
        if best is None or cost < best[1]:
            best = (tau, cost, auto_rate)
    return {"tau": best[0], "total_cost": round(best[1], 2), "auto_rate": round(float(best[2]), 3)}
```

---

## 3. Oversight Patterns

| Pattern | Mechanism | Use for |
|---|---|---|
| **Approval gate** | The agent pauses before an action; a human approves, edits, or rejects | Irreversible or external actions: publish, send, delete, pay, file |
| **Draft review** | The agent produces an artifact; a human edits it before release | Documents, code, legal text |
| **Escalation** | Low confidence or a policy trigger routes the item to a human queue | Classification, triage, exceptions |
| **Clarification** | The agent asks the user a question mid-task (MCP elicitation via MRTR, A2A `INPUT_REQUIRED`) | Ambiguous scope, missing parameters |
| **Supervision / takeover** | A human watches a live session and can pause or take control | High-stakes live operations |
| **Sampling audit** | A random sample of autonomous outputs is reviewed after the fact | Measuring the error rate of L3/L4 flows; detecting drift |
| **Two-person rule** | Two independent approvals | Highest-risk actions |

---

## 4. Implementing Approvals Correctly

### 4.1 Durable pause and resume

Approvals can take hours or days. The agent must **checkpoint and stop**, not hold a thread (05.02 §6, 05.03). With Temporal, a workflow waits on a **signal** with a timer. With LangGraph, a node raises an **interrupt**, and a later call resumes from the checkpoint. With event sourcing, an `ApprovalRequested` event parks the run, and an `ApprovalGranted` event resumes it.

```
 agent ──► propose action A ──► ApprovalRequested{hash(A)} ──► (run parked; no resources held)
                                        │ notify reviewer (UI, email, Teams)
 reviewer ──► sees exactly A (diff, evidence, impact) ──► approves ──► signed token bound to hash(A)
                                        │
 runtime ──► resume ──► recompute hash(A') of the action ABOUT to execute ──► verify token ──► execute
                                        └── mismatch / expired / reused → refuse, re-request
```

### 4.2 Bind approvals to the exact action (no time-of-check/time-of-use gaps)

A classic failure: the human approves "send draft v3", but the agent executes the action with draft v4 or different recipients. **An approval must be cryptographically bound to the canonical action**: the tool, the arguments, and the *versions* of the resources it touches. It also needs an expiry, the approver identity, and single use.

```python
import hashlib
import hmac
import json
import time
import uuid


def action_hash(tool: str, args: dict, resource_versions: dict) -> str:
    canonical = json.dumps({"tool": tool, "args": args, "versions": resource_versions},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class ApprovalService:
    def __init__(self, key: bytes, ttl_s: int = 3600):
        self.key, self.ttl = key, ttl_s
        self.used_nonces: set[str] = set()                   # production: DB table with a unique constraint

    def _sign(self, payload: dict) -> str:
        body = json.dumps(payload, sort_keys=True).encode()
        return hmac.new(self.key, body, hashlib.sha256).hexdigest()

    def grant(self, approver: str, run_id: str, a_hash: str, roles: set[str], required_role: str) -> dict:
        if required_role not in roles:
            raise PermissionError(f"approver lacks role {required_role}")
        payload = {"approver": approver, "run_id": run_id, "action": a_hash,
                   "exp": int(time.time()) + self.ttl, "nonce": uuid.uuid4().hex}
        return {**payload, "sig": self._sign(payload)}

    def verify_and_consume(self, token: dict, run_id: str, tool: str, args: dict, resource_versions: dict,
                           requester: str | None = None) -> None:
        payload = {k: token[k] for k in ("approver", "run_id", "action", "exp", "nonce")}
        if not hmac.compare_digest(self._sign(payload), token.get("sig", "")):
            raise PermissionError("invalid approval signature")
        if token["exp"] < time.time():
            raise PermissionError("approval expired")
        if token["run_id"] != run_id:
            raise PermissionError("approval belongs to a different run")
        if token["action"] != action_hash(tool, args, resource_versions):
            raise PermissionError("action differs from what was approved (args or resource versions changed)")
        if requester is not None and token["approver"] == requester:
            raise PermissionError("self-approval not allowed")
        if token["nonce"] in self.used_nonces:
            raise PermissionError("approval already used")
        self.used_nonces.add(token["nonce"])
```

**Enforce this in the tool runtime (05.01 §4.2), not in the prompt.** A tool annotated `requires_approval` must refuse to execute without a valid token. The model can't talk its way past code.

### 4.3 Timeouts and defaults

Every approval request needs an SLA and a **safe default** on timeout. That is usually *deny and report*, never *proceed*. Escalate to a backup approver before the deadline. Record timeouts as their own outcome.

---

## 5. Designing the Review Experience

Reviews only work if reviewers can *actually evaluate* the proposal quickly and don't slide into **automation bias** (over-trusting the machine).

**Show:**
- **Exactly what will happen**, in domain terms ("Publish AMD-1042 to the House Judiciary folder; 3 recipients").
- **A diff** against the current state (redline for legal text).
- **The evidence:** citations with deep links, and the tool results the agent relied on.
- **The agent's confidence and its reasons for escalating** ("low confidence: two sections conflict").
- **The impact and reversibility.**

**Design the actions:**
- Approve, **edit then approve**, reject with a reason (a structured taxonomy), or ask the agent for a revision.
- Keyboard-efficient. Batch review is allowed only for low-risk, homogeneous items.

**Counter rubber-stamping:**
- Occasionally inject **known-bad "canary" proposals**, and measure reviewer catch rates ($\rho$).
- Require a reason code for approving flagged high-risk items.
- Watch time-on-task per review. Implausibly fast approvals trigger coaching or a secondary review.
- Rotate reviewers, and cap review volume per person per hour.

**Don't bury the human:** if escalation volume grows beyond capacity, quality collapses. Plan capacity (§6) and tune thresholds (§2).

---

## 6. Staffing the Review Queue

The review load in reviewer-hours per hour is

$$
A = \lambda \cdot e \cdot h
$$

where $\lambda$ is the item arrival rate, $e$ the escalation rate, and $h$ the average handling time. With $c$ reviewers and service rate $\mu = 1/h$, the **Erlang C** formula (an M/M/c queue) gives the probability that an item has to wait:

$$
P_{\text{wait}} = \frac{\frac{A^c}{c!}\frac{c}{c - A}}{\sum_{k=0}^{c-1}\frac{A^k}{k!} + \frac{A^c}{c!}\frac{c}{c - A}},
\qquad
\mathbb{E}[W] = \frac{P_{\text{wait}}}{c\mu - \lambda e}
$$

Choose the smallest $c$ that meets the SLA (e.g. mean wait < 30 min, or a percentile via simulation). Queueing theory makes one point very clearly: **utilisation near 100% creates enormous waits**.

```python
from math import factorial


def erlang_c(arrivals_per_h: float, escalation_rate: float, handle_min: float, reviewers: int) -> dict:
    lam = arrivals_per_h * escalation_rate                  # escalations per hour
    mu = 60.0 / handle_min                                   # reviews per hour per reviewer
    A = lam / mu                                             # offered load (Erlangs)
    c = reviewers
    if A >= c:
        return {"stable": False, "load_erlangs": round(A, 2)}
    top = (A ** c / factorial(c)) * (c / (c - A))
    p_wait = top / (sum(A ** k / factorial(k) for k in range(c)) + top)
    wait_h = p_wait / (c * mu - lam)
    return {"stable": True, "load_erlangs": round(A, 2), "utilisation": round(A / c, 3),
            "p_wait": round(p_wait, 3), "mean_wait_min": round(wait_h * 60, 1)}


def min_reviewers(arrivals_per_h, escalation_rate, handle_min, max_mean_wait_min):
    c = 1
    while True:
        r = erlang_c(arrivals_per_h, escalation_rate, handle_min, c)
        if r["stable"] and r["mean_wait_min"] <= max_mean_wait_min:
            return c, r
        c += 1
```

Legislative sessions are bursty (deadlines, floor sessions). Model peak arrival rates, not averages. Add surge policies: temporarily raise the auto-threshold only for *low-risk* categories, and never for irreversible actions.

---

## 7. Learning from Human Feedback

Every human decision is labelled data. Capture it deliberately:

| Signal | Captured as | Used for |
|---|---|---|
| Approve / reject + reason code | Structured label | Calibration, threshold tuning, eval sets |
| **Edits** (before → after) | Diff + edit distance | Prompt and model improvement; few-shot examples (02.01 §3); fine-tuning data |
| Clarification answers | Q/A pairs | Better planners (04.04); default assumptions |
| Canary catch rate | Per-reviewer $\rho$ | Oversight effectiveness; training |
| Time to decision | Latency | Staffing (§6); UX improvements |

```python
from difflib import SequenceMatcher


def edit_stats(drafts: list[str], finals: list[str]) -> dict:
    """How much do humans change agent drafts? Track over releases; a rising edit rate signals drift."""
    ratios = [SequenceMatcher(None, d, f).ratio() for d, f in zip(drafts, finals)]
    untouched = sum(r == 1.0 for r in ratios)
    heavy = sum(r < 0.7 for r in ratios)
    n = len(ratios)
    return {"n": n, "untouched_rate": round(untouched / n, 3), "heavy_edit_rate": round(heavy / n, 3),
            "mean_similarity": round(sum(ratios) / n, 3)}
```

**Close the loop:**
- Weekly: review the reason codes and heavy edits.
- Turn recurring corrections into prompt rules, validators (02.03 §4), or eval cases (02.01 §7).
- Re-measure the override and edit rates after each release.
- A falling override rate **with a stable canary catch rate** is the evidence that justifies raising autonomy.

---

## 8. Metrics and Governance

| Metric | Why |
|---|---|
| Escalation rate (by category) | Capacity and threshold health |
| Approval / edit / reject rates | Agent quality as judged by experts |
| Post-approval error rate (from audits or incidents) | Oversight effectiveness |
| Reviewer canary catch rate $\rho$ | Automation bias detection |
| Time to decision (P50/P95) and SLA breaches | User experience, staffing |
| Timeout-default rate | Process health |
| Autonomous error rate (from sampling audits) | Justifying L3/L4 |

**Audit trail:** every proposal, the evidence shown, the decision, the approver identity, the token, and the executed action hash — immutable (05.03 §2.2). This is what regulators, auditors, and incident reviews need.

---

## 9. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Approval bypass** | The agent executes a gated action without approval | Enforce in the tool runtime with signed, action-bound tokens (§4.2) |
| **TOCTOU** | The approved action differs from the executed one | Hash the tool + args + resource versions; verify at execution; single-use nonces |
| **Rubber-stamping** | High approval rates; errors slip through | Canaries, reason codes, time-on-task monitoring, volume caps, a clear diff UI |
| **Review backlog** | SLA breaches; users bypass the system | Erlang C staffing; threshold tuning; peak planning; batching low-risk items |
| **Blocked runs hold resources** | Threads and containers waiting for days | Durable pause (signals/interrupts); no in-memory waiting |
| **Unsafe timeout defaults** | Actions proceed when nobody answered | Default deny; escalate to a backup; alert |
| **Feedback not used** | The same corrections, week after week | Structured reason codes; the edit-to-eval pipeline; release-over-release tracking |
| **Over-escalation** | Humans see everything; the agent adds no value | Calibrate confidence; the cost-based threshold (§2); category-specific policies |

---

## 10. Hands-On Projects

### Project 1 — Approval-Gated Amendment Drafting Agent

**User stories**
- *As a legislative attorney*, I want an agent that drafts amendment language and prepares it for filing, but never files or shares anything without my explicit approval of the exact text and recipients.
- *As an auditor*, I want a tamper-evident record of who approved what.

**Acceptance criteria**
1. The agent (05.02) with tools `draft_amendment` (L1), `validate_drafting_rules` (auto), `share_with_committee` (L2, `requires_approval`), and `file_amendment` (L2, two-person rule).
2. The durable pause/resume uses Temporal signals or LangGraph interrupts. No resources are held while waiting; runs survive restarts.
3. `ApprovalService` semantics (§4.2): tokens bound to the tool + args + resource versions, with expiry, single use, role checks, and no self-approval. Tests prove that editing the draft after approval invalidates the token.
4. A review UI (React) shows a redline diff against current law, the citations, the validation results, the recipients, and the reversibility. Actions: approve, edit & approve, reject with a reason code.
5. An immutable audit log (event-sourced, hash-chained; 05.03) of proposals, decisions, and executed action hashes. An auditor view reconstructs any filing's approval history.

**Step-by-step**
1. Implement the tools with annotations in the 05.01 runtime; add the approval enforcement in the executor.
2. Build the workflow graph with interrupt nodes before gated tools.
3. Implement `ApprovalService` with Postgres storage for the nonces and audit events.
4. Build the review UI with a diff viewer; integrate SSO for approver identity and roles.
5. Write the security tests (bypass attempts, TOCTOU, replay, self-approval) and the resilience tests (restart while waiting).

---

### Project 2 — Confidence-Based Escalation with Cost-Optimal Thresholds

**User stories**
- *As an operations lead* for bill subject tagging and routing, I want the system to handle confident, low-risk items automatically and escalate the rest, with thresholds justified by cost and a staffing plan that meets our SLA during session peaks.

**Acceptance criteria**
1. A classifier/agent producing calibrated probabilities (Platt or isotonic; ECE < 0.05 on held-out data).
2. Error and review costs elicited per category (e.g. mis-routing to the wrong committee costs more than a wrong subject tag). Uses `choose_threshold` per category, and reports the auto rate, the expected cost, and the cost vs "review everything" and "review nothing".
3. Staffing plan: `min_reviewers` for the average and peak arrival rates from historical session data, with a simulation (e.g. a discrete-event simulation with SimPy) validating the Erlang C predictions.
4. Sampling audits of auto-handled items (e.g. 2%) to estimate the true autonomous error rate, with confidence intervals.
5. A dashboard: escalation rate, queue length, waits, audit error rates, and threshold drift alerts.

**Step-by-step**
1. Build the dataset, train or prompt the classifier, and calibrate it.
2. Run workshops with stakeholders to estimate costs; document the assumptions.
3. Compute the thresholds and simulate the policy on historical data.
4. Model the queue analytically and by simulation; pick the staffing and surge rules.
5. Deploy with the audit sampling and the dashboard.

---

### Project 3 — Human Feedback Flywheel and Automation-Bias Monitoring

**User stories**
- *As the product owner* of a drafting assistant, I want every reviewer edit to make the assistant measurably better, and I want to know whether reviewers are still catching errors as the assistant improves.

**Acceptance criteria**
1. Captures drafts, final versions, reason codes, and review times for all reviewed outputs, with PII handling.
2. `edit_stats` per release and per document type; a clustering of heavy edits into recurring correction themes (LLM-assisted, human-validated).
3. Converts the top 3 correction themes each cycle into concrete fixes (prompt rules, validators, retrieval changes). Each fix is validated on an eval set built from past edits.
4. Canary injection: 2–5% of review items are known-flawed. Tracks $\rho$ per reviewer and overall, and alerts when $\rho$ drops below a threshold.
5. A report over ≥ 3 release cycles: heavy-edit rate, canary catch rate, and time per review — demonstrating (or refuting) improvement without an erosion of oversight.

**Step-by-step**
1. Instrument the review UI and store the events (05.03 event store).
2. Build the edit analytics and the theme-clustering pipeline.
3. Build the canary generator (inject known errors: a wrong citation, a wrong date, a missing section) and the tracking.
4. Run the improvement cycles; for each fix, run the eval gate (02.01 §7).
5. Write up the results and governance recommendations (when to raise or lower autonomy levels).

---

## 11. Foundational Papers & Reading (exact titles)

**Deferral and selective prediction**
- Geifman & El-Yaniv, 2017 — *Selective Classification for Deep Neural Networks*
- Madras, Pitassi & Zemel, 2018 — *Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer*
- Mozannar & Sontag, 2020 — *Consistent Estimators for Learning to Defer to an Expert*
- Guo et al., 2017 — *On Calibration of Modern Neural Networks*

**Human–AI teaming and automation bias**
- Parasuraman & Manzey, 2010 — *Complacency and Bias in Human Use of Automation: An Attentional Integration*
- Skitka, Mosier & Burdick, 1999 — *Does automation bias decision-making?*
- Bansal et al., 2021 — *Does the Whole Exceed its Parts? The Effect of AI Explanations on Complementary Team Performance*
- Amershi et al., 2019 — *Guidelines for Human-AI Interaction*

**Learning from feedback**
- Ouyang et al., 2022 — *Training language models to follow instructions with human feedback*

**Governance**
- Regulation (EU) 2024/1689 (EU AI Act) — Article 14 *Human oversight*
- NIST, 2023 — *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*
- OWASP — *Top 10 for LLM Applications* (Excessive Agency)

**Queueing**
- Gross et al. — *Fundamentals of Queueing Theory* (textbook; Erlang C / M/M/c)

## 12. Essential Tooling

| Tool | Role |
|---|---|
| **Temporal** (signals, timers), **LangGraph** (interrupts, checkpointers) | Durable pause/resume |
| **MCP elicitation (MRTR)**, **A2A `INPUT_REQUIRED`** | Protocol-level human input |
| **React + diff viewers** (e.g. react-diff-viewer), **Microsoft Teams / Slack adaptive cards** | Review UIs and notifications |
| **Label Studio / Argilla** | Structured review and feedback capture |
| **SimPy** | Review-queue simulation |
| **scikit-learn** (calibration) | Calibrated confidence for escalation |
| **Postgres (event store), OpenTelemetry** | Audit trails and traces |
