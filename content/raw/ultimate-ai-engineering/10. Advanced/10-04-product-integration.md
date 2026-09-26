# 10.04 — Product Integration (Shipping AI Features People Trust and Use)

> **Module 10: Advanced** · Subtopic 4 of 4
> **Prerequisites:** 05.06 (HITL), 06.01–06.03 (evals, datasets, online metrics and experiments), 07.01 (guardrails), 08.05 (release lifecycle, governance, HB 178), 09.01 §5.1 (DPO from edits), 10.03 (architecture, SSE).
> **Outcome:** you can decide *how much* autonomy an AI feature should have from its error economics, and design interaction patterns that make AI output verifiable, such as legislative strike/underline diffs with per-hunk acceptance. You can stream responses robustly, capture implicit and explicit feedback that feeds evals and training, and roll out with experiments, adoption metrics, accessibility, disclosure, and change management.

> **Status check (September 2026).**
> - **MCP Apps** (official MCP extension, announced Jan 26, 2026) lets MCP tools return **interactive UI** rendered in sandboxed iframes inside AI hosts. UI resources use the `ui://` scheme, and host and app talk over JSON-RPC via `postMessage`. Supported hosts include Claude, ChatGPT, VS Code, and Goose.
> - **AG-UI** is an open event protocol for streaming agent state and UI events to front-ends.
> - Neither replaces your own product UI. They matter when your capability should also appear *inside* AI assistants.

---

## 1. How Much Autonomy? Decide from Error Economics

Every AI feature sits somewhere on this spectrum:

```
 OFF ─── ASSIST (explain, search) ─── SUGGEST (draft; human edits/approves every item) ─── AUTO + audit/spot-check ─── AUTONOMOUS
```

Pick the position from **measured accuracy** $a$ (06.01), time saved $s$, review time $r$, reviewer catch rate $c$, and the cost of an uncaught error $E$:

$$
\mathbb{E}[\text{suggest}] = a\,s - r - (1-a)(1-c)\,E,
\qquad
\mathbb{E}[\text{auto}] = s - (1-a)\,E,
\qquad
\mathbb{E}[\text{off}] = 0 .
$$

$c < 1$ reflects **automation bias**: reviewers approve plausible AI output they would have caught when writing from scratch (Parasuraman & Manzey). **The UI is the main lever on $c$.**

```python
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    minutes_saved: float      # time saved per item if the AI output is used as-is
    accuracy: float           # P(output correct), from offline evals (06.01)
    error_cost_min: float     # cost of an uncaught error in staff-minutes (rework, correction, reputational)
    review_min: float         # time to review one AI output
    catch_rate: float         # P(reviewer catches an error)

def expected_value(t: Task, mode: str) -> float:
    err = 1 - t.accuracy
    if mode == "off":
        return 0.0
    if mode == "suggest":
        return t.accuracy * t.minutes_saved - t.review_min - err * (1 - t.catch_rate) * t.error_cost_min
    if mode == "auto":
        return t.minutes_saved - err * t.error_cost_min
    raise ValueError(mode)

def choose_mode(t: Task):
    ev = {m: expected_value(t, m) for m in ("off", "suggest", "auto")}
    return max(ev, key=ev.get), {k: round(v, 2) for k, v in ev.items()}

tasks = [
    Task("normalise citation format", 2, 0.995, 10, 0.5, 0.9),
    Task("draft bill summary", 25, 0.92, 120, 6, 0.85),
    Task("fiscal-note figures", 15, 0.97, 2000, 5, 0.8),
    Task("legal effect opinion", 40, 0.70, 5000, 45, 0.7),
]
for t in tasks:
    mode, ev = choose_mode(t)
    print(f"{t.name:28s} -> {mode:8s} {ev}")
assert [choose_mode(t)[0] for t in tasks] == ["auto", "suggest", "off", "off"]

# Product levers change the answer: numbers linked to their source cells + automatic cross-check against the
# fiscal table raise both accuracy and reviewer catch rate.
fiscal_v2 = Task("fiscal-note figures (v2 UI)", 15, 0.99, 2000, 4, 0.98)
print(f"{fiscal_v2.name:28s} -> {choose_mode(fiscal_v2)[0]:8s} {choose_mode(fiscal_v2)[1]}")
assert choose_mode(fiscal_v2)[0] == "suggest"
```

| Task | Decision | Why |
|---|---|---|
| Normalise citation format | **Auto** (+1.95 min/item) | Cheap errors, high accuracy; review costs more than it saves |
| Draft bill summary | **Suggest** (+15.6) | Suggest beats auto because errors are costly and reviewers catch most of them |
| Fiscal-note figures, v1 | **Off** (suggest −2.45, auto −45) | One missed number costs more than all the time saved |
| Fiscal-note figures, **v2 UI** | **Suggest** (+10.45) | Source-linked highlighting and an automatic table cross-check raise accuracy to 0.99 and catch rate to 0.98 |
| Legal-effect opinion | **Off** | Accuracy too low and error cost too high; route to counsel |

Two lessons:
- **The feature design, not the model, flipped the fiscal decision.** Invest in making verification cheap and reliable.
- **Policy overrides arithmetic.** HB 178 requires human review of AI recommendations or decisions affecting a person's rights (08.05 §9), whatever the expected value says.

---

## 2. Interaction Patterns That Make AI Output Verifiable

| Pattern | Use when | Must-haves |
|---|---|---|
| **Draft-then-edit** | Summaries, minutes, letters | Editable output; the edit is captured as a signal (§5); "regenerate" keeps edits |
| **Proposed changes as a diff** | Any AI edit to existing text (bills, amendments) | **Legislative strike/underline rendering**, per-hunk accept/reject, semantic hunk boundaries |
| **Inline suggestions** | Citation completion, defined-term lookup | Low-latency (< 300 ms); Tab to accept; no layout shift |
| **Side-panel assistant** | Research questions while drafting | Answers cite sources; click-through to the exact span (10.01, 10.02 provenance) |
| **Structured autofill** | Forms: fiscal-note templates, bill request intake | Field-level confidence; unfilled rather than guessed when unsure |
| **Explain / "why"** | Search ranking, flags ("this cites a repealed section") | Show the evidence path (e.g. graph path from 10.02 §8) |
| **Background agent + inbox** | Long tasks (impact analysis across a title) | Progress, interruptibility, results as reviewable items (05.06) |

**Diffs are the legislative UI primitive.** Bill drafters read amendments as ~~stricken~~ and underlined text. AI proposals should arrive in the same form, reviewable change by change. One subtlety determines whether per-hunk acceptance is safe: **hunks must follow semantic units**. A word-level diff splits "August 15 → September 1" into two independent changes, and accepting only one produces a date **nobody proposed**:

```python
import difflib, re

def tokenize(s):
    return re.findall(r"\S+|\s+", s)

def legislative_hunks(original, proposed, merge_adjacent=True):
    """Word-level diff as reviewable hunks (op, old, new). merge_adjacent joins change hunks separated only by
    whitespace, so 'August 15' -> 'September 1' is ONE decision. (Production: also merge within dates, amounts,
    MCA cites, and defined terms using an entity tagger.)"""
    a, b = tokenize(original), tokenize(proposed)
    hunks = [(op, "".join(a[i1:i2]), "".join(b[j1:j2]))
             for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()]
    if not merge_adjacent:
        return hunks
    merged = []
    for h in hunks:
        if (h[0] != "equal" and len(merged) >= 2 and merged[-1][0] == "equal" and not merged[-1][1].strip()
                and merged[-2][0] != "equal"):
            ws = merged.pop()[1]; prev = merged.pop()
            merged.append(("replace", prev[1] + ws + h[1], prev[2] + ws + h[2]))
        else:
            merged.append(h)
    return merged

def render_markup(hunks):
    """~~stricken~~ and __new__ text; the UI renders these as strike-through and underline."""
    out = []
    for op, old, new in hunks:
        if op == "equal":
            out.append(old)
        else:
            out.append(" ".join(x for x in (f"~~{old}~~" if old.strip() else "", f"__{new}__" if new.strip() else "") if x))
    return "".join(out)

def apply_hunks(hunks, accepted):
    """Clean text after the drafter accepts a subset of change hunks (indexes over change hunks only)."""
    out, k = [], 0
    for op, old, new in hunks:
        if op == "equal":
            out.append(old); continue
        out.append(new if k in accepted else old); k += 1
    return "".join(out)

orig = "The governing body shall adopt a final budget no later than August 15 after a public hearing."
prop = "The governing body may adopt a final budget no later than September 1 after one public hearing."
fine = legislative_hunks(orig, prop, merge_adjacent=False)
print("word-level hunks:", sum(op != "equal" for op, _, _ in fine), "| accepting only 'September' gives:",
      re.search(r"than \w+ \d+", apply_hunks(fine, {1})).group(0), "<- a date nobody proposed")
h = legislative_hunks(orig, prop)
print(render_markup(h))
for i, (op, old, new) in enumerate(x for x in h if x[0] != "equal"):
    print(f"  hunk {i}: {old!r} -> {new!r}")
assert apply_hunks(h, {0, 1, 2}) == prop and apply_hunks(h, set()) == orig
assert apply_hunks(h, {1}) == orig.replace("August 15", "September 1")
```

The rendered result is: The governing body ~~shall~~ __may__ adopt a final budget no later than ~~August 15~~ __September 1__ after ~~a~~ __one__ public hearing. There are three independent, meaningful decisions.

The drafting UI must also:
- keep the AI proposal **separate from the document of record** until accepted;
- record who accepted what, which serves as audit and feeds §5 signals;
- never auto-apply to enrolled or official text.

---

## 3. Streaming UX That Survives Real Networks

Streaming cuts *perceived* latency, since users read as tokens arrive. It also creates failure modes: dropped connections, proxies buffering responses, half-rendered citations, and screen readers announcing every token. Use **Server-Sent Events** with **typed events** and **resumability**:

| Event | Payload | UI behaviour |
|---|---|---|
| `status` | `{step: "retrieving", sources: 4}` | Progress line ("Searching statutes…"), no layout jump |
| `delta` | text chunk | Append; announce to screen readers only on sentence boundaries |
| `citation` | `{n, mca, chunk_id, span}` | Render a superscript linked to source; hover preview |
| `tool` | `{name, state}` | Show agent actions (05.x) for transparency |
| `error` | `{code, retryable}` | Inline error with retry; keep the partial text, marked incomplete |
| `done` | `{finish, bundle, usage}` | Enable copy/accept; attach the feedback controls (§5) |

```python
import json

def sse_event(event, data, id_=None):
    """Encode one Server-Sent Event; multi-line data becomes multiple 'data:' lines (SSE spec)."""
    lines = [] if id_ is None else [f"id: {id_}"]
    lines.append(f"event: {event}")
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    lines += [f"data: {ln}" for ln in payload.split("\n")]
    return "\n".join(lines) + "\n\n"

def sse_parse(stream_text):
    """Minimal SSE client parser -> list of (id, event, data)."""
    out = []
    for block in stream_text.split("\n\n"):
        if not block.strip():
            continue
        fields = {"id": None, "event": "message", "data": []}
        for line in block.split("\n"):
            if line.startswith(":"):
                continue                                       # comment / heartbeat
            k, _, v = line.partition(":")
            v = v[1:] if v.startswith(" ") else v
            if k == "data":
                fields["data"].append(v)
            elif k in ("id", "event"):
                fields[k] = v
        out.append((fields["id"], fields["event"], "\n".join(fields["data"])))
    return out

class StreamBuffer:
    """Per-response event log so a reconnecting client (Last-Event-ID) resumes without gaps or duplicates."""
    def __init__(self):
        self.events = []
    def emit(self, event, data):
        self.events.append((len(self.events) + 1, event, data))
        return sse_event(event, data, id_=len(self.events))
    def replay_after(self, last_event_id):
        return "".join(sse_event(e, d, id_=i) for i, e, d in self.events if i > int(last_event_id or 0))

buf = StreamBuffer()
wire = [buf.emit("status", {"step": "retrieving", "sources": 4}),
        buf.emit("delta", "The bill revises"), buf.emit("delta", " county budget\ndeadlines"),
        buf.emit("citation", {"n": 1, "mca": "7-6-4005", "chunk": "c-88"}),
        buf.emit("delta", " to September 1."),
        buf.emit("done", {"finish": "stop", "bundle": "3f9a1c", "tokens": 812})]
received = sse_parse("".join(wire[:3]))                        # connection drops after event 3
resumed = sse_parse(buf.replay_after(received[-1][0]))         # reconnect with Last-Event-ID: 3
text = "".join(d for _, e, d in received + resumed if e == "delta")
print([(i, e) for i, e, _ in received + resumed]); print(repr(text))
assert [int(i) for i, _, _ in received + resumed] == [1, 2, 3, 4, 5, 6]
assert text == "The bill revises county budget\ndeadlines to September 1."
```

**Operational details:**
- Send heartbeat comments (`: ping`) every 15 s so idle proxies don't cut the stream.
- Disable response buffering at the gateway/ingress (`X-Accel-Buffering: no` for NGINX).
- **Propagate client cancellation to the model call** so abandoned requests stop spending tokens (08.02).
- Keep the stream buffer (Redis, TTL of minutes) keyed by response ID.
- In Spring, return `Flux<ServerSentEvent<…>>` from WebFlux, or use `SseEmitter`. In React, use `EventSource` (GET) or `fetch` with a streaming reader (POST).

**Accessibility (WCAG 2.2):**
- Stream into an `aria-live="polite"` region that updates **per sentence**, not per token.
- Provide a "stop generating" control reachable by keyboard.
- Respect `prefers-reduced-motion` (no typing animation).
- Make sure citations and diff markup are conveyed semantically (`<del>`/`<ins>` with accessible names), not only by colour or strike-through.

---

## 4. Trust, Transparency, and Disclosure

- **Disclose.** Label AI-generated content, and for public-facing interfaces and unreviewed publications this is required under Montana **HB 178** (08.05 §9). Record reviewer identity when content is reviewed.
- **Cite everything factual.** Link to the exact span, section, and version (10.01, 10.02 provenance). Uncited factual claims are a UX bug.
- **Show uncertainty honestly.** Prefer *"not found in the statutes provided"* over a confident guess, and design the abstention path as a first-class answer. Calibrated confidence displays help only if they're calibrated (06.01), so don't show raw log-probs.
- **Make errors cheap to fix.** Provide undo, version history, one-click regenerate that keeps edits, and "report a problem" with the trace ID attached.
- **Calibrate reliance.** Onboarding should state what the feature is *not* for (e.g. "does not give legal opinions"). Occasional **verification prompts** on high-stakes outputs counter automation bias; measure their effect on catch rate $c$.

---

## 5. Feedback: Implicit Signals Beat Thumbs

Thumbs are sparse (typically < 5% of interactions) and biased toward extremes. **What users do with the output** is denser and more honest:
- accepted verbatim;
- lightly edited;
- heavily rewritten;
- discarded.

The edits themselves are **preference data** (09.01 §5.1).

```python
import difflib
from collections import defaultdict

def edit_ratio(draft, final):
    """Normalised edit amount in [0, 1] at word level: 0 = accepted verbatim, 1 = fully rewritten."""
    return 1 - difflib.SequenceMatcher(a=draft.split(), b=final.split(), autojunk=False).ratio()

def feedback_report(events, heavy_edit=0.3, min_pair_edit=0.15):
    """events: dicts with id, feature, draft, final (None if discarded), thumbs (+1/-1/None).
    Returns per-feature implicit + explicit metrics and DPO pair candidates."""
    by = defaultdict(list)
    for e in events:
        by[e["feature"]].append(e)
    report, pairs = {}, []
    for f, evs in by.items():
        used = [e for e in evs if e["final"] is not None]
        ratios = [edit_ratio(e["draft"], e["final"]) for e in used]
        thumbs = [e["thumbs"] for e in evs if e["thumbs"] is not None]
        report[f] = dict(n=len(evs), acceptance=round(len(used) / len(evs), 2),
                         verbatim=round(sum(r == 0 for r in ratios) / max(1, len(used)), 2),
                         mean_edit=round(sum(ratios) / max(1, len(ratios)), 2),
                         heavy_edit_rate=round(sum(r > heavy_edit for r in ratios) / max(1, len(used)), 2),
                         thumbs_up=round(sum(t > 0 for t in thumbs) / len(thumbs), 2) if thumbs else None,
                         explicit_n=len(thumbs))
        pairs += [{"prompt_ref": e["id"], "chosen": e["final"], "rejected": e["draft"]}
                  for e, r in zip(used, ratios) if r >= min_pair_edit]
    return report, pairs

D = "The bill revises county budget deadlines and requires one public hearing before adoption."
events = [
    {"id": 1, "feature": "summary", "draft": D, "final": D, "thumbs": None},
    {"id": 2, "feature": "summary", "draft": D, "final": D.replace("revises", "changes"), "thumbs": 1},
    {"id": 3, "feature": "summary", "draft": D, "final": "Moves the county budget deadline to September 1 and requires one hearing.", "thumbs": -1},
    {"id": 4, "feature": "summary", "draft": D, "final": None, "thumbs": None},
    {"id": 5, "feature": "citations", "draft": "7-6-4005, 7-6-4006", "final": "7-6-4005, 7-6-4006", "thumbs": None},
    {"id": 6, "feature": "citations", "draft": "7-6-4005", "final": "7-6-4005", "thumbs": None},
]
rep, pairs = feedback_report(events)
for f, m in rep.items():
    print(f, m)
print("DPO pair candidates:", [p["prompt_ref"] for p in pairs])
assert rep["summary"]["acceptance"] == 0.75 and rep["citations"]["verbatim"] == 1.0
assert [p["prompt_ref"] for p in pairs] == [3] and rep["citations"]["thumbs_up"] is None
```

How the pieces connect:
- **Pipeline.** Client events flow to Kafka topic `feedback.*` (10.03). Each event is joined to its trace (bundle digest, retrieved sources) and aggregated per feature and bundle. Results feed:
  - the **online metrics dashboard** (06.03);
  - **eval dataset candidates** from heavy edits and thumbs-down (06.02);
  - **DPO pairs** after style-vs-fact triage (09.01 Project 2).
- **Privacy.** Drafts and edits can contain confidential material. Apply the same classification, retention, and access rules as the documents themselves (07.01, 08.05 §9).
- **Report `thumbs_up` as unknown when there's no explicit signal, not as 0%.** Missing data isn't negative data.

---

## 6. Measuring Product Impact

| Layer | Metric | Notes |
|---|---|---|
| **Adoption** | Weekly active users / eligible users; feature reach; **retention** at week 4 | Novelty inflates week 1; judge on weeks 4–8 |
| **Engagement quality** | Acceptance rate, verbatim rate, heavy-edit rate (§5) | Per feature, per bundle version |
| **Outcome** | Time-on-task (instrumented), throughput (bills summarised per analyst-day), rework rate | Pre/post with a holdout, or A/B (06.03) |
| **Quality/safety guardrails** | Uncaught error reports, correction notices, guard blocks, escalations | Must not regress while adoption rises |
| **Cost** | Cost per accepted output (08.02 `cost_per_success`) | The unit that matters for FinOps |

**Experiment design.**
- Randomise by **user** (or office), not by request, to avoid contamination (06.03).
- Keep a **long-term holdout** (5–10%) to measure durable effects.
- Pre-register the primary metric (e.g. time-on-task for summaries) and the guardrails (error reports).
- **Expect heterogeneity.** AI assistance often helps less-experienced staff most (Brynjolfsson et al.), and can *slow* experts on familiar tasks (METR, 2025). Segment by experience before concluding.

---

## 7. Rollout and Change Management

```
 design partner (1 office, 2 wks) ──► internal pilot (1 committee's staff, 4 wks, holdout) ──► phased GA by division
      co-design, shadow mode            A/B, weekly quality review, feedback triage          feature flags, kill switch,
                                                                                               training, support rota
```

- **Co-design with the people whose work changes** (drafters, fiscal analysts, committee secretaries). They know the error costs in §1 better than engineers do.
- **Train for the failure modes, not the features.** Cover what the assistant gets wrong, how to verify, and how to report.
- **Time launches around the legislative calendar.** Do not change drafting tools during peak session weeks, so ship in the interim and freeze before session (08.05 §6).
- **Feature flags and kill switches** per capability (08.05 §8). Communicate degraded modes ("AI summaries paused; drafts unaffected").
- **Records and governance:** inventory entries, disclosures, and human-review points (08.05 §9) go live *with* the feature.

**Integration surfaces beyond your own UI:**
- **MCP servers** expose capabilities (bill status, section lookup, impact analysis) to any AI host the organisation approves.
- **MCP Apps** add interactive UI (e.g. the diff reviewer) inside those hosts. It runs sandboxed, with user consent for UI-initiated tool calls.
- **Teams/Outlook** add-ins serve notification and summary workflows.
- **APIs** serve other systems.

Every surface calls the same use-case APIs (10.03 §1), with the same auth, guards, evals, and telemetry.

---

## 8. Production Challenges and Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **Automation bias** | Reviewers approve AI errors | Verification-friendly UI (source-linked highlights, cross-checks), spot-check prompts, measure catch rate |
| **Unsafe partial acceptance** | Accepted hunks combine into nonsense | Semantic hunk boundaries (§2) |
| **Streaming breaks behind proxies** | Responses arrive all at once or cut off | Disable buffering, heartbeats, resumable SSE (§3) |
| **Token-by-token screen reader spam** | Inaccessible assistant | Sentence-level live-region updates, stop control, semantic diff markup |
| **Sparse, biased feedback** | Can't tell if quality changed | Implicit signals (acceptance, edits) joined to traces |
| **Novelty-driven metrics** | Week-1 success, week-6 abandonment | Retention cohorts, long-term holdout |
| **Mistimed launch** | Staff overwhelmed during session | Interim launches, pre-session freeze, phased rollout |
| **Disclosure gaps** | Unlabelled AI content published | Disclosure enforced by the publication workflow and CI checks (08.05 §9) |
| **Shadow AI** | Staff paste confidential text into unapproved tools | Provide a good sanctioned tool; policy + training; DLP controls |

---

## 9. Hands-On Projects

### Project 1 — AI-Assisted Amendment Drafting with Legislative Diff Review

**User stories**
- *As a bill drafter*, I want to describe an intended change ("move the budget deadline to September 1 and require only one hearing") and receive a proposed amendment in strike/underline form. I want to accept or reject each change, and nothing touches the bill until I accept.

**Acceptance criteria**
- The AI service (10.03) returns proposed text plus rationale citing the sections involved (10.02 impact analysis). The UI renders semantic hunks (dates, amounts, MCA cites, and defined terms are merged) with per-hunk accept/reject.
- Accepted changes are written through the bill system's normal versioning with an audit record (user, bundle digest, hunks accepted). There is no path to modify enrolled text.
- On 100 drafting tasks, drafters' time-on-task falls ≥ 25% vs control, with **0** accepted-hunk combinations producing unintended text (checked by comparing to the drafter's final intent).
- The UI passes a WCAG 2.2 AA audit (`<del>`/`<ins>` semantics, keyboard-only flow, screen-reader test).

**Step-by-step**
1. Co-design the interaction with 3 drafters and write the §1 error-economics table for this feature.
2. Implement hunk generation with entity-aware merging, then the React diff component with accept/reject and keyboard support.
3. Build the backend proposal endpoint with streaming status, citations, and impact analysis.
4. Wire the audit and versioning integration, and the feedback events (§5).
5. Run the pilot with a control group and write up the results.

### Project 2 — Feedback and Telemetry Pipeline to Quality Dashboard and Training Data

**User stories**
- *As the AI product owner*, I want to see per-feature, per-version quality from real usage, and turn corrections into eval items and training pairs automatically.

**Acceptance criteria**
- Client events (shown, accepted, edited, discarded, thumbs with reason, copy) go to Kafka `feedback.*`. They are joined with traces by response ID and aggregated per feature × bundle × user segment.
- The dashboard shows acceptance, verbatim, heavy-edit, and thumbs rates (with *n*), cost per accepted output, and guard/escalation rates, with week-over-week and version comparisons.
- Heavy edits and thumbs-down are queued for triage and labelled *style* (to DPO pairs) or *fact* (to RAG gap or eval item). ≥ 200 labelled items are added to 06.02 datasets in the first month.
- Privacy review is passed: classification-aware retention and access to drafts and edits.

**Step-by-step**
1. Define the event schema (schema registry) and instrument the React components.
2. Build the stream join with traces (Kafka Streams or Flink) and aggregate into Postgres.
3. Build the dashboard (Grafana or an internal UI) with confidence intervals on rates.
4. Build the triage UI and export jobs (datasets, DPO pairs).
5. Review privacy and retention with records staff.

### Project 3 — Pilot Launch Plan with Experiment, Adoption, and Governance

**User stories**
- *As IT leadership*, I want to launch the bill-summary assistant to one division in the interim, with evidence strong enough to decide on session-wide rollout.

**Acceptance criteria**
- A written launch plan covers:
  - co-design findings;
  - the §1 decision table;
  - a pre-registered experiment (user-level randomisation, primary metric time-on-task, guardrails: error reports and correction notices; power analysis via 06.03 CUPED where applicable);
  - a 10% long-term holdout.
- Governance is live at launch: inventory entry, disclosure labels, human-review points, kill switch, support rota, and a training module on failure modes.
- The decision report at week 8 includes retention cohorts, heterogeneity by experience, cost per accepted summary, and a go/no-go recommendation with confidence intervals.
- A pre-session freeze date and rollback plan are agreed with legislative leadership.

**Step-by-step**
1. Hold stakeholder workshops, then run the error-economics analysis and write the success criteria.
2. Design the experiment (randomisation unit, metrics, power) and configure the flags.
3. Prepare the governance artefacts (08.05 §9) and training.
4. Run the pilot with a weekly quality review and triage.
5. Analyse the results, recommend, and plan the rollout or iteration.

---

## 10. Foundational Papers and Guidance (exact titles)

- Amershi et al., 2019 — *Guidelines for Human-AI Interaction*
- Lee & See, 2004 — *Trust in Automation: Designing for Appropriate Reliance*
- Parasuraman & Manzey, 2010 — *Complacency and Bias in Human Use of Automation: An Attentional Integration*
- Passi & Vorvoreanu, 2022 — *Overreliance on AI: Literature Review*
- Buçinca, Malaya & Gajos, 2021 — *To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI in AI-assisted Decision-making*
- Vaithilingam, Zhang & Glassman, 2022 — *Expectation vs. Experience: Evaluating the Usability of Code Generation Tools Powered by Large Language Models*
- Ziegler et al., 2022 — *Productivity Assessment of Neural Code Completion*
- Brynjolfsson, Li & Raymond, 2023 — *Generative AI at Work*
- Dell'Acqua et al., 2023 — *Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality*
- Becker et al. (METR), 2025 — *Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity*
- Kohavi, Tang & Xu, 2020 — *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*
- Google PAIR — *People + AI Guidebook*; W3C — *Web Content Accessibility Guidelines (WCAG) 2.2*
- WHATWG HTML Living Standard — *Server-sent events*; MCP — *MCP Apps extension specification (2026-01-26)*

## 11. Essential Tooling

| Tool | Use |
|---|---|
| **React** + `EventSource`/fetch streaming, diff libraries (`diff-match-patch`, jsdiff) | Streaming UI, diff rendering |
| **Spring WebFlux** (`Flux<ServerSentEvent>`), Redis stream buffers | Resumable SSE backends |
| **OpenFeature / LaunchDarkly / Unleash** | Flags, phased rollout, kill switches |
| **GrowthBook / Statsig / in-house (06.03)** | Experiments, holdouts, CUPED |
| **Kafka + Kafka Streams/Flink** | Feedback pipelines joined with traces |
| **axe-core, NVDA/JAWS/VoiceOver testing** | Accessibility verification |
| **MCP servers + MCP Apps; Teams/Outlook add-ins** | Integration surfaces beyond your UI |

**Image prompt (diff review UI):** *"Clean product mockup of a bill-drafting editor: left pane shows statute text with AI-proposed changes — red strike-through for deletions, blue underline for insertions — each change in a subtle card with 'Accept' and 'Reject' buttons; right pane shows 'Why this change' with citations to MCA sections and an impact-analysis list. Top banner: 'AI-proposed — not applied until accepted'. Professional government UI style, accessible contrast."*

**Image prompt (autonomy spectrum):** *"Horizontal spectrum graphic from 'Off' to 'Autonomous' with markers 'Assist', 'Suggest', 'Auto + audit'. Example tasks placed on it: 'citation formatting' near Auto, 'bill summary' at Suggest, 'fiscal figures (v2 UI)' at Suggest with an arrow from Off, 'legal opinion' at Off. Caption: 'expected value = time saved − review − uncaught error cost'. Minimal flat infographic."*
