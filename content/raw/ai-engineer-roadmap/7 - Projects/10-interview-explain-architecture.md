# Capstone Project 10 — Explain Your Architecture in an Interview

> **Deliverable:** A rehearsed, structured way of talking about an AI system you built — one that lands with Senior/Staff AI Engineer interviewers. Includes the "one whiteboard picture," the 5-minute pitch, the deep-dive answers to expected follow-ups, and how to handle "what would you do differently?" with confidence rather than defensiveness.
>
> **Time:** 3–5 hours of preparation for one system; less on repeat.
>
> **What you'll be able to say:** "Here's the system, here's why each choice was made, here's what breaks and how I'd know, here's the CI/eval story, and here's what I'd invest in next." Delivered in 5 minutes, with the whiteboard drawn as you go.

---

## 1. Why This Matters

You can build an entire production stack and still bomb the interview if you can't communicate it. Conversely, a middling project explained brilliantly outperforms a great project mumbled through.

The bar for Senior+ AI Engineering roles isn't "can you build one thing?" — it's "can you *reason* about the trade-offs, own the gaps, and speak fluently across the whole lifecycle?" This project is deliberate rehearsal for that.

We build:
1. **The one picture** — one whiteboard-friendly architecture diagram
2. **The pitch** — a 5-minute story you tell
3. **The trade-off matrix** — decisions and their alternatives
4. **The failure catalog** — how it breaks and how you know
5. **The runbook of your own thinking** — anticipated follow-up questions with rehearsed answers
6. **The "next" story** — what you'd invest in if given another sprint

Reference system for the walk-through: the **production-hardened AI research service** from Month 6 (Capstone 08+09 combined).

### Architecture (the one whiteboard picture)
![IMG-CAP10-01](/7%20-%20Projects/images/IMG-CAP10-01.jpg)


### Prerequisites

- One system you've actually built (Capstone 05, 06, 08, or your own)
- Ability to draw the architecture from memory in 90 seconds
- Familiarity with the eval, monitoring, cost, and security patterns from earlier capstones

---

## 2. Step-by-Step Preparation

### Step 1 — Distill the architecture to ONE picture

Constraint: you must draw it on a whiteboard in 90 seconds. That forces you to keep the boxes minimal and the labels human.

Draw yours now (paper or Miro). Rules:
- ≤ 12 boxes
- Every arrow labeled
- No jargon in labels — say "rate limiter" not "SlidingWindowLimiter"
- Data flow left-to-right if possible; feedback loops as circular arrows
- CI/CD on one side, observability on the other

Practice drawing it 10 times until you can do it while talking.

### Step 2 — The 5-minute pitch (structure)

Rehearse this exact structure. The interviewer's brain fills in gaps *if* the shape is familiar; if you meander they lose the thread by minute 2.

```
30s — Problem & scope
       "The project is X. Users do Y. Success looks like Z."

60s — Architecture (draw as you talk)
       "At a high level: user hits [gateway], we auth, retrieve from [store],
        call [LLM], return with citations. Two supporting stacks: [CI/CD]
        and [observability]. Let me draw it."

90s — The three or four decisions I want to highlight
       "The interesting choices were:
        (1) Hybrid retrieval — why and result
        (2) Prompt caching — why and result
        (3) Cost cap per user — how and why
        (4) LLM-as-judge in CI — the key eval move"

60s — How it breaks and how we know
       "Failure modes we monitor: [X, Y, Z]. Alerts fire on [conditions].
        On-call follows runbooks. In practice we've hit [thing] once and
        recovered in [time]."

30s — What I'd do next
       "If I had another sprint: [concrete, specific, prioritized]."

10s — Explicit invitation
       "Happy to go deeper into any of that."
```

Time yourself. If you can't do it in 5 minutes, cut, don't rush.

### Step 3 — Build the trade-off matrix

Interviewers probe *why*. Have a matrix of decisions you can defend and alternatives you rejected with a reason.

| Decision | Alternative considered | Why we chose ours | What would flip it |
|----------|-----------------------|--------------------|--------------------|
| Fly.io | AWS ECS + ALB | Cost floor 10× lower; smaller team; global-by-default | If we hit compliance boundaries or need specific AWS services |
| pgvector | Pinecone / Weaviate | Postgres we already run; RLS for tenancy; no extra vendor | Billion-scale corpus; need managed |
| Sonnet as default | Opus everywhere | Quality/cost sweet spot on eval; routing escalates to Opus for hard | Sonnet quality drops below floor on our eval |
| Reranker (Cohere/local) | Skip reranking | RAGAS context-precision jumped 0.09 → 0.81 | Latency budget tightens under 500ms |
| LLM-as-judge in CI | Human review only | Scales to hundreds of cases per PR; judge calibrated κ=0.78 | If judge drifts or task complexity outgrows judge |
| Prompt caching | Recompute every time | 85% of input tokens served from cache; ~40% cost reduction | If prompt volatility becomes high |
| $ per user cap | Free tier globally | Prevents one abusive user from wrecking the bill; tiering as product | If B2C growth needs different pricing model |

The pattern is important: **every decision has a reason and a reversal condition.** That signals you thought about it, not that you copied a tutorial.

### Step 4 — Own the failures

Interviewers love this question: *"What could go wrong with your system?"* — because it distinguishes people who understand what they built from people who assembled it.

Prepare a **failure catalog**. Four rows per failure:

```
Failure: Anthropic API 5xx spike
Blast radius: All chat responses fail for the duration
Detection: SLO burn-rate fast-alert (2m window)
Mitigation: Circuit-breaker after 5 consecutive fails; fail-open to a
            degraded "we're having trouble, try again in a minute" response;
            optional: fail-over to OpenAI with adapter
```

Have 6–8 of these ready:
- Upstream LLM outage
- Redis outage (rate limit + cache)
- Postgres/pgvector outage
- Bad prompt version deployed
- Runaway agent / cost spike
- Prompt-injection attack in ingested content
- One user driving 90% of cost
- Slow deploy / rollback needed mid-incident

Practice: interviewer says "how does your system fail?" — you answer with three, not just one.

### Step 5 — Cost + latency numbers memorized

You should know your service's key numbers to two significant figures without hedging.

Sample memorization card:
```
Latency:
  TTFT p95: 550ms
  Total p99: 4.2s (streaming), 6.8s (non-streaming)

Throughput:
  Peak observed: 12 rps
  Design headroom: 40 rps per replica

Cost:
  $ per average request: $0.017 (Sonnet, 3k input, 500 output, 82% cache read)
  $ per user per month (median): $0.34
  $ per user per month (p99): $2.10 (capped at $5 by rate limiter)

Cache hit rates:
  Response cache: 22% (semantic layer)
  Provider prompt cache: 82% of input tokens served from cache

Eval scores (last run):
  Golden set pass rate: 100% (40 cases)
  RAGAS faithfulness: 0.87
  RAGAS answer relevance: 0.82
  Cohen's kappa (judge vs human): 0.78

SLO:
  Availability: 99.9% target (0.1% error budget = 43 min/mo)
  Latency: p99 < 5s for /chat
  Current 30d burn: 34% of monthly budget consumed
```

If asked "what's your latency?", say *"p95 TTFT is 550ms, full generation p99 4.2s"* — precise beats waving.

### Step 6 — Practice the deep-dive on each subsystem

For each of the following, be ready with a 90-second explanation and a follow-up question in your back pocket:

**Retrieval (RAG)**
- Q: "How does retrieval work?"
- Structure: chunk → embed → hybrid (vector + BM25) → RRF → rerank (cross-encoder) → top-K feeds LLM prompt.
- Follow-up expected: "How do you tune K?" (empirically; RAGAS context-precision on golden set; too high hurts precision and cost, too low hurts recall).

**LLM client**
- Q: "How do you call the LLM?"
- Structure: async client, retry with jitter capped at 60s, streaming for TTFT, structured output validation via Pydantic + retry-with-error, prompt cache for system prompt, cost accounting per call.
- Follow-up: "What if the API is down?" (retry, circuit-breaker, then degraded response or fail-over).

**Agent loop** (if applicable)
- Q: "How does the agent work?"
- Structure: ReAct — Think → Act (tool call) → Observe → repeat. Three simultaneous budgets: max_iters, max_cost, max_wall_time. Each halts. Tool boundary enforces SSRF prevention, output truncation, timeout.
- Follow-up: "How do you prevent it from looping?" (iteration cap + duplicate-call detection + no-progress detection).

**Auth**
- Q: "How does auth work?"
- Structure: OIDC (Auth0/Cognito) issues JWT; backend validates via JWKS with pinned algorithm; scope check per endpoint; `tenant_id` from JWT claims; Postgres RLS enforces per-row.
- Follow-up: "What if a JWT leaks?" (short expiry, refresh tokens revocable server-side; detected via unusual usage; rotate signing key if key leaks).

**Evaluation**
- Q: "How do you know when the model regresses?"
- Structure: golden set on every PR (blocks merge); nightly full eval with LLM-judge (calibrated κ=0.78); RAGAS on RAG-specific metrics; sampled prod traces with user thumbs.
- Follow-up: "How would you know if the judge itself drifted?" (recalibrate quarterly against fresh human labels; monitor score distribution shifts).

**Monitoring**
- Q: "What do you monitor?"
- Structure: RED + LLM ($ per user, tokens/sec, cache hit rate, refusal rate) + USE. SLOs with multi-window burn-rate alerts. Runbook per alert.
- Follow-up: "Which single metric matters most?" (my answer: SLO burn-rate composite — because it's user-facing and time-integrated; but I'd argue cost/hour is a close second for LLM apps).

**Cost**
- Q: "How do you keep costs under control?"
- Structure: model routing (Haiku/Sonnet/Opus by intent), prompt caching (85% cached-token rate), response cache, per-user $/day cap, output length discipline (max_tokens, "no preamble"), batch API for offline.
- Follow-up: "What's your break-even for self-hosting?" (~1T tokens/mo, dependent on utilization, model size, team capacity).

### Step 7 — The "what would you do differently?" answer

Interviewers ask this to see if you can self-critique. **Have three specific things.** Not "I'd write more tests" (vague, disarming). Something like:

```
Three things I'd change:

1. I built this monolithically. The retrieval + generation + eval modules
   should really be their own services, both for scaling and because
   retrieval will need its own team eventually.

2. I bolted on evaluation late. Next time I'd start with an eval harness
   and let it drive the model choice, not the other way around.

3. My cost accounting is in-band with request handling. In hindsight I
   should have offloaded the price-table calc to a background job so a
   pricing config bug wouldn't stall request handling.
```

Each is specific, technical, non-apologetic. It shows depth without weakness.

### Step 8 — Handling questions you don't know

You will get asked about something you haven't done. Rehearse:

*"I haven't personally built that, but the shape of the problem is [X]. The trade-offs I'd think about are [A vs B]. What I'd want to learn quickly is [the specific thing]."*

Notice: no "I don't know" full stop. Also no bluffing. You're demonstrating that you can reason from adjacent knowledge — which is what senior roles actually require.

### Step 9 — The one killer follow-up you ask them

Interviews are two-way. At the end when they ask "any questions for us?", have a specific one that shows you thought:

- **For an eval-heavy role:** "How do you decide what quality metric to optimize when the LLM-judge disagrees with human reviewers?"
- **For an infra-heavy role:** "What's your P1 incident cadence, and how do you decide whether to freeze deploys during error-budget burn?"
- **For a product-heavy role:** "What's the highest-leverage cost lever you've pulled on your production stack?"

They'll remember you as someone who asked a real question, not "what's the culture like."

### Step 10 — Whiteboard rehearsal

Set a phone timer for 5 minutes. Stand at a whiteboard (or big paper). Explain your architecture to an imaginary interviewer. Record yourself.

Watch it back. Mark:
- Where you paused
- Where you said "um" or "so basically"
- Where the drawing outran the explanation (or vice versa)
- Where you undersold something (self-deprecation is a bug)
- Where you meandered

Redo. Every rep gets tighter. By rep 5-8 you'll have a smooth 5-minute story. That's the deliverable.

`[IMG-CAP10-02]` — *Prompt: A composite illustration of an engineer at a whiteboard mid-explanation. On the whiteboard, a clean hand-drawn architecture diagram of a production AI service with labeled boxes: user, gateway, FastAPI, Redis, pgvector, Anthropic, LangFuse, CI/CD arrow. To the side, three bullet points visible: "1. Hybrid retrieval → +9pt precision", "2. Prompt cache → -40% cost", "3. $ cap per user → prevented runaway". A stopwatch icon showing "4:38" indicating pacing. The mood: focused but confident, professional. Isometric or clean flat-illustration style, warm palette.*

---

## 3. Common Mistakes to Avoid

**Mistake #1 — Overloaded diagram.** Twenty-five boxes with lines crossing everywhere. Interviewer can't follow; assumes you can't communicate. Rule: ≤ 12 boxes; consolidate.

**Mistake #2 — Defensive on "what could go wrong."** "Nothing really — we haven't seen issues." Instant credibility loss. Every real system has failure modes; not knowing them means not watching for them.

**Mistake #3 — Buzzword-heavy.** "We used LangGraph with ReAct agents doing tool-calling via MCP with a vector-space semantic cache." Nobody knows if you built it or read about it. Talk mechanism: "the agent loops through: pick a tool, run it, feed the result back, repeat, up to 20 iterations or a $0.50 cap."

**Mistake #4 — Undercutting yourself.** "It's just a small project" / "I'm not sure this is production-grade." Interviewer takes you at your word. If you shipped it, own it. Constructive self-critique later is fine; opening with weakness is a bug.

**Mistake #5 — Numbers by feel.** "It's pretty fast." Nope. "TTFT p95 is 550ms." Specific numbers signal you measured. If you don't have real numbers, get some before the interview.

**Mistake #6 — Not distinguishing "I built" vs "I know."** Interviewers respect "I read the paper" and "I've integrated this at work" as different claims. Conflate them and you'll get caught on a follow-up.

**Mistake #7 — Ignoring the interviewer's expression.** If they check their phone, you're too long. If they lean in, deep-dive that part. Read the room.

**Mistake #8 — Ending flat.** "So yeah, that's the system." End with your killer forward statement: "*If I had another sprint I'd do X, Y, Z. Happy to dive into anything.*" That last sentence hands the mic back gracefully.

---

## 4. The Anti-Cheat Sheet (Things You Must Not Do)

- Don't claim tools/techniques you haven't used
- Don't say "AI" when you mean "LLM" or "ML" — precision matters at this level
- Don't quote pricing you haven't checked recently (Anthropic and OpenAI prices move)
- Don't claim eval numbers you didn't compute
- Don't skip the "what could break" question
- Don't say "I'd just use LangChain" without owning why (they'll ask)
- Don't downplay ops work (deploys, monitoring, on-call) — senior roles depend on it

---

## 5. Interviewer-Specific Adaptations

Adjust emphasis by who's across the table:

**ML Engineer / DS interviewer:** lean into eval, calibration, data leakage checks, model routing. They will want to hear you say "log-loss" and "Cohen's kappa" and "PSI".

**Infra / Platform Engineer:** lean into SLOs, deploy strategies, rollback drills, cost, secrets management. They want ops maturity.

**Product / Founder / CTO:** lean into what users see and cost per user; the business framing of technical decisions.

**Security-oriented:** lead with tenant isolation, RLS, prompt injection defenses, secrets, audit trail. They want to know you thought about attackers.

**Researcher / applied scientist:** be ready to talk about *why* transformers work, tokenization edge cases, sampling temperature, and the current pace of the field. This is where the Month 3 material earns its keep.

---

## 6. Sample 5-Minute Script (Template)

Use this as a prompt for your own version. Fill in blanks specific to your build.

> "The project I'll walk through is a research-assistant service — users ask a question, the system searches, reads sources, and returns a cited report. Latency budget under 10 seconds, cost budget under 20 cents per query, and it can't hallucinate — every claim needs a source.
>
> [START DRAWING]
>
> At the top, the user hits our HTTPS endpoint. We authenticate via OIDC — Auth0 issues a JWT, backend verifies via JWKS. The FastAPI app is behind Fly.io's edge routing, two replicas, rolling deploys.
>
> Inside the app, three subsystems. First, retrieval: pgvector for embeddings, Postgres full-text for BM25, we do hybrid retrieval with reciprocal rank fusion, then rerank the top 50 with a cross-encoder to top 5.
>
> Second, generation: Anthropic Sonnet with a heavily-cached system prompt — about 85% of our input tokens hit the prompt cache, so cost per query is around $0.017 average.
>
> Third, the agent loop: for multi-step questions, we run a ReAct-style loop with 4 tools — search, fetch, extract, note. Three simultaneous budgets bound it: 20 iterations, $0.50 max cost, 5 minutes wall time.
>
> Around all of that: on the left, GitHub Actions runs lint, tests, and a golden-set eval on every PR; a 40-case set that must pass 100% to merge. Nightly full eval with LLM-judge scoring — Opus judges Sonnet, calibrated against human labels at κ=0.78. On the right, Prometheus for metrics, LangFuse for LLM traces, Grafana for dashboards, PagerDuty for pages driven by SLO burn-rate alerts.
>
> [FINISH DRAWING; 90 seconds elapsed]
>
> The three interesting choices: hybrid retrieval with rerank — that alone moved RAGAS faithfulness from 0.71 to 0.87. Prompt caching — cut input costs about 40% with no other change. Per-user daily $ cap in Redis — has already stopped one runaway integration from wrecking the monthly bill.
>
> Failure modes we watch for: Anthropic 5xx spikes (SLO burn-rate alert, retry+circuit-breaker); cache-hit-rate drops (alert; probably corpus refresh broke keys); cost spike (alert on 15-min rate > 3× baseline); agent loops (iteration cap fires, we log and investigate).
>
> If I had another sprint: one, I'd split retrieval into its own service — it's growing its own quality logic. Two, I'd add pairwise LLM-judging alongside absolute scoring — more reliable at ranking. Three, I'd wire self-hosted Llama 3.1 8B as a fallback for classification-shape queries — probably 20-30% cost reduction on the FAQ traffic.
>
> Happy to go deeper into any of that."

That's ~700 words at natural pace = 4:30-5:00. Practice.

---

## 7. Extensions

- **Record a talk version.** 20-min conference-style deep-dive; upload to your portfolio. Recruiters love it.
- **Blog post.** Same content, written form. Post on your site.
- **Do a mock interview.** With a friend, ex-colleague, or paid coach. Their critique is 10× more useful than your own.
- **Study a real system architecture.** Read a good post-mortem or engineering blog (Netflix, Stripe, Anthropic) and diagram it. Pattern-match to your own.

---

## 8. Interview Talking Points

Meta-talking-points — how to talk about *this project* if asked "how do you prepare for interviews?":

- "I built the system, then I built an interview version — the picture, the pitch, the trade-off matrix, the failure catalog. I rehearse the 5-min pitch on a whiteboard until it's smooth."
- "I keep a cost/latency/eval-scores card memorized for every project. Precision signals I measured."
- "I don't defend decisions I can't defend. If I chose Fly over AWS because I hadn't set up AWS deeply, I say that."
- "My favorite question is 'how does this fail?' — because that's the one where seniors and mids diverge."

---

## 9. Reference: Well-Known Interview Frames You'll Hear

- **"Describe a system you built."** — the 5-min pitch above.
- **"Design an X."** — different beast; use the same "start with problem, then components, then trade-offs, then failures" structure but you're inventing it live.
- **"Why did you use Y?"** — trade-off matrix.
- **"What would you do differently?"** — three specific things, no self-deprecation.
- **"How does it fail?"** — failure catalog, three examples.
- **"How do you know it works?"** — eval story.
- **"How would you scale this 100×?"** — different bottlenecks at different scales; usually retrieval or cost.

---

## 10. Final Checklist

Before an interview where this is the flagship project:

- [ ] Diagram drawn from memory in < 90s (tested this week)
- [ ] 5-min pitch rehearsed in one clean take (recorded self)
- [ ] Cost/latency/eval numbers memorized to 2 sig figs
- [ ] Trade-off matrix — 7 decisions with alternatives and reversal conditions
- [ ] Failure catalog — 8 failure modes with detection + mitigation
- [ ] Three "would-do-differently" specifics, each technical
- [ ] One killer question for them
- [ ] Recent Anthropic pricing memorized (or admitted uncertainty rather than guessed)
- [ ] Github link to project, README polished, deploy URL live

Do all ten. That's a senior-level interview posture, earned rather than performed.

---

## 11. References

- *Cracking the Coding Interview* — outdated on ML/AI, but the meta-technique of "explain your solution as you code" is on-point
- *System Design Interview* (Alex Xu) — the shape of a good architecture pitch
- Julia Evans' zines on debugging and systems — how to *think* about the components
- Charity Majors — blog posts on ops maturity signals
- Any post-mortem library (K8s, Stripe, AWS) — models of how professionals write about their systems

---

## Closing Note

You're not selling the project. You're selling **the way you think about the project** — that's what senior interviewers hire. The system is the vehicle; the reasoning is the payload.

Good luck.
