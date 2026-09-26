# Career 02 — The Interview Loop (Beyond the Architecture Pitch)

> **Goal:** Survive and win a Senior AI Engineer interview loop. Capstone 10 covered the architecture pitch. This covers the rest: recruiter screen, behavioral interviews, coding rounds, take-homes, system design (AI-specific), reference calls. Aim for signal fluency across the whole loop.

---

## The Shape of a 2026 Senior AI Engineer Loop

Loops vary but most follow this shape at senior/staff level:

| Stage | Interviewers | Length | What they're testing |
|-------|-------------|--------|---------------------|
| **Recruiter screen** | Recruiter | 20-30 min | Signal fit; comp expectations; timeline |
| **Hiring manager call** | HM | 45 min | Motivation; culture; broad experience |
| **Technical screen** | Engineer | 45-60 min | Coding OR system-design intro |
| **Take-home** (some cos.) | Async | 4-8 hrs | Real work sample |
| **Onsite / virtual loop** | 4-6 interviewers | Half day | Depth on multiple axes |
| **Bar-raiser / cross-functional** | Senior IC | 45 min | Standards check |
| **Reference calls** | 2-3 references | 30 min each | Confirmation |

Onsite loops typically include:
- **Coding** (1-2 rounds) — general algorithms + LLM-adjacent
- **System design** (1-2 rounds) — one general, one AI-specific
- **Architecture deep-dive** (1 round) — walk through a past project you built
- **Behavioral / leadership** (1 round) — the "tell me about a time" round

Below: how to prep each.

---

## The Recruiter Screen

Skills: brevity, timeline clarity, comp calibration.

**What they ask:**
- Tell me about yourself (60 seconds max)
- Why are you looking?
- Where are you geographically? Remote OK?
- Any current interview processes?
- What's your target comp?
- What's your ideal role?
- When are you looking to start?

**Your 60-second self-introduction template:**
> "I'm Brian, senior software engineer with 10+ years of production experience. Currently at the Montana Legislative Branch, where I build a bill-drafting system used by the whole state legislature — Java/Spring/Kafka/K8s. In the last year I've moved deep into production AI: RAG, agents, fine-tuning, LLM ops. I have a Google Cloud Professional Architect cert and a Master's from [University] finished this year. I'm looking for a remote Senior AI Engineer role at a company where the AI work is core to the product — either scaling a solid team or joining an ambitious early one."

**Comp expectations answer:**
Don't drop a specific number first if you can avoid it. Answers by acceptability:

- **Best**: "I don't have a specific number yet — I'd like to understand the scope and level first. Can you share the range for this role?"
- **Acceptable**: "My target is in the range [X] to [Y] total comp for a senior IC AI role, depending on level, equity, and location. I'm flexible on the mix."
- **Avoid**: dropping an exact figure before knowing what the role actually is.

Some recruiters will insist. If pushed, give a **range** with a wide top: "$[low] to $[high]+ depending on role and location" — this leaves room for their offer to land above your midpoint.

See `career/03-comp-and-negotiation.md` for how to figure out those numbers.

**Timeline answer:**
Truthful, not desperate. "I'm actively looking but not in a rush — I want a strong match. I can start with 4-6 weeks notice at my current role. I'm interviewing at several companies now."

If you're not — say you're evaluating a few opportunities. Never say "you're my only interview."

---

## The Hiring Manager Call

Skills: matching your story to their needs; asking sharp questions back.

Come with:
- A **crisp 60-second version of yourself** (see above)
- 3-4 **specific past projects** you can talk about in ~2-3 minutes each
- 3-4 **questions** for them that make you look thoughtful

**Questions to ask the hiring manager:**
- What are the top 2-3 outcomes you want this role to drive in year 1?
- Who else would I work most closely with?
- Where is the team on the "shipping vs learning" spectrum right now?
- What's the most exciting problem the team's chewing on?
- What made the last person in this role successful — or fall short?
- How does the team make decisions about model choice (proprietary vs open, hosted vs self-hosted)?
- What does the on-call / operations rhythm look like?

**Do NOT ask:**
- Cultural fluff you can find on their careers page ("what's the culture like?")
- Compensation (that's for the recruiter)
- Vague signals of doubt ("Is the funding solid?" — asking directly is fine; hedging is uncomfortable)

**The hiring manager is your biggest ally when things go well.** They advocate internally, negotiate offer components, and often decide the final level. Build rapport.

---

## Coding Rounds

For Senior AI Engineer roles in 2026, the coding round typically **is not LeetCode 30-minute puzzles anymore**. Common formats:

**Format A — Practical LLM-adjacent coding.** "Build a rate limiter for LLM API calls with different limits per user tier." Or "Write a function that takes a prompt template and structured input, formats it correctly, and validates the response against a Pydantic schema." Or "Given this failing prompt, debug why and fix it."

Prep: implement the standard patterns from Capstone 04 and Month 6 (rate limiting, structured output validation, retry logic). Be able to write them without notes.

**Format B — Modest algorithmic.** Simplified LeetCode-style, medium difficulty, expected in 30-40 minutes with dialogue. Two-pointer, hashmap, simple DP. Not the hard graph problems.

Prep: NeetCode's "top 75" is the standard recommendation; take a week to grind if algorithms are rusty. Focus on speed at medium problems more than solving hards.

**Format C — Debugging a codebase.** You're given a small repo with a broken function; find and fix. Signals real-world debugging.

Prep: practice reading unfamiliar code fast. `git log`, tests-first, print-statements-before-fancy-debuggers.

**Format D — Pair programming a small feature.** Interviewer plays product/tech lead; you build a small thing end-to-end. "Add a caching layer to this LLM client." "Add pagination to this endpoint."

Prep: this is genuine practice; the capstones already teach it. Do a few timed builds.

**General rules for any coding round:**
- **Narrate as you go.** Silence = interviewer confusion about your thinking.
- **Clarify the requirements first.** Ask 2-3 clarifying questions before writing.
- **Write tests first (or discuss them upfront).** Even a stubbed test signals discipline.
- **Handle edge cases explicitly** — empty input, None, off-by-one.
- **Keep the interviewer in the loop about trade-offs.** "I'll use a dict here for O(1) lookup; if this needed to scale we'd move to Redis."
- **Do NOT bluff Python idioms you don't know.** "I'd need to look up the exact signature of `functools.lru_cache` here" is fine; making up an incorrect one is fatal.

---

## System Design (AI-Specific)

The most-tested round for Senior AI Engineers. Common prompts:

- "Design a customer support chatbot for a mid-size SaaS."
- "Design a semantic search across a company's internal docs."
- "Design an AI code review assistant."
- "Design a system that generates weekly reports from a data warehouse."
- "Design an AI-powered content moderation pipeline."

The interviewer wants to see:
1. **You gather requirements** — clarify scope, users, scale, constraints
2. **You draw the architecture** — components + data flow + trade-offs
3. **You dive deep on 2-3 specific parts** when prompted
4. **You handle scale** — from 10 to 10M users
5. **You mention operations** — eval, monitoring, cost, security, on-call
6. **You handle failure modes** — what breaks, how you know, how you recover

**The 5-minute AI system design template:**

```
Minute 0-1: Clarify requirements
  - Who's the user? What do they want out of it?
  - What scale — DAU, requests/sec, growth expectations?
  - Latency budget?
  - Any regulatory / privacy constraints?
  - Existing infra we should assume or design against?

Minute 1-3: High-level architecture
  Draw: user → gateway → auth → API → LLM + retrieval → response
  Name specific pieces: FastAPI, pgvector, Redis, Anthropic Sonnet, LangFuse.
  Explain the request path.
  Note where cost/latency live.

Minute 3-4: Deep-dive prompts from the interviewer
  Common ones:
  - "How does retrieval work?" → chunking, hybrid, rerank, RAGAS eval
  - "How do you handle scale?" → autoscaling, caching, rate limits
  - "How do you evaluate?" → golden set + LLM-judge + user feedback
  - "How does it fail?" → 3 specific failure modes + detection + recovery
  - "How do you keep costs down?" → routing, caching, batching, per-user caps

Minute 4-5: Wrap
  Summarize the trade-offs you made.
  Name 2-3 things you'd do if you had more time.
```

Every capstone in your curriculum maps to this template. **The whole point of the six-month track is that you can improvise this now** — Capstone 10 is the deliberate rehearsal.

**System design non-obvious tips:**
- **Draw first, talk while drawing.** Silent designers seem lost.
- **Number your components** so the interviewer can point at them.
- **Explicitly name what you're NOT covering** so scope doesn't drift ("I'm going to focus on the request path and eval; happy to dive into deploy or auth if you want").
- **When you don't know something, say so and reason.** "I haven't personally used Milvus, but the choice between it and Pinecone comes down to X and Y — for this scale I'd default to pgvector because Z."

---

## The Behavioral / Leadership Round

Underestimated by engineers, over-weighted by hiring committees. This is the "tell me about a time..." round.

**Bank of stories to prepare** (aim for 8-10):
- A time you led a technical project end-to-end
- A time you disagreed with a decision and how you handled it
- A time you shipped something and it broke
- A time you had to work with a difficult stakeholder
- A time you had to say no to a request from leadership
- A time you mentored someone
- A time you missed a deadline
- A time you made a technical trade-off you now regret
- A time you had to learn something new fast
- A biggest career mistake and what you learned

**Use the STAR framework** (Situation, Task, Action, Result). Two minutes per story.

**Sample story structure — "a time you shipped something and it broke":**

> S: "We deployed a new bill-status API endpoint to legmt.gov the Monday before session opened. Traffic was 5× normal from staffers preparing for the week."
> 
> T: "By 10am we had multiple reports of stale data and 500 errors. Session opened Wednesday; I was on call. The team was three people that day."
> 
> A: "First I stabilized: rolled back the deploy to the previous version, which restored 90% of traffic. Then I dove into the traces — turned out our new Kafka consumer was double-processing on restart, mangling the local cache. I wrote a repro in a scratch script by 2pm, patched the consumer idempotency check, added a regression test, and shipped forward by 4pm. Post-mortem the next day surfaced two other places we needed idempotent consumers; I filed those and one got prioritized for the sprint."
> 
> R: "No stale-data reports the rest of the week. Session opened Wednesday without incident. Team adopted an idempotency-first pattern for Kafka consumers going forward, which prevented a similar class of bug on the calendar workflow six months later. Post-mortem template we wrote for that incident is still in use."

Notice: specific numbers, ownership without blaming others, actual technical detail, and a positive downstream outcome.

**Pitfalls in behavioral rounds:**
- **Blaming teammates / previous companies.** Even if true, it's a signal you'll blame the next team. Frame around your actions, not others' failures.
- **Vague stories.** "I helped optimize a slow query" is nothing. "I found a 300ms N+1 in the bill-search endpoint by profiling; refactored to a single JOIN with an index; p95 dropped from 800ms to 120ms" is a story.
- **Making everything a solo hero moment.** Senior engineers work with others. Include collaboration, mentoring, cross-team stuff.
- **Zero self-critique.** "I've never made a real mistake" fails the round instantly. Have at least one story where you were wrong, learned, changed.

---

## Take-Home Assignments

Some companies (mostly smaller / startups) send a take-home. Typical shapes:

**Type 1 — "Build a small AI service."** Given prompt: "Build a service that classifies support tickets into 5 categories using an LLM. Bonus for evals."

Prep: this is literally what your capstones taught. Reuse the patterns; polish it as a small standalone repo.

**Type 2 — "Improve/debug an existing repo."** Given: repo with a broken RAG service. Fix the bug and improve retrieval quality.

Prep: read code fast; measure before changing; keep a diff-focused change; write up what you did.

**Type 3 — "Analyze and design."** Given: a scenario. Produce a design doc (no code) with architecture, trade-offs, roadmap.

Prep: use the Capstone 10 structure; be concrete; number your decisions.

**Take-home rules:**
- Ship on time. Late is 90% of failures.
- Don't over-engineer. Match the scope they asked for. Add 1-2 "nice touches" (a test suite, a README with results, a Dockerfile).
- Include a write-up. `WRITEUP.md` at repo root explaining what you built, what you didn't have time for, and what you'd do differently. This document is often the deciding factor.
- Track time. If they said "4 hours max," respect it. Spending 30 hours on a 4-hour take-home signals judgment problems.

---

## The Bar-Raiser / Cross-Functional Round

Amazon-style. A senior engineer from outside the team asks tough questions, looking for signals of company-wide standards. Similar rounds exist at Meta, Anthropic, and many mid-size companies.

**What they test:**
- Do you have opinions on your own work?
- Would you push back on a bad decision from senior leadership?
- Can you critique your own past projects?
- Are you self-aware about strengths / weaknesses?

**Sample questions:**
- What's your biggest career mistake?
- What's a project you'd redo differently if you started over?
- Describe a time you were wrong.
- What's an unpopular technical opinion you hold?

Answer these **specifically and honestly**. Fake modesty ("my greatest weakness is caring too much") fails. Genuine self-knowledge ("I under-invested in test coverage on my first big system and it bit me — I now default to writing tests first") passes.

---

## References

Almost every senior loop ends with 2-3 reference calls. Choose references who:
- Worked with you closely (manager, senior peer, or someone you managed/mentored)
- Can speak specifically to your technical work
- Won't be surprised — brief them in advance, share the JD, remind them of specific projects

**Brief your references** with a short email: "The role is [X] at [Y], hiring manager is [Z]. They'll ask about our work on [specific project] and my role in [specific decision]. Feel free to mention [these outcomes]. Thanks for the vote of confidence."

Bad references (silent, vague, "he's fine I guess") kill offers late. Good references seal them.

For Brian: colleagues at Montana Legislative Branch are natural. If you can find a former manager, senior peer from an earlier role, and one client-facing person, that's a strong reference panel.

---

## Question Bank to Rehearse

Practice these out loud, timed. Record yourself once and cringe-watch.

- Tell me about yourself (60 sec)
- Why are you looking for a new role? (60 sec)
- Walk me through your most impressive project (5 min)
- Design a RAG system for [X] (system design, 45 min)
- Tell me about a time you disagreed with a technical decision (STAR, 2 min)
- What's your biggest weakness / mistake (2 min)
- Why us specifically? (60 sec — tailor per company)
- What questions do you have for us? (3-4 sharp ones)

---

## The Meta-Move

The people who win senior interview loops:

1. **Are already interviewing elsewhere.** BATNA (best alternative to a negotiated agreement) is leverage. Interview at 3-5 places in parallel. Even losses inform your best offer.
2. **Know their story cold.** Same 8-10 stories, adapted for each interviewer. Not scripts — internalized narratives.
3. **Do their homework.** Read the company's engineering blog before the loop. Ask questions that show it.
4. **Manage energy.** A 4-hour virtual onsite is exhausting. Sleep the night before. Eat before you start. Blocked schedule the day of.
5. **Debrief after each interview.** Notes on what went well, what didn't, what to change for next.

---

## Post-Loop

After the loop wraps, send a thank-you note to the recruiter (and hiring manager if you have the address). Two paragraphs: appreciation, one specific thing that resonated, reiteration of interest.

If you get an offer: see `career/03-comp-and-negotiation.md`.

If you don't: ask for feedback (some give it, most don't); check if the door is open in 6 months; move on.

---

**Related files:**
- `career/01-portfolio-and-signal.md` — the surface area that opens the loop
- `career/03-comp-and-negotiation.md` — after the offer
- `capstones/10-interview-explain-architecture.md` — the architecture-pitch deep dive
