# Career 01 — Portfolio & Public Signal

> **Goal:** Turn the six-month curriculum + capstones + fine-tuning + self-hosting work into a public, credible portfolio that a remote Senior AI Engineer recruiter can screen in under 5 minutes and reach a positive verdict. The corpus is the substance; this document is the distribution.

---

## The Recruiter's 5-Minute Path

When a senior AI engineering role opens, a recruiter typically screens candidates by clicking three links:

1. **Your LinkedIn profile** — 30 seconds
2. **Your GitHub or portfolio site** — 2 minutes
3. **Your résumé PDF** — 2 minutes

Time budget: ~5 minutes. Decision: pass to hiring manager, or reject.

If your surface area doesn't clearly answer "does this person build production AI systems?" in that time, you're rejected — regardless of what you actually know. This module is about making that yes obvious.

---

## The Four-Surface Framework

Build these four surfaces in this order:

### Surface 1 — GitHub organization with 3-5 flagship repositories

Not a hundred half-finished repos; not one giant monorepo. **Three to five polished, deployable, README-first projects.**

For your curriculum, the natural flagships are:

**Flagship 1: `chat-with-your-docs` (from Capstone 05)** — RAG service with hybrid retrieval, cross-encoder rerank, grounded generation, RAGAS evaluation, FastAPI + Docker + deployed URL. **This is the anchor.** Every senior AI engineer recruiter wants to see one of these done right.

**Flagship 2: `production-ai-service` (from Month 6 build)** — the hardened service with dashboards, alerts, cost tracking, security. Screenshots of Grafana; runbooks. Signals ops maturity.

**Flagship 3: `research-agent` (from Capstone 06)** — ReAct agent with tool budgets, memory, guardrails, evaluation. Signals agent understanding beyond frameworks.

**Flagship 4 (optional but strong): `llama-finetune-legal-classifier` (from fine-tuning capstone)** — QLoRA fine-tune with dataset curation, evaluation, deployment. Signals you can go deep.

**Flagship 5 (optional): `self-host-llm`** — vLLM deployment with the provider router. Signals infra depth.

For each flagship:

**README structure:**
```
# Project Name

One-sentence description with a screenshot or GIF.

## What it does
2-3 paragraphs.

## Architecture
Diagram (mermaid or an image).

## Live demo
URL if you can (free-tier Fly/Cloud Run keeps it cheap).

## Key results
- Metric 1: number
- Metric 2: number
- Metric 3: number

## Stack
Bullet list of technologies.

## How to run locally
Three commands.

## What I'd do next
Three bullets.

## Why this project
One paragraph about the interesting decisions and trade-offs.
```

**Anti-patterns to avoid:**
- Auto-generated README with no substance
- Screenshots of the code (nobody wants to squint at a code screenshot when the code is right there)
- "Uses OpenAI API" listed as if it's a technology
- 200 lines of "how to install pip" filler
- Emojis every 3 words

**The one screenshot rule:** every flagship repo has at least one image in the README — a screenshot of the deployed UI, a Grafana dashboard, an architecture diagram, or an eval-results chart. Skimmers stop at pictures.

### Surface 2 — Portfolio site

Optional but valuable. A simple static site (Astro, Next.js, or plain HTML) at `brianakashaba.dev` or similar. Content:

- **Home page**: name, one-line pitch ("Senior software engineer building production AI systems"), photo, links to LinkedIn/GitHub/Twitter/email
- **Projects page**: card view of your 3-5 flagships, each with a link, a screenshot, and 2-3 sentence description
- **Writing page**: your blog posts (Surface 3)
- **About page**: longer version of your bio, timeline of roles, values

Hosting: GitHub Pages, Cloudflare Pages, Vercel free tier. Custom domain ~$12/yr.

**Is this necessary?** No. But a portfolio site signals seriousness. It's also fully in your control — LinkedIn can throttle you, GitHub can hide activity, but your site is yours forever.

Time investment: one weekend to launch, ~1 hour/month to maintain.

### Surface 3 — Public writing (blog + LinkedIn)

The highest-leverage credibility builder. Two channels:

**Channel A — Long-form blog**

Aim for 3-5 flagship posts, 1500-3000 words each. Topics from the curriculum:

- "Building a RAG system that doesn't hallucinate: what I learned from 200 evals" (Capstone 05 write-up)
- "Fine-tuning Llama 3.1 on a $10 GPU budget: results and honest surprises" (Fine-tuning capstone write-up)
- "The three cost levers that cut our LLM bill by 90%" (Month 6 Module 14 write-up)
- "SLOs for AI services: why cost belongs in your on-call dashboard" (Month 6 Module 07 + 14)
- "Prompt injection defense: what actually works in production" (Month 5 + Month 6 Module 13)

Style rules for these posts:
- Open with a concrete result ("Our chatbot hit 87% eval score. Here's how.")
- Include numbers, screenshots, and code
- Assume a technical reader; don't over-explain basics
- Link generously to your own code
- End with a specific take or opinion, not a "hope this helps!" throwaway
- Under 15 minutes read time

Publishing platforms: your own site (best), or Medium / Substack / dev.to / Hashnode. Substack has the best distribution for growing an audience; your own site has the best long-term equity.

Cadence: 1 post/month is plenty. Consistency > volume.

**Channel B — LinkedIn posts**

Short-form (150-400 words) posts, 1-2 per week. Content types that perform for AI engineers:

- **Small technical wins** ("Cut our LLM cost 30% by fixing the prompt cache. Turns out we were invalidating it every request.")
- **Contrarian takes** ("Everyone's doing DPO. For our task, plain SFT with better data outperformed by 4 points. Here's why...")
- **Concrete recipes** ("If you're getting started with RAG, don't do X. Do this instead:")
- **Reference material summaries** ("Read the vLLM paper this week. Three things worth knowing:")

Avoid:
- Hustle bro platitudes ("Wake up at 5am to grind")
- Unearned confidence ("I've mastered AI in 6 months")
- Vague thought-leadering with no code
- Reposts of trending news without your own analysis

LinkedIn's algorithm rewards consistency + engagement in the first hour. Post at 9am Mountain / noon Eastern for US audience; morning UK time for European.

For Brian specifically: leverage the Montana Legislative Branch angle — it's memorable, unusual, and gives credibility ("built a system that generates $X million/year in taxpayer value"). Don't overshare specifics obviously; anonymize as needed.

### Surface 4 — Résumé (PDF)

Yes, still. A clean 1-page PDF that lists:

**Header**
- Name + title ("Senior Software Engineer / AI Engineer")
- Location + remote-preference ("Helena, MT, USA — Remote (US, EU)")
- Links: email, LinkedIn, GitHub, portfolio site

**Summary (2-3 lines)**
"Senior software engineer with 10+ years experience shipping production systems, recently focused on production AI engineering: RAG, agents, fine-tuning, LLM ops. Google Cloud Professional Architect, MS in Computer Science."

**Experience (reverse chronological)**
For each role, 3-5 bullets. Each bullet:
- **Action verb** + **specific thing done** + **measurable outcome** where possible
- ✗ "Worked on Kafka"
- ✓ "Redesigned message replay path in a Kafka-based bill drafting system, reducing session recovery time from 45s to 3s"

For the LegMT role specifically, lean into: scale (Montana's whole legislature relies on it), technical depth (Kafka, Spring Boot, K8s, Azure AD B2C — Java experience is a differentiator for AI roles because most AI people are Python-only), reliability (uptime numbers if you have them), and stakeholder communication (elected officials, staff, other agencies).

**Projects (this is where the AI portfolio shows up)**
3-5 bullets, one per flagship. Each: name, one-line description, link, tech stack.
- **chat-with-your-docs** — Full RAG service with hybrid retrieval, cross-encoder rerank, RAGAS eval. Faithfulness 0.87. Live at [URL]. FastAPI, Postgres+pgvector, LangFuse, Docker on Fly.io. [Github link]

**Education**
- MS Computer Science, [University], June 2025
- BS Computer Science or equivalent, [University], [year]

**Certifications**
- Google Cloud Professional Cloud Architect, [year]
- Any others

**Skills (compact list)**
Core: Python, Java, TypeScript, SQL. AI/ML: PyTorch, HuggingFace, vLLM, LangGraph, RAG, fine-tuning (QLoRA, DPO). Infra: Docker, Kubernetes, GitHub Actions, Fly.io/GCP/AWS, Postgres, Redis, Kafka.

**Design rules:**
- One page. Rare exceptions for 15+ year staff engineers with legitimate density.
- Clean sans-serif font (Inter, Söhne, Helvetica). 10-11 pt.
- Enough white space to breathe.
- PDF format. Filename: `Brian_Akashaba_Resume_2026.pdf`.
- No photo, no age, no marital status.
- ATS-friendly: no images-as-text, no fancy columns that break parsing. Use standard section headings.

Tools: **Reactive Resume** (open source), **FlowCV**, or **Overleaf** for LaTeX purists.

---

## The Content Compounder Pattern

Turn each project into multiple assets:

**One flagship repo** →
- **Public repo** (surface 1)
- **Blog post write-up** (surface 3A)
- **LinkedIn announcement** (surface 3B) linking to the blog
- **Résumé bullet** (surface 4)
- **Portfolio card** (surface 2)

**Five flagship repos** = 5 blog posts + 5 LinkedIn threads + 5 résumé bullets + 5 portfolio cards + 5 interview stories.

That's ~30 credibility artifacts from the six-month curriculum, if you invest one weekend of write-up per flagship. Extraordinary leverage.

---

## What Makes AI Engineers Signal Well vs Badly

**Good signal:**
- Deployed URLs (even to free tiers) — "here it is running"
- Screenshots of dashboards, evals, GIFs of interactions
- Specific numbers ("87% eval faithfulness", "$0.017 per request")
- Honest write-ups of what didn't work
- Code you'd be happy to defend line by line
- Consistency over 6+ months of dated commits or posts

**Bad signal:**
- "Trained a model" with no eval numbers
- LangChain quickstarts with your name pasted on
- "Passionate about AI" as a bio adjective
- Silent GitHub for 2 years then a burst of activity right before job hunting
- Certifications without work behind them
- Buzzword salad in bios ("AGI-native full-stack generative AI polymath")
- Cross-posted engagement-farming content

For your background, three things are already differentiators worth foregrounding:
1. **Decade of production software experience** — you're not a bootcamp graduate. Play this up.
2. **Java + AI** — most AI candidates are Python-only. Java for enterprise + AI for the future is a rare combo employers want.
3. **Montana Legislative work** — memorable and civic. Beats "worked at another SaaS."

---

## A 6-Week Portfolio Assembly Plan

Assuming the curriculum work is done, this is how to launch the surface area:

**Week 1** — GitHub cleanup. Create a `.github` org profile repo (pinned bio + top projects). Polish READMEs of 3-5 flagships. Add live demo URLs. Pin them on your profile.

**Week 2** — Portfolio site. Astro or Next.js starter; 4 pages (home, projects, writing, about). Deploy to Cloudflare Pages. Custom domain.

**Week 3** — Résumé. Draft, get one senior peer to review, iterate. Save PDF. Update LinkedIn to match.

**Week 4** — Blog post #1. Pick the most impressive project (usually the RAG or the production-hardening). Write 2000-word post with numbers, screenshots, and code. Publish.

**Week 5** — Blog post #2 (fine-tuning or agents). Post LinkedIn thread linking to post #1.

**Week 6** — Blog post #3. LinkedIn thread linking to #2. By now you have 3 posts and 2 LinkedIn threads live — enough to start applying with credibility.

Weeks 7+: maintain the cadence, add posts as project extensions surface. Apply and interview in parallel.

---

## The One Number That Matters

For each project you present, have **one memorable number** you'd tell a recruiter.

- RAG: "Faithfulness 0.87 on RAGAS after adding cross-encoder rerank"
- Production hardening: "Cut LLM cost by 42% while raising eval score"
- Fine-tuning: "27-point accuracy uplift over base with 2500 examples in 40 minutes on a $2 GPU"
- Self-hosting: "1200 tokens/sec on an A10G at 20× lower cost than Sonnet"
- Agent: "Successfully completes 83% of research tasks under $0.50 budget"

Numbers make people remember. Round to 2-3 significant figures. Never fudge.

---

## Closing

Your résumé and GitHub aren't your work. They're the *distribution layer* for your work. Six months of curriculum, ten capstones, plus a fine-tune and a self-host is genuinely a senior engineer's portfolio-worth of AI signal.

**Don't spend six months building and then two weeks polishing.** Reserve a full month of the last cycle for portfolio + writing + application. That's where the job comes from.

---

**Related files:**
- `career/02-interview-loop-prep.md` — after the pitch works
- `career/03-comp-and-negotiation.md` — after the offer comes
- All flagship capstones — the substance
- `capstones/10-interview-explain-architecture.md` — the whiteboard performance
