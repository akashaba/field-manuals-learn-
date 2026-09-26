# Self-Hosting Open Models — Goals

> **North star:** Be the person on the team who can spin up a 7B or 70B open-weight model behind an OpenAI-compatible API, tune throughput and latency knowingly, and defend the cost model to the CFO. Know when self-hosting wins economically, when it doesn't, and how to make the switch reversible.

Managed LLM APIs (Claude, OpenAI, Gemini) are the right choice for most workloads most of the time. But at sufficient scale, on some tasks, self-hosting an open model wins by 5-20×. Being the engineer who can pull that lever confidently is a real edge.

---

## What "self-hosting" means in 2026

Self-hosting = you run the model on hardware you control (or lease directly), managing inference yourself. The 2026 stack is remarkably standard:

- **Model:** Llama 3.1/3.2, Qwen 2.5, Mistral, Gemma 2, or a fine-tune thereof — open weights, permissive license
- **Server:** vLLM (dominant), TGI (Hugging Face's), SGLang (research/experimental), or llama.cpp (edge/CPU/consumer)
- **API shape:** OpenAI-compatible chat completions endpoint — everything speaks this now
- **Hardware:** rented GPU (RunPod, Modal, AWS/GCP GPU instances) or owned (single 24GB card for 7-8B; multi-GPU for 70B+)
- **Quantization:** 8-bit (bitsandbytes, AWQ, GPTQ) or 4-bit for tight VRAM; bf16 for max quality

The interesting engineering: **throughput optimization** (dynamic batching, prefix caching, speculative decoding), **quantization trade-offs**, and **the OpenAI-compatibility shim** that makes swapping providers a config change.

---

## Learning outcomes

By the end of this track you should be able to:

**Serving**
- Launch vLLM with the right flags for a given workload
- Configure `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`
- Enable and monitor prefix caching for shared-prefix workloads
- Load and swap LoRA adapters at request time via `--enable-lora`

**Quantization**
- Compare fp16/bf16/int8/int4 (GPTQ, AWQ, INT4 NF) on quality-vs-speed-vs-size
- Choose a quantization for a target GPU
- Detect quality degradation caused by too-aggressive quantization

**API compatibility**
- Build a thin FastAPI wrapper that speaks OpenAI's chat completions shape
- Support streaming, tool calls, and structured output on top of an open model
- Route between a managed provider and self-hosted with a single config flag

**Economics**
- Compute the break-even token volume where self-hosting beats a managed API
- Model utilization patterns for sensible sizing
- Argue for or against self-hosting given a specific workload

**Operations**
- Deploy vLLM on a cloud GPU with health checks and metrics
- Monitor tokens/sec, latency, cache hit rate, and utilization
- Handle failover between self-hosted and managed provider

---

## The 3-module track + 1 capstone

| # | Module | Question it answers |
|---|--------|--------------------|
| 01 | vLLM serving | How do I get a model behind an API and how fast can it go? |
| 02 | Quantization | Why does the same 8B model take 16 GB or 4 GB depending on config? |
| 03 | OpenAI-compatible API | How do I make a self-hosted model a drop-in for the Anthropic/OpenAI SDK? |
| Capstone | Self-host Llama 3.1 8B behind an OpenAI-compatible API | End-to-end: launch → tune → deploy → route between it and Anthropic |

**Time budget:** 2-3 weeks of evenings; capstone is a long weekend.

---

## Prerequisites

- Comfortable with Docker (Month 6 Module 02) and cloud deployment (Month 6 Module 03)
- FastAPI fluency (Month 6 Module 01)
- LLM API basics (Month 4)
- Familiarity with the fine-tuning track (this repo's `fine-tuning/`) is helpful but not required — the two tracks are complementary

Plus one physical prerequisite: **a way to rent a GPU by the hour**. Options:
- **RunPod** — cheap ($0.30-1.00/hr for consumer-tier), fast to set up
- **Modal** — Python-native launches, great for scripted workflows
- **Vast.ai** — cheapest, more manual, quality varies
- **Cloud vendors** (AWS/GCP/Azure) — most expensive, best for enterprise
- **Fly.io GPU** — new but simple

The cheapest way to complete the whole track: ~$20-30 of GPU credit for the modules + capstone.

---

## What this track is not

- Not a "buy your own DGX" guide. We assume you rent by the hour.
- Not deep kernel-level GPU optimization. If you want CUDA/Triton internals, that's a separate research track. We treat the serving engine as a fast box; we tune its dials.
- Not a fine-tuning course. See `fine-tuning/` for that. The two tracks combine when you self-host a fine-tuned model.

Let's begin.
