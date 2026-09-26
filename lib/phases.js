// Canonical phase list. Slug matches the repo directory under `phases/`.
// Times and lesson counts are picked up from the fetched manifest when it exists.
export const PHASES = [
  { slug: "00-setup-and-tooling",           num: "00", name: "Setup & Tooling",                blurb: "Everything you install before touching a model — env, GPUs, Docker, keys, notebooks." },
  { slug: "01-math-foundations",            num: "01", name: "Math Foundations",               blurb: "Linear algebra, calculus, probability, optimization — the language ML is written in." },
  { slug: "02-ml-fundamentals",             num: "02", name: "ML Fundamentals",                blurb: "Regression, trees, SVMs, boosting — the tools that were already winning before deep learning." },
  { slug: "03-deep-learning-core",          num: "03", name: "Deep Learning Core",             blurb: "Backprop, initialization, regularization, normalization — the mechanics of neural nets." },
  { slug: "04-computer-vision",             num: "04", name: "Computer Vision",                blurb: "From convolutions to ViT and diffusion — how machines learn to see." },
  { slug: "05-nlp-foundations-to-advanced", num: "05", name: "NLP Foundations to Advanced",    blurb: "Tokens, embeddings, sequence models, attention — the road that led to transformers." },
  { slug: "06-speech-and-audio",            num: "06", name: "Speech & Audio",                 blurb: "STFT, mel-spectrograms, ASR, TTS — models that listen and speak." },
  { slug: "07-transformers-deep-dive",      num: "07", name: "Transformers Deep Dive",         blurb: "Self-attention, KV cache, positional encodings — the architecture behind every LLM." },
  { slug: "08-generative-ai",               num: "08", name: "Generative AI",                  blurb: "GANs, VAEs, diffusion, flow-matching — the family of models that make new things." },
  { slug: "09-reinforcement-learning",      num: "09", name: "Reinforcement Learning",         blurb: "Bandits, Q-learning, policy gradients, PPO — learning by trial and error." },
  { slug: "10-llms-from-scratch",           num: "10", name: "LLMs from Scratch",              blurb: "Tokenizer up through pre-training, SFT, RLHF, DPO, quantization and inference.", deep: true },
  { slug: "11-llm-engineering",             num: "11", name: "LLM Engineering",                blurb: "Prompting, evals, retrieval, structured output — using LLMs as production components." },
  { slug: "12-multimodal-ai",               num: "12", name: "Multimodal AI",                  blurb: "Vision-language, audio-language, any-to-any — models that bridge modalities." },
  { slug: "13-tools-and-protocols",         num: "13", name: "Tools & Protocols",              blurb: "Tool use, function calling, MCP — how models reach out to the world." },
  { slug: "14-agent-engineering",           num: "14", name: "Agent Engineering",              blurb: "Planning, memory, loops, guardrails — building agents that actually finish tasks." },
  { slug: "15-autonomous-systems",          num: "15", name: "Autonomous Systems",             blurb: "Long-horizon agents, self-correction, world models — approaching autonomy carefully." },
  { slug: "16-multi-agent-and-swarms",      num: "16", name: "Multi-Agent & Swarms",           blurb: "Roles, coordination, message buses, marketplaces — many agents working together." },
  { slug: "17-infrastructure-and-production", num: "17", name: "Infrastructure & Production", blurb: "Serving, batching, sharding, cost — running these systems for real users." },
  { slug: "18-ethics-safety-alignment",     num: "18", name: "Ethics, Safety, Alignment",      blurb: "Red-teaming, evals for harm, constitutional methods — building models that behave." },
  { slug: "19-capstone-projects",           num: "19", name: "Capstone Projects",              blurb: "End-to-end builds that tie everything together — ship something you would use." },
];

export const REPO = {
  owner: "rohitg00",
  name: "ai-engineering-from-scratch",
  branch: "main",
};
