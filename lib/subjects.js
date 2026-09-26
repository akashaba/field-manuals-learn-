// Subject registry. Each subject knows: where its content lives on GitHub,
// how its files map onto (group, item) URLs, and how the landing page presents it.

export function slugify(s) {
  return s.toLowerCase().trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

const AI_PHASES = [
  { slug: "00-setup-and-tooling",           num: "00", name: "Setup & Tooling",                blurb: "Everything you install before touching a model." },
  { slug: "01-math-foundations",            num: "01", name: "Math Foundations",               blurb: "Linear algebra, calculus, probability, optimization." },
  { slug: "02-ml-fundamentals",             num: "02", name: "ML Fundamentals",                blurb: "Regression, trees, SVMs, boosting." },
  { slug: "03-deep-learning-core",          num: "03", name: "Deep Learning Core",             blurb: "Backprop, initialization, regularization, normalization." },
  { slug: "04-computer-vision",             num: "04", name: "Computer Vision",                blurb: "From convolutions to ViT and diffusion." },
  { slug: "05-nlp-foundations-to-advanced", num: "05", name: "NLP Foundations to Advanced",    blurb: "Tokens, embeddings, sequence models, attention." },
  { slug: "06-speech-and-audio",            num: "06", name: "Speech & Audio",                 blurb: "STFT, mel-spectrograms, ASR, TTS." },
  { slug: "07-transformers-deep-dive",      num: "07", name: "Transformers Deep Dive",         blurb: "Self-attention, KV cache, positional encodings." },
  { slug: "08-generative-ai",               num: "08", name: "Generative AI",                  blurb: "GANs, VAEs, diffusion, flow-matching." },
  { slug: "09-reinforcement-learning",      num: "09", name: "Reinforcement Learning",         blurb: "Bandits, Q-learning, policy gradients, PPO." },
  { slug: "10-llms-from-scratch",           num: "10", name: "LLMs from Scratch",              blurb: "Tokenizer through pre-training, SFT, RLHF, DPO, quantization.", deep: true },
  { slug: "11-llm-engineering",             num: "11", name: "LLM Engineering",                blurb: "Prompting, evals, retrieval, structured output." },
  { slug: "12-multimodal-ai",               num: "12", name: "Multimodal AI",                  blurb: "Vision-language, audio-language, any-to-any." },
  { slug: "13-tools-and-protocols",         num: "13", name: "Tools & Protocols",              blurb: "Tool use, function calling, MCP." },
  { slug: "14-agent-engineering",           num: "14", name: "Agent Engineering",              blurb: "Planning, memory, loops, guardrails." },
  { slug: "15-autonomous-systems",          num: "15", name: "Autonomous Systems",             blurb: "Long-horizon agents, self-correction, world models." },
  { slug: "16-multi-agent-and-swarms",      num: "16", name: "Multi-Agent & Swarms",           blurb: "Roles, coordination, message buses, marketplaces." },
  { slug: "17-infrastructure-and-production", num: "17", name: "Infrastructure & Production", blurb: "Serving, batching, sharding, cost." },
  { slug: "18-ethics-safety-alignment",     num: "18", name: "Ethics, Safety, Alignment",      blurb: "Red-teaming, evals for harm, constitutional methods." },
  { slug: "19-capstone-projects",           num: "19", name: "Capstone Projects",              blurb: "End-to-end builds that tie everything together." },
];

export const SUBJECTS = [
  {
    slug: "ai-engineering",
    name: "AI Engineering",
    tagline: "from first principles",
    tocLabel: "20 phases · 511 lessons",
    description:
      "A field-manual walkthrough of the modern AI stack — linear algebra and gradient descent, through transformers and LLM training, all the way to agents, multimodal systems, and production infrastructure.",
    accent: "#2a4c72",
    accent2: "#1a3556",
    repo: { owner: "rohitg00", name: "ai-engineering-from-scratch", branch: "main" },
    layout: "phases",
    groups: AI_PHASES,
    matcher(pathStr) {
      const m = pathStr.match(/^phases\/([^/]+)\/([^/]+)\/docs\/en\.md$/);
      return m ? { group: m[1], item: m[2], sourcePath: pathStr, imageBase: `phases/${m[1]}/${m[2]}/docs/` } : null;
    },
  },
  {
    slug: "system-design",
    name: "System Design",
    tagline: "notes from the interview",
    tocLabel: "28 chapters",
    description:
      "Twenty-eight canonical system-design chapters — scaling and estimation, rate limiters and consistent hashing, then the classics: URL shortener, news feed, chat, YouTube, Google Drive, payment, stock exchange.",
    accent: "#8b3a2a",
    accent2: "#6b2b1f",
    repo: { owner: "liquidslr", name: "system-design-notes", branch: "main" },
    layout: "flat",
    groups: [{ slug: "chapters", num: "01", name: "Chapters", blurb: "" }],
    matcher(pathStr) {
      const m = pathStr.match(/^([^/]+)\/(README\.md|Readme\.md|readme\.md)$/);
      if (!m) return null;
      const dir = m[1];
      const item = slugify(dir);
      if (!item) return null;
      return { group: "chapters", item, sourcePath: pathStr, originalDir: dir, imageBase: `${dir}/` };
    },
  },

  // ---- Microsoft ML For Beginners --------------------------------------
  {
    slug: "ml-for-beginners",
    name: "ML for Beginners",
    tagline: "Microsoft's 24-lesson curriculum",
    tocLabel: "9 chapters · 24 lessons",
    description:
      "Microsoft's ML curriculum — Introduction, Regression, Web App, Classification, Clustering, NLP, Time Series, Reinforcement, Real-World. One README per lesson, plus notebooks.",
    accent: "#0078d4",
    accent2: "#003b6a",
    repo: { owner: "microsoft", name: "ML-For-Beginners", branch: "main" },
    layout: "phases",
    groups: [
      { slug: "1-Introduction",   num: "01", name: "Introduction",   blurb: "What ML is and isn't." },
      { slug: "2-Regression",     num: "02", name: "Regression",     blurb: "Linear, polynomial, logistic." },
      { slug: "3-Web-App",        num: "03", name: "Web App",        blurb: "Ship a model behind Flask." },
      { slug: "4-Classification", num: "04", name: "Classification", blurb: "Classifiers on real data." },
      { slug: "5-Clustering",     num: "05", name: "Clustering",     blurb: "K-means and friends." },
      { slug: "6-NLP",            num: "06", name: "NLP",            blurb: "Text prep, sentiment, translation." },
      { slug: "7-TimeSeries",     num: "07", name: "Time Series",    blurb: "ARIMA, SVR forecasts." },
      { slug: "8-Reinforcement",  num: "08", name: "Reinforcement",  blurb: "Q-learning and Peter & the Wolf." },
      { slug: "9-Real-World",     num: "09", name: "Real-World",     blurb: "Ethics, postscript." },
    ],
    matcher(pathStr) {
      const m = pathStr.match(/^([0-9]-[A-Za-z]+)\/([0-9]-[^/]+)\/README\.md$/);
      if (!m) return null;
      return { group: m[1], item: slugify(m[2]), sourcePath: pathStr, originalDir: `${m[1]}/${m[2]}`, imageBase: `${m[1]}/${m[2]}/` };
    },
  },

  // ---- Microsoft AI Agents for Beginners --------------------------------
  {
    slug: "ai-agents-for-beginners",
    name: "AI Agents for Beginners",
    tagline: "Microsoft's agent curriculum",
    tocLabel: "17 lessons",
    description:
      "Setup through production — intro, frameworks, design patterns, tool use, agentic RAG, trustworthiness, planning, multi-agent, metacognition, memory, browser use, deployment.",
    accent: "#6b2f8f",
    accent2: "#472060",
    repo: { owner: "microsoft", name: "ai-agents-for-beginners", branch: "main" },
    layout: "flat",
    groups: [{ slug: "lessons", num: "01", name: "Lessons", blurb: "" }],
    matcher(pathStr) {
      const m = pathStr.match(/^([0-9]{2}-[^/]+)\/README\.md$/);
      if (!m) return null;
      return { group: "lessons", item: slugify(m[1]), sourcePath: pathStr, originalDir: m[1], imageBase: `${m[1]}/` };
    },
  },

  // ---- RAG from Scratch (paired CONCEPT.md + CODE.md) -------------------
  {
    slug: "rag-from-scratch",
    name: "RAG from Scratch",
    tagline: "no black boxes",
    tocLabel: "7 phases",
    description:
      "Peter Guso's build-it-yourself RAG walkthrough — every lesson is a CONCEPT explainer paired with a CODE walkthrough. Local models, in-memory vector store, retrieval strategies.",
    accent: "#c05621",
    accent2: "#8a3d17",
    repo: { owner: "pguso", name: "rag-from-scratch", branch: "main" },
    layout: "phases",
    groups: [
      { slug: "00_how_rag_works",              num: "00", name: "How RAG Works",        blurb: "The 30,000-ft view." },
      { slug: "01_intro_to_llms",              num: "01", name: "Intro to LLMs",        blurb: "Local models via node-llama-cpp." },
      { slug: "02_data_loading",               num: "02", name: "Data Loading",         blurb: "Get documents into memory." },
      { slug: "03_text_splitting_and_chunking",num: "03", name: "Text Splitting",       blurb: "Chunking strategies that keep meaning intact." },
      { slug: "04_intro_to_embeddings",        num: "04", name: "Embeddings",           blurb: "Turn text into vectors." },
      { slug: "05_building_vector_store",      num: "05", name: "Vector Store",         blurb: "In-memory store, NN search, metadata filtering." },
      { slug: "06_retrieval_strategies",       num: "06", name: "Retrieval Strategies", blurb: "Basic, hybrid, multi-query, rewriting." },
    ],
    matcher(pathStr) {
      const m = pathStr.match(/^examples\/([^/]+)(?:\/([^/]+))?\/(CONCEPT|CODE)\.md$/);
      if (!m) return null;
      const phase = m[1];
      const lessonDir = m[2] || null;
      const fileType = m[3];
      const item = lessonDir ? slugify(lessonDir) : "overview";
      const rel = lessonDir ? `${phase}/${lessonDir}` : phase;
      return {
        group: phase,
        item,
        sourcePath: pathStr,
        imageBase: `examples/${rel}/`,
        role: fileType === "CODE" ? "code" : "main",
      };
    },
  },

  // ---- AI Engineer Roadmap (user's own repo) ---------------------------
  {
    slug: "ai-engineer-roadmap",
    name: "AI Engineer Roadmap",
    tagline: "learning path + builds",
    tocLabel: "6 phases + tracks",
    description:
      "A personal AI engineering roadmap — six numbered phases from Foundations through Production, plus side tracks for Career, Fine-tuning, and Self-hosting. Each phase pairs Learning notes with Build projects.",
    accent:  "#0d7c66",
    accent2: "#08544a",
    repo:    { owner: "akashaba", name: "AI-Engineer-Roadmap", branch: "main" },
    layout:  "phases",
    groups: [
      { slug: "1-foundations",             num: "01", name: "Foundations",             blurb: "Python, math, git, APIs." },
      { slug: "2-machine-learning",        num: "02", name: "Machine Learning",        blurb: "Classical ML end-to-end." },
      { slug: "3-deep-learning",           num: "03", name: "Deep Learning",           blurb: "Neural nets, CNNs, RNNs." },
      { slug: "4-llm-engineering",         num: "04", name: "LLM Engineering",         blurb: "Prompting, RAG, evals." },
      { slug: "5-agents-production",       num: "05", name: "Agents + Production",     blurb: "Agent design, tool use." },
      { slug: "6-production-ai-engineer",  num: "06", name: "Production AI Engineer",  blurb: "Ship real systems." },
      { slug: "7-projects",                num: "07", name: "Projects",                blurb: "Portfolio builds." },
      { slug: "career",                    num: "08", name: "Career",                  blurb: "Interviews, growth, comp." },
      { slug: "fine-tuning",               num: "09", name: "Fine-tuning",             blurb: "SFT, LoRA, DPO." },
      { slug: "self-hosting",              num: "10", name: "Self-hosting",            blurb: "Local models, infra." },
    ],
    matcher(pathStr) {
      // <phase>/<file>.md or <phase>/<subdir>/<file>.md
      const m = pathStr.match(/^([^/]+)\/(?:([^/]+)\/)?([^/]+)\.md$/);
      if (!m) return null;
      const phase = m[1], subdir = m[2], file = m[3];
      if (file.toLowerCase() === "readme") return null; // skip phase READMEs
      const item = subdir ? `${slugify(subdir)}-${slugify(file)}` : slugify(file);
      const imageBase = subdir ? `${phase}/${subdir}/` : `${phase}/`;
      return { group: slugify(phase), item, sourcePath: pathStr, imageBase };
    },
  },

  // ---- Ultimate AI Engineering (user's own repo) -----------------------
  {
    slug: "ultimate-ai-engineering",
    name: "Ultimate AI Engineering",
    tagline: "the compact reference",
    tocLabel: "10 chapters",
    description:
      "A tight ten-chapter reference on the modern LLM stack — fundamentals, prompting, representations, RAG, tools and agents, evaluation, safety, LLMOps, adaptation, and advanced topics.",
    accent:  "#6d28d9",
    accent2: "#4c1d95",
    repo:    { owner: "akashaba", name: "Ultimate-AI-Engineering", branch: "main" },
    layout:  "phases",
    groups: [
      { slug: "01-llm-fundamentals",    num: "01", name: "LLM Fundamentals",   blurb: "Attention, tokens, decoding." },
      { slug: "02-prompting-context",   num: "02", name: "Prompting & Context",blurb: "In-context learning, structured output." },
      { slug: "03-representations",     num: "03", name: "Representations",    blurb: "Embeddings and vector search." },
      { slug: "04-rag",                 num: "04", name: "RAG",                blurb: "Retrieval, chunking, re-ranking." },
      { slug: "05-tools-agents",        num: "05", name: "Tools & Agents",     blurb: "Function calling, planning." },
      { slug: "06-evaluation",          num: "06", name: "Evaluation",         blurb: "Offline + online evals." },
      { slug: "07-safety",              num: "07", name: "Safety",             blurb: "Guardrails, red-teaming." },
      { slug: "08-llmops-serving",      num: "08", name: "LLMOps & Serving",   blurb: "Inference, batching, cost." },
      { slug: "09-adaptation",          num: "09", name: "Adaptation",         blurb: "Fine-tuning, alignment." },
      { slug: "10-advanced",            num: "10", name: "Advanced",           blurb: "Beyond the basics." },
    ],
    matcher(pathStr) {
      const m = pathStr.match(/^([^/]+)\/([^/]+)\.md$/);
      if (!m) return null;
      const file = m[2];
      if (file.toLowerCase() === "readme") return null; // skip phase READMEs
      return { group: slugify(m[1]), item: slugify(file), sourcePath: pathStr, imageBase: `${m[1]}/` };
    },
  },

  // ---- Tech Articles (index of X/Twitter threads) ----------------------
  {
    slug: "tech-articles",
    name: "Tech Articles",
    tagline: "@Harry_The_Nerd on X",
    tocLabel: "index of X threads",
    description:
      "A curated index of technical threads on X — high-level design, low-level design, distributed systems, backend engineering, microservices. Each page embeds the original thread live from X and links back to the author.",
    accent:  "#1d9bf0",
    accent2: "#0f6cb2",
    source:  "articles-index",
    medium:  "articles",
    repo:    { owner: "harshit3011", name: "Technical-Engineering-Articles", branch: "main" },
    layout:  "phases",
    groups: [
      { slug: "hld",                    num: "01", name: "High-Level Design",  blurb: "Design large systems from scratch." },
      { slug: "lld",                    num: "02", name: "Low-Level Design",   blurb: "Class-and-object design problems." },
      { slug: "backend-engineering",    num: "03", name: "Backend Engineering",blurb: "Practical backend patterns." },
      { slug: "engineering-articles",   num: "04", name: "Engineering",        blurb: "General engineering essays." },
      { slug: "distributed-systems",    num: "05", name: "Distributed Systems",blurb: "Consensus, replication, sharding." },
      { slug: "microservices",          num: "06", name: "Microservices",      blurb: "Service boundaries and comms." },
      { slug: "machine-learning",       num: "07", name: "Machine Learning",   blurb: "ML applied." },
      { slug: "ai-engineering",         num: "08", name: "AI Engineering",     blurb: "Foundation-model systems." },
    ],
    matcher(pathStr) {
      const m = pathStr.match(/^([^/]+)\/(.+)\.md$/);
      if (!m) return null;
      return { group: m[1], item: m[2], sourcePath: pathStr, imageBase: `${m[1]}/` };
    },
  },

  // ---- Destination FAANG · Java LeetCode solutions ---------------------
  {
    slug: "destination-faang",
    name: "Destination FAANG",
    tagline: "LeetCode, in Java",
    tocLabel: "problems",
    description:
      "Curated FAANG LeetCode problems with Java solutions. Each page pulls the problem statement from LeetCode and embeds the author's video walkthrough alongside the code.",
    accent: "#1c8d5a",
    accent2: "#125d3a",
    repo: { owner: "DestinationFAANG", name: "Destination-FAANG-Java-Solution", branch: "main" },
    layout: "flat",
    groups: [{ slug: "problems", num: "01", name: "Problems", blurb: "" }],
    matcher(pathStr) {
      // top-level dir like "10 Regular Expression Matching" with a .java inside
      const m = pathStr.match(/^([^/]+)\/[^/]+\.java$/);
      if (!m) return null;
      const dir = m[1];
      if (dir.startsWith(".")) return null;
      return {
        group: "problems",
        item: slugify(dir),
        sourcePath: pathStr,
        originalDir: dir,
        imageBase: `${dir}/`,
        sourceType: "java",
      };
    },
  },
];

export function subjectBySlug(slug) {
  return SUBJECTS.find((s) => s.slug === slug);
}
