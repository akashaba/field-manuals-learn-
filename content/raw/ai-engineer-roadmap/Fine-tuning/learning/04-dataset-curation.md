# 04 — Dataset Curation

> **Module goal:** Understand that the dataset — not the model, hyperparameters, or technique — is the biggest lever in fine-tuning success. Master the practices of building, cleaning, deduplicating, and formatting a fine-tuning dataset. Know when to use synthetic data and when it will burn you.

---

## 1. Executive Summary & Core Concepts

**The fine-tuning outcome is 80% dataset, 20% everything else.** Get the dataset wrong and no amount of hyperparameter tuning will save you. Get it right and even hasty training produces useful models. Every experienced practitioner will tell you this; nobody who hasn't done it will believe it until they've wasted a weekend on a broken run.

**What "wrong" looks like in a dataset:**
- Duplicate or near-duplicate examples (model memorizes them, overfits fast)
- Prompt-response format inconsistency (some rows use JSON, some plain text)
- Length distribution skewed (all responses 20 tokens; model becomes terse)
- Chat template mismatch with the base model
- Toxic or contradictory examples the model over-generalizes from
- Label leakage — the answer appears in the prompt in some rows

**What "right" looks like:**
- A single, consistent format across all rows
- Length distribution matching what you want the model to produce
- Quality bar: every example either good-as-shipped or better than what the base model produces
- Deduplication done properly (semantic, not just string)
- 10-20% "diversity padding" from general instruction data
- Cleanly separated train/val/test with no cross-contamination

**Dataset size rules of thumb:**

| Purpose | Minimum | Sweet spot | Overkill |
|---------|---------|-----------|----------|
| Format / style SFT | 500 | 2000-5000 | 50000+ |
| Task capability SFT | 2000 | 10000-30000 | 100000+ |
| DPO preference tuning | 500 | 1000-3000 | 20000+ |
| ORPO/KTO | 500 | 2000-5000 | 20000+ |

Note the tight middle range. Dataset quality drops fast when you cut corners; scale rarely rescues bad data.

---

## 2. Deep-Dive Breakdown

### 2.1 The Data Format

Modern fine-tuning frameworks (TRL, Axolotl, torchtune, Unsloth) accept several formats:

**Chat format (recommended for chat/Instruct models):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a legal research assistant."},
    {"role": "user", "content": "Summarize Marbury v. Madison in one paragraph."},
    {"role": "assistant", "content": "Marbury v. Madison (1803) established..."}
  ]
}
```

**Instruction format (works for base models too):**
```json
{
  "instruction": "Summarize Marbury v. Madison in one paragraph.",
  "input": "",
  "output": "Marbury v. Madison (1803) established..."
}
```

**Preference (for DPO):**
```json
{
  "prompt": "Summarize Marbury v. Madison in one paragraph.",
  "chosen": "Marbury v. Madison (1803) established the principle of judicial review, allowing the Supreme Court to strike down laws it finds unconstitutional. Chief Justice John Marshall's decision fundamentally reshaped the balance of power between the branches of government.",
  "rejected": "It's a Supreme Court case from a long time ago about the government. Chief Justice Marshall wrote it."
}
```

**Preference (for KTO):**
```json
{"prompt": "...", "completion": "...", "label": true}
```

Pick one format for your entire dataset and stick with it. Framework loaders can convert between them, but mixing formats within a single dataset is a bug waiting to happen.

**Storage:** JSONL (one JSON object per line). Loads with `datasets.load_dataset("json", data_files=...)`.

### 2.2 The Quality Bar

Every fine-tuning example must satisfy one of two conditions:

1. **The response is production-ship-worthy** — you'd be happy for your model to output this verbatim to a user
2. **The response is measurably better than what the base model produces** — even if not perfect, it moves the needle

If neither is true, the example is noise. Cut it.

Practical filters:

- **Length filter**: throw out responses shorter than a reasonable minimum (say 20 tokens) unless the task really is short. Also throw out extreme outliers on the long end.
- **Language filter**: FastText or `langdetect` — remove non-target-language rows unless multilingual is the goal.
- **Toxicity filter**: Perspective API or a small classifier — remove obviously toxic content unless your task requires it.
- **Truncation filter**: if the response was cut off mid-sentence in your source data, remove it — the model will learn to produce partial responses.
- **Refusal filter**: if you don't want the model to refuse, remove rows where the "response" is a refusal.
- **Deduplication** (see 2.4 below).

Zhou et al. (2023) "LIMA: Less Is More for Alignment" showed a 1000-example carefully curated SFT set can rival full RLHF for many properties. **Small and clean beats large and noisy.**

### 2.3 Sources of Data

**In order of quality (usually):**

**1. Human-written, task-specific.** Your own team writing examples for your task. Best quality; scales badly. $1-10 per example depending on complexity.

**2. Extracted from historical production data.** If you have logs of expert humans doing this task (customer service transcripts, legal briefs, code review comments), these are gold. Cleaning is the hard part.

**3. Distilled from a stronger model.** Ask Claude Opus or GPT-5 to produce responses to your prompts; use those as training data for your smaller model. This is "teacher-student distillation" or "model distillation" — often the best cost/quality tradeoff. Legal note: check that the source model's terms of service allow using outputs to train other models (this varies by provider and license).

**4. Synthetic from your own model.** Use your base or SFT model to generate responses; filter them; use the filtered set as training data. Cheap but risks compounding the base model's biases and can lead to model collapse over multiple rounds.

**5. Public instruction datasets.** Alpaca, ShareGPT, Dolly, OpenAssistant, WildChat, No Robots, Tulu. Great for general-purpose alignment and as "diversity padding" for a task-specific dataset. Not sufficient alone for a specific task.

Most production fine-tunes combine #2 or #3 (task-specific) with #5 (10-20% diversity padding to prevent forgetting).

### 2.4 Deduplication — Properly

Duplicates and near-duplicates ruin training. A row appearing 5× is trained on 5× — you're overfitting to that specific example. Worse: many "clean" datasets contain **near-duplicates** (paraphrases, minor edits) that string-hash dedup misses.

Layers of deduplication:

**1. Exact string dedup.** Hash `prompt + response`; drop duplicates.

```python
import hashlib
seen = set()
deduped = []
for row in data:
    h = hashlib.sha256(f"{row['prompt']}|{row['response']}".encode()).hexdigest()
    if h not in seen:
        seen.add(h)
        deduped.append(row)
```

**2. Fuzzy dedup (MinHash / SimHash).** For catching lightly-modified duplicates. Standard in large-scale dataset construction.

```python
from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.8, num_perm=128)
minhashes = {}
for i, row in enumerate(data):
    m = MinHash(num_perm=128)
    for word in row["prompt"].split():
        m.update(word.encode())
    minhashes[i] = m
    lsh.insert(i, m)

# Find and remove near-duplicates
to_remove = set()
for i, m in minhashes.items():
    for j in lsh.query(m):
        if j != i and j > i:
            to_remove.add(j)
```

**3. Semantic dedup (embedding-based).** Embed prompts (or prompt+response), compute pairwise cosine similarity, drop rows with cosine > threshold (say 0.95). Slowest but catches paraphrases.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([r["prompt"] for r in data])
# ... nearest-neighbor search, drop near-duplicates
```

For most datasets: run exact dedup + MinHash. Run semantic dedup only if the dataset is small enough for O(n²) search or you have specific concerns.

### 2.5 Length Distribution Analysis

Every SFT dataset should be plotted on two axes: prompt-token length and response-token length. Look for:

- **Extreme outliers**: rows > 4× the median length. Truncated or malformed.
- **Bimodal distributions**: two clumps of very-short and very-long. Usually means two different tasks accidentally mixed.
- **Right-skew of responses**: responses trending long. Model will learn to be verbose. Truncate or ensure the prompt shape matches production.

Simple check:

```python
import numpy as np

prompt_lens = [len(tokenizer.encode(r["prompt"])) for r in data]
resp_lens = [len(tokenizer.encode(r["response"])) for r in data]

print(f"Prompt len — median: {np.median(prompt_lens)}, p95: {np.percentile(prompt_lens, 95)}")
print(f"Response len — median: {np.median(resp_lens)}, p95: {np.percentile(resp_lens, 95)}")
```

If p95 is 5× the median, you have outliers to investigate.

### 2.6 Synthetic Data — the Careful Version

Distilling from a stronger model works well when done carefully:

**Recipe:**
1. **Seed with real prompts** — from your production logs, or a hand-written variety pack, or a public dataset. Do NOT ask the strong model to generate prompts *and* answers — you'll get prompt distribution collapse.
2. **Generate responses at low temperature** (0.3–0.5) with the strong model.
3. **Sample multiple responses per prompt** (say 3). Ask a judge (same or stronger model) to pick the best.
4. **Filter aggressively.** Length, language, quality-classifier score, human spot-check.
5. **Cap the ratio** — don't use synthetic exclusively if you have any real data at all. Mix real and synthetic 30:70 or 50:50.

**When synthetic data burns you:**
- Repetitive patterns: the teacher has a favorite phrase; your model now uses it in every response.
- Compounding errors: teacher-model factual errors are memorized as ground truth.
- Distribution shift: teacher's response style doesn't match your production users' expectations.
- Legal / TOS issues: some providers restrict training on their outputs.

`[IMG-FT04-01]` — *Prompt: A dataset curation pipeline diagram. Left: raw sources (production logs, human-written batches, distilled-from-teacher, public datasets). Middle: a series of stacked filter boxes labeled: "Format normalization → Exact dedup → MinHash near-dedup → Length filter → Language filter → Toxicity filter → Length distribution analysis → Quality-classifier score → Train/val/test split". Right: a clean output dataset icon labeled "N=2400 examples, all format-consistent, deduped". Below the pipeline, tiny sparkline charts showing the row count dropping at each filter: 15000 → 12000 → 8500 → 7800 → 5100 → 4900 → 2400. Clean pipeline diagram with data-quality dashboard aesthetic.*

---

## 3. Mental Models & Analogies

### Model 1: The Chef's Ingredient Selection

A restaurant kitchen's reputation is decided at the loading dock. A mediocre chef with pristine ingredients can produce a memorable meal. A world-class chef with wilted vegetables cannot.

Your training dataset is your ingredient selection. Every subsequent decision — the recipe (SFT vs DPO), the technique (LoRA vs QLoRA), the plating (chat template, prompt masking) — is downstream. If you find yourself tuning hyperparameters for the 4th time in one week, stop and inspect the dataset. Nine times out of ten there's a duplicate cluster, a length outlier group, or a format inconsistency that no amount of hyperparameter voodoo will paper over.

### Model 2: Compound Interest on Bad Data

A dataset error compounds through training epochs. A duplicated example seen twice per epoch, for 3 epochs = 6 gradient updates on that specific example. If it's a *bad* example — inconsistent, toxic, wrong — you've reinforced that pattern 6 times.

Think of dataset cleaning as **payment on principal**. A 5-minute filter for length outliers saves you two days of debugging why your fine-tuned model produces truncated responses. A weekend of semantic deduplication saves you months of "my model has a weird phrase it keeps saying." The interest rate on unpaid dataset debt in fine-tuning is exorbitant.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — "More data is always better."**
Untrue for fine-tuning. Adding low-quality examples to a good dataset makes the fine-tune *worse*, not better — because gradient updates weight all examples the same, and the noise dilutes the signal. LIMA (Zhou et al., 2023) showed 1000 hand-curated examples outperformed 52k of Alpaca. Practical implication: if you're deciding between "50k mediocre examples" and "5k carefully curated," almost always pick 5k.

Corollary: when someone brags about their "100k SFT dataset," ask about their dedup and quality bar. If they don't have crisp answers, the number is meaningless.

**Pitfall #2 — Test-set contamination.**
You split train/val/test, then run your entire training pipeline on the train set. Evaluation on test shows amazing results. You ship. Users complain.

What happened: your val set was drawn *before* deduplication, and the same (or near-same) examples appear in both train and val. Model memorized them; val score reflects memorization, not generalization.

Fix: **deduplicate first, then split.** And when adding new data to an existing dataset, dedup against the *existing* val/test sets too, not just among the new rows.

**Pitfall #3 — Chat template baked into your JSONL.**
Some tutorials show storing the fully-formatted training string (with all the `<|start_header_id|>` control tokens) in the dataset itself. This locks the dataset to one specific model's template. If you switch base models (Llama 3.1 → Qwen 2.5, say), you have to reformat every row.

Fix: **store data in structured format** (messages list or instruction/output), let the framework apply the chat template at training time using the target model's tokenizer. Portable across base models. This is why the `messages` format dominates: it's the most portable structured form.

---

## 5. Self-Assessment Bank

**Q1 (MC):** For a focused SFT task with high-quality examples, the "sweet spot" dataset size is typically:
A) 50-100 examples
B) 500-5000 examples
C) 100000+ examples
D) Millions of examples

**Q2 (short):** What is "test-set contamination" in a fine-tuning context, and how do you prevent it?

**Q3 (MC):** Which of these is a **valid** source for a fine-tuning dataset in most contexts?
A) Only human-written examples
B) Only synthetic data from a stronger model
C) Both, ideally mixed with public instruction data for diversity
D) Neither — always use fresh data every run

**Q4 (short):** Explain the difference between exact dedup, fuzzy (MinHash) dedup, and semantic dedup. When would you use each?

**Q5 (MC):** You're distilling training data from GPT-5 to fine-tune Llama 3.1 8B. Best practice:
A) Ask GPT-5 to generate both prompts and responses
B) Use your own seed prompts, sample multiple GPT-5 responses at low temp, filter aggressively
C) Use GPT-5's responses raw without filtering
D) Use only GPT-5, no real data

**Q6 (short):** Your fine-tuned model produces responses shorter than users want. Where in the dataset pipeline would you look first?

**Q7 (MC):** Storing chat-template-formatted strings in your training JSONL is:
A) Best practice — makes training faster
B) Fine, no impact
C) A portability trap — locks the dataset to one specific base model's template
D) Required by TRL

**Q8 (short):** Give three specific quality filters you would apply to a raw scraped dataset before fine-tuning.

**Q9 (MC):** LIMA (Zhou et al. 2023) argued:
A) Larger models always beat smaller ones
B) 1000 carefully curated examples can rival 52k Alpaca-style examples for alignment
C) DPO always beats SFT
D) Full-parameter fine-tuning is superior to LoRA

**Q10 (short):** Your DPO dataset has 3000 pairs. On inspection, 400 pairs have the same "chosen" response (a canned "I'm sorry, I can't help with that") appearing as the winning answer against varied rejected responses. What's the risk and what would you do?

---

### Answer Key

**A1: B.** 500-5000 is the practical sweet spot for focused SFT. Below 500 the model has too few examples to generalize; above 5000 marginal returns drop rapidly for a focused task. Very large datasets are appropriate for broad capability tuning (general Instruct-style tuning, code, math) but for a specific behavior fine-tune, more data isn't better — quality dominates.

**A2:** Test-set contamination = examples in your test set (used to evaluate quality) also appear (exactly or near-exactly) in the training set. Result: the model has seen the test data during training; test scores reflect memorization, not generalization; production performance is worse than test scores suggest. Prevention: **deduplicate the entire pool first (exact + fuzzy + semantic dedup), then split into train/val/test with a fixed random seed.** When augmenting the dataset later, dedup new rows against existing val/test rows before adding to train. Never leak the test set into the training pool "just to see."

**A3: C.** Real production data + carefully-filtered synthetic distillation + a small dose of public instruction data (for diversity padding) is a standard, high-performing mix. Pure human-written is highest quality but doesn't scale. Pure synthetic risks compounding teacher biases. Pure public data doesn't cover your task. The blend leverages each source's strengths.

**A4:**
- **Exact string dedup**: hash `prompt + response`; drop matches. Fast (O(n) with a set), catches only literal duplicates. Always run.
- **Fuzzy dedup (MinHash / SimHash)**: fingerprints of word-shingles; catches lightly modified duplicates (paraphrases, whitespace changes, minor edits). Standard for large datasets. Faster than semantic; slower than exact. Run on any dataset > ~5000 rows.
- **Semantic dedup (embedding-based)**: compute embeddings, cosine-similarity threshold (e.g., 0.95). Catches actual paraphrases and semantic near-duplicates that MinHash misses. Most expensive; O(n²) or requires ANN index. Run when dataset is small enough or you have specific concerns about paraphrase duplicates.

Layer them: exact first, then MinHash, then semantic if needed.

**A5: B.** Best practice for teacher-student distillation: seed with your own prompts (from production logs or hand-crafted variety), sample multiple responses from the teacher at low temperature (0.3-0.5), filter aggressively (length, quality classifier, best-of-N by judge). (A) is a trap — asking the teacher for prompts too leads to prompt distribution collapse. (C) is careless — teacher outputs still need filtering. (D) is unbalanced — real data alongside synthetic is almost always better.

**A6:** **Response length distribution in the training data.** Symptoms: fine-tuned model producing terse responses often reflects a training set skewed toward short responses. Look at the histogram of response token counts. If the median is 30 and p95 is 60, the model has learned to be terse. Fix by either (a) filtering the source data to include only longer, more complete responses; (b) augmenting with well-formed longer examples; (c) reviewing the source pipeline for truncation bugs. Also check prompt masking — if you accidentally computed loss on prompts, the model may have learned to produce prompt-like continuations, cutting response length.

**A7: C.** Storing pre-formatted strings couples the dataset to one specific base model's template. If you switch base models (say Llama → Qwen → Mistral), you need to strip and reformat every row. The idiomatic modern approach: store as structured messages, apply the tokenizer's `apply_chat_template()` at training time using the actual base model's tokenizer. Portable, correct, and if the base model publishes a template update, you inherit it automatically.

**A8:** Three of many valid filters:
1. **Length filter** — drop rows with response < 20 tokens or > 4× median length (probably truncated or malformed).
2. **Language filter** — FastText or langdetect; drop non-target-language rows (unless multilingual is the goal).
3. **Toxicity filter** — Perspective API or a small classifier; drop toxic content unless the task specifically needs it.
4. **Refusal filter** — drop rows where the response is a refusal (if that's not what you're training).
5. **Duplicate filter** — exact + fuzzy dedup.
6. **Format-consistency filter** — drop rows that violate the target output schema (invalid JSON, missing fields, etc.).
7. **Quality-classifier filter** — a small model rates each response; keep top-N%.
8. **Length-distribution check** — visualize; investigate bimodal or heavy-tail distributions.

**A9: B.** LIMA — "Less Is More for Alignment" — showed that 1000 carefully hand-curated examples produced alignment quality rivaling models trained on much larger datasets (52k Alpaca). The paper is often overstated ("data doesn't matter beyond 1000") — the correct takeaway is that *quality* dominates *quantity* at moderate scale, so effort spent curating a small set beats effort spent scaling a noisy one. Still one of the most-cited applied fine-tuning results.

**A10:** **Risk: refusal collapse / preference dominance.** 400 out of 3000 pairs (13%) all having the same "chosen" response as a refusal template means DPO will learn to strongly prefer the refusal shape whenever the input is even loosely similar to what those 400 prompts covered. The model may start refusing safe, unrelated queries after training. Fixes:
1. **Downsample** the refusal-chosen pairs to a smaller number (say 40-80) so they don't dominate the gradient.
2. **Diversify chosen responses** — if refusal is genuinely the right chosen answer, vary the refusal wording so the model doesn't memorize one specific string.
3. **Balance** — for every refusal-chosen pair, include multiple pairs where the chosen answer is *helpful* (safe requests correctly answered), so the model learns "refuse when appropriate, help when appropriate."
4. **Evaluate refusal rate** on your production eval before shipping — if refusal rate on safe queries has climbed, roll back and rebalance.

The general principle: any single "chosen response pattern" that appears repeatedly in DPO data will be over-learned. Preference data needs diversity in the chosen side just as much as in the rejected side.

---

**Related modules:**
- `learning/01-sft-fundamentals.md` — how the data is consumed
- `learning/03-dpo-preference-tuning.md` — preference-dataset-specific concerns
- `learning/05-fine-tuning-ops.md` — running training on the curated data
- `../production-ai/learning/08-model-evaluation.md` — golden sets have the same discipline

**Practice prompts:**
1. Take any public instruction dataset (Alpaca, Dolly). Run exact + MinHash dedup. Report % removed. Plot the length distribution before and after.
2. Build a 300-example task-specific SFT set by hand or via distillation. Apply the filters listed above. Track the row count at each step.
3. For a preference dataset (real or synthetic), plot the distribution of "chosen response" first-100-chars. Look for the pattern of over-representation the answer to Q10 warns about.

**References:**
- Zhou et al., "LIMA: Less Is More for Alignment" (2023)
- Chen et al., "AlpaGasus: Training A Better Alpaca with Fewer Data" (2023)
- Wang et al., "How Far Can Camels Go? Exploring the State of Instruction Tuning" (2023)
- The Hugging Face `datasets` library's dedup and filter recipes
- Anthropic — HH-RLHF dataset methodology write-up
- Argilla and LangSmith blog posts on data-centric fine-tuning
