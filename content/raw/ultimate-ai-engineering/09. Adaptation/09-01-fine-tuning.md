# 09.01 — Fine-Tuning (SFT, Preference Tuning, and Reinforcement Fine-Tuning)

> **Module 9: Adaptation** · Subtopic 1 of 4
> **Prerequisites:**
> - 01.01 (transformer internals, cross-entropy);
> - 01.03 (tokenization, chat templates);
> - 02.01 (prompting baselines);
> - 04.x (RAG);
> - 06.01–06.02 (evals, datasets);
> - 08.04 (memory and roofline math);
> - 08.05 (release lifecycle).
>
> **Outcome:** you can decide when fine-tuning beats prompting and RAG, and build a clean training set. You can size the GPU memory for full fine-tuning, LoRA and QLoRA, and run supervised fine-tuning (SFT) with correct loss masking. You can add preference tuning (DPO) or reinforcement fine-tuning (RFT/GRPO) with verifiable rewards, and evaluate and ship the result as a governed bundle.

> **Status check (September 2026). Read first.**
> - **Hosted fine-tuning is shrinking and shifting.** OpenAI's docs now say it is *"winding down the fine-tuning platform. The platform is no longer accessible to new users, but existing users … will be able to create training jobs for the coming months."* Fine-tuned models stay available for inference until their base models are deprecated.
> - **Azure AI Foundry** (relevant to an Azure AD shop) still lists:
>   - **SFT** on gpt-4o-mini, gpt-4o, and gpt-4.1 / 4.1-mini / 4.1-nano;
>   - **DPO** on gpt-4o and the gpt-4.1 family;
>   - **RFT** on o4-mini (gpt-5 RFT is invitation-only);
>   - SFT on open models (Llama-3.3-70B-Instruct, Qwen-32B, gpt-oss-20b, Ministral-3B);
>   - three training tiers: Standard (data residency), Global, and Developer (preemptible, no SLA).
> - **Plan for the open-weights path.** It means Hugging Face **TRL 1.13** (SFT/DPO/GRPO/KTO/RLOO/Reward/Distillation trainers) plus **PEFT 0.21**, with serving on vLLM (08.03). Hosted fine-tuning of a vendor model ties you to that vendor's deprecation calendar (08.05 §5).

---

## 1. When to Fine-Tune (and When Not To)

```
 Problem ──► Does a strong prompt + few-shot (02.01) already hit the bar?  ── yes ──► STOP (cheapest, most flexible)
    │ no
    ├─► Is the gap KNOWLEDGE (facts, fresh or private documents)? ── yes ──► RAG (04.x). Fine-tuning does not reliably inject facts
    │ no
    ├─► Is the gap BEHAVIOUR: format, style, task procedure, tool-call reliability, domain register? ── yes ──► SFT
    ├─► Is the gap "which of two acceptable outputs is BETTER" (tone, concision, editorial preference)? ─────► DPO / KTO on top of SFT
    ├─► Is correctness VERIFIABLE by a program (extraction F1, tests pass, math, schema + business rules)? ──► RFT / GRPO
    └─► Is the gap COST/LATENCY (a frontier model works but is too slow or expensive)? ──────────────────────► distill into a small model (09.04)
```

| Fine-tuning is good at | Fine-tuning is bad at |
|---|---|
| Consistent output **format** and **style** (house style for bill summaries, fiscal notes) | Teaching **new facts** reliably. It raises hallucination risk on unseen facts; use RAG |
| Narrow **task procedures** (extract amendments → structured diff) | Knowledge that **changes** weekly (statutes amended each session) |
| **Shorter prompts**: behaviour moves from the prompt into the weights, cutting input tokens and TTFT (08.02) | Anything you can't **evaluate**. Without a held-out eval you are guessing |
| **Smaller models** matching larger ones on one task (with distillation, 09.04) | Broad capability gains. Narrow tuning often causes **regressions elsewhere** (forgetting, §6) |
| Tool-call **argument accuracy** for your schemas | Fixing safety alone. Tuning can *remove* safety behaviour (Qi et al., 2023); re-run 07.x evals |

**Fine-tuning and RAG work together.** Fine-tune the model to *use retrieved context well*: cite passages, say "not found", and follow the output schema. Then feed it fresh facts through RAG. Train on examples whose context comes from your real retriever, including distractor passages (RAFT, Zhang et al., 2024).

---

## 2. The SFT Objective, Precisely

Take a sequence $x_{1:T}$ with a **loss mask** $m_t \in \{0,1\}$, where $m_t = 1$ for tokens the model should learn to produce: assistant turns, not the system prompt, the user turn, or retrieved context. The SFT objective is then

$$
\mathcal{L}_{\text{SFT}}(\theta) \;=\; -\,\frac{1}{\sum_{t} m_t}\sum_{t=1}^{T-1} m_{t+1}\,\log p_\theta\!\left(x_{t+1}\mid x_{\le t}\right).
$$

Four details decide whether this works in practice:

1. **Loss masking** (`assistant_only_loss` / `completion_only_loss` in TRL). Training on prompt tokens wastes capacity, makes the model imitate *user* text, and in RAG setups teaches it to regurgitate context. Masking usually improves task quality; one exception is very short completions with scarce data.
2. **The chat template must be identical at training and at serving** (01.03 §7). This covers special tokens, the system-prompt placement, and the tool-call format. A template mismatch is the most common cause of "the fine-tuned model is worse than base". TRL's `assistant_only_loss` needs a template that marks assistant spans with `{% generation %}` tags, and many model templates lack them. Check that the masked fraction is what you expect (the test below prints it).
3. **Loss normalisation across gradient accumulation.** With a mean per micro-batch, short-completion micro-batches get the same weight as long ones. The correct gradient is **total token loss ÷ total unmasked tokens across the whole effective batch**. Recent Transformers/TRL versions handle this; custom loops often do not.
4. **Packing.** Concatenating examples to fill `max_length` removes padding waste, which can give a 2–5× throughput gain on short examples. It needs **per-document attention isolation** (position-id resets with FlashAttention variable-length kernels, or "padding-free" batching). Otherwise tokens attend across unrelated examples. TRL's default `packing_strategy="bfd"` (best-fit decreasing) is padding-free.

```python
import torch
import torch.nn.functional as F

def masked_nll_sum(logits, labels, ignore_index=-100):
    """Sum of next-token NLL over unmasked targets, plus the unmasked target count.
    logits: (B, T, V); labels: (B, T) aligned with inputs, -100 = masked."""
    sl, st = logits[:, :-1], labels[:, 1:]
    loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1), ignore_index=ignore_index, reduction="sum")
    return loss, int((st != ignore_index).sum())

def build_example(prompt_ids, completion_ids):
    """Loss mask: learn only the completion tokens."""
    return prompt_ids + completion_ids, [-100] * len(prompt_ids) + completion_ids

torch.manual_seed(0)
V = 50
ex_short = build_example(list(range(1, 40)), [7, 8, 9])                     # 3 completion tokens
ex_long = build_example(list(range(1, 10)), list(range(10, 43)))            # 33 completion tokens
logits = [torch.randn(1, len(ex[0]), V) for ex in (ex_short, ex_long)]
labels = [torch.tensor([ex[1]]) for ex in (ex_short, ex_long)]

sums, counts = zip(*(masked_nll_sum(lg, lb) for lg, lb in zip(logits, labels)))
correct = sum(sums) / sum(counts)                                            # global token mean
mean_of_means = sum(s / c for s, c in zip(sums, counts)) / 2                 # naive per-micro-batch mean
print(f"token-weighted loss {correct:.4f} vs mean-of-means {mean_of_means:.4f}; counts={counts}")
assert counts == (3, 33) and abs(float(correct - mean_of_means)) > 1e-3
```

The 3-token example gets **11× its fair weight** under mean-of-means (half the loss for 1/12 of the tokens). At scale this skews training toward short answers.

### 2.1 Running SFT with TRL

This setup is tested end-to-end on a tiny local model. In production, swap in your pinned base model.

```python
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-8B"                      # pin revision in production (model_init_kwargs={"revision": ...})
rows = [{"messages": [
    {"role": "system", "content": "You summarize Montana bills."},
    {"role": "user", "content": f"Summarize the bill in two sentences. {i}"},
    {"role": "assistant", "content": "The bill appropriates $2,000,000 from the general fund."}]} for i in range(64)]
train, val = Dataset.from_list(rows[:56]), Dataset.from_list(rows[56:])

cfg = SFTConfig(
    output_dir="out/sft-bill",
    assistant_only_loss=True,              # needs {% generation %} markers in the chat template
    max_length=4096, packing=False,        # packing=True + padding-free for short examples at scale
    learning_rate=2e-5,                    # full FT; LoRA typically ~10x higher (09.03)
    lr_scheduler_type="cosine", warmup_steps=0.03, num_train_epochs=2,
    per_device_train_batch_size=4, gradient_accumulation_steps=8, gradient_checkpointing=True,
    bf16=True, eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50,
    logging_steps=10, report_to="none")
trainer = SFTTrainer(model=MODEL_ID, args=cfg, train_dataset=train, eval_dataset=val)
batch = next(iter(trainer.get_train_dataloader()))
print("masked fraction of labels:", float((batch["labels"] == -100).float().mean()))   # sanity-check masking
trainer.train()
```

---

## 3. Memory: What Full Fine-Tuning Actually Costs

**Mixed-precision AdamW** keeps, per trainable parameter:
- BF16 weights (2 B);
- BF16 gradients (2 B);
- an FP32 master copy (4 B);
- Adam $m$ and $v$ (4 + 4 B).

That is **16 bytes per parameter** before activations, so an 8B model needs about 128 GB of static memory. **ZeRO/FSDP** shards this across $N$ GPUs:
- **stage 1** shards the optimizer state;
- **stage 2** also shards the gradients;
- **stage 3 / FSDP full-shard** also shards the weights.

**Activation memory** per layer, with FlashAttention (Korthikanti et al.), is $\approx 34\,sbh$ bytes, where $s$ is sequence length, $b$ the micro-batch size, and $h$ the hidden size. **Full activation checkpointing** keeps only each layer's input ($2sbh$ bytes) and recomputes the rest, at about 33% extra compute.

A frequently missed term is the **logits**: $s \cdot b \cdot V$ FP32 values, plus their gradient. With 128k–150k vocabularies at 4k context this is 4+ GB per sequence. **Chunked cross-entropy** (TRL's default `loss_type="chunked_nll"`, Liger kernels) never materialises the full logits.

```python
def train_memory_gb(params_b, method="full", seq=4096, micro_batch=1, layers=32, hidden=4096,
                    vocab=128_256, lora_trainable_b=0.0, checkpointing=True, chunked_logits=False):
    """Rough single-GPU training memory (GB). method: full | lora | qlora."""
    P = params_b * 1e9
    if method == "full":
        static = P * 16                                          # bf16 w + g, fp32 master, Adam m, v
    elif method == "lora":
        static = P * 2 + lora_trainable_b * 1e9 * 16             # frozen bf16 base + trainable adapters
    elif method == "qlora":
        static = P * 4.127 / 8 + lora_trainable_b * 1e9 * 16     # NF4 base + double-quant constants (09.03)
    else:
        raise ValueError(method)
    sbh = seq * micro_batch * hidden
    act = layers * 2 * sbh + 34 * sbh if checkpointing else layers * 34 * sbh
    logits = 0 if chunked_logits else seq * micro_batch * vocab * 4 * 2
    return {k: round(v / 1e9, 1) for k, v in
            dict(static=static, activations=act, logits=logits, total=static + act + logits).items()}

def zero_per_gpu_gb(params_b, n_gpus, stage):
    """Static per-GPU memory under ZeRO stage 0-3 (DeepSpeed) / FSDP equivalents."""
    P = params_b * 1e9
    w, g, o = 2 * P, 2 * P, 12 * P
    if stage >= 1: o /= n_gpus
    if stage >= 2: g /= n_gpus
    if stage >= 3: w /= n_gpus
    return round((w + g + o) / 1e9, 1)

for m in ("full", "lora", "qlora"):
    print(f"8B {m:5s}", train_memory_gb(8.03, m, lora_trainable_b=0.042))
print("8B qlora, chunked logits", train_memory_gb(8.03, "qlora", lora_trainable_b=0.042, chunked_logits=True))
print("ZeRO per GPU (8 GPUs), stages 0-3:", [zero_per_gpu_gb(8.03, 8, s) for s in range(4)])
assert zero_per_gpu_gb(8.03, 8, 0) == 128.5 and zero_per_gpu_gb(8.03, 8, 3) == 16.1
```

For an 8B Llama-style model, 4k context, micro-batch 1, with checkpointing:

| Method | Static | Activations | Logits (unchunked) | Total | Fits on |
|---|---|---|---|---|---|
| **Full FT** | 128.5 GB | 1.6 GB | 4.2 GB | **≈134 GB** | Multi-GPU: ZeRO stages 0→3 on 8 GPUs give 128.5 → 44.2 → 30.1 → 16.1 GB static/GPU |
| **LoRA** (r=16, all linear, 42M trainable) | 16.7 GB | 1.6 GB | 4.2 GB | **≈22.6 GB** | 1×24–48 GB |
| **QLoRA** | 4.8 GB | 1.6 GB | 4.2 GB | **≈10.7 GB** | 1×16–24 GB |

With chunked logits QLoRA drops to ≈6.5 GB, so for an 8B model with QLoRA **the logits can outweigh the model**.

---

## 4. Data: The Part That Decides the Outcome

**Quality and coverage beat volume.** LIMA (Zhou et al., 2023) showed 1,000 carefully curated examples can align a strong base model. Typical sizes by task:
- format or style tasks: **500–5,000** examples;
- complex procedures: 5k–50k;
- anything more needs deduplication and difficulty balancing.

**Sources, in order of value:**
1. **Real production inputs with expert-written or expert-corrected outputs.** For example, legislative services staff edits of AI drafts. These double as DPO pairs (§5).
2. **Historical artefacts** such as past bill summaries and fiscal notes paired with their bills.
3. **Synthetic outputs from a stronger model** (09.04). These need quality filtering and a licence/terms check.
4. Public datasets, for general-capability mixing (§6).

**Hygiene checklist.** Automate all of these (06.02):
- schema and role order;
- non-empty targets;
- a token-length distribution within `max_length`, with the truncation rate reported;
- exact and near-duplicate removal;
- **decontamination against eval sets** by n-gram overlap;
- PII scrubbing (07.01);
- a label-noise audit on a sample;
- slice balance, e.g. appropriations vs policy bills and short vs long.

```python
import hashlib, re
from collections import Counter

def _norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()

def _ngrams(s, n=8):
    w = _norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}

def validate_chat_rows(rows, count_tokens, max_tokens=4096, eval_texts=(), n=8, overlap_thresh=0.2):
    """Returns (clean_rows, report). Checks roles, empty targets, length, exact dups, eval contamination."""
    eval_grams = set().union(*(_ngrams(t, n) for t in eval_texts)) if eval_texts else set()
    seen, clean, issues = set(), [], Counter()
    for r in rows:
        msgs = r.get("messages", [])
        roles = [m["role"] for m in msgs]
        body = [x for x in roles if x != "system"]
        if not body or body[0] != "user" or body[-1] != "assistant" or any(a == b for a, b in zip(body, body[1:])):
            issues["bad_role_order"] += 1; continue
        if any(m["role"] == "assistant" and not m["content"].strip() for m in msgs):
            issues["empty_target"] += 1; continue
        text = "\n".join(m["content"] for m in msgs)
        if count_tokens(text) > max_tokens:
            issues["too_long"] += 1; continue
        h = hashlib.sha256(_norm(text).encode()).hexdigest()
        if h in seen:
            issues["duplicate"] += 1; continue
        g = _ngrams(text, n)
        if eval_grams and g and len(g & eval_grams) / len(g) > overlap_thresh:
            issues["eval_contamination"] += 1; continue
        seen.add(h); clean.append(r)
    return clean, dict(kept=len(clean), dropped=dict(issues))

bill = "An act revising county budget laws and amending section 7-6-4005 to require public hearings before adoption"
rows = [
    {"messages": [{"role": "user", "content": "Summarize: " + bill}, {"role": "assistant", "content": "Requires hearings."}]},
    {"messages": [{"role": "user", "content": "Summarize:  " + bill.upper()}, {"role": "assistant", "content": "requires hearings."}]},
    {"messages": [{"role": "assistant", "content": "Hi"}]},
    {"messages": [{"role": "user", "content": "Summarize HB 2"}, {"role": "assistant", "content": "  "}]},
    {"messages": [{"role": "user", "content": "Summarize HB 9 " + "word " * 5000}, {"role": "assistant", "content": "x"}]},
    {"messages": [{"role": "user", "content": "Summarize SB 12 which creates a grant program for rural broadband deployment in eligible counties"},
                  {"role": "assistant", "content": "Creates a grant program."}]},
]
clean, rep = validate_chat_rows(rows, count_tokens=lambda s: len(s.split()), max_tokens=2000,
                                eval_texts=["SB 12 which creates a grant program for rural broadband deployment in eligible counties"])
print(rep)
assert rep == dict(kept=1, dropped={"duplicate": 1, "bad_role_order": 1, "empty_target": 1,
                                    "too_long": 1, "eval_contamination": 1})
```

**Hyperparameter defaults that usually work.** Tune only after the data is right.

| Knob | Full FT | LoRA/QLoRA (09.03) | Notes |
|---|---|---|---|
| Learning rate | 1e-5 – 2e-5 | 1e-4 – 2e-4 (≈10× full FT) | Cosine, 3% warmup |
| Epochs | 1–3 | 2–4 | Watch eval loss *and* task metric; eval loss rising means overfitting |
| Effective batch | 64–256 sequences (or ~0.5–1M tokens) | 16–64 | Use gradient accumulation |
| Max length | Cover p99 of data | same | Report the truncation rate; truncated targets teach cut-off answers |
| Weight decay | 0–0.1 | 0 | |
| Precision | BF16 | BF16 compute | FP16 needs loss scaling; avoid |

---

## 5. Preference Tuning and Reinforcement Fine-Tuning

### 5.1 DPO — preference optimisation without a reward model

**RLHF** maximises $\mathbb{E}[r(x,y)] - \beta\,\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$, whose optimum is $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\,e^{r(x,y)/\beta}$. **DPO** (Rafailov et al., 2023) substitutes that optimum into the Bradley–Terry preference model. The result is a supervised loss on preference pairs $(x, y_w, y_l)$:

$$
\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\left[\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)\right].
$$

The coefficient $\beta$ controls how far the policy may move from the reference: small $\beta$ lets it move far. The code below checks the theory numerically. Given Bradley–Terry-distributed preferences, DPO converges to exactly $\pi_{\text{ref}}\,e^{r/\beta}/Z$.

```python
import torch
import torch.nn.functional as F

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """Inputs: summed completion log-probs per example. Returns (loss, implicit reward margin)."""
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(margin).mean(), margin.detach()

torch.manual_seed(0)
K = 6                                                        # candidate responses to one prompt
r = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])             # latent utility
ref_logits = torch.randn(K)
i, j = torch.triu_indices(K, K, 1)
p_ij = torch.sigmoid(r[i] - r[j])                            # Bradley-Terry P(i preferred over j)

def fit(beta, steps=3000):
    theta = ref_logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.05)
    ref_lp = F.log_softmax(ref_logits, 0)
    for _ in range(steps):
        lp = F.log_softmax(theta, 0)
        m = beta * ((lp[i] - ref_lp[i]) - (lp[j] - ref_lp[j]))
        loss = -(p_ij * F.logsigmoid(m) + (1 - p_ij) * F.logsigmoid(-m)).mean()   # expected DPO loss
        opt.zero_grad(); loss.backward(); opt.step()
    pi = F.softmax(theta.detach(), 0)
    analytic = F.softmax(ref_lp + r / beta, 0)
    kl = float((pi * (pi.log() - ref_lp)).sum())
    return pi, analytic, kl

for beta in (0.5, 1.0, 2.0):
    pi, analytic, kl = fit(beta)
    print(f"beta={beta}: max|pi - pi*|={float((pi - analytic).abs().max()):.4f}  KL(pi||ref)={kl:.3f}  P(best)={float(pi[-1]):.3f}")
    assert float((pi - analytic).abs().max()) < 1e-3
assert fit(0.5)[2] > fit(1.0)[2] > fit(2.0)[2]               # smaller beta -> further from reference
```

At β = 0.5 / 1 / 2 the fitted policy matches $\pi^*$ to within $10^{-4}$, with KL to the reference of 1.84 / 0.51 / 0.11 and probability on the best response of 0.62 / 0.23 / 0.10. That is **β as a trust-region knob**, confirmed numerically.

**DPO in practice:**
- **Start from an SFT model** and use it as $\pi_{\text{ref}}$.
- Pairs should be **on-policy**: both responses sampled from the SFT model, then ranked by humans or a judge. Off-policy pairs work less well.
- Typical settings: $\beta = 0.05$–$0.5$ (TRL default 0.1), learning rate 5e-7 – 5e-6, 1–3 epochs.
- **Monitor:**
  - reward accuracy and margin on held-out pairs;
  - **chosen log-probability**, which often *falls* even while the margin grows (likelihood displacement);
  - response length. DPO tends to lengthen outputs, so use length-controlled judges (06.01 §4).

**The variants,** by when to use them:
- **IPO** resists overfitting on deterministic preferences.
- **KTO** needs only thumbs-up/down labels, not pairs. It fits production feedback (06.03).
- **ORPO** and **SimPO** are reference-free and cheaper.
- **Online/iterative DPO** re-samples pairs from the current policy.

**A natural source of pairs in a legislature:** (AI draft, staff-edited final) for bill summaries or amendment explanations. The edited version is $y_w$ and the draft is $y_l$. Only use pairs where the edit is substantive (edit distance above a threshold), and exclude purely factual corrections. Those corrections are knowledge gaps that belong in RAG, not style preferences.

```python
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

MODEL_ID = "out/sft-bill"                                     # the SFT checkpoint is the reference policy
pairs = Dataset.from_list([{
    "prompt":   [{"role": "user", "content": f"Summarize the bill in two sentences. {k}"}],
    "chosen":   [{"role": "assistant", "content": "The bill appropriates $2,000,000 from the general fund."}],
    "rejected": [{"role": "assistant", "content": "This bill, which is a bill, does many things relating to money."}],
} for k in range(32)])
cfg = DPOConfig(output_dir="out/dpo-bill", beta=0.1, learning_rate=1e-6, num_train_epochs=1,
                per_device_train_batch_size=4, gradient_accumulation_steps=4, bf16=True,
                logging_steps=10, report_to="none")
DPOTrainer(model=MODEL_ID, args=cfg, train_dataset=pairs).train()
```

### 5.2 RFT / GRPO — optimise a verifiable reward

When correctness is **checkable by a program**, sample $G$ completions per prompt, score each with a reward $r_i$, and push up the ones that beat their siblings. **GRPO** (Shao et al., DeepSeekMath) drops PPO's value network and uses a **group-relative advantage**:

$$
\hat A_i = \frac{r_i - \mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G}) + \epsilon},
\qquad
\mathcal{L} = -\frac{1}{\sum_i |y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\min\!\Big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon_{\text{high}})\hat A_i\Big),
$$

where $\rho_{i,t}$ is the token-level importance ratio between the new and old policy.

The normalisation shown is the token-level ("DAPO") aggregation. It is **TRL 1.13's default** (`loss_type="dapo"`), with `beta=0.0`, meaning no KL penalty by default, and `scale_rewards="group"`.

Consequences of this formula:
- **Groups where every sample gets the same reward produce zero advantage and no learning signal.** Keep only prompts that the model solves *sometimes*, via difficulty filtering or dynamic sampling.
- **Rewards are hackable.** Combine a correctness term with format and length terms, cap lengths, and read samples every run.

```python
import json, re
import numpy as np

def group_advantages(rewards, scale="group", eps=1e-4):
    """rewards: (n_prompts, G). scale: 'group' (GRPO), 'batch', or 'none' (Dr. GRPO-style)."""
    r = np.asarray(rewards, dtype=float)
    a = r - r.mean(axis=1, keepdims=True)
    if scale == "group":
        a = a / (r.std(axis=1, keepdims=True) + eps)
    elif scale == "batch":
        a = a / (r.std() + eps)
    return a

MCA = re.compile(r"\b\d{1,2}-\d{1,2}-\d{3,4}\b")              # Montana Code Annotated section, e.g. 7-6-4005

def citation_reward(completion: str, gold: set[str]) -> float:
    """Verifiable reward for 'list every MCA section this bill amends' as JSON {"sections": [...]}:
    F1 on sections, plus a small format bonus, minus a length penalty for rambling."""
    try:
        pred = set(json.loads(completion)["sections"]); fmt = 0.1
    except Exception:
        pred = set(MCA.findall(completion)); fmt = 0.0
    tp = len(pred & gold)
    f1 = 0.0 if not pred or not gold else 2 * tp / (len(pred) + len(gold))
    return round(f1 + fmt - 0.1 * (len(completion) > 400), 4)

gold = {"7-6-4005", "7-6-4006"}
samples = ['{"sections": ["7-6-4005", "7-6-4006"]}', '{"sections": ["7-6-4005"]}',
           'Amends 7-6-4005 and 7-6-4006.', '{"sections": ["15-30-2101"]}']
R = [[citation_reward(s, gold) for s in samples], [1.0, 1.0, 1.0, 1.0]]
A = group_advantages(R)
print("rewards", R[0], "advantages", A[0].round(2))
assert R[0] == [1.1, 0.7667, 1.0, 0.1] and np.allclose(A[1], 0)           # uniform group -> no signal
assert A[0].argmax() == 0 and A[0].argmin() == 3
```

The TRL training loop for the same reward:

```python
import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "out/sft-bill"
data = Dataset.from_list([{"prompt": [{"role": "user", "content": f"List amended MCA sections as JSON. Bill {k}: ... 7-6-4005 ..."}],
                           "gold": ["7-6-4005"]} for k in range(16)])

def reward_sections(completions, gold, **kw):            # extra dataset columns arrive as kwargs
    """Compact F1 reward; in production call citation_reward() from the block above."""
    out = []
    for c, g in zip(completions, gold):
        text = c[0]["content"] if isinstance(c, list) else c
        try:
            pred = set(json.loads(text)["sections"])
        except Exception:
            pred = set()
        out.append(0.0 if not pred else 2 * len(pred & set(g)) / (len(pred) + len(g)))
    return out

cfg = GRPOConfig(output_dir="out/grpo-cite", num_generations=8, max_completion_length=256, temperature=1.0,
                 learning_rate=1e-6, per_device_train_batch_size=8, gradient_accumulation_steps=4,
                 bf16=True, logging_steps=5, report_to="none")   # use_vllm=True for fast rollouts at scale
GRPOTrainer(model=MODEL_ID, reward_funcs=reward_sections, args=cfg, train_dataset=data).train()
```

The tiny-model test run of this loop shows the failure mode directly. An untrained model never emits valid JSON, so every group scores 0, TRL logs `frac_reward_zero_std = 1`, and the loss is exactly 0. **Warm-start with SFT and filter prompts by difficulty before RL.**

**Hosted RFT** (Azure Foundry on o4-mini; OpenAI for existing customers) takes a *grader* definition (string match, model grader, or Python) instead of a reward function. The same design rules apply: a verifiable signal, difficulty-filtered prompts, and held-out evaluation.

---

## 6. Forgetting, Safety Drift, and Regression Control

Narrow fine-tuning **overwrites** general behaviour. Three failure modes are common:
- **catastrophic forgetting**: general question answering, reasoning, and instruction-following degrade;
- **safety erosion**: even benign fine-tuning data can weaken refusals (Qi et al., 2023);
- **format lock-in**: the model answers everything in the trained format.

Mitigations:

| Mitigation | How |
|---|---|
| **Mix general data** | Add 5–20% general instruction and chat data (licensed) to the task data |
| **Lower LR, fewer epochs; prefer LoRA** | LoRA "learns less and forgets less" (Biderman et al., 2024); see 09.03 |
| **Regression suite** | Always evaluate base vs tuned on: task eval, a general-capability slice, a **safety/refusal suite (07.01–07.03)**, tool-call and schema validity. Gate with 08.04 §4's `quant_gate`-style paired per-slice test |
| **Keep the base on the router** | Route out-of-domain requests to the base or general model (08.01 §3) rather than forcing the tuned model onto everything |

---

## 7. Evaluation and Shipping

- **The comparison that matters** is tuned model vs **best-effort prompted base** vs **prompted frontier model**, on the same held-out items, with CIs (06.01). Many fine-tunes lose to a better prompt, so beat the strongest baseline, not the weakest.
- **Split by time or by source, not at random.** For example, train on 2023–2025 sessions and test on 2026 interim drafts. This prevents near-duplicate leakage and tests generalisation.
- **Learning curves.** Train on 25/50/100% of the data. If the task metric is still rising at 100%, more data helps more than more tuning.
- **Ship as a bundle** (08.05 §2). Record the base model and revision, adapter or merged digest, data manifest hash, training config, eval report, and licence of the base and data. Serve as a merged model or as a LoRA adapter on vLLM (`--enable-lora`, 08.03 Project 3).
- **Plan retraining.** Trigger it on a drift-canary alert or a new session's style guide. **A new base model means redoing the fine-tune**, so budget it into the base model's lifecycle.

---

## 8. Production Challenges and Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **Template mismatch** | Tuned model worse than base; stray special tokens | Same template at train and serve; unit-test rendered prompts; check masked fraction |
| **Loss on the wrong tokens** | Model echoes user or context text | `assistant_only_loss` or completion-only loss; verify labels |
| **Truncated targets** | Answers cut off mid-sentence | Raise `max_length` or drop over-long examples; report the truncation rate |
| **Overfitting** | Eval loss rises after epoch 1; outputs become templated | Fewer epochs, more diverse data, early stopping on the *task metric* |
| **"Fine-tuned it on our statutes, still hallucinates"** | Facts wrong or out of date | Facts belong in RAG; fine-tune for *using* retrieved context (RAFT) |
| **DPO length inflation** | Longer, wordier outputs | Length-controlled eval; length penalty or SimPO; balance chosen and rejected lengths |
| **Reward hacking (GRPO)** | Reward rises, quality falls | Multi-term rewards, length caps, held-out human review, KL anchor ($\beta>0$) if drift appears |
| **Safety erosion** | Refusal rate drops after tuning | Include safety data; run the 07.x suites as release gates |
| **Vendor platform changes** | Hosted fine-tune can't be retrained | Keep datasets and eval portable; maintain an open-weights path |
| **Evaluation leakage** | Great offline, mediocre online | Time/source splits, n-gram decontamination, shadow deploy (08.05 §4) |

---

## 9. Hands-On Projects

### Project 1 — House-Style Bill Summarizer (SFT) vs Prompted Baselines

**User stories**
- *As a legislative services editor*, I want AI bill summaries in our house style (length, structure, neutral register, fiscal-impact line) so that I edit less.
- *As the platform owner*, I want a self-hosted 7–8B model to match a prompted frontier model on this task at a fraction of the cost.

**Acceptance criteria**
- The dataset holds ≥ 2,000 (bill text, final published summary) pairs from past sessions. It passes `validate_chat_rows`, has 0 eval contamination, and has a truncation rate < 1%.
- Splits are by session: train on sessions ≤ 2023, validate on 2025 interim drafts, test on 2025 session bills.
- The tuned model **beats the best-prompted base model** and comes within the non-inferiority margin (−2 pts) of a prompted frontier model on a judge rubric (06.01 §4), plus exact-match on fiscal figures. Every slice (appropriations, policy, long bills) passes.
- The general-capability and safety regression suites show no significant drop.
- The cost memo gives $/1k summaries self-hosted (08.01 TCO) vs the frontier API.

**Step-by-step**
1. Extract the pairs and normalise formatting. Render them with the base model's chat template and add a system prompt (the one used at serving).
2. Validate and deduplicate the data, and decontaminate it against the test sessions.
3. Train LoRA SFT (09.03) with TRL, then run a full-FT comparison if the budget allows. Log the masked fraction and the learning curves (25/50/100%).
4. Evaluate against both baselines with a paired bootstrap per slice.
5. Serve the result with vLLM as an adapter (08.03) and write the bundle manifest and model card (08.05).

### Project 2 — Learning Editorial Preferences with DPO from Staff Edits

**User stories**
- *As an editor*, I want the model to learn from the corrections I already make, without writing labelling guidelines.

**Acceptance criteria**
- The pair pipeline mines (AI draft, staff final) pairs, keeps substantive style edits (normalised edit distance above a threshold), and routes factual-correction pairs to a "knowledge gap" report for RAG. At least 1,000 pairs.
- DPO is trained on the Project 1 SFT model. The β sweep {0.05, 0.1, 0.3} reports held-out reward accuracy, the chosen log-probability trend, and length change.
- On a blind side-by-side, editors prefer the DPO model to SFT ≥ 60% of the time with a 95% CI excluding 50%. Median length change is ≤ +10%.
- A KTO variant using only thumbs-up/down from the UI is compared against it.

**Step-by-step**
1. Join drafts to finals by (bill, version). Compute edit metrics, then classify edits as style or fact with a small judge plus human spot-checks.
2. Build `{prompt, chosen, rejected}` rows and hold out 10% by bill.
3. Run the DPO sweep and track margins, chosen log-probability, and length.
4. Run a blind human evaluation with randomised order. Analyse it with a paired sign test / bootstrap.
5. Document and ship through 08.05 gates.

### Project 3 — RFT for Statute-Citation Extraction with a Verifiable Reward

**User stories**
- *As a bill drafter*, I need every MCA section a bill amends, repeals, or creates listed exactly, because a missed section is a drafting error.

**Acceptance criteria**
- The gold set is built from enrolled bills' section headings (≥ 1,500 bills). The reward is `citation_reward` extended with action type (amend, repeal, new) and exact JSON schema validity.
- Prompts are difficulty-filtered: the SFT model's pass@8 is between 10% and 90%.
- GRPO improves test F1 by ≥ 5 pts over SFT and beats a regex + LLM baseline. JSON validity stays ≥ 99.5%, and reward-hacking checks (length, degenerate outputs) show no regressions.
- An ablation covers `scale_rewards` group vs none, β 0 vs 0.04, and 4 vs 8 generations.

**Step-by-step**
1. Parse enrolled bills into (text, gold sections with actions). Split by session.
2. Run SFT warm-start (Project 1 pipeline) on 300 examples.
3. Estimate pass@8 per prompt and filter.
4. Run GRPO with vLLM-backed generation. Log reward components separately and read 20 samples per checkpoint.
5. Run the ablations, the final eval with CIs, and the bundle release.

---

## 10. Foundational Papers (exact titles)

- Ouyang et al., 2022 — *Training language models to follow instructions with human feedback*
- Wei et al., 2021 — *Finetuned Language Models Are Zero-Shot Learners*
- Zhou et al., 2023 — *LIMA: Less Is More for Alignment*
- Rafailov et al., 2023 — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- Azar et al., 2023 — *A General Theoretical Paradigm to Understand Learning from Human Preferences* (IPO)
- Ethayarajh et al., 2024 — *KTO: Model Alignment as Prospect Theoretic Optimization*
- Hong, Lee & Thorne, 2024 — *ORPO: Monolithic Preference Optimization without Reference Model*
- Meng, Xia & Chen, 2024 — *SimPO: Simple Preference Optimization with a Reference-Free Reward*
- Schulman et al., 2017 — *Proximal Policy Optimization Algorithms*
- Shao et al., 2024 — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO)
- Yu et al., 2025 — *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*
- Liu et al., 2025 — *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO)
- Zhang et al., 2024 — *RAFT: Adapting Language Model to Domain Specific RAG*
- Qi et al., 2023 — *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!*
- Gekhman et al., 2024 — *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*
- Rajbhandari et al., 2019 — *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*
- Korthikanti et al., 2022 — *Reducing Activation Recomputation in Large Transformer Models*

## 11. Essential Tooling

| Tool | Use |
|---|---|
| **Hugging Face TRL 1.13** | SFT, DPO, KTO, GRPO, RLOO, Reward, Distillation trainers; vLLM-backed rollouts |
| **PEFT 0.21** | LoRA/QLoRA and friends (09.02–09.03) |
| **Axolotl, LLaMA-Factory, Unsloth, torchtune** | Config-driven or memory-optimised fine-tuning recipes |
| **DeepSpeed ZeRO / PyTorch FSDP2, Accelerate** | Sharded full fine-tuning |
| **Liger Kernel, FlashAttention** | Fused/chunked losses, padding-free packing |
| **verl, OpenRLHF** | Large-scale RL post-training |
| **Azure AI Foundry fine-tuning** | Hosted SFT/DPO/RFT with data residency tiers |
| **lm-evaluation-harness, your 06.01 harness** | Regression and task evals |
| **Weights & Biases / MLflow** | Run tracking, config and dataset lineage |

**Image prompt (decision flow):** *"Clean flowchart titled 'Should I fine-tune?': start box 'Quality gap' → diamonds 'Prompting enough?', 'Knowledge gap?' (→ RAG), 'Behaviour/format gap?' (→ SFT), 'Preference between good outputs?' (→ DPO/KTO), 'Programmatically verifiable?' (→ RFT/GRPO), 'Cost/latency?' (→ Distillation). Flat vector, white background, colour-coded terminal boxes, sans-serif."*

**Image prompt (loss masking):** *"Diagram of a tokenized chat sequence as a horizontal strip of tokens: system (grey), user (blue), retrieved context (light blue), assistant (green). Below, a loss-mask row with 0s under grey/blue and 1s under green. Caption 'SFT loss only on assistant tokens'. Minimal technical illustration."*
