# 09.04 — Distillation (Compressing a Teacher into a Cheaper Student)

> **Module 9: Adaptation** · Subtopic 4 of 4
> **Prerequisites:** 09.01 (SFT, DPO, GRPO), 09.03 (LoRA/QLoRA), 08.01 (cascades and TCO), 08.04 (quantization as the *other* compression lever), 06.01 (evals).
> **Outcome:** you can pick the right distillation regime (logit, sequence-level, on-policy, rationale, or reasoning-trace) for the access you have to the teacher. You can implement and reason about the losses (temperature-scaled KL, forward vs reverse KL, generalised JSD), build verified synthetic datasets, run on-policy distillation with TRL, and prove with evals and cost math that the student is good enough to replace or front the teacher.

> **Status check (September 2026).**
> - **TRL 1.13** ships a stable **`DistillationTrainer`**. It trains **on-policy**: the student generates its own completions and is trained toward the teacher's token distributions under a **generalised JSD** with `beta`:
>   - `beta=0` is forward KL, `beta=1` is reverse KL, and the default is 1.0;
>   - there is no reference-model KL penalty.
>
>   Experimental variants (GKD, MiniLLM, GOLD, server-/async-distillation) live under `trl.experimental`.
> - **Terms matter as much as losses.** Many API providers' terms restrict using outputs to develop competing models, and open-weights licences can impose naming, attribution, or use conditions on derivatives. Check the teacher's current terms and licence **before** generating data (§7).

---

## 1. Why Distil, and the Four Regimes

A frontier model (or your own 70B fine-tune) solves the task but is too slow or too expensive. **Distillation** transfers its behaviour on *your task distribution* into a smaller student. That can cut cost and latency by 10–100× (08.01 §2 TCO). Distillation complements quantization (08.04), and the two stack: distil 70B → 8B, then FP8 the 8B.

```
 Teacher access ─────────────────────────────► Regime ─────────────────────────────────► Needs
 Only text outputs (API)                      SEQUENCE-LEVEL KD: SFT on teacher outputs      prompts + verifier/judge (§4)
                                              (+ rationale / reasoning-trace distillation)
 Full logits, same tokenizer (open weights)   TOKEN-LEVEL (logit) KD: match p_teacher per     teacher forward passes;
                                              token on fixed data (Hinton KD)                 same vocab (or alignment)
 Logits + can run teacher online              ON-POLICY KD (GKD / MiniLLM): student samples,  teacher inference in the loop
                                              teacher scores every token of the student's
                                              own outputs → fixes exposure bias
 Hidden states, compatible architecture       FEATURE KD (DistilBERT, MiniLM, TinyBERT)      architecture coupling; mostly
                                                                                               for encoders/embedders
```

**Rule of thumb:**
- Start with **sequence-level KD**. It is cheap, works with any teacher, and is usually 80% of the win.
- Add **on-policy KD** when the student must generate long outputs. Errors compound once the student leaves the teacher's distribution (exposure bias), and on-policy training corrects the states the student actually visits.

---

## 2. The Losses

### 2.1 Temperature-scaled KD (Hinton et al., 2015)

With logits $z_s$ and $z_t$, temperature $T$, and softened distributions $p^{T} = \mathrm{softmax}(z/T)$:

$$
\mathcal{L}_{\text{KD}} = (1-\lambda)\,\mathrm{CE}\big(y,\ p_s^{1}\big) \;+\; \lambda\,T^{2}\,\mathrm{KL}\big(p_t^{T}\,\Vert\,p_s^{T}\big).
$$

Two ingredients make this work:
- **Higher $T$** exposes the teacher's "dark knowledge": the relative probabilities of wrong-but-plausible tokens.
- **The $T^2$ factor.** For large $T$, $\partial\,\mathrm{KL}/\partial z_s \approx \frac{1}{NT^2}(z_s - z_t)$, so without the factor the soft-target gradient vanishes as $T^{-2}$ and $\lambda$ would have to be re-tuned for every $T$.

### 2.2 Forward vs reverse KL: which mistakes the student makes

- **Forward KL** $\mathrm{KL}(p_t \Vert p_s)$, used in standard KD and in SFT on teacher samples, is **mode-covering**. The student must put mass wherever the teacher does. A student too small to represent every mode spreads mass *between* them, onto outputs the teacher would never produce: hallucination-like blends.
- **Reverse KL** $\mathrm{KL}(p_s \Vert p_t)$ is **mode-seeking**. The student commits to modes it can represent and avoids regions the teacher considers impossible. It gives up diversity in exchange for precision. MiniLLM (Gu et al.) argues this is the right default for generative students.
- **Generalised JSD** interpolates between the two (GKD; TRL's `beta`):

$$
\mathrm{JSD}_\beta(p_t, p_s) = \beta\,\mathrm{KL}(p_t \Vert m) + (1-\beta)\,\mathrm{KL}(p_s \Vert m),\qquad m = \beta p_t + (1-\beta) p_s .
$$

TRL special-cases $\beta = 0$ as forward KL and $\beta = 1$ as reverse KL.

The experiment below fits a capacity-limited student (a single discretised Gaussian) to a bimodal teacher:

```python
import torch

torch.set_num_threads(1)
V = 60
x = torch.arange(V, dtype=torch.float32)

def disc_gauss(mu, log_sigma):
    return torch.log_softmax(-(x - mu) ** 2 / (2 * torch.exp(log_sigma) ** 2), dim=0)

teacher = torch.logsumexp(torch.stack([disc_gauss(torch.tensor(15.), torch.tensor(3.).log()),
                                       disc_gauss(torch.tensor(45.), torch.tensor(3.).log())]), 0) - torch.log(torch.tensor(2.))

def fit(divergence, steps=4000):
    mu = torch.tensor(20.0, requires_grad=True); ls = torch.tensor(0.7, requires_grad=True)
    opt = torch.optim.Adam([mu, ls], lr=0.1)
    for _ in range(steps):
        s = disc_gauss(mu, ls)
        loss = (teacher.exp() * (teacher - s)).sum() if divergence == "forward" else (s.exp() * (s - teacher)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    s = disc_gauss(mu, ls).detach().exp()
    low = teacher.exp() < 1e-3                                   # outputs the teacher considers ~impossible
    return dict(mu=round(float(mu.detach()), 1), sigma=round(float(ls.detach().exp()), 1),
                mass_on_teacher_impossible=round(float(s[low].sum()), 3),
                teacher_mass_covered=round(float(teacher.exp()[s > 1e-3].sum()), 3))

fwd, rev = fit("forward"), fit("reverse")
print("forward KL:", fwd); print("reverse KL:", rev)
assert fwd["mass_on_teacher_impossible"] > 0.2 and rev["mass_on_teacher_impossible"] < 0.02
assert fwd["teacher_mass_covered"] > 0.95 and rev["teacher_mass_covered"] < 0.6
```

| Objective | Student fit | Mass on outputs the teacher deems impossible | Teacher modes covered |
|---|---|---|---|
| Forward KL | μ≈30.6, σ≈22: straddles both modes | **41.7%** | 100% |
| Reverse KL | μ=15.0, σ=3.0: one mode, exactly | **0.4%** | 50% |

**For extraction, citation, and drafting tasks, where precision beats diversity, prefer reverse-leaning objectives** (`beta` near 1) or on-policy training. For creative or broad assistants, forward or mid JSD keeps coverage. Validate the choice with evals rather than theory.

### 2.3 Implementations: KD loss, sparse top-k teacher logits, generalised JSD

Storing full teacher distributions is infeasible at LLM vocabulary sizes (128k–256k). **Sparse KD** keeps the teacher's top-$k$ log-probs per position and renormalises over them.

```python
import torch, torch.nn.functional as F

torch.set_num_threads(1); torch.manual_seed(0)

def kd_loss(student_logits, teacher_logits, labels=None, T=2.0, lam=0.5, ignore_index=-100):
    """Hinton KD: (1-lam)*CE(labels) + lam*T^2*KL(p_t^T || p_s^T), over unmasked positions."""
    mask = torch.ones(student_logits.shape[:-1], dtype=torch.bool) if labels is None else labels != ignore_index
    s_T = F.log_softmax(student_logits[mask] / T, -1)
    t_T = F.log_softmax(teacher_logits[mask] / T, -1)
    kd = (t_T.exp() * (t_T - s_T)).sum(-1).mean() * T ** 2
    if labels is None or lam == 1.0:
        return kd
    return (1 - lam) * F.cross_entropy(student_logits[mask], labels[mask]) + lam * kd

def topk_kd(student_logits, teacher_topk_logp, teacher_topk_idx):
    """Sparse forward KL using stored teacher top-k log-probs, renormalised over the k tokens."""
    p = torch.softmax(teacher_topk_logp, -1)
    s = F.log_softmax(student_logits, -1).gather(-1, teacher_topk_idx)
    return (p * (p.clamp_min(1e-12).log() - s)).sum(-1).mean()

V, N = 32_000, 64
ranks = torch.arange(1, V + 1, dtype=torch.float32)
zipf = -1.3 * ranks.log()                                                   # Zipf-like next-token distributions
teacher = torch.stack([zipf[torch.randperm(V)] for _ in range(N)]) + 0.3 * torch.randn(N, V)
student = (teacher + torch.randn(N, V) * 2.0).requires_grad_(True)

for T in (1.0, 2.0, 4.0, 8.0):                                               # why the T^2 factor
    g = {}
    for scaled in (False, True):
        s = student.detach().clone().requires_grad_(True)
        (kd_loss(s, teacher, T=T, lam=1.0) / (1 if scaled else T ** 2)).backward()
        g[scaled] = float(s.grad.norm())
    print(f"T={T}: |grad| with T^2 = {g[True]:.4f}   without = {g[False]:.5f}")

g_full, = torch.autograd.grad(kd_loss(student, teacher, T=1.0, lam=1.0), student)
tlp = F.log_softmax(teacher, -1)
cos = {}
for k in (1, 8, 64, 512):
    v, idx = tlp.topk(k, -1)
    g_k, = torch.autograd.grad(topk_kd(student, v, idx), student)
    cos[k] = float(F.cosine_similarity(g_k.flatten(), g_full.flatten(), dim=0))
    print(f"top-{k:<4d} teacher mass kept={float(v.exp().sum(-1).mean()):.3f}  grad cosine vs full KL={cos[k]:.3f}")
assert cos[512] > cos[64] > cos[8] > cos[1]
tokens = 200e6
print(f"storage, {tokens/1e6:.0f}M tokens: full bf16 logits {tokens * V * 2 / 1e12:.1f} TB | "
      f"top-64 (fp16 logp + int32 id) {tokens * 64 * 6 / 1e9:.0f} GB")
```

| Top-k stored | Teacher mass kept | Gradient cosine vs full KL |
|---|---|---|
| 1 (= hard label) | 0.27 | 0.62 |
| 8 | 0.58 | 0.91 |
| **64** | 0.79 | **0.98** |
| 512 | 0.91 | 0.997 |

- **Top-64 recovers 98% of the full-KL gradient direction.** For 200M training tokens it needs **77 GB instead of 12.8 TB** of full BF16 logits at a 32k vocabulary. Real LLM distributions are usually more peaked than this Zipf toy, so small $k$ does even better.
- **The $T^2$ factor** keeps the soft-target gradient roughly constant for $T \ge 4$ (0.0017 → 0.0014). Without it the gradient falls as $T^{-2}$ (0.00011 → 0.00002).

```python
import torch, torch.nn.functional as F

def generalized_jsd(student_logits, teacher_logits, beta=0.5, T=1.0):
    """GKD divergence per position, TRL convention: beta=0 -> KL(t||s) forward; beta=1 -> KL(s||t) reverse;
    otherwise beta*KL(t||m) + (1-beta)*KL(s||m) with m = beta*t + (1-beta)*s."""
    s = F.log_softmax(student_logits / T, -1); t = F.log_softmax(teacher_logits / T, -1)
    kl = lambda p, q: (p.exp() * (p - q)).sum(-1)
    if beta == 0.0: return kl(t, s)
    if beta == 1.0: return kl(s, t)
    b = torch.tensor(beta)
    m = torch.logsumexp(torch.stack([s + torch.log1p(-b), t + torch.log(b)]), 0)
    return b * kl(t, m) + (1 - b) * kl(s, m)

torch.manual_seed(0)
sl, tl = torch.randn(8, 100), torch.randn(8, 100) * 2
ref_fwd = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), reduction="none", log_target=True).sum(-1)
assert torch.allclose(generalized_jsd(sl, tl, 0.0), ref_fwd, atol=1e-6)            # matches PyTorch/TRL forward KL
j = generalized_jsd(sl, tl, 0.5)
assert torch.all(j >= 0) and torch.all(j <= torch.log(torch.tensor(2.0)) + 1e-6)   # symmetric JSD is bounded by ln 2
print(f"forward {ref_fwd.mean():.3f}  reverse {generalized_jsd(sl, tl, 1.0).mean():.3f}  JSD(0.5) {j.mean():.3f}")
```

**Tokenizer mismatch.** Logit KD requires the teacher and student to share a vocabulary: same family (Llama 70B → 8B, Qwen 32B → 4B) or the same tokenizer. Across families, use sequence-level KD, or cross-tokenizer methods (e.g. ULD, *Universal Logit Distillation*, which matches sorted probability vectors), and validate carefully.

---

## 3. On-Policy Distillation

**Off-policy KD**, meaning training on teacher-generated or ground-truth prefixes, teaches the student what to do *in states the teacher visits*. At inference time the student conditions on **its own** earlier tokens. One early deviation puts it in states it never trained on, and errors compound. This is **exposure bias**, and it grows with output length.

**On-policy KD** (GKD, Agarwal et al.; MiniLLM) makes three changes:
1. The student samples $y \sim \pi_s(\cdot|x)$.
2. The teacher computes $p_t(\cdot \mid x, y_{<t})$ **for every token of the student's own sample**.
3. The loss is $\sum_t D\big(p_t(\cdot|x,y_{<t}),\ p_s(\cdot|x,y_{<t})\big)$.

This gives **dense, per-token supervision on the student's own distribution**. That combination is what makes on-policy KD dramatically more sample-efficient than RL with a sparse reward (09.01 §5.2) while fixing exposure bias.

Costs and requirements:
- teacher forward passes over the student's samples;
- student generation in the loop, where vLLM-backed generation matters;
- a shared vocabulary.

The TRL loop below is tested on tiny local models that share a tokenizer:

```python
from datasets import Dataset
from trl import DistillationConfig, DistillationTrainer

STUDENT_ID = "Qwen/Qwen3-1.7B"                    # start from an SFT'd student (sequence-level KD first)
TEACHER_ID = "Qwen/Qwen3-32B"                     # same tokenizer family
prompts = Dataset.from_list([{"prompt": [{"role": "user", "content": f"Summarize the bill in two sentences. {k}"}]}
                             for k in range(32)])
cfg = DistillationConfig(output_dir="out/distill-bill", beta=0.5, temperature=1.0, max_completion_length=64,
                         learning_rate=1e-5, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                         bf16=True, logging_steps=1, report_to="none")   # use_vllm=True for fast student rollouts
trainer = DistillationTrainer(model=STUDENT_ID, teacher_model=TEACHER_ID, args=cfg, train_dataset=prompts)
trainer.train()
```

**The recipe that works in practice** (e.g. the Qwen3 small models, and Thinking Machines' *On-Policy Distillation* write-up):
1. **Sequence-level SFT** on teacher outputs to get the student into the right region.
2. **On-policy KD** with reverse-leaning divergence to polish, especially for long reasoning or structured outputs.
3. **Optionally RL** (GRPO) with verifiable rewards for the last mile.

---

## 4. Sequence-Level (Black-Box) Distillation Done Right

This is **SFT on teacher outputs**. Its quality ceiling is the **quality of the filtered data**, not the teacher's average quality. The pipeline:

```
 real prompts (production logs, PII-scrubbed)     ┌─ teacher @ T>0, n samples/prompt (+ rationale / reasoning trace)
 + synthetic prompt expansion for rare slices ───►│
                                                  └─ VERIFY: programmatic checks (schema, citations exist in source,
                                                     numbers match fiscal tables) · judge rubric (06.01 §4) ·
                                                     self-consistency across samples · dedupe · decontaminate vs eval
                                                  ─► keep best verified (shortest/cleanest) ─► SFT dataset (09.01)
```

```python
import hashlib, json, re
from collections import Counter

def build_distillation_set(prompts, teacher_generate, verify, n=4, min_agree=2, dedupe=True):
    """Sample n teacher outputs per prompt, keep verified ones, require self-consistency on the extracted
    answer, and keep the shortest verified output. teacher_generate(prompt, n) -> list[str];
    verify(prompt, out) -> (ok, answer_key). Returns (rows, stats)."""
    rows, stats, seen = [], Counter(), set()
    for p in prompts:
        outs = teacher_generate(p, n)
        good = [(o, k) for o, ok, k in ((o, *verify(p, o)) for o in outs) if ok]
        stats["sampled"] += len(outs); stats["verified"] += len(good)
        if not good:
            stats["prompt_no_verified"] += 1; continue
        key, votes = Counter(k for _, k in good).most_common(1)[0]
        if votes < min_agree:
            stats["prompt_low_agreement"] += 1; continue
        best = min((o for o, k in good if k == key), key=len)
        h = hashlib.sha256(best.encode()).hexdigest()
        if dedupe and h in seen:
            stats["duplicate"] += 1; continue
        seen.add(h)
        rows.append({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": best}]})
    stats["kept"] = len(rows)
    return rows, dict(stats)

MCA = re.compile(r"\b\d{1,2}-\d{1,2}-\d{3,4}\b")

def verify_sections(prompt, out):
    """Valid iff JSON {"sections": [...]}, non-empty, well-formed MCA cites, and every cite appears in the source."""
    try:
        secs = json.loads(out)["sections"]
    except Exception:
        return False, None
    ok = bool(secs) and all(MCA.fullmatch(s) and s in prompt for s in secs)
    return ok, tuple(sorted(secs))

canned = {  # stand-in for teacher API calls at temperature > 0
    "Bill A amends 7-6-4005 and 7-6-4006.": ['{"sections": ["7-6-4005", "7-6-4006"]}', '{"sections": ["7-6-4006", "7-6-4005"]}',
                                           '{"sections": ["7-6-4005"]}', 'Sections: 7-6-4005, 7-6-4006'],
    "Bill B amends 15-30-2101.": ['{"sections": ["15-30-2101"]}', '{"sections": ["15-30-2101"]}',
                                  '{"sections": ["15-30-2102"]}', '{"sections": ["15-30-2101"]}'],
    "Bill C amends 2-2-102.": ['{"sections": ["2-2-102"]}', '{"sections": ["2-2-104"]}', 'n/a', '{"sections": []}'],
}
rows, stats = build_distillation_set(list(canned), lambda p, n: canned[p][:n], verify_sections)
print(stats)
print([r["messages"][1]["content"] for r in rows])
assert stats == {"sampled": 12, "verified": 7, "prompt_low_agreement": 1, "kept": 2}
```

How each filter behaves:
- The teacher's **hallucinated section** (15-30-2102, absent from the bill) fails verification.
- **Bill C** has only one verified answer, below the agreement threshold, so it is *dropped*, not trained on. Low teacher agreement signals a hard or ambiguous item, and a student imitating one noisy sample learns noise. Route these to human labelling instead.
- The **free-text answer** fails the schema check even though it is correct. The student learns the format you serve.

**Rationale and reasoning-trace distillation.** Train the student on the teacher's *explanation plus answer* (Distilling Step-by-Step, Hsieh et al.), or on full reasoning traces (DeepSeek-R1 distilled into Qwen and Llama students via SFT on ~800k curated samples). This transfers reasoning far better than answer-only data. The costs are longer training sequences and longer student outputs at inference (TTFT/ITL, 08.02). Measure accuracy **per output token**, not just accuracy.

**Capacity gap.** A much larger teacher does not always produce a better student. Very large gaps can hurt (Mirzadeh et al., *teacher assistants*). Try an intermediate teacher, or on-policy KD, when a 70B → 1B jump underperforms 70B → 8B → 1B.

---

## 5. Evaluating a Student, and Deciding to Ship It

| Question | Measurement |
|---|---|
| Is it good enough? | Task eval vs the **teacher** and vs the **prompted student base**, paired, per slice with CIs (06.01); non-inferiority margin agreed with owners |
| Where does it fail? | Slice report: long inputs, rare bill types, numbers-heavy, multi-hop. Distilled students lose the **long tail** first |
| Does it know when it doesn't know? | Calibration (ECE), abstention rate on unanswerable items; students are often over-confident |
| Is it faithful to the teacher? | Agreement rate on held-out prompts; on disagreements, which is right (judge + human spot-check) |
| Is it worth it? | Cost per successful task and p95 latency vs teacher (08.02 `cost_per_success`); break-even volume (08.01 §2) |
| Safety intact? | 07.x suites. Distillation data rarely includes refusals, so add them explicitly |

**Deployment patterns:**
- **Replace** the teacher when the student passes all slices.
- **Cascade** (08.01 §3.2): the student answers, and a confidence or verifier check escalates hard items to the teacher. This often captures 70–90% of the savings with **no** quality loss, and the escalation rate becomes a monitored SLI.
- **Keep improving:** escalated items plus teacher answers become the next distillation round.

---

## 6. Production Challenges and Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **Imitating teacher errors** | Student repeats teacher hallucinations confidently | Verification filters, self-consistency, drop low-agreement items (§4) |
| **Exposure bias** | Good on short outputs, derails on long ones | On-policy KD (§3); reverse-leaning `beta` |
| **Mode-averaging** | Bland or blended outputs that match no valid answer | Reverse KL / JSD, or sequence-level data with one canonical answer per prompt |
| **Tokenizer mismatch** | Can't do logit KD across families | Sequence-level KD; same-family teacher; cross-tokenizer KD with care |
| **Logit storage blow-up** | TBs of teacher logits | Top-k sparse logits (§2.3), or compute teacher logits online |
| **Long-tail collapse** | Rare slices far below teacher | Oversample rare slices in prompts; cascade to teacher; per-slice gates |
| **Distribution shift over time** | Student degrades as bills and styles change | Periodic re-distillation from escalations and new prompts; drift canary (08.05 §5) |
| **Terms/licence violation** | Legal exposure, forced takedown | Check teacher terms/licence first; record provenance in the data manifest (§7) |
| **Reasoning-length blowup** | Distilled reasoning student is slow | Budget reasoning tokens; train on the shortest correct traces; measure accuracy per token |

---

## 7. Governance: Terms, Licences, Provenance

- **API teachers.** Many providers' terms restrict using outputs to develop models that compete with the provider. Some enterprise agreements allow internal-use distillation. Read the **current** terms and your contract, get legal sign-off, and keep a record.
- **Open-weights teachers.** Licences (e.g. community licences for Llama-family models) can impose naming, attribution, acceptable-use, or scale-based conditions on derivatives and on models trained with their outputs. Record the licence in the AI-BOM (07.02 §2) and the model card (08.05 §9).
- **Data provenance manifest:**
  - teacher model and version;
  - prompt sources, including any PII scrubbing;
  - generation parameters;
  - verification rules and pass rates;
  - dedupe and decontamination reports;
  - dataset hash.

  This makes the student reproducible and auditable.
- **Public-sector context.** Student outputs that inform decisions about people fall under the same human-review and disclosure controls as the teacher's (08.05 §9, HB 178).

---

## 8. Hands-On Projects

### Project 1 — Distil a Frontier Model's Bill-Summary Behaviour into an 8B Student

**User stories**
- *As the platform owner*, I want an 8B self-hosted student to replace (or front) a frontier API for bill summaries at ≥ 10× lower cost, with no measurable quality loss on any slice.

**Acceptance criteria**
- Teacher terms and licences are reviewed and recorded. Prompts come from ≥ 5,000 real bill versions across sessions, with rare slices oversampled.
- `build_distillation_set` with verifiers (schema; fiscal numbers match the fiscal-note table; cited sections exist; judge rubric ≥ 4/5) and self-consistency. The pass-rate report is broken down by slice.
- The student (LoRA or full SFT, 09.01/09.03) is non-inferior to the teacher (−2 pt margin, 95% CI) on every slice. Calibration and refusal suites pass.
- The cost and latency report shows $/1k summaries and p95 latency (08.02) plus the break-even volume (08.01).

**Step-by-step**
1. Assemble prompts and write the verifiers, unit-testing each verifier on known good and bad outputs.
2. Generate with n=4 at T=0.7 through the provider's batch API (08.02 §4.2), then filter and deduplicate.
3. Train the student and track learning curves by data fraction.
4. Evaluate against the teacher per slice, and analyse the disagreements.
5. Ship behind a cascade first (§5), then replace the teacher once the escalation rate is low and stable.

### Project 2 — Logit and On-Policy Distillation, Measured Against SFT-Only

**User stories**
- *As an ML engineer*, I want to know whether white-box and on-policy distillation beat plain SFT on teacher outputs for our long-form tasks.

**Acceptance criteria**
- Teacher and student are from the same family (e.g. Qwen3-32B → Qwen3-1.7B). The task is committee-hearing minutes summarisation, with long outputs.
- Four students trained on equal token budgets:
  - (a) sequence-level SFT;
  - (b) (a) + offline top-64 logit KD with $T$ ∈ {1, 2};
  - (c) (a) + on-policy `DistillationTrainer` with `beta` ∈ {0, 0.5, 1};
  - (d) (a) + GRPO with a judge reward.
- Report quality by **output length bucket** (to show exposure bias), diversity (distinct-n), faithfulness (claim-level groundedness, 07.01), and compute cost per variant.
- The written conclusion names a default recipe for future tasks.

**Step-by-step**
1. Generate the SFT data (Project 1 pipeline) and train (a).
2. Dump the teacher's top-64 log-probs on the SFT data (§2.3 storage math) and train (b) with `topk_kd` + CE.
3. Run on-policy KD with vLLM rollouts for each `beta`.
4. Run the GRPO variant, then evaluate everything by length bucket.
5. Write up, including the forward/reverse KL behaviour you observe.

### Project 3 — Distil an Embedding Model or Reranker for the RAG Stack

**User stories**
- *As the search owner*, I want a small, fast reranker (and optionally an embedding model) that matches the large cross-encoder's ranking quality on legislative search, so reranking fits the latency budget (04.03, 08.02).

**Acceptance criteria**
- Teacher: a large cross-encoder or LLM reranker scoring (query, passage) pairs from real query logs (≥ 50k pairs, including hard negatives from hybrid search, 04.02).
- Student: a small cross-encoder trained with **margin-MSE** or listwise KL on teacher scores. Optionally, a bi-encoder embedder distilled with the same scores (03.01 contrastive + KD).
- nDCG@10 is within 1 point of the teacher on the held-out query set. Reranking p95 latency is ≤ 30% of the teacher's at the production candidate count.
- Deployment through the release pipeline (08.05), with an index rebuild if the embedder changes (the `plan_release` rule).

**Step-by-step**
1. Mine queries and candidates from logs, then score them with the teacher (batch).
2. Train the student with margin-MSE: $\big((s^+ - s^-) - (t^+ - t^-)\big)^2$. Tune it against a listwise KL variant.
3. Evaluate nDCG/MRR per query type (bill number, topic, statute), and measure latency on the target hardware.
4. Roll out through canary with online metrics (06.03: click-through, reformulation rate).
5. Document the results and schedule periodic re-distillation.

---

## 9. Foundational Papers (exact titles)

- Hinton, Vinyals & Dean, 2015 — *Distilling the Knowledge in a Neural Network*
- Kim & Rush, 2016 — *Sequence-Level Knowledge Distillation*
- Sanh et al., 2019 — *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*
- Jiao et al., 2019 — *TinyBERT: Distilling BERT for Natural Language Understanding*
- Wang et al., 2020 — *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*
- Mirzadeh et al., 2019 — *Improved Knowledge Distillation via Teacher Assistant*
- Hofstätter et al., 2020 — *Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation* (margin-MSE)
- Agarwal et al., 2023 — *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (GKD)
- Gu et al., 2023 — *MiniLLM: Knowledge Distillation of Large Language Models*
- Hsieh et al., 2023 — *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes*
- Taori et al., 2023 — *Alpaca: A Strong, Replicable Instruction-Following Model* (self-instruct-style distillation; technical report)
- Mukherjee et al., 2023 — *Orca: Progressive Learning from Complex Explanation Traces of GPT-4*
- Boizard et al., 2024 — *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs*
- DeepSeek-AI, 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- Xu et al., 2024 — *A Survey on Knowledge Distillation of Large Language Models*

## 10. Essential Tooling

| Tool | Use |
|---|---|
| **TRL 1.13 `DistillationTrainer`** (+ `trl.experimental` GKD, MiniLLM, GOLD) | On-policy KD with generalised JSD, vLLM rollouts |
| **vLLM** (08.03), provider **Batch APIs** | Cheap bulk teacher generation; top-k log-prob dumps (`logprobs=k`) |
| **distilabel, Argilla** | Synthetic data pipelines with judges and human review |
| **sentence-transformers** | Embedding and cross-encoder distillation (margin-MSE, listwise) |
| **DistillKit, torchtune KD recipes** | Offline logit KD recipes |
| **Your 06.01 harness + 08.05 release pipeline** | Student vs teacher gates; cascade rollout |

**Image prompt (distillation regimes):** *"Four-lane diagram. Lane 1 'Sequence-level': large teacher model emits text samples → filter funnel (verify, vote, dedupe) → small student trained by SFT. Lane 2 'Logit KD': teacher and student side by side over the same token sequence, teacher's probability bars (top-k highlighted) as soft targets. Lane 3 'On-policy KD': student generates text, teacher scores each student token, arrows back to student. Lane 4 'Deploy': student in front with a cascade arrow escalating hard cases to the teacher. Clean flat infographic, teacher in navy, student in teal."*

**Image prompt (forward vs reverse KL):** *"Two panels with the same bimodal teacher distribution (grey). Left 'Forward KL (mode-covering)': a wide teal student curve spanning both peaks, shaded area between the peaks labelled 'mass where teacher ≈ 0'. Right 'Reverse KL (mode-seeking)': a narrow teal curve sitting exactly on one peak, other peak labelled 'dropped mode'. Minimal technical plot style."*
