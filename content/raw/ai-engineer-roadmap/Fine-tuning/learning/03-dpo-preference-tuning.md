# 03 — DPO & Preference Tuning

> **Module goal:** Understand why SFT alone isn't enough for aligned, high-quality LLM behavior — and how modern preference-based methods (DPO, ORPO, KTO) close that gap without the operational nightmare of full RLHF. Be able to explain when you'd use DPO in a real project and what its dataset looks like.

---

## 1. Executive Summary & Core Concepts

**SFT teaches the model to imitate examples.** But "imitation" has a fundamental limit: the model learns to produce responses *like* the ones it saw, without any signal about what makes a response *better than another*.

Concrete symptom: after SFT on high-quality legal explanations, your model produces reasonable-looking legal explanations. But when a user asks something adversarial or edge-case, the model may still produce something *plausible-looking-but-wrong* — because it never learned to distinguish "correct" from "confidently incorrect."

**Preference tuning** solves this by training on **pairs of responses ranked by preference**. Given the same prompt:
- A "chosen" response (better)
- A "rejected" response (worse)

The model learns to prefer chosen-shaped outputs and avoid rejected-shaped outputs.

Three modern methods:

| Method | Data shape | Key idea | Complexity |
|--------|-----------|----------|-----------|
| **RLHF (PPO)** | Preference pairs → train reward model → PPO against it | Full reinforcement learning loop | High — 3 models in memory, brittle, expensive |
| **DPO** | Preference pairs directly | Reformulate the RLHF objective as a supervised classification loss | Low — one training script, similar cost to SFT |
| **ORPO** | Preference pairs, no reference model | Combine SFT and DPO into a single loss | Very low — one epoch on the preference data |
| **KTO** | Just "good" / "bad" labels (no pairs) | Kahneman-Tversky prospect theory loss | Low — often easier to collect labels |

**Direct Preference Optimization (DPO)** — the current default. It bypasses training a reward model entirely: through a clever derivation, the preference-optimization objective can be expressed as a supervised classification loss over the (prompt, chosen, rejected) triples. Same training loop as SFT, one extra term.

The DPO objective:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

where $\pi_\theta$ is the model being trained, $\pi_{\text{ref}}$ is the frozen reference (usually the SFT checkpoint), $y_w$ is the "chosen" (winning) response, $y_l$ is the "loser," and $\beta$ controls how strongly to enforce the preference.

Intuitively: **push the log-probability of chosen up relative to reference; push rejected down; balance so we don't wander too far from the reference.**

**The recipe most teams use in 2026:**
1. Start from an Instruct base model (already SFT'd by the vendor)
2. Do task-specific SFT with LoRA (Modules 01–02)
3. Collect a small preference dataset (500–5000 pairs)
4. Do DPO (or ORPO) on top of your SFT

Step 3 is the hard part — see Module 04 on data curation.

---

## 2. Deep-Dive Breakdown

### 2.1 Why RLHF Was Painful (and Why DPO Is a Big Deal)

Original RLHF loop:
1. Train a **reward model** ($r_\phi$) on preference pairs — a small transformer head predicting "which response is better."
2. Load the SFT model as the **policy** ($\pi_\theta$).
3. Load a frozen **reference** ($\pi_{\text{ref}}$) — usually the SFT model at start.
4. Run **PPO**: policy samples responses, reward model scores them, KL-penalty vs. reference keeps policy close to $\pi_{\text{ref}}$.

Problems:
- **Three models in memory simultaneously** — policy, reference, reward model — VRAM murder.
- **PPO is unstable** on language models. Many failed runs, hyperparameter voodoo.
- **Reward hacking** — the policy finds ways to score high on the reward model that a human would not endorse (e.g., producing overly verbose responses if length correlates with reward).
- Requires an ML team with deep RL expertise.

DPO's insight: **the optimal policy for the RLHF objective has a closed-form relationship to the reward function.** Rewriting the RLHF loss in terms of that relationship eliminates the reward model. You're left with a supervised loss on preference pairs.

Result:
- **Two models in memory** (policy + reference; reference can even be QLoRA-quantized).
- **Stable training** — behaves like standard supervised learning.
- **No reward hacking** in the RLHF sense (though DPO has its own failure modes — see below).
- **Fits alongside SFT tooling** — TRL's `DPOTrainer` is a drop-in.

DPO is now the default in most open-source alignment recipes (Zephyr, Tulu, StarCoder2-Instruct, dozens more). Frontier labs still use PPO/RLHF for last-mile alignment, but for applied engineering, DPO is what you'll use.

### 2.2 The Data — Preference Pairs

A DPO dataset row looks like:

```json
{
  "prompt": "Explain how contract law handles breach of duty in three sentences.",
  "chosen": "In contract law, a breach of duty occurs when a party fails to perform an obligation established by the contract. The non-breaching party may seek remedies including damages, specific performance, or rescission. Whether relief is available depends on the type and materiality of the breach, and on any limitations agreed to in the contract itself.",
  "rejected": "Breach of duty is when someone doesn't do what they said in the contract. You can sue them. It depends on stuff."
}
```

Sources for preference data:

**1. Human annotators.** Highest quality. Expensive (~$0.50–$5 per pair depending on expertise). For high-stakes domains, this is where the money goes.

**2. Model-graded pairs.** Use a stronger model (Claude Opus, GPT-4) to compare pairs sampled from your SFT model. Cheap, moderately reliable. Prone to LLM-judge biases (Module 07 of Month 6 covers).

**3. Rules-based pairs.** For structured tasks: generate a "correct" response by rule; corrupt it deterministically to make the "rejected." Works for JSON schema adherence, function-call formatting, refusal patterns.

**4. Public preference datasets.** UltraFeedback, HH-RLHF, Nectar. Good for general-purpose alignment; not for your specific task.

**Dataset size:** DPO typically needs less data than SFT. **500–5000 high-quality pairs** is a solid range. Beyond ~10k pairs, you're usually not gaining much unless the task is exceptionally broad.

### 2.3 The Beta Hyperparameter

$\beta$ (beta) controls how strongly DPO enforces the preferences vs. how much it stays close to the reference model.

- **Low $\beta$** (0.01–0.05): stays close to reference; smaller behavioral change; safer but less alignment.
- **Standard $\beta$** (0.1–0.3): sweet spot for most applications.
- **High $\beta$** (0.5–1.0): aggressive preference optimization; risk of over-correction, mode collapse, refusal drift.

$\beta = 0.1$ is TRL's default and a good starting point. Sweep it if you're not seeing preference improvement (may need higher) or if the model is becoming brittle (may need lower).

### 2.4 The Reference Model Choice

The reference model $\pi_{\text{ref}}$ anchors the policy — DPO's KL-like regularization pulls the trained model toward it.

Typical choices:
- **Same as init:** reference = SFT checkpoint you start DPO from. Most common; measures "did we improve over SFT?"
- **A stronger reference:** rarely done; can help if you want stability more than change.

Practical detail: the reference model doesn't need gradients — you can quantize it (4-bit) and save VRAM. TRL supports `ref_model` as a separate loaded 4-bit model, or `ref_model=None` (uses the pre-training state, which is more memory-efficient with PEFT).

### 2.5 ORPO — Skipping the Reference Model Entirely

**Odds Ratio Preference Optimization (ORPO, Hong et al., 2024)** collapses SFT and DPO into one loss:

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w \mid x) + \lambda \cdot \mathcal{L}_{\text{OR}}(y_w, y_l \mid x)
$$

The odds-ratio term penalizes the model for producing rejected-style outputs relative to chosen ones, without needing a reference model. Result:
- One training script (no reference model in memory)
- Combines SFT and preference tuning in one pass
- Often works with only 1 epoch on preference data
- Slightly less flexible than DPO's separate hyperparameters, but simpler

If you're VRAM-constrained or want the fastest path to a preference-tuned model, ORPO is a strong choice. In practice its results are comparable to DPO on most benchmarks.

### 2.6 KTO — When You Only Have Thumbs-Up / Thumbs-Down

DPO/ORPO require **paired** preferences: for the same prompt, you need both a chosen and a rejected response.

Real production feedback data is often not paired: users just thumbs-up or thumbs-down a response. You have a bunch of (prompt, response, label) triples where the label is a binary preference — not "this vs that."

**Kahneman-Tversky Optimization (KTO)** handles exactly this shape. It uses a loss inspired by prospect theory (humans weight losses ~2× as much as equivalent gains):

$$
\mathcal{L}_{\text{KTO}} = \begin{cases}
1 - \sigma(\beta \cdot \text{logratio}) & \text{if desirable (thumbs-up)} \\
1 - \sigma(-\beta \cdot \text{logratio}) & \text{if undesirable (thumbs-down)}
\end{cases}
$$

Data shape:
```json
{"prompt": "...", "completion": "...", "label": true}   // thumbs-up
{"prompt": "...", "completion": "...", "label": false}  // thumbs-down
```

Advantages:
- Works with your existing user-feedback data
- No need to construct pairs
- Handles imbalanced good/bad ratios gracefully

Use KTO when you have production user-feedback data; use DPO/ORPO when you're constructing pairs deliberately.

`[IMG-FT03-01]` — *Prompt: A comparison chart of three preference-tuning methods. Rows: RLHF (PPO), DPO, ORPO, KTO. Columns: "Data needed", "Models in memory", "Training stability", "Compute cost", "Typical use case". Fill in with concise labels. Highlight DPO as "modern default" with a subtle bookmark icon. Below the table, three loss-curve sparklines: PPO showing spiky unstable training, DPO showing smooth descent, ORPO showing similarly smooth descent with fewer steps. Clean data-table styling in a soft dark palette.*

---

## 3. Mental Models & Analogies

### Model 1: Feedback vs. Demonstration

Consider teaching a person to write emails.

**SFT is showing them 500 exemplar emails and saying "write like this."** They learn the shape, tone, and structure.

**DPO is showing them 500 pairs of emails — "this one is good, this one is bad, don't ask why yet" — and asking them to internalize the difference.** They learn *what makes one better than another*, not just what a good one looks like.

The difference matters when the person encounters a novel situation. The SFT-only learner produces something that *looks like* their training set but may miss why it's wrong. The DPO learner has internalized the ranking dimension — they can generalize better to "this feels like a good-vs-bad choice I've seen before" even if the specific case is new.

RLHF is the same idea with an intermediary reward-modeler in the middle who's summarizing the preferences into a grading rubric before teaching. DPO cuts out that intermediary because the math worked out.

### Model 2: The Diet vs. The Trainer

**SFT is the diet** — you eat what the good athletes eat and hope to become one.
**RLHF/DPO is the trainer** — someone watches you perform and specifically tells you which movements are better, why, and pushes you toward them.

You can become reasonably fit on diet alone. But diet-only produces athletes who look like they *should* be fit — they can't be corrected when their form is off, because they only ever saw good form, not the delta between good and bad.

A trainer's feedback ("this rep was worse than that one, because of X") is a fundamentally different signal from imitation. It's also lower-bandwidth per rep, but higher-value per rep. That's why preference tuning uses far fewer examples than SFT (500–5000 vs 5000–50000) to achieve substantial gains.

---

## 4. Common Pitfalls & Misconceptions

**Pitfall #1 — Doing DPO before SFT.**
DPO is a *refinement* step, not a replacement for SFT. On a base model that hasn't learned your task, DPO has nothing to work with — the "chosen" vs "rejected" signal degenerates into noise if the model can't produce either shape yet. Correct sequence: base → SFT → DPO. Skipping SFT and jumping to DPO on top of an Instruct model *can* work for very light-touch alignment adjustments but usually doesn't move the needle.

Corollary: your preference dataset should include responses at approximately the quality *level* your SFT model produces. Comparing "expert human-written" chosen vs "random-mush" rejected teaches the model very little that it wouldn't learn from SFT on the chosen alone. The valuable pairs are close-quality pairs where the preference is subtle.

**Pitfall #2 — Preference-collapse and refusal drift.**
DPO with high $\beta$ or too many steps can cause the model to become brittle: it produces the "chosen"-flavored response even when a different response would be appropriate. Common failure: after DPO training on "helpful" vs "refusal" pairs, the model starts refusing safe requests because the refusal shape is now strongly preferred. Symptoms:
- Refusal rate spikes on your eval set
- Response length collapses to a narrow band
- Model becomes strangely repetitive

Fixes: lower $\beta$; fewer epochs (1 is often enough); diversify your preference dataset so no single "type of preference" dominates. Watch a general-capability eval alongside your task metric.

**Pitfall #3 — Building preference pairs by asking the model to critique its own output.**
Tempting: sample two responses from your SFT model, ask the SFT model itself which is better, use that as ground-truth. This is close to "AI feedback" (RLAIF) and has published success stories — but at applied engineering scale, self-critique is usually worse than asking a *stronger* model to judge. Reasons:
- Self-critique inherits the model's own biases
- The critic has no better view of the ground truth than the generator
- You'll amplify whatever tendencies the SFT model already has

If you're using model-graded pairs, use a **stronger judge** — Claude Opus judging your Sonnet-tier SFT, GPT-5 judging your 7B open-model SFT, or human labels for high-stakes domains. Same rule as LLM-judge in eval (Month 6 Module 08).

---

## 5. Self-Assessment Bank

**Q1 (MC):** DPO's key mathematical insight is:
A) A new optimization algorithm faster than PPO
B) A closed-form relationship between the optimal policy and reward, allowing removal of the reward model
C) A better prompt template
D) A way to use RLHF without any preference data

**Q2 (short):** Given preference pairs $(x, y_w, y_l)$, write out (in plain English) what DPO's loss encourages the model to do.

**Q3 (MC):** In DPO, $\beta = 0.1$ vs $\beta = 0.5$ — which one enforces preferences more aggressively?
A) $\beta = 0.1$
B) $\beta = 0.5$
C) They're identical
D) $\beta$ doesn't control this

**Q4 (short):** Your model has been through SFT and works well. You have a small preference dataset from customer thumbs-up/down feedback. Which method should you use and why?

**Q5 (MC):** Compared to RLHF (PPO), DPO uses:
A) Same models in memory, faster training
B) One fewer model (no reward model) and more stable training
C) Two fewer models
D) More models

**Q6 (short):** Why is doing DPO *before* SFT usually a bad idea?

**Q7 (MC):** ORPO's key feature is:
A) Better quality than DPO
B) Combines SFT and preference tuning in one loss, no reference model needed
C) Uses reinforcement learning
D) Only works with 4-bit models

**Q8 (short):** After DPO training, your model refuses many safe requests it previously answered. Diagnose the likely cause and propose two fixes.

**Q9 (MC):** A DPO dataset size of ~2000 preference pairs is:
A) Way too small — you need at least 100k
B) A reasonable range for a focused task
C) Way too big — DPO needs only 10 or so
D) Doesn't matter, DPO is dataset-independent

**Q10 (short):** Explain the difference in *data shape* required by DPO vs KTO, and one production scenario where KTO's shape is more natural to obtain.

---

### Answer Key

**A1: B.** DPO's insight is that the RLHF objective (maximize reward subject to KL-constraint to reference) has a known closed-form solution — the optimal policy is proportional to the reference times $\exp(r/\beta)$. Solving for the reward and substituting into the preference likelihood produces a loss over preference pairs that doesn't require a reward model at all. The math is elegant; the operational win is enormous.

**A2:** DPO's loss encourages the model to increase the log-probability of the chosen response $y_w$ relative to the reference, while decreasing the log-probability of the rejected response $y_l$ relative to the reference. The margin between these two ratios is passed through a sigmoid, so DPO effectively frames the problem as "predict whether $y_w$ is preferred to $y_l$ using a Bradley-Terry-style model," but with the reward function replaced by a log-ratio of policy-to-reference probabilities. Net effect: push chosen up, push rejected down, don't wander too far from the reference (that's what $\beta$ and the reference log-ratios enforce).

**A3: B.** Higher $\beta$ = more aggressive enforcement of preferences (larger reward differential per pair). Lower $\beta$ = model stays closer to reference. Default $\beta = 0.1$ is a moderate setting. Going to $\beta = 0.5$ typically produces stronger preference following but risks over-fitting to the pairs' idiosyncrasies (mode collapse, refusal drift). Sweep $\beta \in \{0.05, 0.1, 0.2, 0.5\}$ and pick based on task metric vs general-cap eval trade-off.

**A4:** **KTO** (Kahneman-Tversky Optimization). Customer thumbs data is *unpaired* — each row is one prompt + one response + one label. DPO requires paired chosen/rejected for the same prompt, which thumbs data doesn't give you (you'd have to synthesize pairs by matching same-prompt responses across users, which usually doesn't exist). KTO natively accepts the label-per-response format. It also handles imbalance (typically far more thumbs-up than thumbs-down) more gracefully than trying to force pairs.

**A5: B.** DPO uses two models in memory (policy + frozen reference). RLHF/PPO uses three (policy + reference + reward model). DPO training is also more stable because it's a standard supervised loss rather than an RL loop, which reduces sensitivity to hyperparameters and rollout-buffer choices.

**A6:** DPO refines preferences *between* possible responses the model can produce. If the model can't yet produce either "chosen" or "rejected" shapes (i.e., pre-SFT), the preference signal has nothing to grip. DPO would try to push probabilities of shapes the model doesn't know how to produce — a weak signal that often just destabilizes the model. Correct order: SFT gives the model the *ability* to produce responses in the target style, then DPO teaches it *which* responses within that style are better. Skip SFT and you're asking DPO to do both jobs at once, which it's not designed for.

**A7: B.** ORPO combines the SFT loss (learn to produce chosen responses) and an odds-ratio preference loss (avoid rejected patterns) into one loss function trainable in a single pass. Because both signals share the same forward pass, no reference model is needed — the ratio is computed within the batch. Advantages: one training script, fewer models in memory, often one epoch is enough. Not necessarily higher quality than DPO, but simpler ops.

**A8:** Likely **refusal drift from preference collapse**: your preference dataset included refusal patterns as "chosen" (or "helpful" as "rejected") in some pairs, and DPO amplified the refusal-shaped response mode. Common when the preference data included safety-oriented pairs alongside quality pairs without careful balance. Fixes: (1) lower $\beta$ (say 0.05) so the model stays closer to reference behavior; (2) reduce epochs — DPO overfits with more steps than SFT; (3) audit the preference dataset — remove or rebalance pairs where "chosen" is refusal-shaped; (4) add explicit safe-response-preferred pairs to the dataset if you removed too many.

**A9: B.** 500–5000 pairs is the reasonable working range for DPO. Below that, you're often not getting signal beyond noise. Above ~10k, gains flatten unless the task is very broad. Compare to SFT which often benefits from 10k-100k+ examples — DPO is a *refinement* step and gains most of its value from a smaller, higher-quality set.

**A10:** **DPO data shape**: `(prompt, chosen, rejected)` — every row has two responses to the same prompt, one preferred over the other. Requires you to have or construct pairs.

**KTO data shape**: `(prompt, completion, label)` — every row has one prompt-response pair with a binary label (good or bad). No pair-construction required.

Natural KTO scenario: you have user thumbs-up/thumbs-down feedback from your production chatbot. Each user interaction is one prompt + one response + one label. You cannot easily construct DPO pairs from this because there's no canonical "same prompt" alternative response — but KTO handles this shape natively. This is the main reason KTO has gained ground in industry deployments: real feedback data is almost always unpaired.

---

**Related modules:**
- `learning/01-sft-fundamentals.md` — the necessary prerequisite step
- `learning/04-dataset-curation.md` — where preference datasets come from
- `learning/05-fine-tuning-ops.md` — running DPO in TRL
- `../production-ai/learning/08-model-evaluation.md` — evaluating preference-tuned models
- `../capstones/07-evaluate-ai-systems.md` — including preference-vs-base A/B methodology

**Practice prompts:**
1. Load a small SFT'd model + a preference dataset (UltraFeedback, or a subset). Run 200 steps of DPO with TRL's `DPOTrainer`. Compare a few generations before and after.
2. Take 20 responses from your SFT model. Ask a stronger model (Claude/GPT) to pairwise-judge them. Format into a DPO dataset.
3. Read the DPO paper's derivation (Rafailov et al., 2023) section 4.1. Verify you follow the substitution from the RLHF-optimal policy formula to the DPO loss.

**References:**
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023) — the DPO paper
- Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model" (2024)
- Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization" (2024)
- TRL docs: DPOTrainer, ORPOTrainer, KTOTrainer — https://huggingface.co/docs/trl
- Zephyr recipe (Tunstall et al.) — canonical DPO reference implementation
