# Probability & Statistics — Master Study Guide

> **Track:** Foundations · **Module:** 08
> **Prerequisites:** Basic algebra; comfort with sums, integrals, and Python.
> **Time budget:** ~25–35 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every ML model is (implicitly or explicitly) a **statistical estimator**. Every A/B test result is a **hypothesis test**. Every prediction has **uncertainty**, and if you don't quantify it, you'll overclaim and mislead. Every claim of "significance" hides assumptions that either hold and give you a real result — or don't and give you noise dressed up as a discovery.

Probability is the **language** of uncertainty; statistics is how we **use data** to reason under that uncertainty. Together, they underpin experimentation, modeling, forecasting, and decision-making. Learning them "properly" means:

1. Being fluent with **random variables, distributions, expectation, variance, and covariance**.
2. Understanding **estimators** and their properties (bias, variance, MSE, consistency).
3. Grasping **the two great pillars — the Law of Large Numbers and the Central Limit Theorem** — and what they let you get away with.
4. Reading and running **hypothesis tests** and **confidence intervals** honestly.
5. Knowing when to reach for **Bayesian** vs **frequentist** framing.

**Fundamental principles you must own:**

1. **Randomness is a *model*, not a property of the world.** We use random variables to represent our uncertainty about deterministic (or effectively deterministic) processes.
2. **All statistics is inference under assumptions.** Every test carries a null hypothesis and modeling assumptions; violating them invalidates the conclusion.
3. **A p-value is not the probability that the null is true.** It's the probability of the observed data (or more extreme) *given* the null. Confusing the two is the most common statistical error in the world.
4. **Distributions are functions, not "things".** A PMF/PDF is a function; expectation is an integral (or sum) against that function.
5. **The bootstrap is your friend.** Analytical CIs assume things; the bootstrap doesn't.

If you retain nothing else: **simulate before you formulate.** For any probability question, write 100 lines of NumPy that samples the process and estimates the answer. Then derive the analytical result and check.

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Random Variables, Distributions, Expectation, Variance

**A random variable (RV)** $X$ is a function from outcomes to real numbers. Discrete RVs have a **probability mass function (PMF)** $p(x) = P(X = x)$; continuous ones have a **probability density function (PDF)** $f(x)$ such that $P(a \leq X \leq b) = \int_a^b f(x)\,dx$.

**Expectation.**

- Discrete: $\displaystyle E[X] = \sum_x x \, p(x)$
- Continuous: $\displaystyle E[X] = \int_{-\infty}^{\infty} x f(x)\,dx$

**Linearity of expectation** (always holds, even without independence):

$$E[aX + bY + c] = a E[X] + b E[Y] + c$$

**Variance.**

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Standard deviation:** $\sigma_X = \sqrt{\text{Var}(X)}$ — same units as $X$.

**Covariance** measures joint linear variation:

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

**Correlation** normalizes to $[-1, 1]$:

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Independence.** $X, Y$ are independent iff their joint PMF/PDF factorizes: $p(x, y) = p_X(x) p_Y(y)$. Independence implies zero correlation; the converse is **not** true (correlation captures only linear relationships).

**Common distributions to memorize:**

| Distribution | Support | Mean | Variance | Use for |
|--------------|---------|------|----------|---------|
| Bernoulli($p$) | $\{0, 1\}$ | $p$ | $p(1-p)$ | Single coin flip |
| Binomial($n, p$) | $\{0, ..., n\}$ | $np$ | $np(1-p)$ | Sum of $n$ Bernoullis |
| Poisson($\lambda$) | $\{0, 1, ...\}$ | $\lambda$ | $\lambda$ | Rare events per interval |
| Geometric($p$) | $\{1, 2, ...\}$ | $1/p$ | $(1-p)/p^2$ | Trials until first success |
| Uniform($a, b$) | $[a, b]$ | $(a+b)/2$ | $(b-a)^2/12$ | Unbounded ignorance |
| Normal($\mu, \sigma^2$) | $\mathbb{R}$ | $\mu$ | $\sigma^2$ | Sums of many small effects |
| Exponential($\lambda$) | $[0, \infty)$ | $1/\lambda$ | $1/\lambda^2$ | Waiting times, memoryless |

**Normal PDF:**

$$f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Where **$\mu$** = mean, **$\sigma^2$** = variance.

---

### 2.2 Law of Large Numbers & Central Limit Theorem

**Law of Large Numbers (LLN).** Given i.i.d. samples $X_1, X_2, \ldots$ with $E[X_i] = \mu$, the sample mean converges to $\mu$ as $n \to \infty$:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \;\xrightarrow{\text{a.s./p.}}\; \mu$$

Practically: given enough data, your sample average is a good estimate of the true mean. This is the foundation of Monte Carlo and empirical estimation.

**Central Limit Theorem (CLT).** Given i.i.d. samples with mean $\mu$ and variance $\sigma^2 < \infty$, the standardized sample mean converges in distribution to a standard normal:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \;\xrightarrow{d}\; \mathcal{N}(0, 1)$$

Practically: the sampling distribution of $\bar{X}_n$ is approximately Normal for large $n$, regardless of the shape of the underlying distribution. This is why the Normal is *everywhere* — it emerges from any aggregation of many independent noisy contributions.

**Standard error of the mean:**

$$\text{SE}(\bar{X}_n) = \frac{\sigma}{\sqrt{n}}$$

If you don't know $\sigma$, replace with $s$ (sample standard deviation) — this is why the $t$-distribution appears when $n$ is small.

**Interpretation caveat.** CLT does not say the *underlying data* is Normal. It says the **sampling distribution of the mean** is approximately Normal. Many datasets are heavy-tailed or skewed; using Normal-based CIs on the raw data would be wrong, but on the *mean* of enough samples, it's fine.

---

### 2.3 Conditional Probability, Bayes' Rule, and Independence

**Conditional probability:**

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Chain rule:**

$$P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)$$

**Bayes' rule** rearranges the chain rule:

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

Named parts, for a hypothesis $H$ and evidence $E$:

$$\underbrace{P(H \mid E)}_{\text{posterior}} = \frac{\overbrace{P(E \mid H)}^{\text{likelihood}} \; \overbrace{P(H)}^{\text{prior}}}{\underbrace{P(E)}_{\text{evidence / marginal}}}$$

**Classic base-rate puzzle.** A disease affects 1 in 1000 people. A test has 99% sensitivity ($P(+ \mid \text{sick}) = 0.99$) and 99% specificity ($P(- \mid \text{healthy}) = 0.99$). You test positive. What's the probability you're actually sick?

$$P(\text{sick} \mid +) = \frac{P(+ \mid \text{sick}) P(\text{sick})}{P(+)}$$

$$P(+) = P(+ \mid \text{sick}) P(\text{sick}) + P(+ \mid \text{healthy}) P(\text{healthy}) = 0.99 \cdot 0.001 + 0.01 \cdot 0.999 \approx 0.011$$

$$P(\text{sick} \mid +) = \frac{0.99 \cdot 0.001}{0.011} \approx 0.09 \;=\; 9\%$$

That's the base-rate fallacy: a "99% accurate" test still leaves you 91% likely to be healthy after a positive result, because the prior is so low.

**Independence:** $P(A \cap B) = P(A) P(B) \iff P(A \mid B) = P(A)$.

---

### 2.4 Estimation, Hypothesis Testing, and Confidence Intervals

**An estimator** $\hat\theta$ is a function of the sample used to estimate a parameter $\theta$. Properties:

- **Bias:** $\text{Bias}(\hat\theta) = E[\hat\theta] - \theta$.
- **Variance:** $\text{Var}(\hat\theta)$.
- **Mean squared error:**

  $$\text{MSE}(\hat\theta) = \text{Bias}(\hat\theta)^2 + \text{Var}(\hat\theta)$$

  The **bias–variance decomposition** is the intellectual scaffolding of nearly all ML tradeoffs.

- **Consistent** if $\hat\theta \to \theta$ in probability as $n \to \infty$.
- **Unbiased** if $\text{Bias} = 0$ for all $\theta$.

**Hypothesis testing.** Set up:

- **Null hypothesis** $H_0$: the default / no-effect claim.
- **Alternative** $H_1$: the claim you want evidence for.
- **Test statistic** $T$: a function of the sample.
- **p-value**: $P(\text{observing } T \text{ as extreme as observed} \mid H_0)$.

**Decision rule.** Fix a significance level $\alpha$ (traditionally 0.05). If p-value < $\alpha$, reject $H_0$; otherwise, fail to reject.

**Error types:**

- **Type I** — reject a true $H_0$ (false positive); rate = $\alpha$.
- **Type II** — fail to reject a false $H_0$ (false negative); rate = $\beta$.
- **Power** = $1 - \beta$; the probability of catching a real effect.

**Common tests:**

- **One-sample t-test** — is $\mu = \mu_0$? Statistic:

  $$t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}}$$

  under $H_0$, $t \sim t_{n-1}$.

- **Two-sample t-test (Welch's)** — do two independent samples have equal means?

  $$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

- **Chi-square test of independence** — is there an association between two categorical variables?

  $$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

  Where $O_{ij}$ = observed, $E_{ij}$ = expected under independence.

- **Permutation test** — nonparametric; shuffle labels, recompute statistic, repeat, compare observed to shuffled distribution. Assumption-light and often the right choice.

**Confidence intervals.** A $(1 - \alpha) \cdot 100\%$ CI is an interval $[L, U]$ constructed so that if you repeated the experiment many times, $(1 - \alpha)$ of the intervals would contain the true parameter. For a Normal-approximated mean:

$$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

For $\alpha = 0.05$, $z_{0.025} \approx 1.96$.

**Critical caveat.** A specific CI *doesn't* contain the true value with probability 95% — either it does or it doesn't. The 95% is a property of the *procedure*, not any single interval.

---

### 2.5 Bootstrap, Bayesian vs Frequentist, and Practical Wisdom

**The bootstrap.** A simulation-based method to estimate the sampling distribution of *any* statistic without assuming a distribution family.

**Algorithm** (nonparametric bootstrap for a statistic $\hat\theta$):

1. Draw a sample $x_1, \ldots, x_n$ from your data.
2. Resample $n$ points **with replacement** from your original sample → a bootstrap sample.
3. Compute $\hat\theta$ on the bootstrap sample.
4. Repeat $B$ times (typically $B \in [1000, 10000]$).
5. The distribution of the $B$ values approximates the sampling distribution of $\hat\theta$.

**Percentile CI**: take the 2.5th and 97.5th percentiles of the bootstrap values.

```python
import numpy as np

def bootstrap_mean_ci(x, B=10_000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(x)
    n = len(x)
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = x[idx].mean()
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1 - alpha/2)])
    return means.mean(), lo, hi
```

**Bayesian vs Frequentist.**

- **Frequentist**: parameters are fixed unknowns; probabilities are long-run frequencies. Confidence intervals and p-values live here.
- **Bayesian**: parameters have probability distributions representing your beliefs; you combine a prior with data via Bayes' rule to get a posterior. Credible intervals live here.

A **95% credible interval** *does* directly mean "given the data and prior, there's a 95% probability the parameter is in this range." Frequentist CIs don't say that.

**When to use which:**

- Small samples with strong prior knowledge → Bayesian is often cleaner.
- Regulatory/traditional contexts (medical trials, most business A/B tests) → Frequentist.
- Sequential / online / decision-under-uncertainty → Bayesian shines.

**Practical wisdom (things every practitioner learns the hard way):**

1. **Report effect sizes, not just p-values.** A "significant" 0.1% improvement on a metric that costs $1M to ship may not be worth it.
2. **Peeking inflates false positives.** In A/B testing, checking every day and stopping when significant destroys your $\alpha$. Use sequential tests or fix your sample size in advance.
3. **Multiple comparisons** — running 100 tests at $\alpha = 0.05$ yields ~5 false positives. Correct with Bonferroni ($\alpha / m$) or Benjamini-Hochberg (FDR).
4. **Correlation is not causation.** Confounders, selection bias, and reverse causation all produce spurious correlation. Randomized experiments and causal inference (DAGs, IVs) are the tools.
5. **Beware Simpson's paradox** — an aggregated trend can reverse within subgroups.

---

## 3. Mental Models & Analogies

### 3.1 The "Two Bags of Marbles" Model (Sampling and Estimation)

Imagine a huge sealed bag (the **population**) containing marbles of unknown color mix. You can't count them all, but you can pull out a handful — that's your **sample**. Your job is to *guess* the population's composition from the sample.

- **The Law of Large Numbers** says: if you pull enough marbles, the sample's color mix will match the bag's.
- **The Central Limit Theorem** says: if you repeat the whole "pull a handful, compute the fraction of red" experiment many times, the fractions themselves form a bell curve around the true fraction, with spread $\sigma / \sqrt{n}$.
- **A confidence interval** is a rule that, applied to your one handful, produces an interval that (over many hypothetical repetitions) would trap the true fraction 95% of the time.
- **A p-value** answers: *if the bag really were 50–50, how surprising is my handful of 45 reds out of 50?* Small p-value = surprising = evidence against the 50–50 model.

The single most important consequence of this model: **your one sample is a snapshot; the reality you care about is the bag.** Any inference is a claim about the bag through the small window of the sample.

### 3.2 The "Weather Forecast" Model (Bayesian Updating)

A meteorologist walks into work with a **prior** belief about tomorrow's weather (climatology says 30% chance of rain in April). New **evidence** arrives (satellite shows a front approaching): the meteorologist updates to a **posterior** (now 70% rain).

$$P(\text{rain} \mid \text{front}) = \frac{P(\text{front} \mid \text{rain}) P(\text{rain})}{P(\text{front})}$$

- **Prior** = what you thought before the data.
- **Likelihood** = how the world would generate this data under each hypothesis.
- **Posterior** = what you think now.

The next day, more evidence arrives. Yesterday's posterior becomes today's prior. That's **sequential updating** — the natural way beliefs evolve. Frequentist procedures reset every time; Bayesian procedures accumulate.

![IMG-NP-01](/1%20—%20Foundations/images/IMG-PS-01.png)
> **Caption:** Left — Bayesian updating; right — frequentist inference from a sample.
> **Placement:** Section 3.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "A p-value Is the Probability the Null Is True"

It is not. A p-value is $P(\text{data at least this extreme} \mid H_0)$. The inverse — $P(H_0 \mid \text{data})$ — is a Bayesian quantity that requires a prior. Confusing the two is called the **prosecutor's fallacy** and is the single most common statistical error in courts, medicine, and business dashboards. A p-value of 0.03 does not mean "3% chance the null is true"; it means "if the null were true, we'd see data this weird 3% of the time."

### 4.2 "Correlation Implies Causation"

If ice-cream sales correlate with drowning deaths, ice cream doesn't cause drowning — summer causes both (confounder). Standard fixes:

- **Randomization** — the gold standard when feasible.
- **Control for confounders** with regression, matching, or stratification.
- **Instrumental variables** for observational settings.
- **Sensitivity analysis** to bound the effect of unobserved confounders.

Absent one of these, "significant correlation" is a hypothesis, not a conclusion.

### 4.3 "Larger Samples Fix Everything"

They fix random error but not systematic error. If your sampling frame is biased (only online shoppers, only respondents), collecting a million responses tightens your CI around **the biased estimate** — you become confidently wrong. Similarly, in ML, adding rows to a mislabeled dataset drives training error down but does not fix ground-truth error. Sample size is a lever for variance, not bias.

---

## 5. Self-Assessment Bank (Probability & Statistics)

### Questions

**Q1 (Short answer).** State the Central Limit Theorem in one sentence. What does it *not* say?

**Q2 (Multiple choice).** For i.i.d. samples $X_i$ with $\mu = E[X]$, $\sigma^2 = \text{Var}(X)$, and sample mean $\bar{X}_n$, what is $\text{Var}(\bar{X}_n)$?
A. $\sigma^2$
B. $\sigma^2 / n$
C. $n \sigma^2$
D. $\sigma^2 / \sqrt{n}$

**Q3 (Short answer).** Explain the base-rate fallacy in the context of a rare-disease diagnostic test.

**Q4 (Multiple choice).** A test rejects at $\alpha = 0.05$ level. Which is a correct statement?
A. If the null is true, there's a 5% probability we'd reject it.
B. There's a 95% probability that our conclusion is correct.
C. The p-value is guaranteed to be under 0.05.
D. There's a 5% chance the null hypothesis is true.

**Q5 (Short answer).** What are the four properties of a good estimator, and what is the bias–variance decomposition of MSE?

**Q6 (Multiple choice).** Zero correlation between $X$ and $Y$ implies:
A. $X$ and $Y$ are independent.
B. $X$ and $Y$ have no relationship.
C. $X$ and $Y$ have no linear relationship — nonlinear relationships are still possible.
D. $E[X] = E[Y]$.

**Q7 (Short answer).** Describe the nonparametric bootstrap algorithm and one situation where it's the right tool.

**Q8 (Multiple choice).** In a 95% confidence interval for the mean, which is TRUE?
A. There is a 95% probability the true mean is in this specific interval.
B. If we repeated the experiment many times and formed the interval each time, 95% of the intervals would contain the true mean.
C. 95% of the data lies inside the interval.
D. The p-value is 0.05.

**Q9 (Short answer).** You run 100 A/B tests at $\alpha = 0.05$ with no true effects. On average, how many will you falsely declare significant? Name one correction technique.

**Q10 (Multiple choice).** Which is a common cause of Simpson's paradox?
A. Randomization.
B. Confounding variables and imbalanced subgroups.
C. Large sample size.
D. A well-defined outcome metric.

---

### Answer Key & Detailed Explanations

**A1.** The CLT: for i.i.d. samples with finite variance, the sampling distribution of the standardized sample mean converges to $\mathcal{N}(0, 1)$ as $n \to \infty$. It does **not** say the raw data are Normal — only the mean of many samples is (approximately). Nor does it say anything about very heavy-tailed distributions with undefined variance.

**A2. B.** $\text{Var}(\bar{X}_n) = \text{Var}\!\left(\frac{1}{n}\sum X_i\right) = \frac{1}{n^2} \sum \text{Var}(X_i) = \frac{\sigma^2}{n}$. This is the reason $\text{SE}(\bar{X}) = \sigma / \sqrt{n}$.

**A3.** Even a "99% accurate" test can produce mostly-false-positive results when the prior probability is very low. If 1 in 1000 has the disease and specificity is 99%, then for every 1 sick person correctly flagged, ~10 healthy people are falsely flagged. So $P(\text{sick} \mid +) \approx 9\%$, not 99%. The prior — the base rate — is what governs the posterior when the test's specificity error dominates the true-positive count.

**A4. A.** $\alpha$ is the **Type I error rate** — the probability of rejecting when the null is true. B and D conflate p-values with posterior probabilities (see A3). C is trivially false.

**A5.** A good estimator ideally is **unbiased**, has low **variance**, is **consistent** (converges in probability to the true value as $n \to \infty$), and is **efficient** (achieves the lowest variance among unbiased estimators, i.e., the Cramér-Rao bound). The bias–variance decomposition:

$$\text{MSE}(\hat\theta) = \left(E[\hat\theta] - \theta\right)^2 + \text{Var}(\hat\theta) = \text{Bias}^2 + \text{Variance}$$

Minimizing MSE often means accepting a bit of bias to sharply cut variance (why regularization works).

**A6. C.** $\rho_{XY} = 0$ means no linear relationship, but $Y = X^2$ with $X$ symmetric around zero has zero correlation despite a perfect nonlinear relationship. Independence is a strictly stronger condition than zero correlation.

**A7.** Nonparametric bootstrap: from your original sample of size $n$, repeatedly draw resamples of size $n$ **with replacement**, compute your statistic on each, and use the distribution of resample statistics as an approximation to the true sampling distribution. It's the right tool when analytical distributions are unavailable or assumption-heavy — e.g., CIs for medians, quantiles, or complex model outputs.

**A8. B.** A CI is a property of the **procedure**: repeating the whole experiment many times, 95% of the constructed intervals would trap the true parameter. A specific realized interval either contains the truth or doesn't; the probability doesn't apply post-hoc to a single interval in the frequentist framework.

**A9.** On average, $100 \times 0.05 = 5$ tests will be falsely significant under the null. Corrections:
- **Bonferroni** — reject only if p < $\alpha / m$ (very conservative).
- **Benjamini-Hochberg** — controls the False Discovery Rate; less conservative and often preferred for many tests.

**A10. B.** Simpson's paradox arises when an aggregate trend reverses within subgroups because a confounder is distributed unevenly. Classic example: a university admits a higher fraction of women overall than men, but within every department admits a higher fraction of men — because women disproportionately applied to competitive departments.

---

## 6. Practice Prompts

1. Simulate 10,000 flips of a biased coin ($p = 0.55$). Plot the running mean and observe convergence to 0.55. This is LLN in your face.
2. Take a heavy-tailed distribution (log-normal). Sample $n = 30$ observations, compute the mean, repeat 10,000 times. Overlay a Normal PDF on the histogram. Increase $n$; watch CLT tighten.
3. Reproduce the disease-test posterior calculation from Section 2.3 for several priors ($10^{-1}, 10^{-3}, 10^{-6}$) and sensitivities/specificities.
4. Bootstrap a 95% CI for the median of a small ($n=25$) dataset. Compare to the theoretical CI (if you can even write one — good luck).
5. Design and run a two-sample permutation test to compare click-through rates of two ads with $n_1 = 500$, $n_2 = 500$.
6. Study Simpson's paradox with a small synthetic dataset (Berkeley admissions style). Confirm that stratifying reverses the effect.

---

## 7. References

- Larry Wasserman, *All of Statistics* (compact and rigorous).
- Bruce, Bruce & Gedeck, *Practical Statistics for Data Scientists* (applied).
- Efron & Tibshirani, *An Introduction to the Bootstrap* (canonical).
- Kruschke, *Doing Bayesian Data Analysis* (approachable Bayesian).
- 3Blue1Brown, *Bayes theorem* video (intuition).
- Blitzstein & Hwang, *Introduction to Probability* (free MOOC and text).
