# *Zero-shot* cloning as source coding: a rate-distortion theory of voice coverage

Working document (teoria2/A), 17 November 2026. CPU only, no training, no repository file modified.
Scripts and outputs in this folder (`*.py`, `out/*.json|csv`), all run with `python3 -I script.py <paths>`.

**Legend.** **[PROVEN]** follows from the hypotheses by a complete calculation; **[HEURISTIC]** calculation with an
uncontrolled approximation; **[CONJECTURE]** modeling hypothesis; **[SIMULATED]** checked numerically (checks the
calculation, not the hypothesis); **[DATA]** measured on the saved latents/results.

---

## 0. Summary

**Main formula.** Let $c$ be the code (whitened latent) of the reference, $v_0$ the default voice (LJ), $P_v$ the
distribution of training voices and $V\sim P_v*\mathcal N(0,I/K)$ a noisy *codeword*. Then

$$
\boxed{\;q(N,K,M)=\mathbb E_c\Big[\big(1-F_c(t(c))\big)^N\Big],\qquad
F_c(t)=\Pr\big(\|c-V\|^2\le t\big),\qquad t(c)=\|c-v_0\|^2-\tau,\quad \tau=2T\ln\tfrac{M_{\rm ef}}{K}\;}
$$

"not copied" $\iff$ the **nearest-neighbor distortion** in a random codebook of $N$ voices,
$D_N(c)=\min_i\|c-v_i\|^2$, exceeds the **tolerance** $t(c)$ that the default-voice attractor leaves to that voice.
Equivalently (information spectrum), $q(N)\approx\Pr_c\big[L(c)>\log_2N\big]$ with
$L(c)=-\log_2F_c(t(c))$ = **bits needed to "hit" voice $c$ with randomly drawn voices**.

**What was measured, fitted and predicted** (38 external voices, $K=32$):

| | $N=8$ | $N=32$ | $N=128$ | parameters taken from the curve |
|---|---|---|---|---|
| observed | 0.434 | 0.302 | 0.211 | — |
| Bayes decoder ($T=1$, $M=2000$ nominal) | 0.092 | 0.039 | 0.000 | 0 — **fails** |
| $\tau$ fitted at $N=128$, real codebooks, $T=1/2/3$ | 0.375/0.401/0.445 | 0.263/0.264/0.279 | (0.211) | 1 ($\tau$) |
| $\tau_0=\|\mu-v_0\|^2$ (identity without fitting, found *post hoc*), $T=2$ | 0.400 | 0.263 | 0.210 | 0 (*post hoc*) |
| same, each run in its **own** latent space | 0.360 | 0.219 | 0.210 | 0 (*post hoc*) |
| $\tau$ from $N=128$, **randomly drawn** codebooks (mean) | 0.32 ± 0.05 | 0.25 ± 0.02 | (0.211) | 1 |
| Gaussian population (measured spectrum), real refs. | 0.353 | 0.274 | 0.253 | 1 |

The one-parameter model, calibrated at one point, predicts the other two with an error of 0.03–0.06 (or 0.00–0.04 with
$T=3$), in the direction of **underestimating the dependence on $N$** when $T=1$. The fully parametric version (Gaussian
voices) errs more (0.08) because the real voices are more clustered than the Gaussian: real $D_N$ falls faster
(median 39.2→19.5 versus 39.6→23.5). Per voice, the predictor ranks the 38 voices with Spearman 0.50–0.58 at each $N$.
**Failures:** the 4 new voices of the grid are harder than the model says (bias −0.19 to −0.24 over 14 training runs);
the effect of $K$ on the trained voices is larger than the $\ln K$ term predicts; the exponent is not identified.

**New theoretical results:**
1. With spectrum $\lambda_j\propto j^{-a}$, the optimal codebook gives $D(R)\propto R^{-(a-1)}$, that is, **a power of
   $\log N$**, with local exponent $\beta_D(N)\approx(a-1)/\ln N$; the random codebook pays **2–4 bits** more
   ($N$ random voices are worth $\approx N/10$ optimal ones); high resolution ($N^{-2/d}$) is not reached up to $N=4096$ [SIMULATED].
2. **Tauberian theorem**: if the fraction of voices with ball probability $p_c\le x$ grows as $x^{\beta}$,
   then $q(N)-q_\infty\sim C\,\Gamma(1+\beta)N^{-\beta}$; with voices near the boundary of the attractor and local
   dimension $d$, $\beta=2/d$. With $d\approx7$–8 (measured) this gives 0.25–0.29, the observed "0.26" [PROVEN given the model; SIMULATED].
3. **Averaging references does not help** follows from the structure: the reference noise $\|\xi\|^2$ cancels exactly between
   $\|c-v_0\|^2$ and $D_N(c)$; only the projection of the noise on the axis $v_{\rm nn}-v_0$ remains [PROVEN].
4. **Floor**: with the current attractor, $q_\infty\approx0.10$ (90% resampling CI 0.05–0.18): voices whose reference
   lies less than $\sqrt\tau$ from the default voice are not copied at any $N$. Prediction: $q(2456)\approx0.15$
   (0.10–0.22), **90% copying is unreachable with more speakers alone**. Without a dominant voice (two-stage voice,
   $v_0\to\mu$), the same calculation gives $q=0.45/0.11/0.00$ at $N=8/32/128$.
5. **Bits**: one utterance carries $I(v;c)\approx10.3$ bits of voice; copying the median voice requires $L\approx2$ bits,
   75% of the voices 7.5 bits, 80% 10 bits ($\approx2^{10}$ randomly drawn speakers), 85% 16 bits — more than what one
   reference carries. The saturation scale is $N^\star\approx2^{I(v;c)}\approx10^3$ speakers.

---

## 1. Setup

### 1.1 Space and measured quantities [DATA, `a1_geometria.py` → `out/a1.json`]

Common space: latent $\mu$ (32 dim.) of the $N=128$ training run (`n128_k32_s1-wide`), whitened by the covariance of the
channel that the decoder sees in training, $W_c=W+\mathrm{diag}\,\mathbb E\sigma^2$ ($W$: within-speaker, LibriTTS). In these
units, a sampled code has noise $I$ around the speaker mean. The codebooks of $N=8$ and $N=32$ are
**subsets** of the 128 (checked in the `train_manifest.csv` files), so the three training runs have trained voices
located in the same space.

| Quantity | Value | Source |
|---|---|---|
| eigenvalues of $\Sigma_b$ (between speakers, noise-corrected) | 8.3; 5.8; 5.2; 3.1; 2.2; 2.0; 1.1; 1.0; 0.85; 0.76; 0.44; … | variance components |
| $\mathrm{tr}\,\Sigma_b$; participation ratio | 32.6; 7.0 | |
| exponent $a$ (first 8 / 16) | 1.03 / 1.55 | |
| noise of one reference $\mathrm{tr}\,W_x$ ($\mu$ of one utterance) | 12.0 | |
| $\|v_0-\mu\|$ (LJ to the LibriTTS center) | 10.8 (5.0 Mahalanobis standard deviations along the 8 directions) | |
| typical distance between two voices $\sqrt{2\,\mathrm{tr}\Sigma_b}$ | 8.1 | |
| $\|c-v_0\|$ of the 38 references | 12.7 ± 1.8 | |
| $I(v;c)=\tfrac12\sum_j\log_2(1+\lambda_j)$ | 10.3 bits (8 directions: 8.0 bits) | `a5`, `a6` |

The value of 10.3 bits reproduces the 9.8–10.2 bits "in $z$" of `03_information.md` (same family of estimator; it is not
independent confirmation). The default voice is an atypical voice, 5 standard deviations away from the population.

### 1.2 The decoder as the test channel of a codebook

**Hypothesis H (Gibbs decoder)** [CONJECTURE]. The voice rendered for the code $c$ is the mean of the training voices
weighted by
$$
\pi_s(c)=\frac{n_s\,e^{-\|c-v_s\|^2/2T}}{\sum_r n_r\,e^{-\|c-v_r\|^2/2T}},\qquad n_s=K\ (\text{LibriTTS}),\ n_0=M\ (\text{LJ}).
$$
This is exactly the form of the Blahut–Arimoto step (the optimal rate-distortion test channel restricted to a
codebook with marginal $n_s/\sum n$, multiplier $\beta=1/T$; *deterministic annealing*, Rose 1998): **the
training masses play the role of the output marginal, and the default-voice attractor is simply the codeword with the
largest mass**. Copying is decided by Lemma 1 of `02_derivation.md` ($m=(\tfrac12-\rho)\|\Delta\|^2$), applied in
latent space: $\rho=1-\langle\hat y-v_0,c-v_0\rangle/\|c-v_0\|^2$, $\Pr(\text{copy}\mid c)=\Phi(\kappa(\tfrac12-\rho))$,
$\kappa=\|\Delta\|^2/\sigma_w=0.65/0.036$ ($\sigma_w$ measured; $\|\Delta\|^2$ is the value assumed in 02; $\kappa$ only
smooths an almost 0/1 result).

**Proposition 1 (Bayes)** [PROVEN]. If the training codes of speaker $s$ are $c=v_s+\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$
(whitened), and the decoder minimizes the squared error of the rendered voice over the training mixture, then
$\hat y(c)=\mathbb E[v_S\mid c]=\sum_s\pi_s(c)v_s$ with $T=1$ and $n_s$ = number of utterances. *Proof:* Bayes' rule,
$\Pr(S=s\mid c)\propto n_s\,\mathcal N(c;v_s,I)$. $\square$

**Test** [DATA, `a2_gibbs_dados.py`]: with $T=1$, $M=2000$, $K=32$, the Bayes decoder predicts
$q=0.09/0.04/0.00$ versus $0.43/0.30/0.21$. The likelihood ratio of LJ against the trained neighbor is
~70 nats, versus $\ln(M/K)=4.1$: **the real decoder is far from the Bayes optimum; it pulls toward the default voice
with a strength equivalent to $\tau/2\approx57$ nats**. The nominal mass of the 2,000 utterances does not explain the attractor; the
natural candidate is the starting *checkpoint* (a single speaker, many epochs), which is not modeled.

**Proposition 2 (Poisson limit)** [PROVEN]. With $\tau\equiv2T\ln(M/K)$ fixed and $T\to0$,
$\pi_0(c)\to\mathbf 1\big[\|c-v_0\|^2-\tau<D_N(c)\big]$, $D_N(c)=\min_i\|c-v_i\|^2$. *Proof:* the log of the ratio
$\pi_0/\pi_{\rm nn}$ is $\big(D_N(c)-\|c-v_0\|^2+\tau\big)/2T$; the remaining terms are of order $e^{-\Delta/2T}$. $\square$
For i.i.d. codebooks from $P_v$, $\Pr(D_N>t\mid c)=(1-F_c(t))^N$ exactly, which gives the formula of the summary.
For Gaussian $P_v$, $F_c$ is a generalized noncentral chi-squared (Imhof; Lugannani–Rice saddle point for the
tail; `rd_theory.py`).

**Verification** [SIMULATED, `t2_verifica_formula.py` → `out/t2.json`]: calibrated Gaussian geometry, 600 voices,
4 codebooks per $N$:

| case | $N$ | formula | information spectrum $\Pr(L>\log_2N)$ | simulated Gibbs |
|---|---|---|---|---|
| $T=1$, $K=32$, $\tau=116.5$ | 8 / 32 / 128 / 512 | 0.514 / 0.424 / 0.363 / 0.322 | 0.542 / 0.445 / 0.373 / 0.332 | 0.505 / 0.412 / 0.353 / 0.310 |
| $T=1$, $K=8$ | 8 / 128 / 512 | 0.562 / 0.413 / 0.366 | 0.593 / 0.437 / 0.377 | 0.545 / 0.406 / 0.347 |
| $T=2$, $\tau=116.5$ | 8 / 128 / 512 | 0.514 / 0.363 / 0.322 | — | 0.490 / 0.321 / 0.261 |
| Bayes, $M=2000$ ($\tau=8.3$) | 8 | 0.002 | 0.002 | 0.004 |

The Poisson formula matches the Gibbs channel to ~0.01 with $T=1$; with $T=2$ it overestimates $q$ at large $N$
(0.06 at $N=512$), because the mass of several nearby *codewords* starts to add up (law-of-large-numbers regime).
The information-spectrum form is off by 0.01–0.03 (Gumbel smoothing of width ~1.4 bits).

---

## 2. Part 1 — $D(R)$, random codebook and nearest neighbor

### 2.1 Reverse water-filling with a power-law spectrum [HEURISTIC in the continuous approximation; the bound is PROVEN]

For a Gaussian source $\mathcal N(0,\mathrm{diag}\lambda_j)$, any codebook of $N$ points has
$\mathbb E\,d\ge D(\log_2N)$ (*one-shot* bound: $I(X;\hat X)\le H(\hat X)\le\log_2N$) [PROVEN]. With $\lambda_j=j^{-a}$,
the water level $\theta=J^{-a}$ activates $J$ directions and
$$
R=\tfrac a2\sum_{j\le J}\log_2\tfrac Jj\approx\frac{aJ}{2\ln2},\qquad
D=J\theta+\sum_{j>J}j^{-a}\approx\frac{a}{a-1}J^{1-a}\ \Rightarrow\
D(\log_2N)\propto\Big(\frac{2\ln N}{a}\Big)^{-(a-1)},\quad \beta_D(N)\equiv-\frac{d\ln D}{d\ln N}\approx\frac{a-1}{\ln N}.
$$
The optimal distortion falls as **a power of $\log N$**; seen as a function of $N$ over a short window, it looks like a power law
with a small exponent that decays as $1/\ln N$. For $a=1$ the fall is even slower ($D\approx\ln(d/J)+1$).

### 2.2 Numerical comparison [SIMULATED, `t1_rd_teoria.py` → `out/t1.json`; $d=32$, trace 32.6]

Total $D$ (sum over 32 dimensions) for the **measured** spectrum:

| $N$ | 8 | 32 | 128 | 1024 | 4096 |
|---|---|---|---|---|---|
| optimal (reverse water-filling) $D(\log_2N)$ | 17.7 | 13.3 | 10.5 | 7.7 | 6.3 |
| random, $N$ i.i.d. voices from $P_v$ (Monte Carlo) | 32.4 | 22.7 | 16.5 | 11.6 | 9.5 |
| random, saddle point (median) | 30.3 | 21.7 | 16.4 | 11.5 | 9.3 |
| high resolution $\Gamma(1+\tfrac2d)(NV_d)^{-2/d}\mathbb E f^{-2/d}$ | 4.7 | 4.3 | 3.9 | 3.5 | 3.2 |
| extra bits of the random codebook, $\log_2N-R_{\rm wf}(D_{\rm rand})$ | 3.0 | 3.5 | 3.5 | 3.9 | 4.1 |

Local exponents $-d\ln D/d\ln N$ between 8 and 128:

| spectrum | optimal | random | $2/d_{\rm eff}$ |
|---|---|---|---|
| $a=1$ | 0.10 | 0.15 | — |
| $a=1.5$ | 0.16 | 0.20 | — |
| $a=2$ | 0.23 | 0.28 | — |
| measured | 0.19 | 0.24 | — |

Interpretation: (i) the high-resolution formula is off by a factor of 4–7 over the whole range: **$N\le4096$ is far from the
$N^{-2/d}$ regime**; the relevant regime is the low-rate one, $R=\log_2N\approx3$–12 bits, in which only the first directions are
"coded"; (ii) a random codebook costs 3–4 bits (a factor of ~10 in $N$) more than the optimal one — the price of
training with randomly drawn speakers instead of chosen ones [SIMULATED]; (iii) the saddle point reproduces Monte Carlo to
~5% [SIMULATED]. Check of the small-ball calculation (4 points × 3 quantiles, 400,000 samples): Imhof and the saddle point
agree with Monte Carlo to within 1–7% relative down to $p=10^{-3}$; the Chernoff bound $e^{-I}$ is 3–6 times higher,
as expected (the Bahadur–Rao prefactor is missing).

### 2.3 Distribution of $D_N$ and of $q$

Given $c$: $\Pr(D_N>t)=(1-F_c(t))^N\approx\exp(-N\,2^{-L_c(t)})$, a Gumbel in $\log_2N$ centered at $L_c(t)$ with
a width of ~1.4 bits [PROVEN]. Over voices, $q(N)=\mathbb E_c\,e^{-Np_c}$, $p_c=F_c(t(c))$: **$q$ is the Laplace transform
of the distribution, across voices, of the ball probability $p_c$**.

**Theorem (Tauberian/Abelian)** [PROVEN]. Let $H(x)=\Pr_c(p_c\le x)$, $q_\infty=H(0)$. If
$H(x)-q_\infty\sim Cx^{\beta}$ as $x\to0$, then $q(N)-q_\infty=\int_0^1e^{-Nx}dH(x)\sim C\,\Gamma(1+\beta)\,N^{-\beta}$.
(Abelian direction of Karamata's theorem.) Hence **$q$ is a power of $N$ if and only if the fraction of "hard" voices
(small $p_c$) has a regularly varying tail at zero**.

**Corollary** [HEURISTIC]. In local high resolution, $p_c\approx f(c)V_d\,(t(c)-n_0)^{d/2}$ ($n_0$: irreducible
distortion). If the density of the tolerances $t(c)$ is positive at the boundary $n_0$ (voices "almost captured" by the
attractor), $H(x)-q_\infty\propto x^{2/d}$ and
$$
q(N)-q_\infty\ \propto\ N^{-2/d}.
$$
The exponent comes from the **voices near the boundary of the attractor** and from the **local dimension of the voice manifold**, not from the
global spectrum. With $d\approx7$–8 (Levina–Bickel and the coverage exponent of `03`), $2/d=0.25$–0.29. This differs from the
argument of 03 ("failure $\propto r^2$"): here the power comes from the tail of the distribution of hard voices.

Verification [SIMULATED, `t3_tauberiano.py`]: voices uniform on $[0,1]^d$ (torus), uniform tolerances with 8% of
voices captured:

| $d$ | slope of $q-q_\infty$ ($N=128$–8192) | predicted $-2/d$ | ratio of decrements $\frac{q_{32}-q_{128}}{q_8-q_{32}}$ | predicted $4^{-2/d}$ | apparent exponent 8–128 without floor |
|---|---|---|---|---|---|
| 4 | −0.503 | −0.500 | 0.49 | 0.50 | 0.36 |
| 8 | −0.250 | −0.250 | 0.76 | 0.71 | 0.20 |
| 12 | −0.175 | −0.167 | 0.75 | 0.79 | 0.15 |

In the data, the ratio of decrements is $0.091/0.132=0.69$, which corresponds to $d\approx7.4$; but the CI of this number
is huge (04: $\alpha_g$ from −0.7 to 1.6), so this is **compatibility, not a test**. Note that the three-point fit
with $d=8$ implies $q_\infty\approx0$, whereas the calibrated model (section 5) says $q_\infty\approx0.10$: the three points
do not separate floor from exponent.

---

## 3. Part 2 — the role of $K$ and $M$

From the Gibbs channel [PROVEN given H]:

1. **$M$ and $K$ enter as masses** and, in the Poisson limit, only through the combination $\tau=2T\ln(M/K)$: they shift the
   tolerance of all voices by the same amount. Doubling $K$ is equivalent to halving $M$ (as in 02).
2. **$K$ also enters as codeword noise**: the trained voice is known through the mean of $K$ utterances,
   $V=v_i+\mathcal N(0,I/K)$, which adds $\approx d/K$ to $D_N$ (32/K in 32 dimensions).
3. **Law-of-large-numbers regime** (large $T$ or large $N$): $\sum_ik(c,v_i)\approx N\,\mathbb E\,k$, and the
   decision depends on $\ln(NK/M)$; the shrinking of the failure region is logarithmic in $NK/M$.

$$
q(N,K,M)=\mathbb E_c\Big[\Big(1-F^{(K)}_c\big(\|c-v_0\|^2-2T\ln\tfrac{M}{K}\big)\Big)^N\Big],\qquad
F^{(K)}_c(t)=\Pr\big(\|c-V\|^2\le t\big),\ V\sim P_v*\mathcal N(0,I/K).
$$

**What the data say about the regime** [DATA, `a2_gibbs_dados.py`, profile with $q_{128}$ fixed]: for each $T$,
choosing the LJ mass that reproduces $q_{128}$, the prediction at $N=8$ is

| $T$ | 1.2 | 1.8 | 2.8 | 4.4 | 6.8 | 10 | 16 | ≥25 |
|---|---|---|---|---|---|---|---|---|
| predicted $q_8$ | 0.38 | 0.40 | 0.42 | 0.53 | 0.68 | 0.79 | 0.97 | 1.0 |
| predicted $q_{32}$ | 0.26 | 0.26 | 0.27 | 0.30 | 0.38 | 0.52 | 0.70 | ≥0.89 |

The mass regime ($T\gtrsim5$, in which $q$ depends on $\ln(NK/M)$) gives too strong a dependence on $N$. The
data lie in the **distortion regime** ($T\approx1$–3): what decides is how close the nearest trained voice is to the
requested voice, not how many voices there are around it. In this regime $T$ and $M_{\rm ef}$ are not separately
identifiable; only $\tau$ is.

**Consequence**: in the distortion regime, $K$ and $M$ weigh little on the new voices (they enter through $2T\ln$, a few
units versus a spread of margins of ~60). Predictions in `a5` (empirical codebook, section 5.3), $N=128$:
$K=8$: 0.26 ($T=1$) to 0.30 ($T=3$); $K=128$: 0.18 to 0.16; reference ($K=32$) 0.25 in the same calculation.

---

## 4. Part 3 — the threshold $\tau$ and the parameter count

### 4.1 What sets $\tau$

- By Proposition 1 (Bayes decoder), $\tau=2\ln(M/K)=8.3$: **refuted** (section 1.2).
- Fitted to the 38 voices at $N=128$: $\tau=114$ ($T=1$), 117 ($T=2$), 121 ($T=3$); hard rule: 113.4.
- **Empirical identity** [DATA, found after the fit]: $\|\mu-v_0\|^2=116.4$, where $\mu$ is the center of the trained
  voices. Check in the own spaces of each `-wide` training run (`a4_tau_zero_param.py`), fitted $\tau$ versus
  $\|\mu_N-v_0\|^2$: 147 versus 135 ($N=8$), 110 versus 107 ($N=32$), 117 versus 116 ($N=128$).

With $\tau=\|\mu-v_0\|^2$ and $\|c-v_0\|^2=\|c-\mu\|^2+2\langle c-\mu,\mu-v_0\rangle+\|\mu-v_0\|^2$, the rule becomes
$$
\text{copied}\iff D_N(c)<\underbrace{\|c-\mu\|^2}_{D(0)\ \text{of the rate-zero code}}+\;2\langle c-\mu,\ \mu-v_0\rangle .
$$
**Rate-distortion reading** [CONJECTURE]: the voice is copied when the trained codebook represents it better than the
rate-zero code (the mean voice), with a linear correction that favors voices on the side opposite the default voice. In
particular, a voice exactly at the center of the population, perfectly covered, lies on the boundary. I have no derivation
of this identity from training (it would require modeling the starting *checkpoint*); I treat it as a zero-parameter
hypothesis to be tested, not as a result.

### 4.2 Link with the metric and with noise

- Lemma 1: $\rho<\tfrac12$ is the copy criterion; in the Gibbs channel, $\rho\approx\pi_0(c)$ when the local mixture is
  close to $c$, so "copied $\iff\pi_0<\tfrac12$" (the decision of Prop. 2) [HEURISTIC; in the calculation I use the exact $\rho$].
- **Reference noise** [PROVEN]: with $c=v+\xi$, the margin
  $s(c)=\|c-v_0\|^2-\|c-v_{\rm nn}\|^2=\|v-v_0\|^2-\|v-v_{\rm nn}\|^2+2\langle\xi,v_{\rm nn}-v_0\rangle$:
  **$\|\xi\|^2$ cancels**. Averaging $R$ references only reduces the cross term (standard deviation ~13 units versus a spread of
  margins of ~60 across voices), and the effect on $q$ is symmetric around the threshold: this explains why $R=4$ did not change
  copying in the 38 voices (66→64%, 81→79%, 88→84%; 03 §5.2).
- **All-or-nothing**: the per-voice result is deterministic given $c$; the only noise is the synthesis noise $\sigma_w$
  (via $\kappa$), which reproduces the near-dichotomy (ICC 0.83–0.94).

### 4.3 Parameter count (honest)

| Item | Status | Origin |
|---|---|---|
| spectrum $\Sigma_b$, center $\mu$, $v_0$, noise $W_x$, trained voices | measured (do not use copying) | latents `n128_k32_s1-wide` |
| $K$, $M$, identity of the speakers of each training run | known | manifests |
| $\sigma_w=0.036$ | measured in `results.csv` (within voice) | 02 |
| $\|\Delta\|^2=0.65$ (in $\kappa$) | assumed (02); only smooths | — |
| $\tau$ | **1 free parameter**, fitted at 1 point ($q_{128}$); or 0 if one accepts $\tau=\|\mu-v_0\|^2$, found *post hoc* | — |
| $T$ | chosen from {1,2,3} (1 = Bayes value); changes $q_8$ by ±0.035 | — |
| choice of the common space (latent of $N=128$) | modeling decision [CONJECTURE: the space of the encoder trained with the most voices approximates the "voice space"] | — |

---

## 5. Part 4 — calibration and prediction

### 5.1 Curve of the 38 voices [DATA; `a3_modelo_codigo.py`, `a4`, `a5`, `a8_tau0_comum.py`]

| Variant | what was fitted | $q_8$ | $q_{32}$ | $q_{128}$ | Spearman per voice (8/32/128) |
|---|---|---|---|---|---|
| observed | — | **0.434** | **0.302** | **0.211** | — |
| Gibbs $T=1$, real codebooks | $\tau$ at $q_{128}$ | 0.375 | 0.263 | (0.211) | 0.54 / 0.54 / 0.50 |
| Gibbs $T=2$ | same | 0.401 | 0.264 | (0.211) | 0.50 / 0.57 / 0.50 |
| Gibbs $T=3$ | same | 0.445 | 0.279 | (0.211) | 0.50 / 0.56 / 0.50 |
| $\tau_0=\|\mu-v_0\|^2$, common space, $T=1/2/3$ | nothing (*post hoc*) | 0.39/0.40/0.39 | 0.26/0.26/0.24 | 0.23/0.21/0.19 | ~0.50 / 0.55–0.58 / 0.50 |
| $\tau_0$, own space of each training run, $T=2$ | nothing (*post hoc*) | 0.360 | 0.219 | 0.210 | 0.50 / 0.62 / 0.50 |
| hard rule, mean over random subsets of the 127 | $\tau$ at $q_{128}$ | 0.32 ± 0.045 | 0.25 ± 0.020 | (0.211) | — |
| Gaussian population, real references | $\tau$ at $q_{128}$ (real codebook) | 0.353 | 0.274 | 0.253 | — |
| Gaussian population, Gaussian references | same | 0.486 | 0.391 | 0.333 | — |
| $\tau$ calibrated on the 8 **trained** voices | $\tau$ at $q_{\rm seen}$ | 0.148 | 0.105 | 0.105 | — |

Comparison: the distances alone give Spearman $\rho(\text{copy},d_{LJ})=0.38/0.49/0.58$ and
$\rho(\text{copy},d_{NN})=-0.44/-0.19/-0.02$ in the common space (`a1`); the margin $d_{LJ}^2-d_{NN}^2$, 0.51/0.57/0.56.
The model combines the two and wins at $N=8$; at $N=128$ it ties with $d_{LJ}$ alone, as it should (the neighbor stops
mattering when coverage improves).

**Interpretation.**
- The mechanism "neighbor distortion versus attractor tolerance" **predicts the direction and ~70–100% of the drop** in
  $q$ between $N=8$ and 128 from a single point. The specific 8-speaker codebook of the training run was worse than
  the average of 8-speaker subsets (0.375 versus 0.32 ± 0.045 under the hard rule): part of the observed drop is luck in the
  draw of speakers, which a single seed does not allow us to separate.
- **The Gaussian hypothesis fails quantitatively**: on the same set of references, the real median $D_N$ is 39.2 / 26.6 /
  19.5 and the Gaussian one 39.6 / 29.5 / 23.5 (±3.6/1.6/1.0). Real voices get covered faster — consistent with
  an intrinsic dimension of 7–10, below the Gaussian null (03). The Gaussian version predicts a curve that is too flat.
- Distances and exponent: the real $D_N$ falls with exponent $\ln(39.2/19.5)/\ln16=0.25$ (noise floor included).

### 5.2 $N\times K$ grid and trained voices [DATA, `a7_grade.py`; without any fit to the grid]

Common space, $\tau$ from the 38 voices, dependence on $K$ from the theory ($\tau(K)=\tau_{32}-2T\ln(K/32)$, noise $1/K$):

| Group (14 training runs) | $T$ | RMSE | bias | correlation |
|---|---|---|---|---|
| 4 new voices | 1 | 0.261 | −0.235 | 0.29 |
| 4 new voices | 3 | 0.213 | −0.186 | 0.67 |
| 8 trained voices | 1 | 0.152 | +0.100 | 0.86 |
| 8 trained voices | 3 | 0.174 | +0.124 | 0.74 |

**Acknowledged failures.**
1. The 4 new voices of the grid: the model says that 3 of the 4 have a large margin at any $N$ (it predicts $q\approx0.25$);
   one observes 0.26–0.74, with half-copied voices. It was already known that these 4 are harder than average (02).
   With $T=3$, $K=8$ gets the right direction (0.59–0.68 versus 0.61–0.74 at $N=8$).
2. Trained voices: the ordering across training runs is well predicted (0.74–0.86), but the benefit of large $K$ is much larger than
   the $2T\ln K$ term: $K=128$ observed 0.09–0.15 versus 0.375 predicted. A **codeword quality** term is missing
   (the decoder learns to render voice $i$ with an error that falls with $K$), absent from the Gibbs channel.
3. $\tau$ calibrated on the trained voices (72–76) does not work for the new ones (114–121). With 8 voices that are nearly 0/1, the
   standard error of $q_{\rm seen}$ is ~0.15 and the difference (0.375 predicted with $\tau_E$ versus 0.22–0.24) is about 1
   standard error; but, together with item 2, it suggests that trained voices gain something beyond geometry.
4. The per-voice prediction is imperfect: of the 4 voices that the hard rule puts at the floor ($\|c-v_0\|^2<\tau$: speakers 4507,
   3729, 5639, 2961), two are already copied at $N=128$ (5639: 0.96; 2961: 0.68). The floor is an aggregate statement,
   not a reliable list of voices.

---

## 6. Part 5 — sample complexity and bits

### 6.1 Semi-parametric extrapolation [`a5_extrapola.py` → `out/a5.json`; HEURISTIC]

Empirical $p_c$ (fraction of the 127 trained voices within $t(c)$) when there are ≥3 neighbors within the tolerance
(28 of 38 voices); otherwise, the Gaussian tail (Imhof) multiplied by the empirical/Gaussian calibration factor measured on the
28 (0.99). Hard rule, $\tau=113.4$. Bootstrap of 120 resamples of the 38 voices, refitting $\tau$ in each.

| $N$ | 128 | 247 | 512 | 1000 | 2048 | 2456 | $10^4$ | $10^5$ | floor $q_\infty$ |
|---|---|---|---|---|---|---|---|---|---|
| $q$ (i.i.d. codebook) | 0.247 | 0.236 | 0.218 | 0.201 | 0.185 | 0.182 | 0.163 | 0.156 | 0.105 |
| 90% bootstrap CI | | | 0.13–0.30 | | | 0.105–0.26 | 0.095–0.24 | | 0.05–0.18 |

The value at 128 (0.247) is above the observed one (0.211) because drawing with replacement from 127 voices covers only ~63%
of them; what matters is the **predicted drop** from $N=128$ to 2456: −0.06 (Gaussian with real references: −0.06 as well).
**Prediction: $q(2456)\approx0.15$ (0.10–0.22).**

### 6.2 Bits per voice [`a6_bits.py`]

$L(c)=-\log_2p_c$ (random codebook) and $R_{\rm wf}(c)$ = reverse water-filling rate of the reference source at
tolerance $t(c)$ (best possible codebook):

| fraction of voices copied | 50% | 75% | 80% | 85% | ≥90% |
|---|---|---|---|---|---|
| bits, random codebook $L$ | 2.0 | 7.5 | 10.1 | 15.9 | ∞ (floor) |
| randomly drawn speakers $\approx2^L$ | 4 | 180 | ~1,100 | ~60,000 | — |
| bits, optimal codebook $R_{\rm wf}$ | 0 | 5.4 | 7.8 | 12.8 | ∞ |
| chosen speakers $\approx2^{R}$ | 1 | 40 | ~220 | ~7,000 | — |

Median difference random − optimal: 2.2 bits (a factor of ~5 in $N$). $N_{90\%}=N_{95\%}=\infty$ at the point estimate (floor
0.105); at the 5th percentile of the bootstrap, $N_{90\%}\approx7.7\times10^3$.

**Relation to the ~17 bits / 7–10 dimensions** [HEURISTIC]:
- What the decoder needs to resolve is not the 17 bits of $\mu$ (a fine description of the voice, including utterance
  noise), but the voice **up to the tolerance** $t(c)$: 2 bits for the median voice, 8–10 bits to reach 80%. The 8 bits
  lie in ~8 directions (1.6; 3.0; 4.3; 5.3; 6.2; 7.0; 7.5; 8.0 bits accumulated over the first 8), in agreement with the
  6–8 useful directions and the intrinsic dimension of 7–10.
- **Saturation**: one reference carries $I(v;c)=10.3$ bits. With $N\approx2^{10}$, $D_{\rm rand}\approx11.6$ is already
  of the order of the reference noise ($\mathrm{tr}W_x+32/K\approx12.9$): above $N^\star\approx2^{I(v;c)}\approx10^3$,
  the codebook becomes finer than what the reference determines, and more speakers yield little. Voices that require
  $L>I(v;c)$ (the hardest 15–20%) are not resolved by coverage with a single reference [CONJECTURE].

---

## 7. Part 6 — consequences for the architecture

### 7.1 Voice module of dimension $k$ [`a5`, DATA + model]

The module renders only the projection onto the $k$ largest directions of $\Sigma_b$; the discarded part adds to the distortion:
$D^{(k)}_N(c)=\min_i\|P_k(c-v_i)\|^2+\|P_\perp(c-\mu)\|^2$. Rate-distortion floor (perfect codebook in the
subspace): $q_{\rm floor}(k)=\Pr\big(\|P_\perp(c-\mu)\|^2>t(c)\big)$, with $\mathbb E\|P_\perp(v-\mu)\|^2=\sum_{j>k}\lambda_j$.

| $k$ | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| $\sum_{j>k}\lambda_j$ | 24.3 | 18.5 | 13.3 | 10.2 | 5.9 | 3.8 | 1.3 | 0.5 | 0 |
| floor $q_{\rm floor}(k)$ | 0.29 | 0.24 | 0.21 | 0.18 | 0.18 | 0.16 | 0.13 | 0.105 | 0.105 |
| $q$ with the real codebook, $N=8$ | 0.29 | 0.24 | 0.29 | 0.29 | 0.32 | 0.32 | 0.34 | 0.34 | 0.37 |
| $N=32$ | 0.29 | 0.24 | 0.26 | 0.24 | 0.24 | 0.24 | 0.26 | 0.26 | 0.26 |
| $N=128$ | 0.29 | 0.24 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 |

(Hard rule; the point $k=32$ reproduces 0.37/0.26/0.21.) Interpretation [HEURISTIC]: reducing the dimension trades **faster
coverage** (in $k$ dimensions the neighbor is closer: $k=4$ gains 0.08 at $N=8$) for **a higher floor**
(discarded tail). With $N=128$, $k\ge3$ loses nothing; with large $N$ the floor dominates and $k\gtrsim12$ is needed for
the floor to fall to that of the attractor (0.105). The optimal dimension grows with $N$: $k^\star(N)\approx$ the number of directions with
$\lambda_j$ above the coverage distortion, the same logic as reverse water-filling ($\lambda_j>\theta(\log_2N)$).

### 7.2 Voice stage trained with untranscribed speakers [model; DATA]

In the two-stage proposal the codebook is the set of $N_{\rm voz}$ speakers with audio but no text; the formula holds
with $N\to N_{\rm voz}$, $K\to K_{\rm voz}$. Two cases:
- **Keeping a dominant voice** (the same attractor): $q(N_{\rm voz})$ follows table 6.1 and stops at the floor ≈0.10.
  The gain is only one of data scale: 2,456 → $10^4$ speakers gives 0.18→0.16 in the i.i.d. calculation.
- **Without a dominant voice** ($M_B\approx K$, $v_0\to\mu$, $\tau\to0$): "copied" becomes "the codebook beats the
  rate-zero code", $D_N(c)<\|c-\mu\|^2$. With the same 38 references and the real codebooks: $q=0.45/0.11/0.00$
  at $N=8/32/128$ (semi-parametric: 0.41/0.08/0.003). **The main gain would come from removing the attractor, not from more
  speakers.** Caveat: the criterion `sim_ref > sim_default` ceases to exist without a default voice; here I compare with the
  mean voice, and the translation to the real metric is [CONJECTURE].

---

## 8. Part 7 — falsifiable predictions

| # | Experiment | Prediction of this theory | Contrast | Decides |
|---|---|---|---|---|
| A1 | $N=512$, $K=32$ (≥200 evaluation voices) | $q=0.18$–0.19 (0.13–0.30) | 02 coverage: ~0.14; power law: 0.15 | weak on its own |
| A2 | $N=2048$ and 2456 | $q\approx0.15$ (0.10–0.22); curve flattening | 02: 0.09; power law 0.10 | $q(2456)<0.09$ refutes the floor |
| A3 | $N\to10^4$ (or two stages with $N_{\rm voz}=10^4$ keeping LJ dominant) | $q\to$ floor 0.10–0.15 | without floor: <0.07 | floor |
| A4 | $M$ halved (1,000 LJ utterances), $N=128$ | $\Delta q=-0.01$ ($T=1$) to $-0.06$ ($T=3$); if $\tau=\|\mu-v_0\|^2$ is geometric, $\Delta q\approx0$ | 02: −0.025 | measures $T$; $\Delta q<-0.08$ refutes the distortion regime |
| A5 | $K=8$ and $K=128$ with $N=128$ (new voices) | 0.26–0.30 and 0.16–0.18 (reference 0.25 in the same calculation) | mass regime: larger dependence | shape in $K$ |
| A6 | Voice latent $k=4$ (projection onto the first 4 directions) | $N=8$: improves ~0.08 (0.29 versus 0.37); $N=128$: same (0.21); floor rises to ~0.18 | 03: $k=8$ without loss | coverage × floor trade-off |
| A7 | $k=2$ | $q\approx0.24$ at all $N$ (floor) | | |
| A8 | Average of $R=16$ references | no systematic effect ($|\Delta q|<0.03$), because $\|\xi\|^2$ cancels | parametric model: gain | already consistent with $R=4$ |
| A9 | Two stages without a dominant voice | $q(N_{\rm voz}=128)<0.05$ in the "ref. versus mean voice" metric | one stage: 0.21 | attractor × coverage |
| A10 | Repeat the three `-wide` training runs with another sample of speakers | $q_8$ varies ±0.045, $q_{32}$ ±0.02 from the choice of voices alone | | separates speaker draw from seed |
| A11 | Resemblyzer embeddings of the 2,456 voices | $D_N$ in the metric space falls ~$N^{-0.25}$ up to $N\sim10^3$ and then saturates at the utterance noise | | links latent space and metric |

---

## 9. Status of each claim

| Claim | Status |
|---|---|
| Gibbs channel = Blahut–Arimoto step with training masses as the marginal | PROVEN (identity of form); that Tacotron behaves this way is CONJECTURE |
| MSE-optimal decoder = Gibbs with $T=1$ and nominal masses (Prop. 1) | PROVEN for Gaussian clouds; **refuted as a description of the data** |
| Poisson limit → distortion rule (Prop. 2) and the formula $\mathbb E_c(1-F_c)^N$ | PROVEN; SIMULATED (error ~0.01 with $T=1$) |
| Information-spectrum form $\Pr(L>\log_2N)$ | PROVEN as a bound/approximation; SIMULATED (0.01–0.03) |
| $D(R)\propto(\ln N)^{-(a-1)}$, local exponent $(a-1)/\ln N$ | HEURISTIC (Stirling); approximately SIMULATED |
| The random codebook pays 2–4 bits | SIMULATED (measured spectrum and $a=1$–2) |
| High resolution not reached up to $N=4096$ | SIMULATED |
| Tauberian theorem and $q-q_\infty\sim N^{-2/d}$ | PROVEN (Abelian) given $H(x)\sim x^\beta$; $\beta=2/d$ HEURISTIC; SIMULATED |
| Distortion regime ($T\lesssim3$) rather than mass regime | DATA (profile of section 3) |
| $\tau\approx\|\mu-v_0\|^2$ | DATA *post hoc* (3 spaces); interpretation CONJECTURE |
| Prediction of $q_8,q_{32}$ from $q_{128}$ | DATA: error 0.03–0.06 ($T=1$), 0.01–0.02 ($T=3$) |
| Gaussian hypothesis for $P_v$ | **quantitatively refuted** (real $D_N$ smaller) |
| Cancellation of the reference noise (averaging references does not help) | PROVEN; consistent with DATA |
| Grid of 4 voices and effect of $K$ on the trained voices | **fails** (bias −0.19/−0.24; benefit of $K$ underestimated) |
| Floor 0.10 and $q(2456)\approx0.15$ | HEURISTIC (semi-parametric extrapolation, wide CI) |
| Bits per copied fraction; $N^\star\approx2^{I(v;c)}$ | HEURISTIC / CONJECTURE |
| Floor of the dimension-$k$ module | PROVEN in the model; the numbers depend on the latent space |

**What is not identified.** With three points and one seed, exponent and floor cannot be separated (section 2.3); $T$ and the
effective mass of the default voice cannot be separated in the distortion regime (only $\tau$ can); the identity $\tau=\|\mu-v_0\|^2$
was noticed after the fit and holds in three spaces, which are not independent (same data, similar
encoders). The choice of the latent space of the $N=128$ training run as the "voice space" is the strongest hypothesis; the
numbers change by 0.02–0.04 if each training run is evaluated in its own space.

---

## Appendix — files

All in `/tmp/claude-0/-home-user-ml2-final-project/4c11fa8d-6b5f-5505-bd7b-deebf5dd9907/scratchpad/teoria2/A/`:

| Script | Does | Output |
|---|---|---|
| `rd_lib.py` | reading of latents/results; whitened space; $\Sigma_b$ by variance components | — |
| `rd_theory.py` | Imhof, saddle point (Lugannani–Rice), Chernoff, reverse water-filling | — |
| `a1_geometria.py` | spectrum, $v_0$, noise, distances, nested codebooks | `out/a1.json`, `out/a1_pervoice_N*.csv` |
| `a2_gibbs_dados.py` | $(T,\ln M)$ grid of the Gibbs channel on the real codebooks; profile | `out/a2.json`, `out/a2_grade.csv` |
| `a3_modelo_codigo.py` | calibration of $\tau$ (external and trained voices), Gaussian population, real × Gaussian $D_N$ | `out/a3.json` |
| `a4_tau_zero_param.py` | $\tau_0=\|\mu_N-v_0\|^2$ in each own space (17 training runs) | `out/a4.csv` |
| `a5_extrapola.py` | semi-parametric, subsets, bootstrap, $K$/$M$, module $k$, without a dominant voice | `out/a5.json` |
| `a6_bits.py` | $L(c)$, $R_{\rm wf}(c)$, bits per direction | `out/a6.json` |
| `a7_grade.py` | $N\times K$ grid without fitting | `out/a7.csv` |
| `a8_tau0_comum.py` | $\tau_0$ in the common space | `out/a8.json` |
| `t1_rd_teoria.py` | optimal × random × saddle-point × high-resolution $D(R)$; Imhof/saddle/MC check | `out/t1.json` |
| `t2_verifica_formula.py` | formula for $q(N,K,M)$ against simulated Gibbs | `out/t2.json` |
| `t3_tauberiano.py` | law $q-q_\infty\sim N^{-2/d}$ | `out/t3.json` |

Data read (read-only): `scratchpad/coverage2/<run>/step_2500/{latents.npz,train_manifest.csv}` and
`reports/coverage/results/<run>/step_2500/strength_1.0/results.csv`.
