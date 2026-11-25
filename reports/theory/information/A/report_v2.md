# *Zero-shot* cloning as source coding — v2 (revision after the critique)

Working document (teoria2/A), 17 November 2026. v1 (`report_v1.md`) stays as it is; this version corrects the parameter
count and answers the 10 points of the critique. Derivations that did not change (Propositions 1–2, verification of the
formula, reverse water-filling × random codebook comparison, Tauberian theorem) are in v1 and are only summarized here.
New scripts: `b1_revisao.py` (points 1, 3, 4, 5, 8, 10), `b2_aug.py` (point 6), `b3_regimes_extrap.py`
(points 5b, 7, 9); outputs in `out/b1.json`, `out/b2.json`, `out/b3.json`. CPU only, `python3 -I`, no repository
touched.

Legend: **[PROVEN]**, **[HEURISTIC]**, **[CONJECTURE]**, **[SIMULATED]**, **[DATA]** as in v1.
**[DEAD]** marks an idea that the revision knocks down.

---

## 0. Summary of v2

**Model (unchanged).** The decoder is treated as a Gibbs test channel over the training codebook
(masses $K$ per LibriTTS voice and $M$ for the default voice $v_0$, temperature $T$). In the Poisson limit:
$$
q(N,K,M)=\mathbb E_c\big[(1-F_c(\|c-v_0\|^2-\tau))^N\big],\qquad \tau=2T\ln(M_{\rm ef}/K),
$$
with $F_c(t)=\Pr(\|c-V\|^2\le t)$ and $V$ a randomly drawn trained voice.

**What changed verdict.**

| v1 claim | v2 |
|---|---|
| "1 free parameter" | **2 parameters** ($\tau$, $T$), plus the choice of the common space. With $T=1$ fixed *a priori*, the test becomes **1 parameter, 3 points, 2 d.o.f.**; these 2 d.o.f. only reject the hypothesis "no dependence on $N$"; they do not distinguish the model from reasonable alternatives |
| $\tau_0=\|\mu-v_0\|^2$ as a parameter-free prediction | **withdrawn**: moves to the section on post hoc hypotheses (§2) |
| $M$ enters as a count of utterances | **[DEAD]**: with $T=1$, $M_{\rm ef}\approx2\times10^{26}$. So $\tau$ is not a count. Prediction for $M/2$: $\Delta q\approx0$ if $T\approx1$; $-0.12$ if $T\approx21$ (§4) |
| trained voices as noisy codewords ($\sigma^2/K$) explain the effect of $K$ | **[DEAD]**: predicts $q_{\rm seen}=0.46/0.38/0.38$ versus $0.55/0.26/0.11$ (§5b) |
| averaging references "cancels exactly" | holds for $\|\xi\|^2$; the cross term does not cancel and explains the gain on the 4 voices of the grid (§8) |
| $q(2456)\approx0.15$, floor 0.10 | depends on the extrapolation model: 0.14–0.19, 90% CI from 0.07 to 0.29; the floor (0.105, CI 0.05–0.21) does not depend on the voice model, only on $\tau$ (§9) |
| bits per voice | the aggregate form stays; **the per-voice form does not predict better than distance** (§10) |
| exponent | no power law in $8\le N\le2456$; the model predicts $\alpha_g\approx0.53$ for the continuous measure (observed 0.55 in `sim_ref`, 0.41 in `sim_spk`), with a wide CI (§7) |

**Honest count and final table** (38 external voices, $K=32$, common space = latent of the $N=128$ training run):

| | fitted at | $q_8$ | $q_{32}$ | $q_{128}$ |
|---|---|---|---|---|
| observed | — | 0.434 | 0.302 | 0.211 |
| **main**: $T=1$ (*a priori*), $\tau$ fitted | $q_{128}$ | **0.375** | **0.263** | (0.211) |
| residual obs − pred, 90% CI (per-voice bootstrap, $\tau$ refitted) | | [−0.15; +0.21] | [−0.11; +0.14] | |
| null "$q$ does not depend on $N$", residual 90% CI | | [+0.14; +0.31] | [+0.02; +0.15] | |
| $T=3$ (chosen from {1,2,3} by looking at the curve: **2 parameters**) | $q_{128}$ | 0.445 | 0.279 | (0.211) |
| space of the $N=32$ training run, $\tau$ fitted at $q_{32}$ | $q_{32}$ | 0.557 | (0.302) | — |
| space of the $N=128$ training run, $\tau$ fitted at $q_{32}$ | $q_{32}$ | 0.455 | (0.302) | — |

The model reproduces 74% (N=8) and 57% (N=32) of the observed drop down to $N=128$. This is compatible with the data and
rules out "no effect of $N$", but with 38 voices and one seed it does not separate this model from any other monotone
curve. The prediction of $q_8$ changes by ~0.1 depending on the latent space.

---

## 1. Parameter count and $T$

**Attempt to fix $T$ by a measurement that is not the curve.** In the Gibbs channel, the copy probability of a voice
as a function of the margin $s=\|c-v_0\|^2-D_N(c)$ is logistic, $\sigma((s-\tau)/2T)$. I fitted this logistic to the
per-voice fractions (38 voices × 3 $N$, binomial with 25 utterances), with a single $\tau$ and a single width $w$. This uses the **positions**
of the voices, not the aggregate curve:
$w=42$ (90% CI 27–60) $\Rightarrow T_{\rm largura}=w/2\approx21$ (`b1.json: p1_width`).

**This value contradicts the curve** [DATA, `b3.json: regime_massa`]. With $T=21$ and $\tau$ fitted at $q_{128}$, the
model predicts $q_8=1.00$ and $q_{32}=0.85$. The measured width includes the error of the geometry: the latent is not the space
where the decoder decides, and so the width is an **upper bound** on $T$, not a measurement of it.
No available measurement fixes $T$ independently.

**Decision.** I fix $T=1$ *a priori*: it is the noise scale of the training codes in the units whitened by the channel,
the temperature of the MSE-optimal decoder (Proposition 1 of v1). With this the test has **1 fitted parameter
($\tau$), 3 points, 2 degrees of freedom**.

**What these 2 d.o.f. are worth.** The per-voice bootstrap, refitting $\tau$ in each resample, gives residuals with a 90% CI of
±0.15–0.2 at $N=8$. The null "no dependence on $N$" falls outside the CI; any reasonable monotone curve falls inside.
Value as a test: **weak**. It shows that the mechanism has the right direction and order of magnitude, not that it is the
right mechanism. Using $T=3$ improves the fit (0.445/0.279), but then there are 2 parameters for 3 points (1 d.o.f.), which
tests almost nothing.

## 2. Post hoc hypothesis to be tested: $\tau=\|\mu-v_0\|^2$

**Status: post hoc hypothesis, not a prediction.** I noticed the coincidence after fitting $\tau$. I have no derivation: the
Gibbs channel gives $\tau=2T\ln(M/K)$, which does not involve geometry. A justification would require modeling the starting
*checkpoint* (single-speaker decoder, fine-tuned for 2,500 steps), which I did not do.

Evidence for and against, with $\tau$ fitted against $\|\mu-v_0\|^2$ in the space of each training run:

| training run | 8 | 32 | 128 |
|---|---|---|---|
| fitted $\tau$ | 146 | 110 | 114 |
| $\|\mu_N-v_0\|^2$ | 135 | 107 | 116 |

The agreement is 2–8%, but the three spaces come from the same data and from similar encoders.

**Prospective test.** In a new training run (for example $N=512$), compute $\|\mu_N-v_0\|^2$ on the latents before
synthesizing and predict $q$ with $T=1$. It fails if the $\tau$ fitted afterwards deviates by more than 15%.

## 3. Circularity of the geometry

In v1, the main table used **a common space**: the latent of the $N=128$ training run, which contains the 127 trained voices
of the three nested codebooks, the 38 references and LJ. Only `a4` used the space of each training run. No available
space is external; the space of the 128 training run is partly a result of training with these voices (circular for
$N=128$, less so for the subsets 8 and 32).

Sensitivity (`b1.json: p3_espacos`), same procedure (fit $\tau$ at $N=32$, predict $N=8$, $T=1$):

| space | $\tau$ | predicted $q_8$ (obs. 0.434) | Spearman per voice |
|---|---|---|---|
| latent of the $N=32$ training run | 110 | 0.557 | 0.59 |
| latent of the $N=128$ training run | 122 | 0.455 | 0.48 |
| own space of each training run | 146 / 110 / 114 | — ($\tau$ is not constant) | — |

In the own spaces, a single $\tau$ does **not** work for the three training runs (146 versus 110–114). In the common spaces,
$q_8$ changes by 0.1 depending on the chosen space. **The uncertainty of the geometry is the same size as the effect to be predicted.**
An external space (speaker-verifier embeddings of the training voices) would resolve this; no embeddings of the
training speakers were saved.

## 4. The failure of $M$ and the effective mass

$\tau=2T\ln(M_{\rm ef}/K)$ gives, with the fitted $\tau$:

| $T$ | $\ln(M_{\rm ef}/K)$ | $M_{\rm ef}$ |
|---|---|---|
| 1 (*a priori*) | 57 | $2\times10^{26}$ |
| 5 | 13 | $1.6\times10^{7}$ |
| 21 (width) | 6.0 | $1.2\times10^{4}$ |

**[DEAD]** With $T\approx1$, $M_{\rm ef}$ is not a count of anything: the attractor does not come from the 2,000 fine-tuning utterances.
With $T\approx21$, $M_{\rm ef}\approx12,400$, curiously close to the 13,100 utterances of the full LJSpeech (pre-training).
But $T=21$ is incompatible with the curve in $N$ (§1). I record it as a coincidence, with no weight.

**Does the theory still predict an effect of $M$/2? It depends on $T$, and that is exactly what the experiment measures**
(`b1.json: p4_M`, `b3.json: Mhalf_dq128`), $N=128$, $K=32$, reducing $M$ from 2,000 to 1,000:

| hypothesis | predicted $\Delta q_{128}$ | 90% CI |
|---|---|---|
| distortion regime, $T=1$ | −0.001 | [−0.07; 0.00] |
| $T=5$ | −0.055 | — |
| $T=10$ | −0.09 | — |
| mass regime, $T=21$ | −0.115 | [−0.19; −0.04] |
| geometric $\tau$ (§2) | 0 | — |

**Registered prediction (2 training runs: $N=128,K=32$ with $M=2000$ and $M=1000$, 38 voices):** $|\Delta q|\le0.03$ if the
distortion theory with $T\approx1$ is right. $\Delta q\le-0.08$ knocks down $T\approx1$ and favors the mass
regime. Power caveat: the standard deviation of $\Delta q$ from the seed alone is ~0.05–0.08 (04), so
2 seeds per arm are needed to separate 0 from −0.1.

## 5. Out-of-sample failures

### 5a. The 4 fixed voices of the grid (`b1.json: p5a_vozes_grade`)

| voice | F0 | $\|c-v_0\|^2$ (percentile among the 38) | margin at $N=8/32/128$ (percentile) | obs. copying 8/32/128 (`-wide`) | predicted |
|---|---|---|---|---|---|
| 4297 | 171 | 201 (84%) | 160 / 187 / 187 (74–89%) | 1.0 / 1.0 / 1.0 | 1 / 1 / 1 |
| 696 | 168 | 161 (45%) | 125 / 139 / 142 (45–50%) | 0.0 / 0.04 / 0.64 | 1 / 1 / 1 |
| 6415 | 198 | 173 (50%) | 130 / 157 / 159 (53–71%) | 0.44 / 0.88 / 0.96 | 1 / 1 / 1 |
| 7800 | 183 | 60 (0%) | 15 / 15 / 27 (5–8%) | 0 / 0 / 0.40 | 0 / 0 / 0 |

Medians of the 38: F0 174 Hz, $\|c-v_0\|^2=168$, distance to the center 6.3 (the 4: 4.4–6.6).

**Explanation.** The 4 voices are **not** atypical: neither far from the codebook, nor close to the default voice, nor
shifted in F0 or in distance to the center. The difference of origin (held-out training speakers: 4297, 696, 6415,
7800, versus test-clean for the 38) does not show up in the geometry. The error comes from two voices (696, 6415) with median margin,
11–45 units above $\tau$, which fail at small $N$ (and 7800, at the floor, copies 40% at $N=128$). The hard-threshold model calls these voices "copied"; the data say that voices
in this range are copied only sometimes.

The same happens in the 38: the measured per-voice width is $w\approx42$, versus $2T=2$ in the model, and 2 of the 4 voices that the
model puts at the floor are already copied at $N=128$. **The model is too sharp per voice.** The aggregate fit on the 38
comes from errors that compensate each other ("copyable" voices that fail and "floor" voices that copy). With 4 voices there is no
compensation. The failure on the grid is not an anomaly of the 4 voices; it is the failure of the per-voice form (§10) showing up in a small
set.

### 5b. Effect of $K$ on the trained voices, $N=8$ (`b1.json: p5b_seen`, `b3.json: regime_massa`)

**Derivation.** A trained voice is the codeword itself, known through the mean of $K$ utterances. For the reference
$c=v+\xi$, $D_{\rm self}=\|\xi-\bar\varepsilon_K\|^2$, with
$\mathbb E D_{\rm self}=\mathrm{tr}W_x+\mathrm{tr}(I)/K=11.9+32/K$ = 15.9 / 12.9 / 12.2 for $K=8/32/128$.
The threshold shifts by $2T\ln(K/32)$. With $T=1$, the margin changes by ~10 units between $K=8$ and 128, versus a spread of
margins across voices of ~60.

| $q_{\rm seen}$, $N=8$ | $K=8$ | 32 | 128 |
|---|---|---|---|
| observed (seeds) | 0.55 / 0.555 | 0.26 | 0.09 / 0.125 |
| noisy codeword, $T=1$ | 0.46 | 0.375 | 0.375 |
| + fitted rendering error $A/K$ ($A=50$) | 0.50 | 0.375 | 0.375 |
| mass regime $T=21$ | 1.00 | 0.97 | 0.70 |
| $T=21$, total mass $\propto N^{0.25}$ | 0.79 | 0.53 | 0.05 |
| $T=10$, $N^{0.25}$ (best compromise) | 0.50 | 0.49 | 0.27 |

**[DEAD]** "Trained voice = noisy codeword $\sigma^2/K$" does not produce the effect of $K$: the estimation noise of the mean
is too small. No combination of ($T$, mass sublinear in $N$) fits at the same time the curve of the trained voices
in $K$ and the curve of the new voices in $N$. The effect of $K$ on the trained voices is another mechanism: how well the decoder learns
to **render** each voice (a learning error, not one of code geometry), which the Gibbs channel does not have.
The theory covers new voices in the distortion regime and does **not** cover trained voices.

## 6. Speed-perturbed pseudo-speakers (`b2_aug.py`, space of each `-aug` training run)

The latents exist. The manifest shows two kinds of pseudo-voice: perturbations of the 8 LibriTTS voices (16 to 47 in the
latents) and **6 perturbations of LJ itself** (`lj_v±2..4`, 251–487 utterances each).

| measure (median) | 8×8-aug | 8×128-aug |
|---|---|---|
| pseudo-voice → its source voice | 6.4 | 5.6 |
| pseudo-voice → nearest other real voice | 7.8 | 8.0 |
| real voice → nearest real voice | 5.7 | 6.2 |
| pseudo-LJ → LJ | 5.8–12.4 | 4.9–9.2 |
| fraction of the variance of the displacements in the 1st direction | 0.39 | 0.62 |

**Interpretation.**
1. The pseudo-voices do **not** stay glued to their source: they are as far from it as two neighboring real voices are from each other.
   But they move mostly along a common direction (the pitch/speed axis; 62% of the variance in
   8×128). They form a family of dimension ~1 around each voice and cover a thin slice of the space, not a
   region.
2. The pseudo-LJ voices place **high-mass** codewords **near the default voice**, that is, they widen the attractor. On the references,
   the margin drops by 18–58 units for trained voices (e.g. 2952: 240→182).
3. Predicted effect on the 4 new voices (margin with real voices only → with pseudo-voices): 8×128: 4297 +15, 696 +24, 6415 −1, 7800 +7.
   8×8: −2 to 0. With $\tau\approx110$–150, only 696 (122→146) can cross the threshold. Prediction: at most 1 voice of 4
   changes, $|\Delta q|\le0.25$, ambiguous direction (pseudo-LibriTTS voices help, pseudo-LJ voices hurt).
   Observed: 8×128 0.51→0.42; 8×8 0.26→0.28 (copied fractions). 696 went from 0.28 to 0.32, without crossing.

**Verdict:** the theory predicts a small effect of uncertain sign, and the null result is compatible with that. It is not a
contradiction. Nor is it a strong confirmation: with 4 voices and one seed, the test has no power. v1 cited the
prediction of 02 (large gain), made in a model in which the pseudo-voices were voices of the population; the measured geometry
shows that they are not (an almost one-dimensional family, plus an extension of the attractor).

## 7. Regimes and exponent

**Explicit crossover** (measured spectrum $a\approx1$–1.5, intrinsic $d$ 7–10):
- **Optimal distortion** $D(\log_2N)\propto(\ln N)^{-(a-1)}$: holds for $D$ as long as the number of coded directions
  $J\approx2\ln N/a$ is smaller than the ~8–14 directions with relevant variance, that is, up to $N\sim e^{4a\text{–}7a}$
  ≈ $10^2$–$10^4$. In the simulation, the high-resolution formula is off by 4–7× up to $N=4096$ (v1 §2.2). **In 8–2456,
  $D_N$ is in the low-rate regime.**
- **$q-q_\infty\sim N^{-2/d}$** (Tauberian) is an **asymptotic** statement: it only holds when the voices that remain
  are those at the boundary of the attractor and each one is already in local high resolution. In the calibrated Gaussian model, the
  local exponent of the excess $q-q_\infty$ is 0.27 / 0.11 / 0.17 / 0.25 / 0.22 / 0.09 / 0.03 / 0.15 between
  $N=8,32,\dots,10^6$ (`b3.json: expoente_local_gauss`). **There is no stable power law in any window from
  8 to $10^6$.** The curve is the survival function of the distribution of $L(c)$ along $\log_2N$
  (§10), full of steps with 38 voices. In 8–2456 neither of the two asymptotic regimes applies.
- The "2/d ≈ 0.26" agreement of v1 **is not evidence**: the binary exponent is not identified (CI from −0.7 to
  1.6) and the model, with the same numbers, does not produce a stable exponent. I withdraw this argument.

**Prediction for the continuous measures** (`b3.json: continua`). If `sim_ref` (and, approximately, `sim_spk`) is linear
in $-\|\hat y-c\|^2$, the model predicts the exponent through the ratio of gains
$\alpha_g=\log_4\frac{y_8-y_{32}}{y_{32}-y_{128}}$:

| quantity (model, real codebooks) | predicted $\alpha_g$ (90% CI over voices) | observed (04) |
|---|---|---|
| $\mathbb E\|\hat y-c\|^2$, $T=1$ | 0.53 [0.15; 0.92] | `sim_ref` 0.55 (−0.17; 1.88); `sim_spk` 0.41 (−0.59; 2.00) |
| $\mathbb E\|\hat y-c\|^2$, $T=21$ | 0.31 [0.25; 0.38] | |
| $D_N$ only (no attractor) | 0.33 | |
| binary $q$, $T=1$ | 0.55 | 0.26 (−0.73; 1.57) |

The point estimates agree with $T=1$, but all the observed CIs cover 0–1.5: **this is not a test**. Registered prediction: with
≥200 evaluation voices and $N\in\{128,512,2048\}$, $\alpha_g$ of `sim_ref` in 0.3–0.9.

## 8. Averaging references

v1 said that the noise cancels. To be precise: in the margin,
$s=\|v-v_0\|^2-\|v-v_{\rm nn}\|^2+2\langle\xi,v_{\rm nn}-v_0\rangle$, **$\|\xi\|^2$ cancels exactly; the cross
term does not** [PROVEN]. Its standard deviation is $\sigma_\times=2\sqrt{(v_{\rm nn}-v_0)^\top W_x(v_{\rm nn}-v_0)}\approx20$
(median over the 38). With $R$ references, $\sigma_\times/\sqrt R$: only voices with $|s-\tau|\lesssim\sigma_\times$ can
flip.

On the 4 voices of the grid (`b1.json: p8_R`), $z=(s-\tau)/\sigma_\times$: 696 has $z=0.49$ ($N=8$) and 1.5 ($N=128$);
6415 has 0.83 and 2.3. These are exactly the two failing voices with a margin **above** $\tau$, that is, whose single
reference had an unfavorable cross term. Reducing this term by averaging pushes them toward copying (regression to the
mean). 4297 ($z=1.9$–3.2) already copies; 7800 ($z=-7$ to $-9$) does not change. Prediction: 1–2 voices of 4 go up, that is,
+10 to +25 points in the per-speaker criterion. Observed: +13 (CI +1 to +35) on the 4 plain runs; 128×8: 47%→67%.

In the 38, only 21% of the voices are less than $1\sigma_\times$ from the threshold, and the flips go both ways.
Prediction ≈0; observed −2.5 points (CI −6.4 to +0.8).

**Reconciliation:** the gain on the 4 voices comes from **selection**. They are voices that failed with a positive geometric margin; the
effect of $R$ is real per voice and null on average over a population.

## 9. Uncertainty of the extrapolation (`b3.json: extrapolacao`)

Three models for $P_v$. All use the same $\tau$ (hard rule fitted at $q_{128}$) and are anchored at
$q_{\rm obs}(128)=0.211$, adding the drop predicted by each model. Bootstrap of 60 resamples of the voices,
refitting $\tau$:

| model of $P_v$ | $q(512)$ | $q(2456)$ | $q(10^4)$ | floor |
|---|---|---|---|---|
| empirical (127 resampled voices; resolution 1/127, no tail) | 0.19 [0.12; 0.29] | 0.19 [0.12; 0.29] | 0.19 | 0.105 [0.05; 0.19] |
| Gaussian (measured spectrum) | 0.18 [0.10; 0.29] | 0.145 [0.07; 0.28] | 0.13 [0.06; 0.28] | 0.105 [0.05; 0.21] |
| mixture of 2 Gaussians | 0.18 [0.09; 0.27] | 0.146 [0.07; 0.24] | 0.12 [0.06; 0.23] | 0.105 [0.05; 0.18] |

Notes:
1. The empirical model saturates because it has no tail: voices with no neighbor among the 127 never get covered. It is a
   pessimistic bound.
2. BIC prefers **1** component (127 points in 32 dimensions do not support a mixture); the mixture of 2 was
   imposed. It does not reproduce the faster real coverage: median $D_N$ 40.1/30.0/23.9 versus real 39.2/26.6/19.5,
   the same error as the Gaussian. **No parametric model tested captures the clustering of the real voices.**
   The parametric models underestimate coverage and therefore overestimate $q$ at large $N$.
3. The **floor** is the only number that does not depend on the voice model: $\Pr(\|c-v_0\|^2<\tau)$, 4 of the 38 voices.
   But the per-voice form is poor (§5a: 2 of the 4 already copy), so the real floor may be lower.

**Revised prediction:** $q(2456)$ between 0.14 and 0.19 (joint CI 0.07–0.29). $q(2456)<0.07$ refutes the floor; above
0.29 refutes the theory. The prediction of the floor-free laws of 02 (~0.09) lies **inside** the CI: the extrapolation does not
discriminate with 38 voices.

## 10. Bits as information spectrum, per voice

**Derivation** [PROVEN]. For an i.i.d. codebook, $\Pr(\text{failure}\mid c)=(1-p_c)^N$ with
$e^{-Np_c/(1-p_c)}\le(1-p_c)^N\le e^{-Np_c}$. Writing $L(c)=-\log_2p_c$ (self-information of the voice's tolerance
under the distribution of trained voices):
$$
q(N)\approx\mathbb E_c\exp\!\big(-2^{\log_2N-L(c)}\big)=\Pr\!\big[L(c)>\log_2N\big]+\int h(\ell)\,g(\ell-\log_2N)\,d\ell,
$$
with $g(x)=e^{-2^{-x}}-\mathbf 1[x>0]$ and $\int g=-\gamma_E/\ln2$. Hence, for a smooth density $h$,
$q(N)\approx\Pr\big[L(c)>\log_2N+\gamma_E/\ln2\big]$, that is, the effective threshold lies 0.83 bits above $\log_2N$.
A voice starts being copied when $\log_2N\gtrsim L(c)-0.83$. The statement "80% copying ≈ 10 bits ≈ 1,000
speakers" is the 80% quantile of $L$ (v1 §6.2). It depends on $\tau$ and on the Gaussian tail of $P_v$ [HEURISTIC].

**Per-voice test** (`b1.json: p10_L`). $-L(c)$ (Gaussian tail, or empirical count) versus simple distance:

| | Spearman $-L$ (Gauss. / emp.) | Spearman $d_{LJ}$ | Spearman margin | AUC $-L$ | AUC $d_{LJ}$ | Brier of $1-e^{-N2^{-L}}$ | constant Brier |
|---|---|---|---|---|---|---|---|
| $N=8$ | 0.50 / 0.52 | 0.38 | 0.51 | 0.80 | 0.73 | 0.209 | 0.178 |
| $N=32$ | 0.54 / 0.56 | 0.49 | 0.57 | 0.85 | 0.80 | 0.136 | 0.130 |
| $N=128$ | 0.53 / 0.53 | 0.58 | 0.56 | 0.74 | 0.78 | 0.121 | 0.108 |

**Verdict.** $L(c)$ ranks the voices better than $d_{LJ}$ at $N=8$ and 32, worse at 128, and never better than the simple
geometric margin, of which it is almost a monotone transformation. The per-voice probabilities implied by the information
spectrum are **worse than a constant** (Brier 0.209 versus 0.178 at $N=8$): they are too sharp
(§5a). **The per-voice form of the information spectrum does not hold up.** The aggregate form (quantiles of $L$ → required $N$)
remains a model calculation, with no per-voice test to support it.

The relation to the 10.3 bits per utterance remains a **conjecture**: the scale $N^\star\approx2^{I(v;c)}\approx10^3$ is where
the $D_N$ of the Gaussian codebook reaches the reference noise, but the parametric models underestimate the real
coverage (§9), so the real $N^\star$ should be smaller.

---

## 11. Final status

| Claim | v2 |
|---|---|
| Proposition 1 (Bayes, $T=1$, nominal masses) | PROVEN; **refuted** as a description (q≈0) |
| Proposition 2 and the formula $\mathbb E_c(1-F_c)^N$ | PROVEN; SIMULATED |
| Mechanism distortion × attractor tolerance for new voices | compatible with DATA (74%/57% of the drop predicted with 1 parameter); weak test (2 d.o.f., CI ±0.15) |
| $T$ | not identified; $T=1$ *a priori*; per-voice width gives 21, incompatible with the curve |
| $\tau=\|\mu-v_0\|^2$ | post hoc hypothesis (§2) |
| $M$ as a count of utterances | DEAD; the $M/2$ experiment measures $T$ (§4) |
| Trained voices as noisy codewords | DEAD (§5b) |
| 4 voices of the grid | explained as excess per-voice sharpness, not as an anomaly (§5a) |
| Pseudo-speakers | compatible null (~1D family + widened attractor); test without power (§6) |
| Asymptotic laws ($\ln N$, $N^{-2/d}$) in 8–2456 | do not apply; no stable power law (§7) |
| "0.26 ≈ 2/d" | withdrawn |
| Averaging references | the cross term explains the selective gain on the 4 voices and the null on the 38 (§8) |
| $q(2456)$, floor | 0.14–0.19 [0.07–0.29]; floor 0.105 [0.05–0.21]; the per-voice floor is unreliable |
| Per-voice information spectrum | DEAD as a per-voice predictor; aggregate is HEURISTIC |

**Parameters, final count.**
- Measured, without using copying: $\Sigma_b$, $\mu$, $v_0$, $W_x$, the voices of each codebook, $\sigma_w$.
- Assumed: $\|\Delta\|^2=0.65$ (only smooths); the common space (the latent of the $N=128$ training run; changes $q_8$ by ~0.1).
- Fixed *a priori*: $T=1$.
- Fitted to the curve: $\tau$ (1 number, at $q_{128}$).
- Remaining degrees of freedom: 2 ($q_8$, $q_{32}$), both inside the CI; only the flat null is rejected.
