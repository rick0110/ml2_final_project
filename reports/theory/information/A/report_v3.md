# *Zero-shot* cloning as source coding — v3: temperature, capacity and coverage

Working document (teoria2/A), 17 November 2026. v1 (`report_v1.md`) and v2 (`report_v2.md`) stay as they are.
Question for this round: are the "per-voice width" that implies $T\approx21$ (v2) and the "decoder uses ~1/22 of the evidence"
(independent Bayesian analysis) the same fact? And does the temperature fall with $N$?
New scripts: `c1_temperatura.py`, `c2_nominal.py`, `c3_extrap.py`, `c4_loro.py`; outputs `out/c1.json`–`out/c4.json`.
CPU only, `python3 -I`, no repository touched.

---

## 0. Summary

**Direct answer to the hypothesis: the temperature does NOT fall with $N$.** Estimated from the shape of the per-voice response in each
training run, $T(8)=18.7$, $T(32)=19.4$, $T(128)=27.1$, with CIs from 11 to 29–71 that overlap.
$\Pr(T_8>T_{128})=0.23$ in the bootstrap. A common $T$ ($21.6$) is not rejected. The capacity used stays at
$C\approx0.8$–$1.1$ bits at every $N$. **All of the drop in $q$ with $N$ comes from the codebook (coverage/mass),
none from capacity.** The convergence with the other analysis is numerical ($1/T\approx1/19$–$1/27$ versus $1/22$), but it is
a **fixed property of the decoder** (or a shared geometry artifact, §6), not a scaling mechanism.

| training run ($K=32$) | $T$ (90% CI) | $1/T$ | $C(T)$ bits (CI) | $\lambda=\ln(M_{\rm ef}/K)$ |
|---|---|---|---|---|
| $N=8$ | 18.7 (11.8–29.1) | 1/19 | 1.13 (0.75–1.70) | 4.26 |
| $N=32$ | 19.4 (11.7–26.3) | 1/19 | 1.09 (0.83–1.71) | 4.98 |
| $N=128$ | 27.1 (11.2–71.1) | 1/27 | 0.80 (0.32–1.77) | 5.04 |
| common | 21.6 | 1/22 | 1.06 | 3.91 / 4.72 / 5.45 |
| nominal $\ln(2000/32)$ | | | | 4.14 |

**Findings that change v2.**
1. **With high $T$, the nominal mass works again.** The level fitted per training run is
   $\lambda=4.3$–$5.0$, versus the nominal $\ln(M/K)=4.14$, that is $M_{\rm ef}\approx2,300$–$5,000$ versus
   2,000 LJ utterances. The "failure of $M$" of v2 ($M_{\rm ef}=10^{26}$) was an artifact of $T=1$.
   v2 rejected high $T$ by assuming a mass fixed across training runs; with the level free per training run, $T\approx21$ is compatible
   with the curve.
2. **Mass dilution.** The level rises by $0.56$ per unit of $\ln N$: the effective mass of the LibriTTS voices grows
   as $N^{0.44}$, not as $N$. In bits: **each doubling of $N$ adds 1 bit of codebook, of which
   ~0.44 bits reach the decision; capacity adds 0.**
3. **Decomposition of $q_8\to q_{128}$** (Shapley, fixed level): coverage −0.40 to −0.48 (CI −0.50 to −0.39);
   capacity +0.10 to +0.12 (CI −0.16 to +0.31, wrong sign and compatible with 0).

**Parameters and predictions** (38 external voices, common space = latent of the $N=128$ training run):

| prediction for training run $r$ | fitted on the level of $r$? | $q_8$ | $q_{32}$ | $q_{128}$ |
|---|---|---|---|---|
| observed | — | 0.434 | 0.302 | 0.211 |
| own $T$ (shape) + **nominal** $\lambda$ | no (0 level parameters) | 0.409 | 0.186 | 0.112 |
| common $T$ (shape) + nominal $\lambda$ | no | 0.481 | 0.214 | 0.090 |
| common $T$ + $\lambda$ interpolated/extrapolated from the other 2 training runs | no (leaving the training run out) | 0.453 | 0.294 | 0.223 |
| *null*: logit$(q)$ linear in $\ln N$, leaving the training run out | no | 0.412 | (interp.) | 0.196 |

The leave-one-training-run-out prediction is accurate to 0.01–0.02, **but the two-parameter null is just as accurate**: with
3 points, this test does not distinguish the theory from a smooth trend. The version with nominal mass (no level
parameter) gets $N=8$ right and misses $N=32$ and 128 by ~0.1, in the direction of predicting too large a drop. The difference is exactly the
mass dilution of item 2.

---

## 1. $T$ per training run

**Method** [DATA, `c1_temperatura.py`]. The Gibbs channel gives the copy logit of each voice:
$$
\operatorname{logit}P(\text{copy}\mid c)=\ln\!\sum_i e^{-\|c-v_i\|^2/2T}+\frac{\|c-v_0\|^2}{2T}-\lambda,\qquad \lambda=\ln\frac{M_{\rm ef}}K .
$$
The fit is binomial, with 25 utterances per voice, real codebooks and the common space (latent of the $N=128$ training run,
whitened). In each training run, $T$ sets the **shape**: how quickly the probability changes with the geometry of the voice.
$\lambda$ sets the **level** (the mean copying). The two are almost orthogonal. The bootstrap (150 resamples of the
38 voices) redoes both fits.

**Result:** table of §0. $T$ does not fall with $N$. The likelihood ratio between "common $T$" and "$T$ per training run"
is 10.4 on the binomial scale. The per-voice overdispersion is ~15×; once corrected, it is ~0.7 against $\chi^2_2$, far from
significant.

**Capacity used** [HEURISTIC]. The Gibbs channel at temperature $T$ is the Bayes posterior for codes with
noise $T$ per direction, instead of 1. The voice information that the decoder "sees" is
$$
C(T)=\tfrac12\sum_j\log_2\!\big(1+\lambda_j/T\big),
$$
with $\lambda_j$ the measured between-speaker spectrum: $C(1)=10.3$, $C(5)=3.4$, $C(10)=2.0$, $C(20)=1.06$,
$C(27)=0.81$ bits. With the measured $T(N)$, $C\approx1$ bit at every $N$.
Comparison with the other analysis (1.1–2.4 of ~14 bits; fraction 1/22): **same order of magnitude**. My $C$ is at the
lower end because the total scale here is 10.3 bits (channel $z$), not 14.

---

## 2. Prediction without fitting on the level of the predicted training run

The calculations are in `c2.json` and `c4.json`.

- **Zero level parameters** ($\lambda=\ln(2000/32)$; shape $T$, own or common): 0.41–0.48 / 0.19–0.21 /
  0.09–0.11. Gets $N=8$ right and predicts too large a drop afterwards.
- **Level from the other two training runs** (interpolation or extrapolation of $\lambda$ in $\ln N$), with common $T$:
  0.453 / 0.294 / 0.223. Errors of 0.02 / 0.01 / 0.01. With each training run's own $T$, noisier:
  0.565 / 0.252 / 0.316.

**Honest parameter count.**
- Measured: spectrum, $v_0$, noise, codebooks.
- Shape: $T$, one number, estimated from the shape of the per-voice responses of the three training runs (including the predicted one, but
  not its level).
- Level: two ($a$ and $b$ in $\lambda=a+b\ln N$), fitted on the other two training runs.
- Total: 3 parameters for 3 points. Each leave-one-training-run-out prediction uses 0 parameters from the level of the predicted
  training run.

**What this is worth.** A logit linear in $\ln N$ (2 parameters) makes the same leave-one-training-run-out prediction with
similar error (0.412 at $N=8$, 0.196 at $N=128$). **The theory does not beat the null trend on the curve.** What it
adds is the *interpretation* of the two level numbers: $\lambda$ around the nominal value, with slope
0.56 (dilution).

---

## 3. Decomposition: coverage × capacity

Counterfactuals in the fitted model (`c1.json: decomposicao`): $q(\text{codebook of }N_a,\ T\text{ of }N_b)$,
averaged over the two orderings (Shapley).

| step | total (model) | coverage | capacity |
|---|---|---|---|
| 8→32 | −0.27 / −0.22 | −0.28 / −0.24 | +0.02 / +0.01 |
| 32→128 | −0.09 / −0.07 | −0.18 / −0.14 | +0.09 / +0.06 |
| 8→128 | −0.36 / −0.30 | −0.48 / −0.40 | +0.12 / +0.10 |
| 90% CI (8→128, bootstrap) | [−0.66; −0.11] | [−0.50; −0.39] | [−0.16; +0.31] |

(First number: $\lambda$ fixed at the mean of the fits; second: nominal $\lambda$.)

**Interpretation.** Coverage explains more than the whole drop; capacity does not help (the point estimate gets worse, because
$T_{128}$ came out larger). The observed drop of 0.43→0.21 is **100% coverage** within the uncertainty.

In information units, in the high-$T$ regime (law of large numbers), the copy logit of a voice is
$$
\ln N+\ln\mathbb E_v\,k_T(c,v)+\frac{\|c-v_0\|^2}{2T}-\lambda(N),\qquad \frac{d\,\operatorname{logit}}{d\ln N}=1-0.56=0.44 .
$$
Each doubling of $N$ gives 1 bit of codebook; dilution takes back 0.56 bits; capacity gives 0.

**Why "coverage" here is not the nearest-neighbor distortion of v1.** With $T\approx21$, the kernel covers the whole population
($\lambda_1=8.3<T$). What matters is the **mass** of trained voices that the channel "sees" around $c$, not the
distance to the nearest neighbor. That is why the Gaussian codebook now works (`c3.json`: 0.40/0.31/0.21 with the
Gaussian, versus 0.42/0.30 with subsets of the real voices), whereas in v1 it failed.

---

## 4. $T$ and $K$

Grid with fixed $N$, fit on the 12 references of each training run (4 new + 8 trained; very wide CIs):

| $N$ | $K=8$ | $K=32$ | $K=128$ |
|---|---|---|---|
| 8 | 5.3 / 8.4 | 9.5 | 15.1 / 18.2 |
| 32 | 13.2 / 8.8 | 15.1 | 15.7 / 15.6 |
| 128 | 12.0 | 15.3 / 11.1 | — |

Level $\lambda$ at $N=8$: 14.5/9.2 ($K=8$) → 6.2 ($K=32$) → 3.2/3.3 ($K=128$).

**Interpretation.** $T$ does not fall with $K$; the point estimate even rises, and capacity does not improve with $K$. The effect of $K$
is in the **level**: $\lambda$ falls by ~8 units from $K=8$ to 128, versus $\ln16=2.8$ for the nominal. This is the strong effect of
$K$ on the trained voices (v2 §5b), which still has no mechanism: **$K$ acts as a mass ~3× stronger than the count
of utterances, not as capacity**.

Caveat: the temperatures of the grid (5–18) are lower than those of the 38 external voices (19–27). The trained voices carry weight
in these 12 references, so $T$ depends on the set of voices. This is not what one would expect of a constant of the
decoder, and it reinforces the caveat of §6.

---

## 5. Consequences for the two-stage architecture

- **The current bottleneck is capacity**: ~1 bit used out of ~10 available. It does not rise with $N$ or with $K$.
  In the model, with nominal mass, lowering $T$ from 20 to 5 ($C$ from 1.1 to 3.4 bits) takes $q_8$ from 0.44 to 0.08
  and $q_{128}$ from 0.084 to 0.043 (`c2.json: varre_T`). **Gaining capacity is worth more than multiplying $N$ by 16.**
- **The voice stage should be sized by capacity, not just fed with more speakers.** More untranscribed speakers
  improve only the coverage term, which yields 0.44 bits of logit per doubling. This is still useful:
  $N_{\rm voz}=2048$ gives ~0.10 in the model. What could lower $T$ does not appear in the available data; candidates
  [CONJECTURE] are a wider or deeper voice conditioning, the KL weight and longer training (the
  2,500 steps start from a single-speaker *checkpoint*).
- **Cheap diagnostic**: measure $T$ from the per-voice width (this method) in every new training run. It is the number that tells whether
  an architecture change increased capacity.

**Falsifiable predictions** (38 voices; better with ≥200):

| training run | capacity term | coverage term | predicted $q$ |
|---|---|---|---|
| $N=512$, $K=32$, current architecture | $T=20$ (11–30), $C\approx1$ bit, same as $N\le128$ | $\lambda(512)=6.2$ (dilution 0.56) | 0.14 (Gaussian) |
| $N=2048$ | same $T$ | $\lambda=7.0$ | 0.096 |
| $N=2456$ | same $T$ | $\lambda=7.1$ | 0.090 |
| $N=512$ without dilution (mass ∝ $N$) | same $T$ | $\lambda=\lambda(128)$ | 0.03 |
| two stages, no dominant voice, $N_{\rm voz}=512$ | if $T$ is a property of the decoder: $T\approx20$ | — | measured $T$ < 10 refutes "fixed capacity" |

Refutation criteria: $T(512)<10$ (CI outside 11–30) refutes "capacity does not depend on $N$";
$q(512)<0.06$ refutes dilution; $q(512)>0.22$ refutes the coverage term.

---

## 6. If $T$ does not depend on $N$ (which is the case)

I say it clearly: **the hypothesis "training with more speakers lowers $T$" is rejected by the available data**,
within wide CIs. The convergence with the Bayesian analysis is then one of two things:

1. **A fixed property of the decoder**: it uses ~1/20 of the voice evidence of the reference at any $N$, which
   points to a limitation of architecture or of training, not of data. Implication: scaling $N$ pays off slowly (0.44 bits of
   logit per doubling), and what would change the curve is increasing capacity.
2. **A shared geometry artifact**: both analyses measure how much the decoder's decision depends on the latent
   of the reference. If the latent (space of the $N=128$ training run) is a noisy measure of the space where the decoder
   decides, both methods see a response "softened" by the same noise, and the agreement 1/22 ≈ 1/21 is not
   independent. Two indications in favor: $T$ changes with the set of voices (grid 5–18 versus external 19–27), and in v2
   §3 the prediction changed by ~0.1 depending on the latent space.

I cannot separate 1 from 2 with the saved data. The test is the architecture diagnostic: if increasing the width of the
voice conditioning lowers the measured $T$ and $q$ together, it is (1); if $q$ changes without $T$ changing, or the reverse, it is (2).

---

## 7. Status (updates v2)

| Claim | v3 |
|---|---|
| $T$ falls with $N$ | **rejected** ($T\approx19$–27, no trend; common $T$ accepted) |
| capacity used $\approx1$ bit, $1/T\approx1/22$ | DATA; same order as the independent analysis; may be a shared artifact (§6) |
| nominal mass $M$ explains the attractor | **rehabilitated** with high $T$ ($M_{\rm ef}$ 2,300–5,000 versus 2,000) |
| dilution: LibriTTS mass ∝ $N^{0.44}$ | DATA (3 points); mechanism not derived |
| drop in $q$ = coverage | DATA within the model; capacity contributes 0 (CI −0.16 to +0.31) |
| $K$ acts through the level (~3× the nominal), not through capacity | DATA on the grid (noisy) |
| leave-one-training-run-out prediction | 0.01–0.02 error, but **same as the** logit-linear **null** |
| distortion regime ($T\approx1$) of v1/v2 | **not preferred**: compatible with the curve, incompatible with the per-voice width; with a free level, high $T$ explains the curve, the width and the nominal $M$ |
| floor 0.10 of v1/v2 | **does not survive** in the high-$T$ regime: mass accumulates and $q\to0$ slowly ($q(10^4)\approx0.06$) |
