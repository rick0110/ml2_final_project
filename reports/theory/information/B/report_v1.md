# *Zero-shot* voice cloning as hypothesis testing: Bayesian evidence, information bottleneck and generalization

Working report, 17/11/2026. CPU only, no training, no GPU. No repository file was changed.
Everything (scripts, outputs, this report) is in
`/tmp/claude-0/-home-user-ml2-final-project/4c11fa8d-6b5f-5505-bd7b-deebf5dd9907/scratchpad/teoria2/B/`.
Complements `reports/theory/02_derivation.md` (Lemma 1, coverage + attractor rule), `03_information.md` (bits in the latent)
and `04_data.md` (variance components).

**Status legend**: **[PROVED]** follows from the stated hypotheses; **[HEURISTIC]** relies on an uncontrolled approximation;
**[CONJECTURE]** modeling hypothesis; **[DATA]** measured in the `results.csv`/`latents.npz`; **[TEST ✓/✗]** prediction compared
with data that did not enter the calibration; **[FITTED]** number chosen to reproduce the very data shown.

---

## 0. Summary

1. **Theorem (decoder = Bayesian estimator)** [PROVED for the ideal squared-error decoder]. A decoder
   trained with squared error on noisy codes returns the posterior mean of the voice under the empirical training mixture
   (default voice with weight $M$, each LibriTTS voice with weight $K$). With Lemma 1 of `02`, **copying is a likelihood-ratio
   test**: copied $\iff \Lambda(c)=\ln\frac{K\sum_i p(c\mid v_i)}{M\,p(c\mid v_0)}>0$. In bits:
   $$\underbrace{E_v}_{\text{evidence against the default}}-\underbrace{C_v}_{\text{coverage deficit}}>\underbrace{\log_2(M/K)}_{=5.97\text{ bits}}.$$
2. **The version with no free parameter is rejected** [TEST ✗]. With the real channel noise (within-speaker + VAE posterior
   noise) the reference carries $\Lambda\approx 100$–$120$ nats (SD 45–50) of evidence: the ideal decoder would copy 95–100%
   of the voices. The real decoder uses $\eta\approx1/20$–$1/25$ of that evidence (**evidence efficiency**).
3. **No single-kernel smoother (tempered Nadaraya–Watson) explains at the same time** (a) trained voices that depend
   only on $K$, (b) the average of 4 references not helping and (c) the weak evidence against the default [PROVED as an
   incompatibility of scales; SIMULATED]: calibrated on the trained voices by $K$, the population model makes the trained voices
   depend on $N$ (not copied $0.48\to0.25\to0.10$ for $N=8\to128$), against the observed flat curve.
4. **The statistic $\Lambda$ ranks the external voices better and more stably than any single distance**
   [TEST ✓]: Spearman(copying, $\Lambda$) $=0.54/0.54/0.56$ at $N=8/32/128$, without using the copying of the external voices; $d_{LJ}$ gives
   $0.34/0.43/0.63$ and $d_{NN}$ $-0.48/-0.22/-0.12$. The switch "neighbor at small $N$, default voice at large $N$" of `03`
   comes out on its own: the coverage term saturates ($\log_2 n_{\rm eff}=0.7\to2.4\to4.0$ bits).
5. **The absolute level does not come out of the trained voices** [TEST ✗]: the law calibrated on the trained voices gives the external voices $q=0.16/0.12/0.08$
   against $0.43/0.30/0.21$. New voices receive less "coverage credit" than the Bayesian sum over trained voices
   would give them. The measured gain with $N$ is $\partial\Lambda/\partial\ln N\approx0.55$ times that of $\partial\Lambda/\partial\ln K$ on the trained voices.
6. **Why averaging references does not help**: the reference noise enters $\Lambda$ only through its projection on the gradient
   $(\bar u-v_0)/s^2$; the noise energy cancels between the hypotheses [PROVED]. The +2.5 to +5 bits of voice that $R=4$ adds to the
   code become **+0.7 to +0.9 bit** on the decision statistic and **+0.03 to +0.13 bit** "used" by the decoder [DATA +
   model]. The predicted per-voice change ($|\Delta p|=0.16/0.10/0.07$) has the right order ($0.07/0.10/0.06$ observed) [TEST, partial].
7. **All-or-nothing = large deviations in the utterance length** [TEST ✓]: the SD of the within-voice margin falls as
   $T^{-0.38\pm0.03}$ (CLT with correlated frames); it predicts, and confirms, that the long half of each voice's utterances is
   more all-or-nothing than the short half (intermediate voices −24 to −35% observed, −21 to −25% predicted). The width of the posterior
   lies **between** a pure posterior mean (ICC 0.99, rejected, $\chi^2=33$–70) and posterior sampling (ICC 0.63, rejected,
   $\chi^2=8$–16): a **tempered posterior**.
8. **Information bottleneck**: the decoder uses **1.1–2.4 bits** of the ~14 bits of voice in the reference. The WER has two terms:
   displacement of the voice away from the default (slope −0.51 WER per unit of `sim_default`, within training) and the
   "nuisance" energy of the reference utterance (calibrated at $R=4$). Without using the strength 0.6 data, it predicts the WER drop at strength
   0.6: $-5.7/-5.9/-5.8$ points against $-7.5/-4.4/-6.4$ [TEST ✓]. But the **drop in copying** at strength 0.6 predicted by the
   geometry is 2.5–3× too large [TEST ✗].
9. **Generalization (Xu–Raginsky, meta-learning)**: the bounds give exactly the form $a+A/\sqrt N+C/\sqrt K$ (the L2 law that
   won the cross-validation in `04`) and predict trained voices $\propto K^{-1/2}$ independent of $N$: measured $K^{-0.46}$ in copying,
   $K^{-0.38\pm0.08}$ in WER [TEST ✓ of form]. The new−trained gap falls as $N^{-0.68\pm0.18}$, compatible with
   a **constant** total decoder–speakers mutual information in $N$ (fixed budget of 2,500 steps) [HEURISTIC].
10. **Sharpest predictions**: (i) **halving $M$ changes almost nothing** ($q$: 0.22 → 0.20–0.22 in the per-voice state) if the weight of the default voice
    comes from the total exposure to LJ (the calibrations on the trained voices give $M_{\rm eff}\approx8\times10^3$–$1.4\times10^5\gg2000$); under the
    count prior, 0.22 → 0.13. (ii) A 2-dimensional latent **does not worsen** the binary criterion ($q\approx0.09$–$0.13$) but loses
    ~63% of the voice bits used; 8 dimensions keep 78–80%. (iii) $N=512/2048$: $q=0.12/0.054$ (per-voice law, $\gamma$ fitted).
    (iv) Leakage in the units of the two-stage architecture: the limit is **per frame** and accumulates; between 0.001 and 0.3 bit/frame
    for $q$ to rise from 0.21 to 0.25, depending on whether the decoder integrates the evidence from the units as an ideal test or with the
    same efficiency $\eta\approx1/25$ measured for the code.

---

## 1. Data, conventions and reproduction

- Latents: `coverage2/<run>/step_2500/latents.npz` (μ, log σ², speaker; `lj` = default voice). Results:
  `reports/coverage/results/<run>/step_2500/strength_*/results.csv`. Runs used: base grid (N ≥ 8; N = 2 only where stated)
  and `-wide` (38 external voices × 25 utterances, one seed).
- **Whitened coordinates**: $W$ = within-speaker covariance (training LibriTTS), regularized
  $W+0.05\,\mathrm{tr}(W)/32\cdot I$ (without this, runs with $N=8,K=8$ have a nearly singular $W$). In this metric an utterance deviates
  from its speaker's mean with variance 1 per direction ($\mathrm{tr}\,W\approx32$).
- Unit of $\Lambda$: **nats of log-count** — 1 unit = multiplying $K$ by $e$ for a trained voice (scale anchored on the
  probit of $q_{\rm seen}$ in $\ln K$, $b=0.42$, from `04`). $\tau$ = utterance noise in this unit.
- Reproduction (all with `python3 -I`, data passed as an argument, nothing written outside this folder):

```
python3 -I b0_extrai.py <coverage2> <results> out/b0.npz   # extracts geometry + per-reference summaries
python3 -I b1_explora.py  out/b0.npz            # parameter-free Bayes (T=1) and tempered           -> out/b1.txt
python3 -I b2_ajuste.py   out/b0.npz out/       # per-run kernel, calibrated on the trained voices  -> out/b2.txt, b2.json
python3 -I b2b_diag.py    out/b0.npz            # diagnostic sweep of s², M_eff                    -> out/b2b.txt
python3 -I b3_fatos.py    out/b0.npz            # WER×copying, duration, R=4, strength 0.6 (no model) -> out/b3.txt
python3 -I b4_pop.py      out/b0.npz out/ n128_k32_s1-wide 20   # population model (no-go)        -> out/b4_n128.txt
python3 -I b5_geo.py      out/b0.npz            # decision axis, ICC on the axis, spread of d²      -> out/b5.txt
python3 -I b6_previsoes.py out/b0.npz 28 0.88   # per-voice predictions (strength, R=4, M, N, dimension) -> out/b6_s*.txt
python3 -I b7_orcamento.py out/b0.npz           # bit budget, duration, WER                        -> out/b7.txt
python3 -I b8_lei.py      out/b0.npz            # atom+coverage law calibrated on the trained voices -> out/b8.txt
python3 -I b9_extras.py   out/b0.npz            # bits used, WER×K, ICC per link, leakage          -> out/b9.txt
python3 -I b10_gap_hist.py out/b0.npz           # generalization gap, per-voice histogram          -> out/b10.txt
python3 -I b11_penalidade.py out/b0.npz         # generalization penalty per N                     -> out/b11.txt
```
Libraries: `bt_lib.py` (log-ratio Λ, links), `bt_pop.py` (population model), `b2_core.py` (definitions of `b2`).

---

## 2. The decoder as a Bayesian estimator of the voice

### 2.1 Hypotheses

- **B1** [CONJECTURE]. The rendered voice is (approximately) linear in the verifier's embedding space, and the voice of each
  training utterance is the voice of its speaker plus a deviation of its own.
- **B2**. Training: $M$ utterances of the default voice $v_0$ and $K$ utterances of each of $N$ voices $v_i$. The decoder receives the
  noisy code $z$ of each utterance; conditional on voice $i$, $z\sim p(\cdot\mid i)$ (Gaussian with covariance
  $\Sigma_c=W+\Sigma_{\rm post}$: within-speaker noise + VAE posterior noise).
- **B3**. The decoder minimizes the squared error and is flexible enough to reach the optimum on the training distribution.

### 2.2 Theorem 1 (posterior mean) [PROVED under B2–B3]

The minimizer of $\mathbb E\|y-f(z)\|^2$ is $f^\star(z)=\mathbb E[y\mid z]$. For the voice part of $y$ (B1),
$$
f^\star_{\rm voice}(z)=\sum_{j\in\{0,1,\dots,N\}}\pi_j(z)\,v_j,\qquad
\pi_j(z)=\frac{n_j\,p(z\mid j)}{\sum_k n_k\,p(z\mid k)},\quad n_0=M,\ n_{i\ge1}=K .
$$
Hence the shrinkage toward the default of `02` is **exactly the posterior weight of the default voice**, $\rho(c)=\pi_0(c)$, and
(in the limit where the posterior-weighted mean of the trained voices falls on the reference axis) Lemma 1
($m=(\tfrac12-\rho)\|\Delta\|^2$) gives:

**Corollary 1 (copying = Neyman–Pearson test)**:
$$
\text{copied}\iff\rho<\tfrac12\iff
\Lambda(c)\equiv\ln\frac{\sum_{i}K\,p(c\mid v_i)}{M\,p(c\mid v_0)}>0 .
$$
It is the likelihood-ratio test "the reference came from the mixture of trained voices" against "it came from the default voice", with
threshold equal to the log of the prior odds $\ln(M/K)$ ($\log_2(2000/32)=5.97$ bits).

### 2.3 Information budget [PROVED for Gaussian kernels of equal width $s^2$]

With $p(c\mid v)=\mathcal N(c;v,s^2I)$ and $c=v^\star+\xi$:
$$
\Lambda=\ln\tfrac KM+\underbrace{\frac{\|c-v_0\|^2}{2s^2}}_{E_v\ \text{(evidence)}}\;-\;\underbrace{\Big(-\ln\sum_i e^{-\|c-v_i\|^2/2s^2}\Big)}_{C_v\ \text{(coverage deficit)}} ,
$$
with $E_v=D_{\rm KL}\big(\mathcal N(v^\star,s^2)\,\|\,\mathcal N(v_0,s^2)\big)$ and $C_v$ a free energy ("soft min" of the divergences to the
trained voices, $C_v=D_{\rm nn}-\ln n_{\rm eff}$, $n_{\rm eff}\in[1,N]$ the number of trained voices within the resolution $s$).
Two properties decide the rest of the report:

- **(P1) The reference noise cancels in energy.** $\|c-v_0\|^2-\|c-v_i\|^2=\|v^\star-v_0\|^2-\|v^\star-v_i\|^2+2\langle\xi,v_i-v_0\rangle$:
  $\|\xi\|^2$ drops out; the noise enters only through its projection on the gradient $g=\nabla_c\Lambda=(\bar u_\pi-v_0)/s^2$, $\bar u_\pi$ the posterior mean
  of the trained voices. Decision variance per reference: $\sigma_{\rm ref}^2=g^\top Wg/R$.
- **(P2) $N$ and $K$ enter through different paths.** $K$ enters in the prior ($\ln K$, in full); $N$ enters only through the coverage
  deficit, at most $\ln N$ (law-of-large-numbers regime, `02` eq. 4.4) and typically much less (Poisson regime,
  dominated by the neighbor).

### 2.4 Scaling function [PROVED as an identity given the Corollary and Gaussian utterance noise]

$$
q(N,K,M)=\mathbb E_{v}\,\Phi\!\Big(-\frac{\Lambda_v(N,K,M)}{\tau}\Big),\qquad
\Lambda_v=\kappa\ln\frac{K}{M_{\rm eff}}+E_v-C_v(N),
$$
$\tau$ = per-utterance decision noise; $\kappa$ = how much the log-prior weighs (1 in the theorem). Per voice, the outcome is an
error probability of a binary test with $T$ frames per utterance; hence the all-or-nothing behavior (section 5).

---

## 3. Tests of the Bayesian model

### 3.1 With no free parameter: rejected; evidence efficiency [DATA; TEST ✗]

`b1_explora.py`: exact Gaussian posterior with $\Sigma_c$ measured in each run (within-speaker covariance + mean VAE
posterior noise, per class), weights $M=2000$ and $K$.

| run (external voices) | observed $q$ | predicted $q$ (T = 1) | median of $\Lambda$ (nats) | SD between voices |
|---|---|---|---|---|
| n8_k32_s1-wide | 0.43 | 0.05 | 100 | 48 |
| n32_k32_s1-wide | 0.30 | 0.02 | 123 | 50 |
| n128_k32_s1-wide | 0.21 | 0.00 | 122 | 44 |

The reference carries ~100–120 nats (~150–180 bits) of evidence against the default voice, against 4 nats (6 bits) of prior.
The real decoder behaves as if the evidence were ~20–25 times smaller: the spread between voices that reproduces the
all-or-nothing behavior and the $q_{\rm seen}(K)$ curve is ~2.2 nats against 44–50 nats for ideal Bayes, that is
**$\eta\approx0.045\approx1/22$**. Tempering the likelihood ($T=30$) makes the right level appear, but not the shape (below).

### 3.2 Single tempered kernel: incompatibility (no-go) [PROVED as a scale argument; SIMULATED]

**Proposition 2.** In a posterior-mean decoder with a Gaussian kernel of common width $s^2$ (any $s^2$),
the three facts below require incompatible scales:

1. *Trained voices depend only on $K$* (exponent of $N$ = 0 ± 0.07, `04`): the mass of the other trained voices seen from a trained
   voice has to be negligible compared with its own, $\sum_{j\ne i}e^{-d_{ij}^2/2s^2}\ll1$, that is $s^2\ll d_{\rm nn}^2/(2\ln N)$
   ($d_{\rm nn}\approx4$ whitened at $N=128$ ⇒ $s^2\lesssim1.6$).
2. *Weak evidence against the default* (SD of $\Lambda$ between voices ≈ 2.2 nats, with SD of $\|v-v_0\|^2$ = 54, `b5`):
   $s^2\approx54/(2\cdot2.2)\approx12$ (or ~28 on the Lemma 1 scale, section 3.5).
3. *$R=4$ does not help* (P1 holds only with a common width; with different widths $s_c^2\ne s_0^2$ a term
   $\tfrac12(\|\xi_1\|^2-\|\xi_4\|^2)(1/s_c^2-1/s_0^2)\approx12(1/s_c^2-1/s_0^2)$ nats enters, which with $s_c^2=1.6$ would give +6 nats, almost total copying).

(1) asks for $s^2\lesssim2$, (2) asks for $s^2\approx12$–28, (3) forbids using different widths to get around it. $\square$

Numerical check (`b4_pop.py`, population model with $\Sigma_b$, $W$, posterior noise and $v_0$ measured in
n128_k32_s1-wide; $s^2$ or $(s^2,M_{\rm eff})$ fitted **only** on $q_{\rm seen}(K)$):

| variant | $s^2$ | $M_{\rm eff}$ | $q_{\rm seen}$ K=8/32/128 (obs 0.50/0.26/0.12) | $q_{\rm seen}$ at N=8/32/128 (obs flat ≈0.3) | $q$ new N=8/32/128 (obs 0.43/0.30/0.21) |
|---|---|---|---|---|---|
| sampling, M=2000 | 29 | 2000 | 0.53/0.25/0.09 | **0.48/0.25/0.10** | 0.55/0.27/0.10 |
| sampling, free $M$ | 14 | 16,500 | 0.51/0.28/0.13 | **0.42/0.28/0.16** | 0.56/0.33/0.16 |
| posterior mean, free $M$ | 9.4 | 143,000 | 0.52/0.30/0.13 | **0.39/0.30/0.18** | 0.61/0.38/0.19 |

All the variants fit the trained voices in $K$ and **make the trained voices improve with $N$** (they borrow coverage from the
neighbors), which the data deny. The same holds with the geometry of each run (`b2_ajuste.py`: predictions of $q_{\rm seen}$
0.31/0.34/0.11 at $N=8/32/128$, $K=32$). **Rejected: the default voice does not compete with a sum over trained voices in a
single kernel.** What survives: (P1) and the model's identity $M/2\equiv2K$ (0.165 vs 0.164 at $N=32$), and the prediction that
$R=4$ does not change the mean (0.556/0.267/0.094 vs 0.550/0.267/0.096).

### 3.3 The statistic $\Lambda$ as a per-voice predictor [TEST ✓]

`b7_orcamento.py` (a). Log-count units, $s^2=12.1$ (from the spread of the trained voices; no data from the external voices used),
Nadaraya–Watson coverage with the trained voices of the run itself, distances corrected for noise ($-\mathrm{tr}W$,
$-\mathrm{tr}W/n_i$).

| run | median $E_v$ (bits) | median $C_v$ (bits) | $\log_2 n_{\rm eff}$ | prior | Spearman copying ~ $\Lambda$ | ~ $E_v$ | ~ $-C_v$ |
|---|---|---|---|---|---|---|---|
| n8_k32-wide | 18.3 (SD 4.8) | +1.3 | 0.7 | −5.97 | **0.54** | 0.42 | 0.46 |
| n32_k32-wide | 12.5 (SD 3.3) | −1.2 | 2.4 | −5.97 | **0.54** | 0.50 | 0.13 |
| n128_k32-wide | 15.2 (SD 3.9) | −3.4 | 4.0 | −5.97 | **0.56** | 0.59 | −0.07 |

Reading: the combination given by the theory is the best per-voice predictor at $N=8$ and 32 and ties at 128, and it is **stable** in $N$
(0.54–0.56), while each component switches roles: with few speakers the coverage varies a lot between voices (the neighbor
decides), with many it saturates ($n_{\rm eff}$ grows from 1.6 to 16) and only the evidence against the default discriminates. This is the
quantitative explanation of the pattern in `03` (section 6): $d_{NN}$ wins at $N=8$, $d_{LJ}$ at $N=128$. Reference ceiling: the rank
correlation of per-voice copying between runs is 0.59–0.75, so 0.55 is close to what the geometry of a single run allows.

### 3.4 Absolute level: it does not transfer from the trained voices [TEST ✗]

`b8_lei.py`: "atom + coverage" law (trained voices: own atom, no borrowing — obeys Proposition 2 by construction;
new voices: NW coverage or only the neighbor), $s^2=12.1$, $\tau=0.83$, a single parameter $\ln M_{\rm eff}$ fitted on the trained voices
($M_{\rm eff}=4\times10^4$).

| | N=8 | N=32 | N=128 |
|---|---|---|---|
| trained (mean over runs) obs / pred | 0.30 / 0.10 | 0.27 / 0.27 | 0.32 / 0.34 |
| external obs | 0.43 | 0.30 | 0.21 |
| external pred (NW / neighbor only) | 0.16 / 0.18 | 0.12 / 0.34 | 0.08 / 0.13 |

The prediction gives too much copying to the external voices and the error is not monotonic in $N$ (`b11`: required penalty 4.0/1.5/3.9 nats with
NW). **The geometry of each run (each with its own encoder) varies more between runs than the effect being sought**; the absolute
level between runs is not identifiable with latents from one run per point. The effects *within* a run (ordering of the
voices, condition changes on the same voice) are what the theory can predict; that is why sections 4–6 work with the
measured **per-voice state** and shifts $\Delta\Lambda$.

### 3.5 Posterior scale: three independent anchors agree on a tempered posterior [HEURISTIC; DATA]

Let $\kappa$ be the effective weight of the log-prior ($\Lambda\ni\kappa\ln K$). Three measurements that do not use the $q(N)$ curve:

1. **Lemma 1 + posterior mean** gives the per-utterance decision noise in true nats: $\tau_{\rm Lemma}=4\sigma_w/\|\Delta\|^2=0.19$
   ($\sigma_w=0.036$, $\|\Delta\|^2=0.75$ from the 4 voices with strength 0) or 0.48 (voices near the threshold, $\|\Delta\|^2\approx0.3$).
2. **Latent ICC** 0.83–0.94 (`04`) with the probit 0.42 of the trained voices: $\tau=0.83$ in log-count units.
3. **Width of the logistic transition** in $d_{LJ}$ (`03`: $d_{50}\approx15.5$, width 2) ⇒ $s^2\approx31$ whitened.

The three agree if $\kappa\approx0.23$–$0.58$ (equivalently: $s_0^2\approx20$–36 instead of 12): **the decoder weighs the
utterances of a voice as $K^{\kappa}$, $\kappa\approx0.4$, not as $K$** — which is the exponent $K^{0.46}$ of the collapse of the trained voices.
Consequence for the all-or-nothing behavior: the pure posterior mean with $\kappa=1$ gives ICC 0.993 and posterior sampling 0.63; the
observed value (0.88) is a **tempered** posterior (section 5).

Important: **the per-voice predictions of sections 4–6 are (almost) invariant to $s^2$** — $\sigma_L$, $\tau$, $\sigma_{\rm ref}$ and $\Delta\Lambda$
scale together with $1/s^2$ (`b6_s12.1.txt` to `b6_s36.txt` give the same numbers for strength 0.6, R = 4 and M; the projection onto k
dimensions varies a little, because of the nonlinearity of the log-sum-exp). The only number that depends on the scale is $\kappa$.

### 3.6 Per-voice law (what remains standing) [HEURISTIC; one parameter FITTED]

With $z_v=\Phi^{-1}(\hat p_v)$ the measured state of each voice in a reference condition ($N_0,K_0,M_0$), in units of $\tau$:
$$
\boxed{\;q(N,K,M)=\frac1{|V|}\sum_v\Phi\Big(-\big[z_v+\underbrace{0.66\ln\tfrac N{N_0}}_{\gamma/\tau,\ \text{FITTED (external)}}+\underbrace{0.35\ln\tfrac K{K_0}}_{\text{grid, 4 voices}}-\underbrace{1.21\ln\tfrac{M_{\rm eff}(M)}{M_{\rm eff}(M_0)}}_{\kappa/\tau,\ \text{trained}}\big]\Big)\;}
$$
(trained voices: $+1.21\ln(K/K_0)$ and nothing in $N$). The coefficients come from three different sets: $1.21=0.42/\sqrt{1-{\rm ICC}}$
(trained), $0.35=0.12/\sqrt{1-{\rm ICC}}$ (4 new voices of the grid), $0.66=0.23/\sqrt{1-{\rm ICC}}$ (external, fitted — it is the only term
taken from the curve itself). In terms of the prior: **a doubling of $N$ is worth, for a new voice, 0.55 of a doubling of $K$ for a trained
voice; a doubling of $K$ is worth, for a new voice, 0.29 of that.** The theorem would give both the same weight; the new voice gains with $N$
more slowly than a posterior over trained voices would allow (3.4) and almost nothing with $K$ (the two-stage design effect
predicts K irrelevant: ICC along the decision axis = 0.76 ⇒ $K_0=0.3$, `b5`; the $K_0=5.8$ of the collapse in `02` is not recovered).

---

## 4. Why more references do not help (evidence × coverage)

**Proposition 3** [PROVED given (P1) and a Gaussian state]. Let $\Lambda_v=\bar\Lambda_v+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma_{\rm ref}^2/R)$.
Averaging $R$ references changes the mean of $\Lambda$ by zero (the noise energy cancels) and changes the decision only for the voices with
$|\bar\Lambda_v|\lesssim\sigma_{\rm ref}$. The information that $R$ adds **about the decision statistic** is
$\tfrac12\log_2\frac{1+V_b R/\sigma_{\rm ref}^2}{1+V_b/\sigma_{\rm ref}^2}\le\tfrac12\log_2R$ = **at most 1 bit for $R=4$**, whatever the gain
about the identity of the voice in the code.

Numbers (`b6`, `b9`):

| | N=8 | N=32 | N=128 |
|---|---|---|---|
| gain in bits about the voice in $z$ (Gaussian, `03`) | +2.5 | +4.6 | +5.0 |
| $\sigma_{\rm ref}/\sigma_L$ (geometry: $\|\bar u-v_0\|/s^2$) | 0.65 | 0.49 | 0.55 |
| gain in bits about the decision statistic | +0.72 | +0.86 | +0.82 |
| voice bits **used** by the decoder, R=1 → R=4 ($s^2=12$–28) | | 1.1–2.4 → +0.03 to +0.13 | |

Per-voice test (external voices, `sim_spk` criterion on both sides; generative simulation with the Gaussian state, 25 utterances, the average of 4
includes the original reference):

| | pred N=8 | obs N=8 | pred N=32 | obs N=32 | pred N=128 | obs N=128 |
|---|---|---|---|---|---|---|
| per-voice $\mathbb E|\Delta p|$ | 0.16 | 0.07 | 0.10 | 0.10 | 0.07 | 0.06 |
| fraction that flips majority | 0.16 | 0.11 | 0.09 | 0.13 | 0.07 | 0.05 |
| corr($p_1,p_4$) | 0.77 | 0.94 | 0.81 | 0.81 | 0.77 | 0.90 |
| mean $\Delta$ copying | +0.02 | −0.02 | +0.02 | −0.02 | +0.03 | −0.04 |

[TEST, partial]: the order of magnitude of the per-voice change and the fact that it concentrates on the intermediate voices (observed:
$|\Delta p|=0.11$–$0.17$ on the intermediate ones against 0.04–0.07 on the extreme ones, `b3`) are right; at $N=8$ the reference noise
is overestimated; the sign of the mean comes out wrong (averaging references shifts the output slightly toward the default,
`sim_default` +0.007, an effect outside the model). **Conclusion: the extra bits go to directions orthogonal to the decision axis and
to a decoder that uses ~1–2 bits; the bottleneck is the prior/coverage, not the evidence.** The same argument explains the
ignored ECAPA vector (6.4).

---

## 5. All-or-nothing as large deviations

**Proposition 4** [PROVED given a per-utterance margin = mean of $T$ per-frame contributions with a sub-Gaussian tail].
For a voice with mean margin $\mu_v$ and frames with variance $\sigma_f^2$ and correlation time $t_c$,
$\Pr(\text{utterance fails}\mid v)\le\exp\!\big(-T\,\mu_v^2/(2\sigma_f^2t_c)\big)$ (Chernoff; Stein for the associated hypothesis test).
The fraction of intermediate voices is $\approx2\,z_{0.9}\,\tau\,g(0)\propto T^{-1/2}$, and the latent ICC $=\sigma_L^2/(\sigma_L^2+\tau^2)\to1$.

Tests (`b3`, `b6`, `b7`, `b10`):

1. SD of the within-voice margin per duration quintile: 0.040 / 0.036 / 0.033 / 0.032 / 0.029 ⇒ **$\tau\propto T^{-0.38\pm0.03}$**
   (−0.5 with independent frames; the smaller exponent indicates correlation between frames or a component that does not average out) [DATA].
2. **Prediction**: the long half of the 25 utterances of each voice is more all-or-nothing than the short half [TEST ✓]:

| run | voices with 0<p<1, short: obs / pred | long: obs / pred | reduction obs / pred |
|---|---|---|---|
| n8-wide | 0.45 / 0.57 | 0.34 / 0.45 | −24% / −21% |
| n32-wide | 0.55 / 0.63 | 0.37 / 0.49 | −33% / −22% |
| n128-wide | 0.37 / 0.53 | 0.24 / 0.40 | −35% / −25% |

   The direction is right in all three runs; the absolute level of intermediate voices is overestimated (the Gaussian state with the $\tau$ from the ICC
   widens the distribution) and the observed reduction is a little larger than predicted.
3. **Width of the posterior** (histogram of the 38 external voices in 6 bins; Gaussian state fitted only on the mean):

| link | $\tau/\sigma_L$ | $\chi^2$ (N=8/32/128) |
|---|---|---|
| posterior mean, $\kappa=1$ (Theorem 1 + Lemma 1) | 0.085 | 35 / 70 / 33 ✗ |
| posterior sampling | 0.76 | 16 / 8 / 14 ✗ |
| tempered (from ICC 0.88) | 0.37 | 1.3 / 5.3 / 1.3 (circular: the ICC comes from the same data) |

   It rejects both extremes. The all-or-nothing behavior comes from $\tau_\mu/\sigma_w\approx2.5$ (`02`), which here reads as "the per-voice
   information ($\sigma_L$) is ~2.7× the per-utterance noise ($\tau$)".
4. **Seeds**: the variation between seeds should concentrate on the voices with $|z_v|\lesssim1$ (≈ 25–35% of the voices); with 38
   voices the predicted SD of $q$ per seed is ≈ 0.02 + the main seed effect (`04`: 0.054), consistent with voice 6415 flipping.

---

## 6. Information bottleneck

### 6.1 Voice bits used [DATA + model]

$I_{\rm used}=\tfrac12\sum_j\log_2(1+\lambda_j/s^2)$ with $\lambda_j$ the whitened between-speaker eigenvalues (n128-wide):

| width | bits | fraction in k = 1 / 2 / 4 / 8 directions |
|---|---|---|
| $s^2=1$ (what the code carries, μ) | 14.3 | 0.13 / 0.24 / 0.42 / 0.66 |
| $s^2=12$ (decoder, count scale) | 2.4 | 0.20 / 0.35 / 0.56 / 0.78 |
| $s^2=28$ (decoder, Lemma 1 scale) | 1.1 | 0.22 / 0.37 / 0.59 / 0.80 |

The channel carries 64–75 bits/utterance (KL), ~14–17 about the voice; the decoder **uses 1–2.4 bits**. The rate is ~30–60× above the
decoder's operating point on the information–distortion plane.

### 6.2 WER in two terms; strength 0.6 and R = 4 [TEST ✓ on WER, ✗ on copying]

New measured facts (`b3`): between voices of the same run, **copying and WER correlate +0.3 to +0.95** (copied voices
are less intelligible), but within a voice the copied and non-copied utterances have the same WER (difference −0.003 ± 0.007,
161 mixed voices). The WER is a function of where the voice is rendered, not of which side of the threshold the utterance fell on.

Model [HEURISTIC]: $\mathrm{WER}=w_0+a_v\,\mathrm{sim}_{\rm default}(\text{output})+a_n\,\|\xi\|^2_{\rm eff}$ (displacement of the voice out of the
well-trained region + "nuisance" energy of the reference utterance that is not voice).

- $a_v=-0.51$ (SE 0.03; $r=-0.64$) calibrated **cross-sectionally**, strength 1 and R = 1 only, between voices within each run.
- $a_n=6.0\times10^{-4}$ per unit of $\|\xi\|^2$, calibrated at R = 4 (energy 32 → 8; predicted voice part −0.3 point, the rest −1.4 points).
- Strength 0.6 scales the code: nuisance $32\to0.36\cdot32$.

| group (plain runs) | obs ΔWER (strength 0.6) | pred. voice | pred. nuisance | pred. total |
|---|---|---|---|---|
| trained (128) | −7.5 pts | −4.5 | −1.2 | **−5.7** |
| new (64) | −4.4 | −4.7 | −1.2 | **−5.9** |
| external (16) | −6.4 | −4.6 | −1.2 | **−5.8** |

(Prediction conditional on the observed `sim_default` shift, +0.09.) **Strength moves along the copying–WER curve; R = 4 moves
off it** (it only reduces the nuisance): this is the "voice × rest" separation of the bottleneck. It explains why strength 0.6 lowers WER and copying
together and R = 4 lowers only WER.

**But copying at strength 0.6 is not predicted** [TEST ✗]: scaling the code by 0.6 in the NW geometry shifts $\Lambda$ by −4 to −7 $\tau$
(invariant to $s^2$), predicting copying of 0.6–6%; the observed values (25% / 19% / 47% for trained / new / external) require −1.7 to −2.2 $\tau$.
The decoder is ~2.5–3× less sensitive to the magnitude of the code than the quadratic evidence predicts (perhaps the style projection
is biased, or the decoder responds to direction more than to magnitude).

**WER constant in N** is only partially explained: the voice term predicts +2 points from N = 8 to 128 on the external voices (`sim_default`
0.651 → 0.612), the observed value is −1.7; it needs a fidelity term that improves with data (as for the trained voices, section 7).

### 6.3 Minimum dimension of the latent [CONJECTURE; model numbers]

Projecting code, trained voices and default voice onto the $k$ principal directions of the (whitened) training covariance and
recomputing $\Lambda$ (`b6`, per-voice state of n128-wide; ranges for $s^2=12$–36):

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| predicted $q$ (binary criterion) | 0.06–0.11 | 0.09–0.13 | 0.11–0.13 | 0.16–0.17 | 0.20 | 0.21 (obs) |
| fraction of the voice bits used | 0.20–0.22 | 0.35–0.38 | 0.56–0.59 | 0.78–0.80 | ~0.95 | 1 |

Two predictions that come apart: **the binary criterion does not get worse (it even improves) with small k**, because it only needs the
default voice–population axis; **identity (`sim_ref`, `sim_spk`) loses** in proportion to the bits used. Minimum rate for copying ≈ the axis
(1 direction, <1 bit); minimum rate for identity ≈ 4–8 directions (60–80% of 1–2 bits).

### 6.4 Pretrained speaker vector ignored [PROVED for the ideal decoder]

If training is for reconstruction and $I(y;e\mid z)=0$ on the training distribution, then $\mathbb E[y\mid z,e]=\mathbb E[y\mid z]$: the optimum
**ignores** $e$. The bound is $I(S;e\mid z)\le H(S\mid z)$: with speaker identification from one utterance already at 2.9 of 3.0 bits
($N=8$), 4.3–4.6 of 5.0 ($N=32$) and 5.1–5.3 of 6.9 ($N=128$) (`03`), ECAPA can add at most 0.1–1.8 bit about
*which* trained speaker it is, and nothing that the objective rewards for new voices. Prediction: ECAPA only starts to be used
when the code is degraded (`-spkdrop`) or when $N$ grows until $H(S\mid z)$ becomes large.

### 6.5 Two stages: leakage accumulates per frame [PROVED for per-frame independent channels; HEURISTIC numbers]

In stage B, the units $u_{1:T}$ leaking voice with divergence $D_u$ per frame (between the voice of A, close to the default, and the requested
voice) add to the test a contrary evidence $T_{\rm eff}\,\eta_u\,D_u$ (Stein: the error exponent grows linearly in the number
of independent frames), while the code enters once. Copied $\iff\Lambda_v>T_{\rm eff}\,\eta_u\,D_u$. With the state of the 38 external voices
at n128 (median margin of the copied ones 1.7 nats; `b9`):

| $T_{\rm eff}$ (independent frames) | $\eta_u$ | $D_u$ for $q$: 0.21 → 0.25 | → 0.30 |
|---|---|---|---|
| 20 | 1 (ideal test) | 0.012 bit/frame | 0.031 |
| 20 | 1/25 (measured efficiency of the code) | 0.30 | 0.77 |
| 200 | 1 | 0.001 | 0.003 |
| 200 | 1/25 | 0.030 | 0.076 |

`02` gave ~0.7 bit per dimension for a single vector; per frame, the limit is 10–700 times smaller. Measurable before training B:
per-frame speaker classifier on the units (gives $D_u$) and accuracy × number-of-frames curve (gives $T_{\rm eff}$).

---

## 7. Information-theoretic generalization

Hypotheses: speakers i.i.d. from $P$, $K$ i.i.d. utterances per speaker; loss $\ell\in[0,1]$ ($\sigma=\tfrac12$-sub-Gaussian); decoder $W$.

- **Trained voice** (Xu–Raginsky conditional on the voice) [PROVED]: $|\mathrm{gap}_i|\le\sqrt{2\sigma^2I(W;S_i\mid v_i)/K}$. If the bits that $W$
  keeps about the utterances of a voice are bounded ($C_v$), the term falls as $K^{-1/2}$ **and does not depend on $N$**.
  Measured: copying $K^{-0.46}$, exponent of $N$ 0 ± 0.07; WER of the trained voices $-w_\infty\propto K^{-0.38\pm0.08}$ (0.186/0.132/0.109 for
  K = 8/32/128) [TEST ✓ of form; the bound is an upper one, so "exponent ≤ 1/2" is what is tested].
- **New voice** (meta-learning, Jose–Simeone / Chen et al. 2021) [PROVED]:
  $\mathrm{gap}\le\sqrt{2\sigma^2I(W;v_{1:N})/N}+\mathbb E_i\sqrt{2\sigma^2I(W;S_i\mid v_{1:N})/K}$ — **the form $a+A/\sqrt N+C/\sqrt K$ (L2), the best law by
  cross-validation in `04`**. If $I(W;v_{1:N})\propto N^{\theta}$, the term falls as $N^{-(1-\theta)/2}$.
- **New − trained gap** (4 new voices of the grid, all K): +0.31 / +0.21 / +0.03 / +0.03 at N = 8/32/128/200 ⇒
  $N^{-0.68\pm0.18}$, compatible with $\theta\le0$ (total information about the set of speakers constant — a fixed budget of
  2,500 steps bounds $I(W;S)$ by the number of steps, Pensia et al.) [HEURISTIC].
- The slow decline of the external voices (apparent exponent 0.26, or 0.4–0.55 in the continuous measures) corresponds to $\theta\approx0.0$–0.5.
  The data do not pin down $\theta$; what the bound says is that **no exponent above 1/2 in $N$ or $K$ is guaranteed**, and that $N$ and $K$ enter in
  separate terms (the $NK/M$ collapse of the LLN regime is not implied by it).

---

## 8. Falsifiable predictions

| # | Experiment | Prediction | Falsifies |
|---|---|---|---|
| F1 | $N=128,K=32$, **$M=1000$** (half of LJ) | exposure prior ($M_{\rm eff}\approx8\times10^3$–$1.4\times10^5$, calibrations on the trained voices): $q$ drops ≤ 2 pts (0.22 → 0.20–0.22); count prior: 0.13; `02`: 0.157 | a drop ≥ 5 pts rejects the exposure prior; < 2 pts rejects count and `02` |
| F2 | $M=4000$ | exposure: +0 to +4 pts (0.22–0.26); count: $q=0.36$ | > 8 pts rejects exposure |
| F3 | latent **k = 2 vs 8 vs 32** (128×32) | binary criterion: $q(2)\approx0.09$–0.13 ≤ $q(8)\approx0.16$ ≤ $q(32)$; `sim_spk` drops with k = 2 (37% of the bits used), not with k = 8 (80%) | k = 2 losing > 10 pts on the binary criterion (H_bits of `03`) |
| F4 | $N=512/2048$, K = 32, ≥ 100 voices | per-voice law: $q=0.12/0.054$; NW count posterior ($\gamma=\kappa$): 0.002/0 (already rejected for 8→128) | $q(2048)>0.10$ rejects constant $\gamma$ (there is a floor) |
| F5 | K = 64 on the external voices | $q$: 0.21 → 0.19 (weight of K for the new voices 0.35/ln) ; count: 0.13 | |
| F6 | utterances 2× longer in evaluation | fraction of intermediate voices falls ×$2^{-0.38}\approx0.77$; ICC rises | unchanged |
| F7 | replace the reference with another utterance of the same speaker | per-voice change $|\Delta p|\approx0.1$, concentrated on the voices with $|z_v|<1$; flips 5–15% | flips > 30% |
| F8 | larger KL weight (rate → 25–30 bits) | copying ±5 pts (it uses 1–2 bits); WER falls by up to ≈ $a_n\cdot32\approx2$ pts | copying falls > 10 pts |
| F9 | two stages: measure $D_u$ and $T_{\rm eff}$ on the units | floor of $q$ given by table 6.5; with $\eta_u$ = 1/25 and $T_{\rm eff}$ = 20, $D_u<0.3$ bit/frame keeps $q\le0.25$ | floor outside the range [ideal, η = 1/25] |
| F10 | no training: `sim_default` of the reference itself | (attractor) $\mathrm{sim}_{\rm default}(\text{output})-\mathrm{sim}_{\rm default}(\text{ref})$ grows with $\rho_v$ estimated by $\Lambda$ | no relation |

---

## 9. Parameter accounting and status

| Parameter | Value | Calibrated on | Used to predict |
|---|---|---|---|
| $s^2$ (evidence scale) | 12 (count) / 28 (Lemma 1) | spread of $d^2$ of the trained voices + probit $q_{\rm seen}(K)$ | per-voice ordering (3.3); per-voice predictions are invariant to it |
| $\tau/\sigma_L$ | 0.37 | latent ICC (`04`) | R = 4, duration, histogram (the last one circular) |
| $\ln M_{\rm eff}$ | 9.0–11.9 | trained voices (six variants, `b2`, `b4`, `b8`) | F1, F2; absolute level (failed) |
| $\kappa/\tau=1.21$ | — | probit of the trained voices in ln K | M, K of the trained voices |
| weight of K for the new voices 0.35 | — | 4 new voices of the grid | F5 |
| $\gamma/\tau=0.66$ | — | **curve of the external voices (FITTED)** | F4 only |
| $a_v$, $a_n$ (WER) | −0.51; 6×10⁻⁴ | cross-sectional strength 1; R = 4 | ΔWER strength 0.6 |
| $T_{\rm eff},\eta_u$ (leakage) | scenarios | not calibrated | F9 |

| Claim | Status |
|---|---|
| Theorem 1 (posterior mean), Corollary (LRT with threshold ln M/K), $E-C$ budget | PROVED (B1–B3) |
| Parameter-free Bayes explains the data | **rejected** (predicts 0–5% failure) |
| Evidence efficiency $\eta\approx1/22$ | DATA + model |
| Single tempered kernel (NW) | **rejected** (Proposition 2; trained voices would depend on N) |
| $\Lambda$ ranks voices (ρ ≈ 0.55 stable) and explains the switch $d_{NN}\to d_{LJ}$ | TEST ✓ |
| Absolute level between runs from the trained voices | **failed** (per-run encoder geometry) |
| Tempered posterior ($\kappa\approx0.4$); extremes rejected | HEURISTIC; histogram ✓ |
| R = 4: ≤ 1 bit about the decision; per-voice change | PROVED / TEST partial (sign of the mean wrong) |
| All-or-nothing by large deviations; short × long prediction | PROVED / TEST ✓ |
| WER in two terms; ΔWER strength 0.6 | HEURISTIC / TEST ✓ |
| Copying at strength 0.6 from the geometry | **rejected** (2.5–3× too strong) |
| ECAPA ignored | PROVED (ideal decoder) |
| Leakage accumulates per frame | PROVED (independent frames); HEURISTIC numbers |
| Forms of the generalization bounds (L2; trained voices $K^{-1/2}$ only) | PROVED (bounds); agreement of form |
| Design effect explains the weight of K for the new voices | **rejected** (ICC on the axis 0.76 ⇒ $K_0=0.3$, not 5.8) |
| Per-voice law and F1–F10 | CONJECTURE / predictions |

## 10. What remains unexplained

- **Why the new voice gains with N only ~1/3 of what a posterior over the trained voices would give it**, and why the trained voices do not
  borrow coverage. The simplest reading is that the decoder has two mechanisms — per-voice memorization (atom, weight
  $K^{\kappa}$) and generalization (a weight that grows as $N^{0.3}$ in bits) — but I did not derive the second exponent.
- **Strength 0.6** moves copying less than the latent geometry predicts (factor 2.5–3).
- **The absolute level** between runs: each run has its own encoder; comparing levels requires a common space (embedding the
  references in a fixed verifier, P1 of `02`, or saving checkpoints).
- **$M_{\rm eff}$** is poorly identified (1.1×10⁴ to 1.4×10⁵ depending on the variant); what is robust is $M_{\rm eff}\gg2000$, and that is what
  supports F1. The coincidence with the 13,100 LJSpeech utterances of pretraining is suggestive, not demonstrated.
