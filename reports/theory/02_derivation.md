# Scaling theory for *zero-shot* voice cloning: derivation

Working document, 17/11/2026. Complements `reports/theory/coverage_theory.md` (recorded facts and predictions),
`reports/proposal/proposal.tex` (proposal) and `01_literature.md` (literature survey). All simulations run on CPU
and are in `analyses/` (`sim_*.py`, outputs in `out/sim_*.txt`). No file in the repository was modified.

**Status legend** (applies to every tagged claim):

- **[PROVED]**: follows from the stated assumptions by a complete calculation (indicated).
- **[HEURISTIC]**: calculation with uncontrolled approximations (local law of large numbers, linearization, etc.).
- **[CONJECTURE]**: modeling assumption, not derived; this is what needs to be tested.
- **[SIMULATED]**: checked numerically in a synthetic model (script indicated). A simulation checks the *calculation*, not the *modeling assumption*.
- **[DATA]**: computed from the per-sample `results.csv` (read only).

---

## 0. Summary

1. **The binary metric is linear in the output** [PROVED]. With normalized speaker embeddings, the margin
   $m=\mathrm{sim}_{\rm ref}-\mathrm{sim}_{\rm default}=\langle \hat e,\ e_{\rm ref}-e_0\rangle=(\tfrac12-\rho)\,\|\Delta\|^2$,
   where $\rho$ is the **shrinkage of the generated voice toward the default voice**, measured only along the axis
   $\Delta=e_{\rm ref}-e_0$. Errors orthogonal to this axis do not count. "Copied" $\iff \rho<\tfrac12$ (up to noise).
2. **Binary link** [PROVED given Gaussian within-voice noise; DATA]: $q=\mathbb E_s\,\Phi(-\mu_s/\sigma_w)$ with
   $\sigma_w\approx0.036$ (constant in $N$) and between-voice deviation $\tau_\mu\approx0.08$–$0.10$. With the measured margins,
   it predicts $q=0.442/0.304/0.217$ against $0.434/0.302/0.211$ observed (38 voices, $N=8/32/128$).
   The per-voice "all-or-nothing" is $\tau_\mu/\sigma_w\approx2.5$.
3. **Three regimes** for $\rho$ (or for the error), with the same structure $E=\text{bias}(N)+\text{variance}(N)\,s^2(K)$,
   $s^2(K)=\sigma_b^2+\sigma^2/K$:
   - (a) parametric: $E_{\rm new}=\sigma_b^2+s^2(K)\,\frac{d}{N-d-1}$ [PROVED, SIMULATED] $\Rightarrow a+A/N+B/(NK)$ with $A/a=d$, $B/A=\sigma^2/\sigma_b^2$;
   - (b) coverage: $E\sim N^{-2/d}$ (nearest neighbor), $(s^2/N)^{2s/(2s+d)}$ (kernel) [known; SIMULATED];
   - (c) spectral ($\lambda_j\propto j^{-\alpha}$): $E\sim (s^2/N)^{(\alpha-1)/\alpha}$ [HEURISTIC, standard; SIMULATED approx.].
4. **Default-voice attractor** [PROVED for the Nadaraya–Watson smoother; CONJECTURE that Tacotron behaves this way]:
   $\rho(v)=\dfrac{M\,k(v,v_0)}{M\,k(v,v_0)+K\sum_i k(v,v_i)}$ ($M=2000$ LJ sentences). With few speakers:
   **copied $\iff \|v-v_0\|^2-\min_i\|v-v_i\|^2>2\ell^2\ln(M/K)$**. With many: it depends on $NK/M$.
5. **Single model** (coverage + attractor, voices with spectrum $\lambda_j=j^{-1}$, $d=16$) reproduces, with 5 parameters
   chosen on a grid for 3 points, the curve $0.43/0.29/0.21$, the all-or-nothing ($\tau_\mu/\sigma_w=2.5$–$2.8$), and the
   **signs** of the per-voice correlations (distance to the default +, F0 −, trained neighbor −) [SIMULATED]. The local exponent
   stays at $0.23$–$0.31$ from $N=8$ to $10^4$, with no floor. Without the power-law spectrum, no isotropic
   configuration reproduces both the slow exponent and the all-or-nothing at the same time.
6. **Sharpest predictions**: (i) $q(2456)\approx0.09$ (coverage) against $\ge0.16$ (laws with a floor), decidable with
   $\ge200$ evaluation voices; (ii) **halving the LJ sentences is equivalent to doubling $K$** for new voices; (iii) in
   two stages, copying follows $N_{\rm voice}$, and voice leakage into the units with weight $w$ creates a floor
   ($w=0.2\Rightarrow q\gtrsim0.15$; requires $I(\text{units};\text{speaker})\lesssim0.7$ bit per effective dimension);
   (iv) a parametric voice module of dimension $k$ has a floor $q_{\rm floor}(k)=\Pr\big(\|P_\perp(v-\mu)\|>\|P_k(v-v_0)\|\big)$,
   computable without training from the spectrum of the embeddings.

---

## 1. What the theory needs to explain

Facts (from `coverage_theory.md` and the proposal), plus regularities I have now measured in the `results.csv`
(`sim_0_dados.py`, output in section 1.2).

### 1.1 Recorded facts

| # | Fact |
|---|---|
| F1 | New voices: non-copied fraction falls with $N$ (4 voices, noisy; 38 voices: $0.434\to0.302\to0.211$ for $N=8,32,128$, $K=32$; apparent power $N^{-0.26}$). |
| F2 | Trained voices: $\approx0.14+2.87/K+0.22/N$ (they depend on $K$). |
| F3 | WER constant (12–13%) in $N$ for new voices. |
| F4 | Stable per-voice difficulty (ranks 0.6–0.75 across $N$); copying correlates with distance to the default voice (+0.5) and with F0 (−0.4 to −0.5; the default is female); distance to the trained neighbor predicts failure at $N=8,32$ ($-0.33$, $-0.40$), not at 128 ($+0.13$). Copying is nearly all-or-nothing per voice. |
| F5 | Negatives: ECAPA vector ignored or memorized; averaging 4 references does not help on the 38 voices; speed-based pseudo-speakers do not count; reference swapping and contrastive loss do not help. |
| F6 | Latent: ~70 bits/sentence (KL), 10–20 bits about the speaker, growing ~$\log N$. |

### 1.2 New regularities measured in the data [DATA]

Per-sample margin $m=\mathrm{sim}_{\rm ref}-\mathrm{sim}_{\rm default}$ (Resemblyzer), 38 external voices, $K=32$:

| $N$ | $q$ | mean margin | deviation **between** voices $\tau_\mu$ | deviation **within** voice $\sigma_w$ | skewness |
|---|---|---|---|---|---|
| 8 | 0.434 | −0.004 | 0.101 | 0.036 | −0.68 |
| 32 | 0.302 | 0.048 | 0.078 | 0.036 | −0.12 |
| 128 | 0.211 | 0.078 | 0.088 | 0.037 | +0.07 |

- $\sigma_w$ is the same in all three trainings; what changes with $N$ is the position of the per-voice margins.
- Correlation of per-voice margins between trainings: 0.75 (8–32), 0.85 (32–128), 0.65 (8–128).
- The margin gain from 8 to 128 correlates $-0.55$ with the margin at $N=8$: the worst voices gain the most.
- The exponent is poorly determined by 3 points: from the mean margin, $\beta=0.39$ (95% CI from $-0.09$ to $0.94$, resampling voices); from the probit of $q$, $0.13$ ($-0.83$ to $1.39$). **The number 0.26 is a description, not a precise measurement.**
- **The variation between seeds is one voice that flips.** For the 4 fixed voices of the grid, seeds 1 and 2 use the same speakers; per-voice copying is, for example, $128\times32$: {696: 0.76/0.72; 4297: 1.0/1.0; **6415: 1.0/0.52**; 7800: 0.2/0.0}; $8\times128$: {0.28/0.12; 1.0/1.0; **0.76/0.36**; 0/0}. Two voices are always copied or never; the difference between seeds comes almost entirely from one voice near the threshold.

---

## 2. Common generative model and the geometry of the metric

### 2.1 Assumptions

- **H1 (voice space)** [CONJECTURE]. Each speaker $s$ has a voice $v_s\in\mathbb R^D$, identified with its mean speaker
  embedding (the space where the metric measures). Voices come from a population $P$ with mean $\bar v$, covariance
  $\Sigma_v$ with eigenvalues $\lambda_1\ge\lambda_2\ge\dots$, and intrinsic dimension $d$ (or spectrum $\lambda_j\propto j^{-\alpha}$).
- **H2 (sentences)**. Sentence $j$ of speaker $s$ realizes the voice $v_s+b_s+\varepsilon_{sj}$: $b_s\sim(0,\sigma_b^2 I)$ is the
  idiosyncratic part that does not fit in the code; $\varepsilon_{sj}\sim(0,\sigma^2 I)$ is sentence noise. The mean of $K$ sentences has noise
  $s^2(K)=\sigma_b^2+\sigma^2/K$ around the "explainable" part $v_s$.
- **H3 (training)**. $M=2000$ sentences of the default voice $v_0$ (LJSpeech, female) + $N$ speakers $\times K$ sentences. The model
  starts from a single-speaker checkpoint, which renders $v_0$ for any code.
- **H4 (inference)**. A reference gives a code $c=v^\*+\xi$ (encoder noise $\sigma_\xi^2$; $R$ references: $\sigma_\xi^2/R$). The decoder
  renders a voice $\hat y(c)$; each generated sentence comes out with noise $\eta\sim(0,\sigma_\eta^2 I)$.
- **H5 (error)**. Rendering error for a voice: $E=\mathbb E\|\hat y-v^\*\|^2$. Trained voices: $v^\*=v_i$; new: $v^\*\sim P$.

### 2.2 Lemma 1: the metric measures only the shrinkage toward the default voice [PROVED]

Let $\hat e$, $e_{\rm ref}$, $e_0$ be unit-norm embeddings (Resemblyzer normalizes) of the generated speech, the reference and the
default voice. Then
$$
m=\mathrm{sim}_{\rm ref}-\mathrm{sim}_{\rm default}=\langle\hat e,e_{\rm ref}\rangle-\langle\hat e,e_0\rangle=\langle \hat e,\Delta\rangle,\qquad \Delta=e_{\rm ref}-e_0 .
$$
Define the shrinkage along the axis $\rho=1-\langle \hat e-e_0,\Delta\rangle/\|\Delta\|^2$. Since
$\langle e_0,\Delta\rangle=\langle e_0,e_{\rm ref}\rangle-1=-\|\Delta\|^2/2$,
$$
\boxed{\,m=\big(\tfrac12-\rho\big)\,\|\Delta\|^2\,}
$$
Equivalently, $m=\tfrac12\big(\|\hat e-e_0\|^2-\|\hat e-e_{\rm ref}\|^2\big)$. Consequences:

1. Any error orthogonal to $\Delta$ (wrong timbre in a direction that does not separate the reference from the default) **does not change** $m$.
   The copy rate measures shrinkage toward the default voice, not the total rendering error.
2. "Copied" $\iff\rho<\tfrac12$: the generated voice has passed the midpoint between the default and the reference.

With sentence noise $\eta$ (H4) and $\hat e=\bar e+\eta$: $m=(\tfrac12-\rho)\|\Delta\|^2+\langle\eta,\Delta\rangle$. If $\eta$ is isotropic Gaussian,
$$
\Pr(\text{copied}\mid\text{voice})=\Phi\!\Big(\frac{(1-2\rho)\,\|\Delta\|}{2\sigma_\eta}\Big). \tag{2.1}
$$
The dependence on distance to the default voice comes from here: with the same $\rho<\tfrac12$, voices far from the default are copied with
more certainty. Voices very close to the default ($\|\Delta\|\lesssim2\sigma_\eta$) stay near 50% even with perfect copying
($\rho=0$): this is the **metric floor**, separate from the model floor.

**Training-free measurement this enables.** For each voice, $\rho_s=\tfrac12-\mu_s/\|\Delta_s\|^2$, with $\mu_s$ the mean margin (already measured)
and $\|\Delta_s\|^2=2(1-\cos(e_{\rm ref},e_0))$ (it suffices to embed the references and the default voice). If $\cos(e_{\rm ref},e_0)\approx0.68$
($\|\Delta\|^2\approx0.65$; assumed value, to be measured), the mean shrinkage is $\bar\rho\approx0.51/0.43/0.38$ for $N=8/32/128$ [HEURISTIC].

### 2.3 Proposition 1: continuous → binary link [PROVED given H; DATA]

Let $\mu_s=\mathbb E[m\mid s]$ and $m=\mu_s+w$, $w\sim\mathcal N(0,\sigma_w^2)$ within the voice. Then the non-copied fraction is
$$
q=\mathbb E_s\big[\Phi(-\mu_s/\sigma_w)\big]. \tag{2.2}
$$
If in addition $\mu_s\sim\mathcal N(\bar\mu,\tau_\mu^2)$, $q=\Phi\big(-\bar\mu/\sqrt{\tau_\mu^2+\sigma_w^2}\big)$.

Check with the measured per-voice margins and $\sigma_w=0.036$ (`sim_6_previsoes.py`):

| $N$ | observed $q$ | (2.2) with measured $\mu_s$ | Gaussian probit |
|---|---|---|---|
| 8 | 0.434 | 0.442 | 0.514 |
| 32 | 0.302 | 0.304 | 0.289 |
| 128 | 0.211 | 0.217 | 0.206 |

Form (2.2) with the per-voice margins is right to within 1 point; the Gaussian version misses at $N=8$ because the margins
are skewed ($-0.68$). **All-or-nothing**: the fraction of voices with copying between 10% and 90% is
$\Pr(|\mu_s|<1.28\,\sigma_w)\approx 2\cdot1.28\,\sigma_w\, g(0)$, with $g$ the density of the per-voice margins; with $\tau_\mu/\sigma_w=2.5$
this gives about 0.35–0.40, and I measured 0.34/0.37/0.26.

---

## 3. Regime (a): parametric / hierarchical (linear random effects)

### 3.1 Theorem 1 [PROVED]

Assumptions: code $u_s\sim\mathcal N(0,I_d)$, output $y_{sj}=Wu_s+b_s+\varepsilon_{sj}\in\mathbb R^D$ (H2), known intercept
(fixed by the $M$ default-voice sentences), $W$ estimated by least squares on the per-speaker means
$\bar y_s=Wu_s+e_s$, $e_s\sim\mathcal N(0,s^2(K)I_D)$, $N>d+1$. For a new voice $u^\*\sim\mathcal N(0,I_d)$:
$$
\frac1D\,\mathbb E\|\hat Wu^\*-(Wu^\*+b^\*)\|^2=\sigma_b^2+s^2(K)\,\frac{d}{N-d-1}. \tag{3.1}
$$
*Proof.* $\hat W-W=E^\top U(U^\top U)^{-1}$, with $E$ independent of $U$; hence
$\mathbb E\|(\hat W-W)u^\*\|^2=Ds^2\,\mathbb E[u^{\*\top}(U^\top U)^{-1}u^\*]=Ds^2\,\mathrm{tr}\,\mathbb E(U^\top U)^{-1}=Ds^2\,d/(N-d-1)$
(mean of the inverse Wishart). The term $b^\*$ is unpredictable and adds $D\sigma_b^2$. $\square$

Expanding, for $N\gg d$:
$$
E_{\rm new}\approx\underbrace{\sigma_b^2}_{a}+\underbrace{\frac{d\,\sigma_b^2}{N}}_{A/N}+\underbrace{\frac{d\,\sigma^2}{NK}}_{B/(NK)},\qquad \frac{A}{a}=d,\quad \frac BA=K_0\equiv\frac{\sigma^2}{\sigma_b^2}. \tag{3.2}
$$
The $A/N$ term exists only because there is variation **between speakers** that the code does not explain ($\sigma_b^2>0$): it is the
cluster noise of two-stage sampling. If $\sigma_b=0$, only $d\sigma^2/(NK)$ remains (the form of Du et al. 2021).

**Trained voices** (BLUP: per-speaker table shrunk toward $\hat Wu_s$, with $\lambda_K=\sigma_b^2/s^2(K)$) [HEURISTIC: I neglect $O(d/N^2)$]:
$$
E_{\rm trained}\approx\frac{\sigma_b^2\,\sigma^2/K}{\sigma_b^2+\sigma^2/K}+(1-\lambda_K)^2 s^2(K)\frac dN\ \xrightarrow{K\gg K_0}\ \frac{\sigma^2}{K}+O\!\Big(\frac{d}{NK^2}\Big). \tag{3.3}
$$

**Encoder floor**: with a noisy code $c=u^\*+\xi$, $\|W\|^2\sigma_\xi^2/R$ is added: **in this regime, averaging $R$
references must help** [PROVED].

**Simulation** (`sim_1_linear_gauss.py`, $d=4$, $D=24$, $\sigma_b^2=0.05$, $\sigma^2=0.5$, 400 replicates) [SIMULATED]:
(3.1) matches the simulation in all 24 cells with $N\ge16$ (e.g. $N=32,K=8$: 0.0667 against 0.0667); at $N=8$ the
difference reaches 8% (tail of the inverse Wishart with $N-d-1=3$). (3.3) matches in all cells. Fitting
$a+A/N+B/(NK)$ to the simulated data recovers $a=0.049$ (true 0.050), $K_0=8.9$ (10) and $\hat d=A/a=5.3$ (4):
the bias in $\hat d$ comes from approximating $d/(N-d-1)$ by $d/N$. Slope of the excess: $-1.09$ and $-1.11$ (predicted $-1$). Averaging
references: $R=1,4,16$ gives 0.358/0.129/0.071 against 0.352/0.127/0.071 predicted.

### 3.2 Reading the data in this regime [HEURISTIC]

If $q$ were proportional to the error, the grid fit ($a=0.39$, $A=1.08$, $B=10.3$) would give $d\approx A/a\approx2.8$ and
$K_0\approx9.5$. Three things weigh against this regime being the bottleneck: (i) averaging references does not help on the 38 voices
(regime (a) requires that it help); (ii) the measured exponent is well below 1; (iii) regime (a) gives nearly
uniform shrinkage across voices (attenuation by error in the code, $\rho_a\approx\sigma_\xi^2/(1+\sigma_\xi^2)$), and does not explain the dependence
on distance to the trained neighbor.

---

## 4. Regime (b): coverage (the decoder interpolates between trained voices)

### 4.1 Known rates [known from the literature; SIMULATED]

If the rendering $g$ is $L$-Lipschitz (or $s$-Hölder) in a space of intrinsic dimension $d$ and the decoder
renders the new voice from the neighboring trained voices:

- nearest neighbor (no noise): $E\asymp L^2\,\mathbb E\,r_{\rm nn}^2\asymp N^{-2/d}$;
  with noise, $E\to s^2(K)+O(N^{-2/d})$: **$K$ enters only in the floor** (the neighbor does no averaging);
- kernel smoothing with optimal bandwidth: $E-\sigma_b^2\asymp\big(s^2(K)/N\big)^{2s/(2s+d)}$, $s=2$ for a symmetric kernel and smooth $g$.

`sim_2_naoparam.py` ($v\sim U[0,1]^d$, $g=\sin(Av+c)$): nearest neighbor, slopes $-1.01/-0.53/-0.23$ against
$-1/-0.5/-0.25$ ($d=2,4,8$); Nadaraya–Watson with optimal bandwidth, $-0.69/-0.51/-0.28$ against $-0.67/-0.50/-0.33$.
In the bias-limited regime, **$N$ dominates and $K$ hardly matters**: $N=64,K=128$ (error 0.079) is worse than
$N=256,K=8$ (0.043), even though it has smaller $N/s^2$. The exponent $0.26$ would correspond to $d\approx8$ (neighbor, $2/d$) or
$d\approx11$ (kernel, $4/(4+d)$), if $q$ were proportional to the error, which Lemma 1 refutes (section 7).

### 4.2 Proposition 2: the attractor in a per-sentence smoother [PROVED for Nadaraya–Watson]

Decoder model [CONJECTURE]: the voice rendered for code $c$ is a weighted average **per training sentence**
of the voices the decoder has heard, with kernel $k(c,v)=e^{-\|c-v\|^2/2\ell^2}$:
$$
\hat y(c)=\frac{M\,k(c,v_0)\,v_0+K\sum_i k(c,v_i)\,\bar y_i}{M\,k(c,v_0)+K\sum_i k(c,v_i)}. \tag{4.1}
$$
The weight of the default voice is $\omega_0(c)=Mk(c,v_0)/Z(c)$. In the limit where the local mean of the trained voices is the
requested voice itself, $\hat y-v_0=(1-\omega_0)(v^\*-v_0)$, and by Lemma 1 $\rho=\omega_0$. Therefore
$$
\boxed{\ \text{copied}\iff K\sum_i k(v^\*,v_i)>M\,k(v^\*,v_0)\ } \tag{4.2}
$$
"the local mass of training sentences around the voice must exceed the mass of the default voice as seen from that point".
Outside this limit, the bias of the local mean is added (it pulls toward the center of the population, not toward $v_0$).

**Poisson regime (few speakers)**: the sum is dominated by the nearest neighbor, $\sum_i k\approx e^{-r_{\rm nn}^2/2\ell^2}$, and (4.2) becomes
$$
\boxed{\ \text{copied}\iff \|v^\*-v_0\|^2-r_{\rm nn}(v^\*)^2>2\ell^2\ln\frac MK\ } \tag{4.3}
$$
The voice must be closer to some trained voice than to the default voice, with a handicap $2\ell^2\ln(M/K)$.
This gives, at once, the positive correlation with distance to the default and the negative one with distance to the trained neighbor
(F4), and explains why $K$ matters little for new voices: it enters only in the logarithm. It also explains F5 (ECAPA): a better
code does not change $r_{\rm nn}$; with 8 speakers the decoder only renders voices near the 8 it has heard ("memorizes").

**Law-of-large-numbers regime (many speakers)**: $\sum_i k(v,v_i)\approx N\,c_\ell\,\tilde p(v)$, with $\tilde p=p*k$ the
smoothed density. Then (4.2) depends on $(N,K,M)$ **only through $NK/M$**:
$$
\text{copied}\iff \frac{NK}{M}\,c_\ell\,\tilde p(v^\*)\,e^{\|v^\*-v_0\|^2/2\ell^2}>1 . \tag{4.4}
$$
The failure region has two parts: a ball around $v_0$ of radius$^2\approx2\ell^2\ln\!\big(M/(NKc_\ell\tilde p)\big)$ (voices
captured by the attractor) and the tail of the population where $\tilde p$ is small (atypical voices, coverage holes). Both
shrink as the **logarithm** of $NK/M$ [HEURISTIC].

**Transition between the two** [HEURISTIC]: the expected number of trained voices within scale $\ell$ of a new
Gaussian voice is $N\prod_j(1+2\lambda_j/\ell^2)^{-1/2}$. The dependence on the nearest neighbor disappears when this exceeds ~1:
$$
N^\*=\prod_j\Big(1+\frac{2\lambda_j}{\ell^2}\Big)^{1/2}. \tag{4.5}
$$
The data (the correlation with the neighbor disappears between $N=32$ and 128) suggest $N^\*\sim50$–$100$; for isotropic voices this is
$d=6$, $\ell=0.75\tau$ ($N^\*=95$) or $d=8$, $\ell=\tau$ (81) (`sim_9_piso_e_Nstar.py`). The spectral model that best
fits $q$ (section 6) has $N^\*\approx860$: tension discussed in section 12.

### 4.3 Gaussian process version [HEURISTIC; SIMULATED]

Fine-tuning from a single-speaker checkpoint = regression of the residual $r(v)=g(v)-v_0$ with **prior mean $v_0$**
(Gaussian process or tangent kernel) and an almost noise-free observation at $v_0$. In the well-sampled region,
$\rho(v)\approx\tilde s^2/(\tilde s^2+n_{\rm loc}(v))$, $\tilde s^2=s^2(K)/a^2$: the effect of $K$ saturates at $\sigma_b^2$. Simulated (`sim_3_atrator.py gp`),
this version gives a drop that is too fast ($q\approx0.54\to0.26\to0.10$) when the all-or-nothing is required; the per-sentence version (4.1)
does better (section 6).

---

## 5. Regime (c): spectral (power-law eigenvalues)

### 5.1 Proposition 3 [HEURISTIC; standard kernel regression results, Caponnetto–De Vito, Bordelon et al. 2020]

Voices with covariance $\Sigma_v=\mathrm{diag}(\lambda_j)$, $\lambda_j=j^{-\alpha}$, $\alpha>1$; the decoder learns the map
(target: identity) by ridge from $N$ voices with noise $s^2(K)$. A direction $j$ is learned when
$N\lambda_j\gtrsim\kappa$ ($\kappa$: effective regularization $\sim s^2$). The number of learned directions is $J(N)\asymp(N/\kappa)^{1/\alpha}$ and
$$
E_{\rm new}\asymp\sum_{j>J(N)}\lambda_j+\kappa\frac{J(N)}N\asymp\Big(\frac{s^2(K)}{N}\Big)^{(\alpha-1)/\alpha};\qquad
\text{no noise (interpolation)}:\ E\asymp N^{-(\alpha-1)}. \tag{5.1}
$$
`sim_7_espectral.py` ($D=1500$): with noise and optimal ridge, slopes $-0.28/-0.42/-0.59$ against $-0.33/-0.50/-0.67$
($\alpha=1.5/2/3$); without noise, $-0.85/-1.25/-2.12$ against $-0.5/-1/-2$ (finite-$D$ effect as $N\to D$). Agreement
of 0.05–0.1 in the exponent, with deviation in the direction expected from finite-size effects [SIMULATED, approximate].

### 5.2 The spectrum measured in the latent [DATA from the previous attempt, `out/speaker.json`]

Between-speaker covariance of the codes (trainings with $N=128,200$): participation ratio 9–10 (SNR-weighted) and 6
(raw); 13–14 directions with SNR > 1; 6 directions explain 90% of the variance; decay exponent
$\alpha_{\rm raw}\approx2.0$–$2.1$ and $\alpha_{\rm snr}\approx1.05$–$1.1$. By (5.1), $\beta_E=(\alpha-1)/\alpha$ lies between $0.05$ ($\alpha=1.05$) and
$0.5$ ($\alpha=2$): the observed 0.26 is inside, but the interval is too wide to be a test. Caveat: this is the spectrum
of the learned latent, not that of the metric space; the right thing is to measure the spectrum of the Resemblyzer embeddings of the 2,456
LibriTTS voices (no training; section 10).

### 5.3 Unifying (c) with the parametric floor [HEURISTIC]

If the decoder renders the $J$ learned directions well and nothing in the others, Lemma 1 gives
$\rho(v)=\sum_{j>J}\Delta_j^2/\|\Delta\|^2$ when $v_0-\bar v$ lies in the learned directions, and
$$
q(N)\approx\Pr\Big(\textstyle\sum_{j>J(N)}(v-\bar v)_j^2>\sum_{j\le J(N)}(v-v_0)_j^2\Big)=q_{\rm floor}\big(J(N)\big). \tag{5.2}
$$
This is the same function as the floor of the parametric module (section 9.4), evaluated at the number of directions the data resolve.

---

## 6. The default-voice attractor and the model that puts it all together

### 6.1 Why the default voice is an attractor [modeling CONJECTURE, consequences PROVED in (4.1)]

Two routes, both present in our training: (i) **prior mean**: the starting checkpoint renders $v_0$ for any
code, and fine-tuning is short (2,500 steps), so where the data do not reach, the output stays near $v_0$; (ii) **heavy
anchor**: the $M=2000$ LJ sentences are the largest single mass in training (from 98% of the total in $8\times8$ to 33% in
$128\times32$). In (4.1) both become the same term $Mk(c,v_0)$.

Pitch enters through $\Delta$: the default is female and high-pitched, so high-pitched voices have smaller $\|\Delta\|$ and (i) fall more easily into
the capture ball of (4.4), (ii) have a smaller margin by (2.1) even when copied. Both effects give a negative correlation
between copying and F0 and a positive one with distance to the default (F4).

### 6.2 Fitting and testing the coverage + attractor model [SIMULATED]

`sim_3_atrator.py`: voices $v\sim\mathcal N(0,\mathrm{diag}(j^{-\alpha}))$ in $\mathbb R^d$, coordinate 0 = pitch, $v_0=\delta e_0$; decoder (4.1)
or Gaussian process; codes with noise $\sigma_\xi=0.2$, sentences with noise $\sigma_\eta$; 25 sentences per voice; Lemma 1 criterion.

**Negative result first.** With isotropic voices ($\alpha=0$, $d=4$–16), none of the 240 configurations fits the
three points while keeping the all-or-nothing: the best misses by 0.07 (RMSE) and decays with exponent ~0.45–0.5
(`out/sim_3_grade.txt`). The slow exponent is obtained only with large sentence noise ($\sigma_\eta\ge1$), which destroys the all-or-nothing
(the fraction of voices between 10% and 90% rises to 0.9). **A low-dimensional isotropic voice space does not explain F1+F4 together.**

With a power-law spectrum ($\alpha\in\{1,2\}$, $d\in\{16,32\}$), the best configuration of the per-sentence smoother is
$\alpha=1$, $d=16$, $\ell=0.5$, $\delta=2$, $\sigma_\eta=0.1$ (RMSE 0.017 on the three points; `out/sim_3_grade_espectro.txt`).
The $\alpha=1$ chosen by the grid coincides with the $\alpha_{\rm snr}\approx1.05$–$1.1$ measured in the latent, which was not imposed.
Running this configuration with 6 seeds (`out/sim_3.txt`):

| Quantity | Model ($d=16$) | Model ($d=32$, $\ell=0.3$) | Observed |
|---|---|---|---|
| $q$ at $N=8/32/128$ | 0.427 / 0.292 / 0.212 | 0.418 / 0.297 / 0.207 | 0.434 / 0.302 / 0.211 |
| $q$ at $N=247/1000/2456/10^4$ | 0.177 / 0.115 / 0.086 / 0.062 | 0.173 / 0.126 / 0.102 / 0.072 | — |
| local exponent | 0.23–0.31, no floor | 0.23–0.27 | ~0.26 |
| $\tau_\mu/\sigma_w$ (all-or-nothing) | 2.5 / 2.7 / 2.8 | 2.1 | ~2.5 |
| voices with copying between 10% and 90% | 0.27 / 0.22 / 0.20 | 0.34 / 0.29 / 0.27 | 0.34 / 0.37 / 0.26 |
| ranks copying × distance to default | +0.77 / +0.85 / +0.81 | +0.72 / +0.78 / +0.78 | +0.47 / +0.58 / +0.53 |
| ranks copying × pitch | −0.83 / −0.89 / −0.85 | −0.79 / −0.83 / −0.83 | −0.53 / −0.52 / −0.37 |
| ranks copying × distance to trained neighbor | −0.25 / −0.18 / −0.15 | −0.35 / −0.30 / −0.21 | −0.33 / −0.40 / +0.13 |
| ranks of copying across $N$ (8–32, 32–128, 8–128) | 0.86 / 0.91 / 0.80 | 0.82 / 0.84 / 0.76 | 0.75 / 0.74 / 0.59 |
| corr(gain 8→128, margin at 8) | −0.40 | −0.55 | −0.55 |

What the model gets right without having been fitted for it: the all-or-nothing, the signs of the three per-voice correlations, the
stability of difficulty across $N$, the "worst gain the most". What it gets wrong: the correlations with distance and pitch are
too strong (in the model pitch is the only axis of the default and there is no measurement noise in the covariates; in the data, F0 and
distance in the latent are noisy measurements of the right coordinate), and the correlation with the neighbor does not change sign at $N=128$
(in the data $+0.13$, with standard error ~0.16 over 38 voices: the difference is not significant).

On the $N\times K$ grid of the 4 fixed voices, the same model (without refitting) predicts the order of the 10 trainings with correlation
0.97 and RMSE 0.041 after a line $q_{\rm obs}=0.11+1.19\,q_{\rm model}$ (the 4 voices are harder than average;
`out/sim_6.txt`). The RMSE is below the seed noise (~0.05–0.08).

### 6.3 What the model says about the negatives (F5) [SIMULATED; `sim_4_variantes.py`]

- **Averaging references**: $R=1,4,16$ gives $q=0.437/0.439/0.438$ ($N=8$) and $0.213/0.222/0.223$ ($N=128$). No effect, as on the
  38 voices: the bottleneck is $\rho$ (what the decoder knows how to render), not the code noise. The gains on the 4 grid voices
  come from voices at the threshold (section 12).
- **Pseudo-speakers**: if the speed-altered copies were voices from the same population shifted along the pitch axis,
  the model predicts a large gain ($N=8$: $0.437\to0.33$, close to $5N$ real voices, $0.28$). The data say they do not count.
  **Tension**: either the pseudo-voices fall outside the distribution of the test voices (in the metric space), or the coverage
  model is wrong. Training-free test in section 10 (P9).
- **ECAPA, reference swapping, contrastive loss**: they change the code, not $n_{\rm loc}$ or $M$; (4.2) does not change.

---

## 7. From continuous error to binary fraction: when a power law comes out

By (2.2), $q$ is an average of probits. Three cases:

**(i) Uniform shift** [PROVED given $\mu_s=\bar\mu(N)+\delta_s$, $\delta_s\sim\mathcal N(0,\tau^2)$]: $q=\Phi(-\bar\mu(N)/\sigma_T)$,
$\sigma_T^2=\tau^2+\sigma_w^2$. The local exponent is
$$
\beta_{\rm loc}(N)=-\frac{d\ln q}{d\ln N}=\frac{\varphi(z)}{\Phi(z)}\cdot\frac{1}{\sigma_T}\frac{d\bar\mu}{d\ln N},\qquad z=-\bar\mu/\sigma_T . \tag{7.1}
$$
With the data ($d\bar\mu/d\ln N=0.037$ between 8 and 32 and 0.022 between 32 and 128; $\sigma_T\approx0.097$): $\beta_{\rm loc}\approx0.39$ and $0.29$.
This matches the 0.26 up to noise. If $\bar\mu(N)=\mu_\infty-AN^{-\gamma}$, $q\to\Phi(-\mu_\infty/\sigma_T)>0$: **a power-law error with a
finite limit gives $q$ with a floor, not a power law**. $q$ is a pure power only if $\bar\mu$ grows without bound as
$\sqrt{2\beta\ln N}\,\sigma_T$ (then $q\approx N^{-\beta}/\sqrt{4\pi\beta\ln N}$) [PROVED, Mills asymptotics].

**(ii) Per-voice threshold ("quanta" view)** [PROVED given the assumption]: if each voice has a threshold $N_v$ and is copied for
$N>N_v$, then $q(N)=\Pr(N_V>N)$. Power law $\iff$ $N_V$ has a Pareto tail. In (4.4), $N_v=M/(Kc_\ell\tilde p(v)e^{\|v-v_0\|^2/2\ell^2})$:
the exponent is a property of the **distribution of the local densities of the voices**, not of the decoder. For isotropic Gaussian voices
this gives $\chi^2_d$-type tails (exponent rising to ~1 with $N$); for a $j^{-1}$ spectrum the simulation gives an
exponent ~0.25 constant over three decades (section 6.2). I have no proof that $\alpha=1$ gives an exact power; it is a
simulation fact [SIMULATED; explanation CONJECTURE: as resolution improves, new directions of the spectrum appear,
and the effective dimension $d_{\rm eff}(r)=r^{-2/\alpha}$ grows, compensating for the thinning of the tail].

**(iii) All-or-nothing** is the regime $\tau_\mu\gg\sigma_w$ of (2.2): every voice is far from the threshold except a fraction
$\approx2.6\,\sigma_w g(0)$. It comes from $\rho_v$ being a deterministic function of the voice's position (4.2) and from $\sigma_w$ being small.

---

## 8. Unified scaling function, collapse, convergence, minimum $N$, $N/K$ split

### 8.1 Scaling function [PROVED as an identity given Lemma 1 and (2.2); the form of $\rho$ depends on the regime]

$$
\boxed{\ q(N,K,M)=\mathbb E_{v\sim P}\ \Phi\!\Big(-\frac{\big(\tfrac12-\rho_{N,K,M}(v)\big)\|\Delta(v)\|^2}{\sigma_w}\Big)\ } \tag{8.1}
$$

| Regime | $\rho_{N,K,M}(v)$ | Collapse variable |
|---|---|---|
| (a) parametric | $\rho_\infty+O\big(d\,s^2(K)/N\big)$, nearly uniform | $N/(1+K_0/K)$ |
| (b) Poisson ($N<N^\*$) | $\big[1+\tfrac KM\sum_i e^{(\|\Delta\|^2-\|v-v_i\|^2)/2\ell^2}\big]^{-1}$ | $N$ (and $K/M$ only in the log) |
| (b) LLN ($N>N^\*$) | $\big[1+\tfrac{NK}{M}c_\ell\tilde p(v)e^{\|\Delta\|^2/2\ell^2}\big]^{-1}$ | $NK/M$ |
| (c) spectral | $\sum_{j>J(N)}\Delta_j^2/\|\Delta\|^2$, $J\propto(N/s^2(K))^{1/\alpha}$ | $N/(1+K_0/K)$ |

**Collapse test on the 16 trainings** (`sim_8_colapso.py`; $q\sim a+b\ln X+c\ln^2X$; seed noise ~0.06):
$X=N$: RMSE 0.074; $X=N/(1+K_0/K)$: 0.067 ($K_0=5.8$); $X=NK$: 0.137; mixture $N^{0.83}(NK)^{0.17}$: 0.068.
Only with $N\ge32$: $X=N$ gives 0.065 and $X=NK$ 0.049. Reading [HEURISTIC]: below ~32 speakers only $N$ counts (Poisson);
above, $K$ starts to count and the variable $NK$ improves, as (4.4) predicts. With 4 voices per training this is weak evidence.

**Collapse prediction for new trainings**: in the LLN regime, $q$ is a function of $NK/M$. In particular, in the model,
$(K=64,M=2000)$ and $(K=32,M=1000)$ give the same $q$ (0.267 and 0.268 at $N=32$; 0.185 and 0.185 at $N=128$), and
$(K=128,M=2000)\approx(K=32,M=500)$ (0.238/0.240; 0.156/0.157) [SIMULATED].

### 8.2 Convergence

- $K\to\infty$, $N$ fixed: (a) $q\to$ a function of $a+A/N$ (cluster floor); (b) the handicap $2\ell^2\ln(M/K)$ in (4.3)
  falls and becomes negative for $K>M$, but $\Pr\big(r_{\rm nn}(v)>\|\Delta(v)\|\big)$ remains: **more sentences do not cover holes**. This is F1/F2.
- $N\to\infty$, $K$ fixed: (a) $q\to$ the $\Phi$-floor from $\sigma_b$ and from misspecification error; (b) $q\to0$ as $\Pr(N_V>N)$ (logarithmic to
  power, depending on the distribution of the densities); only the metric floor $\Pr(\|\Delta\|\lesssim2\sigma_\eta)$ of the voices
  indistinguishable from the default remains.

### 8.3 Minimum $N$ for 90% and 95% copying ($K=32$, 38 voices; `sim_6_previsoes.py`)

| Law (fitted to the 3 points) | $q(247)$ | $q(1000)$ | $q(2456)$ | $N_{90\%}$ | $N_{95\%}$ |
|---|---|---|---|---|---|
| power $0.746\,N^{-0.26}$ | 0.178 | 0.123 | 0.098 | $2.2\times10^3$ | $3.2\times10^4$ |
| probit linear in $\ln N$: $\Phi(0.31-0.23\ln N)$ | 0.166 | 0.097 | 0.066 | $9\times10^2$ | $4.5\times10^3$ |
| $a+A/\sqrt N$ ($a=0.145$) | 0.198 | 0.171 | 0.162 | never | never |
| $a+A/N$ ($a=0.22$) | 0.226 | 0.221 | 0.220 | never | never |
| coverage + attractor (section 6, $d=16$) | 0.177 | 0.115 | 0.086 | $\approx1.7\times10^3$ | $\approx2\times10^4$ |
| mean margin with floor (`sim_5_ajuste_previsoes.py`, $1/\sqrt N$) | 0.200 | 0.174 | 0.165 | never | never |

**With all of LibriTTS (2,456 speakers) the best case is ~90% and the laws with a floor give ~84%.** 95% requires $10^4$ or more
speakers under any law without a floor, or a change of regime (section 9).

### 8.4 Optimal split of a budget $NK=B$ [HEURISTIC; SIMULATED]

- (a): $q\approx a+A/N+B'/B$; with $B$ fixed only $N$ counts $\Rightarrow$ minimum $K$.
- Law $a+A/\sqrt N+C/\sqrt K$: $K^\*=(C/A)\sqrt B$; with $C/A=0.42$: $K^\*\approx27$ for $B=4096$, $54$ for $B=16384$.
- (b) LLN: $q$ depends on $NK/M$ $\Rightarrow$ flat in $K$; in the model, $B=4096$: $q=0.205$ ($K=2$) to $0.246$ ($K=128$);
  $B=16384$: 0.144 to 0.169. (b) Poisson: prefers $N$. **Recommendation**: $K$ between 8 and 32 and the largest possible $N$; $K$ below 8
  only if the encoder still separates speaker from sentence (not modeled here) and if the trained voices do not matter (they want $K$, F2).

---

## 9. The two-stage proposal

Stage A: text → content units $u$, trained with $N_{\rm text}$ transcribed speakers. Stage B: $(u,z)\mapsto$ mel,
trained by reconstruction with $N_{\rm voice}$ untranscribed speakers ($K_{\rm voice}$ sentences each; $M_B$ sentences from any dominant
voice that enters B).

### 9.1 Proposition 4 (separation) [PROVED in the model of sections 2–6, given assumptions S1–S3]

- S1: the units carry no voice: $u\perp v\mid$ content.
- S2: the distribution of the units predicted by A at inference is the same as that of the units extracted when training B.
- S3: B's voice code sees only audio (as today).

Then rendering the voice in B is the same problem as in sections 4–6 with $(N,K,M)\to(N_{\rm voice},K_{\rm voice},M_B)$, and
$$
q_{2}(N_{\rm text},N_{\rm voice})=q_1(N=N_{\rm voice},K=K_{\rm voice},M=M_B),\qquad
\mathrm{WER}_2\approx\mathrm{WER}_A(N_{\rm text},\text{transcribed hours})+\epsilon_B(N_{\rm voice}K_{\rm voice}). \tag{9.1}
$$
Copying does not depend on $N_{\rm text}$; WER does not depend on $N_{\rm voice}$ except through the interaction term $\epsilon_B$, which is now
learned from all the untranscribed audio. **Extra gain**: if B is not dominated by one voice ($M_B\approx K_{\rm voice}$), the attractor disappears.
In the model, $q_2=0.25/0.16/0.10/0.06$ for $N_{\rm voice}=8/32/128/1000$, against $0.41/0.30/0.22/0.12$ in one stage with
$M=2000$ [SIMULATED, `sim_4_variantes.py` (6)].

### 9.2 Voice leakage into the units [PROVED for linear Gaussian channels; SIMULATED]

If the units leak voice ($u$ contains $v+\xi_\ell$ with per-dimension SNR $\mathrm{SNR}_\ell$) and the code has $\mathrm{SNR}_c$, the
decoder that minimizes reconstruction combines the two channels with precision weights. At inference the leaked channel
says "the voice of A's speakers" ($v_A$, close to the default), so
$$
\hat y=(1-w)\,\hat y_{\rm code}+w\,v_A,\qquad w=\frac{\mathrm{SNR}_\ell}{\mathrm{SNR}_\ell+\mathrm{SNR}_c}, \tag{9.2}
$$
and the shrinkage becomes $\rho_2=w+(1-w)\rho_B$. Copied requires $(1-w)(1-\rho_B)>\tfrac12$. Simulation (floor as $N_{\rm voice}\to\infty$):

| $w$ | 0 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|
| $q_2(N_{\rm voice}=8/128/2456)$ | 0.25/0.10/0.05 | 0.33/0.14/0.09 | 0.47/0.21/0.15 | 0.67/0.37/0.31 | 0.86/0.67/0.67 |

In bits, with $\mathrm{SNR}=2^{2I}-1$ per effective dimension: $w\le0.1\iff I_\ell\le\tfrac12\log_2\!\big(1+(2^{2I_c}-1)/9\big)$. With the current
code (~16 bits over ~8 effective dimensions, $I_c\approx2$ bits/dim) this gives $I_\ell\lesssim0.7$ bit/dim, or ~6 bits in total:
**the units can carry at most about one third of the code's speaker information** [HEURISTIC in the conversion to bits].
Measurable before training B: the same Gaussian information estimator used on the latent, applied to the units.
Remedy: train B with units extracted from audio with perturbed voice (the leakage stops being informative, $\mathrm{SNR}_\ell^{\rm eff}\to0$).

### 9.3 Cross-language transfer [PROVED in the threshold model; the rest HEURISTIC]

The failure region $F_N$ of (4.2) depends only on B's training. For voices from another population (Portuguese),
$q_{\rm PT}(N)=\int p_{\rm PT}\mathbf 1_{F_N}\le\big\|p_{\rm PT}/p_{\rm EN}\big\|_\infty\,q_{\rm EN}(N)$: coverage transfers to the extent that the Portuguese
voices fall inside the distribution of the English ones. Additional conditions: (i) unit inventory: the fraction
$\pi_{\rm miss}$ of Portuguese frames with units that are rare or absent in B's training enters as rendering error
(and WER) proportional to $\pi_{\rm miss}$; multilingual units reduce $\pi_{\rm miss}$; (ii) language-linked accent and prosody cannot
live in the voice code, otherwise the copied voice brings the English accent; (iii) S2 (mismatch between predicted and
extracted units) is language-independent. Untranscribed Portuguese audio (abundant) in B solves (i) and the density ratio.

### 9.4 Low-dimensional parametric voice module: exponent and price [floor PROVED in the model; exponent HEURISTIC]

If B conditions the voice through a family of dimension $k$ (affine, "eigenvoices" style: $\hat y=\hat\mu+\hat P_k(c-\hat\mu)$), then:

- **Floor** (Lemma 1 applied to $\hat y=\mu+P_k(v-\mu)$, with $v_0-\mu$ in the subspace):
  $$q_{\rm floor}(k)=\Pr\big(\|P_\perp(v-\mu)\|>\|P_k(v-v_0)\|\big). \tag{9.3}$$
  The voice is copied if the family captures more of its distance to the default than the part left out.
- **Excess**: the error in estimating $(\mu,P_k)$ is $O(k\,s^2/N)$ (Theorem 1), and the excess of $q$ over the floor is linear in this error
  (expansion of (2.2)), hence **exponent 1** asymptotically, against ~0.26 today.

Numbers from (9.3) with spectra that bracket the measured one (`sim_9_piso_e_Nstar.py`, $d=32$, default at $\delta$ deviations on the 1st axis):

| spectrum | $\delta$ | $k=4$ | $k=6$ | $k=8$ | $k=12$ |
|---|---|---|---|---|---|
| $\alpha=2$ (raw latent) | 1 | 0.045 | 0.010 | 0.002 | 0.000 |
| $\alpha=2$ | 2 | 0.011 | 0.002 | 0.000 | 0.000 |
| $\alpha=1.05$ (latent/SNR) | 1 | 0.374 | 0.218 | 0.114 | 0.022 |
| $\alpha=1.05$ | 2 | 0.133 | 0.069 | 0.033 | 0.006 |

Simulation (`sim_4_variantes.py` (5), `sim_5b_parametrico.py`; spectrum $j^{-1}$, $d=16$): without noise, $q$ converges to 0.075–0.086
($k=4$; (9.3) gives 0.086) and 0.009–0.012 ($k=8$; 0.009). With $k=8$: $q=0.069/0.036/0.025/0.018$ at $N=8/32/128/1000$, against
$0.43/0.29/0.21/0.12$ for the coverage decoder in the same world. The measured exponent of the excess lies between 0.5 and 1 and is
noisy (the excess is only a few points); "exponent 1" is the asymptotic value, not confirmed numerically with precision.
**Price**: if the real voice needs more dimensions than $k$, or if rendering is nonlinear in the code, the floor (9.3)
plus the nonlinear misspecification error remain forever. **Big caveat**: in the simulated world rendering is linear
(identity), which favors the parametric module by construction [CONJECTURE that it holds for Tacotron/B].

---

## 10. Falsifiable predictions

### 10.1 Without training (measurements)

| # | Measurement | Prediction and what it decides |
|---|---|---|
| P1 | Spectrum of the per-speaker mean Resemblyzer embeddings of the 2,456 LibriTTS voices (and intrinsic dimension by TwoNN). | The model of section 6 needs $\alpha\approx1$ in the metric space. If $\alpha\ge1.5$ and $d_{\rm int}\le8$, the spectral model is wrong and the slow exponent must come from something else (for example, from the metric floor). $\alpha$ also fixes the price (9.3) of a module of dimension $k$. |
| P2 | Per-voice $\rho_s=\tfrac12-\mu_s/\|\Delta_s\|^2$, with $\cos(e_{\rm ref},e_0)$ of the 38 references. | Bimodal distribution, far from 1/2, except for ~1/3 of the voices; $\bar\rho$ falls with $N$. Voices never copied should have $\rho_s>1/2$ with large $\|\Delta_s\|$; voices with $\|\Delta_s\|<2\sigma_\eta$ should stay near 50%, not 0% (metric floor). |
| P3 | Poisson rule (4.3) on the 38 speakers with the saved latents: predictor $\|\Delta\|^2-r_{\rm nn}^2$. | At $N=8,32$ it separates copied from non-copied better (AUC) than either distance alone; at $N=128$ the $r_{\rm nn}$ part loses value. |
| P9 | Distance (Resemblyzer) from each test voice to the nearest pseudo-voice (`-aug`) and to the nearest real trained voice. | If the pseudo-voices are closer to the test voices than the real ones and still did not help, the coverage model (4.1) is refuted. If they are farther (out of distribution), the negative result is explained. |
| P8 | Per-voice variance between seeds (same speakers). | Concentrated on voices with $|\mu_s|\lesssim\sigma_w$ in the other seed; with 4 voices, deviation of $q$ ~5–8 points; with 38, ~2 points. |

### 10.2 New single-stage trainings

| # | Training | Prediction (38 voices or more) | Decides |
|---|---|---|---|
| P4 | $N=247,1000,2456$, $K=32$ | Coverage: $q=0.18/0.12/0.09$ ($\pm0.03$); power: 0.18/0.12/0.10; laws with a floor: $\ge0.16$ at 2456. | $q(2456)<0.12$ rejects the laws with a floor; $>0.14$ rejects coverage and pure power. **Requires $\ge200$ evaluation voices** (standard error $\sqrt{q(1-q)/n_v}\approx0.02$); with 38, the CI is ±0.1 and does not decide. |
| P5 | $N=128$, $K=32$, with $M=1000$ and $M=500$ LJ sentences | $q$ falls as if $K$ doubled/quadrupled: 0.21 → 0.185 → 0.157 (model); the voices that gain are those near the default (gain correlates negatively with $\|\Delta\|$). | Direct test of the attractor: if $M$ does not matter, section 6 falls. |
| P6 | Budget $NK=4096$: $(512,8)$, $(128,32)$, $(32,128)$ | LLN coverage: differences $\le4$ points, small $K$ slightly better; square-root law: optimum at $K\approx27$; (a): small $K$ much better. | Distinguishes the regimes by the shape in $K$. |
| P10 | Voice latent 32 → 16, 8, 4 (E4 of the proposal) | Copying unchanged down to $k$ with small $q_{\rm floor}(k)$: $k\approx6$–8 if $\alpha\approx2$; $k\approx12$ if $\alpha\approx1$. | Links P1 to a training. |

### 10.3 Two stages ($N_{\rm text}=8$, sweeping $N_{\rm voice}$)

| # | Prediction | Fails if |
|---|---|---|
| P7a | $q_2(N_{\rm voice})\le q_1(N=N_{\rm voice})$ at each $N_{\rm voice}$ (without LJ dominating B, up to ~15 points better at $N_{\rm voice}\le128$). | $q_2$ follows $N_{\rm text}$ (stays at the single-stage $N=8$ level). |
| P7b | WER flat in $N_{\rm voice}$ (±2 points), at the level set by A. | WER rises with $N_{\rm voice}$ (content × voice interaction poorly learned). |
| P7c | Floor of $q_2$ given by (9.2) from $I(u;\text{speaker})$ measured beforehand: $w=0.2\Rightarrow q_2\gtrsim0.15$ for every $N_{\rm voice}$. | Floor different from the one predicted by the measured information. |
| P7d | Stage A in Portuguese: $q_{\rm PT}\le\|p_{\rm PT}/p_{\rm EN}\|_\infty\,q_{\rm EN}$; WER in Portuguese close to the current system (10%) if $\pi_{\rm miss}$ is small. | Copying in Portuguese much worse with Portuguese voices inside the English distribution. |

---

## 11. Verification by simulation: summary

| Claim | Script | Predicted | Simulated | Status |
|---|---|---|---|---|
| (3.1) new-voice error, regime (a) | `sim_1` | $\sigma_b^2+s^2d/(N-d-1)$ | equal to ~1% for $N\ge16$ | confirmed |
| excess $\propto1/N$ in (a) | `sim_1` | −1 | −1.09; −1.11 | confirmed |
| (3.3) trained voices | `sim_1` | formula | equal in all cells | confirmed |
| encoder floor $\sigma_\xi^2/R$ | `sim_1` | 0.352/0.127/0.071 | 0.358/0.129/0.071 | confirmed |
| nearest neighbor $N^{-2/d}$ | `sim_2` | −1/−0.5/−0.25 | −1.01/−0.53/−0.23 | confirmed |
| kernel $N^{-4/(4+d)}$ | `sim_2` | −0.67/−0.5/−0.33 | −0.69/−0.51/−0.28 | confirmed ($d=8$ pre-asymptotic) |
| spectral $(\alpha-1)/\alpha$ | `sim_7` | −0.33/−0.5/−0.67 | −0.28/−0.42/−0.59 | approximate |
| binary link (2.2) on the data | `sim_6` | 0.442/0.304/0.217 | obs. 0.434/0.302/0.211 | confirmed (DATA) |
| isotropic coverage explains F1+F4 | `sim_3` | — | no (RMSE ≥0.07 with all-or-nothing) | **refuted** |
| spectral coverage ($\alpha=1$) + attractor | `sim_3` | curve + signs | curve, all-or-nothing, signs; magnitudes too strong | partial |
| $N\times K$ grid (order) | `sim_6` | — | corr 0.97, RMSE 0.041 after line | confirmed |
| $q$ depends on $K/M$ | `sim_4` | equality | 0.267=0.268; 0.185=0.185 | confirmed (in the model) |
| averaging references does not help | `sim_4` | ~0 | ±0.01 | consistent with data |
| pseudo-speakers do not count | `sim_4` | — | model says they would count | **tension** |
| parametric floor (9.3) | `sim_4/5b` | 0.086; 0.009 | 0.075–0.086; 0.009–0.012 | confirmed |
| parametric excess exponent 1 | `sim_5b` | 1 | 0.5–1, noisy | not confirmed |
| leakage (9.2) | `sim_4` | floor increasing in $w$ | 0.05/0.09/0.15/0.31/0.67 | illustrates |

---

## 12. Critique: weak assumptions, what remains unexplained, status of each claim

### 12.1 Weakest assumptions

1. **Tacotron as a per-sentence smoother in the metric space** (4.1) is the assumption that carries almost everything and was not
   derived. A neural decoder may extrapolate linearly in some directions (regime (a)) and interpolate in others
   (regime (b)); the model picks just one. The Gaussian process version (prior mean $v_0$) does worse, which suggests that the
   **per-sentence** weight of the default voice matters; P5 tests this.
2. **Metric space = model space.** Lemma 1 is exact in the Resemblyzer space; the coverage assumptions are
   about the space where the decoder interpolates. If the two are not approximately isometric, the numbers in
   sections 4–6 change (the signs do not).
3. **Gaussian voices with a $j^{-1}$ spectrum.** Chosen on a grid of 2 values of $\alpha$; the agreement with the latent's $\alpha_{\rm snr}$ is
   suggestive, not a test. P1 decides.
4. **Fitting with little data.** The model of section 6 has 5 free parameters ($d,\ell,\delta,\sigma_\eta,\alpha$) chosen for 3 points.
   Fitting the curve tests nothing; the tests are the quantities that did not enter the choice (all-or-nothing, correlations,
   grid, $K$). And the magnitudes of the correlations with pitch and distance come out too strong.
5. **The exponent 0.26** comes from 3 points with a CI that includes 0 and 0.9 (section 1.2). Power, logarithm and floor cannot be distinguished
   up to $N\sim10^3$ with 38 voices.

### 12.2 What remains unexplained

- **Between-seed variance of 18 points**: explained as **one voice at the threshold that flips** (DATA: voice 6415, 1.0 → 0.52), which is
  consistent with (2.2) and with the all-or-nothing; but why the seed shifts $\rho$ for that voice by ~0.1 was not modeled (variance
  of the learned function due to initialization and data order). With fixed speakers, the model reproduces this only if one adds
  a per-seed function noise; I did not do this.
- **Averaging references**: the model explains the absence of effect on the 38 voices (the bottleneck is $\rho$, not $\sigma_\xi$), but not the gain
  of up to 20 points on the 4 grid voices; the only available explanation is that those voices were at the threshold (same
  mechanism as the seeds), which is post hoc.
- **Pseudo-speakers**: the model predicts they would count if they were voices from the population; they did not count. The way out "they are outside the
  distribution" (the latent of the `-aug` trainings concentrated the between-speaker variance: participation ratio 1.9 in
  $8\times8$-aug and 4.1 in $8\times128$-aug, against 4.0–4.6 in the normal trainings with $N=8$) is plausible and testable (P9), but not demonstrated.
- **Sign change of the correlation with the neighbor at $N=128$**: the fitted spectral model has $N^\*\approx860$ and does not change sign;
  an isotropic model with $d\approx6$–8 changes sign at ~100 but decays too fast. The measured value (+0.13) is ~1.7 standard errors from −0.15.
- **WER constant in $N$**: compatible with the decomposition (content in the long LJ line, voice in the code), but not derived.
- **Bits in the latent growing with $\log N$** (F6): compatible with a code that only resolves the voices the decoder knows how to
  render, but I did not derive the rate.

### 12.3 Status of each claim

| Claim | Status |
|---|---|
| Lemma 1 ($m=(\tfrac12-\rho)\|\Delta\|^2$) | PROVED (normalized embeddings) |
| Link (2.2) and all-or-nothing as $\tau_\mu\gg\sigma_w$ | PROVED given Gaussian noise; DATA agree |
| Theorem 1 / form $a+A/N+B/(NK)$ and meaning of $A/a=d$, $B/A=K_0$ | PROVED in the linear model; SIMULATED |
| Regime (a) is not the bottleneck of the data | HEURISTIC (three arguments in section 3.2) |
| Coverage rates $N^{-2/d}$, $N^{-2s/(2s+d)}$ | known; SIMULATED |
| Copy rule (4.2)–(4.4) | PROVED for Nadaraya–Watson; CONJECTURE for Tacotron |
| $N^\*$ (4.5) | HEURISTIC |
| Spectral rate $(\alpha-1)/\alpha$ | HEURISTIC (standard); SIMULATED approximately |
| Isotropic coverage does not explain F1+F4 | SIMULATED (within the tested family) |
| Spectral coverage + attractor explains F1, F4, grid | SIMULATED, partial; fit with 5 parameters |
| $q$ depends on $NK/M$ above $N^\*$ | PROVED in (4.4); weak evidence in the data (section 8.1); P5 decides |
| Power law in $q$ requires margin $\sim\sqrt{\ln N}$ or Pareto thresholds | PROVED |
| Two stages: copying follows $N_{\rm voice}$ (9.1) | PROVED given S1–S3; S1–S3 are CONJECTURE |
| Leakage floor (9.2) and bound in bits | PROVED for Gaussian channels; conversion to bits HEURISTIC |
| Cross-language transfer via density ratio | PROVED in the threshold model |
| Parametric module: floor (9.3) | PROVED; SIMULATED |
| Parametric module: exponent → 1 | HEURISTIC; simulation does not confirm with precision |
| Numerical predictions for $N=247$–2456 | depend on the model; with 38 voices they do not decide |

---

## Appendix: files

Scripts (in `analyses/`): `sim_0_dados.py` (reading the results, margins; from the previous attempt),
`sim_1_linear_gauss.py`, `sim_2_naoparam.py`, `sim_3_atrator.py` (`grade gp|nw [alphas]`, `run "dict(...)"`),
`sim_4_variantes.py`, `sim_5_ajuste_previsoes.py` (fits on the margin; previous attempt), `sim_5b_parametrico.py`,
`sim_6_previsoes.py`, `sim_7_espectral.py`, `sim_8_colapso.py`, `sim_9_piso_e_Nstar.py`.
Outputs: `out/sim_1.txt` … `out/sim_9.txt`, `out/sim_3_grade*.txt`. The scripts that read data open the `results.csv`
read-only and write auxiliary CSVs (`dados_*.csv`) in `analyses/`.
