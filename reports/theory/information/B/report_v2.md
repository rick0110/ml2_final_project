# Voice capacity of the decoder: a scaling law measured on the audio (v2)

Version 2, 17/11/2026. Answers the round-1 critique. v1 (`report_v1.md`) was kept. CPU only, no training; no
repository changed. Folder: `/tmp/claude-0/-home-user-ml2-final-project/4c11fa8d-6b5f-5505-bd7b-deebf5dd9907/scratchpad/teoria2/B/`.
Legend: **[MEASURED]** number taken from the data with no model; **[PROVED]** follows from the hypotheses; **[HEURISTIC]**; **[CONJECTURE]**;
**[PREDICTED ✓/✗]** compared with data that did not enter any calibration; **[FITTED]** chosen to reproduce the data shown.

## 0. Summary

1. **Voice capacity measured on the audio [MEASURED].** $C(N)=I(\text{output voice};\text{reference identity})$, estimated by
   identification among 38 external voices (Resemblyzer on the saved outputs; references from LibriTTS test-clean):

   | | Fano bound (accuracy) | cross-entropy bound (LOVO) | verifier ceiling (real utterance) | voice bits in the latent |
   |---|---|---|---|---|
   | $N=8$ | **0.16–0.31** [CI up to 0.78] | **0.61–0.75** [0.16, 1.20] | 4.2–4.7 (Fano) / 4.3–5.0 (CE) | μ: 17.1; z: 10.0 |
   | $N=32$ | **not measured** (there is no saved audio for N = 32) | — | | |
   | $N=128$ | **0.89–1.13** [0.39, 1.93] | **1.99–2.03** [1.35, 2.68] | | |
   | $\log_2 38$ | 5.25 | | | |

   $C(128)-C(8)=+1.24$ bit (CE, 95% CI [+0.66, +1.82]). Per doubling of $N$: **+0.31 bit** (CE) or +0.21 (Fano). The decoder
   passes ~1–2 of the 10 voice bits that the code z carries, and ~1/5 of what the verifier gets from a real utterance.
2. **The 1/22 factor explained [MEASURED + HEURISTIC].** The KL costs 7 bits (17.1 in μ → 10.0 in z). The rest (10 → 1–2 bits) is due to the
   decoder: (a) copied voices also have little information (2.3 bits against 1.2 for the non-copied ones), so the prior of the default
   voice is not the main bottleneck; (b) at strength 0.6 the capacity drops by 0.5–0.8 bit; noise in the KL only predicts a drop of 0,
   noise in the decoder only predicts 1.5–1.8 bit. About 1/3 to 1/2 of the effective noise sits after the style scale.
3. **Minimal model: 3 parameters** ($\mu_{128}$, $t$, $\gamma$). The ordering of the voices comes from a geometric statistic with no
   parameter. Everything at $N=32$ is predicted: $q=0.306$ against 0.302; per-voice histogram $\chi^2=4.4$ ($p=0.49$).
4. **A single mechanism, capacity, predicts strength 0.6** [PREDICTED ✓, with the Fano estimator]: predicted copying
   0.34/0.18/0.45 (trained/new/external) against 0.26/0.19/0.47. For R = 4 the sign comes out right (negative), but the size
   comes out too large.
5. **Dead**: (i) a Bayesian posterior over the latent geometry with width given by the capacity (zero parameters):
   $q=0.76/0.08/0.08$ against 0.43/0.30/0.21; (ii) the transfer of level from the trained voices to the new ones; (iii) the design
   effect as an explanation of the weight of $K$.
6. **$M_{\rm eff}=2M+NK$** comes out of the exposure: the style channel only exists from the `en_vae_v1` stage on, with 2,500 steps ×
   48 of LJ, the same as the fine-tuning. With this, halving $M$ gives $q$ 0.22 → **0.18**; v1 said ≤ 2 points, which was wrong.
   The theory of `02` gives 0.157, and the naive count prior gives 0.13. **Two runs do not separate these predictions**.
7. **Duration [PREDICTED ✓, reinforced].** Long utterances are more all-or-nothing: $T=+0.035$ [95% CI +0.015, +0.053], permutation
   $p=0.0002$. The intermediate voices drop by 14 points [−20, −8], with $p<2\times10^{-4}$. The ratio of margin SDs is 0.850
   [0.808, 0.894], against 0.829 predicted ($T^{-0.38}$).
8. **Leakage (two stages), a bound**: $I_{\rm utt}(\text{units};\text{speaker})\le C(N_{\rm voice})/9\approx0.2$ bit per utterance,
   with the premise stated in section 7.

---

## 1. The voice capacity of the decoder, without a model (point 1)

### 1.1 Definition

$C\equiv I(Y;S)$, with $S$ uniform over $|S|$ test voices and $Y$ the embedding (Resemblyzer, the project's verifier) of an
output synthesized from a reference of $S$. The text sentence is the same in all outputs (`__s0`, "would not agree
with…"), so the variation is in the voice. Two valid lower bounds:

- **Fano** [PROVED]: $I(Y;S)\ge I(\hat S;S)\ge\log_2|S|-h(P_e)-P_e\log_2(|S|-1)$, with $\hat S$ the neighbor with the highest cosine.
- **Variational cross-entropy** (Barber–Agakov) [PROVED]: $I(Y;S)\ge H(S)-\mathbb E[-\log_2 q(S\mid Y)]$ for any
  $q$. I use $q\propto e^{\beta\cos}$, with $\beta$ chosen leaving the evaluated voice out (LOVO), to avoid optimism.
- The **confusion-matrix plug-in** (with Miller–Madow correction) gives 2.6–4.0 bits, but with one sample per class it is
  dominated by bias: the correction *increases* the number. It is reported and **not used**.
- CI: bootstrap over the 38 voices (1,000 replicates).

### 1.2 Execution

- `pip install --target pylib resemblyzer webrtcvad-wheels`: it worked, with weights included.
- Reference audio: LibriTTS test-clean via the indicated mirror, with only the `.wav` files extracted into `libritts_tc/` (1.4 GB, 4,837 files).
- Outputs: `wavs_wide.tgz`, `wavs_n128.tgz` and `wavs_n200.tgz` extracted into `wavs_ext/`.
- **`wavs_wide.tgz` only has N = 8 and N = 128** (50 references × {R = 1, R = 4}): **C(32) cannot be measured**. I did not interpolate
  as if it were a measurement.
- Pipeline validation: the output–reference cosine here correlates 0.93 (N = 8) and 0.89 (N = 128) with the `sim_ref` of
  `results.csv` (max. difference 0.07). It is probably a difference in preprocessing or version; the same yardstick is used in all the
  comparisons below.
- Scripts: `b12_embed.py` (embeddings) and `b13_capacidade.py` (bounds and CI); outputs `out/b12_emb.npz`, `out/b13.txt`.

### 1.3 Results [MEASURED]

| condition | accuracy | top-5 | Fano (bits) | CE-LOVO (bits) |
|---|---|---|---|---|
| held-out real utterance × centroid (verifier ceiling) | 0.947 | 0.97 | 4.68 [4.00, 5.25] | 4.96 [4.54, 5.23] |
| real utterance × reference utterance | 0.895 | 0.95 | 4.21 | 4.31 |
| N = 8, R = 1, × centroid / × ref utterance | 0.184 / 0.132 | 0.45 / 0.42 | 0.31 / 0.16 | 0.75 / 0.61 |
| N = 8, R = 4 | 0.132 / 0.105 | 0.42 / 0.39 | 0.16 / 0.10 | 0.57 / 0.39 |
| N = 128, R = 1 | 0.342 / 0.395 | 0.74 / 0.71 | 0.89 / 1.13 | 1.99 / 2.03 |
| N = 128, R = 4 | 0.211 / 0.211 | 0.74 / 0.61 | 0.39 / 0.39 | 1.57 / 1.37 |

Paired differences (CE, centroid):

- $C(128)-C(8)=+1.24$ [+0.66, +1.82].
- $C(R4)-C(R1)=-0.18$ [−0.50, +0.11] at N = 8 and $-0.41$ [−0.86, +0.05] at N = 128.
- **R = 4 does not increase the capacity**: the gain is ≤ +0.11 bit with 95% confidence, against +2.5 to +5 more bits in the code.

Output–output identification: same reference and same condition, utterance s0 against s1, in the plain runs (16 voices, log₂16 = 4):

| run | 1.0 | 0.6 | R = 4 | R = 16 |
|---|---|---|---|---|
| n128_k8 (Fano / CE) | 2.97 / 3.68 | 1.30 / 2.44 | 3.42 / 3.82 | 2.57 / 3.35 |
| n128_k32 | 2.97 / 3.06 | 2.57 / 3.00 | 3.42 / 3.49 | 3.42 / 3.86 |
| n200_k32 | 1.88 / 2.64 | 1.58 / 2.45 | 2.21 / 2.72 | — |

This measures **consistency** ($I(Y_{s0};Y_{s1})$, includes 8 trained voices), not fidelity to the reference. Averaging references
increases consistency (+0.1 to +0.45 bit) and not fidelity (previous table): the averaged code gives the same voice in different
utterances, but not closer to the reference.

### 1.4 The scaling law

- **$C(N)\approx C(8)+\theta\ln(N/8)$, with $\theta=0.45$ bit per unit of ln N (CE) or 0.30 (Fano)**: +0.31 / +0.21 bit per
  doubling of N [MEASURED at 2 points; the log-linear form is a CONJECTURE, a power law $C\propto N^{0.35}$ (CE) also fits].
- In scale: at N = 128 the decoder passes ≈ 1–2 bits, 20–40% of $\log_2 38$, 10–20% of the 10 voice bits in z, 6–12% of the
  17 bits in μ.
- Keeping the log-linear rate (CE), reaching the verifier ceiling (~4.7 bits) would require $\ln(N/8)\approx8.8$, that is
  $N\sim5\times10^4$ speakers. A 2-point extrapolation, only to give a sense of scale.
- **Per voice**: at N = 128 the copied voices have 2.26 bits and the non-copied ones 1.17 (Spearman copying × $c_v$ = 0.23); at N = 8,
  0.45 against 0.54 (Spearman 0.13). Capacity is not the same thing as the binary copying criterion: it measures resolution *between*
  voices, the criterion measures the side of the default voice.

## 2. Where the 1/22 factor comes from (point 2)

v1 took 1/22 from a fitted temperature (ratio of log-likelihood spreads). Now, in measured bits:

| stage | voice bits | loss | where the number comes from |
|---|---|---|---|
| encoder μ (Gaussian, variance components) | 17.1 | — | `b17` |
| z = μ + posterior noise (what the decoder saw in training) | 10.0 | −7.1 bits: **KL / posterior temperature** | `b17` |
| decoder output, N = 128 | 1.1 (Fano) – 2.0 (CE) | −8 to −9 bits: **decoder** | `b13` |
| output, N = 8 | 0.3–0.75 | | `b13` |

Four candidates:

1. **Prior of the default voice.** It would make the loss come from the attracted voices. But the copied voices also pass only ~2.3 bits.
   Even if the 21% non-copied voices collapsed to a single point, the upper limit would be ~4.4 bits. **It is not the main one.**
2. **Style scale.** Strength 1.0 is nominal; it does not reduce information.
3. **KL / posterior temperature.** Explains 7 of the ~15 bits lost.
4. **Decoder capacity.** Discriminating test, at strength 0.6: noise applied *before* the scale (KL, encoder) is
   scaled together with the signal, and the information does not change [PROVED: invertible transformation]. Fixed noise *after* the scale
   (decoder) gives $I(\alpha)=\tfrac12\sum_j\log_2(1+\alpha^2\lambda_j/s^2)$.

   | run | observed drop (CE / Fano) | predicted, noise in the decoder only | predicted, noise in the KL only |
   |---|---|---|---|
   | n128_k8 | −1.24 / −1.67 | −2.1 / −1.8 | 0 |
   | n128_k32 | −0.06 / −0.40 | −1.8 / −1.8 | 0 |
   | n200_k32 | −0.19 / −0.30 | −1.6 / −1.1 | 0 |
   | mean | −0.50 / −0.79 | −1.8 / −1.5 | 0 |

   The observed value lies between the two: with $s^2=s^2_{\rm pre}+s^2_{\rm post}$, ~30–50% of the effective noise is after the scale
   [HEURISTIC].

**Conclusion:** the "1/22" is (KL) × (decoder capacity), with the decoder ≈ 4–5× the KL loss as a ratio of bits;
it is neither prior nor scale. And **it is not constant**: the capacity grows with N (0.3–0.75 → 1.1–2.0 bits), that is, the effective factor
improves from ~1/40 to ~1/10 between N = 8 and 128.

## 3. Minimal model (point 3)

**Killed first: posterior with width given by the capacity (zero parameters)** [PREDICTED ✗]. `b14_cap.py`:

- Decoding width $s^2(N)$ chosen so that $\tfrac12\sum\log_2(1+\lambda_j/s^2)=C(N)$ measured.
- $M_{\rm eff}=2M+NK$ derived (section 5); posterior over the trained voices in the geometry of each run; Lemma 1 link.
- Result: $q=0.76/0.08/0.08$ (CE) or 1.00/0.64/0.08 (Fano) against 0.43/0.30/0.21.
- With a shift fitted at N = 128, N = 8 and 32 go to 1.00 and so do the trained voices. The variation of $s^2$ with N
  (43 → 15) enters a quadratic function of the distance and gives an effect much stronger than the observed one.
- **Capacity cannot enter as the width of a posterior over the latent geometry.**

**3-parameter model** (`b16_min3.py`). Per-voice latent state $L_v\sim\mathcal N(\mu_N,1)$; each utterance is copied with
probability $\Phi(L_v/t)$; $\mu_N=\mu_{128}+\gamma\ln(N/128)$. The ordering of the voices is predicted by the geometric
likelihood-ratio statistic $\Lambda_v$ (v1, section 3.3: $s^2$ from the spread of the trained voices, no external copying used; the ordering
does not depend on $\mu,t,\gamma$).

| parameter | value | calibrated on |
|---|---|---|
| $\mu_{128}$ | 0.908 | per-voice counts at N = 128 (marginal likelihood) — **test curve, 1 point** |
| $t$ (latent ICC $1/(1+t^2)=0.87$) | 0.393 | same, N = 128 |
| $\gamma$ | 0.263 per ln N | $q(8)$ — **test curve, 1 point** |
| (ordering: $\Lambda_v$) | no parameter | spread of the trained voices, a different basis |

**Fitted × predicted table:**

| quantity | N = 8 | N = 32 | N = 128 |
|---|---|---|---|
| $q$ obs / model | 0.434 / 0.434 *(fitted)* | 0.302 / **0.306** *(predicted)* | 0.211 / 0.199 *(fitted)* |
| voices between 10 and 90%, obs / model | 0.34 / **0.37** *(predicted)* | 0.37 / **0.33** *(predicted)* | 0.26 / 0.26 *(fitted)* |
| histogram in 6 bins, $\chi^2$ (p by simulation) | **1.6 (0.91)** *(predicted)* | **4.4 (0.49)** *(predicted)* | 1.5 (0.91) *(fitted)* |
| ordering of the voices, Spearman($\Lambda_v$, copying) | **0.54** | **0.54** | **0.56** *(all predicted)* |

Caveats:

- Predicting N = 32 from 8 and 128 tests only the linearity of the mean state in ln N; it is a weak test, but a genuine one.
- The histograms at N = 8 and 32 are real predictions: the ICC came only from N = 128.
- Extrapolation: $q(512)=0.118$, $q(2048)=0.064$.

**Link to capacity.** Replacing $\gamma\ln N$ with $\beta\,C(N)$, with $C$ measured on the audio, $\beta=\gamma/(dC/d\ln N)=0.58$
(CE) or 0.89 (Fano) state SDs per bit. It still has 3 parameters, but now the N term is **measured capacity**: with $C$
measured at another N, $q$ is predicted without touching the curve. "Each bit of voice capacity is worth ~0.6–0.9 between-voice SD in the copying state."

## 4. The four failures of v1 (point 4) — test of the capacity-limited channel

| v1 failure | with the measured capacity | status |
|---|---|---|
| **Absolute level of the new voices** (0.16/0.12/0.08 against 0.43/0.30/0.21) | The limited channel explains *qualitatively* why the latent geometry overestimates the evidence: the decoder passes 1–2 bits, not 10–17. But the quantitative model of capacity as posterior width fails worse (section 3). The level stays as 1 parameter ($\mu_{128}$). | **not resolved**; the idea "level = latent geometry" is dead |
| **Copying at strength 0.6** (1–6% predicted) | The state shift is $\beta\cdot\Delta C$, with $\Delta C$ measured output–output at strength 0.6 (−0.79 bit, Fano) and $\beta=0.89$ (from N and C, without strength data): predicted 0.34/0.18/0.45, observed 0.26/0.19/0.47. With CE ($\Delta C=-0.50$, $\beta=0.58$): 0.57/0.34/0.73, little drop. | **resolved with Fano**, not with CE; the choice of estimator is a *post hoc* degree of freedom |
| **Sign of the mean for R = 4** (v1: +0.02, observed −0.02/−0.04) | The fidelity capacity *drops* with R = 4 (−0.18/−0.41 bit) ⇒ prediction −0.04/−0.12 (CE) or −0.05/−0.25 (Fano) against −0.02/−0.04. | **sign resolved**, size 2–6× too large; the CIs of $\Delta C$ touch zero |
| **Trained voices improving with N** in the calibrated model | Output–output identification R1→R4 among the 8 trained voices: 0.875 (N = 8) → 1.000 (N = 128), already at the ceiling; external (8 drawn): 0.66 → 0.78. The capacity of the trained voices is saturated (memorization), that of the new ones grows. In addition, $M_{\rm eff}=2M+NK$ grows with N, which pulls the trained voices down (−0.64 nat from 8 to 128) and offsets part of the borrowing. | **partially**: 8 voices at the ceiling, no power; the minimal model has no N term for the trained voices, by construction |

## 5. Effective weight of the default voice, derived (point 5)

Facts from the code (read-only):

- `scripts/cloud/stage_setup.sh` and `README.md`: the `en_vae_v1` stage starts from NVIDIA's Tacotron 2 (no style input) and
  trains the VAE on LJSpeech with `--max-steps 2500 --batch-size 48`.
- `scripts/coverage/colab_job.sh`: the coverage fine-tuning starts from `ckpt/en_final` with `BATCH=48` and 2,500 steps.

**Derivation [HEURISTIC; premise in bold].** **The prior over style codes can only have been learned where a style
channel existed**: in `en_vae_v1` (120,000 presentations, all LJ) and in the fine-tuning. NVIDIA does not count, because it had no
style input. In the fine-tuning each utterance is presented $r=120{,}000/(M+NK)$ times. Measuring the weight in "fine-tuning utterances":
$$M_{\rm eff}=M+\frac{120{,}000}{r}=M+(M+NK)=2M+NK .$$

- For 128×32: $M_{\rm eff}=8{,}096$. It sits at the low end of the v1 calibrations (8×10³ to 1.4×10⁵); it matches the `b2`
  variants (8,232 and 11,089) and not the larger ones.
- **$M$ = 1,000**: $M_{\rm eff}=6{,}096$, $\Delta\ln M_{\rm eff}=-0.28$ nat (naive count: −0.69). With the coefficient of the
  trained voices (1.21 τ per nat): **q 0.222 → 0.179 (−4.3 pts)**. Naive count: → 0.129 (−9 pts). Theory `02` ($M/2\equiv2K$):
  0.21 → 0.157 (−5.4 pts). Prior dominated by the NVIDIA pretraining (v1, $M_{\rm eff}\ge10^5$): ≤ 1 pt.
- **v1 was wrong** to state "≤ 2 pts" as a prediction of the theory; that only holds if the style-free pretraining counted, which the
  derivation above denies.
- **Power:**
  - The difference between this derivation and `02` is **1.1 pt**. Between this one and the naive count, ~5 pts.
  - The SD of the main seed effect is 0.054 on the fraction (`04`), hence 0.076 for the difference of two runs.
  - Separating 4.3 from 5.4 pts is out of reach. Separating 4.3 from 9 pts with power 0.8 requires ≈ 2(1.96 + 0.84)²·0.054²/0.047² ≈ 21
    runs per arm. Separating "≤ 1 pt" from "−9 pts" requires ~6 per arm.
  - **A 2-run experiment does not discriminate any of the hypotheses.** Not with R = 4 voices either, because the seed dominates (`04`).

## 6. Duration: the *a priori* prediction confirmed, with CI and permutation (point 6)

`b15_min.py`. For each external voice in the 3 `-wide` runs (114 voice-runs), I split the 25 utterances into 12 short and 12 long
(sorted by synthesized duration; the middle one is left out).

- Statistic: $T=\overline{|p_{\rm long}-\tfrac12|-|p_{\rm short}-\tfrac12|}$.
- CI: speaker bootstrap (the 3 replicates of each speaker together, 2,000 replicates).
- Null: permutation of the utterances within the voice (5,000 replicates, one-sided).

| measure | effect | 95% CI | p (permutation) | predicted (large deviations, $\tau\propto T^{-0.38}$) |
|---|---|---|---|---|
| $T$ (distance from ½) | **+0.035** | [+0.015, +0.053] | **0.0002** | > 0 |
| fraction of intermediate voices (long − short) | **−0.140** | [−0.202, −0.079] | **< 0.0002** | −0.12 to −0.14 (v1, `b7`) |
| ratio of margin SD, long/short | **0.850** | [0.808, 0.894] | — | 0.829 (exponent −0.38); 0.781 (−0.5) |

Mean ratio of long/short durations: 1.64. The exponent −0.38 came from the spread per duration quintile across all groups
(`b3`); the prediction for the halves of each voice uses only that exponent, and it is right within the CI. Caveat: the 25 text sentences are
the same for all the voices, so "long" also means "these texts". The within-voice permutation controls for chance,
not for the text.

## 7. Leakage in the two-stage architecture: a bound (point 7)

**Premise (the only one)** [CONJECTURE, supported by section 1]: the stage-B decoder pools the voice from all inputs into a
per-utterance budget limited by its own capacity $C$; it is not an ideal integrator of per-frame evidence. The measured
capacity (1–2 bits against 10 available) is what supports this.

With this premise, the code and the units are two Gaussian channels over the same voice directions, combined by
precision. The weight of the leaked voice is $w=\mathrm{SNR}_\ell/(\mathrm{SNR}_\ell+\mathrm{SNR}_c)$ (`02`, eq. 9.2), and in the low-SNR regime
we are in, $I\approx\mathrm{SNR}\cdot k/(2\ln2)$, so $w\approx I_\ell/(I_\ell+I_c)$. The target $w\le0.1$ (floor of $q$ ≈ 0.09–0.14
for $N_{\rm voice}$ = 128–2,456, `02`, table 9.2) gives:
$$
\boxed{\,I_{\rm utt}(\text{units};\text{speaker})\ \le\ C(N_{\rm voice})/9\ \approx\ 0.2\ \text{bit per utterance (with }C(128)=2.0\text{, CE)}\,}
$$

How to measure it, with the same protocol as section 1: identify the speaker among 38 from the units aggregated per utterance. 0.2 bit
of Fano corresponds to ~14% accuracy (chance 2.6%). If the premise fails (ideal per-frame integration), the bound becomes ~0.2/$T_{\rm eff}$
bit per frame (0.01 bit/frame with $T_{\rm eff}=20$). That is why the test of the premise comes first: measure the capacity of B for a
*known* voice channel injected only into the units, and see whether it saturates at ~C or grows with the number of frames.

## 8. Predictions (updated)

| # | experiment | v2 prediction | decides |
|---|---|---|---|
| P1 | $C(N)$ at N = 32, 512, 2,048 (synthesis + embeddings only) | CE: 1.38 / 2.64 / 3.27 bits (log-linear) | $C$ saturating ≤ 2.5 bits at 2,048 rejects the log form |
| P2 | $q(N)$ with measured $C(N)$, $\beta=0.58$–0.89 | $q(512)\approx0.12$, $q(2048)\approx0.06$ (via $\gamma$) | — |
| P3 | $M$ = 1,000 | 0.22 → 0.18 (derived); count 0.13; `02` 0.157 | ≥ 6 runs per arm to separate "≤ 1 pt" from "−9" |
| P4 | two stages | $I_{\rm utt}(u;\text{spk})\le C/9\approx0.2$ bit keeps $q$ within ≤ +5 pts of one stage | |
| P5 | utterances 2× longer | intermediate voices ×0.77; margin SD ×0.77 | |
| P6 | strength 0.4 | $\Delta C$ (measured) × β predicts copying; noise in the KL only predicts $\Delta C=0$ | a null drop in $C$ rejects the "after the scale" part |
| P7 | latent k = 2 / 8 | k = 8 keeps $C$; k = 2 lowers $C$ by ~60% (fraction of $\tfrac12\sum\log_2$ outside the top 2) | |

## 9. Parameter accounting and status

- **Minimal model: 3 parameters**, two of them fitted on 2 of the 3 points of the test curve ($\mu_{128}$, $\gamma$) and one on the
  per-voice spread at N = 128 ($t$). The statistic that orders the voices comes from a different basis (trained voices).
- **Measured without a model**: $C(8)$, $C(128)$ and their differences, the cost of the KL, $\Delta C$ at strength 0.6 and at R = 4, the duration effect.
- **Predicted**: all of N = 32, histograms at N = 8 and 32, the ordering of the voices at the 3 N, copying at strength 0.6 (via measured $\Delta C$;
  only with Fano), the sign of R = 4.
- **Rejected in this round**: posterior with width given by the capacity (zero parameters); v1's "≤ 2 pts for M/2";
  design effect; level coming from the latent geometry.

Reproduction (in addition to the v1 scripts):
```
python3 -I b12_embed.py wavs_ext libritts_tc ref_ids.txt out/b12_emb.npz 30
python3 -I b13_capacidade.py out/b12_emb.npz <results_root> 1000     # C(N), bounds, CI       -> out/b13.txt, b13.json
python3 -I b14_cap.py out/b0.npz ce|fano                             # capacity as width     -> out/b14_*.txt (dead)
python3 -I b15_min.py out/b0.npz out/b12_emb.npz                     # per-voice info, duration, M/2 -> out/b15.txt
python3 -I b16_min3.py out/b0.npz                                    # 3-parameter model     -> out/b16.txt
python3 -I b17_forca.py out/b0.npz out/b13.json                      # mechanism of the factor -> out/b17.txt
python3 -I b18_capshift.py out/b0.npz out/b13.json out/b16.json      # strength 0.6 and R=4 via ΔC -> out/b18.txt
python3 -I b19_seen.py out/b0.npz out/b12_emb.npz                    # trained × external    -> out/b19.txt
```
