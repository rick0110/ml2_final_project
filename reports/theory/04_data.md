# Statistical analysis of the saved coverage results (N speakers × K utterances)

CPU only, no new training. All numbers come from the `results.csv` files saved in `reports/coverage/results/<training>/step_2500/strength_<s>[_r<R>]/`. Main condition: step 2,500, style strength 1.0, R = 1. "Copied" = `sim_ref > sim_default`; continuous margin `m = sim_ref − sim_default`; per-speaker margin `ms = sim_spk − sim_default`; q = 1 − fraction copied (fraction **not** copied).

Scripts (all in [`analyses/`](analyses/), run with `python3 -I script.py <results_dir> out [B]`):

| Script | Task | Outputs in `out/` |
|---|---|---|
| `dados_comum.py`, `dados_modelos.py`, `dados_leis_bin.py` | loader; fitting engines | — |
| `dados_00_check.py` | check (identical reference sets across training runs; available strengths) | — |
| `dados_01_tabela.py` | 1 | `tabela_runs.csv` |
| `dados_02_margem.py` | 2 | `02.txt`, `varcomp_por_treino.csv`, `seed_sd.json`, `precisao_por_treino.csv`, `leis_margem_unseen.csv`, `boot_dev_margem_unseen.npy` |
| `dados_03_link.py` | 3 | `03.txt`, `link_probit.csv`, `tudo_ou_nada.csv` |
| `dados_04_leis_binario.py` | 4 | `04.txt`, `leis_binario.csv`, `previsoes_binario.csv` |
| `dados_05_colapso.py` | 5 | `05.txt`, `colapso.csv` |
| `dados_06_artefato.py` | 6 | `06.txt`, `artefato_expoentes.csv`, `artefato_por_locutor.csv` |
| `dados_07_R.py` | 7 | `07.txt`, `efeito_R.csv` |
| `dados_08_poder.py`, `dados_08b_poder_extra.py` | 8 | `08.txt`, `08b.txt`, `poder.csv`, `poder_extra.csv` |

**Verification of the previous (interrupted) work.** The files `dados_comum.py`, `dados_01_tabela.py`, `dados_02_margem.py`, `dados_modelos.py` and the outputs in `out/` were re-read and checked: (i) the 33 training runs use exactly the same 4 "unseen" references; 30 use the same 8 "seen" ones (the 3 N = 2 training runs use 8 references from 2 speakers); the 3 `-wide` evaluations have the same 38 external speakers × 25 utterances (950 samples); the 4 plain training runs with external voices (N = 128, 200) use the same 4. This validates stacking training runs × speakers × utterances in a crossed ANOVA. (ii) The table reproduces the numbers of [`coverage_theory.md`](coverage_theory.md) (e.g., 128 × 32 s1: 74% unseen; `-wide`: 57/70/79% external). (iii) Strength 0 exists only in the `-spk`/`-spkdrop` training runs and does not include the 38 external voices (relevant to task 6). The previous results were reused; nothing in them was corrected, except what is indicated below.

---

## 1. Table of training runs

Grid (plain) and wide evaluation (`-wide`, same checkpoint as the plain run of the same name, re-synthesized). Copied = fraction; WER unseen/trained/external; mean margin m (SD across samples). "Ext." in the plain runs = 4 external voices; in the `-wide` runs = 38.

| training run | N | K | seed | copied unseen | copied trained | copied ext. | WER unseen / trained / ext | m unseen | m trained | m ext. | ms ext. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n2_k128_s1 | 2 | 128 | 1 | 0.04 | 0.69 | — | 0.079 / 0.129 / — | -0.112 (0.065) | +0.098 (0.134) | — | — |
| n2_k128_s2 | 2 | 128 | 2 | 0.02 | 0.73 | — | 0.055 / 0.114 / — | -0.114 (0.048) | +0.091 (0.103) | — | — |
| n8_k8_s1 | 8 | 8 | 1 | 0.26 | 0.45 | — | 0.143 / 0.207 / — | -0.052 (0.097) | -0.005 (0.120) | — | — |
| n8_k8_s2 | 8 | 8 | 2 | 0.39 | 0.45 | — | 0.244 / 0.262 / — | -0.033 (0.096) | -0.009 (0.112) | — | — |
| n8_k32_s1 | 8 | 32 | 1 | 0.47 | 0.74 | — | 0.137 / 0.173 / — | -0.024 (0.103) | +0.105 (0.114) | — | — |
| n8_k32_s1-wide | 8 | 32 | 1 | 0.36 | 0.78 | 0.57 | 0.106 / 0.140 / 0.134 | -0.040 (0.088) | +0.105 (0.115) | -0.004 (0.106) | +0.026 (0.108) |
| n8_k128_s1 | 8 | 128 | 1 | 0.51 | 0.91 | — | 0.079 / 0.112 / — | -0.011 (0.106) | +0.172 (0.120) | — | — |
| n8_k128_s2 | 8 | 128 | 2 | 0.37 | 0.88 | — | 0.094 / 0.104 / — | -0.029 (0.107) | +0.157 (0.118) | — | — |
| n32_k8_s1 | 32 | 8 | 1 | 0.47 | 0.58 | — | 0.126 / 0.172 / — | +0.009 (0.114) | +0.047 (0.107) | — | — |
| n32_k8_s2 | 32 | 8 | 2 | 0.47 | 0.51 | — | 0.085 / 0.169 / — | -0.006 (0.109) | +0.019 (0.087) | — | — |
| n32_k32_s1 | 32 | 32 | 1 | 0.52 | 0.80 | — | 0.077 / 0.130 / — | +0.026 (0.104) | +0.088 (0.099) | — | — |
| n32_k32_s1-wide | 32 | 32 | 1 | 0.48 | 0.76 | 0.70 | 0.077 / 0.121 / 0.116 | +0.015 (0.118) | +0.095 (0.112) | +0.048 (0.085) | +0.071 (0.085) |
| n32_k128_s1 | 32 | 128 | 1 | 0.61 | 0.86 | — | 0.052 / 0.121 / — | +0.045 (0.105) | +0.147 (0.117) | — | — |
| n32_k128_s2 | 32 | 128 | 2 | 0.57 | 0.85 | — | 0.050 / 0.097 / — | +0.037 (0.097) | +0.149 (0.115) | — | — |
| n128_k8_s1 | 128 | 8 | 1 | 0.55 | 0.52 | 0.86 | 0.081 / 0.121 / 0.126 | +0.007 (0.086) | +0.034 (0.114) | +0.087 (0.065) | +0.063 (0.070) |
| n128_k32_s1 | 128 | 32 | 1 | 0.74 | 0.76 | 0.86 | 0.077 / 0.120 / 0.157 | +0.058 (0.075) | +0.090 (0.117) | +0.091 (0.070) | +0.096 (0.064) |
| n128_k32_s1-wide | 128 | 32 | 1 | 0.75 | 0.78 | 0.79 | 0.071 / 0.120 / 0.117 | +0.056 (0.077) | +0.095 (0.117) | +0.078 (0.094) | +0.101 (0.088) |
| n128_k32_s2 | 128 | 32 | 2 | 0.56 | 0.66 | 0.86 | 0.076 / 0.128 / 0.105 | +0.016 (0.082) | +0.063 (0.100) | +0.076 (0.058) | +0.055 (0.079) |
| n200_k32_s1 | 200 | 32 | 1 | 0.64 | 0.67 | 0.91 | 0.073 / 0.122 / 0.105 | +0.030 (0.073) | +0.073 (0.116) | +0.097 (0.071) | +0.077 (0.084) |

Variants (one seed unless stated; 4 unseen × 25 and 8 trained × 25):

| training run | copied unseen | copied trained | WER unseen / trained | m unseen | m trained |
|---|---|---|---|---|---|
| n2_k128_s1-spk | 0.04 | 0.69 | 0.079 / 0.139 | -0.101 | +0.097 |
| n8_k8_s1-aug | 0.28 | 0.50 | 0.139 / 0.175 | -0.042 | -0.004 |
| n8_k8_s1-contrast | 0.27 | 0.42 | 0.199 / 0.273 | -0.073 | -0.023 |
| n8_k8_s2-contrast | 0.31 | 0.47 | 0.208 / 0.229 | -0.072 | -0.026 |
| n8_k8_s1-spk | 0.30 | 0.43 | 0.140 / 0.183 | -0.053 | -0.003 |
| n8_k8_s1-swap | 0.27 | 0.52 | 0.206 / 0.230 | -0.075 | -0.026 |
| n8_k8_s2-swap | 0.32 | 0.47 | 0.245 / 0.254 | -0.062 | -0.007 |
| n8_k128_s1-aug | 0.42 | 0.83 | 0.077 / 0.113 | -0.009 | +0.141 |
| n8_k128_s1-contrast | 0.39 | 0.85 | 0.095 / 0.138 | -0.041 | +0.145 |
| n8_k128_s1-spk | 0.46 | 0.88 | 0.084 / 0.128 | -0.014 | +0.164 |
| n8_k128_s1-spkdrop | 0.37 | 0.88 | 0.076 / 0.141 | -0.019 | +0.165 |
| n8_k128_s1-swap | 0.38 | 0.87 | 0.079 / 0.122 | -0.047 | +0.145 |
| n32_k8_s1-spkdrop | 0.60 | 0.57 | 0.147 / 0.218 | +0.029 | +0.035 |
| n32_k128_s1-spkdrop | 0.62 | 0.83 | 0.056 / 0.119 | +0.038 | +0.142 |

Full table (includes the SD of `ms`, mean `sim_ref` and `sim_default`, and copying by `sim_spk`): [`out/tabela_runs.csv`](analyses/out/tabela_runs.csv) and [`out/tabela_runs.md`](analyses/out/tabela_runs.md). In the variants, all the margins of the unseen voices fall within the range of the two plain seeds of the same configuration ± 2 seed SDs (0.015), except `-spkdrop` 32 × 8 (+0.029 against +0.009/−0.006), which sits ~1.5 SD above: no variant has a demonstrable effect with one seed.

---

## 2. Continuous margin

### 2.1 Methods

- **Per training run and group**: balanced two-way ANOVA speaker × utterance (random effects, method-of-moments estimators), separating the SD between speakers, between utterances, and the residual (which includes the speaker × utterance interaction).
- **Joint**: crossed three-way ANOVA training run × speaker × utterance (16 plain training runs × 4 unseen × 25; 14 training runs × 8 trained without N = 2; 3 `-wide` training runs × 38 external × 25). The "training-run effect" contains the fixed effect of (N, K) and the seed; for this reason the seed variance is estimated separately.
- **Seeds**: 6 pairs of replicates (2×128, 8×8, 8×128, 32×8, 32×128, 128×32). For each pair, D = (Y₁ − Y₂)/√2 per speaker × utterance; E[D̄²] estimates the seed variance of the training-run mean (with the evaluation set fixed). It is decomposed into a seed main effect and seed × speaker and seed × utterance interactions (ANOVA of D). 95% chi-square CI with 6 d.f.
- **Re-synthesis**: the 3 plain/`-wide` pairs are the same checkpoint synthesized again; they give the variance of the synthesis alone.

### 2.2 Variance components (SD)

| Source | Margin m, unseen (3-way, 16 training runs) | Binary, unseen | Margin m, external (38, 3 training runs) | Binary, external |
|---|---|---|---|---|
| speaker | 0.095 | 0.374 | 0.076 | 0.309 |
| training run × speaker | 0.033 | 0.196 | 0.046 | 0.209 |
| utterance | 0.007 | 0.024 | 0.009 | 0.025 |
| training run × utterance | 0.008 | 0.030 | 0.006 | 0.022 |
| speaker × utterance | 0.016 | 0.077 | 0.014 | 0.055 |
| residual | 0.028 | 0.265 | 0.031 | 0.261 |
| training run (includes the effect of N, K) | 0.048 | 0.171 | 0.041 | 0.107 |

The utterance barely matters (SD 0.007–0.009 on the margin); **the variability is from the speaker** (ICC of the margin within a training run: median 0.91 for unseen, 0.92 for trained, 0.81–0.85 for external).

SD **between seeds** of a training-run mean (fixed evaluation), 6 pairs:

| | Total SD (95% CI) | seed main effect | part coming from seed × speaker/utterance interactions |
|---|---|---|---|
| margin, unseen | 0.0150 (0.0097–0.033) | 0.0127 | 0.0080 |
| binary, unseen | 0.077 (0.050–0.169) | 0.054 | 0.054 |
| margin, trained | 0.0125 (0.0081–0.028) | 0.0098 | 0.0078 |
| binary, trained | 0.039 (0.025–0.085) | 0.000 | 0.041 |

Re-synthesis of the same checkpoint (3 pairs): SD of the mean 0.0080 (margin, unseen) and 0.048 (binary, unseen). That is, **about 40% of the "seed" variance of the binary measure on the 4 unseen voices is just re-synthesis** (decoder sampling), not training.

Central implication: the seed main effect (0.0127 on the margin ≈ 0.05 on the fraction) **does not shrink with more evaluation voices**. With 38 voices, the speaker part of the uncertainty drops to ~0.046/√38 ≈ 0.007, and the seed becomes dominant.

### 2.3 Precision gain of the margin over the binary measure

Comparison on the same scale (fraction copied), by three routes:

| Route | Variance ratio binary / margin | Interpretation |
|---|---|---|
| Within a training run: speaker × utterance bootstrap of p̂ vs Φ(m̄/s) | median 1.00 (unseen), 1.56 (trained), 1.31 (external, 38) | almost no gain with 4 voices, because the uncertainty is from the speaker and affects both measures |
| Between seeds (seed SD × slope dp/dμ across training runs = 3.83 unseen, 2.68 trained) | 1.78 (unseen), 1.32 (trained) | the margin is equivalent to 1.3–1.8× more training runs |
| Re-synthesis | 2.4 (unseen), 7.8 (trained) | the margin is much more stable to decoder sampling |

Signal-to-noise ratio (SD between configurations / seed SD): fraction 2.38, margin 3.13 (unseen). **The gain is real but modest (1.3–2×)**: the margin does not solve the main problem, which is speaker and seed variance.

### 2.4 Laws for the mean margin (unseen, 16 training runs, 10 configurations)

Unweighted least squares per training run; CV leaving one configuration out (LOCO: both seeds are left out together); AICc/BIC with k+1 parameters (σ).

| Law | k | RMSE | CV-RMSE (LOCO) | ΔAICc | ΔBIC | AICc weight |
|---|---|---|---|---|---|---|
| a + b/N | 2 | 0.0196 | 0.044 | 11.8 | 12.7 | 0.002 |
| a + b/N + c/(NK) [L1] | 3 | 0.0146 | 0.033 | 6.0 | 6.0 | 0.033 |
| a + b/√N | 2 | 0.0172 | 0.019 | 7.6 | 8.5 | 0.015 |
| a + b/√N + c/√K [L2] | 3 | 0.0121 | **0.016** | 0.0 | 0.0 | 0.65 |
| a + b ln N | 2 | 0.0252 | 0.034 | 19.9 | 20.8 | 0 |
| a + b ln N + c ln K | 3 | 0.0240 | 0.038 | 22.0 | 22.0 | 0 |
| a + b N^−α | 3 | 0.0170 | 0.141 | 10.8 | 10.9 | 0.003 |
| a + b N^−α + c ln K | 4 | 0.0111 | 0.069 | 1.6 | 0.0 | 0.30 |
| K only / constant | 2 / 1 | 0.048 / 0.049 | 0.065 / 0.055 | ≥38 | ≥39 | 0 |

- Achievable RMSE floor per training run = seed SD = 0.015. **L2 is already at the noise level** (0.012 in-sample, 0.016 out-of-sample); nothing more complex is justified.
- Without N = 2 (14 training runs): L2 and a + b N^−α + c ln K remain ahead (CV 0.016 and 0.015), but 1/N, 1/√N and ln N come within ≤0.01 of them.
- Exponent α in the form a + b N^−α + c ln K: 0.64, speaker × utterance bootstrap CI [0.08, 1.08]; without N = 2 the point moves to 1.24. **The exponent of the margin is not identified.**
- Bootstrap (B = 1,000) of the CV-RMSE differences relative to a + b ln N + c ln K: only "K only" and "constant" are worse with certainty (P = 1.00 and 0.99). L2 is better in 90% of the resamples, without significance. (Caveat: the bootstrap resamples the 4 speakers, which with 4 speakers creates degenerate sets; in the following tasks the 4 speakers are kept fixed.)
- **Identifiability** (parametric bootstrap: data = fitted law + N(0, 0.015²) per training run, B = 1,000): choosing by AICc, the correct law is chosen in 57–85% of cases; the power law a + b N^−α is confused with a + b/√N in 73% and a + b N^−α + c ln K with L2 in 46%. On the current grid, forms with a free exponent are indistinguishable from 1/√N.
- **Trained** (margin): only K matters. a + c ln K (without N = 2: AICc weight 0.75); adding ln N gives a coefficient of −0.0006. This confirms fact 1 of the theory with the continuous measure.

---

## 3. Probit link and all-or-nothing per speaker

### 3.1 Does Φ(mean/SD) predict the binary fraction?

For each training run × group (45 plain + wide cases):

| Predictor of p | mean bias | RMSE | correlation | max. deviation |
|---|---|---|---|---|
| global Φ(m̄/s) (single normal) | +0.030 | 0.050 | 0.984 | 0.103 |
| per-speaker mixture: mean_s Φ(m̄_s/s_w), s_w = within-speaker SD | +0.003 | **0.018** | 0.996 | 0.062 |

The global probit overestimates copying, especially for the trained voices (bias +0.05): the distribution of the margin is a mixture of speakers, not a normal. With the per-speaker mixture the fit is within sampling error (the median |z| of the global probit is already 0.28 bootstrap SEs). **The binary fraction is, to a good approximation, a deterministic function of the per-speaker mean margins and of a within-speaker SD (~0.034).** This justifies using the margin as the primary measure and converting to "% copied" only at the end.

### 3.2 All-or-nothing

| Group | speakers with copying ≤10% or ≥90% (observed) | expected under a homogeneous binomial | latent ICC (probit, random intercept, ML with Gauss-Hermite quadrature) | ICC of the margin |
|---|---|---|---|---|
| unseen (4) | median 50–75% | ~0–3% | median 0.87 (0.62–1.00) | 0.91 |
| trained (8) | 69–75% | ~0–30% | 0.93 | 0.92 |
| external (38, wide) | 66% (25, 24, 28 of 38) | 0–8% | 0.83–0.91 | 0.82–0.89 |

Overdispersion chi-square of the per-speaker fraction: 584–687 with 37 d.f. for the external voices (p ≈ 0). Histogram of the 38 external voices (≤10% | … | ≥90%): N = 8: 10 | 4 | 3 | 1 | 5 | 15; N = 32: 4 | 3 | 4 | 5 | 2 | 20; N = 128: 4 | 1 | 3 | 2 | 4 | 24. The difficulty belongs to the voice: rank correlation of the per-speaker rate between N = 8–32, 32–128, 8–128: 0.75 / 0.74 / 0.59 (binary) and 0.78 / 0.84 / 0.72 (margin). **Consequence:** the effective sampling unit is the speaker, not the sample; 25 utterances per voice are far more than needed (the utterance has SD 0.007), while 4 voices are far too few.

---

## 4. Laws L1–L4 for the binary fraction and predictions

### 4.1 Methods

- Response: q per training run (fraction not copied). Laws with 3 parameters (4 voices, with K) or 2 (38 voices, N only):
  - L1: a + A/N + B/(NK) (coefficients ≥ 0, NNLS); L1n: a + A/N.
  - L2: a + A/√N + C/√K (≥ 0); L2n: a + A/√N.
  - L3: c N^−α K^−β (grid α ∈ [0.01, 1.5], β ∈ [−0.5, 1.5], c by LS); L3n: c N^−α.
  - L4: q = Φ(a + b ln N + c ln K), nonlinear LS on q ("direct probit").
  - L4m: probit **of the margin**: the mean margin μ(N, K) = a + b ln N + c ln K (or a + b/√N + c/√K, L4m′) is fitted and converted as q = mean_s Φ(−(μ + u_s)/σ), with u_s = fixed effect of each evaluation speaker and σ² = variance within training run×speaker + training run × speaker interaction (section 3.1). The nuisances u_s, σ are estimated with all training runs (minimal leakage in the CV).
- LOCO CV by configuration. AICc/BIC.
- Bootstrap B = 500: (A) "evaluation" — resamples utterances; the 4 speakers stay **fixed** (with 4 speakers, resampling speakers produces degenerate sets; the 4-voice metric is, by definition, about those 4); for the 38 external voices it resamples speakers and utterances, paired across training runs. (B) "evaluation + seed" — adds N(0, s²) per training run, s = 0.077 (total seed SD of the fraction, 4 voices) or 0.054 (seed main effect, used for the 38 voices, where the seed was not measured). (B) is conservative (the observed seed effect is already in the data).

### 4.2 Metric on 4 unseen voices (16 plain training runs)

| Law | RMSE | CV-RMSE | ΔAICc | Predicted copying 247×32 | 1000×32 | 2456×32 | 128×128 |
|---|---|---|---|---|---|---|---|
| L1 | 0.060 | 0.103 | 2.2 | 0.60 | 0.61 | 0.61 | 0.60 |
| L2 | 0.056 | **0.075** | **0** | 0.67 | 0.70 | 0.71 | 0.68 |
| L3 (α = 0.25, β = 0.04) | 0.078 | 0.136 | 10.5 | 0.72 | 0.80 | 0.84 | 0.69 |
| L4 direct | 0.099 | 0.146 | 18.3 | 0.74 | 0.87 | 0.92 | 0.71 |
| L4m (ln margin) | 0.103 | 0.147 | 19.4 | 0.75 | 0.87 | 0.93 | 0.71 |
| L4m′ (√ margin) | 0.070 | 0.079 | 7.1 | 0.68 | 0.71 | 0.72 | 0.69 |

95% intervals (evaluation + seed bootstrap) for the predictions: L1 1000×32: 0.61 [0.55, 0.66]; L2: 0.70 [0.63, 0.77]; L3: 0.80 [0.73, 0.85]; L4: 0.87 [0.78, 0.93]; 2456×32: L1 0.61 [0.55, 0.66], L2 0.71 [0.64, 0.78], L3 0.84 [0.78, 0.89], L4 0.92 [0.84, 0.96]; 128×128: 0.60–0.71 for all laws (L2 0.68 [0.61, 0.76]). These intervals are **conditional on the law**; the uncertainty between laws (0.61 to 0.93 at N = 2,456) is much larger than that of each one.

Bootstrap of the CV-RMSE differences relative to L2 (evaluation + seed): L1 +0.041 [−0.02, +0.18] (P worse 0.84); L3 +0.044 [0.000, +0.076] (0.97); L4 +0.052 [+0.013, +0.079] (0.996); L4m +0.056 [+0.009, +0.11] (0.99); L4m′ +0.0005 [−0.026, +0.029]. But **this is entirely the N = 2 point** (q ≈ 0.97), which the laws with ln N cannot follow. Without N = 2 (14 training runs): CV-RMSE 0.114 (L1), 0.058 (L2), 0.066 (L3), 0.064 (L4), 0.074 (L4m), 0.059 (L4m′); ΔAICc ≤ 1.3 among L2, L3, L4, L4m, L4m′; CV differences relative to L2 with CIs containing 0 for all except L1 (+0.053 [+0.015, +0.098] in the evaluation bootstrap only; with seed, [−0.025, +0.17]). L3 without N = 2 gives α = 0.17, β = 0.08.

**Identifiability** (data generated from each fitted law + N(0, 0.077²) per training run, B = 200): choosing by AICc, the correct law is recovered in 62–80% of cases for L1–L4, but L4m (law on the margin) is taken for L4 in 73% (they are almost the same function) and L4m′ for L2 in 47%. By LOCO CV recovery drops to 28–59%. **The grid data do not distinguish the laws outside N = 2; they only distinguish "with N = 2 included, forms with a floor/square root fit better than log-probit".**

### 4.3 Metric on 38 external voices (3 `-wide` training runs, K = 32, **one seed**)

Caution: these are 3 points and 2 parameters per law (1 residual d.f.); there is no K to vary; the seed was not replicated in this evaluation. AICc with 3 points is not informative (L3n passes exactly through the 3 points: q = 0.744 N^−0.26, because the ratios 0.302/0.434 = 0.696 and 0.211/0.302 = 0.698 are practically equal).

| Law | RMSE | CV (leave 1 of 3 out) | 247 | 1000 | 2456 | 95% CI at 2456 (evaluation) | 95% CI at 2456 (evaluation + seed 0.054) |
|---|---|---|---|---|---|---|---|
| L1n a + A/N | 0.021 | 0.142 | 0.77 | 0.78 | 0.78 | [0.66, 0.88] | [0.62, 0.92] |
| L2n a + A/√N | 0.008 | 0.035 | 0.80 | 0.83 | 0.84 | [0.71, 0.95] | [0.66, 0.98] |
| L3n c N^−α (α = 0.26) | 0.000 | 0.000 | 0.82 | 0.88 | 0.90 | [0.79, 0.97] | [0.72, 0.98] |
| L4n Φ(a + b ln N) | 0.005 | 0.019 | 0.84 | 0.90 | 0.93 | [0.82, 0.99] | [0.71, 1.00] |
| L4mn (margin ln N) | 0.043 | 0.040 | 0.84 | 0.92 | 0.95 | [0.89, 0.99] | [0.85, 1.00] |
| L4mn′ (margin 1/√N) | 0.046 | 0.064 | 0.79 | 0.82 | 0.83 | [0.74, 0.91] | [0.72, 0.94] |

Speaker bootstrap (38) of the CV differences relative to L2n: all differences have CIs containing 0 (e.g., L1n +0.10 [−0.06, +0.22], P worse 0.83; L3n −0.008 [−0.05, +0.04]). Probability of each law being the best: 0.10–0.25 for all of them. **With 3 points from one seed there is no way to distinguish the laws.** Honest prediction for N = 2,456, K = 32 on the 38 voices: between ~0.78 (floor in 1/N) and ~0.95 (probit in ln N), with each law still carrying ±0.1–0.15 of its own uncertainty. The proposal's extrapolation "88% with 1,000 speakers" is that of law L3n, one of six equally compatible laws.

The levels of the two metrics are not comparable: the 4 unseen voices are harder (36–75% against 57–79% copied on the same checkpoint).

---

## 5. Data collapse: x = N^a K^b

### 5.1 Methods

Only the ratio b/a is identifiable in a nonparametric collapse (f(N^a K^b) = g(N^{ca} K^{cb})). For this reason I estimated a and b in four ways:
- (i) power law on the fraction not copied: q = c N^−a K^−b (grid a ∈ [−0.5, 1.5], b ∈ [−0.5, 2], step 0.01);
- (ii) probit: Φ⁻¹(1 − q) = c₀ + a ln N + b ln K (LS);
- (iii) mean margin: m̄ = c₀ + a ln N + b ln K (LS);
- (iv) nonparametric collapse: direction θ in the (ln N, ln K) plane that minimizes the SSE of an increasing isotonic regression of y on x = cos θ ln N + sin θ ln K; b/a = tan θ.

95% intervals: bootstrap (B = 400) of utterances with fixed speakers + seed noise N(0, s²) per training run (s = total measured seed SD: 0.077/0.015 for fraction/margin on the unseen voices; 0.039/0.0125 on the trained ones); and the same also resampling the speakers (4 unseen, 8 trained). Trained: without N = 2 (different references).

### 5.2 Results

| Group | Estimator | a (exponent of N) | b (exponent of K) | b/a |
|---|---|---|---|---|
| unseen, with N = 2 | (i) power | 0.25 [0.19, 0.31] (spk.: [0.07, 1.18]) | 0.04 [−0.03, 0.10] | 0.16 |
| | (ii) probit | 0.41 [0.29, 0.51] | 0.01 [−0.09, 0.12] | 0.03 |
| | (iii) margin | 0.032 [0.026, 0.037] | 0.007 [0.000, 0.013] | 0.22 |
| | (iv) isotonic | — | — | ≈0 (θ = 0.5°) |
| unseen, without N = 2 | (i) power | 0.17 [0.08, 0.25] (spk.: [0.00, 0.68]) | 0.08 [0.01, 0.16] (spk.: [−0.02, 0.33]) | 0.47 |
| | (ii) probit | 0.21 [0.12, 0.33] | 0.12 [0.02, 0.22] | 0.54 |
| | (iii) margin | 0.022 [0.015, 0.028] | 0.012 [0.006, 0.019] | 0.55 |
| | (iv) isotonic | — | — | 0.51 [0.01, 1.33] (θ = 27° [0.5°, 53°]) |
| trained, without N = 2 | (i) power | 0.01 [−0.06, 0.07] (spk.: [−0.11, 0.15]) | 0.46 [0.38, 0.56] (spk.: [0.20, 0.99]) | ≈ 46 |
| | (ii) probit | −0.015 [−0.09, 0.05] | 0.42 [0.35, 0.53] | — |
| | (iii) margin | −0.001 [−0.006, 0.006] | 0.050 [0.044, 0.056] | — |
| | (iv) isotonic | — | — | 2.36 (θ = 67°; the bootstrap does not vary: see below) |

Interpretation:
- **Trained: x ≈ K^0.46 (N^0)**, robust across all forms. The exponent of N has a CI centered on 0 with width ±0.07.
- **Unseen: the collapse depends on N = 2.** With N = 2, the extreme point (q ≈ 0.97) forces x ≈ N^0.25 and K barely enters (b/a ≈ 0–0.2). Without N = 2, the three parametric estimators agree on **b/a ≈ 0.5** (x ≈ N K^0.5, or N^0.17 K^0.08 in the power form), with b > 0 significant when the speakers are kept fixed and not significant when the 4 speakers are also resampled. This means that, between N = 8 and 200, K helps the unseen voices with about half the weight of N (in log) — consistent with the C/√K term of L2 and with 128 × 32 > 128 × 8.
- With speaker resampling the CIs for a, b on the unseen voices cover 0 to ~0.7: with 4 voices, the collapse is descriptive.
- The isotonic collapse is fragile with 14–16 points: the SSE is constant over a wide range of θ (several directions produce the same ordering), and the argmin picks the edge of the range; hence the degenerate (trained) or huge (unseen) CIs. It should not be used as an estimate.
- The 38 voices do not enter here (K = 32 only).

---

## 6. Measurement artifact or real effect (38 external voices)

### 6.1 Proximity of the reference to the default voice

The ideal quantity would be c_s = similarity between the output in the default voice (strength 0) and the reference. It exists only for the 4 unseen voices (`-spk` training runs at strength 0, where `sim_default` = 0.905): c_s = 0.600 / 0.610 / 0.603 / 0.690. I tried to estimate it without strength 0, by regressing `sim_ref` on `sim_default` within speaker and extrapolating to `sim_default` = 0.905; **validation on the 4 voices failed** (estimates 0.66–0.73 against 0.60–0.69, unstable slopes), so this estimate was discarded. Two proxies remain, as requested:

- f0 of the reference (exogenous; the default voice is female);
- the speaker's `sim_default` level (mean over the 3 training runs): how similar the output for that voice is to the default voice. It is **endogenous** (measured on the output and part of the margin itself), so it gives an upper bound on what "attraction toward the default voice" explains.

Rank correlation between the two proxies: 0.69.

### 6.2 How much of the per-speaker failure each thing explains

Sum-of-squares decomposition over the 38 voices × 3 N (means per speaker and N):

| Measure | N | speaker | remainder (speaker × N + noise) | of the speaker effect: R² by f0 | by sim_default | both |
|---|---|---|---|---|---|---|
| q (binary) | 0.06 | 0.74 | 0.21 | 0.28 | 0.55 | 0.55 |
| margin m | 0.13 | 0.72 | 0.16 | 0.30 | 0.59 | 0.59 |
| sim_ref | 0.08 | 0.72 | 0.20 | 0.03 | 0.09 | 0.09 |
| sim_spk | 0.06 | 0.72 | 0.22 | 0.02 | 0.06 | 0.06 |

Sample-by-sample logistic regression (McFadden pseudo-R²): ln N alone 0.031; f0 0.117; speaker's sim_default 0.234; ln N + f0 0.154; ln N + sim_default 0.278; ceiling with speaker as a factor (+ ln N) 0.540.

Interpretation:
- **Voice identity explains ~12 times more than N** (74% against 6% of the variation of q in this range of N).
- On the metric that subtracts the default voice (q, m), about 30% of the speaker effect comes with f0 and ~55–59% with the `sim_default` level. On the metrics that do **not** use the default voice (`sim_ref`, `sim_spk`), f0 and `sim_default` explain almost nothing (2–9%). Therefore the part of the difficulty tied to high-pitched voices / voices close to the default is, to a large extent, **constructed by the criterion** `sim_ref > sim_default`, whereas the difficulty measured only by similarity to the reference is something else (and has no simple predictor here).
- Correlations per N: copying × f0: −0.53, −0.52, −0.37; copying × sim_default: −0.64, −0.67, −0.61; `sim_ref` × f0: +0.14, +0.30, −0.01. High-pitched voices are not reproduced worse in terms of similarity to the reference; they lose on the binary criterion because the output also ends up similar to the default.

### 6.3 The curve in N with measures independent of the default voice

Methodological note: since the same 38 speakers enter all 3 training runs, **any additive per-speaker correction (for example `sim_ref − c_s`) cancels in the differences between N**; the shape of the curve only changes with nonlinear measures (threshold) or measures that depend on something else (`sim_default` changes with N). I used the exponent from the ratio of gains, which is free of the floor: α_g = log₄[(y₃₂ − y₈)/(y₁₂₈ − y₃₂)] (valid for y = y_∞ − A N^−α); CI by bootstrap of the 38 speakers (paired across N, B = 2,000). For the binary measure, also α₀ = −log₁₆(q₁₂₈/q₈) (power law without a floor).

| Measure (mean over the 38) | N=8 | 32 | 128 | gain 8→32 | 32→128 | α_g (95% CI) | P(α_g > 0.5) |
|---|---|---|---|---|---|---|---|
| copying (sim_ref > sim_default) | 0.566 | 0.698 | 0.789 | 0.132 | 0.092 | 0.26 (−0.73, 1.57); α₀ = 0.26 | 0.31 |
| margin m | −0.004 | 0.048 | 0.078 | 0.052 | 0.030 | 0.39 (−0.09, 0.95) | 0.32 |
| per-speaker margin ms | 0.026 | 0.071 | 0.101 | 0.045 | 0.029 | 0.31 (−0.22, 0.81) | 0.22 |
| copying by sim_spk > sim_default | 0.661 | 0.805 | 0.878 | 0.144 | 0.073 | 0.50 (−0.30, 1.80); α₀ = 0.37 | 0.50 |
| **sim_ref** (no default voice) | 0.648 | 0.677 | 0.690 | 0.029 | 0.014 | 0.55 (−0.17, 1.88) | 0.56 |
| **sim_spk** (no default voice) | 0.678 | 0.700 | 0.713 | 0.022 | 0.013 | 0.41 (−0.59, 2.00) | 0.42 |
| −sim_default | −0.651 | −0.629 | −0.612 | 0.022 | 0.017 | 0.21 (−0.95, 1.54) | 0.27 |

Adding seed noise to the 3 points (main effect measured on the 4 voices: 0.054 on the fraction, 0.0127 on the margin), the CI of α_g for copying becomes (−1.9, 2.3) and that for the margin (−0.9, 2.0); P(α_g < 0) = 0.42 and 0.28.

Subgroups (19 + 19 speakers):

| Subgroup | copying 8 → 32 → 128 | sim_ref 8 → 32 → 128 |
|---|---|---|
| f0 below the median (low-pitched) | 0.76 → 0.83 → 0.88 | 0.635 → 0.649 → 0.680 (gain accelerates) |
| f0 above the median (high-pitched) | 0.37 → 0.57 → 0.70 | 0.661 → 0.704 → 0.700 (stops at 32) |
| output far from the default | 0.81 → 0.92 → 0.91 (ceiling) | 0.637 → 0.654 → 0.689 (accelerates) |
| output close to the default | 0.32 → 0.48 → 0.67 (gain accelerates) | 0.659 → 0.699 → 0.692 (stops) |

**Conclusion of task 6.** (i) "Slows down with N" survives as a point signal: on every measure, including those that do not use the default voice (`sim_ref`, `sim_spk`), the gain from 32→128 is smaller than that from 8→32, and the point estimates of α_g lie between 0.2 and 0.55. (ii) **The value 0.26 does not survive as a number**: the measure without the default voice gives 0.41–0.55; and even without seed noise no CI excludes 0 or 1. (iii) The aggregate curve of the binary measure is the sum of two opposite behaviors: low-pitched voices / voices far from the default hit the ceiling of the binary measure already at N = 32 while the continuous similarity keeps rising; high-pitched voices / voices close to the default are still crossing the threshold. The "slow power law without a floor" shape of the binary measure therefore depends on the composition of the voice set relative to the default voice — it is partly an artifact of the criterion. Caveat: the `sim_default` subgroup is endogenous (regression to the mean is possible); the f0 subgroup is exogenous and shows the same pattern.

---

## 7. Effect of R (mean of the codes of 4/16 references)

Paired by sample (same training run, speaker and utterance; R = 1 versus R). Mean difference over training runs and speakers; 95% CI by bootstrap of the speakers (the same in all training runs of the set); Wilcoxon on the per-speaker means (only with ≥ 5 non-tied speakers). Training runs with R = 4: the 4 plain ones with N ≥ 128 and the 3 `-wide`; R = 16: 128×8 and 128×32 s1.

| Group (training runs) | n spk | Δ copying (sim_ref) | Δ per-speaker copying (sim_spk) | Δ margin m | Δ ms | Δ sim_spk | Δ sim_default | Δ WER |
|---|---|---|---|---|---|---|---|---|
| external, wide (3), R=4 | 38 | −0.097 [−0.169, −0.035], p=0.006 | −0.025 [−0.064, +0.008], 14↑ 18↓ | −0.020 [−0.033, −0.009] | −0.002 [−0.010, +0.006] | +0.005 [−0.002, +0.012] | +0.007 [−0.003, +0.017] | **−0.018 [−0.028, −0.009]**, p=0.009 |
| external, plain (4), R=4 | 4 | −0.24 [−0.73, +0.02] | +0.105 [−0.02, +0.32] | −0.042 | +0.009 | +0.029 [+0.003, +0.071] | +0.020 | −0.033 [−0.071, +0.005] |
| unseen, plain (4), R=4 | 4 | −0.28 [−0.58, +0.03] | **+0.13 [+0.01, +0.35]** (3↑ 0↓) | −0.041 | +0.024 [−0.003, +0.052] | +0.030 | +0.005 | +0.002 |
| unseen, wide (3), R=4 | 4 | −0.24 [−0.49, −0.08] | +0.04 [−0.03, +0.14] | −0.034 | +0.008 | +0.030 [+0.005, +0.061] | +0.023 | −0.000 |
| trained, plain (4), R=4 | 8 | −0.14 [−0.31, −0.01] | −0.065 [−0.27, +0.08] | −0.033 [−0.062, −0.005] | −0.016 | +0.003 | +0.018 | −0.016 [−0.036, +0.003] |
| trained, wide (3), R=4 | 8 | −0.13 [−0.34, +0.03] | −0.07 [−0.20, +0.02] | −0.032 | −0.011 | +0.006 | +0.017 | −0.016 |
| unseen, R=16 (2) | 4 | −0.38 [−0.70, −0.05] | +0.09 [−0.08, +0.33] | −0.068 | +0.017 | +0.028 | +0.011 | +0.016 |
| external plain, R=16 (2) | 4 | −0.17 | +0.085 [−0.075, +0.30] | −0.047 | +0.008 | +0.040 [+0.007, +0.074] | +0.033 [+0.016, +0.049] | −0.049 [−0.112, −0.008] |

Interpretation:
- By the per-utterance measure (`sim_ref`), R > 1 always lowers copying — as expected, since the code is no longer that of that particular utterance; this measure is not suitable for evaluating R.
- By the per-speaker measure, the only gain with a CI above 0 is that of the **4 unseen voices** in the 4 plain training runs (+13 points, CI [+1, +35], 3 speakers go up, 0 go down). On the **38 external voices the effect is null or negative** (−2.5 points, CI [−6.4, +0.8]; 14 go up, 18 go down). On the trained voices, negative without significance.
- `sim_spk` rises by ~+0.03 on the 4 unseen/external plain voices, but `sim_default` rises with it (+0.02): the mean code moves toward the center, and the per-speaker margin `ms` barely changes (+0.002 to +0.024).
- The consistent effect is **lower WER** on the external voices: −1.8 points with 38 voices (CI [−2.8, −0.9]; p = 0.009 per speaker); in the other groups the CI contains 0.
- Conclusion: with 38 voices, R = 4 is an intelligibility stabilizer, not an improvement in copying; the gain on the 4 unseen voices is compatible with chance over 4 speakers (lower CI bound +0.01).

---

## 8. Power by simulation and recommended design

### 8.1 Methods

Per-sample generative model on the margin scale (components measured in sections 2.2 and 2.2-seeds):

m = μ(N) + u_s + v_{training,s} + w_{training,utterance} + z_utterance + x_{s,utterance} + ε + δ_seed; copied = m > 0,

with SDs: speaker 0.0764, speaker × utterance 0.0143, utterance 0.0092, training run × utterance 0.0062, residual 0.0314 (ANOVA of the 38 external voices), seed main effect 0.0127 (seed pairs), training run × speaker 0.046 ("conservative": includes the real speaker × N interaction of the 38 external voices, treated as independent noise per training run) or 0.0148 ("seed only": the measured seed × speaker interaction). μ(N) is calibrated so that the population fraction not copied follows each hypothesis, all fitted to the 3 points measured on the 38 voices:

| Hypothesis | q(8) | q(32) | q(128) | q(512) | q(2048) |
|---|---|---|---|---|---|
| H0.26: 0.744 N^−0.26 (no floor) | 0.433 | 0.302 | 0.211 | 0.147 | 0.102 |
| H0.5: 0.145 + 0.828/√N | 0.437 | 0.291 | 0.218 | 0.181 | 0.163 |
| Hfloor: 0.219 + 1.762/N | 0.439 | 0.274 | 0.233 | 0.223 | 0.220 |

In each simulated experiment the evaluation voices are the same at every N (pairing, as in the `-wide` evaluation). Analysis: q̂ per N (mean over seeds); fit of the three 2-parameter forms (c N^−α with free α, a + A/√N, a + A/N); the one with the lowest SSE is chosen. Also: spread of α̂ (form c N^−α) and a test of "there is still a gain from 512 to 2,048" (paired difference per speaker, 95% CI = speaker bootstrap + seed variance assumed known). 400 simulations per cell (200 with 200 voices); "P correct" = probability of choosing the true hypothesis (chance = 0.33). The estimate by the probit mixture of the margin (section 3.1) gave the same results as the binary fraction (±0.03) and is not repeated.

### 8.2 Results (conservative scenario; "seed only" gives the same numbers ±0.05)

| Voices × utterances | Seeds per N | N grid | Training runs | P correct: H0.26 / H0.5 / Hfloor | α̂ under H0.26 (2.5–97.5%) | P(detect gain 512→2048) under H0.26 / Hfloor |
|---|---|---|---|---|---|---|
| 38 × 25 (**current**) | 1 | 8, 32, 128 | 3 | 0.44 / 0.16 / 0.62 | 0.06–0.53 | — |
| 4 × 25 | 1 | 8 … 2048 (5) | 5 | 0.18 / 0.40 / 0.58 | 0.04–1.18 | 0.03 / 0.04 |
| 4 × 25 | 3 | 5 points | 15 | 0.35 / 0.33 / 0.64 | 0.08–1.00 | 0.08 / 0.04 |
| 38 × 25 | 1 | 5 points | 5 | 0.57 / 0.37 / 0.62 | 0.17–0.47 | 0.12 / 0.03 |
| 38 × 25 | 2 | 5 points | 10 | 0.67 / 0.50 / 0.70 | 0.17–0.40 | 0.19 / 0.02 |
| 38 × 25 | 3 | 5 points | 15 | 0.74 / 0.57 / 0.72 | 0.18–0.37 | 0.29 / 0.04 |
| 38 × 25 | 5 | 5 points | 25 | 0.81 / 0.70 / 0.78 | 0.19–0.36 | 0.46 / 0.04 |
| 100 × 10 | 1 | 5 points | 5 | 0.63 / 0.41 / 0.68 | 0.18–0.41 | 0.16 / 0.04 |
| 100 × 25 | 1 | 5 points | 5 | 0.63 / 0.41 / 0.67 | 0.17–0.39 | 0.17 / 0.03 |
| 100 × 25 | 3 | 5 points | 15 | 0.78 / 0.63 / 0.80 | 0.20–0.34 | 0.38 / 0.03 |
| 100 × 25 | 5 | 5 points | 25 | 0.80 / 0.78 / 0.84 | 0.21–0.33 | 0.57 / 0.03 |
| 200 × 10 | 2 | 5 points | 10 | 0.73 / 0.55 / 0.78 | 0.20–0.32 | 0.31 / 0.03 |
| 200 × 25 | 3 | 5 points | 15 | 0.78 / 0.62 / 0.77 | 0.21–0.33 | 0.51 / 0.04 |
| 100 × 25 | 1 | 9 points (8, 16, …, 2048) | 9 | 0.67 / 0.53 / 0.70 | 0.18–0.36 | 0.19 / 0.04 |
| 100 × 25 | 2 | 9 points | 18 | 0.78 / 0.67 / 0.75 | 0.20–0.35 | 0.27 / 0.03 |
| 100 × 25 | 3 | 9 points | 27 | 0.78 / 0.74 / 0.84 | 0.21–0.34 | 0.39 / 0.02 |

Interpretation:
- **The current design (38 voices, 1 seed, N ≤ 128) is close to chance** for choosing the law (0.16–0.62), and α̂ ranges from 0.06 to 0.53 even when the truth is 0.26: the observed 0.26 is compatible with any of the three hypotheses.
- **4 voices are useless** for this, even with 3 seeds and N up to 2,048.
- **Beyond ~40–100 voices, the bottleneck is the seed.** From 38 to 200 voices (3 seeds) the correct-choice rate rises 0.74 → 0.78 under H0.26; from 1 to 5 seeds (38 voices) it rises 0.57 → 0.81. 10 utterances per voice are enough (100 × 10 ≈ 100 × 25), as predicted by the near-zero utterance SD.
- **Distinguishing 0.26 from 0.5 (H0.26 against H0.5) is the hardest part**: the two differ by 3.4 points at N = 512 and 6 points at N = 2,048. With 5 seeds per N and 38–100 voices one reaches 70–80% accuracy; with fewer than 3 seeds, no.
- **Testing for a floor** through the 512 → 2,048 difference has low power (12–57%) even with 5 seeds, with correct type I error (2–5%); fitting the whole curve is more informative than this test.
- A dense grid (9 values of N, 1 seed) yields the same as 5 values with 2 seeds, for the same number of training runs: **what counts is the total number of training runs**, as long as N goes up to ~2,000.

### 8.3 Recommended design for the next GPU rounds

1. **Fixed, paired evaluation set**: 60–100 never-seen voices (LibriTTS test-clean + dev-clean/other), **10 utterances** each, the same in all training runs; stratified by f0 and by distance to the default voice. Evaluation is cheap compared with training.
2. **Seeds**: ≥ 3 per value of N, ideally 5 (this is what pays off most). Grid N ∈ {8, 32, 128, 512, 2048} with fixed K (32); 15–25 training runs. If the budget is ~10 training runs, prefer 5 N × 2 seeds or 9 N × 1 seed over fewer N with more voices.
3. **Re-synthesize each checkpoint 2×** (cheap) to separate synthesis noise from training noise (today ~40% of the "seed" variance of the binary measure is synthesis).
4. **Save strength 0 for each evaluation voice** (once per voice, with any checkpoint): it gives c_s = similarity between the default voice and the reference, and enables measures that do not depend on the `sim_ref > sim_default` threshold (task 6).
5. **Continuous primary measure** (`sim_ref` / `sim_spk` and per-speaker margin), with the binary fraction derived through the probit mixture; report both. For K, a small second axis (K ∈ {8, 128} at N = 128 and 512, 2 seeds) is enough to check b/a ≈ 0.5.
6. **Pre-register** the decision rule: fit of the three forms, criterion by SSE/AICc and CV leaving one N out, with bootstrap over voices + seeds; and declare as "indistinguishable" any result in which the winning hypothesis has < 80% recovery in the corresponding simulation.

---

## What the data support and what they do not

**They support (with the indicated strength):**
1. **Trained voices depend on K, not on N** — on the continuous margin, a + c ln K explains everything; the coefficient of ln N is ≈ 0. Strong.
2. **Unseen voices depend on N** — mean margin: K-only and constant are worse with certainty (bootstrap P ≥ 0.99); the margin rises from −0.11 (N = 2) to +0.03–0.06 (N = 128–200). Strong, but most of the evidence comes from N = 2 versus the rest.
3. **On the 38 external voices, more speakers help from 8 to 128** (copying 57 → 70 → 79%; margin −0.004 → 0.048 → 0.078), and the gain from 32 → 128 is smaller than that from 8 → 32 on all measures, including those that do not use the default voice. Moderate: one seed; the seed SD of the fraction (~0.05) is of the order of the 32 → 128 gain (0.09).
4. **Copying is nearly all-or-nothing per voice** (latent ICC 0.83–0.94; 2/3 of the external voices at ≤ 10% or ≥ 90%), and the difficulty is stable across training runs (ρ ≈ 0.75). Strong.
5. **The binary fraction is a function of the per-speaker margins** (probit mixture, RMSE 0.018). Strong. The margin is ~1.3–2× more precise than the binary measure across seeds, and much more stable to re-synthesis; it is not a solution to speaker and seed noise.
6. **The utterance matters little** (SD 0.007 on the margin); speaker and seed dominate.
7. **R = 4 does not improve copying on the 38 voices**, but it reduces WER by ~1.8 points.
8. **Collapse**: for the trained voices, x ≈ K^0.46 with an N exponent ≈ 0 (±0.07); for the unseen voices between N = 8 and 200, K enters with about half the weight of N (b/a ≈ 0.5, wide CIs; depends on excluding N = 2).
9. **The precision bottleneck is the seed**, not the number of voices, from ~40 voices on: the simulation shows that separating 0.26 from 0.5 requires ≥ 3–5 seeds per N with N up to ~2,000.

**They do not support:**
1. **The exponent 0.26.** It is exactly what two parameters produce when passed through three points from one seed. The ratio of gains has CI (−0.7, 1.6) with speaker resampling alone and (−1.9, 2.3) with seed; measures without the default voice give 0.41–0.55. Claiming "≈ 0.26, d ≈ 4" or extrapolating "82% with 247 / 88% with 1,000 speakers" has no statistical support.
2. **Any choice between L1 (floor in 1/N), L2 (square roots), L3 (power law) and L4 (probit in ln N).** On the 4-voice grid, the preference for L2 comes from the N = 2 point; without it, all of them (except L1, marginally) tie. On the 38 voices, 3 points and 1 d.f. For N = 2,456 the laws predict from 61% to 93% (4 voices) and from 78% to 95% (38 voices).
3. **The existence or absence of a floor.** Not identifiable with N ≤ 200.
4. **"There is no ceiling" or "more speakers yield less and less" as a quantitative claim.** Both readings fit within the noise.
5. **Effects of the variants** (`-swap`, `-contrast`, `-spk`, `-spkdrop`, `-aug`): one seed each; all within ~2 seed SDs of the baseline.
6. **That the external voices are "easier to imitate"** than the unseen voices of the grid: the difference in level comes largely from the distance to the default voice, which the binary criterion rewards.
7. **Interpreting the failure on high-pitched voices as a difficulty of the decoder**: in terms of similarity to the reference (`sim_ref`, `sim_spk`), high-pitched voices are not reproduced worse; they lose on the binary criterion because the output also resembles the female default voice.
