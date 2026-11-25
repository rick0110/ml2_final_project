# 03 — Information-theoretic analysis of the saved latents

Scope: CPU only, saved data only (`coverage2/<run>/step_2500/`: `latents.npz`, `metrics.jsonl`, `strength_1.0*/results.csv`). No decoder was saved, so everything here is about the **encoder** (what the 32-dimensional style vector carries) and about the **association** between the geometry of that vector and the copying outcome. Nothing here measures what the decoder does with the vector.

## 0. Summary in numbers

| Quantity | Value (reference runs) | Interpretation |
|---|---|---|
| KL rate per utterance | 64–75 bits (base), 41–61 bits (variants `-spkdrop`, `-swap`, `-contrast`); 32 of 32 dimensions active, 1.35–4.0 bits per dimension | ~93% of the rate is the "precision term" (posterior σ ≈ 0.16), not information |
| Gaussian I(x; z) (aggregate) | 17–29 bits; grows ≈ 1.6 bit per doubling of N and of K | the rate exceeds the information by ~45 bits |
| Information about the voice, corrected Gaussian model (μ) | 17.2–17.7 bits for N = 128–200, K = 32 (± 0.3) | stable from N = 32 to 200: capacity of the encoder, not of the number of voices |
| Same thing in the channel z = μ + σε | 9.8–10.2 bits (N = 128–200) | what the decoder sees in training |
| Discrete lower bound I(μ; identity), LOO classifier | 5.1–5.3 of 6.9 bits (N = 128), 4.6 of 5.0 (N = 32), 2.9 of 3.0 (N = 8) | permutation: ≤ −0.4 bits |
| Directions carrying voice information | PR(Σ_b) ≈ 5.8; whitened PR ≈ 9.7; 6–8 directions with SNR > 1 in z; 90% of the bits in 11–12 directions; spectrum ~ k^−2 | |
| Intrinsic dimension of the voices (noise-free means, N ≥ 124) | TwoNN 9.7–10.6; Levina–Bickel 7.0–7.5; coverage exponent γ = 0.12–0.14 ⇒ d ≈ 7–8 | below the Gaussian null with the same spectrum (TwoNN 13–14, γ ≈ 0.10) |
| Exponent 0.26 | linear coverage would require d ≈ 3.8 (rejected by the measurement); squared error in the distance requires d ≈ 7.7 (compatible) | |
| Default-voice attractor (38 external) | ρ(copy, d_LJ) = 0.34 / 0.43 / 0.63 for N = 8 / 32 / 128; ρ(copy, d_NN) = −0.48 / −0.22 / −0.12 | d_LJ beats d_NN at N = 128 (P = 0.99); d_NN wins at N = 8 |
| Copying transition | d50 ≈ 15.5 whitened units for N = 32 and 128, 19.5 for N = 8 (≈ distance LJ center–LibriTTS center), width ≈ 2 (noise of one utterance = 1) | |

## 1. Data and conventions

- `latents.npz`: μ and log σ² (n × 32) of ~1,500 **training** utterances per run (subsample: LJ + LibriTTS), plus the references (`source = refs`). In the `-wide` runs the external references are **one utterance per speaker** (38 speakers from LibriTTS test-clean). `results.csv`: 25 synthesized utterances per external reference; "copied" = `sim_ref > sim_default` (the project's criterion). Non-copy rates on the external voices: 0.43 / 0.30 / 0.21 (N = 8 / 32 / 128), matching the previous report.
- Bits = log₂. In the script-generated tables the decimal separator is the point. "Whitened" = coordinates W^−1/2 μ, with W the within-speaker covariance (LibriTTS, training). In this metric, a single utterance departs from its speaker's mean with SD 1 per direction (‖e‖ ≈ √32 = 5.7).
- Σ_b = between-speaker covariance, estimated without bias by variance components: Σ_b = (MSB − MSW)/n₀ (unbalanced multivariate ANOVA).
- Main runs: base grid (16 runs) and `-wide` (3). `n8_k32_s1-wide12` ignored. Training variants appear only in section 3.4.
- Uncertainty: bootstrap over speakers (2,000 replicates for the correlations, 1,000 for the pooled model), parametric bootstrap (100 simulations) for the bias of the Gaussian estimators, 80% subsampling of speakers for the neighbor estimators (the bootstrap with replacement breaks TwoNN).

## 2. Task 1 — Rate (KL) per dimension and total

Method: KL(q(z|x) ‖ N(0, I)) per dimension, in bits, averaged over the training utterances; decomposed into a mean term ½μ²/ln2 and a precision term ½(σ² − ln σ² − 1)/ln2. "Gaussian" I(x; z): ½ log₂ det(Cov μ + diag E σ²) − ½ log₂ det diag E σ² (approximation of the aggregate posterior by a Gaussian; it is not a rigorous bound in any sense, because the aggregate is not Gaussian and I used E σ² in place of E log σ²). The Monte Carlo estimator over the empirical mixture saturates at log₂ n = 10.55 and is of no use (column omitted; it is in `out/rate.json`).

| run | N | K | total rate (bits/utterance) | LJ rate | Libri rate | precision term | mean term | bits/dim min–max | Gaussian I(x;z) |
|---|---|---|---|---|---|---|---|---|---|
| n2_k128_s1 | 2 | 128 | 64.4 | 64.9 | 61.0 | 61.8 | 2.7 | 1.35–3.32 | 18.4 |
| n2_k128_s2 | 2 | 128 | 65.5 | 65.7 | 64.0 | 62.8 | 2.7 | 1.42–3.34 | 18.7 |
| n8_k8_s1 | 8 | 8 | 64.3 | 64.2 | 66.5 | 62.3 | 1.9 | 1.47–3.32 | 17.5 |
| n8_k8_s2 | 8 | 8 | 67.8 | 67.8 | 68.5 | 65.8 | 2.0 | 1.46–3.35 | 18.6 |
| n8_k32_s1 | 8 | 32 | 66.1 | 65.7 | 68.7 | 63.4 | 2.7 | 1.42–3.36 | 20.0 |
| n8_k128_s1 | 8 | 128 | 68.6 | 70.0 | 66.0 | 64.6 | 4.0 | 1.44–3.59 | 22.8 |
| n8_k128_s2 | 8 | 128 | 70.2 | 71.5 | 67.7 | 66.2 | 3.9 | 1.41–3.68 | 23.5 |
| n32_k8_s1 | 32 | 8 | 66.9 | 66.7 | 68.2 | 64.4 | 2.5 | 1.40–3.32 | 20.2 |
| n32_k8_s2 | 32 | 8 | 68.4 | 68.2 | 70.0 | 65.8 | 2.6 | 1.59–3.45 | 20.3 |
| n32_k32_s1 | 32 | 32 | 71.3 | 71.9 | 70.0 | 67.6 | 3.6 | 1.51–3.69 | 24.8 |
| n32_k128_s1 | 32 | 128 | 71.6 | 77.6 | 68.6 | 66.7 | 4.8 | 1.51–3.81 | 27.6 |
| n32_k128_s2 | 32 | 128 | 74.5 | 79.3 | 72.1 | 69.8 | 4.7 | 1.56–3.97 | 28.9 |
| n128_k8_s1 | 128 | 8 | 70.2 | 71.0 | 68.6 | 66.6 | 3.5 | 1.46–3.57 | 23.1 |
| n128_k32_s1 | 128 | 32 | 72.4 | 77.3 | 70.0 | 68.0 | 4.4 | 1.52–3.87 | 26.8 |
| n128_k32_s2 | 128 | 32 | 72.6 | 77.7 | 70.1 | 67.3 | 5.3 | 1.55–3.90 | 26.2 |
| n200_k32_s1 | 200 | 32 | 71.6 | 79.1 | 69.1 | 67.3 | 4.3 | 1.55–3.75 | 25.6 |
| n8_k8_s1-aug | 8 | 8 | 68.8 | 68.2 | 69.4 | 65.8 | 3.0 | 1.40–3.22 | 20.1 |
| n8_k128_s1-aug | 8 | 128 | 71.7 | 74.4 | 70.4 | 67.2 | 4.5 | 1.54–3.61 | 25.0 |
| n8_k8_s1-contrast | 8 | 8 | 54.0 | 53.9 | 57.8 | 47.8 | 6.2 | 1.35–2.16 | 3.7 |
| n8_k8_s2-contrast | 8 | 8 | 48.9 | 48.5 | 63.3 | 43.7 | 5.2 | 1.19–1.98 | 3.3 |
| n8_k128_s1-contrast | 8 | 128 | 54.9 | 54.6 | 55.7 | 51.5 | 3.4 | 1.31–2.24 | 14.5 |
| n2_k128_s1-spk | 2 | 128 | 66.6 | 66.8 | 64.9 | 64.1 | 2.4 | 1.53–3.52 | 18.4 |
| n8_k8_s1-spk | 8 | 8 | 63.5 | 63.5 | 63.0 | 61.8 | 1.8 | 1.43–3.35 | 16.5 |
| n8_k128_s1-spk | 8 | 128 | 68.4 | 69.8 | 65.5 | 64.9 | 3.5 | 1.45–3.57 | 21.2 |
| n8_k128_s1-spkdrop | 8 | 128 | 40.7 | 42.1 | 37.9 | 39.4 | 1.3 | 0.91–2.46 | 7.0 |
| n32_k8_s1-spkdrop | 32 | 8 | 45.6 | 45.5 | 46.2 | 44.4 | 1.1 | 1.09–2.48 | 8.2 |
| n32_k128_s1-spkdrop | 32 | 128 | 42.5 | 46.7 | 40.5 | 40.4 | 2.1 | 0.85–2.49 | 9.9 |
| n8_k8_s1-swap | 8 | 8 | 49.9 | 49.9 | 51.1 | 49.5 | 0.5 | 1.25–2.13 | 3.3 |
| n8_k8_s2-swap | 8 | 8 | 50.0 | 49.9 | 54.2 | 49.5 | 0.5 | 1.22–1.99 | 3.2 |
| n8_k128_s1-swap | 8 | 128 | 61.0 | 60.5 | 62.0 | 57.8 | 3.2 | 1.33–2.58 | 14.7 |
| n8_k32_s1-wide | 8 | 32 | 66.6 | 66.5 | 66.8 | 63.9 | 2.7 | 1.44–3.48 | 20.1 |
| n32_k32_s1-wide | 32 | 32 | 72.4 | 74.3 | 68.9 | 68.7 | 3.7 | 1.54–3.81 | 24.8 |
| n128_k32_s1-wide | 128 | 32 | 73.2 | 78.6 | 70.6 | 69.0 | 4.2 | 1.59–3.77 | 27.0 |

Regression over the 16 base runs (standard error in parentheses):

```
taxa total  = 58,0 + 1,39 (0,20)·log₂N + 0,97 (0,24)·log₂K    RMSE 1,5 bit
taxa Libri  = 64,4 + 0,92 (0,26)·log₂N − 0,09 (0,31)·log₂K    RMSE 1,9
taxa LJ     = 50,0 + 2,50 (0,25)·log₂N + 1,99 (0,30)·log₂K    RMSE 1,9
I(x;z) gauss=  7,8 + 1,63 (0,17)·log₂N + 1,51 (0,20)·log₂K    RMSE 1,3
```

(In the block above, `taxa` = rate.)

Interpretation:

1. **The rate barely depends on the data** (64 → 75 bits from N = 2 to 200), and almost all of it (62–70 bits) is the precision term: with σ ≈ 0.16 in each dimension, each one pays ≈ 2 bits for "being precise", whether or not it carries anything that varies across utterances. The mean term is only 2–5 bits. All 32 dimensions are active (minimum 1.35 bit/dim; participation ratio of the rate profile ≈ 31 of 32).
2. The effective information I(x; z) is 17–29 bits: **~45 bits of the rate are mismatch between the aggregate posterior and the prior** (the aggregate occupies a ball of SD ≈ 0.4 inside a prior of SD 1), not information about the utterance.
3. The KL weight was still rising at step 2,500 (0.0225 → 0.0238 over the last 25 steps of `metrics.jsonl`): the models are not at a rate–distortion equilibrium point. The KL in the training log (≈ 1.6–1.9 nat per dimension ≈ 74–86 bits) is somewhat above the one measured on the saved latents (64–75 bits); the difference is plausibly a train/eval mode effect and was not investigated.
4. The rate of the LJ utterances grows with N and K (2.5 and 2.0 bits per doubling); that of the LibriTTS utterances grows little with N and not at all with K. The mechanism of the increase in the LJ rate was not decomposed; the information per LibriTTS utterance (Libri rate) barely changes.

## 3. Task 2 — Information about the speaker

### 3.1 Estimators

- **Gaussian between/within** (the "first pass"): I = ½ log₂ det(T)/det(W) with unbiased T and W ("plug-in"), and the variance-components version I = ½ Σ log₂(1 + λᵢ), λᵢ generalized eigenvalues of (Σ_b, W) with the negative part truncated ("corr."). Channel version: W + diag E σ² (what reaches the decoder in training, z sampled).
- **Bias correction by parametric bootstrap**: I simulate 100 datasets from the fitted model (same number of speakers and of utterances per speaker), re-estimate, and subtract the mean bias (I_corr = 2·I − mean of the simulations). I also estimated Σ_b by the cross-covariance between disjoint halves of each speaker's utterances (noise-free by construction): it gives the same value as the variance-components version, with a difference ≤ 0.5 bit (`out/speaker2.json`, `I_split`).
- **Permutation**: speaker labels shuffled (200 times for the Gaussian, 10 for the classifier).
- **Discrete lower bound**: I(μ; S) ≥ H(S) − CE, with CE the cross-entropy (bits) of a shared-covariance Gaussian classifier under **exact leave-one-out** validation (predictive with W(1 + 1/n_s)), and of a logistic regression with 5 stratified folds. Fano: I ≥ H(S) − h(P_e) − P_e log₂(S − 1).

### 3.2 Results

| run | S | utterances/spk. | log2 S | H(S) | Gauss plug-in | Gauss corr. (mu) | Gauss corr. (z) | perm. plug/vc | acc LDA-LOO | I_CE (LDA) | I_Fano | I_CE (logreg 5-fold) | I_CE perm. | LJ×Libri I/H | dist. LJ–Libri (SD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n2_k128_s1 | 2 | 90.5 | 1.00 | 1.00 | 3.0 | 4.4 ± 1.2 | 3.4 | 0.01 / 0.25 | 1.00 | 1.00 | 1.00 | 0.99 | -0.21 | 0.53/0.53 | 7.3 |
| n2_k128_s2 | 2 | 90.5 | 1.00 | 1.00 | 3.1 | 4.4 ± 1.2 | 3.4 | 0.02 / 0.26 | 1.00 | 1.00 | 1.00 | 0.99 | -0.22 | 0.53/0.53 | 7.9 |
| n8_k32_s1 | 8 | 22.6 | 3.00 | 3.00 | 11.7 | 14.9 ± 0.8 | 10.0 | 0.09 / 0.91 | 0.98 | 2.69 | 2.83 | 2.89 | -1.45 | 0.53/0.53 | 6.1 |
| n8_k128_s1 | 8 | 64.1 | 3.00 | 2.99 | 12.5 | 16.0 ± 1.0 | 10.9 | 0.01 / 0.29 | 0.99 | 2.94 | 2.89 | 2.95 | -0.37 | 0.93/0.93 | 8.9 |
| n8_k128_s2 | 8 | 64.1 | 3.00 | 2.99 | 12.5 | 15.7 ± 1.0 | 10.9 | 0.01 / 0.30 | 0.99 | 2.92 | 2.90 | 2.96 | -0.40 | 0.93/0.93 | 9.1 |
| n32_k8_s1 | 32 | 5.7 | 5.00 | 4.97 | 16.7 | 17.9 ± 0.6 | 10.0 | 0.51 / 2.33 | 0.91 | 3.66 | 4.05 | 3.96 | -5.96 | 0.53/0.53 | 4.8 |
| n32_k8_s2 | 32 | 5.7 | 5.00 | 4.97 | 15.9 | 17.4 ± 0.6 | 9.7 | 0.51 / 2.33 | 0.89 | 3.70 | 3.92 | 3.85 | -5.97 | 0.53/0.53 | 4.7 |
| n32_k32_s1 | 32 | 16.0 | 5.00 | 4.98 | 17.0 | 19.2 ± 0.6 | 11.3 | 0.05 / 0.67 | 0.95 | 4.31 | 4.48 | 4.55 | -1.70 | 0.92/0.93 | 6.0 |
| n32_k128_s1 | 32 | 31.4 | 5.00 | 4.99 | 19.1 | 21.8 ± 0.6 | 12.3 | 0.01 / 0.33 | 0.97 | 4.59 | 4.63 | 4.74 | -0.84 | 0.91/0.91 | 5.5 |
| n32_k128_s2 | 32 | 31.4 | 5.00 | 4.99 | 19.4 | 22.0 ± 0.5 | 12.8 | 0.01 / 0.33 | 0.96 | 4.59 | 4.59 | 4.74 | -0.82 | 0.91/0.91 | 5.7 |
| n128_k8_s1 | 124 | 4.1 | 6.95 | 6.90 | 15.5 | 15.6 ± 0.4 | 9.0 | 0.25 / 1.58 | 0.69 | 3.73 | 3.83 | 4.15 | -6.73 | 0.92/0.93 | 5.7 |
| n128_k32_s1 | 127 | 7.9 | 6.99 | 6.92 | 17.2 | 17.7 ± 0.3 | 10.2 | 0.06 / 0.72 | 0.83 | 5.31 | 5.05 | 5.08 | -3.31 | 0.91/0.91 | 4.5 |
| n128_k32_s2 | 127 | 7.9 | 6.99 | 6.92 | 16.7 | 17.2 ± 0.3 | 9.8 | 0.05 / 0.71 | 0.80 | 5.11 | 4.81 | 4.98 | -3.32 | 0.91/0.91 | 4.4 |
| n200_k32_s1 | 197 | 5.7 | 7.62 | 7.52 | 17.1 | 17.4 ± 0.2 | 9.8 | 0.08 / 0.82 | 0.74 | 4.75 | 4.72 | 5.04 | -4.66 | 0.81/0.81 | 3.7 |
| n8_k32_s1-wide | 8 | 22.6 | 3.00 | 3.00 | 11.3 | 14.4 ± 0.9 | 9.4 | 0.10 / 0.91 | 0.98 | 2.64 | 2.78 | 2.88 | -1.36 | 0.53/0.53 | 6.0 |
| n32_k32_s1-wide | 32 | 16.0 | 5.00 | 4.98 | 17.0 | 19.1 ± 0.6 | 11.1 | 0.05 / 0.67 | 0.95 | 4.29 | 4.40 | 4.55 | -1.73 | 0.92/0.93 | 6.1 |
| n128_k32_s1-wide | 127 | 7.9 | 6.99 | 6.92 | 17.0 | 17.5 ± 0.3 | 10.2 | 0.05 / 0.71 | 0.83 | 5.26 | 5.08 | 5.07 | -3.32 | 0.91/0.91 | 4.6 |

(`dist. LJ–Libri` = √(λ₁ − 1), λ₁ the largest generalized total/within eigenvalue in the binary LJ × LibriTTS problem; it equals √(π(1−π)) times the Mahalanobis distance.)

What can be claimed:

1. **The Gaussian estimate was not biased upward**, as had been feared. The permutation null gives 0.01–0.5 bit (plug-in) and the parametric bias is **negative** (−0.2 to −2.6 bits; larger with few speakers, because Σ_b from S speakers has rank ≤ S − 1). The problem with the first pass is one of **interpretation**, not of bias: the Gaussian number (12–22 bits) exceeds H(S) = log₂ S, so it is not I(z; identity among the S trained speakers). It is the information about a **continuous voice vector** (infinite population of voices) under the Gaussian model, μ = v + e. With N = 8 it is limited by the rank (7 directions), which is why it grows from N = 8 to 32 and then stops: **17.2–17.7 bits in μ and 9.8–10.2 bits in z for N = 128 and 200**, against 17.4–22 bits with N = 32. The encoder does not gain capacity to describe voice with more speakers; what grows with K is the precision (less within-speaker noise).
2. **Discrete bound**: the classifier recovers almost all of the entropy with few speakers (2.9 of 3.0 bits with N = 8; 4.3–4.6 of 5.0 with N = 32 and K ≥ 32) and loses more with many (5.1–5.3 of 6.9 with N = 128, K = 32; 4.7–5.0 of 7.5 with N = 200; 3.7 of 6.9 with N = 128, K = 8). The deficit H − I_CE grows with S (≈ 0.1–0.4 bit with S = 8; 0.4–0.7 with S = 32 and K ≥ 32; 1.6–1.8 with S = 127; 2.8 with S = 197). In terms of I_CE/log₂ S, for K ≥ 32: 0.90–0.98 (S = 8), 0.86–0.92 (S = 32), 0.73–0.76 (S = 127), 0.62 (S = 197). A single utterance identifies the speaker among 127 with 83% accuracy (LDA) / 72% (logistic).
3. **Default voice × LibriTTS**: perfect separation (accuracy 1.00 in all runs; I = H of the binary label, 0.53–0.93 bit depending on the proportion). The default voice is an isolated cluster, at a distance ≈ 15–17.5 whitened units from the center of the LibriTTS voices — more than the typical distance between two LibriTTS voices (≈ 11 = √(2 tr Σ_b) whitened, noise-free) and three times the nearest-neighbor distance among 127 (≈ 5).

### 3.3 How many directions carry the voice

| run | S | PR(Σb) | PR(W^-1/2 Σb W^-1/2) | PR(SNR) | dir. SNR>1 (mu) | dir. SNR>1 (z) | k 50% bits (z) | k 90% (z) | k 99% (z) | k 90% (mu) | power-law α (Σb, 16 dir.) | predicted gain R=4 (mu/z, bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n8_k32_s1 | 8 | 2.8 | — | 4.0 | 7 | 6 | 3 | 5 | 7 | 6 | 2.37 [1.40, 2.73] | 6.5 / 2.6 |
| n8_k128_s1 | 8 | 3.8 | — | 4.6 | 7 | 6 | 3 | 6 | 7 | 6 | 1.76 [1.06, 2.12] | 6.6 / 2.8 |
| n8_k128_s2 | 8 | 3.7 | — | 4.3 | 7 | 7 | 3 | 6 | 7 | 6 | 1.82 [1.25, 2.07] | 6.6 / 2.8 |
| n32_k8_s1 | 32 | 5.0 | 7.7 | 7.6 | 12 | 7 | 4 | 9 | 16 | 14 | 2.60 [2.80, 3.43] | 14.7 / 4.1 |
| n32_k8_s2 | 32 | 5.0 | 7.5 | 7.4 | 13 | 6 | 3 | 10 | 16 | 14 | 2.61 [2.78, 3.60] | 14.5 / 4.0 |
| n32_k32_s1 | 32 | 5.6 | 8.7 | 8.7 | 13 | 7 | 4 | 11 | 18 | 15 | 2.37 [2.55, 3.06] | 15.5 / 4.7 |
| n32_k128_s1 | 32 | 6.1 | 9.7 | 9.7 | 15 | 8 | 4 | 11 | 19 | 16 | 2.15 [2.27, 2.78] | 16.9 / 4.9 |
| n32_k128_s2 | 32 | 6.1 | 9.7 | 9.7 | 14 | 8 | 4 | 12 | 19 | 16 | 2.14 [2.28, 2.78] | 16.9 / 5.2 |
| n128_k8_s1 | 124 | 5.2 | 8.4 | 8.4 | 13 | 6 | 4 | 11 | 19 | 18 | 2.24 [2.26, 2.43] | 16.3 / 4.4 |
| n128_k32_s1 | 127 | 5.8 | 9.8 | 9.8 | 14 | 7 | 4 | 12 | 21 | 19 | 2.03 [2.05, 2.20] | 17.8 / 4.9 |
| n128_k32_s2 | 127 | 5.7 | 9.8 | 9.8 | 13 | 6 | 4 | 11 | 21 | 19 | 2.10 [2.12, 2.29] | 17.4 / 4.8 |
| n200_k32_s1 | 197 | 5.8 | 8.9 | 8.9 | 14 | 7 | 4 | 11 | 21 | 20 | 2.06 [2.06, 2.20] | 17.8 / 4.6 |
| n8_k32_s1-wide | 8 | 3.0 | — | 4.0 | 7 | 6 | 3 | 5 | 7 | 6 | 2.46 [1.46, 2.77] | 6.5 / 2.5 |
| n32_k32_s1-wide | 32 | 5.6 | 8.7 | 8.7 | 13 | 7 | 4 | 11 | 18 | 15 | 2.34 [2.50, 3.09] | 15.4 / 4.6 |
| n128_k32_s1-wide | 127 | 5.9 | 9.6 | 9.6 | 13 | 8 | 4 | 12 | 22 | 19 | 1.99 [2.01, 2.16] | 17.5 / 5.0 |

- The spectrum of Σ_b is concentrated: participation ratio ≈ 5.8 (raw) and ≈ 9.7 (whitened, that is, relative to the within-speaker noise); power-law decay λ_k ∝ k^−α with α ≈ 2.0–2.1 over the first 16 directions for N ≥ 128 (≈ 2.1–2.6 for N = 32). The bootstrap CI for α is shifted upward relative to the point estimate (the bootstrap with repeated speakers broadens the spectrum); treat α as ≈ 2 ± 0.3.
- In z (with the posterior noise), **6–8 directions have SNR > 1**; 50% of the voice bits are in 4 directions, 90% in 11–12, 99% in ~21. In μ (without the posterior noise) there are 13–15 directions with SNR > 1.
- Cumulative bits in z (N = 128, `-wide`): 1.6 (1 dir.), 3.0 (2), 5.3 (4), 6.9 (6), 7.9 (8), 9.2 (12), 9.7 (16), 10.0 (32).

### 3.4 Training variants (separate)

| run | I gauss corr. μ / z (bits) | I_CE / H | rate (bits) |
|---|---|---|---|
| n8_k128_s1 (base) | 13.4 / 9.2 | 2.94 / 2.99 | 68.6 |
| -spk | 12.3 / 8.1 | 2.89 / 2.99 | 68.4 |
| -spkdrop | 8.9 / 3.0 | 2.76 / 2.99 | 40.7 |
| -contrast | 23.0 / 14.9 | 2.99 / 2.99 | 54.9 |
| -swap | 23.4 / 14.3 | 2.85 / 2.99 | 61.0 |
| -aug (58 pseudo-speakers) | 18.4 / 11.4 | 2.93 / 4.84 | 71.7 |
| n32_k128_s1 (base) / -spkdrop | 19.5 / 11.3 → 12.7 / 4.3 | 4.59 → 4.32 / 4.99 | 71.6 → 42.5 |

The voice information in z varies by a factor of 3–5 across variants (3 to 15 bits), and according to [`coverage_theory.md`](coverage_theory.md) (fact 4) none of them changed the dependence of copying on N. This is consistent with section 5: **the voice bits in the latent are not the bottleneck** (weak inference: I do not have the per-variant copy rates with CIs here).

## 4. Task 3 — Intrinsic dimension of the voice manifold

Method: per-speaker means of μ for the trained speakers, in whitened coordinates. The raw means are noisy (with ~8 utterances per speaker the noise has SD 1/√8 per direction, and a noise-only null already gives TwoNN ≈ 11), so I remove the noise with the BLUP predictor v̂_s = Σ_b (Σ_b + W/n_s)^−1 (m_s − m̄). Estimators: TwoNN (Facco et al. 2017, maximum-likelihood form), Levina–Bickel k = 5 (MacKay–Ghahramani version), and a **coverage exponent** measured inside the latent space: median distance from a speaker to the nearest of m other randomly drawn ones, r_NN(m) ∝ m^−γ, fitted over m = 4…64 (d_cov = 1/γ). Each number comes with a **Gaussian null**: same Σ_b, same number of speakers and of utterances, same procedure (simulated 100 times; 20 for the coverage exponent). I only use S ≥ 30.

| run | S | TwoNN raw means | TwoNN noise only | TwoNN noise-free means [80% subsample CI] | TwoNN Gaussian null [95% CI] | Levina–Bickel k=5 noise-free | LB null | coverage exponent γ (r_NN ∝ m^-γ) | γ Gaussian null |
|---|---|---|---|---|---|---|---|---|---|
| n32_k8_s1 | 32 | 5.6 | 9.0 | 5.2 [4.5, 8.3] | 9.3 [6.7, 12.7] | 4.9 [3.9, 5.5] | 5.8 | — | — |
| n32_k8_s2 | 32 | 6.4 | 8.7 | 5.9 [4.9, 9.9] | 9.4 [6.3, 12.5] | 5.7 [4.7, 6.4] | 5.8 | — | — |
| n32_k32_s1 | 32 | 7.2 | 9.7 | 8.8 [6.5, 11.7] | 10.4 [7.1, 14.5] | 5.8 [4.8, 6.2] | 6.4 | — | — |
| n32_k128_s1 | 32 | 6.6 | 10.2 | 9.2 [6.8, 12.7] | 10.9 [7.5, 14.8] | 6.1 [5.0, 6.6] | 6.7 | — | — |
| n32_k128_s2 | 32 | 6.9 | 10.0 | 9.7 [7.3, 13.9] | 11.0 [7.9, 15.7] | 5.9 [5.1, 6.7] | 6.8 | — | — |
| n128_k8_s1 | 124 | 8.9 | 11.4 | 10.1 [8.5, 12.3] | 12.5 [10.5, 15.4] | 6.7 [5.9, 6.9] | 8.3 | 0.128 [0.119, 0.159] → d≈7.8 | 0.099 ± 0.009 |
| n128_k32_s1 | 127 | 7.7 | 11.3 | 10.6 [9.3, 12.3] | 13.6 [11.3, 16.1] | 7.3 [6.6, 7.5] | 8.8 | 0.122 [0.107, 0.153] → d≈8.2 | 0.096 ± 0.007 |
| n128_k32_s2 | 127 | 7.8 | 11.5 | 10.5 [9.1, 12.4] | 13.2 [10.7, 16.3] | 7.5 [6.5, 7.5] | 8.8 | 0.138 [0.111, 0.156] → d≈7.3 | 0.095 ± 0.008 |
| n200_k32_s1 | 197 | 8.9 | 11.6 | 9.7 [8.8, 11.0] | 13.8 [11.9, 15.8] | 7.0 [6.4, 7.0] | 9.2 | 0.134 [0.110, 0.155] → d≈7.4 | 0.097 ± 0.008 |
| n32_k32_s1-wide | 32 | 7.7 | 9.8 | 7.7 [6.0, 10.0] | 10.3 [7.1, 13.8] | 5.7 [4.8, 6.2] | 6.4 | — | — |
| n128_k32_s1-wide | 127 | 8.0 | 11.4 | 10.4 [9.0, 11.4] | 13.2 [11.2, 15.7] | 7.1 [6.4, 7.4] | 8.8 | 0.122 [0.103, 0.155] → d≈8.2 | 0.100 ± 0.009 |

Interpretation:

- The local estimators give d ≈ 7–10 (Levina–Bickel ≈ 7, TwoNN ≈ 10, coverage ≈ 7–8), consistently **below** what a Gaussian with the same spectrum would give (≈ 9–14). There is a weak signal of nonlinear structure (clusters, curvature, or a heavy tail) that reduces the effective dimension by 2–3 units relative to the linear spectrum, but there is no very low-dimensional manifold (d = 2–4 lies outside all intervals with S ≥ 124).
- With S = 32 the intervals are wide (TwoNN 5–14) and the numbers are not informative.
- The numbers come from a sample of 124–197 voices from LibriTTS train-clean, in one latent space per run. Neighbor estimators with ~130 points in 32 dimensions are strongly biased downward, which is why the null is the comparison that matters, not the absolute value.

### 4.1 The exponent 0.26 and the dimension each theory requires

Non-copying on the 38 external voices: 0.43 / 0.30 / 0.21 ⇒ p ∝ N^−0.26 (fit without a floor; three points, one seed).

| Theory (failure p as a function of N) | Predicted exponent | d required for 0.26 | Compatible with the measurement (d ≈ 7–10; γ ≈ 0.12–0.14)? |
|---|---|---|---|
| Linear coverage: p ∝ r, r ∝ N^−1/d | 1/d | 3.8 | **No**: would require γ ≈ 0.26; measured 0.12–0.14 (CI 0.10–0.16) |
| Coverage with squared error: p ∝ r², r ∝ N^−1/d | 2/d | 7.7 | **Yes**: 2γ = 0.24–0.28 (CI 0.21–0.31) |
| Nonparametric regression with smoothness β (Stone): N^−2β/(2β+d) | 2β/(2β+d) | 5.7·β | Yes for β ≈ 1.2–1.8 |
| Coverage at fixed ε (mass outside the balls) | exponential decay | — | Does not describe a power law |
| Two-level law a + A/N (section 3 of the coverage report) | 1 (no floor) | — | Only with a floor a > 0; not a question of d |

The most discriminating measurement is γ inside the latent space of the N = 128 model: **the distance to the nearest trained neighbor falls as m^−0.12…−0.14, half the exponent of the failure**. This rules out the "failure proportional to distance" form at the measured dimension and is compatible with "failure proportional to the square of the distance" (which is what one expects from a smooth, second-order error around each trained voice). Strong caveats: (i) the coincidence 2γ ≈ 0.26 comes from a 3-point fit; (ii) across runs, the (noise-corrected) distance from the external reference to the nearest trained neighbor falls faster, as N^−0.23 (table below), but this correction is crude (the minimum over noisy neighbors is biased downward, more so the larger N is) and each run has its own encoder; with this number, linear coverage would become compatible again; (iii) per speaker, d_NN does **not** predict failure at N = 128 (section 6), so coverage is not the whole mechanism at high N.

| run | d_NN (whitened) | noise floor √E‖e‖² | corrected d_NN [95% CI] | d_LJ | dist. LJ center–Libri center | mean copy |
|---|---|---|---|---|---|---|
| n8_k32_s1-wide | 9.64 | 5.78 | 7.37 [6.18, 8.62] | 20.0 | 17.5 | 0.57 |
| n32_k32_s1-wide | 8.02 | 5.83 | 5.26 [4.58, 5.94] | 17.1 | 14.7 | 0.70 |
| n128_k32_s1-wide | 7.42 | 6.00 | 3.90 [3.12, 4.76] | 18.9 | 16.5 | 0.79 |

## 5. Task 4 — Rate–distortion and minimal latent

### 5.1 Precision needed for copying to flip

On the `-wide` runs, I fitted a binomial logistic regression (25 utterances per speaker) of copying on d_LJ (whitened distance from the reference to the mean of the default voice). The transition lies at **d50 ≈ 19.5 / 15.5 / 15.6** units (N = 8 / 32 / 128; 95% CI 17.6–21.3 / 13.2–16.4 / 11.5–17.7), with **width 1/β ≈ 2** (CI ≈ 0.7–5.6). For comparison: the noise of one utterance is 1 per direction; the spread of d_LJ across the 38 speakers is 2.1–2.8; the distance LJ center → LibriTTS center is 14.7–17.5. That is: a voice is copied when the reference falls farther from the default voice than the center of the trained voices does, and the decision is sensitive to ≈ 2 units along a single axis, twice the noise of one utterance.

In bits: the per-speaker outcome (copies / does not copy) carries at most 1 bit, and little of it is predicted by the geometry — leave-one-speaker-out validation reduces the predictive loss from 0.77 → 0.63 bit per sample for the best model (N = 128; table in section 6). The voice identity in the latent uses ≈ 10 bits (z) / 17 bits (μ) in 6–12 directions; the copy decision depends, to first order, on ≈ 1–2 bits along one axis (distance to the default voice), plus F0.

### 5.2 More bits at the input do not change copying

The Gaussian model predicts that averaging R = 4 references adds **+2.5 / +4.6 / +5.0 bits** of voice information in z (N = 8 / 32 / 128; +6.5 / +15 / +18 bits in μ). The coverage report (8.8) measured, on the same 38 voices, per-speaker copying 66 → 64%, 81 → 79%, 88 → 84% (R = 1 → 4). Five more bits about the voice did not produce more copying. Together with 3.4, this supports that **the bottleneck is not the amount of voice information in the code**, but what the decoder is able to render (the hypothesis of the coverage report). This is an inference by association: the `sim_spk` criterion used at R = 4 also measures something different from `sim_ref`.

### 5.3 Falsifiable predictions (minimal latent and KL weight)

Two hypotheses compete and make different predictions:

- **H_bits** (copying needs all of the voice information the code holds): copying on the 38 external voices, training 128 × 32, stays within ±10 points of 79% as long as the code retains ≥ 90% of the voice bits in z, that is, **latent with d ≥ 12**; it drops by more than 10 points with **d ≤ 4** (≈ 53% of the bits).
- **H_axis** (copying, by the `sim_ref > sim_default` criterion, depends mainly on the distance to the default voice): even **d = 4** keeps copying within ±10 points.

Registered prediction (my bet): **d_min ≈ 6–8** (the number of directions with SNR > 1 in z): copying unchanged (±10 points) with d = 8, drop > 10 points with d = 2. It is falsified if d = 12 already loses > 10 points (the decoder uses more than the voice: prosody, channel) or if d = 2 does not lose (H_axis).

KL weight: the current rate (≈ 72 bits) contains ≈ 45 bits of "precision without information". Prediction: **raising the final KL weight until the measured rate drops to ≈ 25–30 bits does not change copying (±10 points)**; copying drops by > 10 points only when the rate falls below ≈ 12–15 bits (≈ voice information in z + 2–5 bits). I cannot convert this into a weight β without training (the weight was still ramping, 0.024 at step 2,500); the suggested sweep is β_final ∈ {0.024, 0.1, 0.3, 1}, reading the rate from `metrics.jsonl` itself. The `-spkdrop` runs (rate 41–46 bits, 3–5 bits of voice in z) are the closest existing data point to this sweep.

"Reference swap" prediction: if geometry is the operative variable, swapping each external speaker's reference utterance for another utterance of the same speaker (noise 1 per direction) changes the majority outcome for **≈ 15% / 15% / 4% of the speakers** (≈ 5–6 / 5–6 / 1–2 of 38, N = 8 / 32 / 128; `out/flip.json`). Much more than that (> 30%) means that the "all or nothing per voice" does not come from the reference latent.

## 6. Task 5 — Default-voice attractor

Predictors per external speaker (one reference each): d_LJ (whitened distance to the LJ mean), d_NN (to the mean of the nearest trained speaker), log F0 of the reference. Outcomes: fraction copied (25 utterances), mean margin `sim_ref − sim_default`, `sim_ref`, `sim_default`. Spearman with CI by bootstrap over speakers; partial rank correlation; binomial logistic regression with leave-one-speaker-out validation (loss in bits per sample; lower is better); pooled model with a run fixed effect, predictors standardized within each run, bootstrap by speaker (the same 38 in all runs).

| run | copy | ρ(copy, d_LJ) | ρ(copy, d_NN) | ρ(copy, log F0) | P(abs ρ_LJ > abs ρ_NN) | partial ρ d_LJ given F0 | partial ρ d_NN given d_LJ | ρ(sim_ref, d_LJ) | ρ(sim_default, d_LJ) |
|---|---|---|---|---|---|---|---|---|---|
| n8_k32_s1-wide | 0.57 | 0.34 [-0.02, 0.62] | -0.48 [-0.68, -0.20] | -0.53 [-0.75, -0.26] | 0.24 | 0.12 | -0.41 | -0.14 [-0.44, 0.19] | -0.61 [-0.83, -0.29] |
| n32_k32_s1-wide | 0.70 | 0.43 [0.10, 0.67] | -0.22 [-0.55, 0.14] | -0.52 [-0.76, -0.23] | 0.77 | 0.35 | -0.33 | -0.36 [-0.62, -0.06] | -0.67 [-0.84, -0.42] |
| n128_k32_s1-wide | 0.79 | 0.63 [0.39, 0.78] | -0.12 [-0.41, 0.21] | -0.37 [-0.63, -0.08] | 0.99 | 0.60 | -0.14 | 0.09 [-0.23, 0.41] | -0.72 [-0.84, -0.53] |

| run | intercept only | d_LJ | d_NN | log F0 | d_LJ+d_NN | d_LJ+F0 | d_LJ+d_NN+F0 | d50 (whitened) | width 1/β | overdispersion | all-or-nothing fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n8_k32_s1-wide | 1.016 | 0.904 | 0.883 | 0.889 | 0.836 | 0.865 | 0.829 | 19.5 [17.6, 21.3] | 2.4 [1.4, 5.6] | 18 | 0.66 |
| n32_k32_s1-wide | 0.908 | 0.790 | 0.889 | 0.797 | 0.797 | 0.726 | 0.726 | 15.5 [13.2, 16.4] | 1.7 [1.0, 3.8] | 15 | 0.63 |
| n128_k32_s1-wide | 0.769 | 0.673 | 0.803 | 0.708 | 0.701 | 0.626 | 0.633 | 15.6 [11.5, 17.7] | 2.0 [0.7, 4.1] | 16 | 0.74 |
| pooled (run fixed effect, LOSO) | 0.898 | 0.771 | 0.869 | 0.791 | 0.798 | — | 0.726 | — | — | — | — |

Pooled (three runs, standardized coefficients, 95% CI by speaker bootstrap): d_LJ **+0.91 [0.47, 1.80]**, d_NN −0.20 [−0.85, 0.27], log F0 **−0.85 [−1.43, −0.29]**. Within-run Spearman, pooled: d_LJ 0.47 [0.26, 0.62], d_NN −0.27 [−0.48, −0.05], F0 −0.48 [−0.67, −0.23].

Interpretation:

1. **The answer to "does d_LJ predict better than d_NN?" depends on N.** With N = 8, d_NN predicts better (ρ = −0.48 against 0.34; P(|ρ_LJ| > |ρ_NN|) = 0.24); with N = 32 it is a tie (0.77); with N = 128 d_LJ clearly wins (0.63 against −0.12; P = 0.99). In the aggregate, d_LJ wins (LOSO loss 0.771 against 0.869 bit/sample; intercept 0.898). This is the pattern of the coverage report (section 8.9), now with CIs and with a whitened metric.
2. **F0 and d_LJ are partially redundant** (ρ(d_LJ, F0) = −0.48 / −0.27 / −0.26). With N = 8, F0 explains almost everything (partial d_LJ | F0 = 0.12); with N = 128 it is the opposite (partial 0.60 against −0.28 for F0). The best model at N = 32 and 128 is d_LJ + F0.
3. **Attractor or measurement artifact? They cannot be separated with the saved data.** d_LJ does not predict `sim_ref` (ρ = −0.14 / −0.36 / +0.09) and strongly predicts `sim_default` (−0.61 / −0.67 / −0.72). The effect of d_LJ on "copying" passes entirely through the similarity of the output to the default voice. This is exactly what the `sim_ref > sim_default` criterion would produce even with perfect copying (a reference similar to LJ produces an output similar to LJ), and also what an attractor would produce. The test that separates them: measure `sim_default` **of the reference itself**. An artifact predicts `sim_default(output) ≈ sim_default(reference)`; an attractor predicts `sim_default(output) > sim_default(reference)`, with the difference growing as d_LJ decreases.
4. Copying is nearly all-or-nothing per voice: overdispersion 15–18 times the binomial; 63–74% of the speakers have copying ≤ 10% or ≥ 90%.

## 7. Task 6 — What cannot be measured and how to measure it

Without decoder checkpoints, the following cannot be measured:

- **I(z; output)** and how much of each latent direction the decoder actually uses (Jacobian, per-direction sensitivity);
- **the minimal latent directly**: project z onto the first k (between/within) directions and synthesize — with the checkpoint this tests the prediction of section 5.3 **without training anything**;
- **bits per parameter**: how much of the decoder's capacity is spent on each trained voice, which is the quantity that the reading "the decoder does not render new voices" calls for;
- **the attractor × artifact separation** (needs `sim_default` of the reference, or a listening test);
- **within-speaker variability for the external speakers**: there is only one reference utterance per external speaker in the latents; the R = 4 average was not saved.

Experiments that do measure them (from cheapest to most expensive):

1. Save the final checkpoint of each training run (zero cost). With it: projection onto k directions (5.3), "strength" sweep along the LJ axis, per-direction decoder Jacobian.
2. Extract latents for the 25 utterances of each external speaker (encoder only) and repeat the synthesis with another reference: tests the reference-swap prediction (≈ 15% / 15% / 4%).
3. `sim_default` of the references themselves (speaker encoder only): separates attractor from artifact.
4. Sweep of d ∈ {2, 4, 8, 12, 32} and of β_final, training 128 × 32, evaluation on the 38 external voices: tests H_bits × H_axis and the rate prediction.
5. Bits per parameter: train the same model with decoders of different widths (½×, 1×, 2× parameters) and N ∈ {32, 128, 512}; measure the number of trained voices copied above a threshold vs. parameters, and the description length of the decoder (quantize the weights to b bits until copying of the trained voices drops, or a prequential code). The "capacity" prediction is that the exponent in N changes with the width; the "coverage" prediction is that it does not.

## 8. Falsifiable predictions (list)

1. **Minimal latent**: with d = 8, copying on the 38 external voices (128 × 32) stays at 79% ± 10 points; with d = 2 it drops > 10 points. (Falsified if d = 12 loses > 10 or d = 2 does not lose.)
2. **KL weight**: rate reduced to 25–30 bits without loss of copying (±10 points); loss only below ~12–15 bits.
3. **Reference swap**: changes the majority outcome for ≈ 15% / 15% / 4% of the speakers (N = 8 / 32 / 128); > 30% falsifies the role of the reference latent.
4. **R = 16**: adds ≈ +7.6–8.3 bits of voice in z (Gaussian model), and per-speaker copying changes by < 5 points (as with R = 4).
5. **Larger N (≥ 512)**: ρ(copy, d_NN) ≈ 0 and ρ(copy, d_LJ) ≥ 0.5; failures concentrate at d_LJ below the LJ center–LibriTTS center distance.
6. **Exponent**: if failure is ∝ r² with d ≈ 7–8, the non-copy exponent in N stays at 0.26 ± 0.05 also between N = 128 and 512 (≈ 0.21 → 0.15); if it is the linear-coverage behavior measured across runs (γ ≈ 0.23), also ≈ 0.23–0.26 — **this test does not separate the two**; what separates them is the shape of per-speaker failure as a function of the distance to the neighbor (linear vs. quadratic), which calls for dozens of voices per distance bin.

## 9. Limitations

- Everything is about the encoder and associations. No causal claim about the decoder.
- The Gaussian estimators assume μ = v + e with v and e Gaussian and W common to all speakers. The "bits about a continuous voice vector" numbers depend on this assumption; the discrete bound (classifier) does not, but it is only a lower bound.
- With N = 8 (and N = 2), Σ_b has rank ≤ 7 (≤ 1): the estimates are of the projection onto that subspace, not of the population.
- Each run has its own encoder: distances across runs are not strictly comparable (whitening by W helps, it does not solve it).
- The 38 external voices have one reference each; one seed per N in the `-wide` runs; the copy criterion is binary and comes from a single speaker verifier.
- The exponent 0.26 is a 3-point fit without a floor; with a floor the exponent changes (the coverage report fits laws with a floor).
- The coincidence 2γ ≈ 0.26 is suggestive, not a demonstration.

## Appendix — reproduction

Scripts (all run with `python3 -I`, paths as arguments), in [`analyses/`](analyses/):

```
python3 -I info_rate.py      ../coverage2 out        # tarefa 1 → out/rate.json, out/rate_table.md
python3 -I info_speaker.py   ../coverage2 out 200 200 # tarefa 2-3 (estimadores, permutação, LOO, TwoNN bruto) → out/speaker.json
python3 -I info_speaker2.py  ../coverage2 out 100     # viés paramétrico, metades, direções, ID sem ruído, cobertura → out/speaker2.json
python3 -I info_attractor.py ../coverage2 out 2000    # tarefas 4-5 → out/attractor.json
python3 -I info_flip.py      out                      # previsão de troca de referência → out/flip.json
python3 -I info_summary.py   out                      # tabelas → out/summary_tables.md
```

(Comments in the block above: task 1; tasks 2–3 (estimators, permutation, LOO, raw TwoNN); parametric bias, halves, directions, noise-free ID, coverage; tasks 4–5; reference-swap prediction; tables.)

`info_speaker_lib.py` contains the functions of `info_speaker.py` without side effects; `info_common.py` the utilities.
