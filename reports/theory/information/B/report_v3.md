# Voice capacity C(N, K) measured on the audio: curve, decomposition and information chain (v3)

Version 3, 17/11/2026, round 3 of the critique. Versions v1 and v2 were kept. CPU only, no training; no repository changed; saved
data read with `python3 -I`. Folder: `/tmp/claude-0/-home-user-ml2-final-project/4c11fa8d-6b5f-5505-bd7b-deebf5dd9907/scratchpad/teoria2/B/`.

Legend: **[MEASURED]** no model; **[MODEL]** uses measured covariances in a Gaussian model; **[CONJECTURE]**.
"CE" is the cross-entropy lower bound with β chosen leaving the voice out (LOVO). "Fano" is the bound from the accuracy rate.
Both are **lower** bounds on $I(\hat Y;S)$.

## 0. Summary

| set of voices (ceiling) | N=2 | N=8 | N=32 | N=128 | N=200 | slope (bits per doubling) |
|---|---|---|---|---|---|---|
| **38 external, never heard (5.25), CE** | — | **0.75** [0.28, 1.20] | no audio | **1.99** [1.35, 2.60] | — | **s_N = 0.31** [0.17, 0.47] |
| 38 external, Fano | — | 0.31 | — | 0.89–1.13 | — | 0.15–0.21 |
| 12 common = 8 trained + 4 new (3.58), CE | — | 1.38–2.29 | 1.63–2.58 | 2.13–2.30 | 2.05 | s_N = +0.10 [−0.09, +0.30]; s_K = +0.10 [−0.03, +0.24] |
| 8 trained (3.0), CE | — | 1.79–2.99 | 1.83–2.26 | 1.77–2.12 | 1.92 | s_N = −0.11; s_K = +0.17 |

- **Decomposition** (38 external voices, Gaussian in the 6 principal directions):
  - $H(\hat Y)$ rises **+1.44 bit** [+0.68, +2.32] from N=8 to 128;
  - $H(\hat Y\mid S)$ stays at **+0.07** [−1.2, +1.5];
  - $I$ rises **+1.37**.
  - **The gain with N is in the spread of the reachable voices (coverage), and the per-voice noise does not change**. This is compatible with the
    constant temperature of analysis A.
  - Caveat: in the total (scalar) variance, the per-voice noise measured as R=1 × R=4 also grows a little (+0.032).
- **Chain** (38 voices, Fano):

  | link | N=8 | N=128 |
  |---|---|---|
  | real speech [MEASURED] | 4.2–4.7 | 4.2–4.7 |
  | μ [MODEL] | 5.1 | 5.2 |
  | z [MODEL] | 4.4 | 4.8 |
  | output [MEASURED] | 0.3 | 0.9–1.1 |

  Almost all of the loss is in the **decoder**: ~4 bits at N=8 and ~3.7 at N=128. The KL costs ~0.5 bit on this scale.
- **Extrapolation** (log-linear, 38 external voices, CE):
  - $C(512)=2.61$ [1.86, 3.54] and $C(2048)=3.23$ [2.25, 4.45].
  - The output would carry as much as real speech (4.96 bits CE) at $N^\*\approx10^5$ speakers (CI $10^{3.6}$ to $10^{8.2}$).
- **Variants**: the ECAPA vector **alone (strength 0) carries 0.0 bit** in the `-spk` runs; the decoder ignores it (measured).
  When the style is removed in half of training (`-spkdrop`), ECAPA starts to carry 0.6–1.3 bit. Across runs, copying and C
  correlate −0.49 (Spearman, 12 runs).
- **Strength 0.6** lowers C in 8 of 10 runs (by up to −1.4 bit at K=8). **R = 4 and R = 16** change C by between −0.40 and +0.31 bit, with no
  consistent direction: this confirms v2.

---

## 1. Data and method

### Contents of the archives

Listed with `tar tzf`; extracted into `wavs_ext/` and `wavs_ext2/`.

| archive | runs | conditions | utterances |
|---|---|---|---|
| `wavs_wide` | n8_k32-wide, n128_k32-wide (50 refs: 38 external + 8 trained + 4 new) | 1.0; 1.0_r4 | s0 |
| `wavs_n128` | n128_k8, n128_k32 (16 refs: 8 + 4 + 4 external) | 1.0; 0.6; r4; r16 | s0, s1 |
| `wavs_n200` | n200_k32 (16) | 1.0; 0.6; r4 | s0, s1 |
| `wavs_spk_cov`, `_cov2` | n2_k128-spk, n8_k128-spk, n8_k8-spk (12 refs) | 0.0; 0.6; 1.0 | s0, s1 |
| `wavs_spkdrop_cov`, `_cov3` | n32_k128-spkdrop, n32_k8-spkdrop, n8_k128-spkdrop (12) | 0.0; 0.6; 1.0 | s0, s1 |
| `wavs_aug` | n8_k8-aug, n8_k128-aug (12) | 0.6; 1.0 | s0, s1 |

There is no audio from any base run with N = 8 or 32 other than `n8_k32-wide`. There is also no N = 32 for the 38 external voices. The
N = 8 and 32 points of the grid come from **variants** (`-spk`, `-aug`, `-spkdrop`), which in `04` do not differ from the base run by more than ~2 seed
SDs.

### Common set of voices

The **12 references** (8 trained + 4 new) are the same in all runs with N ≥ 8.
- At N = 2 the trained voices are other utterances from 2 speakers; only the 4 new ones are common.
- Real audio of these 12 voices: LibriTTS train-clean-100, only the 12 speakers, extracted from `openslr.elda.org/resources/60` into
  `libritts_tc100/` (3,538 files).
  - The first attempt failed because the names in the archive start with `./`; it was redone.
- The 38 external voices use the test-clean from v2.

### Estimators

- Resemblyzer embedding, output identified against the centroid of up to 30 real utterances of each voice (excluding the reference
  utterance); Fano and CE-LOVO.
- Bootstrap over voices. For slopes, the bootstrap is paired: the same voices are resampled in all runs.
- The **ceiling** is $\log_2$ of the number of voices. With 12 voices the ceiling is 3.58 bits and the verifier on real speech already gives 3.52, so the
  small set **saturates and compresses differences**. The 38 external voices (ceiling 5.25; real speech 4.2–5.0) are the main measure.

### Scripts

`b20_embed2.py` (embeddings), `b21_curva.py` (consistency), `b22_cadeia_latente.py` (μ and z for the 8 trained voices), `b23_latente38.py`
(μ and z for 38 voices), `b24_lei.py` (curve, slopes, q × C, strength and R, decomposition on 12 voices), `b25_decomp38.py`
(decomposition on the 38), `b26_extrap.py` (extrapolation), `b27_extra.py` (real speech, trained voices, strength 0).
Outputs in `out/b2x.txt`.

## 2. The C(N, K) curve [MEASURED]

Fidelity, strength 1.0, R = 1. More than one utterance per voice where available (s0 and s1).

| run | N | K | variant | 12 voices CE [95% CI] | Fano | 8 trained CE | 4 new CE | q (trained + new) |
|---|---|---|---|---|---|---|---|---|
| n2_k128-spk | 2 | 128 | spk | — | — | — | 0.22 | 0.52 |
| n8_k8-aug | 8 | 8 | aug | 1.63 [0.72, 2.68] | 1.16 | 1.79 | 0.30 | 0.57 |
| n8_k8-spk | 8 | 8 | spk | 1.67 [0.80, 2.57] | 1.33 | 2.26 | 0.57 | 0.61 |
| n8_k32-wide | 8 | 32 | base | 1.38 [0.59, 2.02] | 1.16 | 1.92 | (1 utterance) | 0.36 |
| n8_k128-aug | 8 | 128 | aug | 2.29 [1.44, 3.57] | 1.51 | 2.51 | 0.63 | 0.30 |
| n8_k128-spk | 8 | 128 | spk | 1.98 [0.99, 2.82] | 1.51 | 2.99 | 0.74 | 0.26 |
| n8_k128-spkdrop | 8 | 128 | spkdrop | 1.63 [0.43, 2.79] | 1.71 | 2.98 | 0.08 | 0.29 |
| n32_k8-spkdrop | 32 | 8 | spkdrop | 1.63 [1.00, 2.42] | 0.86 | 1.83 | 0.61 | 0.42 |
| n32_k128-spkdrop | 32 | 128 | spkdrop | 2.58 [2.12, 3.45] | 1.91 | 2.26 | 0.84 | 0.24 |
| n128_k8 | 128 | 8 | base | 2.30 [1.55, 3.34] | 2.13 | 1.77 | 1.88 | 0.47 |
| n128_k32 | 128 | 32 | base | 2.13 [1.50, 2.68] | 1.16 | 1.95 | 0.68 | 0.25 |
| n128_k32-wide | 128 | 32 | base (same checkpoint) | 2.16 [1.45, 3.24] | 1.51 | 2.12 | 0.65 | 0.23 |
| n200_k32 | 200 | 32 | base | 2.05 [1.35, 2.93] | 1.33 | 1.92 | 0.56 | 0.34 |
| **38 external** n8_k32-wide | 8 | 32 | base | **0.75** [0.28, 1.20] | 0.31 | | | q = 0.43 |
| **38 external** n128_k32-wide | 128 | 32 | base | **1.99** [1.35, 2.60] | 0.89–1.13 | | | q = 0.21 |

Law $I=c+s_N\log_2N+s_K\log_2K$ on the 12 voices (10 runs, N ≥ 8, without `-wide`):
- $c=1.01$, $s_N=+0.10$ [−0.09, +0.30], $s_K=+0.10$ [−0.03, +0.24]; residual RMSE 0.23 bit (within the per-run CIs).
- Separating the voices:
  - **trained**: $s_N=-0.11$, $s_K=+0.17$ — their C depends on K, not on N, like copying (K^0.46);
  - **new** (only 4, ceiling 2): $s_N=+0.12$, $s_K=-0.04$.

What this says:
1. **On the never-heard voices the log-linear law has a slope of 0.31 bit per doubling of N** [0.17, 0.47]. There are only 2 points (8 and 128),
   so the "log" form is **not tested**; a power law ($C\propto N^{0.35}$) fits equally well.
2. On the set of 12, the slope in N is small and not significant. This is expected: 2/3 of it are trained voices, whose
   capacity does not depend on N, and the 3.58-bit ceiling is close (real speech: 3.52).
   **The 12-voice set is too small to measure the curve of the new voices**: 4 new voices, ceiling 2 bits, CI of ±1 bit.
3. In K the signal only appears on the trained voices (+0.17 bit per doubling): K improves the memorization of the trained voices, not
   generalization.

## 3. Decomposition $I(\hat Y;S)=H(\hat Y)-H(\hat Y\mid S)$ — the test that reconciles the two theories

**Question.** Analysis A sees no change in the per-voice response temperature across N; here C grows by +1.24 bit. The two things
are only compatible if the gain is in $H(\hat Y)$ (more distinct output voices: coverage), with $H(\hat Y\mid S)$ constant
(fixed per-voice noise).

**Method.**
- Unit embeddings projected onto a common PCA basis, fitted on all the outputs plus the real centroids.
- Gaussian entropies $\tfrac12\log_2\det(2\pi e\Sigma)$:
  - $\Sigma_{\rm tot}$ of the outputs;
  - $\Sigma_{\rm within}$ of the difference between two outputs of the same voice. On the 38 external voices, R = 1 × R = 4, because there is only utterance s0;
    on the 12 voices, utterances s0 × s1.
- Scalar version (no PCA): between-voice and within-voice variance.
- CI: bootstrap over voices, paired across N.

**38 external voices, N = 8 → 128** (`b25`):

| quantity | N = 8 | N = 128 | difference [95% CI] |
|---|---|---|---|
| $H(\hat Y)$, k = 6 | −8.40 | −6.96 | **+1.44** [+0.68, +2.32] |
| $H(\hat Y\mid S)$, k = 6 | −13.53 | −13.46 | **+0.07** [−1.24, +1.54] |
| $I_G$, k = 6 | 5.13 | 6.50 | +1.37 [−0.10, +2.69] |
| $H(\hat Y)$, k = 3 | −2.45 | −1.96 | +0.49 [−0.03, +0.99] |
| $H(\hat Y\mid S)$, k = 3 | −5.75 | −6.48 | −0.72 [−1.54, +0.25] |
| spread between voices (scalar) | 0.178 | 0.221 | +0.043 [+0.018, +0.066] |
| per-voice noise R1 × R4 (scalar) | 0.095 | 0.127 | +0.032 [+0.017, +0.048] |
| error output → real centroid | 0.667 | 0.546 | −0.121 [−0.156, −0.089] |

(The Gaussian $I_G$ is not a bound; at k = 6 it overestimates because $\Sigma_{\rm within}$ comes from only 38 pairs.)

**Reading.**
- The information gain with N is a **gain in spread**: the entropy of the outputs rises by 1.4 bit in the principal directions.
- In those same directions the conditional entropy does not change (+0.07, wide CI, centered on zero), and the error relative to the real voice falls.
- **This is the signature of coverage with fixed per-voice noise**, the same as the constant temperature of analysis A.
- Honest caveat: in the total variance (scalar, all 256 dimensions) the R1 × R4 "noise" grows by 34%. This noise is not only from the
  decoder: it measures how much the output moves when the code changes. In a decoder that follows the code more (larger N), it
  grows, and pure synthesis noise cannot be separated with one utterance per condition.
- The conclusion holds for the dominant subspace: **constant per-voice noise, growing spread**.

**12 voices (8 trained + 4 new)**, k = 6, regression over the 10 runs:
- $H(\hat Y)$: +0.07 bit per doubling of N and +0.25 per doubling of K;
- $H(\hat Y\mid S)$: +0.16 per doubling of N and −0.04 per doubling of K.

Nothing grows with N, again consistent with the capacity of the trained voices not depending on N. The effect of K is in $H(\hat Y)$:
memorizing the trained voices better spreads their outputs.

## 4. Information chain (data processing inequality)

The estimator is the same in all links: nearest-centroid identification + Fano. In the latent links, whitened Euclidean distance
instead of cosine.

**38 external voices** (ceiling 5.25):

| link | N = 8 | N = 128 | source |
|---|---|---|---|
| real speech (held-out utterance × centroid) | 4.21–4.68 | 4.21–4.68 | [MEASURED] `b13` |
| encoder μ (one utterance) | 5.14 | 5.23 | [MODEL] `b23`: $\Sigma_b$, $W$ of the run; true centroids |
| z = μ + posterior noise | 4.36 | 4.76 | [MODEL] `b23` |
| decoder output | 0.31 | 0.89–1.13 | [MEASURED] `b13` |

**8 trained voices** (ceiling 3.0), all measured:

| link | N = 8 | N = 128 |
|---|---|---|
| real speech | 3.00 | 3.00 |
| μ (held-out utterance) | 2.94 | 2.81 |
| z | 2.43 | 2.18 |
| output | 1.49 (wide) | 2.11 (wide); 1.49 (plain); n200: 1.23 |

**Where the bits are lost.**
- For the new voices, almost all in the decoder: 4.1 bits at N = 8, 3.7–3.9 at N = 128 (from z to the output).
  - The encoder and the KL lose ≤ 0.8 bit.
  - N changes the z link little (+0.4 bit, due to the covariance of each run) and recovers +0.6 to +0.8 bit in the decoder link.
- For the trained voices the decoder loses ~0.5–1.3 bit and **does not improve with N**.

The μ link lying above real speech for the 38 voices does not violate the inequality: the estimators are not identical (latent with
true centroids of a Gaussian model against audio with estimated centroids).

## 5. Extrapolation (38 external voices, CE; `b26`, 2,000 paired replicates)

| | point | 95% CI |
|---|---|---|
| slope $s$ | 0.310 bit per doubling | [0.166, 0.467] |
| $C(512)$ | 2.61 | [1.86, 3.54] |
| $C(2048)$ | 3.23 | [2.25, 4.45] |
| real speech (same estimator) | 4.96 | |
| $N^\*$ (output = real speech) | $\approx1.0\times10^5$ | $10^{3.6}$ to $10^{8.2}$ |

**Registered prediction**: with N = 2,048 (K = 32), on the same 38 voices, $C\in[2.25, 4.45]$ bits (CE), point 3.23.
- **Falsifies the log-linear law**: $C(2048)<2.25$, which indicates saturation/a floor, or $C(512)>3.54$.
- **Falsifies the coverage reading**: $H(\hat Y\mid S)$ changing by more than 1 bit between N = 128 and 2,048 in the same directions.

Even at the optimistic end of the CI, matching real speech requires thousands of speakers, and at the central point ~10⁵ (more than the whole of
LibriTTS, 2,456).

## 6. C and copying across runs and variants (negatives)

- Across the 12 runs with 12 voices: **Spearman(q, C) = −0.49**. More capacity, fewer non-copied voices.
  - Exception: n128_k8 has high C (2.30) and high q (0.47).
  - On the 4 new voices, n128_k8 has 1.88 bit and q = 0.45: with K = 8 the outputs are distinct but pulled toward the default voice.
  - Capacity and copying are not the same thing (v2, §1.4).
- **Pretrained speaker vector** (`b27`, strength 0 = no VAE style; in the `-spk` runs only ECAPA remains):

  | run | strength 0.0 (ECAPA only) | strength 1.0 |
  |---|---|---|
  | n2_k128-spk | 0.00 | 0.42 |
  | n8_k8-spk | 0.00 | 1.33 |
  | n8_k128-spk | 0.00 | 1.51 |
  | n8_k128-spkdrop | 1.33 | 1.71 |
  | n32_k8-spkdrop | 0.01 | 0.86 |
  | n32_k128-spkdrop | 0.59 | 1.91 |

  (Fano, 12 voices.) **The `-spk` decoder passes 0 bit through ECAPA**, the direct proof that the vector is ignored (v2, 6.4: reconstruction
  optimum with a redundant input). Removing the style in half of training, ECAPA passes 0.6–1.3 bit, but the total C with
  style (1.7–1.9) does not exceed the other runs with the same N. This is consistent with copying not improving.
- `-aug` with N = 8: C = 1.63 (K = 8) and 2.29 (K = 128), within the range of the other N = 8 runs. The pseudo-speakers do not add
  capacity.
  - Caveat: the base runs n8_k8 and n8_k128 have no saved audio, so the comparison is with other variants of the same N, not with
    the exact pair.
- **Unified reading** [supported CONJECTURE]: the interventions that did not improve copying did not increase the voice capacity of the
  decoder on new voices. Capacity only rises with the number of real voices seen.

## 7. Strength 0.6 and reference averaging on the larger set (`b24`, 16 voices where there are external ones, otherwise 12)

| run | 1.0 | 0.6 | R = 4 | R = 16 |
|---|---|---|---|---|
| n8_k8-aug | 1.63 | **0.25** | | |
| n8_k8-spk | 1.67 | **0.30** | | |
| n8_k128-aug | 2.29 | 1.55 | | |
| n8_k128-spk | 1.98 | 2.06 | | |
| n8_k128-spkdrop | 1.63 | 1.75 | | |
| n32_k8-spkdrop | 1.63 | 0.99 | | |
| n32_k128-spkdrop | 2.58 | 2.41 | | |
| n128_k8 | 2.13 | **0.68** | 2.04 | 2.36 |
| n128_k32 | 2.08 | 1.44 | 2.08 | 2.09 |
| n200_k32 | 1.85 | 1.50 | 2.16 | |
| n8_k32-wide (38 + 12) | 1.36 | | 1.58 | |
| n128_k32-wide | 2.47 | | 2.07 | |

(CE, bits; ceiling 4 with 16 voices.)
- **Strength 0.6**: drops in 8 of 10 runs. The drop is large with K = 8 (−1.4 to −1.5 bit) and small with K = 128 (−0.2 to +0.1).
  The style scale knocks down capacity precisely where per-voice memorization is weak.
- **R = 4 / 16**: −0.40 to +0.31, mean ≈ +0.02 bit. Confirms: **averaging references does not increase capacity**.

## 8. Status

| claim | status |
|---|---|
| C grows with N on the new voices: 0.31 bit per doubling [0.17, 0.47] | MEASURED (2 points, 38 voices) |
| Log-linear form | not tested (missing N = 32 or ≥ 512 with audio) |
| C of the trained voices depends on K, not on N | MEASURED (12 voices; wide CIs) |
| Gain in $H(\hat Y)$, $H(\hat Y\mid S)$ constant (principal subspace) | MEASURED, wide CI on $H(\hat Y\mid S)$; does not hold for the total variance |
| The loss along the chain is in the decoder | MEASURED (output, real speech), MODEL (μ, z for 38 voices) |
| ECAPA ignored = 0 bit | MEASURED |
| Extrapolation $N^\*\sim10^5$ | CONJECTURE (log-linear) with a CI of 4 orders of magnitude |
