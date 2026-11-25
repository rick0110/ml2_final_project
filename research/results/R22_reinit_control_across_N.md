# R22 — Re-initializing the VAE heads costs 0.35–0.78 bit at every N, and the 8-dimensional latent stays below the control everywhere

- **Date:** 2026-11-25 (cluster)
- **Path:** [05 — voice representation](../paths/05_voice_representation/README.md)
- **Verdict:** the predictions mostly hit (one value just below the predicted band); path 05 can close

## Question

Is the cost of re-initializing the VAE heads ([R13](R13_smaller_latent_and_control.md)) similar at every N, so the 8-dimensional latent can be compared with the right control along the curve?

## Setup

`-zr32` (32-dimensional latent, heads re-initialized), original recipe, K = 32, 2,500 steps, N ∈ {32, 128, 241}. Predictions registered in [`paths/05_voice_representation/in_progress/reinit_control.md`](../paths/05_voice_representation/in_progress/reinit_control.md).

## Result

| N | Original (3 seeds) | `-zr32` | Re-init cost | `-z8` |
|---|---|---|---|---|
| 32 | 1.23 ± 0.02 | 0.88 ± 0.03 (2) | 0.35 | 0.62 ± 0.07 (3) |
| 128 | 1.75 ± 0.04 | 0.97 ± 0.06 (3) | 0.78 | 0.57 ± 0.05 (3) |
| 241 | 1.76 ± 0.08 | 1.02 ± 0.04 (2) | 0.74 | 0.42 (1) |

Voice bits (seeds in parentheses).

## Against the registered predictions

- "The re-initialization cost lies between 0.4 and 0.9 bit at N = 32 and at 241": **241 hit** (0.74), **N = 32 just below** (0.35).
- "zr32 stays above z8 at every N": **hit**.

## Conclusion

- After re-initializing the heads, 2,500 steps recover only 0.88–1.02 bits whatever N is: the bits barely grow with N, unlike the intact model. The re-initialized latent is the bottleneck in these runs.
- The 8-dimensional latent costs a further 0.26–0.60 bit on top of the re-initialization, and its cost grows with N. So a small latent limits how many voices can be told apart, which is what the information view predicts.
- Path 05 closes here; the latent size only returns if a long no-KL run recovers from the re-initialization.

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant base zr32 z8 --step 2500 --K 32`.
