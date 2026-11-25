# R15 — The KL ramp was limiting training: with it, longer training hurts; without it, it helps a lot

- **Date:** 2026-11-24 (cluster)
- **Path:** [03 — training recipe](../paths/03_training_recipe/README.md)
- **Verdict:** confirmed with 3 seeds

## Question

Why did training for 5,000 steps instead of 2,500 lower the voice bits (1.75 → 1.42 at N = 128)?

## Setup

In the original recipe the KL weight stays at 0 until step 2,000 and then rises by 0.05 every 1,000 steps. Control `-nokl`: KL weight always 0. K = 32.

KL over training in the original recipe (run `n241_k32_s1`):

| Step | KL weight | KL (nats) |
|---|---|---|
| 2,000 | 0 | 3.40 |
| 2,500 | 0.025 | 1.56 |
| 3,000 | 0.05 | 1.07 |
| 4,000 | 0.10 | 0.72 |
| 4,975 | 0.15 | 0.58 |

## Result

| Recipe | N | 2,500 steps | 5,000 steps | 10,000 steps |
|---|---|---|---|---|
| original | 128 | 1.75 ± 0.04 (3) | 1.42 (1) | — |
| original | 241 | 1.76 ± 0.08 (3) | 1.43 ± 0.07 (3) | — |
| FiLM | 128 | 2.06 ± 0.02 (3) | 1.64 (1) | — |
| no KL | 128 | 1.81 ± 0.06 (3) | 2.44 ± 0.05 (3) | 2.84 ± 0.13 (2) |
| no KL | 241 | — | 2.53 ± 0.06 (3) | 2.90 (1) |

Voice bits (seeds in parentheses). WER falls with training in both recipes: in the original, at N = 241, from 11.9% to 8.2%; without KL, at N = 128, from 12.0% (2,500) to 8.3% (5,000) and 6.8% (10,000).

## Conclusion

- The KL ramp compresses the latent (KL falls from 3.4 to 0.6 nats between steps 2,000 and 5,000) and strips the voice out of it: with the KL rising, training longer **loses** about 0.33 bit.
- Without the ramp, training longer **gains**: +0.63 bit from 2,500 to 5,000 steps and another +0.40 by 10,000, at N = 128.
- Up to step 2,500 both recipes are nearly equal (1.75 vs 1.81), as expected: the ramp only starts at step 2,000.
- Consequence: all later experiments use `-nokl` and longer training ([R16](R16_N_curve_without_kl.md)–[R18](R18_training_length.md)).

## Caveats

Without KL the VAE is no longer regularized; sampling styles from the prior no longer makes sense (it is not used for cloning, which always starts from a reference).

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `metrics.jsonl` of each run on the machines and on Drive.
