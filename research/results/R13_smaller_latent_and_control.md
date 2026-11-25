# R13 — 8- and 2-dimensional latents: most of the loss comes from re-initializing the VAE heads

- **Date:** 2026-11-23/24 (cluster)
- **Path:** [05 — voice representation](../paths/05_voice_representation/README.md)
- **Verdict:** the prediction "an 8-dimensional latent changes nothing" failed, but the control shows the comparison was confounded

## Question

How many dimensions does the style latent need? Registered prediction (section 8.10 of the theory note): with 8 dimensions, copying and bits unchanged; with 2, they drop.

## Setup

`-z8`, `-z2`: a smaller latent, with the VAE heads (fc1–fc3) re-initialized when loading the 32-dimensional checkpoint. Control `-zr32`: a 32-dimensional latent with the same heads re-initialized. Original recipe, K = 32, 2,500 steps.

## Result

| Variant | N | Seeds | Voice bits | Copied (%) | WER (%) |
|---|---|---|---|---|---|
| original (32, not re-initialized) | 128 | 3 | 1.75 ± 0.04 | 79 ± 1 | 11.1 ± 0.3 |
| `-zr32` (32, re-initialized) | 128 | 3 | 0.97 ± 0.06 | 72 ± 3 | 14.6 ± 0.6 |
| `-z8` | 128 | 3 | 0.57 ± 0.05 | 63 ± 6 | 15.4 ± 2.1 |
| `-z2` | 128 | 2 | 0.06 ± 0.01 | 51 ± 12 | 23.2 ± 0.8 |
| `-z8` | 8 | 3 | 0.45 ± 0.02 | 49 ± 4 | 17.1 ± 1.7 |
| `-z8` | 32 | 3 | 0.62 ± 0.07 | 57 ± 9 | 16.0 ± 1.8 |
| `-z8` | 241 | 1 | 0.42 | 72 | 13.2 |

## Conclusion

- **Re-initializing the heads costs 0.78 bit** (1.75 → 0.97) in 2,500 steps: the model does not recover the latent in time.
- **The size itself costs 0.40 bit** (zr32 0.97 → z8 0.57, same re-initialization). The 2-dimensional latent zeroes the bits (0.06) and doubles the WER.
- With z8 the bits do not even grow with N (0.45 → 0.62 → 0.57 → 0.42): the small latent becomes the bottleneck.
- The prediction "z8 changes nothing" is wrong even after discounting the re-initialization, but a fair comparison would need longer training (without the KL ramp, [R15](R15_kl_ramp.md)).

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant base z8 z2 zr32 --step 2500 --K 32`.
