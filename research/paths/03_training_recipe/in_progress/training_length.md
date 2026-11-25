# Training length without KL

**Started:** 2026-11-25 · **Path:** [03](../README.md)

## Goal

Measure where no-KL training saturates, to fix the length of the next experiments. The runs are the 20,000-step ones of [path 01](../../01_scaling_law/in_progress/long_curve_without_kl.md) and [path 02](../../02_decoder_bottleneck/in_progress/film_without_kl_seeds.md): `n128_k32_s2-nokl`, `n241_k32_s1-nokl` and `n241_k32_s1-filmnokl`.

## Predictions registered before the results (2026-11-25)

- **N = 128:** from 10,000 to 20,000 steps, a gain of 0.05 to 0.20 bit (seed 1 gained 0.12).
- **N = 241:** the gain from 10,000 to 20,000 steps is larger than at 128, because more speakers need more training ([R16](../../../results/R16_N_curve_without_kl.md)).
- **Decision:** if the gain at N = 241 exceeds 0.2 bit, the next experiments with N ≥ 128 use 20,000 steps; otherwise, 10,000.
