# Long curve without KL: seeds at 10,000 steps and 20,000 steps at 241 speakers

**Started:** 2026-11-25 · **Path:** [01](../README.md)

## Goal

1. Complete 3 seeds per point of the no-KL curve at 10,000 steps ([R16](../../../results/R16_N_curve_without_kl.md) has 1–3).
2. Find out whether the flattening from 128 to 241 speakers is too few steps.

## Runs (queued or running; status in the inventory)

| Run | Steps | Why |
|---|---|---|
| `n8_k32_s2-nokl`, `n32_k32_s2-nokl`, `n128_k32_s3-nokl` | 10,000 (resume from 5,000) | curve seeds |
| `n241_k32_s2-nokl`, `n241_k32_s3-nokl` | 10,000 (resume from 5,000) | seeds at 241 |
| `n241_k32_s1-nokl` | 20,000 (resumes from 10,000) | is the flattening too few steps? |
| `n128_k32_s2-nokl` | 20,000 (resumes from 10,000) | reference at 128 with the same number of steps |
| `n8/n16/n32/n64_k32_s2/s3-nokl` | 5,000 and 10,000 | seeds of the intermediate points (several already finished) |

## Predictions registered before the results (2026-11-25)

- **Seeds at 10,000 steps:** the new points fall within ±0.15 bit of the current means (8: 1.07; 32: 2.11; 128: 2.84; 241: 2.90).
- **20,000 steps:** at N = 128, seed 2 gives 2.90–3.05 (seed 1 gave 2.96).
- **Decision rule for the flattening:**
  - if it is **too few steps**, N = 241 at 20,000 steps ends at least 0.2 bit above N = 128 at 20,000 steps (≥ 3.16 if 128 stays at 2.96);
  - if it is **something else** (sentences per speaker or a real limit), the difference stays below 0.1 bit.
  - Between 0.1 and 0.2 bit, the conclusion waits for more seeds.

## When it ends

Write the result (next free number in `results/`), add a note to R16 if the mean changes, and move this file to `../done/`.
