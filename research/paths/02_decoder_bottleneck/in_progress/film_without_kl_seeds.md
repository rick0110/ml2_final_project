# FiLM without KL: more seeds and 20,000 steps

**Started:** 2026-11-25 · **Path:** [02](../README.md)

## Goal

In [R17](../../../results/R17_film_without_kl.md) FiLM adds +0.17 to +0.45 bit at 5,000 steps, but at 10,000 the advantage nearly vanishes (one seed). Two readings:

- **FiLM raises the ceiling:** the advantage reappears with more seeds and with 20,000 steps;
- **FiLM only speeds things up:** both converge to the same value.

## Runs (queued or running)

| Run | Steps | Why |
|---|---|---|
| `n128_k32_s2-filmnokl`, `n241_k32_s2-filmnokl` | 10,000 (resume from 5,000) | seed 2 at 10,000 |
| `n16_k32_s2/s3-filmnokl`, `n64_k32_s2/s3-filmnokl` | 5,000 and 10,000 | seeds of the intermediate points |
| `n241_k32_s1-filmnokl` | 20,000 (resumes from 10,000) | ceiling at 241 |
| `n16_k32_s1/s2/s3-film`, `n64_k32_s1/s2/s3-film` | 2,500 (original recipe) | complete the FiLM curve of R12 |

## Predictions registered before the results (2026-11-25)

- **Main decision rule:** compare `n241_k32_s1-filmnokl` with `n241_k32_s1-nokl` at 20,000 steps.
  - If the difference is **≥ 0.2 bit**, FiLM raises the ceiling.
  - If it is **< 0.1 bit**, it only speeds up learning.
  - Between 0.1 and 0.2 bit, wait for more seeds.
- **At 10,000 steps, with 2 seeds:** FiLM's mean advantage at N ≥ 64 lies between 0 and +0.2 bit.
- **With the original recipe, at 2,500 steps:** FiLM at N = 64 lands between 1.8 and 2.0 bits (original: 1.66).
