# K = 64 without KL, and the K grid with the original recipe

**Started:** 2026-11-25 · **Path:** [04](../README.md)

## Goal

1. Find out whether the flattening from 128 to 241 speakers ([R16](../../../results/R16_N_curve_without_kl.md)) comes from too few sentences per speaker.
2. Complete the N × K grid with seeds, to fit a law with both factors.

## Runs (queued or running)

| Run | Steps | Why |
|---|---|---|
| `n241_k64_s1/s2-nokl`, `n128_k64_s1/s2-nokl` | 10,000 | larger K at large N |
| `n64_k64_s1-nokl`, `n32_k64_s1-nokl` | 10,000 | larger K at small N |
| `n128_k16_s1/s2/s3`, `n241_k8_s1/s2` | 2,500 (original) | K grid |
| `n8_k128_s2/s3`, `n32_k128_s2/s3` | 2,500 (original) | grid seeds |
| `n8_k128_s2-nokl`, `n32_k128_s2-nokl`, `n128_k8_s2-nokl`, `n241_k8_s1-nokl` | 5,000 | no-KL seeds |

Note: `coverage_manifest.py` warns when a speaker lacks K sentences and uses the ones it has. With K = 64, some of the 241 speakers may have fewer; check the warning in the log before interpreting.

## Predictions registered before the results (2026-11-25)

- **If the flattening is too few sentences:** `n241_k64` at 10,000 steps ends at least 0.2 bit above `n241_k32` (≥ 3.1), and the gap between 128 and 241 widens.
- **If it is not:** `n241_k64` lands between 2.9 and 3.05.
- **At fixed N:** the gain from K = 32 to 64 is smaller than that from N = 128 to 256 at fixed K. The latter can only be measured in [path 01](../../01_scaling_law/in_progress/beyond_241_speakers.md); until then, compare with the slope per doubling of N (0.38–0.41 bit).
