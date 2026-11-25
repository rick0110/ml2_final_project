# R11 — N curve with the original recipe, 3 seeds: +0.27 bit per doubling up to 128, flat at 241

- **Date:** 2026-11-23 (cluster)
- **Path:** [01 — scaling law](../paths/01_scaling_law/README.md)
- **Verdict:** confirmed with 3 seeds per point

## Question

With 3 seeds per point and the 38-voice evaluation, do the voice bits grow regularly with N? Up to where?

## Setup

Original recipe ([protocol](../protocol.md)), K = 32, 2,500 steps, seeds 1–3, N ∈ {4, 8, 16, 32, 64, 128, 241}. 241 is the maximum of LibriTTS train-clean-100 with K = 32.

## Result

| N | Seeds | Voice bits | Copied (%) | WER (%) |
|---|---|---|---|---|
| 4 | 1 | 0.51 | 35 | 14.9 |
| 8 | 3 | 0.70 ± 0.01 | 57 ± 1 | 13.6 ± 0.9 |
| 16 | 3 | 1.10 ± 0.06 | 65 ± 6 | 12.1 ± 1.1 |
| 32 | 3 | 1.23 ± 0.02 | 70 ± 1 | 11.3 ± 0.3 |
| 64 | 3 | 1.66 ± 0.01 | 77 ± 4 | 13.4 ± 0.9 |
| 128 | 3 | 1.75 ± 0.04 | 79 ± 1 | 11.1 ± 0.3 |
| 241 | 3 | 1.76 ± 0.08 | 81 ± 1 | 11.9 ± 0.6 |

Linear fit in log2 N (all seeds): **0.27 ± 0.03 bit per doubling** from N = 8 to 128; 0.22 ± 0.04 including 241.

## Conclusion

- The growth is regular and the seeds agree very well (0.01–0.08 bit standard deviation), far better than the binary measure of the Colab phase.
- The slope confirms the estimate of [R10](R10_voice_bits_decoder_bottleneck.md) (0.31 [0.17; 0.47]) with a much tighter interval.
- **No gain from 128 to 241** with this recipe. Much of that limit later turned out to be the training recipe ([R15](R15_kl_ramp.md), [R16](R16_N_curve_without_kl.md)).

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant base --K 32 --step 2500`.
