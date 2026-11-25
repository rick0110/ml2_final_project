# R17 — FiLM without KL: 3.08 bits with 241 speakers, the best result so far

- **Date:** 2026-11-24/25 (cluster)
- **Path:** [02 — decoder bottleneck](../paths/02_decoder_bottleneck/README.md)
- **Verdict:** promising; few seeds at 10,000 steps

## Question

Does the FiLM gain ([R12](R12_film.md)) add to the gain from removing the KL ramp ([R15](R15_kl_ramp.md))?

## Setup

`-filmnokl`, K = 32, 5,000 and 10,000 steps.

## Result

| N | No KL, 5,000 | FiLM no KL, 5,000 | No KL, 10,000 | FiLM no KL, 10,000 |
|---|---|---|---|---|
| 8 | 0.89 ± 0.03 (3) | 1.10 ± 0.01 (2) | 1.07 ± 0.01 (2) | 1.20 (1) |
| 16 | 1.56 ± 0.06 (3) | 1.73 (1) | 1.72 ± 0.06 (3) | 1.74 (1) |
| 32 | 1.69 ± 0.06 (3) | 2.00 ± 0.00 (2) | 2.11 ± 0.10 (2) | 2.11 (1) |
| 64 | 2.14 ± 0.06 (3) | 2.39 (1) | 2.38 ± 0.07 (3) | 2.53 (1) |
| 128 | 2.44 ± 0.05 (3) | 2.63 ± 0.05 (3) | 2.84 ± 0.13 (2) | 2.67 (1) |
| 241 | 2.53 ± 0.06 (3) | 2.98 ± 0.08 (2) | 2.90 (1) | **3.08** (1) |

Voice bits (seeds in parentheses). FiLM without KL at 10,000 steps, N = 241: 97% of samples copied, WER 5.9%.

## Conclusion

- At 5,000 steps FiLM adds +0.17 to +0.45 bit at every N, with the largest gain at 241.
- At 10,000 steps the advantage shrinks and stays within the noise at most points (one seed). FiLM seems to speed up learning more than it raises the ceiling; this still needs seeds.
- 3.08 bits is about 60% of the 5.25 ceiling and 65–70% of what real speech carries (4.2–4.7 bits, [R10](R10_voice_bits_decoder_bottleneck.md)).

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant nokl filmnokl --K 32`.
