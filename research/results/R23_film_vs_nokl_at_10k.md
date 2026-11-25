# R23 — At 10,000 steps FiLM is about even with plain no-KL training; at 241 speakers with 20,000 steps it reaches 3.18 bits

- **Date:** 2026-11-25 (cluster)
- **Path:** [02 — decoder bottleneck](../paths/02_decoder_bottleneck/README.md)
- **Verdict:** the 10,000-step prediction hit; the main decision (20,000 steps at N = 241) is still pending

## Question

Does FiLM raise the ceiling, or only speed up learning? ([R17](R17_film_without_kl.md))

## Setup

`-filmnokl` vs `-nokl`, K = 32, 10,000 steps with 2–3 seeds, plus one FiLM run at 20,000 steps (N = 241). Predictions registered in [`paths/02_decoder_bottleneck/in_progress/film_without_kl_seeds.md`](../paths/02_decoder_bottleneck/in_progress/film_without_kl_seeds.md).

## Result

| N | No KL, 10,000 (3 seeds) | FiLM no KL, 10,000 | Difference |
|---|---|---|---|
| 8 | 1.08 ± 0.01 | 1.24 ± 0.04 (2) | +0.16 |
| 16 | 1.72 ± 0.06 | 1.74 (1) | +0.02 |
| 32 | 2.07 ± 0.10 | 2.05 ± 0.06 (2) | −0.02 |
| 64 | 2.38 ± 0.07 | 2.53 ± 0.00 (2) | +0.15 |
| 128 | 2.83 ± 0.11 | 2.72 ± 0.05 (2) | −0.11 |
| 241 | 2.96 ± 0.13 | 3.05 ± 0.05 (3) | +0.09 |

FiLM without KL at 20,000 steps, N = 241: **3.18 bits** (1 seed), the best result so far.

## Against the registered predictions

- "At 10,000 steps, with 2 seeds, FiLM's mean advantage at N ≥ 64 lies between 0 and +0.2 bit": **hit** (+0.04).
- Main decision rule (FiLM vs no KL at N = 241, 20,000 steps): **pending**; the no-KL run at 20,000 steps is running.

## Conclusion

At 10,000 steps the two recipes are within the seed spread at most N. The large FiLM gains seen at 2,500 and 5,000 steps ([R12](R12_film.md), [R17](R17_film_without_kl.md)) look mostly like faster learning. Whether FiLM also raises the ceiling at N = 241 is decided by the pending 20,000-step comparison.

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant nokl filmnokl --K 32 --step 10000 20000`.
