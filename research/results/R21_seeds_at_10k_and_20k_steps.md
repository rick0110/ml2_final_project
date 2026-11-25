# R21 — The no-KL curve holds with 3 seeds; at N = 128, 20,000 steps add almost nothing over 10,000

- **Date:** 2026-11-25 (cluster)
- **Path:** [01 — scaling law](../paths/01_scaling_law/README.md), [03 — training recipe](../paths/03_training_recipe/README.md)
- **Verdict:** curve confirmed; the 20,000-step predictions at N = 128 slightly missed (gain smaller than predicted)

## Question

Does the no-KL curve at 10,000 steps ([R16](R16_N_curve_without_kl.md)) hold with 3 seeds per point? How much do 20,000 steps add?

## Setup

`-nokl`, K = 32. Seeds 2–3 at 10,000 steps; seed 2 at N = 128 continued to 20,000 steps. Predictions registered in [`paths/01_scaling_law/in_progress/long_curve_without_kl.md`](../paths/01_scaling_law/in_progress/long_curve_without_kl.md) and [`paths/03_training_recipe/in_progress/training_length.md`](../paths/03_training_recipe/in_progress/training_length.md).

## Result

| N | 10,000 steps, 3 seeds | Mean before (R16) | Predicted band |
|---|---|---|---|
| 8 | 1.08 ± 0.01 | 1.07 | ±0.15 → hit |
| 16 | 1.72 ± 0.06 | 1.72 | — |
| 32 | 2.07 ± 0.10 | 2.11 | ±0.15 → hit |
| 64 | 2.38 ± 0.07 | 2.38 | — |
| 128 | 2.83 ± 0.11 | 2.84 | ±0.15 → hit |
| 241 | 2.96 ± 0.13 | 2.90 | ±0.15 → hit |

At N = 128, 20,000 steps: 2.86 ± 0.10 (2 seeds; seed 1 gave 2.96, so seed 2 about 2.76).

- Predicted for seed 2 at 20,000 steps: 2.90–3.05 → **missed** (about 2.76).
- Predicted gain from 10,000 to 20,000 steps at N = 128: 0.05–0.20 bit → **missed** (+0.03 on the 2-seed mean).

## Conclusion

- The no-KL curve at 10,000 steps is stable across seeds (standard deviation 0.01–0.13 bit), and the slope of about 0.4 bit per doubling stands.
- At N = 128, training is saturated by 10,000 steps: more steps add less than the seed-to-seed spread. Pending: the same comparison at N = 241 (running), which decides whether the flattening is too few steps.

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant nokl --K 32 --step 10000 20000`.
