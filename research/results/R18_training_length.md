# R18 — Training length without KL: diminishing gains after 10,000 steps

- **Date:** 2026-11-25 (cluster)
- **Path:** [03 — training recipe](../paths/03_training_recipe/README.md)
- **Verdict:** indication (one seed at 20,000 steps)

## Question

Without the KL ramp, how long is it worth training?

## Setup

`-nokl`, N = 128, K = 32. Longer runs resume from the checkpoints of shorter ones (the learning rate is constant after warm-up).

## Result

| Steps | Seeds | Voice bits | Copied (%) | WER (%) |
|---|---|---|---|---|
| 2,500 | 3 | 1.81 ± 0.06 | 80 | 12.0 |
| 5,000 | 3 | 2.44 ± 0.05 | 88 | 8.3 |
| 10,000 | 2 | 2.84 ± 0.13 | 91 | 6.8 |
| 20,000 | 1 | 2.96 | 92 | 6.0 |

## Conclusion

Each doubling of steps yields less: +0.63, +0.40, +0.12 bit. With N = 128 and K = 32 training is close to saturating around 10,000–20,000 steps. Since more speakers need more training ([R16](R16_N_curve_without_kl.md)), a fair comparison across N should use runs near each N's saturation. Under way: 20,000 steps at N = 241.

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant nokl --N 128 --K 32`.
