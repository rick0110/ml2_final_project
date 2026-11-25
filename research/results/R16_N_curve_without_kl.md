# R16 — N curve without KL: about +0.4 bit per doubling, still flattening from 128 to 241

- **Date:** 2026-11-24/25 (cluster)
- **Path:** [01 — scaling law](../paths/01_scaling_law/README.md)
- **Verdict:** confirmed (2–3 seeds per point, except 241 at 10,000 steps)

## Question

Without the KL ramp ([R15](R15_kl_ramp.md)) and with longer training, does the bits-vs-N curve get steeper? Does the flattening at 241 go away?

## Setup

`-nokl`, K = 32, 5,000 and 10,000 steps, N ∈ {8, 16, 32, 64, 128, 241}.

## Result

| N | Original, 2,500 (3 seeds) | No KL, 5,000 | No KL, 10,000 |
|---|---|---|---|
| 8 | 0.70 ± 0.01 | 0.89 ± 0.03 (3) | 1.07 ± 0.01 (2) |
| 16 | 1.10 ± 0.06 | 1.56 ± 0.06 (3) | 1.72 ± 0.06 (3) |
| 32 | 1.23 ± 0.02 | 1.69 ± 0.06 (3) | 2.11 ± 0.10 (2) |
| 64 | 1.66 ± 0.01 | 2.14 ± 0.06 (3) | 2.38 ± 0.07 (3) |
| 128 | 1.75 ± 0.04 | 2.44 ± 0.05 (3) | 2.84 ± 0.13 (2) |
| 241 | 1.76 ± 0.08 | 2.53 ± 0.06 (3) | 2.90 (1) |

Voice bits (seeds in parentheses). Copied at 10,000 steps: from 67% (N = 8) to 95% (N = 241). WER: from 8.8% to 6.0%.

Slope in log2 N (all seeds):

| Recipe | N from 8 to 128 | N from 8 to 241 |
|---|---|---|
| original, 2,500 | 0.27 ± 0.03 | 0.22 ± 0.04 |
| no KL, 5,000 | 0.37 ± 0.05 | 0.33 ± 0.04 |
| no KL, 10,000 | 0.41 ± 0.06 | 0.38 ± 0.05 |

## Conclusion

- Training longer without KL lifts the whole curve and makes it steeper: the gain per doubling of N goes from 0.27 to 0.41 bit.
- More speakers need more training: the gain from 5,000 to 10,000 steps is larger with large N (+0.40 at 128) than with small N (+0.18 at 8).
- The flattening from 128 to 241 shrinks but remains (+0.06 at 10,000 steps, with one seed at 241). Open hypotheses: too few steps, too few sentences per speaker (K = 32) or a real limit; under test in [path 01](../paths/01_scaling_law/README.md) and [path 04](../paths/04_speakers_vs_sentences/README.md).

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant nokl --K 32`.
