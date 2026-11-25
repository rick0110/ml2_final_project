# R19 — With the same amount of data, more speakers yield more bits than more sentences per speaker

- **Date:** 2026-11-24/25 (cluster)
- **Path:** [04 — speakers vs sentences](../paths/04_speakers_vs_sentences/README.md)
- **Verdict:** confirmed with both recipes; several cells with one seed

## Question

With the same total number of LibriTTS sentences, is it better to have many speakers with few sentences or few speakers with many?

## Setup

N × K pairs with the same total N·K. Original recipe (2,500 steps) and no KL (5,000 steps).

## Result

| Total sentences | N × K | Original, 2,500 | No KL, 5,000 |
|---|---|---|---|
| 1,024 | 8 × 128 | 0.74 (1) | 0.94 (1) |
| 1,024 | 32 × 32 | 1.23 ± 0.02 (3) | 1.69 ± 0.06 (3) |
| 1,024 | 128 × 8 | 1.36 ± 0.08 (3) | 1.93 (1) |
| 4,096 | 32 × 128 | 1.38 (1) | 1.84 ± 0.03 (2) |
| 4,096 | 128 × 32 | 1.75 ± 0.04 (3) | 2.44 ± 0.05 (3) |
| 1,928 / 7,712 | 241 × 8 / 241 × 32 | — | 2.16 (1) / 2.53 ± 0.06 (3) |

Voice bits (seeds in parentheses).

## Conclusion

- With 1,024 sentences, going from 8 to 128 speakers nearly doubles the bits (0.74 → 1.36; 0.94 → 1.93). With 4,096, going from 32 to 128 speakers gives +0.37 to +0.60 bit.
- Sentences per speaker also help at fixed N (128 × 8 → 128 × 32: +0.39 and +0.51 bit), but less than the same increase in data spent on speakers.
- Confirms, in bits and on the 38 voices, the reading of the old grid ([R01](R01_N_x_K_grid.md)): for unseen voices, what counts most is the number of voices.

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --K 8 128` and `--N 128 --K 32`.
