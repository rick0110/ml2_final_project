# Results data

## `cluster_runs.csv`

One row per evaluation done on the cluster (one run evaluated at one number of steps). Columns:

| Column | Meaning |
|---|---|
| `run` | run name (`n{N}_k{K}_s{seed}[-variant]`, see [protocol](../../protocol.md)) |
| `variant` | variant (`base` = original recipe) |
| `N`, `K`, `seed` | speakers, sentences per speaker, seed |
| `step` | training steps of the evaluated checkpoint |
| `bits_ce` | voice bits, cross-entropy bound, 38 external voices (main metric) |
| `bits_fano` | voice bits, Fano bound (more conservative) |
| `copied` | share of external samples with `sim_ref > sim_default` |
| `wer` | word error rate (Whisper) on the external samples |
| `machine` | cluster machine holding the run (1–10) |

The full files of each evaluation (per-sample `results.csv`, `voice_bits.json` with intervals, `embeddings.npz`, audio) and the checkpoints live on the machines and on Google Drive ([infrastructure](../../infrastructure.md)).

**Updating:** the CSV is produced on the control PC by reading the machines (script `collect_csv.py` in `~/ml2_backup/tools/`, kept outside the repository because it uses the addresses). After updating, regenerate the tables of the affected results with `scripts/coverage/results_table.py`.

Last update: 2026-11-25, 122 evaluations.

## Colab phase

Results R01–R10 come from `reports/coverage/results/` and the analyses in `reports/theory/`.
