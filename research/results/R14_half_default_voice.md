# R14 — Half of the default-voice sentences: +4 points of copying, same bits

- **Date:** 2026-11-23/24 (cluster)
- **Path:** [06 — default voice pull](../paths/06_default_voice_pull/README.md)
- **Verdict:** the copying prediction confirmed at its lower end; bits do not change, which points to an effect of the measure, not of the voice

## Question

Do the 2,000 default-voice utterances (LJSpeech) pull the outputs toward that voice? Registered prediction (section 8.11 of the theory note): with half the utterances (M = 1,000), 4 to 9 fewer points of uncopied voices at N = 128; telling the theories apart needs about 6 runs per condition.

## Setup

`-lj1000`: 1,000 default-voice utterances, same test sentences. N = 128, K = 32, 2,500 steps, 8 seeds.

## Result

| Variant | Seeds | Voice bits | Copied (%) | WER (%) |
|---|---|---|---|---|
| original (2,000 utterances) | 3 | 1.75 ± 0.04 | 79 ± 1 | 11.1 ± 0.3 |
| `-lj1000` (1,000 utterances) | 8 | 1.83 ± 0.08 | 83 ± 2 | 10.0 ± 0.8 |

## Conclusion

- Copying rises by 4 points, at the lower end of the prediction.
- The bits rise only 0.08 (within the seed-to-seed deviation). The "copied" measure compares with the default voice; with less default voice in training the output moves away from it and the measure improves, without the decoder reaching more voices.
- Supports using the bits as the main metric.

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant base lj1000 --N 128 --K 32 --step 2500`.
