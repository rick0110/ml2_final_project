# R20 — 64 sentences per speaker raise the whole curve by about 0.2 bit; the gap from 128 to 241 speakers stays small

- **Date:** 2026-11-25 (cluster)
- **Path:** [04 — speakers vs sentences](../paths/04_speakers_vs_sentences/README.md), [01 — scaling law](../paths/01_scaling_law/README.md)
- **Verdict:** mixed against the registered predictions (one branch hit, one missed); 1–2 seeds per point, more running

## Question

Is the flattening from 128 to 241 speakers ([R16](R16_N_curve_without_kl.md)) caused by too few sentences per speaker (K = 32)?

## Setup

`-nokl`, 10,000 steps, K = 64, N ∈ {32, 64, 128, 241}. Predictions registered in [`paths/04_speakers_vs_sentences/in_progress/k64_without_kl.md`](../paths/04_speakers_vs_sentences/in_progress/k64_without_kl.md).

**Caveat found while preparing more runs:** LibriTTS train-clean-100 (after the 1.5–10 s duration filter) has 245 speakers, but only **203 have at least 32 sentences, 150 have at least 64 and 74 have at least 128** (median 87). `coverage_manifest.py` uses whatever a speaker has when it has fewer than K. So the "241 speakers" points were never 241 × K sentences: the speakers added from 128 to 241 contribute fewer sentences each, and more so at K = 64.

## Result

| N | K = 32, 10,000 steps (3 seeds) | K = 64, 10,000 steps | Difference |
|---|---|---|---|
| 32 | 2.07 ± 0.10 | 2.24 (1) | +0.17 |
| 64 | 2.38 ± 0.07 | 2.65 (1) | +0.27 |
| 128 | 2.83 ± 0.11 | 3.09 (1) | +0.26 |
| 241 | 2.96 ± 0.13 | 3.17 ± 0.10 (2) | +0.21 |

Voice bits (seeds in parentheses). With K = 64 at N = 241: 97% of samples copied, WER 6.2%.

## Against the registered predictions

- "If the flattening is too few sentences, `n241_k64` at 10,000 steps reaches at least 3.1": **hit** (3.17).
- "…and the gap between 128 and 241 widens": **missed**. The gap is 0.13 bit with K = 32 and 0.08 bit with K = 64.

## Conclusion

- More sentences per speaker help at every N, by about 0.2–0.27 bit. That is comparable to one doubling of N (0.38–0.41 bit).
- They do not remove the flattening between 128 and 241 speakers. Part of that flattening may come from the caveat above: the extra speakers bring fewer sentences each. A clean test needs a corpus where every speaker has K sentences (path 01, beyond 241 speakers).
- K = 128 is only possible up to N = 74 with this corpus.

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant nokl --step 10000`.
