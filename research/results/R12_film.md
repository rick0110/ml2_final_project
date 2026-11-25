# R12 — FiLM in the decoder: the gain grows with N (up to +0.44 bit at 241)

- **Date:** 2026-11-23/24 (cluster)
- **Path:** [02 — decoder bottleneck](../paths/02_decoder_bottleneck/README.md)
- **Verdict:** partial support: the registered prediction (≥ +0.5 bit at N = 8 and at N = 128) was not reached, but the gain is real and grows with N

## Question

If the bottleneck is where the style enters the decoder, does letting the style vector modulate the LSTM states at every step (FiLM) raise the voice bits?

## Setup

`-film` ([protocol](../protocol.md)): FiLM zero-initialized (the model starts identical to the original). Original recipe, K = 32, 2,500 steps. Prediction registered beforehand (section 8.12 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md)): C rises by at least 0.5 bit at N = 8 and at N = 128; if it stays within the interval of the normal run, the style entry point is not the bottleneck.

## Result

| N | Original: bits (seeds) | FiLM: bits (seeds) | Difference | FiLM: copied | FiLM: WER |
|---|---|---|---|---|---|
| 8 | 0.70 ± 0.01 (3) | 0.71 ± 0.07 (3) | +0.01 | 53% | 10.9% |
| 16 | 1.10 ± 0.06 (3) | 1.22 (1) | +0.12 | 71% | 13.4% |
| 32 | 1.23 ± 0.02 (3) | 1.45 ± 0.01 (3) | +0.22 | 75% | 12.4% |
| 128 | 1.75 ± 0.04 (3) | 2.06 ± 0.02 (3) | +0.31 | 85% | 11.3% |
| 241 | 1.76 ± 0.08 (3) | 2.20 ± 0.05 (3) | +0.44 | 88% | 10.7% |

Slope: 0.33 ± 0.03 bit per doubling (8–128), against 0.27 ± 0.03 with the original recipe.

## Conclusion

- At N = 8 FiLM changes nothing; the gain appears and grows with N, and at 241 FiLM keeps rising where the original recipe stalls.
- Reading: with few voices the limit is coverage (the decoder has heard few voices); with many, the way of conditioning starts to limit, and FiLM helps.
- No cost in intelligibility (equal or lower WER).

## Caveats

N = 16 has one seed. FiLM only on the decoder LSTMs (not on the post-net or the attention).

## Data

[`data/cluster_runs.csv`](data/cluster_runs.csv); `python scripts/coverage/results_table.py --variant base film --step 2500 --K 32`.
