# R05 — Speed-changed pseudo-speakers do not count as speakers (negative, as predicted)

- **Date:** November 2026 (Colab phase)
- **Path:** [07 — voice augmentation](../paths/07_voice_augmentation/README.md)
- **Verdict:** negative; the "they don't count" prediction hit both numbers

## Question

Does creating new voices by changing the speed of the recordings (pitch and formants shift together) raise the effective N for free?

## Setup

`-aug` (`scripts/coverage/augment_voices.py`): each training sentence gets a copy with its speed shifted by 1.5 to 4 semitones (41 and 54 pseudo-speakers). Same number of steps (2,500), old evaluation. Predictions registered beforehand, in section 5 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md).

## Result

| Run | Copied: trained | Copied: unseen | WER unseen | Predicted if they don't count | Predicted if they count |
|---|---|---|---|---|---|
| 8 × 8 normal (2 seeds) | 45% / 44% | 26% / 39% | 14% / 24% | | |
| 8 × 8 with pseudo-speakers | 50% | **28%** | 14% | ~31% | ~50% |
| 8 × 128 normal (2 seeds) | 91% / 88% | 51% / 37% | 8% / 9% | | |
| 8 × 128 with pseudo-speakers | 84% | **42%** | 8% | ~44% | ~55% |

## Conclusion

The N that counts is the number of **real** voices. Changing the speed moves a voice along a single direction of voice space; unseen voices differ in many (timbre, glottal source, articulation, rhythm). What remains is augmenting voices with a voice converter trained on many speakers, which is blocked ([path 07](../paths/07_voice_augmentation/README.md)).

## Data

Section 6 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md).
