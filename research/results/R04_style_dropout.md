# R04 — Style dropout: the speaker vector gets used, but the need for speakers stays (negative)

- **Date:** November 2026 (Colab phase)
- **Path:** [05 — voice representation](../paths/05_voice_representation/README.md)
- **Verdict:** negative

## Question

If the voice is forced through the pretrained speaker vector (zeroing the VAE on half of the items), do unseen voices improve?

## Setup

`-spkdrop`: ECAPA vector + VAE with style dropout 0.5. Three runs, one seed, 2,500 steps, old evaluation. "Vector only" = VAE zeroed at evaluation; "vector + VAE" = style at 100%.

## Result

| N × K | Variant | WER trained | WER unseen | Copied: trained | Copied: unseen |
|---|---|---|---|---|---|
| 8 × 128 | normal | 11% | 8% | 91% | 51% |
| 8 × 128 | dropout, vector only | 9% | 6% | 71% | 5% |
| 8 × 128 | dropout, vector + VAE | 14% | 8% | 88% | 37% |
| 32 × 8 | normal (2 seeds) | 17% / 17% | 13% / 9% | 58% / 51% | 47% / 47% |
| 32 × 8 | dropout, vector only | 11% | 7% | 9% | 1% |
| 32 × 8 | dropout, vector + VAE | 22% | 15% | 57% | 60% |
| 32 × 128 | normal (2 seeds) | 12% / 10% | 5% / 5% | 86% / 86% | 61% / 57% |
| 32 × 128 | dropout, vector only | 9% | 6% | 64% | 47% |
| 32 × 128 | dropout, vector + VAE | 12% | 6% | 83% | 62% |

## Conclusion

With 8 speakers the vector alone memorizes the trained voices (71%) and barely copies unseen ones (5%). With 32 speakers and 128 sentences it already copies 47% of unseen voices on its own, but the combination equals the normal run. An encoder trained on thousands of voices does not spare the decoder from hearing many speakers: the limit is how many voices the decoder has heard, whatever path the voice takes to reach it.

## Data

README, section 9.
