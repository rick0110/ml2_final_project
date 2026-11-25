# R02 — Reference swap and contrastive loss do not reduce the data needed (negative)

- **Date:** November 2026 (Colab phase)
- **Path:** [05 — voice representation](../paths/05_voice_representation/README.md)
- **Verdict:** negative

## Question

Is the problem that the style code carries the sentence content? If so, taking the style from another sentence of the same speaker, or forcing similar codes for the same speaker, should help.

## Setup

Three runs repeated two ways: style taken from another sentence of the same speaker (`-swap`) and, on top of that, a contrastive loss on the latent with weight 0.5 (`-contrast`). 2,500 steps, style at 100%, old evaluation (4 unseen voices).

## Result

| N × K | Variant | WER trained | WER unseen | Copied: trained | Copied: unseen |
|---|---|---|---|---|---|
| 8 × 8 | normal | 21% / 26% | 14% / 24% | 45% / 44% | 26% / 39% |
| 8 × 8 | swap | 23% / 25% | 21% / 24% | 52% / 48% | 27% / 32% |
| 8 × 8 | swap + contrastive | 27% / 23% | 20% / 21% | 42% / 48% | 27% / 31% |
| 8 × 128 | normal (seed 1) | 11% | 8% | 91% | 51% |
| 8 × 128 | swap (seed 1) | 12% | 8% | 87% | 38% |
| 8 × 128 | swap + contrastive (seed 1) | 14% | 10% | 85% | 39% |

## Conclusion

Nothing changed beyond seed-to-seed variation. With style at 60%, the contrastive loss trades copying for WER without improving the trade-off. Stopping the latent from carrying the sentence does not reduce the amount of data needed: the bottleneck seems to be how many voices and sentences the decoder has seen.

## Caveats

The weight and temperature of the contrastive loss were not tuned; few seeds.

## Data

README, section 9; `reports/coverage/results/`.
