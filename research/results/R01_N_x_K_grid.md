# R01 — N × K grid: sentences decide trained voices, speakers decide unseen ones

- **Date:** November 2026 (Colab phase)
- **Path:** [01 — scaling law](../paths/01_scaling_law/README.md), [04 — speakers vs sentences](../paths/04_speakers_vs_sentences/README.md)
- **Verdict:** two-level structure confirmed; the unseen-voice numbers are noisy (4 voices per evaluation)

## Question

When a voice appears in training with few sentences, can the model read new text with it? How does that change with the number of speakers (N) and of sentences per speaker (K)?

## Setup

Original recipe ([protocol](../protocol.md)), 2,500 steps, style at 100%. Old evaluation: 25 sentences, 8 trained voices and 4 unseen voices from the same corpus (without the 38 external ones). Where there are two numbers, they are two seeds.

## Result

| N | K | WER trained | WER unseen | Copied: trained | Copied: unseen |
|---|---|---|---|---|---|
| 2 | 128 | 13% / 11% | 8% / 5% | 70% / 74% | 4% / 2% |
| 8 | 8 | 21% / 26% | 14% / 24% | 45% / 44% | 26% / 39% |
| 8 | 32 | 17% | 14% | 74% | 47% |
| 8 | 128 | 11% / 10% | 8% / 9% | 91% / 88% | 51% / 37% |
| 32 | 8 | 17% / 17% | 13% / 9% | 58% / 51% | 47% / 47% |
| 32 | 32 | 13% | 8% | 80% | 52% |
| 32 | 128 | 12% / 10% | 5% / 5% | 86% / 86% | 61% / 57% |

## Conclusion

- **K decides the trained voices:** with 8 speakers, copying goes from 45% (K = 8) to 74% (32) and 90% (128).
- **N decides the unseen voices:** with K = 128, from 3% (N = 2) to 44% (8) and 59% (32).
- **Many speakers make up for few sentences:** 32 × 8 (256 sentences) copies unseen voices as well as 8 × 128 (1,024 sentences).
- **The latent explains the error:** over 144 references, the farther the style code is from the default voice, the more the voice is copied (rank correlation 0.83) and the more words come out wrong (0.45).
- A two-level sampling law fits the grid (not copied ≈ a + A/N + B/(N·K) for unseen voices; a' + C/K + A'/N for trained ones), but the shape of the decay was not identified later ([R06](R06_128_and_200_speakers.md), [R08](R08_wide_eval_38_voices.md)).

## Caveats

4 unseen voices per evaluation; between seeds, copying of unseen voices varies by up to 18 points. A binary measure with a single speaker encoder.

## Data

`reports/coverage/results/` (per-sample results), `scripts/coverage/summarize.py`, `scripts/coverage/fit_law.py`. Details in sections 1–4 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md).
