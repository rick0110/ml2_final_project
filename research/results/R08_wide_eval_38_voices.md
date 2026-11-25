# R08 — Evaluation on 38 external voices: the gain from 32 to 128 speakers is real

- **Date:** November 2026 (Colab phase)
- **Path:** [01 — scaling law](../paths/01_scaling_law/README.md)
- **Verdict:** confirmed; the 38-voice evaluation became the standard

## Question

With a less noisy evaluation (38 LibriTTS test-clean speakers, none heard by any model), does the gain with N continue above 32 speakers?

## Setup

Three runs with K = 32 (`-wide`, N = 8, 32, 128), one seed, 2,500 steps, original recipe. 950 samples per run (38 voices × 25 sentences); intervals by resampling the speakers (10,000 times), differences paired on the same voices. Predictions in section 8.7 of the theory note.

## Result

| Run | External copied (95% CI) | WER external |
|---|---|---|
| 8 × 32 | 57% (43–70) | 13% |
| 32 × 32 | 70% (58–81) | 12% |
| 128 × 32 | 79% (68–89) | 12% |

## Conclusion

- Gain from 32 to 128: 9 points (95% CI 1.5 to 17), against 13 points from 8 to 32. This overturns the reading that there is no gain above 128, which came from the 4 voices.
- WER does not change with N: more speakers improve copying at no cost in intelligibility.
- Copying is nearly all-or-nothing per voice (every run has speakers at 0% and at 100%).
- The later statistical review (section 8.10) showed that three points from one seed neither fix the exponent nor choose the law. That led to 3 seeds per point on the cluster ([R11](R11_N_curve_original_recipe.md)).

## Data

Sections 8.7, 8.8 and 8.10 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md); `reports/theory/04_data.md`.
