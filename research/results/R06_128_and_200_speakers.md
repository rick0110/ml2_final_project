# R06 — 128 and 200 speakers in the old evaluation: the law hits K = 8, the rest is seed noise

- **Date:** November 2026 (Colab phase)
- **Path:** [01 — scaling law](../paths/01_scaling_law/README.md)
- **Verdict:** risky prediction partly confirmed; evaluating on 4 unseen voices proved too noisy (which led to [R08](R08_wide_eval_38_voices.md))

## Question

The law fitted on the N × K grid predicts almost no gain above 32 speakers (56–57% of unseen voices copied with 128). Does that hold?

## Setup

`n128_k8_s1`, `n128_k32_s1`, replicate `n128_k32_s2` and `n200_k32_s1`. 2,500 steps, style at 100%. Predictions in sections 8.1 and 8.5 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md).

## Result

| Run | Trained copied (predicted → measured) | Unseen copied (predicted → measured) | External (4 voices) | WER unseen |
|---|---|---|---|---|
| 128 × 8 | 52% → 52% | 56% → 55% | 86% | 8% |
| 128 × 32, seed 1 | 82% → 76% | 57% → 74% | 86% | 8% |
| 128 × 32, seed 2 | — → 66% | 60% (replicate) → 56% | 86% | 8% |
| 200 × 32, seed 1 | — → 66% | 60–68% → 64% | 91% | 7% |

## Conclusion

- With K = 8 the law was within 1 point.
- The 74% of 128 × 32 was largely seed luck: the replicate gave 56% (mean 65%).
- From 128 to 200 speakers there was no measurable gain **in this evaluation**. Two laws (decay as 1/N or as 1/√N) fit the 16 runs equally well.
- The measure was what limited the conclusion: 4 unseen voices × 25 sentences give seed-to-seed noise of up to 18 points. This led to the evaluation on 38 voices ([R08](R08_wide_eval_38_voices.md)), which showed a real gain from 32 to 128.

## Data

Sections 8.1–8.6 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md).
