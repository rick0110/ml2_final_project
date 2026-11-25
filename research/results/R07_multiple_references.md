# R07 — Averaging the code of several references: helps on the 4 unseen voices, not on the 38 external ones

- **Date:** November 2026 (Colab phase)
- **Path:** [05 — voice representation](../paths/05_voice_representation/README.md)
- **Verdict:** negative for copying; positive only as a stabilizer (lower WER)

## Question

Is the floor of uncopied unseen voices the noise of estimating a voice from a single sentence? If so, using the mean style code of R sentences from the same speaker should help.

## Setup

No new training: at evaluation, code = mean of the codes of R = 1, 4 or 16 sentences (`--n-refs` in `scripts/eval/synth_check.py`). Per-speaker measure (`sim_spk > sim_default`).

## Result

Old evaluation (4 unseen voices), R = 1 → 4:

| Run | Trained | Unseen | External (4) |
|---|---|---|---|
| 128 × 8 | 66% → 66% | 47% → 67% | 77% → 83% |
| 128 × 32, seed 1 | 84% → 71% | 74% → 78% | 95% → 94% |
| 128 × 32, seed 2 | 73% → 68% | 59% → 69% | 74% → 97% |
| 200 × 32 | 77% → 68% | 57% → 75% | 76% → 90% |

Wide evaluation (38 external voices), per speaker, R = 1 → 4: 8 × 32: 66% → 64%; 32 × 32: 81% → 79%; 128 × 32: 88% → 84%. WER: 13% → 12%, 12% → 10%, 12% → 9%.

## Conclusion

On the 38 voices, the mean of 4 references does not raise copying (−2.5 points, CI [−6.4; +0.8]) and changes the voice bits by −0.40 to +0.31 with no consistent direction. The gain seen on the 4 unseen voices does not generalize. What the mean consistently does is lower the WER. Together with [R03](R03_pretrained_speaker_vector.md)/[R04](R04_style_dropout.md): the bottleneck is not describing the new voice but the decoder being able to render it.

## Data

Sections 8.2, 8.4, 8.6, 8.8 and 8.10 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md).
