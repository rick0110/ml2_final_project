# R09 — Which voices fail: those close to the default voice, and those far from the trained ones when N is small

- **Date:** November 2026 (Colab phase)
- **Path:** [06 — default voice pull](../paths/06_default_voice_pull/README.md)
- **Verdict:** descriptive; part of the difficulty is an artifact of the binary measure

## Question

Is failing to copy a voice chance, or a property of the voice? What characterizes the hard voices?

## Setup

No new training: rank correlations over the 38 external speakers of the runs in [R08](R08_wide_eval_38_voices.md) (`scripts/coverage/voice_hardness.py`).

## Result

- **Difficulty belongs to the voice:** the per-speaker copy rate correlates 0.75 between N = 8 and 32, 0.74 between 32 and 128 and 0.59 between 8 and 128.
- **High-pitched voices fail more:** correlation of −0.53, −0.52 and −0.37 between copying and the reference F0 (the default voice is female).
- **Copying rises with the distance from the reference to the default voice:** 0.47, 0.58 and 0.53.
- **The distance to the nearest trained voice predicts failure with few speakers and stops predicting it with many:** −0.33 (N = 8), −0.40 (32), +0.13 (128).

## Conclusion

With few speakers the decoder only reproduces voices close to the ones it has heard (the coverage reading); with 128, almost every voice has a neighbor, and the remaining failures come from the pull toward the default voice. The binary measure amplifies this: with normalized vectors, the margin is m = (½ − ρ)·‖e_ref − e_0‖², so voices similar to the default voice flip sides by noise (section 8.10 of the theory note). This is why the main metric became the voice bits ([R10](R10_voice_bits_decoder_bottleneck.md)).

## Data

Sections 8.9 and 8.10 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md).
