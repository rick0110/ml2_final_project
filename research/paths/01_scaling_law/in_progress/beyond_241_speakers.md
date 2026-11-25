# Beyond 241 speakers (planned, not started)

**Path:** [01](../README.md)

## Goal

Test whether the law continues beyond 241 speakers, the maximum of LibriTTS train-clean-100. It is the strongest test of the law: extrapolating to an N that no fit has seen.

## What is missing

1. **A larger corpus.** LibriTTS train-clean-360 has about 900 speakers. Today one of the machines has only the **LibriTTS-R** version (restored audio) of train-clean-100, train-clean-360 and test-clean, downloaded for [path 08](../../08_flow_matching/README.md). Mixing LibriTTS-R with the original LibriTTS changes the audio domain. To keep the comparison, there are two options:
   - download the original train-clean-360;
   - redo the whole curve on LibriTTS-R (more expensive).
2. **Preprocessing** identical to train-clean-100 (`scripts/preprocess/`) and manifests with `coverage_manifest.py`, with nested selection (the current 241 inside the 512).
3. **Runs:** `n512_k32` and `n900_k32` without KL, 10,000 and 20,000 steps, 2–3 seeds.

## Predictions registered (2026-11-25, before any training)

From the no-KL slope at 10,000 steps (0.38–0.41 bit per doubling, [R16](../../../results/R16_N_curve_without_kl.md)), starting from 128 speakers (2.84):

- N = 512: 3.6 bits if the law continues; 2.9–3.1 if the flattening at 241 is real.
- N = 900: 3.9 bits if the law continues.
- **The law fails** if N = 512 stays below 3.1 bits with the same number of steps and the same K.
