# F5-TTS setup (paused)

**Started:** 2026-11-23 · **Path:** [08](../README.md)

## Done

- Environment installed on one of the cluster machines (folder `~/f5/`): `f5-tts` 1.1.22 and the F5-TTS code cloned, with the configurations in `src/f5_tts/configs/`.
- LibriTTS-R data downloaded (restored version of LibriTTS, as parquet): train-clean-100, train-clean-360 and test-clean, about 41 GB.

## Still to do

1. Convert the parquet files to the F5-TTS format. F5's `prepare_libritts.py` expects folders with wav and `.normalized.txt`, so it needs adapting.
2. Build subsets with varying N and the same total hours.
3. Small configurations at 2–3 sizes; measure speed and memory on a free GPU.
4. Run the curve with the voice-bits evaluation.

## Why it is paused

The GPUs are busy with the Tacotron experiments, which answer the questions of paths 01–04 faster. Resume when one machine can be dedicated to it.

## Predictions

Not registered yet: register them before the first run.
