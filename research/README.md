# Research: how many voices does it take to clone one?

## Central question

A speech synthesis model reads a text in the voice of a reference recording it has never heard (zero-shot cloning). We want to know:

1. **How much of the reference voice reaches the generated audio**, measured in bits (identification among 38 voices; ceiling 5.25 bits).
2. **How this grows with the number of training speakers (N)**: is there a scaling law? Does it saturate?
3. **Where the information is lost**: in the encoder, the latent, or the decoder.

The answer guides the next architecture (and dubbing): if the decoder is the limit and the law is regular, we can predict how many voices and how much compute are needed.

## Current state (2026-11-25, updated)

- **The decoder is the bottleneck.** The reference and the latent carry 4–5 bits of the voice; the output carried 0.3–1 bit with the original recipe ([R10](results/R10_voice_bits_decoder_bottleneck.md)).
- **Bits grow regularly with N.** With the original recipe, +0.27 bit per doubling from 8 to 128 speakers, flattening at 241 ([R11](results/R11_N_curve_original_recipe.md)).
- **The KL ramp was holding everything back.** Training longer made things worse; without the ramp, training longer helps a lot ([R15](results/R15_kl_ramp.md)). With 10,000 steps and no KL: 1.07 bits with 8 speakers, 2.84 with 128 and 2.90 with 241, about +0.4 bit per doubling ([R16](results/R16_N_curve_without_kl.md)).
- **Conditioning the whole decoder helps, more so as N grows.** FiLM without KL reaches 3.08 bits with 241 speakers, the best result so far ([R12](results/R12_film.md), [R17](results/R17_film_without_kl.md)).
- **Speakers are worth more than sentences** for the same amount of data ([R19](results/R19_speakers_vs_sentences.md)), but sentences per speaker still help: 64 instead of 32 adds about 0.2 bit at every N, reaching 3.17 bits with 241 speakers ([R20](results/R20_more_sentences_per_speaker.md)). FiLM without KL at 20,000 steps reaches 3.18 bits ([R23](results/R23_film_vs_nokl_at_10k.md)).
- **Caveat on the 241-speaker points:** only 203 LibriTTS train-clean-100 speakers have 32 or more sentences, and 150 have 64 or more, so the speakers added from 128 to 241 bring fewer sentences each ([R20](results/R20_more_sentences_per_speaker.md)).
- **Open:** why the gain from 128 to 241 speakers is small (too few sentences per speaker? too few steps? a real limit?) and whether the curve continues beyond 241 speakers (requires preparing LibriTTS train-clean-360).

## How to navigate

| Where | What it holds |
|---|---|
| [`protocol.md`](protocol.md) | How each run is set up and evaluated; the definition of each metric; run naming |
| [`infrastructure.md`](infrastructure.md) | Cluster, job queues, where the data lives and how to retrieve it |
| [`results/`](results/README.md) | One file per result, with an index; raw data in `results/data/` |
| [`paths/`](paths/README.md) | Lines of inquiry; each with a history (`README.md`) and what is under way (`in_progress/`) |
| [`../reports/theory/coverage_theory.md`](../reports/theory/coverage_theory.md) | Theory note with the predictions registered during the Colab phase (before this folder existed) |

## Conventions

- **Results** are named `RNN_short_name.md`, numbered in the order they were obtained. A result is never rewritten: if a conclusion changes, a new result points to the old one, and the old one gets a note at the top.
- **Paths** are named `NN_name/`. Each `README.md` has the question, the hypothesis, the history (with links to results) and the status: *active*, *paused*, *closed* or *blocked*.
- **`in_progress/`** holds one file per ongoing exploration: what is running, the predictions registered beforehand and the decision rule. When the exploration ends, the file moves to `done/` in the same path and the result becomes a file in `results/`.
- Every results table states the number of seeds behind each number.
- Everything in the repository is written in English.
