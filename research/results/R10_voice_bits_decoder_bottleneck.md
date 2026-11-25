# R10 — Information theory: the decoder is the bottleneck, and the gain with N is coverage

- **Date:** November 2026 (Colab phase, analysis without new training)
- **Path:** [02 — decoder bottleneck](../paths/02_decoder_bottleneck/README.md), [01 — scaling law](../paths/01_scaling_law/README.md)
- **Verdict:** confirmed; set the main metric (voice bits)

## Question

At which point of the chain reference → encoder → latent → decoder → audio is the information about the voice lost?

## Setup

Measured on the generated audio of the 38 external voices: C = I(output voice; reference), bounds from the identification confusion matrix; ceiling log2 38 = 5.25 bits. Two theories proposed by independent agents and critiqued over three rounds each (`reports/theory/information/A` and `B`).

## Result

| Link in the chain | N = 8 | N = 128 |
|---|---|---|
| Real speech (verifier) | 4.2–4.7 | 4.2–4.7 |
| μ (Gaussian model) | 5.1 | 5.2 |
| z (Gaussian model) | 4.4 | 4.8 |
| Output, Fano bound | 0.3 | 0.9–1.1 |
| Output, cross-entropy bound | 0.75 [0.28; 1.20] | 1.99 [1.35; 2.60] |

## Conclusion

- **The decoder loses about 4 bits**; the encoder and the latent are not the problem.
- **The gain with N is coverage, not precision:** H(Ŷ) rises by 1.44 bits [0.68; 2.32] from N = 8 to 128, and H(Ŷ|S) does not change. The decoder reaches more distinct voices, with the same noise per voice.
- **Law in bits, from two points:** 0.31 bit per doubling of N [0.17; 0.47]. Matching real speech would take about 10^5 speakers at that rate.
- **A priori prediction confirmed:** longer references make copying more all-or-nothing (p = 0.0002).
- **Consequence:** the architecture target is to enlarge the set of voices the decoder can reach, with more voices in training or conditioning throughout the decoder. This led to FiLM ([R12](R12_film.md)).

## Caveats

The N = 128 output values here are from the Colab phase (one seed); on the cluster, with 3 seeds, 128 × 32 gave 1.75 ± 0.04 bits ([R11](R11_N_curve_original_recipe.md)).

## Data

Section 8.11 of [`reports/theory/coverage_theory.md`](../../reports/theory/coverage_theory.md); `reports/theory/03_information.md`; `scripts/coverage/voice_bits.py`.
