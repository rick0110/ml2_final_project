# R03 — A pretrained speaker vector is ignored by training (negative)

- **Date:** November 2026 (Colab phase)
- **Path:** [05 — voice representation](../paths/05_voice_representation/README.md)
- **Verdict:** negative

## Question

Does a speaker encoder trained on thousands of voices (ECAPA-TDNN, frozen) give the model a better description of unseen voices?

## Setup

A second style input: the ECAPA vector, added to the encoded text next to the VAE vector (`-spk`). Three runs, seed 1, 2,500 steps, old evaluation.

## Result

| N × K | Variant | WER trained | WER unseen | Copied: trained | Copied: unseen |
|---|---|---|---|---|---|
| 8 × 8 | normal | 21% | 14% | 45% | 26% |
| 8 × 8 | with speaker vector | 18% | 14% | 43% | 30% |
| 8 × 128 | normal | 11% | 8% | 91% | 51% |
| 8 × 128 | with speaker vector | 13% | 8% | 88% | 46% |
| 2 × 128 | normal | 13% | 8% | 70% | 4% |
| 2 × 128 | with speaker vector | 14% | 8% | 70% | 4% |

With the VAE zeroed and only the speaker vector, no voice is copied (0% in all three runs).

## Conclusion

The model does not use the new input: during training the VAE sees the target audio and already delivers the voice, so training has no reason to use the vector. In the bits measure ([R10](R10_voice_bits_decoder_bottleneck.md)), ECAPA alone carries 0.00 bit.

## Data

README, section 9.
