# Path 01 — Scaling law in N

**Status:** active

## Question

How many bits of the reference voice reach the generated audio as a function of the number of training speakers (N)? Is the relationship regular (a law)? Does it saturate, and where?

## Current hypothesis

Bits grow linearly in log2 N (each doubling of speakers adds a fixed number of bits), because the decoder comes to reach more distinct voices (coverage), with the same noise per voice ([R10](../../results/R10_voice_bits_decoder_bottleneck.md)). The slope depends on the training recipe; the apparent ceiling at 241 speakers may be too little training or too few sentences per speaker, not a real limit.

## History

| When | What we learned | Result |
|---|---|---|
| Colab | Speakers decide unseen voices; sentences decide trained ones | [R01](../../results/R01_N_x_K_grid.md) |
| Colab | With 4 test voices, no gain could be seen above 128 speakers (seed noise) | [R06](../../results/R06_128_and_200_speakers.md) |
| Colab | On 38 voices, the gain from 32 to 128 is real | [R08](../../results/R08_wide_eval_38_voices.md) |
| Colab | In bits: 0.31 bit per doubling, from two points | [R10](../../results/R10_voice_bits_decoder_bottleneck.md) |
| 2026-11-23 | With 3 seeds: 0.27 ± 0.03 bit per doubling up to 128; no gain from 128 to 241 | [R11](../../results/R11_N_curve_original_recipe.md) |
| 2026-11-24/25 | Without the KL ramp and with 10,000 steps: 0.41 ± 0.06 per doubling; 2.90 bits at 241, still flattening | [R16](../../results/R16_N_curve_without_kl.md) |

## In progress

- [`in_progress/long_curve_without_kl.md`](in_progress/long_curve_without_kl.md): seeds at 10,000 steps and 20,000 steps at 241 speakers.
- [`in_progress/beyond_241_speakers.md`](in_progress/beyond_241_speakers.md): planned; requires preparing another corpus.

## Open questions

- Is the flattening from 128 to 241 too few steps, too few sentences per speaker, or a real limit?
- Does the law continue beyond 241 speakers? At the current rate, how many speakers would bring the output to the level of real speech (4.2–4.7 bits)?
