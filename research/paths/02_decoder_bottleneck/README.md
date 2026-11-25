# Path 02 — Decoder bottleneck

**Status:** active

## Question

Where is the voice information lost, and what changes if the decoder is conditioned differently?

## Current hypothesis

The decoder is the bottleneck: the reference and the latent carry 4–5 bits, the output far fewer ([R10](../../results/R10_voice_bits_decoder_bottleneck.md)). With few speakers the limit is coverage (the decoder has heard few voices). With many, the way the style is injected also limits, and conditioning every decoder step (FiLM) helps. It is not yet clear whether FiLM raises the ceiling or only speeds up learning.

## History

| When | What we learned | Result |
|---|---|---|
| Colab | Encoder and latent are not the problem; the decoder loses ~4 bits | [R10](../../results/R10_voice_bits_decoder_bottleneck.md) |
| 2026-11-23/24 | FiLM: no gain at N = 8, +0.31 at 128, +0.44 at 241 (2,500 steps) | [R12](../../results/R12_film.md) |
| 2026-11-24/25 | FiLM without KL: 3.08 bits at 241 with 10,000 steps (best result); at 10,000 steps the edge over "no KL" shrinks | [R17](../../results/R17_film_without_kl.md) |
| 2026-11-25 | With 2–3 seeds at 10,000 steps FiLM is about even (+0.04 at N ≥ 64); FiLM at 241 with 20,000 steps: 3.18 bits | [R23](../../results/R23_film_vs_nokl_at_10k.md) |

## In progress

- [`in_progress/film_without_kl_seeds.md`](in_progress/film_without_kl_seeds.md): seeds and 20,000 steps to separate "higher ceiling" from "learns faster".

## Ideas not yet tested

- Condition the attention and the post-net as well.
- An architecture where the style enters every layer by construction (see [path 08](../08_flow_matching/README.md)).
