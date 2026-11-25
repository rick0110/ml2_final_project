# Path 08 — Modern architecture: flow matching (F5-TTS)

**Status:** paused

## Question

Do the scaling law in N and the decoder bottleneck hold for a modern architecture? Here, flow matching with a diffusion transformer (F5-TTS), which generates the whole spectrogram at once, conditions every layer and scales better than Tacotron 2.

## Why it matters

Tacotron 2 is slow and does not scale. If the law measured here repeats on a current architecture, it becomes a result about voice cloning in general, not about an old model. It also tells how many speakers and how much compute a new architecture would need.

## Plan

1. Small models at 2–3 sizes (for example 4–8 layers, width 256–512), as labs do to predict scaling with few resources.
2. Subsets with the same total hours and varying N, with nested selection as with Tacotron.
3. The same voice-bits evaluation on the same 38 external voices ([protocol](../../protocol.md)).

## History

No results yet.

## In progress

- [`in_progress/f5_setup.md`](in_progress/f5_setup.md)
