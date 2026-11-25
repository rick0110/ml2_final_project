# Path 05 — Voice representation (encoder, latent, reference)

**Status:** nearly closed

## Question

Is copying of unseen voices limited by how the voice is described: what the latent carries, where the vector comes from, how many references are used, how many dimensions?

## Conclusion so far

No. Changing the representation does not reduce the need for speakers: the voice reaches the decoder (4–5 bits in the latent, [R10](../../results/R10_voice_bits_decoder_bottleneck.md)), and the limit is the decoder's ability to render it. The exception is shrinking the latent too much (2–8 dimensions), which becomes a bottleneck of its own.

## History

| When | What we learned | Result |
|---|---|---|
| Colab | Reference swap and contrastive loss: nothing changes | [R02](../../results/R02_reference_swap_and_contrastive.md) |
| Colab | Pretrained speaker vector: ignored | [R03](../../results/R03_pretrained_speaker_vector.md) |
| Colab | Style dropout: vector used, no gain | [R04](../../results/R04_style_dropout.md) |
| Colab | Averaging several references: does not help on the 38 voices; lowers WER | [R07](../../results/R07_multiple_references.md) |
| 2026-11-23/24 | 8-dimensional latent: −0.40 bit (discounting re-initialization); a 2-dimensional latent zeroes the bits | [R13](../../results/R13_smaller_latent_and_control.md) |

## In progress

- [`in_progress/reinit_control.md`](in_progress/reinit_control.md): the cost of re-initializing the VAE heads at other N.
