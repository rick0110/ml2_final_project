# Path 03 — Training recipe

**Status:** active

## Question

How much of the results depends on the training recipe (KL, training length, optimization) rather than on N or the architecture? A fair comparison across N requires runs close to saturation.

## Current hypothesis

The KL ramp compressed the latent and threw the voice away ([R15](../../results/R15_kl_ramp.md)). Without it, training saturates around 10,000–20,000 steps at N = 128 ([R18](../../results/R18_training_length.md)), and more speakers need more steps ([R16](../../results/R16_N_curve_without_kl.md)).

## History

| When | What we learned | Result |
|---|---|---|
| 2026-11-23 | Training for 5,000 steps lowered the bits (1.75 → 1.42) | [R15](../../results/R15_kl_ramp.md) |
| 2026-11-24 | The cause is the KL ramp; without it, 1.81 → 2.44 → 2.84 bits (2,500 → 5,000 → 10,000) | [R15](../../results/R15_kl_ramp.md) |
| 2026-11-25 | 20,000 steps at N = 128: 2.96 (+0.12 over 10,000) | [R18](../../results/R18_training_length.md) |

## In progress

- [`in_progress/training_length.md`](in_progress/training_length.md): training-length curve at N = 128 and 241.

## Ideas not yet tested

- Learning-rate decay at the end of training (today it is constant).
- A small, constant KL (or *free bits*) instead of zero, to keep the latent regular without erasing the voice.
- A larger batch: the GPU uses 9 of 20 GB.
