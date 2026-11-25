# Cost of re-initializing the VAE heads at other N

**Started:** 2026-11-24 · **Path:** [05](../README.md)

## Goal

In [R13](../../../results/R13_smaller_latent_and_control.md), re-initializing the VAE heads cost 0.78 bit at N = 128. Measuring that cost at N = 32 and 241 lets the 8-dimensional latent be compared with the right control along the whole curve.

## Runs (queued or running)

`n32_k32_s1/s2-zr32` and `n241_k32_s1/s2-zr32`, at 2,500 steps (original recipe).

## Predictions registered before the results (2026-11-25)

- The re-initialization cost (original − zr32) lies between 0.4 and 0.9 bit at N = 32 and at 241.
- zr32 stays above z8 at every N (z8: 0.62 at N = 32, 0.42 at 241).

## Afterwards

If the cost is similar at every N, this path closes with a note in R13. The latent-size question only returns if a long no-KL run ([path 03](../../03_training_recipe/README.md)) recovers from the re-initialization.
