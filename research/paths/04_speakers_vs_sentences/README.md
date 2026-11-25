# Path 04 — Speakers vs sentences per speaker

**Status:** active

## Question

With the same amount of data, is it better to have many voices with few sentences or few voices with many sentences? And do the sentences per speaker (K) explain the flattening at 241 speakers?

## Current hypothesis

For unseen voices, the number of voices weighs more than the sentences per voice ([R01](../../results/R01_N_x_K_grid.md), [R19](../../results/R19_speakers_vs_sentences.md)). But K also helps at fixed N, and with large N, K = 32 may be the limiting factor.

## History

| When | What we learned | Result |
|---|---|---|
| Colab | K decides trained voices; N decides unseen ones; 32 × 8 copies as well as 8 × 128 | [R01](../../results/R01_N_x_K_grid.md) |
| 2026-11-24/25 | In bits, with 1,024 sentences: 8 speakers 0.74; 32 speakers 1.23; 128 speakers 1.36 (original); the same without KL | [R19](../../results/R19_speakers_vs_sentences.md) |
| 2026-11-25 | K = 64 adds 0.17–0.27 bit at every N (3.17 at 241); the corpus has only 150 speakers with ≥ 64 sentences and 74 with ≥ 128 | [R20](../../results/R20_more_sentences_per_speaker.md) |

## In progress

- [`in_progress/k64_without_kl.md`](in_progress/k64_without_kl.md): K = 64 without KL and the K grid with the original recipe.
