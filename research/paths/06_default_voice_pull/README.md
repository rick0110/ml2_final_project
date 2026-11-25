# Path 06 — Default voice pull

**Status:** paused

## Question

Do the 2,000 default-voice utterances (LJSpeech) pull the outputs toward that voice? How much of the "uncopied voice" is a real pull and how much is an artifact of the measure, which compares against the default voice?

## Conclusion so far

The binary measure (`sim_ref > sim_default`) distorts: the margin is m = (½ − ρ)·‖e_ref − e_0‖², so voices similar to the default voice flip sides by noise ([R09](../../results/R09_hard_voices.md)). Halving the default voice raises copying by 4 points but barely changes the bits ([R14](../../results/R14_half_default_voice.md)). That is why the main metric became the voice bits, which do not use the default voice.

## History

| When | What we learned | Result |
|---|---|---|
| Colab | High-pitched voices and voices close to the default fail more; part of it is a metric artifact | [R09](../../results/R09_hard_voices.md) |
| 2026-11-23/24 | Half the default voice: +4 points of copying, same bits (8 seeds) | [R14](../../results/R14_half_default_voice.md) |

## In progress

Nothing. See [`in_progress/README.md`](in_progress/README.md).
