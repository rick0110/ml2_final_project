# Path 07 — Voice augmentation

**Status:** blocked

## Question

If what counts is the number of real voices, can voices be manufactured for training without recording new speakers?

## Conclusion so far

Changing the speed of the recordings does not work: the pseudo-voices do not count as speakers, as the theory predicted ([R05](../../results/R05_pseudo_speakers.md)). The version that should work is converting the default-voice sentences into hundreds of real voices with a voice converter trained without transcriptions.

## History

| When | What we learned | Result |
|---|---|---|
| Colab | Speed-changed pseudo-speakers: the effective N does not change | [R05](../../results/R05_pseudo_speakers.md) |

## Why it is blocked

The scripts are ready (`scripts/coverage/vc_cascade.py`, `vc_voices.py`, variant `-vcaug`), with predictions registered in section 7 of [`reports/theory/coverage_theory.md`](../../../reports/theory/coverage_theory.md). They use a third-party pretrained voice converter (kNN-VC, through `torch.hub`), and the environment's permission controls blocked running it. It depends on authorization from the project owner.

## In progress

Nothing. See [`in_progress/README.md`](in_progress/README.md).
