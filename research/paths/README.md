# Research paths

Each path is a line of inquiry. Each folder holds:

- `README.md`: question, hypothesis, history (with links to results) and status;
- `in_progress/`: one file per ongoing exploration, with what is running, the predictions registered **before** the results and the decision rule;
- `done/` (when there is one): finished explorations, already turned into results.

| Path | Question | Status | In progress |
|---|---|---|---|
| [01 — Scaling law in N](01_scaling_law/README.md) | How do voice bits grow with the number of speakers? Do they saturate? | **active** | long curve without KL; beyond 241 speakers (planned) |
| [02 — Decoder bottleneck](02_decoder_bottleneck/README.md) | Is the decoder the limit? Does conditioning it better help? | **active** | FiLM without KL with more seeds and 20,000 steps |
| [03 — Training recipe](03_training_recipe/README.md) | KL, training length and optimization | **active** | training length without KL |
| [04 — Speakers vs sentences](04_speakers_vs_sentences/README.md) | More voices or more sentences per voice? | **active** | K = 64 without KL; K grid with the original recipe |
| [05 — Voice representation](05_voice_representation/README.md) | Do the encoder, latent and reference limit copying? | nearly closed | re-initialization control at other N |
| [06 — Default voice pull](06_default_voice_pull/README.md) | Does the default voice pull the outputs? How much is a metric artifact? | paused | nothing |
| [07 — Voice augmentation](07_voice_augmentation/README.md) | Can speakers be manufactured? | blocked | nothing |
| [08 — Flow matching (F5-TTS)](08_flow_matching/README.md) | Does the same law hold for a modern, scalable architecture? | paused | F5-TTS setup |
| [09 — Dubbing](09_dubbing/README.md) | End application: dubbing films | future | nothing |
