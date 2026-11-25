# Results

One file per result, in the order they were obtained. Each file has the question, setup, table, conclusion, caveats and where the data comes from. To add a result, take the next free number and follow [`AGENTS.md`](../../AGENTS.md).

"Bits" are the voice bits on the 38 external voices (ceiling 5.25), the main metric since R10 ([protocol](../protocol.md)).

## Colab phase (evaluation on 4 unseen voices, later 38 voices)

| ID | Result | Path | Verdict |
|---|---|---|---|
| [R01](R01_N_x_K_grid.md) | Sentences decide trained voices, speakers decide unseen ones | 01, 04 | confirmed |
| [R02](R02_reference_swap_and_contrastive.md) | Reference swap and contrastive loss do not help | 05 | negative |
| [R03](R03_pretrained_speaker_vector.md) | A pretrained speaker vector is ignored by training | 05 | negative |
| [R04](R04_style_dropout.md) | With style dropout the vector is used, but the need for speakers stays | 05 | negative |
| [R05](R05_pseudo_speakers.md) | Speed-changed pseudo-speakers do not count as speakers | 07 | negative (predicted) |
| [R06](R06_128_and_200_speakers.md) | 128 and 200 speakers: the law hits K = 8; 4 voices are too noisy | 01 | partial |
| [R07](R07_multiple_references.md) | Averaging several references does not help on the 38 voices; it only lowers WER | 05 | negative |
| [R08](R08_wide_eval_38_voices.md) | On 38 voices, the gain from 32 to 128 speakers is real | 01 | confirmed |
| [R09](R09_hard_voices.md) | Voices close to the default voice fail more; with small N, those far from the trained ones | 06 | descriptive |
| [R10](R10_voice_bits_decoder_bottleneck.md) | Information theory: the decoder is the bottleneck; the gain with N is coverage | 02, 01 | confirmed |

## Cluster phase (38 external voices, 2–3 seeds)

| ID | Result | Path | Key number | Verdict |
|---|---|---|---|---|
| [R11](R11_N_curve_original_recipe.md) | N curve with the original recipe | 01 | +0.27 bit per doubling; 1.76 bits at 241 | confirmed |
| [R12](R12_film.md) | FiLM in the decoder: the gain grows with N | 02 | +0.44 bit at 241 | partial |
| [R13](R13_smaller_latent_and_control.md) | Smaller latent: the loss comes mostly from re-initialization | 05 | re-init −0.78; 32 → 8 dimensions −0.40 | prediction failed (confounded) |
| [R14](R14_half_default_voice.md) | Half the default voice: +4 points of copying, same bits | 06 | +0.08 bit | effect of the measure |
| [R15](R15_kl_ramp.md) | The KL ramp made longer training worse | 03 | no KL: 1.81 → 2.44 → 2.84 | confirmed |
| [R16](R16_N_curve_without_kl.md) | N curve without KL | 01 | +0.41 bit per doubling; 2.90 at 241 | confirmed |
| [R17](R17_film_without_kl.md) | FiLM without KL | 02 | 3.08 bits at 241 (best) | promising |
| [R18](R18_training_length.md) | Training length: saturating near 10–20k steps | 03 | +0.63, +0.40, +0.12 per doubling | indication |
| [R19](R19_speakers_vs_sentences.md) | With the same data, speakers yield more than sentences | 04 | 1,024 sentences: 0.74 (8 spk.) vs 1.36 (128 spk.) | confirmed |
| [R20](R20_more_sentences_per_speaker.md) | 64 sentences per speaker raise the curve; the 128→241 gap stays small | 04, 01 | 241 speakers × 64: 3.17 bits | mixed |
| [R21](R21_seeds_at_10k_and_20k_steps.md) | No-KL curve holds with 3 seeds; 20k steps add little at N = 128 | 01, 03 | 2.83 → 2.86 (10k → 20k) | confirmed / missed |
| [R22](R22_reinit_control_across_N.md) | Re-init cost 0.35–0.78 bit at every N; z8 below the control everywhere | 05 | — | mostly hit |
| [R23](R23_film_vs_nokl_at_10k.md) | At 10k steps FiLM ≈ no KL; FiLM at 241, 20k steps: 3.18 bits | 02 | +0.04 mean advantage (N ≥ 64) | hit; main test pending |

## Data

[`data/`](data/README.md): a table of every cluster evaluation. The tables of R11 onward come from it via `python scripts/coverage/results_table.py`.
