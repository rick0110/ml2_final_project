# Experimental protocol

How the runs of the coverage experiment are set up and evaluated. It applies to every result from R11 on (cluster). Results R01–R10 (Colab phase) use the same setup, with the differences stated in each file.

## Setting up a run

- **Starting point:** the English model already trained (Tacotron 2 + VAE, fine-tuned from NVIDIA's pretrained Tacotron 2), always the same checkpoint (`ckpt/en_final`).
- **Training data:** 2,000 LJSpeech utterances (the "default voice", one speaker) + **N LibriTTS train-clean-100 speakers with K sentences each**. Selections are nested: the speakers of N = 8 are inside N = 32, the sentences of K = 8 inside K = 32 (`scripts/coverage/coverage_manifest.py`). At most 241 speakers are available.
- **Training:** batch 48, learning rate 1e-4 with 500 warm-up steps, a checkpoint every 500 steps, 2,500 steps by default.
- **KL ramp (original recipe):** the KL weight stays at 0 until step 2,000 and then rises by 0.05 every 1,000 steps (0.15 at step 5,000). The `-nokl` variant turns the ramp off.
- **Seeds:** the number after `_s` in the run name changes the data order and the initialization of anything new.

## Run names

`n{N}_k{K}_s{seed}[-variant]`, for example `n128_k32_s2-nokl`. Variants (defined in `scripts/coverage/colab_job.sh`):

| Suffix | What changes |
|---|---|
| (none) | original recipe |
| `-film` | the style vector also modulates, through FiLM, the decoder LSTM states at every step (zero-initialized) |
| `-nokl` | no KL ramp (KL weight always 0) |
| `-filmnokl` | FiLM and no KL ramp |
| `-z8`, `-z2` | 8- or 2-dimensional latent (VAE heads re-initialized) |
| `-zr32` | 32-dimensional latent with the VAE heads re-initialized (control for `-z8`/`-z2`) |
| `-lj1000` | 1,000 default-voice utterances instead of 2,000 |
| `-swap`, `-contrast`, `-spk`, `-spkdrop`, `-aug`, `-vcaug` | Colab-phase variants (R02–R05) |

The number of steps goes in the job queue, not in the name: the same run can be evaluated at 5,000 and 10,000 steps. Asking for more steps on an existing run resumes from its last checkpoint (the learning rate is constant after warm-up, so resuming is equivalent to training straight through).

## Evaluation

For each evaluated checkpoint:

- **25 test sentences** outside the training set.
- **References:** 8 trained voices, 4 unseen voices from the same corpus and **38 external voices** (LibriTTS test-clean, which no model has heard). Conclusions use the 38 external voices: with 4 voices, the noise between seeds reached 18 points.
- **Synthesis** at full style strength (1.0) with the universal HiFi-GAN vocoder.

## Metrics

- **Voice bits** (`bits_ce`, the main metric): mutual information between the voice of the generated audio and the identity of the reference, C = I(output voice; reference), estimated by the cross-entropy bound from identification among the 38 external voices (`scripts/coverage/voice_bits.py`). Ceiling: log2 38 = 5.25 bits. The Fano bound (`bits_fano`, more conservative), H(Y) and H(Y|S) are computed too.
- **Copied:** share of samples whose audio is closer to the reference than to the default voice (`sim_ref > sim_default`, Resemblyzer speaker encoder). Easy to read, but it depends on each voice's distance to the default voice (see [path 06](paths/06_default_voice_pull/README.md)).
- **WER:** word error rate, transcribing the generated audio with Whisper. It measures intelligibility.

## Generating tables

```
python scripts/coverage/results_table.py                         # everything
python scripts/coverage/results_table.py --variant nokl --step 10000
```

The data come from `research/results/data/cluster_runs.csv`: one row per evaluation (run, variant, N, K, seed, steps, bits, copied, WER, machine).
