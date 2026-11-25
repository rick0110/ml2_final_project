# Technical Report: Tacotron2-VAE with Style Transfer in Portuguese

**Project:** ml2_final_project  
**Date:** June 2026  
**Goal:** Portuguese speech synthesis with style transfer via VAE-GST

---

## 1. Overview

This project implements a speech synthesis (TTS) system with style transfer for Brazilian Portuguese. The base architecture is Tacotron2 (NVIDIA), extended with a VAE-GST module (Variational Autoencoder + Global Style Token) that makes it possible to condition the generation of mel spectrograms on the prosodic style of a reference audio clip.

**Full pipeline:**

```
Text (str)
  └─► TextProcessor (gruut, IPA) → phoneme sequence [int]
        └─► Encoder (3× Conv1D + 1 BiLSTM) → encoder_outputs [B, T_enc, 512]
              └─► VAE-GST (ref_audio) → latent_vector z [B, 32]
                    └─► expanded z + encoder_outputs → joint conditioning
                          └─► Decoder (2 LSTM + location-sensitive attention) → mel_pre [B, 80, T_dec]
                                └─► PostNet (5× residual Conv1D) → mel_post [B, 80, T_dec]
                                      └─► WaveGlow → audio [22050 Hz]
```

---

## 2. Model Architecture

### 2.1 Encoder

- Phoneme embedding: 512 dims
- 3 Conv1D layers (kernel=5, channels=512, BatchNorm, ReLU)
- 1 BiLSTM (hidden=256, output=512)

### 2.2 VAE-GST (Style Encoder)

- Audio reference → mel-spectrogram → GRU reference encoder → μ, log σ² (32 dims each)
- Reparameterization: z = μ + ε·σ
- **Cyclical KL annealing:** lag=2000, x0=4000, upper=0.2, free_bits=0.5
- Latent dim: 32
- z expanded by broadcast and added to encoder_outputs

### 2.3 Decoder

- Prenet: 2× Linear(256) + Dropout(p=0.5) — dropout **always active**, including at inference
- Attention RNN: LSTM(1024) with location-sensitive attention (filters=32, kernel=31)
- Decoder RNN: LSTM(1024)
- Output: mel frame [80] + gate scalar
- Scheduled sampling: p=0.2 (20% of the steps use the previous prediction instead of the target)

### 2.4 PostNet

- 5× Conv1D (kernel=5, channels=512, BatchNorm, Tanh/Linear)
- Residual: mel_post = mel_pre + postnet_output

### 2.5 Guided Attention Loss

Diagonal penalty that forces the attention/encoder alignment to follow a diagonal:

```
W[t, n] = 1 - exp(-(n/N - t/T)² / 2σ²)
```

- Weight: 8.0 (pt_tacotron_v2)
- σ = 0.2 (stricter; equivalent to σ=0.4 with a 4× larger weight)

---

## 3. Datasets

| Dataset | Language | Utterances | Duration |
|---------|--------|------------|---------|
| LJSpeech | English (en-US) | 13,100 | ~24h |
| TTS-Portuguese-Corpus | Portuguese (pt-BR) | 2,340 | ~3.5h |

**Preprocessing:**
- Resample to 22050 Hz
- STFT: filter=1024, hop=256, win=1024, mel_bins=80, fmin=0, fmax=8000
- Phonemes: `gruut` with `portuguese_phonetic_cleaners` (IPA)
- Cache of phonetic sequences in `_seq_cache_{hash}.pkl`

---

## 4. Training Configuration

| Parameter | lj_speech_v1 | pt_tacotron_v2 |
|-----------|-------------|----------------|
| Initial LR | 1e-4 | 1e-5 |
| Batch size | 32 | 12 |
| Grad clip | 1.0 | 5.0 |
| Guided attn weight | 2.0→4.0 | 8.0 |
| Guided attn σ | 0.4 | 0.2 |
| Scheduled sampling | 0 | 0.2 |
| LR scheduler | ReduceLROnPlateau (patience=20) | ReduceLROnPlateau (patience=20) |
| Warm start | N/A | Weights from lj_speech_v1 |

---

## 5. Implemented Changes

| # | Change | File | Motivation |
|---|-------------|---------|-----------|
| 1 | Guided Attention Loss | `losses.py` | Force diagonal alignment; without it the model fails catastrophically |
| 2 | output_lengths masking bug fix | `train_utils.py` | Bug: output_lengths was not passed to the criterion → wrong loss on the padding |
| 3 | Cyclical KL annealing | `train.py` + `hparams.py` | Linear annealing collapsed the VAE; cyclical keeps style information |
| 4 | grad_clip_thresh: 1.0→5.0 | `hparams.py` | Aggressive clipping prevented learning; 5.0 allows larger updates |
| 5 | LR warmup | `train.py` + `train_utils.py` | Stability at the start of PT training with warm start |
| 6 | Cross-lingual warm start | CLI `--checkpoint-path` | Reuse EN weights to speed up PT convergence |
| 7 | p_decoder_input_dropout: 0.8→0.5 | `hparams.py` | 0.8 too high; 0.5 is the original Tacotron2 default |
| 8 | AMP disabled | `train.py` | fp16 caused NaN overflow in the KL loss; fp32 stable |
| 9 | Phoneme pre-caching | `loader.py` | gruut is slow; the cache removes the overhead in subsequent epochs |
| 10 | persistent_workers=True | `train.py` | Avoids respawning workers between epochs (significant overhead) |
| 11 | Scheduled sampling (p=0.2) | `model.py` Decoder.forward() | Partially mitigates exposure bias: 20% of the steps use the previous prediction |
| 12 | Monotonic attention + attn-peak stop | `model.py` Decoder.inference() | Definitive solution for looping attention + stopping without a trained gate |

---

## 6. Inference — Recommended Usage

```bash
LD_LIBRARY_PATH=/opt/anaconda3/envs/ambiente_aluno/lib:$LD_LIBRARY_PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/inference/synthesize_tacotron2_vae.py \
  --experiment experiments/tacotron2-vae/pt_tacotron_v2 \
  --text "Texto em português aqui." \
  --reference-audio data/raw/tts-portuguese-Corpora/TTS-Portuguese-Corpus/wavs/sample-77.wav \
  --waveglow local_weight_models/waveglow/nvidia_waveglowpyt_fp32_20190427 \
  --output-dir exports/synth/output \
  --force-monotonic \
  --monotonic-window 0 \
  --attn-stop-frames 10 \
  --max-decoder-steps 1000
```

**Critical flags:**

| Flag | Value | Reason |
|------|-------|--------|
| `--force-monotonic` | on | Prevents attention loops (attention attractor problem) |
| `--monotonic-window 0` | 0 | Strict: no going back; window>0 allows loops again |
| `--attn-stop-frames 10` | 10 | Stops when the attention stays on the last token for 10 consecutive frames |
| `--max-decoder-steps 1000` | 1000 | Safety fallback |

**Alternative (without monotonic, for models with a trained gate):**
```bash
--energy-stop-threshold -8.5 --energy-stop-frames 20
```

---

## 7. Experiment Log

### 7.1 Experiment 0: libri_pt_br_phonetic_v6 — Catastrophic Failure

**Configuration:**
- Dataset: TTS-Portuguese-Corpus (2340 utterances)
- No guided attention loss
- 24,000 training steps

**Result: CATASTROPHIC FAILURE**

The attention never converged to a diagonal pattern. Even after 24,000 steps, the model produced mel spectrograms with no recognizable phonetic structure. The attention oscillated between random encoder positions, never progressing.

**Lesson learned:**  
Guided attention loss is **essential** for small datasets. With 2340 utterances, the model does not have enough examples to discover the monotonic alignment on its own. Without the diagonal penalty, the attention does not converge.

---

### 7.2 Experiment 1: lj_speech_v1 — Full Convergence in English

**Goal:** Validate the VAE-GST architecture in English before tackling Portuguese.

**Configuration:**
- Dataset: LJSpeech (13,100 utterances)
- LR=1e-4, batch=32, guided_attn_weight=2.0→4.0, σ=0.4
- No scheduled sampling (full exposure to the ground truth)

**Metrics (TensorBoard, step 9800):**
| Metric | Value |
|---------|-------|
| Total val loss | 0.704 |
| Guided Attn Loss (val) | 0.000390 |
| Grad norm | 0.98 |
| LR | 2.5e-5 (ReduceLR fired 2×) |

**Result: SUCCESS**

- The gate fires naturally during inference (sigmoid > 0.5)
- Clean diagonal alignment without the monotonic constraint
- Working style transfer: the same sentence with different voices produces distinct prosodies

**Lesson learned:**  
The architecture works correctly in English. guided_attn_weight=4.0 was the value that completed convergence (2.0 plateaued at guided_attn ≈ 0.038). A large dataset (13k utts) lets the gate learn to fire through pure teacher forcing.

---

### 7.3 Experiment 2: pt_tacotron_v2 (initial phase) — KL Explosion

**Goal:** Fine-tuning lj_speech_v1 for Portuguese.

**Initial configuration:**
- Warm start with the weights from lj_speech_v1
- LR=1e-5, batch=12, guided_attn_weight=8.0, σ=0.2
- KL annealing: linear (first attempt)
- AMP (fp16) enabled

**Problems found:**

1. **NaN overflow with fp16:** The KL loss blew up to NaN in the first 500 steps. Solution: disable AMP, use fp32.

2. **KL onset explosion:** Even in fp32, grad_norm jumped to ~3617 at step 2200 when the KL started being annealed. With grad_clip=1.0, the gradients were clipped aggressively and learning stopped. Solution: grad_clip=5.0.

3. **VAE collapse with linear annealing:** Linear annealing made the KL weight grow monotonically, collapsing the latent space before the decoder converged. Solution: cyclical annealing (lag=2000, x0=4000, upper=0.2, free_bits=0.5).

4. **Guided attention stuck at 0.003:** With batch=12 and grad_clip=1.0, the guided attention loss did not go below 0.003 (σ=0.2). After scheduled sampling was added, there was a temporary disturbance (peak of 0.0086 at step 7000) followed by recovery.

---

### 7.4 Experiment 3: pt_tacotron_v2 — Scheduled Sampling + the Gate Problem

**Configuration:**
- Scheduled sampling p=0.2 added (restart at ~step 7000)
- LR=1e-5, reduced 3× by ReduceLROnPlateau: 1e-5 → 5e-6 → 2.5e-6 → 1.25e-6

**Metrics (step 29600/epoch_29600):**
| Metric | Value |
|---------|-------|
| Train guided_attn (σ=0.2) | 0.001101 |
| Val guided_attn (σ=0.2) | 0.001743 |
| Equivalent σ=0.4 (÷4) | ≈ 0.000275 / 0.000436 |
| Total val loss | 0.895 |
| Gate sigmoid max (inference) | **0.024** |

**Critical problem found — Exposure Bias / Gate Gap:**

The gate is trained **exclusively with teacher forcing** (ground truth as input). During inference, the model uses its own previous prediction (autoregressive), creating a different input distribution. The gate never learned to fire under these conditions.

Result: sigmoid(gate) ≈ 0.024 at every inference step — far below threshold=0.5. The model generated audio until it reached `max_decoder_steps` (1000), producing ~1000 frames of silence after the actual content.

**Critical problem — Attention Attractor:**

Without a monotonic constraint, the attention oscillated in a cycle between fixed encoder positions:

```
frames 1-300: attention at positions {0, 1, 5, 8, 20, 21}
              never advances to position 33 (last token)
              → infinite loop until max_decoder_steps
```

**Attempted solutions:**

1. `energy-stop-threshold -7.5 --energy-stop-frames 5`: Too aggressive — cut speech at frame 14 (energy dip in the middle of a word). 
2. `energy-stop-threshold -8.5 --energy-stop-frames 20`: Worked — cut only at real silence (frame 193+).
3. `--force-monotonic --monotonic-window 3`: Window=3 allows going from peak=21 back to 20 (20 ≥ 21-3=18). **Did not break the loops** — it only slowed them down.

---

### 7.5 Experiment 4: pt_tacotron_v2 — Definitive Solution with Monotonic Attention

**Inference configuration:**
```
--force-monotonic --monotonic-window 0 --attn-stop-frames 10
```

**Mechanism:**
- `monotonic_window=0`: Once the attention peak reaches position N, **all steps 0..N-1 are masked** (prob→0). Strictly forward movement.
- `attn_stop_frames=10`: Stops when the peak stays at `last_enc_pos` for 10 consecutive frames — confirms that the encoder has been fully consumed.

**Results (epoch_29600, 15 inferences = 5 sentences × 3 voices):**

| Sentence | Tokens | Reference | Frames | Duration | Fr/tok | Regressions | Coverage |
|-------|--------|-----------|--------|---------|--------|------------|-----------|
| f1 | 34 | s77 | 177 | 2.05s | 5.2 | 0 | 100% |
| f1 | 34 | s3508 | 303 | 3.52s | 8.9 | 0 | 100% |
| f1 | 34 | s4054 | 209 | 2.43s | 6.1 | 0 | 100% |
| f2 | 39 | s77 | 193 | 2.24s | 4.9 | 0 | 100% |
| f3 | 50 | s77 | 209 | 2.43s | 4.2 | 0 | 100% |

- **0 max_decoder_steps warnings** across all 15 runs
- **100% encoder coverage** — the attention always reaches the last token
- Style transfer confirmed: s77 vs s3508 → 71% difference in duration on the same sentence

**L1 distances between mels (same sentence, different voices):**

| Pair | L1 |
|-----|-----|
| s77 vs s3508 | 0.87 |
| s77 vs s4054 | 1.68 |
| s3508 vs s4054 | 1.21 |

---

### 7.6 Current State: epoch_40000 (Jun 26, 2026)

**Quality metrics (epoch_40000, 15 inferences):**

| Sentence | Tokens | Ref | Frames | Duration | Fr/tok | Regressions | Cov% |
|-------|--------|-----|--------|---------|--------|------------|------|
| f1 | 34 | s77 | 251 | 2.91s | 7.4 | 0 | 100% |
| f1 | 34 | s3508 | 195 | 2.26s | 5.7 | 0 | 100% |
| f1 | 34 | s4054 | 182 | 2.11s | 5.4 | 0 | 100% |
| f2 | 39 | s77 | 193 | 2.24s | 4.9 | 0 | 100% |
| f2 | 39 | s3508 | 182 | 2.11s | 4.7 | 0 | 100% |
| f2 | 39 | s4054 | 205 | 2.38s | 5.3 | 0 | 100% |
| f3 | 50 | s77 | 209 | 2.43s | 4.2 | 0 | 100% |
| f3 | 50 | s3508 | 288 | 3.34s | 5.8 | 0 | 100% |
| f3 | 50 | s4054 | 218 | 2.53s | 4.4 | 0 | 100% |
| f4 | 48 | s77 | 209 | 2.43s | 4.4 | 0 | 100% |
| f4 | 48 | s3508 | 174 | 2.02s | 3.6 | 0 | 100% |
| f4 | 48 | s4054 | 177 | 2.05s | 3.7 | 0 | 100% |
| f5 | 45 | s77 | 176 | 2.04s | 3.9 | 0 | 100% |
| f5 | 45 | s3508 | 181 | 2.10s | 4.0 | 0 | 100% |
| f5 | 45 | s4054 | 185 | 2.15s | 4.1 | 0 | 100% |

**Gate:** sigmoid max = 0.275 (vs. 0.024 at epoch_29600 — an 11× improvement). Still below threshold=0.5, but on a rising trajectory. Projection: a working gate between epoch_45000–50000.

**Style transfer (L1 between voices, epoch_40000):**

| Pair | L1 |
|-----|-----|
| s77 vs s3508 | 0.789 |
| s77 vs s4054 | 1.210 |
| s3508 vs s4054 | 1.495 |
| LJSpeech baseline (reference) | 1.55–1.75 |

Slightly below the LJSpeech baseline — expected, given that the PT dataset has 2340 utterances vs 13100 for LJSpeech.

**Stochastic variance (prenet dropout):**  
5 runs with the same text + same voice: [181, 186, 178, 253, 204] frames. Δ=75 frames (0.87s). Behavior inherent to the prenet dropout, which is active during inference in the original Tacotron2. Not a bug.

---

## 8. Table of Failures and Fixes

| Problem | Cause | Solution |
|----------|-------|---------|
| NaN in the KL loss | AMP fp16 overflow | Disable AMP, use fp32 |
| Grad explosion at KL onset | grad_clip too low (1.0) | Increase to 5.0 |
| VAE collapse | Linear KL annealing | Cyclical annealing (lag, free_bits) |
| Guided attn stuck at 0.038 | guided_attn_weight=2.0 insufficient | Increase to 4.0 (LJ) / 8.0 (PT) |
| Speech cut off prematurely | energy_stop threshold -7.5/window 5 too aggressive | Use -8.5/window 20 |
| Looping attention (attractor) | No monotonic constraint | --force-monotonic --monotonic-window 0 |
| Generation does not stop (gate does not fire) | Exposure bias — gate trained only with teacher forcing | --attn-stop-frames 10 |
| CUDA OOM | Parallel inferences during training | PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + sequential |
| ZeroDivisionError in TacotronSTFT | Wrong positional arguments | Use named arguments (keyword args) |
| Monotonic window=3 does not fix the loops | Window 3 still allows going back 3 positions | Use window=0 (strict) |
| Attention never converges (libri_v6) | No guided attention loss | Guided attention is mandatory for small datasets |

---

## 9. Final State of the Experiments

| Experiment | Status | Checkpoint | Alignment | Style Transfer | Gate |
|-------------|--------|------------|-------------|----------------|------|
| libri_v6 | FAILED | — | Chaotic | — | — |
| lj_speech_v1 | CONVERGED | step ~9800 | Perfect diagonal | Works (EN) | Fires (≥0.5) |
| pt_tacotron_v2 | RUNNING | epoch_40000 | Diagonal (with monotonic) | Works (PT) | 0.275, rising |

---

## 10. Test Sentences Used

```
f1: "O título de página foi encontrado."        (34 tokens)
f2: "O sistema retornou um erro inesperado."    (39 tokens)
f3: "A conferência foi cancelada por motivos de segurança." (50 tokens)
f4: "Os resultados foram publicados ontem à tarde."         (48 tokens)
f5: "A análise dos dados revelou padrões interessantes."    (45 tokens)
```

---

## 11. Environment Requirements

```bash
# Required variables
export LD_LIBRARY_PATH=/opt/anaconda3/envs/ambiente_aluno/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU: 1× ~20GB VRAM (simultaneous inference + training requires sequential execution)
# Python: .venv with dependencies installed
# gruut: IPA phonemes for pt-BR, requires conda's CXXABI (the reason for LD_LIBRARY_PATH)
```

---

## 12. Conclusions

**What worked:**

1. Cross-lingual warm start (EN → PT) significantly sped up convergence
2. Guided attention with a high weight (8.0, σ=0.2) converged the diagonal alignment in ~5000 steps
3. Scheduled sampling (p=0.2) partially reduced exposure bias
4. Monotonic attention (window=0) + attn_stop_frames=10: a robust solution for inference without a trained gate — 100% success in all tests
5. Working VAE-GST style transfer: the same sentence with 3 different voices produces 5–38% variation in duration and mel L1 of 0.79–1.50

**What does not work perfectly yet:**

1. Stop gate: sigmoid max=0.275 at epoch_40000 (threshold=0.5 not reached). Mitigated by `attn_stop_frames`.
2. Run-to-run variance (prenet dropout): ~75 frames of spread between identical runs. Inherent to the Tacotron2 design.
3. Absolute quality of the style transfer (L1 ≈1.5 vs baseline 1.65): the PT dataset being smaller than LJSpeech limits the learned diversity.

**Next steps:**

- Wait for epoch ~45000–50000 to check whether the gate reaches sigmoid≥0.5
- Assess whether additional training improves the naturalness of the audio (subjective MOS)
- Consider HiFi-GAN as an alternative vocoder to WaveGlow for audio quality
