# ml2_final_project

This repository implements a model for **prosody and style transfer in audio for Portuguese**, designed to work with low computational cost and minimal labelled data.

## Problem Statement

Most open-source voice conversion and TTS models are trained on English and Chinese data. High-quality Portuguese (particularly Brazilian Portuguese) datasets are scarce, making it difficult to train models from scratch.

This project proposes a transfer-learning approach that:

1. Uses a pretrained **HuBERT** model (English/multilingual) as a frozen content encoder.
2. Trains an **intermediate mapping network** to bridge HuBERT's latent space and the VITS decoder's feature space — requiring only a small Portuguese corpus.
3. Incorporates a **Reference Encoder + Global Style Tokens (GST)** for unsupervised prosody / style transfer.
4. Adds a **Variance Adaptor** (pitch, duration, energy predictors) for fine-grained prosody control.
5. Decodes features using a **HiFi-GAN vocoder** (VITS-style decoder).

---

## Architecture

```
 Source Audio ──► HuBERT (frozen) ──► content features
                                              │
 Reference Audio ──► GST Reference Encoder ──► style embedding
                                              │
                        ┌─────────────────────┤
                        ▼                     │
                 MappingNetwork  ◄─ style emb (FiLM)
                        │
                        ▼
                 VarianceAdaptor  (duration / pitch / energy)
                        │
                        ▼
                    Decoder  (MelPredictor + HiFi-GAN)
                        │
                 Output Waveform
```

### Sub-modules

| Module | File | Description |
|---|---|---|
| `ContentEncoder` | `src/models/content_encoder.py` | HuBERT wrapper with optional projection |
| `GlobalStyleToken` | `src/models/reference_encoder.py` | GST reference encoder (convolutional + multi-head attention) |
| `MappingNetwork` | `src/models/mapping_network.py` | FiLM-conditioned residual mapping (HuBERT → decoder space) |
| `VarianceAdaptor` | `src/models/variance_adaptor.py` | Pitch / energy / duration predictors + length regulator |
| `Decoder` | `src/models/decoder.py` | Mel predictor + HiFi-GAN waveform generator |
| `ProsodyStyleTransferModel` | `src/models/full_model.py` | Full end-to-end model |

---

## Datasets

| Dataset | Purpose | Module |
|---|---|---|
| **TTS-Portuguese Corpus** | Neutral base mapping training | `TTSPortugueseDataset` |
| **LibriVox PT-BR** | Unsupervised prosody learning | `LibriVoxPTBRDataset` |
| **VERBO** | Style / emotion reference at inference | `VERBODataset` |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Edit data paths in configs/config.yaml first
python scripts/train.py --config configs/config.yaml

# Resume from checkpoint
python scripts/train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch0010.pt
```

---

## Inference

```bash
# Transfer prosody from reference.wav to source.wav
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --source path/to/source.wav \
    --reference path/to/reference.wav \
    --output output.wav

# Neutral style (no reference)
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --source path/to/source.wav \
    --output output.wav

# Batch processing
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --source_dir path/to/sources/ \
    --reference path/to/reference.wav \
    --output_dir outputs/
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ml2_final_project/
├── requirements.txt
├── setup.py
├── configs/
│   └── config.yaml          # Hyperparameters and data paths
├── src/
│   ├── models/
│   │   ├── content_encoder.py   # HuBERT content encoder
│   │   ├── reference_encoder.py # GST reference encoder
│   │   ├── variance_adaptor.py  # Variance adaptor
│   │   ├── mapping_network.py   # HuBERT → VITS mapping network
│   │   ├── decoder.py           # HiFi-GAN decoder
│   │   └── full_model.py        # Full end-to-end model
│   ├── data/
│   │   ├── preprocessing.py     # Audio feature extraction
│   │   └── dataset.py           # Dataset loaders
│   ├── training/
│   │   ├── losses.py            # Loss functions
│   │   └── trainer.py           # Training loop
│   └── inference/
│       └── pipeline.py          # Inference pipeline
├── scripts/
│   ├── train.py                 # Training entry-point
│   └── inference.py             # Inference entry-point
└── tests/
    ├── test_models.py           # Model unit tests
    ├── test_data.py             # Data pipeline unit tests
    └── test_losses.py           # Loss function tests
```

---

## References

- HuBERT: [Hsu et al., 2021](https://arxiv.org/abs/2106.07447)
- VITS: [Kim et al., 2021](https://arxiv.org/abs/2106.06103)
- Global Style Tokens: [Wang et al., 2018](https://arxiv.org/abs/1803.09017)
- FastSpeech 2: [Ren et al., 2021](https://arxiv.org/abs/2006.04558)
- HiFi-GAN: [Kong et al., 2020](https://arxiv.org/abs/2010.05646)
- TTS-Portuguese Corpus: [Casanova et al., 2022](https://arxiv.org/abs/2201.01756)
- VERBO: Brazilian Portuguese emotional speech corpus
