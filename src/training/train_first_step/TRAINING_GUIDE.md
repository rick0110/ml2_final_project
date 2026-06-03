# Training Guide - Complete Reference

## Overview

This is a complete training pipeline for a first-step Text-to-Speech (TTS) model. The system combines multiple neural networks to generate speech from text with style control.

## Architecture

```
Text Input
    ↓
┌───────────────────────────────┐
│   Text Encoder (FastPitch)    │ ← frozen
│   "codec text to features"    │
└───────────────────────────────┘
    ↓ h_text

Mel Spectrogram (target)
    ↓
┌───────────────────────────────┐
│   GST (Style Extractor)       │ ← trainable
│   "extract style features"    │
└───────────────────────────────┘
    ↓ z_style

┌──────────────────────────────────────────────────┐
│  Concatenate: [h_text, z_style]                  │
└──────────────────────────────────────────────────┘
    ↓

┌───────────────────────────────────────────────────────────┐
│  Acoustic Decoder (LSTM)                                  │ ← trainable
│  "decode text+style features to mel spectrogram"         │
└───────────────────────────────────────────────────────────┘
    ↓ M_hat (predicted mel)

┌───────────────────────────────────────────────────┐
│   HiFi_GAN Vocoder                                │ ← frozen
│   "convert mel to waveform"                       │
└───────────────────────────────────────────────────┘
    ↓
Audio Output x_hat(t)
```

## Loss Functions

### 1. L1 Reconstruction Loss
Minimizes absolute difference between predicted and target mel spectrograms:
$$L_{recon} = |M̂ - M|_1$$

Encourages the acoustic decoder to accurately reconstruct mel spectrograms.

### 2. Style Diversity Loss
Prevents style embeddings from collapsing into the same space:
$$L_{div} = \text{mean}\left(\text{ReLU}\left(\text{cos\_sim}(e_i, e_j) - (1 - m)\right)\right)$$

Encourages different utterances to have diverse style representations.

### Total Loss
$$L_{total} = w_{recon} \cdot L_{recon} + w_{div} \cdot L_{div}$$

## File Structure

```
src/training/train_first_step/
│
├── 📝 Documentation
│   ├── README.md                 # Full documentation
│   ├── QUICKSTART.md             # Quick start guide
│   └── TRAINING_GUIDE.md         # This file
│
├── 🧠 Core Training Scripts
│   ├── train.py                  # Main training loop (run this!)
│   ├── losses.py                 # Loss functions
│   ├── model_loader.py           # Model initialization
│   └── train_utils.py            # Training utilities (epoch loop, checkpointing, logging)
│
├── 🛠️ Utilities
│   ├── text_processing.py        # Text tokenization
│   ├── configs.py                # Pre-configured training settings
│   ├── checkpoint_utils.py       # Checkpoint inspection tools
│   └── run_training.py           # Convenience script for pre-configs
│
├── 🧪 Testing & Setup
│   ├── test_setup.py             # Verify setup works
│   └── __init__.py               # Python package init
```

## Quick Start Commands

### 1. Verify Setup
```bash
python src/training/train_first_step/test_setup.py
```

### 2. Quick Test (5 epochs, 8 batch size)
```bash
python src/training/train_first_step/run_training.py quick_test
```

### 3. Balanced Training (100 epochs, 32 batch size) - Recommended
```bash
python src/training/train_first_step/run_training.py balanced
```

### 4. Production Training (200 epochs, 64 batch size)
```bash
python src/training/train_first_step/run_training.py production
```

### 5. Custom Training
```bash
python src/training/train_first_step/train.py \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --weight-reconstruction 1.0 \
    --weight-diversity 0.5
```

## Training Pipeline Details

### Data Loading
- Loads from two sources:
  - LibriSpeech-PT (Portuguese subset)
  - TTS-Portuguese (Portuguese corpus)
- Automatically combines datasets
- Splits into train/validation (default 90/10)
- Uses PyTorch DataLoader with multiple workers

### Model Training
1. **Batch loading:** Load mel spectrograms and text from dataset
2. **Text encoding:** Convert text to token IDs
3. **Forward pass:**
   - Text → FastPitch → h_text
   - Mel → GST → z_style
   - [h_text, z_style] → LSTM Decoder → M_hat
4. **Loss computation:** Combine reconstruction + diversity losses
5. **Backward pass:** Update trainable parameters (Decoder, GST)
6. **Checkpoint saving:** Save best model

### Validation
- Run after each epoch
- Compute losses on validation set
- Save checkpoint if validation loss improves
- Log metrics to TensorBoard

## Hyperparameters

### Training
- `--num-epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Optimizer learning rate
- `--weight-decay`: L2 regularization strength
- `--num-workers`: Data loading workers

### Loss Weighting
- `--weight-reconstruction`: Weight for mel reconstruction loss (default 1.0)
- `--weight-diversity`: Weight for style diversity loss (default 0.5)
- `--diversity-margin`: Minimum margin for diversity (default 0.1)

### Model Architecture
- `--acoustic-decoder-hidden-size`: LSTM hidden size (default 256)
- `--acoustic-decoder-num-layers`: Number of LSTM layers (default 3)
- `--style-embedding-dim`: Dimension of style embeddings (default 128)

### Experiment Control
- `--use-amp`: Enable Automatic Mixed Precision (faster, less memory)
- `--val-split`: Validation split ratio (default 0.1)
- `--resume`: Path to checkpoint to resume from
- `--seed`: Random seed for reproducibility
- `--experiment-name`: Custom name for experiment (auto-generated if not provided)

## Output Structure

```
experiments/
└── step_1/
    └── attempt_YYYYMMDD_HHMMSS/              # Each run creates timestamped folder
        ├── config.json                        # Saved hyperparameters
        ├── checkpoints/                       # Model snapshots
        │   ├── epoch_0001.pt                  # Checkpoint after each epoch
        │   ├── epoch_0002.pt
        │   ├── ...
        │   └── best.pt                        # Best by validation loss
        ├── tensorboard/                       # TensorBoard event files
        │   └── events.out.tfevents.XXXXX
        └── logs/                              # Optional: text logs
```

### Checkpoint Contents
Each checkpoint contains:
- `epoch`: Current epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `metrics`: Loss values and other metrics

## Monitoring Training

### Console Output
- Real-time progress bars with tqdm
- Epoch summaries with train/val losses
- Component breakdowns (reconstruction vs diversity)

### TensorBoard
```bash
tensorboard --logdir experiments/step_1/attempt_YYYYMMDD_HHMMSS/tensorboard
# View at http://localhost:6006
```

Plots include:
- `train/loss`, `train/recon_loss`, `train/div_loss`
- `val/loss`, `val/recon_loss`, `val/div_loss`
- Model architecture and parameter information

### Checkpoint Inspection
```bash
# List all experiments
python src/training/train_first_step/checkpoint_utils.py list-experiments

# List checkpoints in experiment
python src/training/train_first_step/checkpoint_utils.py list my_experiment

# Inspect specific checkpoint
python src/training/train_first_step/checkpoint_utils.py inspect path/to/checkpoint.pt

# Compare multiple checkpoints
python src/training/train_first_step/checkpoint_utils.py compare path/to/ckpt1.pt path/to/ckpt2.pt

# Extract metrics from all checkpoints
python src/training/train_first_step/checkpoint_utils.py metrics my_experiment --metric loss
```

## Pre-configured Experiments

### 1. Quick Test
```python
configs.py QUICK_TEST
```
- For debugging and development
- 5 epochs, small model
- Fast iteration

### 2. Balanced (Recommended)
```python
configs.py BALANCED
```
- Good quality with reasonable time
- 100 epochs, medium model
- ~2-4 hours training

### 3. Production
```python
configs.py PRODUCTION
```
- Best quality results
- 200 epochs, large model
- ~8-12 hours training
- Requires good GPU

### 4. High Diversity
```python
configs.py HIGH_DIVERSITY
```
- Emphasizes style diversity
- Higher diversity loss weight
- Better for style variation

### 5. Lightweight
```python
configs.py LIGHTWEIGHT
```
- For limited resources
- Small model, low batch size
- Faster but lower quality

## Training Tips & Best Practices

### Getting Started
1. Start with `quick_test` to verify everything works
2. Move to `balanced` for real training
3. Use `production` only for final models

### Performance Optimization
- Enable `--use-amp` for ~30% faster training
- Increase `--num-workers` for faster data loading
- Use larger `--batch-size` if GPU memory allows
- Monitor GPU utilization with `nvidia-smi`

### Quality Improvements
- Increase `--num-epochs` for better convergence
- Adjust `--learning-rate` if loss not decreasing
- Tune loss weights for your application:
  - Higher `--weight-reconstruction` for accurate mel
  - Higher `--weight-diversity` for style variation
- Larger `--acoustic-decoder-hidden-size` for higher capacity

### Debugging
- If training is slow: increase batch size, enable AMP, check GPU usage
- If loss not decreasing: try higher learning rate or check data
- If memory errors: reduce batch size or model size
- If diverging: reduce learning rate or check loss weights

## Resuming Training

To resume from a checkpoint:
```bash
python src/training/train_first_step/train.py \
    --num-epochs 200 \
    --resume experiments/step_1/my_experiment/checkpoints/best.pt
```

This will:
- Load model weights from checkpoint
- Load optimizer state
- Start from the next epoch
- Continue training with same or different hyperparameters

## Model Architecture Details

### Text Encoder (Frozen)
- FastPitch from NeMo
- Pre-trained on English TTS
- Converts text to 384-dim features
- Frozen during training

### Acoustic Decoder (Trainable)
- 2-4 layer LSTM
- Input: concatenated [h_text, z_style]
- Output: 80-dimensional mel spectrogram
- ~200k-500k parameters

### Style Extractor (Trainable)
- Global Style Tokens (GST)
- Convolution layers + GRU + multi-head attention
- Extracts style from mel spectrogram
- Outputs 64-256 dimensional embeddings

### Vocoder (Frozen)
- HiFi_GAN from NeMo
- Converts mel spectrogram to waveform
- Pre-trained on high-quality audio
- Frozen during training

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Solution: Reduce `--batch-size` or `--style-embedding-dim`
- Or: Enable `--use-amp` for mixed precision
- Or: Use smaller model (fewer layers, smaller hidden size)

**Slow Data Loading**
- Solution: Increase `--num-workers`
- Or: Pre-cache data to SSD if possible
- Or: Use faster storage medium

**Training Loss Not Decreasing**
- Solution: Check learning rate (might be too high or too low)
- Or: Try different `--weight-decay` value
- Or: Verify data is loading correctly

**Cannot Load Pre-trained Models**
- Solution: Check internet connection
- Or: Download manually to `local_weight_models/`
- Or: Check NeMo installation with `pip install --upgrade nemo-toolkit`

**Checkpoint Loading Error**
- Solution: Ensure checkpoint path is correct
- Or: Verify checkpoint was saved successfully
- Or: Try loading with CPU: `torch.load(..., map_location='cpu')`

## Advanced Usage

### Mixed Precision Training
```bash
python src/training/train_first_step/train.py \
    --batch-size 64 \
    --use-amp  # Enable automatic mixed precision
```

Benefits:
- ~30% faster training
- ~50% less GPU memory
- Similar accuracy

### Custom Data Loaders
Modify `create_datasets()` in `train.py` to:
- Load from different sources
- Apply custom preprocessing
- Change train/val split

### Loss Function Tuning
Adjust in command line:
```bash
python src/training/train_first_step/train.py \
    --weight-reconstruction 1.5 \  # Higher: better accuracy
    --weight-diversity 1.0 \        # Higher: more style variation
    --diversity-margin 0.2          # Higher: stronger separation
```

## References

### Architectures
- **FastPitch**: [arXiv:2106.00606](https://arxiv.org/abs/2106.00606)
- **HiFi_GAN**: [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)
- **GST**: [arXiv:1803.10135](https://arxiv.org/abs/1803.10135)

### Frameworks
- **PyTorch**: Deep learning framework
- **NeMo**: NVIDIA neural modules for TTS
- **TensorBoard**: Visualization toolkit

## Next Steps

1. **Run test setup**: Verify everything is working
2. **Try quick_test**: Get familiar with the pipeline
3. **Run balanced**: Start real training
4. **Monitor with TensorBoard**: Watch losses improve
5. **Generate samples**: Use trained model for inference
6. **Evaluate**: Compare with baseline models

---

For questions or issues, check:
- README.md for full documentation
- QUICKSTART.md for quick reference
- checkpoint_utils.py for inspecting results
- TensorBoard for visualization
