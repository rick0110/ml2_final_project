# First-Step TTS Model Training

Complete training pipeline for a Text-to-Speech model combining:
- **Text Encoder**: FastSpeech (frozen)
- **Acoustic Decoder**: LSTM-based (trainable)
- **Style Extractor**: Global Style Tokens (trainable)
- **Vocoder**: HiFi_GAN (frozen)

## Architecture Overview

```
Text Input
    ↓
[Text Encoder (FastPitch)] → h_text
    ↓
Mel Spectrogram Input
    ↓
[GST] → z_style
    ↓
[Concatenate: h_text + z_style]
    ↓
[Acoustic Decoder (LSTM)] → M̂ (Predicted Mel)
    ↓
[HiFi_GAN Vocoder] → x̂(t) (Audio)
```

## Loss Functions

### 1. L1 Reconstruction Loss
Minimizes the absolute difference between predicted and target mel spectrograms:
$$L_{recon} = \sum_{i,j} |M̂_{i,j} - M_{i,j}|$$

### 2. Style Diversity Loss
Penalizes style embeddings from collapsing into the same space by maintaining minimum distance between embeddings:
$$L_{diversity} = \text{mean}(\text{ReLU}(\cos\_sim - (1 - \text{margin})))$$

### 3. Combined Loss
$$L_{total} = w_{recon} \cdot L_{recon} + w_{diversity} \cdot L_{diversity}$$

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch`: Deep learning framework
- `tqdm`: Progress bars
- `tensorboard`: Visualization
- `nemo[tts]`: Text-to-speech models (FastPitch, HiFi_GAN)
- `transformers`: HuBERT model

## Basic Usage

### Training from Scratch

```bash
cd /home/richard/project/ml2_final_project

python src/training/train_first_step/train.py \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --num-workers 4
```

### With Custom Hyperparameters

```bash
python src/training/train_first_step/train.py \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --weight-reconstruction 1.0 \
    --weight-diversity 0.5 \
    --diversity-margin 0.1 \
    --acoustic-decoder-hidden-size 256 \
    --acoustic-decoder-num-layers 3 \
    --style-embedding-dim 128 \
    --use-amp \
    --experiment-name my_experiment_v1
```

### Resume from Checkpoint

```bash
python src/training/train_first_step/train.py \
    --num-epochs 200 \
    --resume experiments/step_1/attempt_YYYYMMDD_HHMMSS/checkpoints/best.pt
```

## Command-Line Arguments

### Training Hyperparameters
- `--num-epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--weight-decay`: L2 regularization (default: 1e-5)
- `--num-workers`: Data loading workers (default: 4)

### Loss Weights
- `--weight-reconstruction`: L1 loss weight (default: 1.0)
- `--weight-diversity`: Style diversity loss weight (default: 0.5)
- `--diversity-margin`: Minimum margin for style diversity (default: 0.1)

### Model Architecture
- `--acoustic-decoder-hidden-size`: LSTM hidden size (default: 256)
- `--acoustic-decoder-num-layers`: LSTM layers (default: 3)
- `--style-embedding-dim`: Style embedding dimension (default: 128)

### Training Configuration
- `--use-amp`: Use Automatic Mixed Precision
- `--resume`: Path to checkpoint for resuming
- `--val-split`: Validation set ratio (default: 0.1)
- `--experiment-name`: Custom experiment name
- `--seed`: Random seed (default: 42)

## Output Structure

The training script automatically creates the following structure:

```
experiments/
└── step_1/
    └── attempt_YYYYMMDD_HHMMSS/
        ├── config.json                 # Saved hyperparameters
        ├── checkpoints/                # Model checkpoints
        │   ├── epoch_0001.pt
        │   ├── epoch_0002.pt
        │   ├── ...
        │   └── best.pt                 # Best checkpoint by validation loss
        ├── tensorboard/                # TensorBoard logs
        │   └── events.out.tfevents...
        └── logs/                       # Training logs (placeholder)
```

## Monitoring Training

### Using TensorBoard

```bash
# Navigate to tensorboard directory
tensorboard --logdir experiments/step_1/attempt_YYYYMMDD_HHMMSS/tensorboard

# View at: http://localhost:6006
```

TensorBoard logs include:
- **train/loss**: Total training loss
- **train/recon_loss**: L1 reconstruction loss
- **train/div_loss**: Style diversity loss
- **val/loss**: Total validation loss
- **val/recon_loss**: Validation reconstruction loss
- **val/div_loss**: Validation diversity loss
- **Model info**: Parameter counts

## File Structure

```
src/training/train_first_step/
├── __init__.py              # Module initialization
├── train.py                 # Main training script
├── losses.py                # Loss function definitions
├── model_loader.py          # Model loading and initialization
├── train_utils.py           # Training utilities (epoch loop, checkpointing, logging)
└── README.md                # This file
```

### Key Modules

#### `losses.py`
- `L1ReconstructionLoss`: MSE between mel spectrograms
- `StyleDiversityLoss`: Penalizes style embedding collapse
- `CombinedTTSLoss`: Combined loss function

#### `model_loader.py`
- `FirstStepTTSModel`: Complete TTS pipeline
- `load_tts_models()`: Initialize all components
- `get_model_size_info()`: Model parameter information

#### `train_utils.py`
- `train_epoch()`: Single epoch training loop with tqdm
- `validate_epoch()`: Validation loop
- `save_checkpoint()`: Save model and optimizer state
- `load_checkpoint()`: Load from checkpoint
- `TensorBoardLogger`: TensorBoard integration
- `MetricsTracker`: Track metrics across batches

#### `train.py`
- Complete end-to-end training pipeline
- Argument parsing and configuration
- Dataset loading (LibriSpeech-PT and TTS Portuguese)
- Training/validation loops
- Checkpoint management
- TensorBoard logging

## Training Tips

1. **Start with small batch size** if memory is limited
2. **Adjust loss weights** based on your priority:
   - Higher `weight-reconstruction` for better mel quality
   - Higher `weight-diversity` for more style variation
3. **Use AMP** (`--use-amp`) for faster training on GPU
4. **Monitor validation loss** to detect overfitting
5. **Save checkpoints frequently** - best.pt saves automatically
6. **Use TensorBoard** to visualize training progress

## Example Training Session

```bash
# Full training with logging
python src/training/train_first_step/train.py \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --weight-reconstruction 1.0 \
    --weight-diversity 0.5 \
    --acoustic-decoder-hidden-size 512 \
    --acoustic-decoder-num-layers 4 \
    --style-embedding-dim 256 \
    --use-amp \
    --num-workers 8 \
    --experiment-name first_training
```

Monitor with:
```bash
tensorboard --logdir experiments/step_1/first_training/tensorboard
```

## Notes

- **Text Encoder (FastPitch)** and **Vocoder (HiFi_GAN)** are frozen during training
- **Acoustic Decoder** and **GST** are trainable
- The script handles dataset loading from both LibriSpeech-PT and TTS Portuguese
- Text tokenization is currently simplified (random IDs); implement proper tokenizer for production
- All progress bars use tqdm for real-time monitoring
- Checkpoints include full training state for resuming

## Future Improvements

- [ ] Implement proper text tokenizer
- [ ] Add audio visualization to TensorBoard
- [ ] Add learning rate scheduling
- [ ] Add gradient clipping
- [ ] Implement distributed training (DDP)
- [ ] Add inference script
- [ ] Add evaluation metrics (MCD, mel loss statistics)
