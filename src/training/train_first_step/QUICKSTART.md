# Quick Start Guide

## Installation

1. **Navigate to project directory:**
   ```bash
   cd /home/richard/project/ml2_final_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Testing Setup

Before training, verify everything is configured correctly:

```bash
python src/training/train_first_step/test_setup.py
```

This will test:
- Device availability (CUDA/CPU)
- Model loading
- Forward pass
- Loss functions
- Experiment directory structure

## Training Options

### Option 1: Quick Test (Fastest)

For development and debugging:

```bash
python src/training/train_first_step/run_training.py quick_test
```

**Settings:**
- Epochs: 5
- Batch Size: 8
- Model: Small
- Expected time: ~5-10 minutes

### Option 2: Balanced Training (Recommended)

For normal training:

```bash
python src/training/train_first_step/run_training.py balanced
```

**Settings:**
- Epochs: 100
- Batch Size: 32
- Model: Medium
- Expected time: ~2-4 hours

### Option 3: Production Training (Best Quality)

For best results (requires good GPU):

```bash
python src/training/train_first_step/run_training.py production
```

**Settings:**
- Epochs: 200
- Batch Size: 64
- Model: Large
- Expected time: ~8-12 hours

### Option 4: Custom Training

For custom hyperparameters:

```bash
python src/training/train_first_step/train.py \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --weight-reconstruction 1.0 \
    --weight-diversity 0.5 \
    --acoustic-decoder-hidden-size 256 \
    --acoustic-decoder-num-layers 3 \
    --style-embedding-dim 128 \
    --use-amp
```

## List Configurations

To see all available pre-configured training settings:

```bash
python src/training/train_first_step/configs.py
# or
python src/training/train_first_step/run_training.py --list
```

## Monitoring Training

### Real-time Progress

The training script shows:
- Progress bars with tqdm
- Loss values at each epoch
- Train vs Validation losses
- Best checkpoint indicator

### TensorBoard

After training starts, in another terminal:

```bash
# Find your experiment directory (see terminal output)
tensorboard --logdir experiments/step_1/attempt_YYYYMMDD_HHMMSS/tensorboard

# View at: http://localhost:6006
```

TensorBoard displays:
- Training curves (total, reconstruction, diversity losses)
- Validation curves
- Loss distributions
- Model information

## Output Structure

Training creates the following structure:

```
experiments/step_1/
└── attempt_YYYYMMDD_HHMMSS/
    ├── config.json              # Hyperparameters used
    ├── checkpoints/
    │   ├── epoch_0001.pt
    │   ├── epoch_0002.pt
    │   ├── ...
    │   └── best.pt              # Best model by validation loss
    ├── tensorboard/             # TensorBoard logs
    └── logs/                    # Optional: training logs
```

## Resume from Checkpoint

Continue training from a previous checkpoint:

```bash
python src/training/train_first_step/train.py \
    --num-epochs 200 \
    --resume experiments/step_1/attempt_YYYYMMDD_HHMMSS/checkpoints/best.pt
```

## Common Commands

### View available configs
```bash
python src/training/train_first_step/configs.py
```

### Dry run (see command without executing)
```bash
python src/training/train_first_step/run_training.py balanced --dry-run
```

### Check GPU status
```bash
nvidia-smi
```

### Monitor memory usage during training
```bash
watch -n 1 nvidia-smi
```

## Tips for Success

1. **Start small:** Use `quick_test` to verify setup works
2. **Monitor GPU:** Keep `nvidia-smi` running to watch memory usage
3. **Use TensorBoard:** Monitor losses in real-time during training
4. **Save often:** Checkpoints are saved every epoch
5. **Resume as needed:** Training can be interrupted and resumed

## File Structure

```
src/training/train_first_step/
├── train.py                 # Main training script
├── losses.py                # Loss function definitions
├── model_loader.py          # Model initialization
├── train_utils.py           # Training utilities
├── text_processing.py       # Text tokenization
├── configs.py               # Pre-configured settings
├── run_training.py          # Convenience script
├── test_setup.py            # Setup verification
├── __init__.py              # Module initialization
├── README.md                # Full documentation
└── QUICKSTART.md            # This file
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size`
- Use `--use-amp` for mixed precision
- Use smaller model (less layers, smaller hidden size)

### Slow Training
- Increase `--num-workers` for data loading
- Use `--use-amp` for faster computation
- Ensure GPU is being used (check `nvidia-smi`)

### Data Loading Issues
- Verify dataset files exist in `data/processed/`
- Check dataset metadata CSV files are valid
- Increase `--num-workers` if data loading is bottleneck

### Model Loading Issues
- Ensure pre-trained models can be downloaded
- Check internet connection
- Try `pip install --upgrade nemo-toolkit`

## Next Steps

After training completes:

1. **Evaluate best checkpoint:**
   - Load `experiments/step_1/.../checkpoints/best.pt`
   - Test on validation set
   - Generate sample outputs

2. **Inference:**
   - Create inference script using best checkpoint
   - Generate audio from text samples

3. **Analysis:**
   - Compare mel-spectrograms (predicted vs target)
   - Listen to generated audio
   - Analyze style diversity metrics

For full documentation, see [README.md](README.md).
