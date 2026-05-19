# Pre-Training Checklist

Use this checklist to ensure everything is ready before starting training.

## Environment Setup

- [ ] Python 3.8+ installed (`python --version`)
- [ ] CUDA 11.0+ installed (for GPU) (`nvidia-smi`)
- [ ] PyTorch with CUDA support installed (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Project directory accessible (`cd /home/richard/project/ml2_final_project`)

## Dependencies

- [ ] `requirements.txt` packages installed (`pip list | grep torch`)
- [ ] NeMo TTS models available (`pip list | grep nemo`)
- [ ] TensorBoard installed (`pip list | grep tensorboard`)
- [ ] TQDM installed (`pip list | grep tqdm`)

## Data

- [ ] LibriSpeech-PT dataset exists (`ls data/processed/libriSpeech-pt/`)
- [ ] TTS Portuguese dataset exists (`ls data/processed/tts_portuguese/`)
- [ ] Metadata CSV files present in dataset folders
- [ ] Mel spectrogram .pt files accessible

## Model Files

- [ ] Pre-trained models can be downloaded:
  - [ ] FastPitch (for text encoding)
  - [ ] HiFi-GAN (for vocoding)
  - [ ] Check `local_weight_models/` folder (optional)

## Training Scripts

- [ ] `train.py` exists and is executable
- [ ] `losses.py` exists with loss function definitions
- [ ] `model_loader.py` exists with model initialization
- [ ] `train_utils.py` exists with training utilities
- [ ] All imports can be resolved (`python test_setup.py`)

## Directories

- [ ] `experiments/` directory exists or will be created
- [ ] `experiments/step_1/` will be created on first run
- [ ] `logs/` directory accessible for logs

## Configuration

- [ ] Choose training preset:
  - [ ] `quick_test` (for testing)
  - [ ] `balanced` (for normal training)
  - [ ] `production` (for best results)
  - [ ] Custom (with specific hyperparameters)

- [ ] Hyperparameters decided:
  - [ ] Batch size (default: 32)
  - [ ] Learning rate (default: 1e-3)
  - [ ] Number of epochs (default: 100)
  - [ ] Loss weights (reconstruction: 1.0, diversity: 0.5)

## Hardware

- [ ] GPU available with sufficient VRAM:
  - [ ] Quick test: 4GB minimum
  - [ ] Balanced: 8GB recommended
  - [ ] Production: 16GB+ recommended

- [ ] Disk space available:
  - [ ] Checkpoints: ~100MB per checkpoint (will have multiple)
  - [ ] TensorBoard logs: ~50MB for full training
  - [ ] Total: Plan for ~5-10GB

- [ ] CPU cores available for data loading:
  - [ ] Check: `nproc --all`
  - [ ] Set `--num-workers` to half of CPU cores

## Testing

- [ ] Run setup test:
  ```bash
  python src/training/train_first_step/test_setup.py
  ```
  - [ ] Device test passes
  - [ ] Model loading succeeds
  - [ ] Forward pass works
  - [ ] Loss functions compute
  - [ ] Experiment structure creates

- [ ] Verify quick test can run:
  ```bash
  python src/training/train_first_step/run_training.py quick_test --dry-run
  ```

## Final Checks

- [ ] Working directory is correct: `/home/richard/project/ml2_final_project`
- [ ] Virtual environment activated (if using)
- [ ] GPU is available and not in use by other processes
- [ ] Monitor command ready (`nvidia-smi` or `watch nvidia-smi`)
- [ ] TensorBoard ready to launch

## Starting Training

Once all boxes are checked:

1. **Start training:**
   ```bash
   python src/training/train_first_step/train.py \
       --num-epochs 100 \
       --batch-size 32 \
       --learning-rate 1e-3
   ```

2. **Monitor in another terminal:**
   ```bash
   tensorboard --logdir experiments/step_1/attempt_*/tensorboard
   ```

3. **Watch GPU (optional):**
   ```bash
   watch -n 1 nvidia-smi
   ```

## First Run Expectations

- **Startup (~2-5 min):** Model loading and data preparation
- **First epoch (~5-15 min):** Depending on dataset size and batch size
- **Subsequent epochs (~5-15 min each):** Should be relatively consistent
- **Memory usage:** Should stabilize after first epoch
- **GPU utilization:** Should be >80% during training

## Troubleshooting During Training

If something goes wrong:

1. **Check error message** carefully
2. **Stop training** (Ctrl+C)
3. **Consult troubleshooting section** in [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
4. **Adjust parameters** as needed
5. **Resume training** from best checkpoint if appropriate

## Emergency Contacts

If you encounter issues:

1. Check `QUICKSTART.md` for common solutions
2. Read `TRAINING_GUIDE.md` troubleshooting section
3. Review error messages and logs
4. Try reducing batch size or model size
5. Verify data loading with smaller sample

---

**Ready to start?** ✓

Run the setup test to verify:
```bash
python src/training/train_first_step/test_setup.py
```

Good luck! 🚀
