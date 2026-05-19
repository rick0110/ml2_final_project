# Index - First-Step TTS Training Pipeline

Welcome to the complete TTS training implementation! This index helps you navigate the training system.

## 📖 Documentation (Start Here!)

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[QUICKSTART.md](QUICKSTART.md)** | Fast setup and common commands | 5 min |
| **[README.md](README.md)** | Complete technical documentation | 15 min |
| **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** | Detailed architecture and tips | 20 min |

### Choose Your Starting Point

**👤 New User?**
→ Start with **QUICKSTART.md**

**🔧 Developer/Technical?**
→ Start with **README.md**

**📚 Want Full Details?**
→ Start with **TRAINING_GUIDE.md**

---

## 🚀 Quick Commands

```bash
# Test your setup
python test_setup.py

# Quick test (5 epochs)
python run_training.py quick_test

# Balanced training (100 epochs) - RECOMMENDED
python run_training.py balanced

# Production training (200 epochs)
python run_training.py production

# Custom training
python train.py --num-epochs 100 --batch-size 32 --learning-rate 1e-3

# List configurations
python run_training.py --list

# Inspect checkpoints
python checkpoint_utils.py list-experiments
python checkpoint_utils.py inspect experiments/step_1/.../checkpoints/best.pt
```

---

## 📁 File Reference

### Core Training Files
| File | Purpose | Type |
|------|---------|------|
| `train.py` | Main training script | **Executable** |
| `losses.py` | L1 + Diversity loss | Module |
| `model_loader.py` | Model initialization | Module |
| `train_utils.py` | Training utilities | Module |

### Utilities
| File | Purpose | Type |
|------|---------|------|
| `text_processing.py` | Text tokenization | Module |
| `configs.py` | Pre-configured settings | **Executable** |
| `run_training.py` | Convenience runner | **Executable** |
| `checkpoint_utils.py` | Checkpoint inspection | **Executable** |
| `test_setup.py` | Setup verification | **Executable** |

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | Full documentation |
| `QUICKSTART.md` | Quick start guide |
| `TRAINING_GUIDE.md` | Complete reference |
| `INDEX.md` | This file |

---

## 🏗️ System Architecture

```
Text Input → [Text Encoder] → h_text
                                  ↓
Mel Input → [GST] → z_style → [Concatenate]
                                  ↓
                      [LSTM Decoder] ← TRAINABLE
                                  ↓
                            M_hat (Mel)
                                  ↓
                        [HiFi-GAN] (frozen)
                                  ↓
                           Audio Output
```

### Loss Functions
- **L1 Reconstruction**: `|M_predicted - M_target|_1`
- **Style Diversity**: Penalizes style embedding collapse
- **Total Loss**: `L_total = w_recon * L_recon + w_div * L_div`

---

## 📊 Output Structure

```
experiments/step_1/
└── attempt_YYYYMMDD_HHMMSS/
    ├── config.json              ← Saved hyperparameters
    ├── checkpoints/             ← Model snapshots
    │   ├── epoch_0001.pt
    │   ├── ...
    │   └── best.pt
    └── tensorboard/             ← TensorBoard logs
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Test (5 minutes)
```bash
1. python test_setup.py              # Verify setup
2. python run_training.py quick_test # Run training
3. tensorboard --logdir experiments/step_1/*/tensorboard
```

### Workflow 2: Production Training (Full Day)
```bash
1. python run_training.py balanced   # Start training
2. # Monitor with TensorBoard in another terminal
   tensorboard --logdir experiments/step_1/*/tensorboard
3. # Wait for training to complete
4. Check results in experiments/step_1/*/checkpoints/best.pt
```

### Workflow 3: Resume Training
```bash
1. python train.py --num-epochs 200 \
   --resume experiments/step_1/attempt_X/checkpoints/best.pt
```

### Workflow 4: Inspect Results
```bash
1. python checkpoint_utils.py list-experiments
2. python checkpoint_utils.py list my_experiment
3. python checkpoint_utils.py inspect experiments/.../best.pt
4. python checkpoint_utils.py metrics my_experiment
```

---

## 🔧 Configuration Presets

| Config | Epochs | Batch | Model | Time | Use Case |
|--------|--------|-------|-------|------|----------|
| `quick_test` | 5 | 8 | Small | 5min | Development |
| `balanced` | 100 | 32 | Medium | 2-4h | Normal training |
| `production` | 200 | 64 | Large | 8-12h | Best quality |
| `high_diversity` | 100 | 32 | Medium | 2-4h | Style variation |
| `lightweight` | 50 | 8 | Small | 1-2h | Limited GPU |

---

## 📈 Monitoring

### Real-time Console
- tqdm progress bars
- Loss values per epoch
- Best model indicators

### TensorBoard
```bash
tensorboard --logdir experiments/step_1/*/tensorboard
# Visit http://localhost:6006
```

Metrics tracked:
- `train/loss`, `train/recon_loss`, `train/div_loss`
- `val/loss`, `val/recon_loss`, `val/div_loss`

### Checkpoints
```bash
python checkpoint_utils.py metrics my_experiment
# Shows all loss values across epochs
```

---

## 💡 Tips

1. **Start small**: Use `quick_test` first
2. **Monitor GPU**: Keep `nvidia-smi` running
3. **Use AMP**: Add `--use-amp` for 30% speedup
4. **Check TensorBoard**: Monitor losses in real-time
5. **Save often**: Checkpoints auto-saved every epoch
6. **Resume as needed**: Interrupted training resumes easily

---

## ⚙️ Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--batch-size`, enable `--use-amp` |
| Slow training | Increase `--num-workers`, enable `--use-amp` |
| Loss not decreasing | Check learning rate, data loading |
| Model loading error | Check internet, NeMo installation |

See TRAINING_GUIDE.md for detailed troubleshooting.

---

## 🎓 Learning Resources

- **Architecture Overview**: See TRAINING_GUIDE.md
- **Training Details**: See README.md
- **Quick Start**: See QUICKSTART.md
- **Code Examples**: Check comments in source files

---

## 📞 Quick Reference

```bash
# Verify setup works
python test_setup.py

# See available configurations
python run_training.py --list
python configs.py

# List your experiments
python checkpoint_utils.py list-experiments

# Start training
python run_training.py balanced

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir experiments/step_1/*/tensorboard
```

---

## 🗂️ Full File Listing

```
train_first_step/
├── 📄 INDEX.md                  ← You are here
├── 📄 QUICKSTART.md             ← Start here!
├── 📄 README.md                 ← Full documentation
├── 📄 TRAINING_GUIDE.md         ← Complete reference
│
├── 🚀 Executable Scripts
│   ├── train.py                 ← MAIN: Run this to train
│   ├── run_training.py          ← Convenience runner
│   ├── test_setup.py            ← Verify setup
│   ├── configs.py               ← Show configurations
│   └── checkpoint_utils.py      ← Inspect results
│
├── 🧠 Core Modules
│   ├── losses.py                ← Loss functions
│   ├── model_loader.py          ← Model initialization
│   ├── train_utils.py           ← Training utilities
│   ├── text_processing.py       ← Text tokenization
│   └── __init__.py              ← Package init
```

---

## 🎯 Next Steps

1. Read **QUICKSTART.md** (5 min read)
2. Run `python test_setup.py` (verify)
3. Run `python run_training.py quick_test` (test)
4. Run `python run_training.py balanced` (train)
5. Monitor with TensorBoard
6. Check results in `experiments/step_1/`

---

Good luck with your training! 🚀

For detailed documentation, see the respective markdown files.
