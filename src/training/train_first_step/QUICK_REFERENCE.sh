#!/usr/bin/env bash
# Quick Reference Card - Save this for easy access!

# 🚀 QUICK START COMMANDS
# ========================

# VERIFY YOUR SETUP (always start here!)
python src/training/train_first_step/test_setup.py

# QUICK TEST (5 minutes - for testing)
python src/training/train_first_step/run_training.py quick_test

# BALANCED TRAINING (2-4 hours - RECOMMENDED)
python src/training/train_first_step/run_training.py balanced

# PRODUCTION TRAINING (8-12 hours - best quality)
python src/training/train_first_step/run_training.py production

# HIGH DIVERSITY TRAINING (2-4 hours - focus on style)
python src/training/train_first_step/run_training.py high_diversity

# LIGHTWEIGHT TRAINING (1-2 hours - limited GPU)
python src/training/train_first_step/run_training.py lightweight

# CUSTOM TRAINING (with your own hyperparameters)
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

# RESUME FROM CHECKPOINT
python src/training/train_first_step/train.py \
    --num-epochs 200 \
    --resume experiments/step_1/attempt_XXXXX/checkpoints/best.pt

# MONITOR WITH TENSORBOARD (run in another terminal)
tensorboard --logdir experiments/step_1/*/tensorboard
# Then visit: http://localhost:6006

# INSPECT YOUR EXPERIMENTS
python src/training/train_first_step/checkpoint_utils.py list-experiments

# LIST CHECKPOINTS IN AN EXPERIMENT
python src/training/train_first_step/checkpoint_utils.py list my_experiment_name

# INSPECT A SPECIFIC CHECKPOINT
python src/training/train_first_step/checkpoint_utils.py inspect \
    experiments/step_1/attempt_XXXXX/checkpoints/best.pt

# COMPARE CHECKPOINTS
python src/training/train_first_step/checkpoint_utils.py compare \
    experiments/step_1/attempt_1/checkpoints/best.pt \
    experiments/step_1/attempt_2/checkpoints/best.pt

# VIEW CONFIGURATIONS
python src/training/train_first_step/configs.py

# LIST ALL AVAILABLE COMMANDS
python src/training/train_first_step/run_training.py --help

# 🛠️ BASH HELPER (easier syntax if you prefer)
# ====================================

# List all commands
./tts_training.sh help

# Verify setup
./tts_training.sh test

# Quick test
./tts_training.sh quick-test

# Balanced training
./tts_training.sh balanced

# Production training
./tts_training.sh production

# View TensorBoard
./tts_training.sh tensorboard

# List experiments
./tts_training.sh list-experiments

# 📚 DOCUMENTATION READING ORDER
# ==============================

# 1. This file (quick reference)
# 2. QUICKSTART.md (5 minutes)
# 3. README.md (15 minutes) 
# 4. TRAINING_GUIDE.md (20 minutes)
# 5. CHECKLIST.md (before training)

# 🎯 TYPICAL WORKFLOW
# ===================

# Step 1: Verify
python src/training/train_first_step/test_setup.py

# Step 2: Train
python src/training/train_first_step/run_training.py balanced

# Step 3: Monitor (in another terminal)
tensorboard --logdir experiments/step_1/*/tensorboard

# Step 4: Inspect results
python src/training/train_first_step/checkpoint_utils.py list-experiments

# ⚙️ COMMON CUSTOMIZATIONS
# ========================

# Train with Mixed Precision (30% faster)
python src/training/train_first_step/train.py --use-amp

# Train with more workers (for faster data loading)
python src/training/train_first_step/train.py --num-workers 8

# Train with higher learning rate
python src/training/train_first_step/train.py --learning-rate 5e-4

# Train with more emphasis on diversity
python src/training/train_first_step/train.py --weight-diversity 1.0

# Train with larger model
python src/training/train_first_step/train.py \
    --acoustic-decoder-hidden-size 512 \
    --acoustic-decoder-num-layers 4 \
    --style-embedding-dim 256

# Train with custom experiment name
python src/training/train_first_step/train.py --experiment-name my_first_tts

# 💾 CHECKPOINT MANAGEMENT
# ========================

# Find your latest experiment
ls -lt experiments/step_1/ | head -1

# Load best model
checkpoint_path="experiments/step_1/attempt_XXXXX/checkpoints/best.pt"
python -c "import torch; cp = torch.load('$checkpoint_path'); print(cp.keys())"

# Extract only model weights
model_state = torch.load(checkpoint_path)['model_state_dict']

# Save metrics from checkpoint
metrics = torch.load(checkpoint_path)['metrics']

# 📊 MONITORING & ANALYSIS
# =========================

# Watch GPU usage live
watch -n 1 nvidia-smi

# Check current GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# See all experiments with timestamps
find experiments/step_1 -type d -name "attempt_*" | sort

# Extract all validation losses
python src/training/train_first_step/checkpoint_utils.py metrics my_experiment --metric val_loss

# 🔧 TROUBLESHOOTING COMMANDS
# ============================

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check NeMo installation
python -c "from nemo.collections.tts.models import FastPitchModel; print('NeMo OK')"

# Test data loading
python -c "from src.data.first_step_data_loaders.datasets import LibriSpeechPTDataset; d = LibriSpeechPTDataset(); print(f'Dataset size: {len(d)}')"

# 📁 USEFUL PATHS
# ===============

# Training directory
/home/richard/project/ml2_final_project/src/training/train_first_step/

# Experiments directory
/home/richard/project/ml2_final_project/experiments/step_1/

# Data directory
/home/richard/project/ml2_final_project/data/processed/

# Model source directory
/home/richard/project/ml2_final_project/src/models/

# 🆘 QUICK HELP
# ==============

# If you're stuck:
# 1. Check QUICKSTART.md
# 2. Run test_setup.py
# 3. Read CHECKLIST.md
# 4. See TRAINING_GUIDE.md troubleshooting
# 5. Check code comments

# 🎓 RECOMMENDED READING
# ======================

# cat src/training/train_first_step/QUICKSTART.md     # Quick start (5 min)
# cat src/training/train_first_step/README.md         # Full docs (15 min)
# cat src/training/train_first_step/TRAINING_GUIDE.md # Detailed (20 min)

# 🚀 TL;DR - JUST RUN THIS:
# ==========================
python src/training/train_first_step/test_setup.py && \
python src/training/train_first_step/run_training.py balanced

# That's it! Your model will be training in experiments/step_1/

# Save this file for quick reference! 📌
