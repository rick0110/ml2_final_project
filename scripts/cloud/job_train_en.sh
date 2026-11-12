#!/bin/bash
# Cloud job: English Tacotron2-VAE, fine-tuned from NVIDIA's pretrained Tacotron 2 on LJSpeech.
# Safe to rerun: every step is skipped if already done and training resumes from the last checkpoint.
set -ex
BUCKET="$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)"
BRANCH=claude/film-dubbing-project-amuqaf
EXP=en_vae_v1
MAX_STEPS=${MAX_STEPS:-6000}

cd /work
nvidia-smi
# Find the Python that ships with PyTorch on the Deep Learning VM image and put it first on PATH
for cand in /opt/conda/bin /opt/python/bin /usr/local/bin /usr/bin; do
  for exe in python python3; do
    if [ -x "$cand/$exe" ] && "$cand/$exe" -c "import torch" 2>/dev/null; then
      mkdir -p /work/bin && ln -sf "$cand/$exe" /work/bin/python
      export PATH="/work/bin:$cand:$PATH"
      break 2
    fi
  done
done
python -c "import sys, torch; print(sys.executable, torch.__version__, 'cuda', torch.cuda.is_available())"
[ -d repo ] || git clone -b "$BRANCH" https://github.com/rick0110/ml2_final_project repo
cd repo
git fetch origin "$BRANCH" && git reset --hard "origin/$BRANCH"
python -m pip install -q gruut gruut-lang-pt num2words inflect librosa soundfile tensorboard matplotlib scikit-learn tqdm

command -v bzip2 >/dev/null || (apt-get update -qq && apt-get install -y -qq bzip2)
mkdir -p data/raw data/processed ckpt
if [ ! -d data/raw/LJSpeech-1.1 ]; then
  curl -sL https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar xj -C data/raw
fi
[ -f ckpt/tacotron2_nvidia.pt ] || curl -sL -o ckpt/tacotron2_nvidia.pt \
  "https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427"
if [ ! -f data/processed/LJSpeech/mels_metadata.csv ]; then
  python scripts/preprocess/preprocess_ljSpeech.py --input-dir data/raw/LJSpeech-1.1 \
    --out-dir data/processed/LJSpeech/mels --plot-samples 0
fi

EXP_DIR=experiments/tacotron2-vae/$EXP
# Copy light artifacts (metrics, figures, TensorBoard logs) to the bucket every 2 minutes
# and the newest checkpoint every 10 minutes.
(
  n=0
  while true; do
    sleep 120
    [ -d "$EXP_DIR" ] || continue
    gsutil -q -m rsync -r -x 'checkpoints/.*' "$EXP_DIR" "gs://$BUCKET/experiments/$EXP" || true
    n=$((n + 1))
    if [ $((n % 5)) -eq 0 ]; then
      latest=$(ls "$EXP_DIR/checkpoints" | sort -t_ -k2 -n | tail -1)
      [ -n "$latest" ] && gsutil -q cp "$EXP_DIR/checkpoints/$latest" "gs://$BUCKET/experiments/$EXP/checkpoints/latest" || true
    fi
  done
) &
SYNC_PID=$!

COMMON="--artifacts-dir data/processed/LJSpeech --text-cleaner english_cleaners --batch-size 48
  --num-workers 6 --val-split 0.03 --learning-rate 1e-4 --warmup-steps 500
  --iters-per-checkpoint 500 --keep-last-checkpoints 2 --max-steps $MAX_STEPS
  --guided-attention-weight 2.0 --guided-attention-sigma 0.4 --lr-scheduler-patience 1000"
if ls "$EXP_DIR"/checkpoints/epoch_* >/dev/null 2>&1; then
  python src/training/training-tacotron2-vae/train.py $COMMON --resume-experiment "$EXP_DIR"
else
  python src/training/training-tacotron2-vae/train.py $COMMON \
    --pretrained-tacotron2 ckpt/tacotron2_nvidia.pt --experiment-name "$EXP"
fi

kill $SYNC_PID || true
gsutil -q -m rsync -r -x 'checkpoints/.*' "$EXP_DIR" "gs://$BUCKET/experiments/$EXP"
latest=$(ls "$EXP_DIR/checkpoints" | sort -t_ -k2 -n | tail -1)
gsutil -q cp "$EXP_DIR/checkpoints/$latest" "gs://$BUCKET/experiments/$EXP/checkpoints/final"
echo "JOB DONE: $EXP at $latest"
