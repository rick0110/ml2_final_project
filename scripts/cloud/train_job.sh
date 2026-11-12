#!/bin/bash
# Cloud training job for one stage of the pipeline. Run by the VM job runner
# (scripts/cloud/vm_startup.sh) through a small wrapper that sets STAGE.
#
#   STAGE=en     English: LJSpeech, warm-started from NVIDIA's pretrained Tacotron 2
#   STAGE=pt     Portuguese: TTS-Portuguese Corpus, warm-started from the English model
#   STAGE=verbo  Emotional Portuguese: VERBO, warm-started from the Portuguese model
#
# Everything needed to resume lives in the bucket, so a preempted spot VM (or a
# brand-new VM in another zone) picks up from the last uploaded checkpoint.
set -ex
BUCKET="$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)"
STAGE=${STAGE:?set STAGE to en, pt or verbo}
MAX_STEPS=${MAX_STEPS:-6000}
# A value in gs://$BUCKET/control/max_steps_$STAGE overrides MAX_STEPS, so a run can be
# shortened or extended without editing the queued job (takes effect on the next start).
override=$(gsutil -q cat "gs://$BUCKET/control/max_steps_$STAGE" 2>/dev/null || true)
[ -n "$override" ] && MAX_STEPS=$override
cd /work/repo

# The Deep Learning VM image ships PyTorch for one specific interpreter: find it
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
python -m pip install -q gruut gruut-lang-pt num2words inflect librosa soundfile tensorboard matplotlib scikit-learn tqdm
command -v bzip2 >/dev/null && command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq bzip2 unzip)
mkdir -p data/raw data/processed ckpt

case "$STAGE" in
  en)
    EXP=en_vae_v1; DATA=data/processed/LJSpeech; CLEANER=english_cleaners
    if [ ! -f "$DATA/mels_metadata.csv" ]; then
      [ -d data/raw/LJSpeech-1.1 ] || curl -sL https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar xj -C data/raw
      python scripts/preprocess/make_manifest.py --corpus ljspeech --input-dir data/raw/LJSpeech-1.1 --out-dir "$DATA" --num-workers 8
    fi
    [ -f ckpt/tacotron2_nvidia.pt ] || curl -sL -o ckpt/tacotron2_nvidia.pt \
      "https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427"
    INIT="--pretrained-tacotron2 ckpt/tacotron2_nvidia.pt"
    EXTRA="--batch-size 48 --val-split 0.03 --guided-attention-weight 2.0 --guided-attention-sigma 0.4"
    ;;
  pt)
    EXP=pt_vae_v1; DATA=data/processed/TTS-Portuguese; CLEANER=portuguese_phonetic_cleaners
    if [ ! -f "$DATA/mels_metadata.csv" ]; then
      if [ ! -d data/raw/tts-portuguese ]; then
        curl -sL -o /work/tts_pt.zip "https://www.dropbox.com/s/ohpc7epowv9ct7o/TTS-Portuguese-Corpus.zip?dl=1"
        unzip -q /work/tts_pt.zip -d data/raw/tts-portuguese && rm /work/tts_pt.zip
      fi
      ROOT=$(dirname "$(find data/raw/tts-portuguese -name texts.csv | head -1)")
      python scripts/preprocess/make_manifest.py --corpus tts_portuguese --input-dir "$ROOT" --out-dir "$DATA" --num-workers 8
    fi
    gsutil -q cp "gs://$BUCKET/experiments/en_vae_v1/checkpoints/final" ckpt/en_final
    INIT="--checkpoint-path ckpt/en_final"
    EXTRA="--batch-size 80 --val-split 0.04 --guided-attention-weight 5.0 --guided-attention-sigma 0.3"
    ;;
  verbo)
    EXP=verbo_vae_v1; DATA=data/processed/VERBO; CLEANER=portuguese_phonetic_cleaners
    if [ ! -f "$DATA/mels_metadata.csv" ]; then
      gsutil -q -m cp -r "gs://$BUCKET/datasets/VERBO" data/raw/
      python scripts/preprocess/make_manifest.py --corpus verbo --input-dir data/raw/VERBO --out-dir "$DATA" --num-workers 8 ${VERBO_SPEAKERS:+--speakers $VERBO_SPEAKERS}
    fi
    gsutil -q cp "gs://$BUCKET/experiments/pt_vae_v1/checkpoints/final" ckpt/pt_final
    INIT="--checkpoint-path ckpt/pt_final"
    EXTRA="--batch-size 16 --val-split 0.1 --guided-attention-weight 5.0 --guided-attention-sigma 0.3"
    ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac

# Compute every mel once, in parallel, so training never waits on the CPU
python scripts/preprocess/cache_mels.py --data-dir "$DATA" --text-cleaner "$CLEANER" --num-workers 8

EXP_DIR=experiments/tacotron2-vae/$EXP
REMOTE="gs://$BUCKET/experiments/$EXP"
# New VM with no local state: restore the experiment from the bucket
if ! ls "$EXP_DIR"/checkpoints/epoch_* >/dev/null 2>&1; then
  remote_ckpt=$(gsutil ls "$REMOTE/checkpoints/epoch_*" 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
  if [ -n "$remote_ckpt" ]; then
    mkdir -p "$EXP_DIR/checkpoints"
    gsutil -q -m rsync -r -x 'checkpoints/.*' "$REMOTE" "$EXP_DIR"
    gsutil -q cp "$remote_ckpt" "$EXP_DIR/checkpoints/"
  fi
fi

upload_state() {
  [ -d "$EXP_DIR" ] || return 0
  gsutil -q -m rsync -r -x 'checkpoints/.*' "$EXP_DIR" "$REMOTE" || true
  local latest
  latest=$(ls "$EXP_DIR/checkpoints" 2>/dev/null | sort -t_ -k2 -n | tail -1)
  if [ -n "$latest" ] && ! gsutil -q stat "$REMOTE/checkpoints/$latest"; then
    gsutil -q cp "$EXP_DIR/checkpoints/$latest" "$REMOTE/checkpoints/$latest" || true
    for old in $(gsutil ls "$REMOTE/checkpoints/epoch_*" | grep -v "/$latest$"); do gsutil -q rm "$old" || true; done
  fi
}
( while true; do sleep 120; upload_state; echo "[gpu] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
SYNC_PID=$!

COMMON="--artifacts-dir $DATA --text-cleaner $CLEANER --num-workers 6 --learning-rate 1e-4 --warmup-steps 500
  --iters-per-checkpoint 250 --keep-last-checkpoints 2 --max-steps $MAX_STEPS --lr-scheduler-patience 100000 $EXTRA"
if ls "$EXP_DIR"/checkpoints/epoch_* >/dev/null 2>&1; then
  python src/training/training-tacotron2-vae/train.py $COMMON --resume-experiment "$EXP_DIR"
else
  python src/training/training-tacotron2-vae/train.py $COMMON $INIT --experiment-name "$EXP"
fi

kill $SYNC_PID || true
upload_state
latest=$(ls "$EXP_DIR/checkpoints" | sort -t_ -k2 -n | tail -1)
gsutil -q cp "$EXP_DIR/checkpoints/$latest" "$REMOTE/checkpoints/final"
echo "JOB DONE: $EXP at $latest"
