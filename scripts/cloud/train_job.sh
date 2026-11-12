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
source scripts/cloud/stage_setup.sh

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
