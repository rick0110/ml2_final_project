#!/bin/bash
# Cloud job: synthesize test sentences with the newest checkpoint of a stage,
# check them with speech recognition and upload audio + results to the bucket.
#   STAGE=en|pt|verbo|mix   (same meaning as in train_job.sh)
set -ex
BUCKET="$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)"
STAGE=${STAGE:?set STAGE to en, pt, verbo or mix}
cd /work/repo
source scripts/cloud/stage_setup.sh
python -m pip install -q faster-whisper
[ -f ckpt/hifigan_universal_v1.pt ] || curl -sL -o ckpt/hifigan_universal_v1.pt \
  https://huggingface.co/csdc-atl/hifigan-universal_v1/resolve/main/g_02500000

EXP_DIR=experiments/tacotron2-vae/$EXP
if ! ls "$EXP_DIR"/checkpoints/epoch_* >/dev/null 2>&1; then  # fresh VM: take the final checkpoint from the bucket
  mkdir -p "$EXP_DIR/checkpoints"
  gsutil -q cp "gs://$BUCKET/experiments/$EXP/checkpoints/final" "$EXP_DIR/checkpoints/epoch_999999"
fi
CKPT="$EXP_DIR/checkpoints/$(ls "$EXP_DIR/checkpoints" | sort -t_ -k2 -n | tail -1)"
OUT=exports/eval/$EXP
rm -rf "$OUT"
python scripts/eval/synth_check.py --checkpoint "$CKPT" --hifigan ckpt/hifigan_universal_v1.pt \
  --data-dir "$DATA" --text-cleaner "$CLEANER" --language "$LANG_CODE" --out-dir "$OUT" ${EVAL_ARGS:-}
gsutil -q -m rsync -r "$OUT" "gs://$BUCKET/eval/$EXP"
echo "EVAL DONE: $EXP ($CKPT)"
