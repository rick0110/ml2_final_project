#!/bin/bash
# Cloud job: synthesize test sentences with the newest checkpoint of a stage,
# check them with speech recognition and upload audio + results to the bucket.
#   STAGE=en|pt|verbo   (same meaning as in train_job.sh)
set -ex
BUCKET="$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)"
STAGE=${STAGE:?set STAGE to en, pt or verbo}
cd /work/repo
export PATH="/work/bin:$PATH"
python -m pip install -q faster-whisper
mkdir -p ckpt
[ -f ckpt/hifigan_universal_v1.pt ] || curl -sL -o ckpt/hifigan_universal_v1.pt \
  https://huggingface.co/csdc-atl/hifigan-universal_v1/resolve/main/g_02500000

case "$STAGE" in
  en)    EXP=en_vae_v1;    DATA=data/processed/LJSpeech;       CLEANER=english_cleaners;             LANG_CODE=en ;;
  pt)    EXP=pt_vae_v1;    DATA=data/processed/TTS-Portuguese; CLEANER=portuguese_phonetic_cleaners; LANG_CODE=pt ;;
  verbo) EXP=verbo_vae_v1; DATA=data/processed/VERBO;          CLEANER=portuguese_phonetic_cleaners; LANG_CODE=pt ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
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
