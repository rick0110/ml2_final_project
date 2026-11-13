#!/bin/bash
# Cloud job: the whole coverage experiment. For each run (N speakers x K sentences) it
# trains, synthesizes test sentences with seen and unseen reference voices at two style
# strengths, saves the latent codes and uploads everything to gs://$BUCKET/coverage/<run>.
# A run that already has results in the bucket is skipped, so a preempted VM continues.
#   RUNS="cov_n8_k8 cov_n8_k32 ..."   MAX_STEPS=800
set -ex
BUCKET="$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)"
RUNS=${RUNS:?set RUNS}
export MAX_STEPS=${MAX_STEPS:-800}
cd /work/repo
for RUN in $RUNS; do
  if gsutil -q stat "gs://$BUCKET/coverage/$RUN/done"; then continue; fi
  if ! gsutil -q stat "gs://$BUCKET/experiments/$RUN/checkpoints/final"; then
    STAGE=$RUN bash scripts/cloud/train_job.sh
  fi
  export STAGE=$RUN
  source scripts/cloud/stage_setup.sh
  python -m pip install -q faster-whisper
  [ -f ckpt/hifigan_universal_v1.pt ] || curl -sL -o ckpt/hifigan_universal_v1.pt \
    https://huggingface.co/csdc-atl/hifigan-universal_v1/resolve/main/g_02500000
  gsutil -q cp "gs://$BUCKET/experiments/$RUN/checkpoints/final" ckpt/cov_final
  OUT=exports/coverage/$RUN
  rm -rf "$OUT"
  for S in 1.0 0.6; do
    python scripts/eval/synth_check.py --checkpoint ckpt/cov_final --hifigan ckpt/hifigan_universal_v1.pt \
      --data-dir "$DATA/refs" --all-references --text-cleaner english_cleaners --language en \
      --style-strength $S --out-dir "$OUT/strength_$S"
  done
  python scripts/eval/latent_dump.py --checkpoint ckpt/cov_final --manifest "$DATA" --manifest "$DATA/refs" \
    --max-per-manifest 1500 --out "$OUT/latents.npz"
  cp "$DATA/mels_metadata.csv" "$OUT/train_manifest.csv"
  gsutil -q -m rsync -r "$OUT" "gs://$BUCKET/coverage/$RUN"
  echo ok | gsutil -q cp - "gs://$BUCKET/coverage/$RUN/done"
  echo "RUN FINISHED: $RUN"
done
echo "COVERAGE ALL DONE"
