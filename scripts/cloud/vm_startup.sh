#!/bin/bash
# Startup script for the GCP training VM: a minimal job runner.
#
# The VM polls gs://$BUCKET/jobs/next.sh. When a job appears it is moved to the
# local disk and run; its log is copied to gs://$BUCKET/logs/ every minute.
# A job that was interrupted (spot preemption) is rerun on the next boot, so
# jobs must be safe to run twice. The VM powers itself off after
# IDLE_LIMIT_MIN minutes without a job, so a forgotten VM stops costing money.

BUCKET="$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)"
IDLE_LIMIT_MIN=30
WORK=/work
mkdir -p "$WORK" && cd "$WORK"
export HOME=/root

run_job() {
  local id="$1"
  bash "$WORK/current_job.sh" > "$WORK/$id.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    gsutil -q cp "$WORK/$id.log" "gs://$BUCKET/logs/$id.log"
    sleep 60
  done
  wait "$pid"
  echo "[runner] job exited with code $?" >> "$WORK/$id.log"
  gsutil -q cp "$WORK/$id.log" "gs://$BUCKET/logs/$id.log"
  rm -f "$WORK/current_job.sh" "$WORK/current_job.id"
}

# Resume a job interrupted by a preemption
if [ -f "$WORK/current_job.sh" ]; then
  run_job "$(cat "$WORK/current_job.id")"
fi

idle=0
while true; do
  if gsutil -q stat "gs://$BUCKET/jobs/next.sh"; then
    id="job_$(date +%Y%m%d_%H%M%S)"
    gsutil -q cp "gs://$BUCKET/jobs/next.sh" "$WORK/current_job.sh"
    gsutil -q rm "gs://$BUCKET/jobs/next.sh"
    echo "$id" > "$WORK/current_job.id"
    run_job "$id"
    idle=0
  else
    sleep 30
    idle=$((idle + 1))
    if [ "$idle" -ge $((IDLE_LIMIT_MIN * 2)) ]; then
      poweroff
    fi
  fi
done
