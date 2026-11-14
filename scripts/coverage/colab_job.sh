#!/bin/bash
# Coverage experiment on a Colab GPU (the repository is expected at /content/repo and the
# English checkpoint at /content/repo/ckpt/en_final).
#
# For each run (N LibriTTS speakers x K sentences each, one seed) it trains from the English
# model and, at each step count of STEPS, synthesizes the test sentences with seen and unseen
# reference voices, measures word error rate and voice similarity, and saves the latent codes.
# Results go to /content/out/<run>/step_<n>; a finished evaluation is never repeated, so the
# job can be started again after an interruption.
#
# One training step costs the same on any GPU (the decoder runs frame by frame), so the way
# to go faster is to start several copies of this job on one GPU, each with its own RUNS.
#
#   RUNS="n8_k8_s1 n8_k128_s1"   STEPS="2000 4000"   STRENGTHS="1.0 0.6"
set -ex
RUNS=${RUNS:?set RUNS, e.g. "n8_k8_s1 n8_k128_s1"}
STEPS=${STEPS:-"2000 4000"}
STRENGTHS=${STRENGTHS:-"1.0 0.6"}
BATCH=${BATCH:-48}
WORKERS=${WORKERS:-6}   # data-loading processes per training run (lower it when several runs share the machine)
CPUS=$(nproc)
cd /content/repo

python -m pip install -q gruut num2words inflect faster-whisper webrtcvad-wheels
python -m pip install -q --no-deps resemblyzer
# The recognizer looks for the cuDNN that ships inside the PyTorch wheels
export LD_LIBRARY_PATH="$(python - <<'PY'
import glob, os, site
print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))
PY
):${LD_LIBRARY_PATH:-}"

mkdir -p data/raw data/processed ckpt /content/out
if [ ! -f data/processed/LJSpeech/mels_metadata.csv ]; then
  [ -d data/raw/LJSpeech-1.1 ] || curl -sL https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar xj -C data/raw
  python scripts/preprocess/make_manifest.py --corpus ljspeech --input-dir data/raw/LJSpeech-1.1 \
    --out-dir data/processed/LJSpeech --num-workers "$CPUS"
fi
if [ ! -f data/processed/LibriTTS/mels_metadata.csv ]; then
  [ -d data/raw/LibriTTS/train-clean-100 ] || curl -sL https://www.openslr.org/resources/60/train-clean-100.tar.gz | tar xz -C data/raw
  python scripts/preprocess/make_manifest.py --corpus libritts --input-dir data/raw/LibriTTS/train-clean-100 \
    --out-dir data/processed/LibriTTS --min-duration 1.5 --max-duration 10 --num-workers "$CPUS"
fi
[ -f ckpt/hifigan_universal_v1.pt ] || curl -sL -o ckpt/hifigan_universal_v1.pt \
  https://huggingface.co/csdc-atl/hifigan-universal_v1/resolve/main/g_02500000

for RUN in $RUNS; do
  N=${RUN#n}; N=${N%%_*}; K=${RUN#*_k}; K=${K%%_*}; SEED=${RUN##*_s}
  DATA=data/processed/cov_n${N}_k${K}
  EXP_DIR=experiments/tacotron2-vae/cov_$RUN
  [ -f "$DATA/test_sentences.txt" ] || python scripts/coverage/coverage_manifest.py --lj data/processed/LJSpeech \
    --libritts data/processed/LibriTTS --n "$N" --k "$K" --out "$DATA"
  python scripts/preprocess/cache_mels.py --data-dir "$DATA" --text-cleaner english_cleaners --num-workers "$CPUS"
  for STEP in $STEPS; do
    OUT=/content/out/$RUN/step_$STEP
    [ -f "$OUT/done" ] && continue
    echo "$(date -u +%H:%M:%S) training $RUN to step $STEP" >> /content/out/progress.txt
    COMMON="--artifacts-dir $DATA --text-cleaner english_cleaners --num-workers $WORKERS --learning-rate 1e-4 --warmup-steps 500
      --iters-per-checkpoint 500 --keep-last-checkpoints 1 --max-steps $STEP --lr-scheduler-patience 100000
      --batch-size $BATCH --val-split 0.03 --guided-attention-weight 2.0 --guided-attention-sigma 0.4 --seed $SEED"
    if ls "$EXP_DIR"/checkpoints/epoch_* >/dev/null 2>&1; then
      python src/training/training-tacotron2-vae/train.py $COMMON --resume-experiment "$EXP_DIR"
    else
      python src/training/training-tacotron2-vae/train.py $COMMON --checkpoint-path ckpt/en_final --experiment-name "cov_$RUN"
    fi
    CKPT=$EXP_DIR/checkpoints/epoch_$STEP
    echo "$(date -u +%H:%M:%S) evaluating $RUN at step $STEP" >> /content/out/progress.txt
    for S in $STRENGTHS; do
      # An evaluation interrupted after the synthesis continues from the similarity
      head -n 1 "$OUT/strength_$S/results.csv" 2>/dev/null | grep -q sim_ref && continue
      [ -f "$OUT/strength_$S/results.csv" ] || python scripts/eval/synth_check.py --checkpoint "$CKPT" --hifigan ckpt/hifigan_universal_v1.pt \
        --data-dir "$DATA/refs" --all-references --sentences-file "$DATA/test_sentences.txt" \
        --text-cleaner english_cleaners --language en --style-strength "$S" --out-dir "$OUT/strength_$S"
      python scripts/coverage/speaker_similarity.py --eval-dir "$OUT/strength_$S" --refs "$DATA/refs" --train "$DATA"
      # Keep the audio of the first three sentences only (for listening); the numbers are in results.csv
      find "$OUT/strength_$S" -name '*.wav' ! -name '*__s0.wav' ! -name '*__s1.wav' ! -name '*__s2.wav' -delete
    done
    python scripts/eval/latent_dump.py --checkpoint "$CKPT" --manifest "$DATA" --manifest "$DATA/refs" \
      --max-per-manifest 1500 --out "$OUT/latents.npz"
    cp "$EXP_DIR/metrics.jsonl" "$OUT/metrics.jsonl"
    cp "$DATA/mels_metadata.csv" "$OUT/train_manifest.csv"
    touch "$OUT/done"
    echo "$(date -u +%H:%M:%S) finished $RUN at step $STEP" >> /content/out/progress.txt
  done
done
echo "$(date -u +%H:%M:%S) ALL DONE" >> /content/out/progress.txt
