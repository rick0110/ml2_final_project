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
#   RUNS="n8_k8_s1 n8_k128_s1"   STEPS="2000 4000"   STRENGTHS="1.0 0.6"   (n8_k8_s1-swap: see below)
set -ex
RUNS=${RUNS:?set RUNS, e.g. "n8_k8_s1 n8_k128_s1"}
STEPS=${STEPS:-"2000 4000"}
STRENGTHS=${STRENGTHS:-"1.0 0.6"}
VC_REFS=${VC_REFS:-""}  # e.g. "1 8": also convert every evaluation to the reference voice (vc_cascade.py)
MULTI_REFS=${MULTI_REFS:-""}  # e.g. "4 16": also evaluate with the latent code averaged over that many utterances of the reference speaker
BATCH=${BATCH:-48}
WORKERS=${WORKERS:-6}   # data-loading processes per training run (lower it when several runs share the machine)
CPUS=$(nproc)
POOL="--pool data/processed/LibriTTS --pool data/processed/LibriTTS_test"  # more utterances of the reference speakers
cd /content/repo

# Several copies of this job may start together: only one installs and prepares the data, the others wait
exec 9>/content/setup.lock
flock 9
python -m pip install -q gruut num2words inflect faster-whisper webrtcvad-wheels speechbrain
python -m pip install -q --no-deps resemblyzer
# librosa's pitch tracker crashes with the numba that ships on some Colab images
python -m pip install -q -U numba llvmlite
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
# Speakers that neither the models trained here nor the voice-conversion model have heard
if [ ! -f data/processed/LibriTTS_test/mels_metadata.csv ]; then
  [ -d data/raw/LibriTTS/test-clean ] || curl -sL https://www.openslr.org/resources/60/test-clean.tar.gz | tar xz -C data/raw
  python scripts/preprocess/make_manifest.py --corpus libritts --input-dir data/raw/LibriTTS/test-clean \
    --out-dir data/processed/LibriTTS_test --min-duration 1.5 --max-duration 10 --num-workers "$CPUS"
fi
[ -f ckpt/hifigan_universal_v1.pt ] || curl -sL -o ckpt/hifigan_universal_v1.pt \
  https://huggingface.co/csdc-atl/hifigan-universal_v1/resolve/main/g_02500000
flock -u 9

for RUN in $RUNS; do
  # Variants, by suffix. n8_k8_s1-swap: style reference from another utterance; -contrast: that
  # plus the contrastive loss; -spk: pretrained speaker embedding as a second style input;
  # -spkswap: speaker embedding plus the swapped reference; -spkdrop: speaker embedding plus
  # style dropout (half of the items train without the VAE style vector); -aug: every training
  # utterance also in a second, speed-changed voice (scripts/coverage/augment_voices.py);
  # -vcaug: every default-voice utterance also converted to one of 200 other real voices
  # (scripts/coverage/vc_voices.py); -notrain: no training, the English model is evaluated as is
  BASE=${RUN%%-*}; SWAP=""
  case "$RUN" in
    *-swap) SWAP="--swap-reference" ;;
    *-contrast) SWAP="--contrastive-weight 0.5" ;;
    *-spk) SWAP="--speaker-embeddings" ;;
    *-spkswap) SWAP="--speaker-embeddings --swap-reference" ;;
    *-spkdrop) SWAP="--speaker-embeddings --style-dropout 0.5" ;;
  esac
  N=${BASE#n}; N=${N%%_*}; K=${BASE#*_k}; K=${K%%_*}; SEED=${BASE##*_s}
  DATA=data/processed/cov_n${N}_k${K}
  EXP_DIR=experiments/tacotron2-vae/cov_$RUN
  RUN_STEPS=$STEPS; RUN_STRENGTHS=$STRENGTHS; RUN_VC=$VC_REFS
  case "$RUN" in *-notrain) RUN_STEPS=0; RUN_STRENGTHS="0.0 1.0"; RUN_VC="1 4 16 64" ;; esac
  flock 9  # runs that share a data folder prepare it one at a time
  [ -f "$DATA/test_sentences.txt" ] || python scripts/coverage/coverage_manifest.py --lj data/processed/LJSpeech \
    --libritts data/processed/LibriTTS --external data/processed/LibriTTS_test --n "$N" --k "$K" --out "$DATA"
  case "$RUN" in *-vcaug)
    [ -f "${DATA}_vc/test_sentences.txt" ] || python scripts/coverage/vc_voices.py --data-dir "$DATA" \
      --libritts data/processed/LibriTTS --audio-dir data/processed/vc_voices --out "${DATA}_vc"
    DATA=${DATA}_vc ;;
  esac
  case "$RUN" in *-aug)
    [ -f "${DATA}_aug/test_sentences.txt" ] || python scripts/coverage/augment_voices.py --data-dir "$DATA" --out "${DATA}_aug"
    DATA=${DATA}_aug ;;
  esac
  python scripts/preprocess/cache_mels.py --data-dir "$DATA" --text-cleaner english_cleaners --num-workers "$CPUS"
  case "$SWAP" in *speaker-embeddings*) python scripts/preprocess/cache_speaker_embeddings.py --data-dir "$DATA" ;; esac
  flock -u 9
  for STEP in $RUN_STEPS; do
    OUT=/content/out/$RUN/step_$STEP
    [ -f "$OUT/done" ] && continue
    echo "$(date -u +%H:%M:%S) training $RUN to step $STEP" >> /content/out/progress.txt
    COMMON="--artifacts-dir $DATA --text-cleaner english_cleaners --num-workers $WORKERS --learning-rate 1e-4 --warmup-steps 500
      --iters-per-checkpoint 500 --keep-last-checkpoints 1 --max-steps $STEP --lr-scheduler-patience 100000
      --batch-size $BATCH --val-split 0.03 --guided-attention-weight 2.0 --guided-attention-sigma 0.4 --seed $SEED $SWAP"
    CKPT=$EXP_DIR/checkpoints/epoch_$STEP
    if [ "$STEP" = 0 ]; then
      CKPT=ckpt/en_final
    elif ls "$EXP_DIR"/checkpoints/epoch_* >/dev/null 2>&1; then
      python src/training/training-tacotron2-vae/train.py $COMMON --resume-experiment "$EXP_DIR"
    else
      python src/training/training-tacotron2-vae/train.py $COMMON --checkpoint-path ckpt/en_final --experiment-name "cov_$RUN"
    fi
    echo "$(date -u +%H:%M:%S) evaluating $RUN at step $STEP" >> /content/out/progress.txt
    # With a speaker embedding, strength 0 shows what the embedding alone does
    case "$SWAP" in *speaker-embeddings*) RUN_STRENGTHS="$STRENGTHS 0.0" ;; esac
    for S in $RUN_STRENGTHS; do
      # An evaluation interrupted after the synthesis continues from the similarity. The voice
      # conversion needs the audio, which is deleted at the end, so a missing one starts over.
      MISSING=0
      for R in $RUN_VC; do head -n 1 "$OUT/strength_${S}_vc$R/results.csv" 2>/dev/null | grep -q sim_ref || MISSING=1; done
      if head -n 1 "$OUT/strength_$S/results.csv" 2>/dev/null | grep -q sim_ref; then
        [ "$MISSING" = 0 ] && continue
        ls "$OUT/strength_$S"/*__s5.wav >/dev/null 2>&1 || rm -f "$OUT/strength_$S/results.csv"
      fi
      [ -f "$OUT/strength_$S/results.csv" ] || python scripts/eval/synth_check.py --checkpoint "$CKPT" --hifigan ckpt/hifigan_universal_v1.pt \
        --data-dir "$DATA/refs" --all-references --sentences-file "$DATA/test_sentences.txt" \
        --text-cleaner english_cleaners --language en --style-strength "$S" --out-dir "$OUT/strength_$S"
      python scripts/coverage/speaker_similarity.py --eval-dir "$OUT/strength_$S" --refs "$DATA/refs" --train "$DATA" $POOL
      for R in $RUN_VC; do
        V=$OUT/strength_${S}_vc$R
        head -n 1 "$V/results.csv" 2>/dev/null | grep -q sim_ref && continue
        python scripts/coverage/vc_cascade.py --eval-dir "$OUT/strength_$S" --out-dir "$V" --refs "$DATA/refs" \
          $POOL --n-refs "$R"
        python scripts/coverage/speaker_similarity.py --eval-dir "$V" --refs "$DATA/refs" --train "$DATA" $POOL
        find "$V" -name '*.wav' ! -name '*__s0.wav' ! -name '*__s1.wav' ! -name '*__s2.wav' -delete
      done
      # Keep the audio of the first three sentences only (for listening); the numbers are in results.csv
      find "$OUT/strength_$S" -name '*.wav' ! -name '*__s0.wav' ! -name '*__s1.wav' ! -name '*__s2.wav' -delete
    done
    for R in $MULTI_REFS; do
      V=$OUT/strength_1.0_r$R
      head -n 1 "$V/results.csv" 2>/dev/null | grep -q sim_ref && continue
      python scripts/eval/synth_check.py --checkpoint "$CKPT" --hifigan ckpt/hifigan_universal_v1.pt \
        --data-dir "$DATA/refs" --all-references --sentences-file "$DATA/test_sentences.txt" \
        --text-cleaner english_cleaners --language en --style-strength 1.0 --n-refs "$R" $POOL --out-dir "$V"
      python scripts/coverage/speaker_similarity.py --eval-dir "$V" --refs "$DATA/refs" --train "$DATA" $POOL
      find "$V" -name '*.wav' ! -name '*__s0.wav' ! -name '*__s1.wav' ! -name '*__s2.wav' -delete
    done
    python scripts/eval/latent_dump.py --checkpoint "$CKPT" --manifest "$DATA" --manifest "$DATA/refs" \
      --max-per-manifest 1500 --out "$OUT/latents.npz"
    [ "$STEP" = 0 ] || cp "$EXP_DIR/metrics.jsonl" "$OUT/metrics.jsonl"
    cp "$DATA/mels_metadata.csv" "$OUT/train_manifest.csv"
    touch "$OUT/done"
    echo "$(date -u +%H:%M:%S) finished $RUN at step $STEP" >> /content/out/progress.txt
  done
done
echo "$(date -u +%H:%M:%S) ALL DONE" >> /content/out/progress.txt
