# Sourced by train_job.sh and eval_job.sh (needs STAGE and BUCKET): finds the PyTorch
# interpreter, installs the dependencies, prepares the corpus of the stage and sets
# EXP, DATA, CLEANER, LANG_CODE, INIT and EXTRA. Safe to run again on a prepared VM.
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
    EXP=en_vae_v1; DATA=data/processed/LJSpeech; CLEANER=english_cleaners; LANG_CODE=en
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
    EXP=pt_vae_v1; DATA=data/processed/TTS-Portuguese; CLEANER=portuguese_phonetic_cleaners; LANG_CODE=pt
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
    EXP=verbo_vae_v1; DATA=data/processed/VERBO; CLEANER=portuguese_phonetic_cleaners; LANG_CODE=pt
    if [ ! -f "$DATA/mels_metadata.csv" ]; then
      gsutil -q -m cp -r "gs://$BUCKET/datasets/VERBO" data/raw/
      python scripts/preprocess/make_manifest.py --corpus verbo --input-dir data/raw/VERBO --out-dir "$DATA" --num-workers 8 ${VERBO_SPEAKERS:+--speakers $VERBO_SPEAKERS}
    fi
    gsutil -q cp "gs://$BUCKET/experiments/pt_vae_v1/checkpoints/final" ckpt/pt_final
    INIT="--checkpoint-path ckpt/pt_final"
    EXTRA="--batch-size 16 --val-split 0.1 --guided-attention-weight 5.0 --guided-attention-sigma 0.3"
    ;;
  mix)
    # Portuguese corpus + VERBO in one manifest: VERBO alone has only 14 sentences, so
    # fine-tuning on it alone teaches emotion but forgets how to read new text.
    EXP=ptverbo_vae_v1; DATA=data/processed/PT-VERBO; CLEANER=portuguese_phonetic_cleaners; LANG_CODE=pt
    if [ ! -f "$DATA/mels_metadata.csv" ]; then
      (STAGE=pt; source scripts/cloud/stage_setup.sh)
      (STAGE=verbo; source scripts/cloud/stage_setup.sh)
      mkdir -p "$DATA"
      cat data/processed/TTS-Portuguese/mels_metadata.csv > "$DATA/mels_metadata.csv"
      tail -n +2 data/processed/VERBO/mels_metadata.csv >> "$DATA/mels_metadata.csv"
    fi
    gsutil -q cp "gs://$BUCKET/experiments/pt_vae_v1/checkpoints/final" ckpt/pt_final
    INIT="--checkpoint-path ckpt/pt_final"
    EXTRA="--batch-size 48 --val-split 0.04 --guided-attention-weight 5.0 --guided-attention-sigma 0.3"
    ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
