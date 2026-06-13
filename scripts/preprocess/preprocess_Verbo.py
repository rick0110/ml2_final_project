from pathlib import Path
import argparse
import sys
import torch
import csv
import torchaudio
    

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "tacotron2_vae"))
from layers import TacotronSTFT

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess Verbo raw data into FastPitch-compatible mel-spectrogram tensors.")
    parser.add_argument("--input_root", type=Path, help="Root directory of the raw dataset", default="./data/raw/VERBO-Dataset")
    parser.add_argument("--out_root", type=Path, help="Root directory to save the preprocessed dataset", default="./data/preprocessed/VERBO-Dataset")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=8000.0)

    args = parser.parse_args()

    mel_processor = MelSpectrogramProcessor(
        sampling_rate=args.sample_rate,
        filter_length=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mel_channels=args.n_mels,
        mel_fmin=args.fmin,
        mel_fmax=args.fmax
    )

    return args, mel_processor

def find_audio_files(root: Path):
    audio_files = Path(root / "Audios").rglob("*.wav")
    return list(audio_files)

class MelSpectrogramProcessor:
    def __init__(self, sampling_rate=22050, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0.0, mel_fmax=8000.0):
        self.stft = TacotronSTFT(
            sampling_rate=sampling_rate,
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax
        )

    def __call__(self, audio):
        mel_spec = self.stft.mel_spectrogram(audio)
        return mel_spec

def process_audio_file(audio_path: Path, out_root: Path, mel_processor: MelSpectrogramProcessor):
    audio_path = audio_path.resolve()
    audio_id = str(audio_path)
    out_path = (out_root / "mels").resolve()

    audio, sr = torchaudio.load(audio_path)
    mel = mel_processor(audio)

    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(mel, out_path / f"{Path(audio_id).stem}.pt")
    







if __name__ == "__main__":
    args, mel_processor = get_args()
    audio = find_audio_files(Path("./data/raw/VERBO-Dataset"))
    process_audio_file(Path(audio[0]), Path("./data/preprocessed/VERBO-Dataset"), mel_processor)