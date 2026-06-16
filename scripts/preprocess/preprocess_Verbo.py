#!/usr/bin/env python3
"""
Preprocess VERBO-Dataset into FastPitch-compatible mel-spectrogram tensors.
Aligned with the exact structure used in preprocess_mels_tts_portuguese.py
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FastPitch-compatible mel spectrograms for VERBO-Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Diretórios padronizados conforme a arquitetura do projeto
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/VERBO-Dataset/"),
        help="Root directory containing 'Audios' folder and metadata.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/VERBO-Dataset/"),
        help="Output directory for mel tensors and metadata",
    )

    parser.add_argument("--target-sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument("--log-zero-guard-value", type=float, default=1e-5)

    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=20.0) # Aumentado um pouco para frases longas do VERBO
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=5,
        help="Number of mel images to save for validation (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class FastPitchMelTransform(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        log_zero_guard_value: float = 1e-5,
    ) -> None:
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
            window_fn=torch.hann_window,
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
        )
        self.log_zero_guard_value = float(log_zero_guard_value)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.spec(waveform)
        mel = self.mel_scale(spec)
        log_mel = torch.log(mel + self.log_zero_guard_value)
        return log_mel


def find_examples(input_dir: Path) -> List[Path]:
    return sorted((input_dir / "Audios").rglob("*.wav"))


def create_text_lookup(input_dir: Path) -> Dict[str, str]:
    texts_file = input_dir / "metadata.csv" 
    if not texts_file.exists():
        raise FileNotFoundError(f"Arquivo de metadados não encontrado em: {texts_file}. Rode o generate_verbo_metadata.py primeiro.")

    lookup: Dict[str, str] = {}
    with texts_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            
            # Formato gerado pelo nosso script: nome_do_audio sem extensão | texto
            parts = line.split("|", maxsplit=1)
            if len(parts) == 2:
                audio_id = parts[0].strip()
                text = parts[1].strip()
                lookup[audio_id] = text

    return lookup


def text_from_path(wav_path: Path, lookup: Dict[str, str]) -> str:
    audio_id = wav_path.stem
    return lookup.get(audio_id, "")


def process_example(
    wav_path: Path,
    out_root: Path,
    mel_transform: FastPitchMelTransform,
    target_sr: int,
    text_lookup: Dict[str, str],
    min_duration: float,
    max_duration: float,
) -> Optional[Dict[str, Any]]:
    
    text = text_from_path(wav_path, text_lookup)
    if not text:
        return None

    waveform, sr = torchaudio.load(wav_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        sr = target_sr

    # --- HIGIENIZAÇÃO DO ÁUDIO ---
    # Normaliza o volume para nunca ultrapassar [-1.0, 1.0]
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    # Garante o clamp absoluto para satisfazer o assert do modelo
    waveform = torch.clamp(waveform, min=-1.0, max=1.0)
    # -----------------------------

    duration = waveform.shape[1] / sr
    if duration < min_duration or duration > max_duration:
        return None

    with torch.no_grad():
        log_mel = mel_transform(waveform)

    # Cria pastas para os mels e para os áudios limpos
    out_mels_dir = out_root / "mels"
    out_wavs_dir = out_root / "wavs_clean"
    out_mels_dir.mkdir(parents=True, exist_ok=True)
    out_wavs_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_mels_dir / f"{wav_path.stem}.pt"
    clean_wav_path = out_wavs_dir / f"{wav_path.stem}.wav"

    # Salva a cópia limpa do áudio que será lida pelo Dataloader no treino
    torchaudio.save(clean_wav_path, waveform, target_sr)

    payload = {
        "waveform": waveform.cpu(),
        "mel": log_mel.cpu(),
        "sr": sr,
        "duration": float(duration),
        "text": text,
        "source_wav": str(clean_wav_path.resolve()), # <--- APONTA PARA O ÁUDIO LIMPO!
    }
    torch.save(payload, out_path)

    return {
        "mel_path": str(out_path.resolve()),
        "duration": float(duration),
        "text": text,
        "source_wav": str(clean_wav_path.resolve()), # <--- APONTA PARA O ÁUDIO LIMPO!
    }

def compute_dataset_stats(manifest: List[Dict[str, Any]]) -> Tuple[float, float]:
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0

    for row in tqdm.tqdm(manifest, desc="Calculando estatísticas globais (Z-score)"):
        data = torch.load(row["mel_path"], map_location="cpu")
        mel = data["mel"].to(torch.float64)

        total_sum += mel.sum().item()
        total_sum_sq += (mel * mel).sum().item()
        total_count += mel.numel()

    mean = total_sum / total_count
    var = total_sum_sq / total_count - mean * mean
    std = var ** 0.5
    return mean, std


def add_normalized_mels(manifest: List[Dict[str, Any]], mean: float, std: float) -> None:
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)

    for row in tqdm.tqdm(manifest, desc="Injetando mels normalizados"):
        path = Path(row["mel_path"])
        data = torch.load(path, map_location="cpu")
        mel = data["mel"].to(torch.float32)
        mel_normalizado = (mel - mean_t) / std_t
        data["mel"] = mel
        data["mel_normalized"] = mel_normalizado
        torch.save(data, path)


def plot_mels(manifest: List[Dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 42) -> None:
    if not manifest or plt is None: return
    rnd = random.Random(seed)
    samples = rnd.sample(manifest, min(n, len(manifest)))
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(samples, start=1):
        data = torch.load(row["mel_path"], map_location="cpu")
        mel = data["mel"].squeeze(0).numpy()
        mel_norm = data["mel_normalized"].squeeze(0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        im0 = axes[0].imshow(mel, origin="lower", aspect="auto", interpolation="nearest")
        axes[0].set_title(f"{Path(row['mel_path']).stem} - mel")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(mel_norm, origin="lower", aspect="auto", interpolation="nearest")
        axes[1].set_title(f"{Path(row['mel_path']).stem} - mel_normalized")
        fig.colorbar(im1, ax=axes[1])

        out_path = out_dir / f"mel_{i:02d}_{Path(row['mel_path']).stem}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_dir = args.input_dir.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "mels").mkdir(parents=True, exist_ok=True)

    examples = find_examples(input_dir)
    text_lookup = create_text_lookup(input_dir)

    if not examples:
        raise FileNotFoundError(f"Nenhum arquivo .wav encontrado na pasta: {input_dir / 'Audios'}")

    mel_transform = FastPitchMelTransform(
        sample_rate=args.target_sr, n_mels=args.n_mels, n_fft=args.n_fft,
        win_length=args.win_length, hop_length=args.hop_length,
        f_min=args.f_min, f_max=args.f_max, log_zero_guard_value=args.log_zero_guard_value,
    )

    manifest: List[Dict[str, Any]] = []

    print(f"Iniciando pré-processamento de {len(examples)} arquivos do VERBO-Dataset...")
    for wav_path in tqdm.tqdm(examples, desc="Extraindo espectrogramas"):
        res = process_example(
            wav_path=wav_path, out_root=out_root, mel_transform=mel_transform,
            target_sr=args.target_sr, text_lookup=text_lookup,
            min_duration=args.min_duration, max_duration=args.max_duration,
        )
        if res is not None:
            manifest.append(res)

    if not manifest:
        raise RuntimeError("Nenhum áudio sobrou após o filtro de duração ou falta de textos!")

    mean, std = compute_dataset_stats(manifest)

    stats_path = out_root / "mel_stats.pt"
    torch.save({
        "mean": float(mean), "std": float(std), "var": float(std * std),
        "n_mels": args.n_mels, "sample_rate": args.target_sr,
        "n_fft": args.n_fft, "win_length": args.win_length, "hop_length": args.hop_length,
        "f_min": args.f_min, "f_max": args.f_max, "log_zero_guard_value": args.log_zero_guard_value,
    }, stats_path)

    add_normalized_mels(manifest, mean, std)

    manifest_csv = out_root / "mels_metadata.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mel_path", "duration", "text", "source_wav"])
        for row in manifest:
            writer.writerow([row["mel_path"], f"{row['duration']:.6f}", row.get("text", ""), row.get("source_wav", "")])

    if args.plot_samples > 0:
        plot_mels(manifest, out_root / "figures", n=args.plot_samples, seed=args.seed)

    print(f"\n✅ SUCESSO! {len(manifest)} tensores mel salvos e normalizados em: {out_root}")
    print(f"📄 Metadados salvos em: {manifest_csv}")

if __name__ == "__main__":
    main()