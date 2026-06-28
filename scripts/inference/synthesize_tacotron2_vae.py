#!/usr/bin/env python3
"""
Synthesize audio from a Tacotron2-VAE checkpoint with style transfer.

Loads a trained Tacotron2-VAE model, converts input text to speech conditioned on
a reference audio (which encodes the speaking style via the VAE-GST). Optionally
uses a pretrained WaveGlow vocoder to convert mel spectrograms to audio.

Usage:
    # English (lj_speech_v1), mel-only output:
    python scripts/inference/synthesize_tacotron2_vae.py \
        --experiment experiments/tacotron2-vae/lj_speech_v1 \
        --text "Hello, this is a test." \
        --reference-audio path/to/reference.wav \
        --output-dir exports/synth

    # Portuguese (pt_tacotron_v1), with WaveGlow audio:
    LD_LIBRARY_PATH=/opt/anaconda3/envs/ambiente_aluno/lib:$LD_LIBRARY_PATH \
    python scripts/inference/synthesize_tacotron2_vae.py \
        --experiment experiments/tacotron2-vae/pt_tacotron_v1 \
        --text "O título de página foi encontrado." \
        --reference-audio path/to/reference.wav \
        --waveglow local_weight_models/waveglow/nvidia_waveglowpyt_fp32_20190427 \
        --output-dir exports/synth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "loader_vae_tacotron"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "tacotron2_vae"))

from models.tacotron2_vae.hparams import Tacotron2VAEHparams, create_hparams
from models.tacotron2_vae.model import Tacotron2, load_tacotron2_vae_model
from models.tacotron2_vae.layers import TacotronSTFT
from text_processing import TextProcessor
from train_utils import load_checkpoint, find_latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize audio from Tacotron2-VAE with style transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", type=Path, required=True,
                        help="Experiment directory (contains hparams.json, symbols.json, checkpoints/)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Specific checkpoint file (default: latest in experiment/checkpoints/)")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text to synthesize")
    parser.add_argument("--reference-audio", type=Path, default=None,
                        help="Reference audio for style transfer (VAE style input). "
                             "If omitted, uses zero latent vector (neutral style).")
    parser.add_argument("--waveglow", type=Path, default=None,
                        help="Path to pretrained WaveGlow checkpoint. "
                             "If omitted, only mel spectrogram is saved.")
    parser.add_argument("--output-dir", type=Path, default=Path("exports/synth"))
    parser.add_argument("--sigma", type=float, default=0.6,
                        help="WaveGlow inference sigma (smaller = less noise, more stable)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    return parser.parse_args()


def load_experiment(experiment_dir: Path) -> Tuple[Tacotron2VAEHparams, TextProcessor]:
    hparams_path = experiment_dir / "hparams.json"
    symbols_path = experiment_dir / "symbols.json"

    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.json not found in {experiment_dir}")
    if not symbols_path.exists():
        raise FileNotFoundError(f"symbols.json not found in {experiment_dir}")

    with open(hparams_path) as f:
        hparams_dict = json.load(f)
    hparams = create_hparams(hparams_dict)

    with open(symbols_path) as f:
        symbols_data = json.load(f)
    symbols: List[str] = symbols_data.get("symbols", symbols_data)
    # symbols.json records the actual cleaner used; hparams.json may have defaults
    cleaner_names: List[str] = symbols_data.get("cleaner_names", hparams_dict.get("text_cleaners", ["english_cleaners"]))

    text_processor = TextProcessor(symbols=symbols, cleaner_names=cleaner_names)
    return hparams, text_processor


def load_reference_mel(audio_path: Path, device: torch.device) -> Tensor:
    stft = TacotronSTFT(
        filter_length=1024, hop_length=256, win_length=1024,
        sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0
    ).to(device)

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 22050:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=22050)
    waveform = torch.clamp(waveform / waveform.abs().max(), -1.0, 1.0)

    mel = stft.mel_spectrogram(waveform.to(device))  # (1, n_mels, T)
    return mel


def save_mel_plot(mel: Tensor, path: Path, title: str = "Mel Spectrogram") -> None:
    data = mel.squeeze(0).cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_alignment_plot(alignment: Tensor, path: Path) -> None:
    data = alignment.squeeze(0).cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data.T, origin="lower", aspect="auto", interpolation="nearest", cmap="hot")
    ax.set_title("Attention Alignment")
    ax.set_xlabel("Decoder step")
    ax.set_ylabel("Encoder step")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_waveglow(waveglow_path: Path, device: torch.device):
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "waveglow"))
    from glow import WaveGlow  # type: ignore
    checkpoint = torch.load(str(waveglow_path), map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        waveglow = checkpoint["model"]
    else:
        config = checkpoint.get("config", {})
        state_dict = checkpoint["state_dict"]
        # Strip DataParallel 'module.' prefix if present
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        waveglow = WaveGlow(**config)
        waveglow.load_state_dict(state_dict)

    if hasattr(waveglow, "remove_weightnorm"):
        waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(device).eval()
    return waveglow


def griffin_lim_vocoder(mel: Tensor, device: torch.device, n_iter: int = 60) -> Tensor:
    """Convert log-mel spectrogram to waveform via Griffin-Lim."""
    SR, N_FFT, HOP, WIN = 22050, 1024, 256, 1024
    N_MELS, FMIN, FMAX = 80, 0.0, 8000.0

    # Un-log the mel
    linear_mel = torch.exp(mel.squeeze(0).float().cpu())  # (n_mels, T)

    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=SR,
        f_min=FMIN,
        f_max=FMAX,
        norm=None,
    )
    gl = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        n_iter=n_iter,
        power=1.0,
    )
    spec = inv_mel(linear_mel)   # (n_fft//2+1, T)
    waveform = gl(spec)          # (T_samples,)
    return waveform.unsqueeze(0) # (1, T_samples)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load experiment config
    print(f"Loading experiment from {args.experiment}...")
    hparams, text_processor = load_experiment(args.experiment)

    # Resolve checkpoint
    checkpoint_dir = args.experiment / "checkpoints"
    checkpoint_path = args.checkpoint or find_latest_checkpoint(checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    # Build model
    model = load_tacotron2_vae_model(hparams, device=device)
    model, _, _, iteration = load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"Model loaded at iteration {iteration}")

    # Process text
    text_ids = text_processor.text_to_sequence(args.text)
    text_tensor = torch.LongTensor(text_ids).unsqueeze(0).to(device)  # (1, T)
    print(f"Text: '{args.text}' → {len(text_ids)} tokens")

    # Load reference audio (or use zeros for neutral style)
    if args.reference_audio is not None:
        print(f"Loading reference audio: {args.reference_audio}")
        ref_mel = load_reference_mel(args.reference_audio, device)
        style_label = args.reference_audio.stem
    else:
        print("No reference audio — using zero latent (neutral style)")
        ref_mel = torch.zeros(1, hparams.n_mel_channels, 100).to(device)
        style_label = "neutral"

    # Run inference
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.experiment.name}_{style_label}"

    print("Running inference...")
    mel_pre, mel_post, alignments = model.inference_mel(text_tensor, ref_mel)

    # Save mel spectrogram plots
    mel_plot_path = args.output_dir / f"{stem}_mel.png"
    align_plot_path = args.output_dir / f"{stem}_alignment.png"
    save_mel_plot(mel_post, mel_plot_path, f"Synthesized Mel: '{args.text[:50]}'")
    save_alignment_plot(alignments, align_plot_path)
    print(f"Saved mel plot: {mel_plot_path}")
    print(f"Saved alignment plot: {align_plot_path}")

    # Save mel tensor
    mel_tensor_path = args.output_dir / f"{stem}_mel.pt"
    torch.save(mel_post.cpu(), mel_tensor_path)
    print(f"Saved mel tensor: {mel_tensor_path}")

    # WaveGlow audio synthesis
    if args.waveglow is not None:
        print(f"Loading WaveGlow from {args.waveglow}...")
        waveglow = load_waveglow(args.waveglow, device)
        with torch.no_grad():
            audio = waveglow.infer(mel_post, sigma=args.sigma)
        audio_path = args.output_dir / f"{stem}_audio.wav"
        torchaudio.save(str(audio_path), audio.cpu(), 22050)
        print(f"Saved audio: {audio_path}")
    else:
        print("No WaveGlow provided — falling back to Griffin-Lim vocoder (lower quality).")
        audio = griffin_lim_vocoder(mel_post, device)
        audio_path = args.output_dir / f"{stem}_audio.wav"
        torchaudio.save(str(audio_path), audio.cpu(), 22050)
        print(f"Saved audio (Griffin-Lim): {audio_path}")

    summary = {
        "experiment": str(args.experiment),
        "checkpoint": str(checkpoint_path),
        "iteration": iteration,
        "text": args.text,
        "n_tokens": len(text_ids),
        "reference_audio": str(args.reference_audio) if args.reference_audio else None,
        "mel_shape": list(mel_post.shape),
        "mel_plot": str(mel_plot_path),
        "alignment_plot": str(align_plot_path),
    }
    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
