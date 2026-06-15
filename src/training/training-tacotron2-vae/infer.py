"""
Inference script for Tacotron 2 VAE.

Responsibilities:
    - Load a trained Tacotron2-VAE checkpoint and symbol set.
    - Load a reference mel-spectrogram for style transfer.
    - Synthesize mel-spectrograms from input text using the reference style.
    - Convert predicted mel-spectrograms to audio using a HiFi-GAN vocoder.
    - Save the final synthesized waveform as a WAV file.

Main Functions:
    - synthesize: Perform style-conditioned text-to-mel synthesis.
    - load_reference_mel: Utility to load and format reference style tensors.
    - main: Primary entry point for inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Any

import torch
from torch import Tensor
import torchaudio

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))

try:
    from models.tacotron2_vae.hparams import Tacotron2VAEHparams
    from models.tacotron2_vae.model import load_tacotron2_vae_model, Tacotron2
    from text_processing import TextProcessor
except ImportError:
    # Handle absolute paths
    from src.models.tacotron2_vae.hparams import Tacotron2VAEHparams
    from src.models.tacotron2_vae.model import load_tacotron2_vae_model, Tacotron2
    from src.training.training_tacotron2_vae.text_processing import TextProcessor


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for inference.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--reference-mel", type=Path, required=True, help="Path to .pt sample with mel")
    parser.add_argument("--output-wav", type=Path, default=Path("output_tacotron2_vae.wav"))
    parser.add_argument("--symbols", type=Path, default=None)
    parser.add_argument("--max-decoder-steps", type=int, default=1000)
    parser.add_argument("--waveglow-checkpoint", type=Path, default=Path("local_weight_models/waveglow/nvidia_waveglowpyt_fp32_20190427"))
    return parser.parse_args()


def load_reference_mel(path: Path) -> Tensor:
    """
    Load a reference mel-spectrogram from a .pt file.

    Args:
        path (Path): Path to the saved tensor.

    Returns:
        Tensor: Formatted reference mel. Shape (1, n_mels, T).
    """
    sample: Dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    mel: Tensor = sample["mel"] # (B, n_mels, T) or (n_mels, T)
    
    # Standardize to (1, n_mels, T)
    if mel.dim() == 4:
        mel = mel.squeeze(0)
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    return mel.unsqueeze(0)


@torch.inference_mode()
def synthesize(
    model: Tacotron2,
    text_processor: TextProcessor,
    text: str,
    reference_mel: Tensor,
    waveglow: torch.nn.Module,
    device: torch.device,
    max_decoder_steps: int,
) -> Tensor:
    """
    Generate audio from text conditioned on reference style using WaveGlow.

    Args:
        model (Tacotron2): Loaded model.
        text_processor (TextProcessor): Text to ID converter.
        text (str): Input text.
        reference_mel (Tensor): Style reference.
        waveglow (torch.nn.Module): WaveGlow vocoder.
        device (torch.device): Device.
        max_decoder_steps (int): Safety limit for autoregressive decoding.

    Returns:
        Tensor: Generated audio.
    """
    sequence_list: List[int] = text_processor.text_to_sequence(text)
    sequence: Tensor = torch.LongTensor(sequence_list).unsqueeze(0).to(device) # (1, T_text)
    reference_mel = reference_mel.to(device) # (1, n_mels, T_ref)

    model.decoder.max_decoder_steps = max_decoder_steps
    audio = model.infer(sequence, reference_mel, waveglow)
    
    return audio


def main() -> None:
    """
    Main inference workflow.
    """
    args: argparse.Namespace = parse_args()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model state
    checkpoint: Dict[str, Any] = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hparams: Tacotron2VAEHparams = Tacotron2VAEHparams.from_dict(checkpoint["hparams"])

    # Load text processor
    symbols_path: Path = args.symbols or args.checkpoint.parent.parent / "symbols.json"
    text_processor: TextProcessor = TextProcessor.load(symbols_path)

    # Initialize model
    model: Tacotron2 = load_tacotron2_vae_model(hparams, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load vocoder
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "waveglow"))
    try:
        from glow import WaveGlow
        waveglow_ckpt = torch.load(args.waveglow_checkpoint, map_location=device, weights_only=False)
        waveglow = waveglow_ckpt['model'] if 'model' in waveglow_ckpt else waveglow_ckpt
        if hasattr(WaveGlow, 'remove_weightnorm'):
            waveglow = WaveGlow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(device).eval()
    except Exception as e:
        print(f"Error loading WaveGlow vocoder from {args.waveglow_checkpoint}: {e}")
        return

    # Inference
    reference_mel: Tensor = load_reference_mel(args.reference_mel)
    audio = synthesize(
        model,
        text_processor,
        args.text,
        reference_mel,
        waveglow,
        device,
        args.max_decoder_steps,
    )

    # Save output
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    torchaudio.save(str(args.output_wav), audio.cpu(), sample_rate=22050)
    print(f"Saved synthesized audio to {args.output_wav}")


if __name__ == "__main__":
    main()
