#!/usr/bin/env python3
"""
Copy-synthesis script to test WaveGlow reconstruction quality using ground truth audio.

Usage:
    python copy_synthesis.py --waveglow-checkpoint experiments/waveglow/checkpoints/epoch_20000.pt
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torchaudio
from torch import Tensor

# Add project root to path
PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "waveglow"))

try:
    from models.tacotron2_vae.layers import TacotronSTFT
    from glow import WaveGlow
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct audio from ground-truth mel spectrogram using WaveGlow.")
    parser.add_argument(
        "--input-wav",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "tts-portuguese-Corpora" / "TTS-Portuguese-Corpus" / "wavs" / "sample-0.wav",
        help="Path to ground truth WAV file"
    )
    parser.add_argument(
        "--waveglow-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "local_weight_models" / "waveglow" / "nvidia_waveglowpyt_fp32_20190427",
        help="Path to WaveGlow checkpoint file"
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=PROJECT_ROOT / "copy_synthesis_output.wav",
        help="Path to output WAV file"
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check input file
    if not args.input_wav.exists():
        print(f"Error: Input WAV file not found: {args.input_wav}")
        sys.exit(1)

    if not args.waveglow_checkpoint.exists():
        print(f"Error: WaveGlow checkpoint not found: {args.waveglow_checkpoint}")
        sys.exit(1)

    # 1. Load ground truth audio
    print(f"Loading ground truth audio: {args.input_wav}")
    audio, sr = torchaudio.load(str(args.input_wav))
    if sr != 22050:
        print(f"Warning: Audio sample rate is {sr}Hz, resampling to 22050Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        audio = resampler(audio)
        sr = 22050

    # Ensure mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # 2. Setup TacotronSTFT with aligned parameters (1024/256/1024)
    print("Setting up STFT engine (1024 filter length, 256 hop length, 1024 window)...")
    stft = TacotronSTFT(
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=sr,
        mel_fmin=0.0,
        mel_fmax=8000.0
    ).to(device)

    # Compute mel spectrogram
    audio_dev = audio.to(device)
    mel = stft.mel_spectrogram(audio_dev)  # Shape: (1, n_mels, T)
    print(f"Computed mel-spectrogram shape: {mel.shape}")

    # 3. Load WaveGlow vocoder
    print(f"Loading WaveGlow checkpoint: {args.waveglow_checkpoint}")
    waveglow_ckpt = torch.load(args.waveglow_checkpoint, map_location=device, weights_only=False)
    
    # Check if checkpoint is a serialized WaveGlow model or state dict
    if isinstance(waveglow_ckpt, dict) and 'model' in waveglow_ckpt and not isinstance(waveglow_ckpt['model'], dict):
        waveglow = waveglow_ckpt['model']
    else:
        # It's a state dict checkpoint. Instantiate a new WaveGlow model
        print("Instantiating new WaveGlow model...")
        
        # Extract state dict first to inspect shapes
        if isinstance(waveglow_ckpt, dict):
            if 'state_dict' in waveglow_ckpt:
                state_dict = waveglow_ckpt['state_dict']
            elif 'model' in waveglow_ckpt and isinstance(waveglow_ckpt['model'], dict):
                state_dict = waveglow_ckpt['model']
            else:
                state_dict = waveglow_ckpt
        else:
            state_dict = waveglow_ckpt
            
        # Clean state dict keys (remove module. prefix if present)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # Dynamically determine n_channels from state dict
        # WN.0.in_layers.0.bias shape is [2 * n_channels]
        bias_key = next((k for k in new_state_dict.keys() if 'WN.0.in_layers.0.bias' in k), None)
        if bias_key is not None:
            n_channels = new_state_dict[bias_key].shape[0] // 2
            print(f"Dynamically detected n_channels from checkpoint: {n_channels}")
        else:
            n_channels = 256
            print(f"Could not detect n_channels, falling back to: {n_channels}")
            
        waveglow_config = {
            "n_mel_channels": 80,
            "n_flows": 12,
            "n_group": 8,
            "n_early_every": 4,
            "n_early_size": 2,
            "WN_config": {
                "n_layers": 8,
                "n_channels": n_channels,
                "kernel_size": 3
            }
        }
        waveglow = WaveGlow(**waveglow_config)
        waveglow.load_state_dict(new_state_dict)
    
    if hasattr(WaveGlow, 'remove_weightnorm'):
        waveglow = WaveGlow.remove_weightnorm(waveglow)
    
    waveglow = waveglow.to(device).eval()

    # 4. Generate audio using WaveGlow
    print("Synthesizing waveform using WaveGlow vocoder...")
    # WaveGlow inference expects log-mel scaled input (which TacotronSTFT outputs)
    reconstructed_audio = waveglow.infer(mel, sigma=0.6)  # Shape: (1, S) or (S,)
    print(f"Synthesized audio shape: {reconstructed_audio.shape}")

    # Save reconstructed audio
    reconstructed_audio = reconstructed_audio.cpu()
    if reconstructed_audio.dim() == 2:
        reconstructed_audio = reconstructed_audio.squeeze(0)
    if reconstructed_audio.dim() == 1:
        reconstructed_audio = reconstructed_audio.unsqueeze(0)

    torchaudio.save(str(args.output_wav), reconstructed_audio, sample_rate=sr)
    print(f"Success! Saved reconstructed copy-synthesis output to: {args.output_wav}")


if __name__ == "__main__":
    main()
