#!/usr/bin/env python3
"""Tacotron2-VAE inference script (style transfer from reference mel)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))

from models.HiFi_GAN import load_hifigan_model
from models.tacotron2_vae.hparams import Tacotron2VAEHparams
from models.tacotron2_vae.model import load_tacotron2_vae_model
from text_processing import TextProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--reference-mel", type=Path, required=True, help="Path to .pt sample with mel")
    parser.add_argument("--output-wav", type=Path, default=Path("output_tacotron2_vae.wav"))
    parser.add_argument("--symbols", type=Path, default=None)
    parser.add_argument("--max-decoder-steps", type=int, default=1000)
    return parser.parse_args()


def load_reference_mel(path: Path) -> torch.Tensor:
    sample = torch.load(path, map_location="cpu", weights_only=False)
    mel = sample["mel"]
    if mel.dim() == 4:
        mel = mel.squeeze(0)
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    return mel.unsqueeze(0)


@torch.inference_mode()
def synthesize(
    model,
    text_processor: TextProcessor,
    text: str,
    reference_mel: torch.Tensor,
    device: torch.device,
    max_decoder_steps: int,
):
    sequence = torch.LongTensor(text_processor.text_to_sequence(text)).unsqueeze(0).to(device)
    input_lengths = torch.LongTensor([sequence.size(1)]).to(device)
    reference_mel = reference_mel.to(device)

    transcript_embedded_inputs = model.transcript_embedding(sequence).transpose(1, 2)
    transcript_outputs = model.encoder.inference(transcript_embedded_inputs)

    latent_vector, _, _, _ = model.vae_gst(reference_mel)
    latent_vector = latent_vector.unsqueeze(1).expand_as(transcript_outputs)
    encoder_outputs = transcript_outputs + latent_vector

    model.decoder.max_decoder_steps = max_decoder_steps
    mel_outputs, gate_outputs, alignments = model.decoder.inference(encoder_outputs)
    mel_outputs_postnet = model.postnet(mel_outputs)
    mel_outputs_postnet = mel_outputs + mel_outputs_postnet
    return mel_outputs_postnet, alignments


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hparams = Tacotron2VAEHparams.from_dict(checkpoint["hparams"])

    symbols_path = args.symbols or args.checkpoint.parent.parent / "symbols.json"
    text_processor = TextProcessor.load(symbols_path)

    model = load_tacotron2_vae_model(hparams, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    reference_mel = load_reference_mel(args.reference_mel)
    mel, _ = synthesize(
        model,
        text_processor,
        args.text,
        reference_mel,
        device,
        args.max_decoder_steps,
    )

    _, vocoder = load_hifigan_model(freeze=True)
    vocoder = vocoder.to(device).eval()

    with torch.no_grad():
        audio = vocoder.convert_spectrogram_to_audio(spec=mel)

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output_wav), audio.cpu(), sample_rate=22050)
    print(f"Saved synthesized audio to {args.output_wav}")


if __name__ == "__main__":
    main()
