#!/usr/bin/env python
"""
Inference entry-point for the Prosody and Style Transfer model.

Usage::

    # Transfer prosody from reference.wav to source.wav
    python scripts/inference.py \\
        --checkpoint checkpoints/best.pt \\
        --source path/to/source.wav \\
        --reference path/to/reference.wav \\
        --output output.wav

    # Neutral style (no reference)
    python scripts/inference.py \\
        --checkpoint checkpoints/best.pt \\
        --source path/to/source.wav \\
        --output output.wav

    # Batch processing
    python scripts/inference.py \\
        --checkpoint checkpoints/best.pt \\
        --source_dir path/to/sources/ \\
        --reference path/to/reference.wav \\
        --output_dir outputs/
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.preprocessing import AudioPreprocessor
from inference.pipeline import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prosody style transfer inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--source", default=None, help="Source audio file")
    parser.add_argument("--source_dir", default=None, help="Directory of source audio files")
    parser.add_argument("--reference", default=None, help="Reference audio file (for style)")
    parser.add_argument("--output", default="output.wav", help="Output file (single file mode)")
    parser.add_argument("--output_dir", default="outputs", help="Output directory (batch mode)")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg: dict = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    audio_cfg = cfg.get("audio", {})
    preprocessor = AudioPreprocessor(
        sample_rate=audio_cfg.get("sample_rate", 22050),
        n_mels=audio_cfg.get("n_mels", 80),
        n_fft=audio_cfg.get("n_fft", 1024),
        hop_length=audio_cfg.get("hop_length", 256),
        win_length=audio_cfg.get("win_length", 1024),
        f_min=audio_cfg.get("f_min", 0.0),
        f_max_mel=audio_cfg.get("f_max_mel", 8000.0),
        f_min_pitch=audio_cfg.get("f_min_pitch", 50.0),
        f_max_pitch=audio_cfg.get("f_max_pitch", 600.0),
    )

    model_cfg = cfg.get("model", {})
    pipeline = InferencePipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_config=model_cfg if model_cfg else None,
        preprocessor=preprocessor,
        device=args.device,
    )
    logger.info("Model loaded. Running inference…")

    if args.source_dir is not None:
        source_dir = Path(args.source_dir)
        exts = {".wav", ".flac", ".mp3"}
        source_files = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in exts)
        if not source_files:
            logger.error("No audio files found in %s", source_dir)
            sys.exit(1)
        out_paths = pipeline.transfer_batch(
            source_paths=source_files,
            reference_path=args.reference,
            output_dir=args.output_dir,
        )
        logger.info("Processed %d files → %s", len(out_paths), args.output_dir)

    elif args.source is not None:
        waveform = pipeline.transfer(
            source_path=args.source,
            reference_path=args.reference,
        )
        pipeline.save(waveform, args.output, sample_rate=preprocessor.sample_rate)
        logger.info("Saved output to %s", args.output)

    else:
        logger.error("Provide --source or --source_dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
