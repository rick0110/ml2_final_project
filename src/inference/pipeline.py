"""
Inference pipeline for prosody and style transfer.

The pipeline takes a source audio file (whose *content* should be preserved)
and an optional reference audio file (whose *prosody / style* should be
transferred), and returns a synthesised waveform.

Typical usage::

    from inference.pipeline import InferencePipeline

    pipeline = InferencePipeline.from_checkpoint("checkpoints/best.pt")
    waveform = pipeline.transfer(
        source_path="path/to/source.wav",
        reference_path="path/to/reference.wav",
    )
    pipeline.save(waveform, "output.wav")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torchaudio

from ..models.full_model import ProsodyStyleTransferModel
from ..data.preprocessing import AudioPreprocessor

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference pipeline.

    Args:
        model: Pretrained :class:`~models.ProsodyStyleTransferModel`.
        preprocessor: :class:`~data.preprocessing.AudioPreprocessor` instance
            used to extract features from input audio files.
        device: Inference device.
    """

    def __init__(
        self,
        model: ProsodyStyleTransferModel,
        preprocessor: AudioPreprocessor | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.preprocessor = preprocessor or AudioPreprocessor()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: dict[str, Any] | None = None,
        preprocessor: AudioPreprocessor | None = None,
        device: str | torch.device | None = None,
    ) -> "InferencePipeline":
        """Build a pipeline from a saved checkpoint.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint produced by
                :class:`~training.Trainer`.
            model_config: Keyword arguments forwarded to
                :class:`~models.ProsodyStyleTransferModel`.  If ``None`` the
                defaults are used.
            preprocessor: Optional pre-configured preprocessor.
            device: Inference device.

        Returns:
            Configured :class:`InferencePipeline` instance.
        """
        cfg = model_config or {}
        model = ProsodyStyleTransferModel(**cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
        return cls(model, preprocessor=preprocessor, device=device)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def transfer(
        self,
        source_path: str | Path,
        reference_path: str | Path | None = None,
    ) -> torch.Tensor:
        """Perform prosody / style transfer.

        The *content* (phonemes, linguistic information) is taken from
        *source_path* while the *style* (prosody, emotion) is drawn from
        *reference_path*.  When no reference is provided the model produces
        neutral speech.

        Args:
            source_path: Path to the source audio file.
            reference_path: Path to the reference audio file.  If ``None``
                neutral style is used.

        Returns:
            Generated waveform as a 1-D tensor with the sample rate from the
            preprocessor (default 22050 Hz).
        """
        # Load and prepare source audio
        source_wav, _ = self.preprocessor.load(str(source_path))
        source_waveform = source_wav.squeeze(0).unsqueeze(0).to(self.device)  # (1, T)

        # Load and prepare reference mel (if provided)
        ref_mel = None
        if reference_path is not None:
            ref_wav, _ = self.preprocessor.load(str(reference_path))
            ref_mel = self.preprocessor.mel(ref_wav).unsqueeze(0).to(self.device)  # (1, n_mels, T)

        output = self.model.infer(source_waveform, ref_mel=ref_mel)
        waveform = output["waveform"].squeeze()  # (T_wav,)
        return waveform.cpu()

    @staticmethod
    def save(
        waveform: torch.Tensor,
        path: str | Path,
        sample_rate: int = 22050,
    ) -> None:
        """Save a generated waveform to an audio file.

        Args:
            waveform: 1-D waveform tensor to save.
            path: Output file path.  Format is inferred from the extension
                (e.g., ``.wav``, ``.flac``).
            sample_rate: Sample rate of the waveform.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(str(path), waveform, sample_rate)
        logger.info("Saved waveform to %s", path)

    @torch.no_grad()
    def transfer_batch(
        self,
        source_paths: list[str | Path],
        reference_path: str | Path | None = None,
        output_dir: str | Path = "outputs",
    ) -> list[Path]:
        """Apply style transfer to a list of source files.

        Args:
            source_paths: List of paths to source audio files.
            reference_path: Common reference audio for all sources.  If
                ``None`` neutral style is used for all.
            output_dir: Directory where output files are saved.

        Returns:
            List of paths to the generated audio files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for src in source_paths:
            waveform = self.transfer(src, reference_path)
            out_path = output_dir / (Path(src).stem + "_transferred.wav")
            self.save(waveform, out_path, sample_rate=self.preprocessor.sample_rate)
            output_paths.append(out_path)
        return output_paths
