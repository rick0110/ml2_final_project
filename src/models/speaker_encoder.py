"""
Pretrained speaker encoder used as a frozen second style input.

The model is ECAPA-TDNN trained on VoxCeleb (speechbrain/spkrec-ecapa-voxceleb, 192
dimensions). It is a different network from the one that scores voice similarity in the
evaluation (Resemblyzer), so the metric is not computed with the conditioning model.
"""
from __future__ import annotations

from typing import Any

import torch
import torchaudio
from torch import Tensor

SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000


def load_speaker_encoder(device: torch.device, savedir: str = "ckpt/spkrec-ecapa-voxceleb") -> Any:
    """Download (once) and load the speaker encoder."""
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source=SOURCE, savedir=savedir, run_opts={"device": str(device)})


@torch.inference_mode()
def speaker_embedding(encoder: Any, waveform: Tensor, sample_rate: int) -> Tensor:
    """
    Embedding of one utterance.

    Args:
        encoder: Model from load_speaker_encoder.
        waveform (Tensor): Audio (S,) or (1, S).
        sample_rate (int): Sampling rate of the audio.

    Returns:
        Tensor: Unit-norm embedding (192,), on the CPU.
    """
    wav = waveform.reshape(1, -1).float()
    if sample_rate != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE)
    emb = encoder.encode_batch(wav.to(encoder.device)).reshape(-1)
    return torch.nn.functional.normalize(emb, dim=0).cpu()
