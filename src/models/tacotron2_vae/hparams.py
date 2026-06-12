"""Hyperparameters matching tacotron2-vae-master defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Tacotron2VAEHparams:
    ################################
    # Experiment Parameters
    ################################
    epochs: int = 300
    iters_per_checkpoint: int = 500
    seed: int = 42
    dynamic_loss_scaling: bool = True
    distributed_run: bool = False
    dist_backend: str = "nccl"
    dist_url: str = "tcp://localhost:54321"
    cudnn_enabled: bool = True
    cudnn_benchmark: bool = True

    ################################
    # Data Parameters
    ################################
    load_mel_from_disk: bool = True
    training_files: str = ""
    validation_files: str = ""
    text_cleaners: List[str] = field(default_factory=lambda: ["portuguese_cleaners"])
    sort_by_length: bool = False

    ################################
    # Audio Parameters
    ################################
    max_wav_value: float = 32768.0
    sampling_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0

    ################################
    # Model Parameters
    ################################
    n_symbols: int = 80
    symbols_embedding_dim: int = 512

    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embedding_dim: int = 512


    n_emotions: int = 1
    emotion_embedding_dim: int = 16

    E: int = 512
    ref_enc_filters: List[int] = field(default_factory=lambda: [32, 32, 64, 64, 128, 128])
    ref_enc_size: List[int] = field(default_factory=lambda: [3, 3])
    ref_enc_strides: List[int] = field(default_factory=lambda: [2, 2])
    ref_enc_pad: List[int] = field(default_factory=lambda: [1, 1])
    ref_enc_gru_size: int = 256

    z_latent_dim: int = 32
    anneal_function: str = "logistic"
    anneal_k: float = 0.0025
    anneal_x0: int = 10000
    anneal_upper: float = 0.2
    anneal_lag: int = 50000

    prosody_n_convolutions: int = 6
    prosody_conv_dim_in: List[int] = field(default_factory=lambda: [1, 32, 32, 64, 64, 128])
    prosody_conv_dim_out: List[int] = field(default_factory=lambda: [32, 32, 64, 64, 128, 128])
    prosody_conv_kernel: int = 3
    prosody_conv_stride: int = 2
    prosody_embedding_dim: int = 128

    n_frames_per_step: int = 1
    decoder_rnn_dim: int = 1024
    prenet_dim: int = 256
    max_decoder_steps: int = 1000
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1

    attention_rnn_dim: int = 1024
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31

    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5

    ################################
    # Optimization Hyperparameters
    ################################
    use_saved_learning_rate: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1.0
    batch_size: int = 32
    mask_padding: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tacotron2VAEHparams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def create_hparams(overrides: Optional[Dict[str, Any]] = None) -> Tacotron2VAEHparams:
    hparams = Tacotron2VAEHparams()
    if overrides:
        for key, value in overrides.items():
            if hasattr(hparams, key):
                setattr(hparams, key, value)
    return hparams
