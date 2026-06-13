"""Hyperparameters for Tacotron 2 VAE model.

This module defines the default hyperparameters for the Tacotron 2 model
with Variational Autoencoder (VAE) components for prosody modeling.
It uses Python's `dataclasses` for a clean and organized structure.

Dependencies:
    - dataclasses: For creating data classes.
    - typing: For type hinting (Any, Dict, List, Optional).

Typical Usage:
    >>> from src.models.tacotron2_vae.hparams import Tacotron2VAEHparams, create_hparams
    >>> # Create default hyperparameters
    >>> hparams = Tacotron2VAEHparams()
    >>> # Or create with overrides
    >>> overrides = {"epochs": 500, "batch_size": 64}
    >>> hparams_override = create_hparams(overrides)
    >>> print(hparams_override.epochs)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Tacotron2VAEHparams:
    """
    Hyperparameters for the Tacotron 2 VAE model.

    This dataclass encapsulates all configurable parameters for the model,
    covering experiment settings, data loading, audio processing, model architecture,
    and optimization.

    Attributes:
        # Experiment Parameters
        epochs (int): Total number of training epochs.
        iters_per_checkpoint (int): Number of iterations between saving checkpoints.
        seed (int): Random seed for reproducibility.
        dynamic_loss_scaling (bool): Whether to use dynamic loss scaling.
        distributed_run (bool): Whether to run in distributed training mode.
        dist_backend (str): Backend for distributed training (e.g., 'nccl').
        dist_url (str): URL for distributed training initialization.
        cudnn_enabled (bool): Whether to enable cuDNN.
        cudnn_benchmark (bool): Whether to use cuDNN benchmarking for optimization.

        # Data Parameters
        load_mel_from_disk (bool): Whether to load pre-computed mel-spectrograms from disk.
        training_files (str): Path to the training file list.
        validation_files (str): Path to the validation file list.
        text_cleaners (List[str]): List of text cleaning functions to apply.
        sort_by_length (bool): Whether to sort training data by sequence length.

        # Audio Parameters
        max_wav_value (float): Maximum possible value for waveform samples.
        sampling_rate (int): Audio sampling rate.
        filter_length (int): FFT filter length.
        hop_length (int): Hop length for STFT.
        win_length (int): Window length for STFT.
        n_mel_channels (int): Number of mel-spectrogram channels.
        mel_fmin (float): Minimum frequency for mel filter bank.
        mel_fmax (float): Maximum frequency for mel filter bank.

        # Model Parameters
        n_symbols (int): Number of symbols in the vocabulary (e.g., characters).
        symbols_embedding_dim (int): Dimension of the text symbol embedding.
        encoder_kernel_size (int): Kernel size for encoder convolutional layers.
        encoder_n_convolutions (int): Number of convolutional layers in the encoder.
        encoder_embedding_dim (int): Dimension of the encoder output.
        n_emotions (int): Number of distinct emotion categories (if applicable).
        emotion_embedding_dim (int): Dimension of emotion embeddings.
        E (int): Dimension of the style embedding (output of VAE_GST).
        ref_enc_filters (List[int]): Filter sizes for the reference encoder's convolutional layers.
        ref_enc_size (List[int]): Kernel sizes for the reference encoder's convolutional layers.
        ref_enc_strides (List[int]): Strides for the reference encoder's convolutional layers.
        ref_enc_pad (List[int]): Padding for the reference encoder's convolutional layers.
        ref_enc_gru_size (int): Hidden size of the GRU in the reference encoder.
        z_latent_dim (int): Dimensionality of the latent space (z) in the VAE.
        anneal_function (str): KL divergence annealing function ('logistic', 'linear').
        anneal_k (float): Steepness parameter for annealing function.
        anneal_x0 (int): Midpoint parameter for annealing function.
        anneal_upper (float): Upper bound for KL divergence weight during annealing.
        anneal_lag (int): Number of steps before KL divergence annealing starts.
        prosody_n_convolutions (int): Number of convolutional layers for prosody processing.
        prosody_conv_dim_in (List[int]): Input dimensions for prosody convolutional layers.
        prosody_conv_dim_out (List[int]): Output dimensions for prosody convolutional layers.
        prosody_conv_kernel (int): Kernel size for prosody convolutional layers.
        prosody_conv_stride (int): Stride for prosody convolutional layers.
        prosody_embedding_dim (int): Dimension of the final prosody embedding.
        n_frames_per_step (int): Number of mel-frames generated per decoder step.
        decoder_rnn_dim (int): Hidden dimension of the decoder RNN.
        prenet_dim (int): Dimension of the Prenet layers.
        max_decoder_steps (int): Maximum number of decoder steps during inference.
        gate_threshold (float): Threshold for the gate output to stop generation.
        p_attention_dropout (float): Dropout probability for the attention RNN.
        p_decoder_dropout (float): Dropout probability for the decoder RNN.
        attention_rnn_dim (int): Hidden dimension of the attention RNN.
        attention_dim (int): Dimension for attention query/key/value.
        attention_location_n_filters (int): Number of filters in the attention location layer.
        attention_location_kernel_size (int): Kernel size in the attention location layer.
        postnet_embedding_dim (int): Dimension of the postnet's hidden layers.
        postnet_kernel_size (int): Kernel size for postnet convolutional layers.
        postnet_n_convolutions (int): Number of convolutional layers in the postnet.

        # Optimization Hyperparameters
        use_saved_learning_rate (bool): Whether to restore learning rate from checkpoint.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for the optimizer.
        grad_clip_thresh (float): Gradient clipping threshold.
        batch_size (int): Training batch size.
        mask_padding (bool): Whether to mask padded outputs in the model.
    """

    # Experiment Parameters
    epochs: int = 300
    iters_per_checkpoint: int = 500
    seed: int = 42
    dynamic_loss_scaling: bool = True
    distributed_run: bool = False
    dist_backend: str = "nccl"
    dist_url: str = "tcp://localhost:54321"
    cudnn_enabled: bool = True
    cudnn_benchmark: bool = True

    # Data Parameters
    load_mel_from_disk: bool = True
    training_files: str = ""
    validation_files: str = ""
    text_cleaners: List[str] = field(default_factory=lambda: ["portuguese_cleaners"])
    sort_by_length: bool = False

    # Audio Parameters
    max_wav_value: float = 32768.0
    sampling_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0

    # Model Parameters
    n_symbols: int = 80  # Example: Assuming 80 phonemes or characters
    symbols_embedding_dim: int = 512

    # Encoder parameters
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embedding_dim: int = 512

    # Emotion embedding parameters (if applicable)
    n_emotions: int = 1
    emotion_embedding_dim: int = 16

    # VAE_GST parameters
    E: int = 512  # Style embedding dimension
    ref_enc_filters: List[int] = field(default_factory=lambda: [32, 32, 64, 64, 128, 128])
    ref_enc_size: List[int] = field(default_factory=lambda: [3, 3]) # Kernel sizes, not used directly in current ref_encoder impl.
    ref_enc_strides: List[int] = field(default_factory=lambda: [2, 2]) # Strides, not used directly in current ref_encoder impl.
    ref_enc_pad: List[int] = field(default_factory=lambda: [1, 1]) # Padding, not used directly in current ref_encoder impl.
    ref_enc_gru_size: int = 256  # GRU hidden size in reference encoder

    z_latent_dim: int = 32  # Latent space dimension (z)
    anneal_function: str = "logistic"
    anneal_k: float = 0.0025
    anneal_x0: int = 10000
    anneal_upper: float = 0.2
    anneal_lag: int = 50000

    # Prosody parameters (if distinct from VAE_GST, currently seems redundant)
    prosody_n_convolutions: int = 6
    prosody_conv_dim_in: List[int] = field(default_factory=lambda: [1, 32, 32, 64, 64, 128])
    prosody_conv_dim_out: List[int] = field(default_factory=lambda: [32, 32, 64, 64, 128, 128])
    prosody_conv_kernel: int = 3
    prosody_conv_stride: int = 2
    prosody_embedding_dim: int = 128

    # Decoder parameters
    n_frames_per_step: int = 1
    decoder_rnn_dim: int = 1024
    prenet_dim: int = 256
    max_decoder_steps: int = 1000
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1

    # Attention parameters
    attention_rnn_dim: int = 1024
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31

    # Postnet parameters
    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5

    # Optimization Hyperparameters
    use_saved_learning_rate: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1.0
    batch_size: int = 16
    mask_padding: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Converts the hyperparameters dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tacotron2VAEHparams":
        """Creates a Tacotron2VAEHparams instance from a dictionary."""
        # Filter dictionary to include only fields defined in the dataclass
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def create_hparams(overrides: Optional[Dict[str, Any]] = None) -> Tacotron2VAEHparams:
    """
    Creates a Tacotron2VAEHparams instance, applying optional overrides.

    Args:
        overrides (Optional[Dict[str, Any]]): A dictionary of parameters to override
                                               from the default values.

    Returns:
        Tacotron2VAEHparams: An instance of the hyperparameters object.
    """
    hparams = Tacotron2VAEHparams()
    if overrides:
        for key, value in overrides.items():
            # Set attribute if it exists in the hyperparameters object
            if hasattr(hparams, key):
                setattr(hparams, key, value)
    return hparams
