"""
Hyperparameters for Tacotron 2 VAE model.

Responsibilities:
    - Define all configurable parameters for the Tacotron 2 VAE model.
    - Provide utility functions to create, load, and convert hyperparameters.
    - Maintain defaults for experiment, data, audio, model, and optimization settings.

Main Classes:
    - Tacotron2VAEHparams: Data class containing all hyperparameter settings.

Main Functions:
    - create_hparams: Factory function to create hparams with optional overrides.
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
        epochs (int): Total number of training epochs.
        iters_per_checkpoint (int): Number of iterations between saving checkpoints.
        seed (int): Random seed for reproducibility.
        training_files (str): Path to the training file list.
        validation_files (str): Path to the validation file list.
        text_cleaners (List[str]): List of text cleaning functions to apply.
        sort_by_length (bool): Whether to sort training data by sequence length.
        max_wav_value (float): Maximum possible value for waveform samples.
        sampling_rate (int): Audio sampling rate.
        filter_length (int): FFT filter length.
        hop_length (int): Hop length for STFT.
        win_length (int): Window length for STFT.
        n_mel_channels (int): Number of mel-spectrogram channels.
        mel_fmin (float): Minimum frequency for mel filter bank.
        mel_fmax (float): Maximum frequency for mel filter bank.
        n_symbols (int): Number of symbols in the vocabulary (e.g., characters).
        symbols_embedding_dim (int): Dimension of the text symbol embedding.
        encoder_kernel_size (int): Kernel size for encoder convolutional layers.
        encoder_n_convolutions (int): Number of convolutional layers in the encoder.
        encoder_embedding_dim (int): Dimension of the encoder output.
        z_latent_dim (int): Dimensionality of the latent space (z) in the VAE.
        anneal_function (str): KL divergence annealing function ('logistic', 'linear', 'constant').
        anneal_k (float): Steepness parameter for annealing function.
        anneal_x0 (int): Midpoint parameter for annealing function.
        anneal_upper (float): Upper bound for KL divergence weight during annealing.
        anneal_lag (int): Number of steps before KL divergence annealing starts.
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
    cudnn_enabled: bool = True
    cudnn_benchmark: bool = True

    # Data Parameters
    load_mel_from_disk: bool = True
    training_files: str = ""
    validation_files: str = ""
    text_cleaners: List[str] = field(default_factory=lambda: ["portuguese_cleaners"])

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

    # VAE_GST parameters
    E: int = 512  # Style embedding dimension
    ref_enc_filters: List[int] = field(default_factory=lambda: [32, 32, 64, 64, 128, 128])
    ref_enc_size: List[int] = field(default_factory=lambda: [3, 3]) # Kernel sizes
    ref_enc_strides: List[int] = field(default_factory=lambda: [2, 2]) # Strides
    ref_enc_pad: List[int] = field(default_factory=lambda: [1, 1]) # Padding
    ref_enc_gru_size: int = 256  # GRU hidden size in reference encoder

    z_latent_dim: int = 32  # Latent space dimension (z)
    anneal_function: str = "cyclical"
    anneal_k: float = 0.0025
    anneal_x0: int = 4000   # Cycle length for cyclical; midpoint for logistic
    anneal_upper: float = 0.2
    anneal_lag: int = 2000  # Start KL earlier so it regularises from the start
    free_bits: float = 0.5  # Raised from 0.25 to combat posterior collapse

    # Guided attention loss
    guided_attention_weight: float = 2.0  # Weight for guided attention loss (encourages monotonic alignment)
    guided_attention_sigma: float = 0.4   # Gaussian width; larger = more permissive diagonal

    # Decoder parameters
    n_frames_per_step: int = 1
    decoder_rnn_dim: int = 1024
    prenet_dim: int = 256
    max_decoder_steps: int = 1000
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1
    p_decoder_input_dropout: float = 0.5  # Reduced from 0.8; standard Tacotron2 uses 0.5

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
    learning_rate: float = 1e-4  # Raised from 1e-5; was causing plateau
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1.0
    batch_size: int = 32
    mask_padding: bool = True
    warmup_steps: int = 0       # Linear LR warmup; 0 = disabled
    warmup_start_lr: float = 1e-6  # LR at step 0 when warmup is enabled

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the hyperparameters dataclass to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all hyperparameter settings.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Tacotron2VAEHparams:
        """
        Creates a Tacotron2VAEHparams instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing settings.

        Returns:
            Tacotron2VAEHparams: Initialized hyperparameters object.
        """
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

    Example:
        >>> hparams = create_hparams({"batch_size": 32})
        >>> print(hparams.batch_size)
        32
    """
    hparams = Tacotron2VAEHparams()
    if overrides:
        for key, value in overrides.items():
            # Set attribute if it exists in the hyperparameters object
            if hasattr(hparams, key):
                setattr(hparams, key, value)
    return hparams
