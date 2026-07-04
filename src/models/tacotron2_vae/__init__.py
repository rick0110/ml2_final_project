"""
Tacotron 2 VAE Model package.

Responsibilities:
    - Expose primary model classes and hyperparameter utilities.
    - Provide a centralized interface for model initialization.

Main Symbols:
    - Tacotron2: Top-level model class.
    - Tacotron2VAEHparams: Configuration data class.
    - create_hparams: Factory for configuration.
    - load_tacotron2_vae_model: Factory for model initialization.
"""
from models.tacotron2_vae.hparams import Tacotron2VAEHparams, create_hparams
from models.tacotron2_vae.model import Tacotron2, load_tacotron2_vae_model

__all__ = [
    "Tacotron2VAEHparams",
    "create_hparams",
    "Tacotron2",
    "load_tacotron2_vae_model",
]
