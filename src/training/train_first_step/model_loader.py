"""Model loading and preparation for first-step TTS training.

Handles loading:
- Text Encoder (FastSpeech from HiFi_GAN)
- Acoustic Decoder (LSTM-based)
- Style Extractor (GST)
- Vocoder (HiFi_GAN)
"""

import sys
import importlib.util
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.AcousticDecoder import LSTM_AcousticDecoder
from models.GST import GST
from models.TextEncoder import TextEncoder, TextEncoderMultiHeadAttention
from models.DurationPredictor import DurationPredictor
from models.LengthRegulator import length_regulator


def _load_hifigan_model_loader():
    """Load `load_hifigan_model` from src/models supporting hyphenated filenames."""
    hifigan_path = PROJECT_ROOT / "src" / "models" / "HiFi_GAN.py"
    if hifigan_path.exists():
        spec = importlib.util.spec_from_file_location("hifigan_module", hifigan_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {hifigan_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "load_hifigan_model"):
            raise ImportError(f"`load_hifigan_model` not found in {hifigan_path}")
        return module.load_hifigan_model

    # Fallback for environments where file may be renamed to a valid module name.
    from models.HiFi_GAN import load_hifigan_model  # type: ignore

    return load_hifigan_model


class FirstStepTTSModel(nn.Module):
    """Complete TTS model for first-step training.
    
    Pipeline:
    1. Text → Text Encoder (FastPitch) → h_text
    2. Mel → GST → z_style
    3. [h_text, z_style] → Acoustic Decoder → M_hat (predicted mel)
    4. M_hat → HiFi_GAN → x_hat (predicted audio)
    """
    
    def __init__(
            self,
            text_encoder: nn.Module,
            acoustic_decoder: nn.Module,
            style_extractor: nn.Module,
            duration_predictor: Optional[nn.Module] = None,
            text_encoder_frozen: bool = False,
        ):
        """Initialize the complete TTS model.
        
        Args:
            text_encoder: FastPitch text encoder
            acoustic_decoder: LSTM-based acoustic decoder
            style_extractor: GST-based style extractor
            text_encoder_frozen: Whether to freeze text encoder
        """
        super().__init__()
        
        self.text_encoder = text_encoder
        self.acoustic_decoder = acoustic_decoder
        self.style_extractor = style_extractor
        self.duration_predictor = duration_predictor
        self.text_fallback_embedding = nn.Embedding(num_embeddings=4096, embedding_dim=384)
        self._warned_text_fallback = False
        
        if text_encoder_frozen:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
    
    def forward(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            text_ids: Token IDs from text, shape (batch_size, max_text_length)
            target_mel: Target mel spectrogram, shape (batch_size, n_mels, time_steps)
        
        Returns:
            Tuple of (predicted_mel, style_embeddings)
        """
        # Step 1: Text → Text Encoder → h_text
        with torch.no_grad() if self.text_encoder.training is False else torch.enable_grad():
            h_text = self.text_encoder(text_ids) # [batch_size, seq_len, text_hidden_size]

        # If a duration predictor is provided, predict durations and expand text states
        if self.duration_predictor is not None:
            # predicted durations are positive floats; convert to integers
            pred_durs = self.duration_predictor(h_text)  # (batch, seq_len)
            # convert to integer durations (at least 1 frame)
            durations_int = torch.clamp(torch.round(pred_durs), min=1).long()
            h_aligned = length_regulator(h_text, durations_int)
            # h_aligned: (batch, frames, text_hidden_size)
        else:
            h_aligned = h_text
        
        # Step 2: Mel → GST → z_style (GST expects input shape: batch, 1, n_mels, time)
        z_style_vec = self.style_extractor(target_mel.unsqueeze(1))  # [batch_size, style_embedding_dim]

        # Step 3: Concatenate aligned text states and an expanded z_style for decoder input
        z_style_exp = z_style_vec.unsqueeze(1).expand(-1, h_aligned.size(1), -1)  # [batch_size, frames, style_embedding_dim]
        concat_z_h = torch.cat([h_aligned, z_style_exp], dim=-1)  # [batch_size, frames, text_hidden_size + style_embedding_dim]

        # Step 4: [h_text, z_style] → Acoustic Decoder → M_hat (predicted mel)
        predicted_mel = self.acoustic_decoder(concat_z_h)  # [batch_size, seq_len, n_mels]

        # Return predicted mel and the original 2D style embedding (batch_size, style_embedding_dim)
        return predicted_mel, z_style_vec
    
    def get_trainable_parameters(self):
        """Get only trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


def load_tts_models(
    device: torch.device,
    vocab_size: int,
    acoustic_decoder_hidden_size: int = 256,
    acoustic_decoder_num_layers: int = 1,
    style_embedding_dim: int = 128,
    text_encoder_hidden_size: int = 256
) -> FirstStepTTSModel:
    """Load and initialize all TTS components.
    
    Args:
        device: Device to load models to
        acoustic_decoder_hidden_size: Hidden size for LSTM decoder
        acoustic_decoder_num_layers: Number of LSTM layers
        style_embedding_dim: Dimension of style embeddings
    
    Returns:
        FirstStepTTSModel with all components initialized
    """
    
    # Get text encoder output size
    text_hidden_size = text_encoder_hidden_size
    
    print("Initializing Acoustic Decoder...")
    acoustic_decoder = LSTM_AcousticDecoder(
        input_size=text_hidden_size + style_embedding_dim,
        hidden_size=acoustic_decoder_hidden_size,
        num_layers=acoustic_decoder_num_layers,
        output_size=80,  # n_mels = 80
    ).to(device)
    print(f"  ✓ Acoustic Decoder initialized (trainable)")
    
    print("Initializing GST (Style Extractor)...")
    style_extractor = GST(
        n_conv_layers=4,
        hidden_size=style_embedding_dim,
        n_style_tokens=30,
        n_mels=80,
        n_heads=4,
    ).to(device)
    print(f"  ✓ GST initialized (trainable)")

    print("Loading (Text Encoder)...")
    text_encoder = TextEncoderMultiHeadAttention(
        vocab_size,
        embedding_dim=text_encoder_hidden_size,
        n_heads=8,
        n_steps=4,
        ff_dim=256,
        dropout=0.0,
    ).to(device)

    print("Initializing Duration Predictor...")
    duration_predictor = None # DurationPredictor(input_dim=text_encoder_hidden_size, conv_channels=256).to(device)
    print("  ✓ Duration Predictor initialized (trainable)")
    
    # Create complete model
    model = FirstStepTTSModel(
        text_encoder=text_encoder,
        acoustic_decoder=acoustic_decoder,
        style_extractor=style_extractor,
        duration_predictor=duration_predictor,
        text_encoder_frozen=False,
    )
    
    print("\nModel summary:")
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    return model.to(device)


def get_model_size_info(model: FirstStepTTSModel) -> dict:
    """Get detailed information about model parameter counts.
    
    Args:
        model: FirstStepTTSModel instance
    
    Returns:
        Dictionary with parameter counts per component
    """
    info = {
        "text_encoder": sum(p.numel() for p in model.text_encoder.parameters()),
        "acoustic_decoder": sum(p.numel() for p in model.acoustic_decoder.parameters()),
        "style_extractor": sum(p.numel() for p in model.style_extractor.parameters()),
    }
    info["trainable"] = sum(p.numel() for p in model.get_trainable_parameters())
    info["total"] = sum(p.numel() for p in model.parameters())
    
    return info
