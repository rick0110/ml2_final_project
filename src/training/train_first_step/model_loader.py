"""Model loading and preparation for first-step TTS training.

Handles loading:
- Text Encoder (FastSpeech from HiFi-GAN)
- Acoustic Decoder (LSTM-based)
- Style Extractor (GST)
- Vocoder (HiFi-GAN)
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


def _load_hifigan_model_loader():
    """Load `load_hifigan_model` from src/models supporting hyphenated filenames."""
    hifigan_path = PROJECT_ROOT / "src" / "models" / "HiFi-GAN.py"
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
    4. M_hat → HiFi-GAN → x_hat (predicted audio)
    """
    
    def __init__(
        self,
        text_encoder: nn.Module,
        acoustic_decoder: nn.Module,
        style_extractor: nn.Module,
        vocoder: nn.Module,
        text_encoder_frozen: bool = True,
        vocoder_frozen: bool = True,
    ):
        """Initialize the complete TTS model.
        
        Args:
            text_encoder: FastPitch text encoder
            acoustic_decoder: LSTM-based acoustic decoder
            style_extractor: GST-based style extractor
            vocoder: HiFi-GAN vocoder
            text_encoder_frozen: Whether to freeze text encoder
            vocoder_frozen: Whether to freeze vocoder
        """
        super().__init__()
        
        self.text_encoder = text_encoder
        self.acoustic_decoder = acoustic_decoder
        self.style_extractor = style_extractor
        self.vocoder = vocoder
        # Fallback representation used only if the external text encoder call fails.
        self.text_fallback_embedding = nn.Embedding(num_embeddings=4096, embedding_dim=384)
        self._warned_text_fallback = False
        
        if text_encoder_frozen:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        if vocoder_frozen:
            for param in self.vocoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor,
        use_vocoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            text_ids: Token IDs from text, shape (batch_size, max_text_length)
            target_mel: Target mel spectrogram, shape (batch_size, n_mels, time_steps)
            use_vocoder: Whether to apply vocoder (generates audio)
        
        Returns:
            Tuple of (predicted_mel, style_embeddings) if not use_vocoder
            Tuple of (predicted_audio, style_embeddings) if use_vocoder
        """
        # Step 1: Text → Text Encoder → h_text
        with torch.no_grad() if self.text_encoder.training is False else torch.enable_grad():
            try:
                h_text = self.text_encoder(text=text_ids)
                # h_text shape: (batch_size, seq_len, hidden_size)
                if isinstance(h_text, tuple):
                    h_text = h_text[0]  # Some encoders return tuples
            except Exception as exc:
                if not self._warned_text_fallback:
                    print(
                        "Warning: text encoder forward call failed; using fallback embedding. "
                        f"Original error: {exc}"
                    )
                    self._warned_text_fallback = True
                safe_ids = text_ids.clamp(min=0, max=self.text_fallback_embedding.num_embeddings - 1)
                h_text = self.text_fallback_embedding(safe_ids)
        
        # Step 2: Mel → GST → z_style
        # Ensure target_mel has channel dimension: (batch_size, 1, n_mels, time_steps)
        if target_mel.dim() == 3:
            target_mel_gst = target_mel.unsqueeze(1)
        else:
            target_mel_gst = target_mel
        
        style_output = self.style_extractor(target_mel_gst)
        z_style = style_output[0] if isinstance(style_output, tuple) else style_output
        # Normalize style shape to (batch_size, style_embedding_dim)
        if z_style.dim() == 3 and z_style.size(1) == 1:
            z_style = z_style.squeeze(1)
        elif z_style.dim() > 2:
            z_style = z_style.reshape(z_style.size(0), -1)
        
        # Step 3: Concatenate h_text and z_style
        # First, we need to repeat z_style to match temporal dimension of h_text
        batch_size, seq_len, text_hidden = h_text.shape
        style_dim = z_style.shape[-1]
        
        # Repeat z_style for each time step
        z_style_expanded = z_style.unsqueeze(1).expand(-1, seq_len, -1)
        # z_style_expanded shape: (batch_size, seq_len, style_dim)
        
        # Concatenate along feature dimension
        combined_features = torch.cat([h_text, z_style_expanded], dim=-1)
        # combined_features shape: (batch_size, seq_len, text_hidden + style_dim)
        
        # Step 4: [h_text, z_style] → Acoustic Decoder → M_hat
        predicted_mel = self.acoustic_decoder(combined_features)
        # predicted_mel shape: (batch_size, seq_len, n_mels)
        
        # Step 5 (optional): M_hat → HiFi-GAN → x_hat
        if use_vocoder:
            # Vocoder expects (batch_size, n_mels, time_steps)
            if predicted_mel.dim() == 3:
                predicted_mel_vocoder = predicted_mel.transpose(1, 2)
            else:
                predicted_mel_vocoder = predicted_mel
            
            with torch.no_grad() if self.vocoder.training is False else torch.enable_grad():
                predicted_audio = self.vocoder(predicted_mel_vocoder)
            
            return predicted_audio, z_style
        
        return predicted_mel, z_style
    
    def get_trainable_parameters(self):
        """Get only trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


def load_tts_models(
    device: torch.device,
    acoustic_decoder_hidden_size: int = 256,
    acoustic_decoder_num_layers: int = 1,
    style_embedding_dim: int = 128,
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
    print("Loading HiFi-GAN and FastPitch models...")
    load_hifigan_model = _load_hifigan_model_loader()
    text_encoder, vocoder = load_hifigan_model(freeze=True)
    text_encoder = text_encoder.to(device)
    vocoder = vocoder.to(device)
    print(f"  ✓ Text Encoder (FastPitch) loaded and frozen")
    print(f"  ✓ Vocoder (HiFi-GAN) loaded and frozen")
    
    # Get text encoder output size
    text_hidden_size = 384  # FastPitch default hidden size
    
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
        n_conv_layers=6,
        hidden_size=style_embedding_dim,
        n_style_tokens=10,
        n_mels=80,
        n_heads=4,
    ).to(device)
    print(f"  ✓ GST initialized (trainable)")
    
    # Create complete model
    model = FirstStepTTSModel(
        text_encoder=text_encoder,
        acoustic_decoder=acoustic_decoder,
        style_extractor=style_extractor,
        vocoder=vocoder,
        text_encoder_frozen=True,
        vocoder_frozen=True,
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
        "vocoder": sum(p.numel() for p in model.vocoder.parameters()),
    }
    info["trainable"] = sum(p.numel() for p in model.get_trainable_parameters())
    info["total"] = sum(p.numel() for p in model.parameters())
    
    return info
