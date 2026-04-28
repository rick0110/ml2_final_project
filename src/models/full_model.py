"""
Full Prosody and Style Transfer Model.

Integrates all sub-modules into a single end-to-end model:

    Source Audio  ──► ContentEncoder (HuBERT, frozen)
                           │
                           ▼
    Reference Audio ──► GlobalStyleToken (GST)
                           │  style_emb
                           ▼
                      MappingNetwork ──► mapped content features
                           │
                           ▼
                      VarianceAdaptor ──► adapted features (T_dec)
                           │
                           ▼
                        Decoder ──► mel-spectrogram + waveform

During inference only the source audio and an optional reference audio are
required; style defaults to a zero embedding (neutral prosody) when no
reference is provided.
"""

import torch
import torch.nn as nn

from .content_encoder import ContentEncoder
from .reference_encoder import GlobalStyleToken
from .variance_adaptor import VarianceAdaptor
from .mapping_network import MappingNetwork
from .decoder import Decoder


class ProsodyStyleTransferModel(nn.Module):
    """End-to-end prosody and style transfer model for Portuguese TTS.

    Args:
        hubert_model_name: Hugging Face identifier for the HuBERT model.
        freeze_hubert: Whether to freeze HuBERT weights during training.
        n_mels: Number of mel-filter bins.
        d_model: Internal feature dimension (mapping network output / variance
            adaptor input).
        style_dim: Style embedding dimension (GST output).
        mapping_hidden_dim: Hidden dimension for the mapping network.
        mapping_num_layers: Number of residual blocks in the mapping network.
        gst_num_tokens: Number of global style tokens.
        gst_token_dim: Dimension of each GST embedding.
        gst_num_heads: Number of attention heads in the GST attention layer.
        variance_num_conv: Number of conv layers in variance predictors.
        decoder_upsample_rates: Upsampling rates for the HiFi-GAN vocoder.
        decoder_initial_channels: Initial channel count for the vocoder.
    """

    def __init__(
        self,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        freeze_hubert: bool = True,
        n_mels: int = 80,
        d_model: int = 256,
        style_dim: int = 128,
        mapping_hidden_dim: int = 512,
        mapping_num_layers: int = 4,
        gst_num_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_num_heads: int = 8,
        variance_num_conv: int = 2,
        decoder_upsample_rates: tuple[int, ...] = (8, 8, 2, 2),
        decoder_initial_channels: int = 512,
    ) -> None:
        super().__init__()

        # 1. Content encoder (HuBERT)
        self.content_encoder = ContentEncoder(
            model_name=hubert_model_name,
            freeze=freeze_hubert,
        )
        hubert_dim = self.content_encoder.out_dim

        # 2. Reference encoder with Global Style Tokens
        self.reference_encoder = GlobalStyleToken(
            n_mels=n_mels,
            num_tokens=gst_num_tokens,
            token_dim=gst_token_dim,
            num_heads=gst_num_heads,
            ref_embedding_dim=style_dim,
        )

        # 3. Mapping network (HuBERT → decoder space)
        self.mapping_network = MappingNetwork(
            input_dim=hubert_dim,
            output_dim=d_model,
            hidden_dim=mapping_hidden_dim,
            num_layers=mapping_num_layers,
            style_dim=style_dim,
        )

        # 4. Variance adaptor
        self.variance_adaptor = VarianceAdaptor(
            d_model=d_model,
            style_dim=style_dim,
            num_conv_layers=variance_num_conv,
        )

        # 5. Decoder (mel predictor + HiFi-GAN vocoder)
        self.decoder = Decoder(
            input_dim=d_model,
            n_mels=n_mels,
            mel_hidden_dim=d_model,
            upsample_rates=decoder_upsample_rates,
            upsample_initial_channel=decoder_initial_channels,
        )

    def forward(
        self,
        source_waveform: torch.Tensor,
        ref_mel: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        target_durations: torch.Tensor | None = None,
        target_pitch: torch.Tensor | None = None,
        target_energy: torch.Tensor | None = None,
        target_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the full model.

        Args:
            source_waveform: Raw waveform input, shape ``(B, T_wav)``.
            ref_mel: Reference mel-spectrogram for style extraction, shape
                ``(B, n_mels, T_ref)``.  When ``None`` a zero (neutral) style
                embedding is used.
            attention_mask: Attention mask for the HuBERT encoder,
                shape ``(B, T_wav)``.
            target_durations: Teacher-forced duration targets for training,
                shape ``(B, T_enc)``.
            target_pitch: Teacher-forced pitch targets, shape ``(B, T_enc, 1)``.
            target_energy: Teacher-forced energy targets, shape ``(B, T_enc, 1)``.
            target_len: Target decoder frame length.

        Returns:
            Dictionary containing:
                - ``"mel"``: Predicted mel-spectrogram ``(B, n_mels, T_dec)``.
                - ``"waveform"``: Generated waveform ``(B, 1, T_wav_out)``.
                - ``"pred_durations"``: Predicted log-durations ``(B, T_enc, 1)``.
                - ``"pred_pitch"``: Predicted pitch ``(B, T_enc, 1)``.
                - ``"pred_energy"``: Predicted energy ``(B, T_enc, 1)``.
                - ``"style_emb"``: Style embedding from GST ``(B, style_dim)``.
                - ``"content_features"``: Raw HuBERT features ``(B, T_enc, hubert_dim)``.
        """
        # Step 1: Extract content features via HuBERT
        content_features = self.content_encoder(source_waveform, attention_mask)

        # Step 2: Extract style embedding from reference audio
        if ref_mel is not None:
            style_emb = self.reference_encoder(ref_mel)  # (B, style_dim)
        else:
            B = source_waveform.size(0)
            style_emb = source_waveform.new_zeros(B, self.reference_encoder.style_dim)

        # Step 3: Map content features to decoder space
        mapped = self.mapping_network(content_features, style_emb)  # (B, T_enc, d_model)

        # Step 4: Variance adaption (duration, pitch, energy)
        variance_out = self.variance_adaptor(
            mapped,
            style_emb,
            target_durations=target_durations,
            target_pitch=target_pitch,
            target_energy=target_energy,
            target_len=target_len,
        )
        adapted = variance_out["output"]  # (B, T_dec, d_model)

        # Step 5: Decode to mel-spectrogram and waveform
        decoder_out = self.decoder(adapted)

        return {
            "mel": decoder_out["mel"],
            "waveform": decoder_out["waveform"],
            "pred_durations": variance_out["pred_durations"],
            "pred_pitch": variance_out["pred_pitch"],
            "pred_energy": variance_out["pred_energy"],
            "style_emb": style_emb,
            "content_features": content_features,
        }

    @torch.no_grad()
    def infer(
        self,
        source_waveform: torch.Tensor,
        ref_mel: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference helper (no teacher-forcing, gradient-free).

        Args:
            source_waveform: Raw waveform, shape ``(B, T_wav)``.
            ref_mel: Reference mel-spectrogram for style, shape
                ``(B, n_mels, T_ref)``.
            attention_mask: Attention mask, shape ``(B, T_wav)``.

        Returns:
            Same dictionary as :meth:`forward`.
        """
        return self.forward(source_waveform, ref_mel=ref_mel, attention_mask=attention_mask)
