from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.GST import GST
from models.NormalizingFlow import NormalizingFlow
from models.PosteriorEncoder import PosteriorEncoder
from vocoder import HiFiGenerator, HiFiGanDiscriminators


class TextEncoderXLMR(nn.Module):
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.pad_token_id = self.backbone.config.pad_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            if self.pad_token_id is not None:
                attention_mask = (input_ids != self.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self, num_layers: int) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        if num_layers <= 0:
            return
        encoder = getattr(self.backbone, "encoder", None)
        if encoder is None or not hasattr(encoder, "layer"):
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        layers = encoder.layer
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


class E2EFlowModel(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        style_dim: int = 256,
        latent_dim: int = 192,
        flow_layers: int = 4,
        flow_hidden: int = 192,
        gst_tokens: int = 30,
        text_model_name: str = "xlm-roberta-base",
    ):
        super().__init__()
        self.n_mels = n_mels
        self.style_dim = style_dim
        self.latent_dim = latent_dim

        self.text_encoder = TextEncoderXLMR(model_name=text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, latent_dim)
        self.gst = GST(n_conv_layers=6, hidden_size=style_dim, n_style_tokens=gst_tokens, n_mels=n_mels, n_heads=4)
        self.style_proj = nn.Linear(style_dim, latent_dim)
        self.posterior_encoder = PosteriorEncoder(n_mels=n_mels, hidden_channels=latent_dim, latent_channels=latent_dim)
        self.flow = NormalizingFlow(channels=latent_dim, cond_channels=latent_dim, n_flows=flow_layers, hidden_channels=flow_hidden)
        self.vocoder = HiFiGenerator(input_channels=latent_dim)

    def align_text(self, text_states: torch.Tensor, target_length: int) -> torch.Tensor:
        if text_states.dim() != 3:
            raise ValueError(f"text_states must be 3D (batch, time, channels), got {tuple(text_states.shape)}")
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        projected = self.text_proj(text_states).transpose(1, 2)
        return F.interpolate(projected, size=target_length, mode="linear", align_corners=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mel: torch.Tensor,
        generate_audio: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if target_mel.dim() != 3:
            raise ValueError(f"target_mel must be 3D (batch, n_mels, time), got {tuple(target_mel.shape)}")

        text_states = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        style_ref = self.gst(target_mel.unsqueeze(1))
        z_post, post_mean, post_log_std = self.posterior_encoder(target_mel)

        aligned_text = self.align_text(text_states, target_length=z_post.size(-1))
        style_cond = self.style_proj(style_ref).unsqueeze(-1).expand(-1, -1, z_post.size(-1))
        cond = aligned_text + style_cond

        z_prior, log_det = self.flow(z_post, cond)
        if generate_audio:
            with torch.no_grad():
                generated_audio = self.vocoder(z_post)
        else:
            generated_audio = torch.empty(0, device=z_post.device)

        return {
            "text_states": text_states,
            "style_ref": style_ref,
            "z_post": z_post,
            "post_mean": post_mean,
            "post_log_std": post_log_std,
            "z_prior": z_prior,
            "log_det": log_det,
            "generated_audio": generated_audio,
            "cond": cond,
        }

    def freeze_text_encoder(self) -> None:
        self.text_encoder.freeze()

    def unfreeze_text_encoder_last_layers(self, num_layers: int) -> None:
        self.text_encoder.unfreeze_last_layers(num_layers)

    def freeze_vocoder(self) -> None:
        for param in self.vocoder.parameters():
            param.requires_grad = False

    def unfreeze_vocoder(self) -> None:
        for param in self.vocoder.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> Tuple[nn.Parameter, ...]:
        return tuple(param for param in self.parameters() if param.requires_grad)

    def initialize_identity(self) -> None:
        """
        Initialize only the normalizing flow close to identity so the
        posterior encoder can keep its standard initialization and learn
        meaningful latent representations.
        """
        self.flow.initialize_identity()


def build_discriminators() -> HiFiGanDiscriminators:
    return HiFiGanDiscriminators()


def get_model_size_info(model: E2EFlowModel) -> Dict[str, int]:
    return {
        "text_encoder": sum(param.numel() for param in model.text_encoder.parameters()),
        "gst": sum(param.numel() for param in model.gst.parameters()),
        "posterior_encoder": sum(param.numel() for param in model.posterior_encoder.parameters()),
        "flow": sum(param.numel() for param in model.flow.parameters()),
        "vocoder": sum(param.numel() for param in model.vocoder.parameters()),
        "trainable": sum(param.numel() for param in model.get_trainable_parameters()),
        "total": sum(param.numel() for param in model.parameters()),
    }
