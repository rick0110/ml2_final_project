"""
Content Encoder based on HuBERT.

HuBERT (Hidden-Unit BERT) is used as a self-supervised speech representation model.
It provides rich content features that are language-agnostic when fine-tuned appropriately,
making it suitable as a content encoder for cross-lingual prosody transfer.
"""

import torch
import torch.nn as nn
from transformers import HubertModel


class ContentEncoder(nn.Module):
    """Wraps a pretrained HuBERT model to produce content representations.

    The encoder can optionally be frozen (all HuBERT weights fixed) so that
    only the downstream mapping network is trained, enabling low-resource
    adaptation to Portuguese.

    Args:
        model_name: Hugging Face model identifier for HuBERT.
        freeze: If ``True`` all HuBERT parameters are frozen.
        output_dim: Optional projection dimension.  When ``None`` no projection
            is applied and the raw HuBERT hidden size is used.
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        freeze: bool = True,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        self.hidden_size: int = self.hubert.config.hidden_size

        if freeze:
            for param in self.hubert.parameters():
                param.requires_grad = False

        if output_dim is not None and output_dim != self.hidden_size:
            self.projection: nn.Module = nn.Linear(self.hidden_size, output_dim)
            self.out_dim = output_dim
        else:
            self.projection = nn.Identity()
            self.out_dim = self.hidden_size

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract content features from raw waveform samples.

        Args:
            input_values: Raw waveform tensor of shape ``(B, T_wav)``.
            attention_mask: Boolean mask of shape ``(B, T_wav)``.

        Returns:
            Content features of shape ``(B, T_frames, out_dim)``.
        """
        outputs = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # (B, T_frames, hidden_size)
        return self.projection(hidden_states)
