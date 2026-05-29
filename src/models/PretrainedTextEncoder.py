import torch
import torch.nn as nn
from transformers import AutoModel


class PretrainedTextEncoder(nn.Module):
    """Hugging Face pretrained text encoder with optional frozen backbone.

    Returns contextual token embeddings with shape (batch, seq_len, output_dim).
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        output_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone

        hidden_size = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(hidden_size, output_dim)

        # Prefer config pad token id; fallback for safety.
        self.pad_token_id = self.backbone.config.pad_token_id

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Build attention mask from pad token to ignore padded positions.
        if self.pad_token_id is not None:
            attention_mask = (input_ids != self.pad_token_id).long()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        return self.proj(hidden_states)
