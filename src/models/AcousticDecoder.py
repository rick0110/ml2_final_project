from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

class LSTM_AcousticDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out)
        return output # (batch_size, seq_len, n_mels)


class PretrainedTransformerAcousticDecoder(nn.Module):
    """Acoustic decoder backed by a pretrained Transformer encoder.

    The pretrained backbone stays trainable by default (not frozen).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 80,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        h = self.input_proj(x)
        if attention_mask is None:
            attention_mask = torch.ones(h.size(0), h.size(1), dtype=torch.long, device=h.device)

        outputs = self.backbone(inputs_embeds=h, attention_mask=attention_mask)
        return self.output_proj(outputs.last_hidden_state)  # (batch_size, seq_len, n_mels)
