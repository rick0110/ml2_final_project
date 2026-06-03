import sys
from pathlib import Path
import torch 
import torch.nn as nn
from typing import Union, Tuple

# Garante que a pasta 'models' está no path (evita ModuleNotFoundError)
MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from HiFi_GAN import load_hifigan_model
from GST import GST

class TTS_GST(nn.Module):
    def __init__(self, num_gst_tokens: int = 10, gst_token_dim: int = 256):
        super(TTS_GST, self).__init__()
        self.spec_generator, self.vocoder = load_hifigan_model()
        self.gst = GST(n_style_tokens=num_gst_tokens, hidden_size=gst_token_dim)

    def forward(self, text_embed: torch.Tensor, audio_inputs: torch.Tensor, return_att_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if audio_inputs.dim() == 3:
            audio_inputs = audio_inputs.unsqueeze(1)
            
        if return_att_weights:
            style_embedding, att_weights = self.gst(audio_inputs, return_att_weights=True)
        else:
            style_embedding = self.gst(audio_inputs, return_att_weights=False)

        print(f"Style embedding shape: {style_embedding.shape}")  # Debugging line
        print(f"Text embedding shape: {text_embed.shape}")  # Debugging line
        
        text_embed_conditioned = text_embed + style_embedding.unsqueeze(1)
        
        mel_outputs = self.spec_generator(text_embed_conditioned)
        audio_outputs = self.vocoder(mel_outputs)
        
        if return_att_weights:
            return audio_outputs, att_weights
        else:
            return audio_outputs