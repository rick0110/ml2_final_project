import torch 
import torch.nn as nn
from HiFi_GAN import load_hifigan_model
from GST import GST

class TTS_GST(nn.Module):
    def __init__(self, num_gst_tokens: int = 10, gst_token_dim: int = 256):
        super(TTS_GST, self).__init__()
        self.spec_generator, self.vocoder = load_hifigan_model()
        self.gst = GST(n_style_tokens=num_gst_tokens, hidden_size=gst_token_dim)

    def forward(self, text_embed: torch.Tensor, audio_inputs: torch.Tensor) -> torch.Tensor:
        if audio_inputs.dim() == 3:
            audio_inputs = audio_inputs.unsqueeze(1)
            
        style_embedding = self.gst(audio_inputs)
        
        text_embed_conditioned = text_embed + style_embedding.unsqueeze(1)
        
        mel_outputs = self.spec_generator(text_embed_conditioned)
        audio_outputs = self.vocoder(mel_outputs)
        
        return audio_outputs