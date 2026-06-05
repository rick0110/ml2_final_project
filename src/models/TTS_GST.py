import sys
from pathlib import Path
import torch 
import torch.nn as nn
from typing import Union, Tuple

MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from HiFi_GAN import load_hifigan_model
from GST import GST

class TTS_GST(nn.Module):
    def __init__(self, num_gst_tokens: int = 10, gst_token_dim: int = 384, train_hifigan: bool = True):
        super(TTS_GST, self).__init__()
        self.spec_generator, self.vocoder = load_hifigan_model(freeze=not train_hifigan)
        self.gst = GST(n_style_tokens=num_gst_tokens, hidden_size=gst_token_dim)
        
        self.spec_generator.fastpitch.encoder._current_style = None
        self.spec_generator.fastpitch.encoder.register_forward_hook(self._injection_hook)

    @staticmethod
    def _injection_hook(module, inputs, output):
        style = getattr(module, "_current_style", None)
        
        if style is None:
            return output
            
        if isinstance(output, tuple):
            return (output[0] + style,) + output[1:]
        return output + style

    def _extract_style(self, audio_inputs: torch.Tensor, return_att_weights: bool) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        gst_inputs = audio_inputs.unsqueeze(1)
        
        if return_att_weights:
            style_embedding, att_weights = self.gst(gst_inputs, return_att_weights=True)
        else:
            style_embedding = self.gst(gst_inputs, return_att_weights=False)
            att_weights = None
            
        style_expanded = style_embedding.view(style_embedding.size(0), 1, -1)
        
        return style_expanded, att_weights

    def forward(self, text_tokens: torch.Tensor, audio_inputs: torch.Tensor, return_att_weights: bool = False, generate_audio: bool = True) -> Union[torch.Tensor, Tuple]:
        style_expanded, att_weights = self._extract_style(audio_inputs, return_att_weights)
        
        self.spec_generator.fastpitch.encoder._current_style = style_expanded
        
        was_training = self.spec_generator.training
        self.spec_generator.eval() 
        
        try:
            mel_outputs = self.spec_generator.generate_spectrogram(tokens=text_tokens)
        finally:
            self.spec_generator.fastpitch.encoder._current_style = None
            
            if was_training:
                self.spec_generator.train()
                
        audio_outputs = None
        if generate_audio:
            audio_outputs = self.vocoder(spec=mel_outputs[0:1])
        
        if return_att_weights:
            return audio_outputs, mel_outputs, att_weights
        
        return audio_outputs, mel_outputs