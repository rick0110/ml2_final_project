import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

code1 = """import torch
import IPython.display as ipd
import numpy as np
import sys
from pathlib import Path

# Configurar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando device: {device}")"""

code2 = """# 1. Carregar a arquitetura do Tacotron 2 da NVIDIA via PyTorch Hub
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:shared', 'nvidia_tacotron2', pretrained=False)

# 2. Carregar os pesos locais (o modelo pre-treinado que vc tem)
checkpoint_path = 'local_weight_models/tacotron2/nvidia_tacotron2pyt_fp32_20190427'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Pegar o state_dict e remover 'module.' se necessário
state_dict = checkpoint.get('state_dict', checkpoint)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

tacotron2.load_state_dict(state_dict)
tacotron2 = tacotron2.to(device).eval()
print("Pesos do Tacotron 2 carregados com sucesso!")"""

code3 = """# 3. Carregar o Vocoder (WaveGlow) da NVIDIA
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:shared', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to(device).eval()
print("WaveGlow carregado com sucesso!")"""

code4 = """# 4. Text Processing
# Para usar os cleaners originais (english_cleaners), vamos importar diretamente
# do repositório que o torch.hub acabou de clonar na sua pasta de cache.
hub_dir = Path(torch.hub.get_dir()) / 'NVIDIA_DeepLearningExamples_shared' / 'PyTorch' / 'SpeechSynthesis' / 'Tacotron2'
if str(hub_dir) not in sys.path:
    sys.path.append(str(hub_dir))

from text import text_to_sequence

texto = "Hello! This is a test of the pre-trained NVIDIA Tacotron 2 model."
print(f"Texto: {texto}")

# Converter texto em sequência numérica de tokens
sequence = np.array(text_to_sequence(texto, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).to(device).long()"""

code5 = """# 5. Geração e Áudio
with torch.no_grad():
    # Passar texto pelo Tacotron2
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(sequence)
    
    # Passar Espectrograma Mel pelo WaveGlow para gerar áudio
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

audio_numpy = audio[0].data.cpu().numpy()

# 6. Tocar o Áudio no Jupyter
ipd.Audio(audio_numpy, rate=22050)"""

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Teste do Tacotron 2 Original da NVIDIA"),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5)
]

with open('testing_nvidia_tarcotron2.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook testing_nvidia_tarcotron2.ipynb criado com sucesso.")
