import os
import glob
import torchaudio
from torch.utils.data import Dataset
import torch

class VerboEmotionDataset(Dataset):
    def __init__(self, verbo_audios_dir, processor):
        """
        verbo_audios_dir: Caminho para a pasta "Audios" do VERBO
        processor: O processador do HuBERT carregado 
        """
        self.processor = processor
        
        self.emotion_map = {
            'ale': 0,  # Alegria
            'des': 1,  # Desgosto / Nojo
            'med': 2,  # Medo
            'neu': 3,  # Neutro
            'rai': 4,  # Raiva
            'sur': 5,  # Surpresa
            'tri': 6   # Tristeza
        }
        
        
        self.file_paths = glob.glob(os.path.join(verbo_audios_dir, "**", "*.wav"), recursive=True)
        
        if len(self.file_paths) == 0:
            print(f" Alerta: Nenhum arquivo .wav encontrado em: {verbo_audios_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_name = os.path.basename(file_path)
        
        
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Passando os audios para a forma padrão do HuBERT em 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # (Audio Mono / Estéreo) - HuBERT espera mono, então se for estéreo, convertemos para mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        speech = waveform.squeeze(0).numpy()
        
        prefix = file_name.split('-')[0].lower()[:3] 
        
        label = self.emotion_map.get(prefix, 3) 
        
        inputs = self.processor(
        speech, 
        sampling_rate=16000, 
        padding="max_length", 
        max_length=64000,  #4 seg de áudio
        truncation=True, 
        return_tensors="pt"
        )

        return inputs.input_values.squeeze(0), label
    
