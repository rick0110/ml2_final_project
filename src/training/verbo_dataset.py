import os
import re
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

class VerboDataset(Dataset):
    def __init__(self, dataset_root, target_sample_rate=22050, n_mels=80):
        """
        Dataset customizado para o VERBO-Dataset focado em extração de Mel-espectrogramas.
        
        Args:
            dataset_root (str): Caminho até a pasta principal do VERBO-Dataset (contendo a pasta 'Audios').
            target_sample_rate (int): Taxa de amostragem que o modelo espera (ex: 22050 ou 24000 Hz).
            n_mels (int): Número de canais Mel (padrão do Tacotron/FastPitch é 80).
        """
        self.audio_dir = os.path.join(dataset_root, "Audios")
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.file_list = []
        
        # Dicionário para mapear as strings de emoção para IDs numéricos (útil para validação interna)
        self.emotion_to_id = {
            "ale": 0, "des": 1, "med": 2, "neu": 3, 
            "rai": 4, "sur": 5, "tri": 6
        }
        
        # Inicializa o transformador para gerar o Espectrograma Mel
        # Nota: Os parâmetros n_fft, hop_length e win_length devem idealmente coincidir 
        # com os encontrados em configs.py do seu repositório.
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=self.n_mels,
            power=2.0
        )
        
        self._build_index()

    def _build_index(self):
        """Varre as 12 pastas e extrai os metadados com um filtro mais flexível."""
        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError(f"Diretório de áudios não encontrado em: {self.audio_dir}")
            
        # Regex hiper-flexível:
        # Pega a Emoção (grupo 1), Locutor (grupo 2) e a Frase/Código (grupo 3)
        # Separados por hífen ou underline
        pattern = re.compile(r"^([a-zA-Z]+)[-_]([a-zA-Z0-9]+)[-_](.+)(?:\.wav)$", re.IGNORECASE)

        arquivos_ignorados = []

        for folder in os.listdir(self.audio_dir):
            folder_path = os.path.join(self.audio_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    # Aceita tanto .wav quanto .WAV
                    if filename.lower().endswith(".wav"):
                        match = pattern.match(filename)
                        if match:
                            emotion_str, speaker, phrase_id = match.groups()
                            
                            # Padroniza pegando só as 3 primeiras letras em minúsculo (ex: 'Ale')
                            emo_key = emotion_str.lower()[:3]
                            emotion_id = self.emotion_to_id.get(emo_key, -1)
                            
                            self.file_list.append({
                                "file_path": os.path.join(folder_path, filename),
                                "emotion_str": emo_key,
                                "emotion_id": emotion_id,
                                "speaker": speaker,
                                "phrase_id": phrase_id
                            })
                        else:
                            arquivos_ignorados.append(filename)

        # Sistema de Debug: Mostra o que ficou de fora
        if arquivos_ignorados:
            print(f"\n[ALERTA] {len(arquivos_ignorados)} arquivos foram ignorados por causa do nome.")
            print(f"Exemplos do que foi ignorado: {arquivos_ignorados[:10]}\n")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        meta = self.file_list[index]
        
        # 1. Carregar o arquivo de áudio bruto (.wav)
        waveform, sample_rate = torchaudio.load(meta["file_path"])
        
        # 2. Garantir que o áudio seja Mono (reduzir canais se houver estéreo)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 3. Resampling se a taxa de amostragem original diferir do alvo do modelo
        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            
        # 4. Computar o Espectrograma Mel
        mel_spectrogram = self.mel_transform(waveform)  # Shape: [1, n_mels, time_frames]
        mel_spectrogram = mel_spectrogram.squeeze(0)    # Shape: [n_mels, time_frames]
        
        # 5. Aplicar compressão logarítmica (Log-Mel Spectrogram)
        # O acréscimo de 1e-5 evita o cálculo de log(0) que resultaria em menos infinito
        log_mel = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        
        # 6. Placeholder para o pipeline de texto
        # Como o VERBO foca em transferência e controle de estilo, se vocês não tiverem as 
        # transcrições em texto de 'l1', 'l2', passamos um tensor de tokens vazio ou IDs padrão.
        # Caso possuam um arquivo de transcrições, ele pode ser mapeado aqui.
        dummy_text_tokens = torch.tensor([1, 2, 3], dtype=torch.long) 
        
        return {
            "text": dummy_text_tokens,
            "mel": log_mel,
            "emotion_id": meta["emotion_id"],
            "speaker": meta["speaker"]
        }
    
    