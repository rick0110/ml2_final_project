import csv
import re
import torch
from pathlib import Path
from torch.utils.data import Dataset

try:
    from num2words import num2words
except ImportError:
    num2words = None

class TextNormalizerEN:
    def __init__(self):
        self.number_pattern = re.compile(r'\d+')
        
    def _replace_numbers(self, match):
        if num2words is None:
            return match.group(0)
        number = int(match.group(0))
        return num2words(number, lang='en')

    def normalize(self, text: str) -> str:
        text = self.number_pattern.sub(self._replace_numbers, text)
        replaces = {'–': '-', '—': '-', '−': '-', '·': '', '"': '', '\'': ''}
        for old_char, new_char in replaces.items():
            text = text.replace(old_char, new_char)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()


class DatasetLibriSpeechTacotronVAE(Dataset):
    def __init__(self, text_processor, data_dir=Path("data/processed/libriSpeech-en-tacotron-vae")):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "librispeech_mels_metadata.csv"
        self.normalizer = TextNormalizerEN()
        self.text_processor = text_processor # Processador que transforma texto em IDs
        self.files = self._load_files_list()

    def _load_files_list(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Arquivo de metadados não encontrado: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.files[idx]
        sample = torch.load(row["mel_path"], map_location="cpu", weights_only=False)
        
        # Normaliza o texto e converte para sequência de IDs
        clean_text = self.normalizer.normalize(row["text"])
        sequence_list = self.text_processor.text_to_sequence(clean_text)
        
        # IMPORTANTE: Camadas de Embedding no PyTorch exigem LongTensor (Int64)
        text_sequence = torch.LongTensor(sequence_list)
        
        # Garante que o Mel tem 2 dimensões [80, T]
        mel_tensor = sample["mel"].squeeze(0) if sample["mel"].dim() == 3 else sample["mel"]

        # Tacotron2-VAE espera tensores one-hot para locutor e emoção.
        # Estamos a usar 1 locutor e 4 emoções de acordo com os seus hparams padrão.
        speaker = torch.zeros(1, dtype=torch.float32)
        speaker[0] = 1.0  # Ativa o locutor padrão
        
        emotion = torch.zeros(4, dtype=torch.float32)
        emotion[0] = 1.0  # Ativa a emoção 0 (Neutra)

        # Retorna exatamente a tupla que o utils.py e o TextMelCollate esperam:
        # [0] = text, [1] = mel, [2] = speaker, [3] = emotion
        return text_sequence, mel_tensor, speaker, emotion