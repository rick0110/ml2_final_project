from torch.utils.data import Dataset
import csv
import torch
from pathlib import Path
import re
from num2words import num2words

class TextNormalizerBR:
    def __init__(self):
        self.number_pattern = re.compile(r'\d+')
        
    def _replace_numbers(self, match):
        number = int(match.group(0))
        return num2words(number, lang='pt-BR')

    def normalize(self, text: str) -> str:
        text = self.number_pattern.sub(self._replace_numbers, text)
        replaces = {
            '–': '-',
            '—': '-',
            '−': '-',
            '·': '',
            'ı': 'õ',
        }
        for old_char, new_char in replaces.items():
            text = text.replace(old_char, new_char)
        return text

class DatasetTTSPortuguese(Dataset):
    def __init__(self, data_dir=Path("data/processed/tts-portuguese-Corpora")):
        self.data_dir = Path(data_dir)
        self.files = self._load_files_list()
        self.normalizer = TextNormalizerBR()

    def _load_files_list(self):
        with open(self.data_dir / "mels_metadata.csv", "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.files[idx]
        sample = torch.load(row["mel_path"], weights_only=False)
        
        clean_text = self.normalizer.normalize(row["text"])

        return {
            "mel": sample["mel"],
            "mel_normalized": sample["mel_normalized"],
            "text": clean_text,
            "duration": float(row["duration"]),
            "waveform": sample["waveform"],
        }