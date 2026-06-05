from torch.utils.data import Dataset
import csv
import torch
from pathlib import Path

class DatasetTTSPortuguese(Dataset):

    def __init__(self, data_dir=Path("data/processed/tts-portuguese-Corpora")):
        self.data_dir = Path(data_dir)
        self.files = self._load_files_list()

    def _load_files_list(self):
        with open(self.data_dir / "mels_metadata.csv", "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.files[idx]

        sample = torch.load(row["mel_path"])

        return {
            "mel": sample["mel"],
            "text": row["text"],
            "duration": float(row["duration"]),
            "waveform": sample["waveform"],
        }