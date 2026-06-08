import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.loader_latent_transfer.DataSet import DatasetTTSPortuguese

def pad_sequence(sequences, padding_value):
    sequences = [seq.squeeze(0) if seq.dim() > 1 else seq for seq in sequences]
    max_len = max(seq.size(0) for seq in sequences)
    padded_seqs = [F.pad(seq, (0, max_len - seq.size(0)), value=padding_value) for seq in sequences]
    return torch.stack(padded_seqs)

def tts_collate_fn(batch):
    mels = [item["mel"] for item in batch]
    texts = [item["text"] for item in batch]
    max_mel_len = max(m.size(-1) for m in mels)
    padded_mels = [F.pad(m, (0, max_mel_len - m.size(-1)), value=0.0) for m in mels]
    return {"mel": torch.stack(padded_mels), "text": texts}

def create_dataset():
    return DatasetTTSPortuguese()

def create_dataloaders(dataset, batch_size, num_workers=4):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=tts_collate_fn, pin_memory=True)
    return train_loader, None