from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from data.loader_TTS_GST.DataSet import DatasetTTSPortuguese
from torch.utils.data import DataLoader, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]

def create_dataset(root = PROJECT_ROOT / "data" / "processed" / "tts-portuguese-Corpora"):
    return DatasetTTSPortuguese(data_dir=root)

def create_dataloader(dataset, batch_size: int = 32, num_workers: int = 32, split_ratio: int = 0.1, seed: int = 42):
    total_size = len(dataset)
    val_size = int(total_size * split_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    return DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers), DataLoader(val_dataset, batch_size, num_workers=num_workers)


def create_experiment_dir(experiment_name: Optional[str] = None) -> Path:
    experiments_root = PROJECT_ROOT / "experiments" / "step_1"
    experiments_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = experiments_root / (experiment_name or f"attempt_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "tensorboard").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    return experiment_dir
