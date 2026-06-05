from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
from data.loader_TTS_GST.DataSet import DatasetTTSPortuguese
from torch.utils.data import DataLoader, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_dataset(root = PROJECT_ROOT / "data" / "processed" / "tts-portuguese-Corpora"):
    return DatasetTTSPortuguese(data_dir=root)


def tts_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Função customizada para juntar o batch. 
    Aplica padding (zeros) nos Mel-spectrogramas e Waveforms 
    para que todos tenham o tamanho do maior elemento do batch.
    """
    texts = [item["text"] for item in batch]
    srs = [item.get("sr", 22050) for item in batch]

    # Processar os Mel-Spectrogramas (Shape esperado: [..., Frequência, Tempo])
    mels = [item["mel"] for item in batch]
    max_mel_len = max(m.size(-1) for m in mels)
    
    padded_mels = []
    for m in mels:
        # Calcula quanto falta para chegar no tamanho máximo
        pad_amount = max_mel_len - m.size(-1)
        # Aplica padding apenas na última dimensão (Tempo), à direita
        m_pad = F.pad(m, (0, pad_amount), value=0.0)
        padded_mels.append(m_pad)
        
    mels_tensor = torch.stack(padded_mels)

    # Processar os Waveforms (Áudio Bruto), se estiverem no dataset
    waveforms_tensor = None
    if "waveform" in batch[0] and batch[0]["waveform"] is not None:
        waveforms = [item["waveform"] for item in batch]
        max_wav_len = max(w.size(-1) for w in waveforms)
        
        padded_wavs = []
        for w in waveforms:
            pad_amount = max_wav_len - w.size(-1)
            w_pad = F.pad(w, (0, pad_amount), value=0.0)
            padded_wavs.append(w_pad)
            
        waveforms_tensor = torch.stack(padded_wavs)

    return {
        "text": texts,            
        "mel": mels_tensor,        
        "waveform": waveforms_tensor, 
        "sr": srs              
    }


def create_dataloaders(dataset, batch_size: int = 32, num_workers: int = 4, split_ratio: float = 0.1, seed: int = 42):
    total_size = len(dataset)
    val_size = int(total_size * split_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )

    # Adicionando a collate_fn customizada
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=tts_collate_fn
    )
    
    # Shuffle False na validação
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=tts_collate_fn
    )

    return train_loader, val_loader


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