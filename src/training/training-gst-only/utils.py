import torch
import torch.nn as nn
import torch.nn.functional as F

def build_dataset(config: TrainingConfig) -> Dataset:
    """Build dataset for training."""
    # This is a placeholder implementation - actual dataset building
    # would depend on the specific data loading requirements
    from src.data import CombinedFirstStepDataset
    
    return CombinedFirstStepDataset(
        processed_root=config.processed_root,
        split="train",
        max_seq_len=config.max_seq_len
    )
