import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """TensorBoard logging utility for training."""
    
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """Log training metrics to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, step)
    
    def log_audio_examples(
        self,
        step: int,
        sample_rate: int,
        original_audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        predicted_audio: torch.Tensor,
        prefix: str = "examples/"
    ):
        """Log audio examples to TensorBoard."""
        self.writer.add_audio(f"{prefix}original", original_audio, step, sample_rate=sample_rate)
        self.writer.add_audio(f"{prefix}reconstructed", reconstructed_audio, step, sample_rate=sample_rate)
        self.writer.add_audio(f"{prefix}predicted", predicted_audio, step, sample_rate=sample_rate)
    
    def flush(self):
        """Flush TensorBoard logs."""
        self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
