import os
import torch

def load_checkpoint(
    checkpoint_path: str,
    fastpitch: torch.nn.Module,
    gst: torch.nn.Module,
    mel_decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    fastpitch.load_state_dict(checkpoint["fastpitch"])
    gst.load_state_dict(checkpoint["gst"])
    mel_decoder.load_state_dict(checkpoint["mel_decoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["epoch"]

def list_checkpoints(checkpoint_dir: str) -> List[str]:
    """List all available checkpoints in the directory."""
    return [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Inspect the contents of a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return torch.load(checkpoint_path)
