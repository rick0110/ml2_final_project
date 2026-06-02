from transformers import AutoProcessor, AutoModelForCTC
import torch
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.parent


def freeze_model(model: torch.nn.Module) -> None:
    """Freeze model parameters."""
    
    for param in model.parameters():
        param.requires_grad = False


def load_hubert_model(path: pathlib.Path = ROOT / "local_weight_models" / "hubert", freeze: bool = True):
    """Load Hubert model from local path or download it if not present.

    Args:
        path (pathlib.Path): Path to the directory containing the model files.
        freeze (bool, optional): Whether to freeze the model parameters. Default is True.

    Returns:
        tuple[AutoProcessor, AutoModelForCTC]: Processor and Hubert model for CTC tasks.
    """
    
    if path.exists():
        processor = AutoProcessor.from_pretrained(str(path / "processor"))
        model = AutoModelForCTC.from_pretrained(str(path / "model"))
        if freeze:
            freeze_model(model)
        return processor, model
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir=str(ROOT / "local_weight_models" / "hubert" / "processor"))
    model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir=str(ROOT / "local_weight_models" / "hubert" / "model"))

    if freeze:
        freeze_model(model)
    return processor, model
