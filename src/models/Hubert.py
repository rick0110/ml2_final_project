from transformers import AutoProcessor, AutoModelForCTC
import torch
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.parent

def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

def load_hubert_model(path: pathlib.Path = ROOT / "local_weight_models" / "hubert", freeze: bool = True):
    if path.exists():
        processor = AutoProcessor.from_pretrained(str(path))
        model = AutoModelForCTC.from_pretrained(str(path))
        if freeze:
            freeze_model(model)
        return processor, model
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    if freeze:
        freeze_model(model)
    return processor, model


