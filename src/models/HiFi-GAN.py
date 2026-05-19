from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
import pathlib
import torch

ROOT = pathlib.Path(__file__).parent.parent.parent

def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

def load_hifigan_model(path: pathlib.Path = ROOT / "local_weight_models" / "HiFi-GAN", freeze: bool = True):
    if path.exists():
        spec_generator = FastPitchModel.from_pretrained(str(path))
        model = HifiGanModel.from_pretrained(str(path))
        if freeze:
            freeze_model(model)
        return spec_generator, model
    spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
    model = HifiGanModel.from_pretrained("nvidia/tts_hifigan")
    if freeze:
        freeze_model(model)
        freeze_model(spec_generator)
    return spec_generator, model
