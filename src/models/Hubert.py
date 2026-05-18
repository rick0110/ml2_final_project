from transformers import AutoProcessor, AutoModelForCTC
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.parent

def load_hubert_model(path: pathlib.Path = ROOT / "local_weight_models" / "hubert"):
    if path.exists():
        processor = AutoProcessor.from_pretrained(str(path))
        model = AutoModelForCTC.from_pretrained(str(path))
        return processor, model
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    return processor, model



