import argparse
from src.models.tacotron2_vae.hparams import create_hparams
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--iters-per-checkpoint", type=int, default=500)
    return parser.parse_args()

args = parse_arguments()
hparams = create_hparams({"epochs": args.epochs})
print(f"hparams.epochs: {hparams.epochs}")
