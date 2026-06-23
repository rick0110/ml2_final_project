
"""
WaveGlow training script.

Responsibilities:
    - Load configurations and initialize distributed training (if multi-GPU).
    - Setup Mel2Samp dataset and DataLoaders.
    - Initialize WaveGlow model, NLL loss, and Adam optimizer.
    - Orchestrate the training loop across epochs and iterations.
    - Handle checkpoint saving and loading for model persistence.

Main Functions:
    - train: Core training routine.
    - load_checkpoint: Restore training state.
    - save_checkpoint: Persist training state.

Tensor Conventions:
    B = batch size
    T = sequence length
    n_mels = mel channels
"""

import argparse
import json
import os
import sys
import torch
from torch import Tensor
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

#=====START: ADDED FOR DISTRIBUTED======
try:
    from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
except ImportError:
    # Handle if running from different context
    from src.training.training_waveglow.distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader

ROOT_DIR: Path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "waveglow"))
sys.path.insert(0, str(ROOT_DIR / "src" / "data" / "loader_waveglow"))

try:
    from glow import WaveGlow, WaveGlowLoss
    from loader_waveglow import Mel2Samp
except ImportError:
    from src.models.waveglow.glow import WaveGlow, WaveGlowLoss
    from src.data.loader_waveglow.loader_waveglow import Mel2Samp


def load_checkpoint(
    checkpoint_path: str, 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """
    Restore model and optimizer state from a checkpoint.

    Args:
        checkpoint_path (str): Path to .pt file.
        model (Module): Model to load.
        optimizer (Optimizer): Optimizer to load.

    Returns:
        Tuple: model, optimizer, iteration.
    """
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict: Dict[str, Any] = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    iteration: int = checkpoint_dict.get('iteration', 0)
    if 'optimizer' in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    
    # Handle if checkpoint directly contains model or state_dict
    if 'model' in checkpoint_dict:
        model_for_loading = checkpoint_dict['model']
        state_dict = model_for_loading.state_dict() if hasattr(model_for_loading, 'state_dict') else model_for_loading
    elif 'state_dict' in checkpoint_dict:
        state_dict = checkpoint_dict['state_dict']
    else:
        state_dict = checkpoint_dict
        
    # Remove 'module.' prefix from distributed training
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    print("Loaded checkpoint '{}' (iteration {})" .format(checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    learning_rate: float, 
    iteration: int, 
    filepath: str
) -> None:
    """
    Persist training state.

    Args:
        model (Module): Current model.
        optimizer (Optimizer): Current optimizer.
        learning_rate (float): Current LR.
        iteration (int): Current step.
        filepath (str): Save path.
    """
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))

    # We create a clean instance to save without distributed wrappers if necessary
    model_for_saving: WaveGlow = WaveGlow(**waveglow_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model_for_saving,
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate
    }, filepath)


def train(
    num_gpus: int, 
    rank: int, 
    group_name: str, 
    output_directory: str, 
    epochs: int, 
    learning_rate: float,
    sigma: float, 
    iters_per_checkpoint: int, 
    batch_size: int, 
    seed: int, 
    fp16_run: bool,
    checkpoint_path: str
) -> None:
    """
    Main training loop for WaveGlow.

    Args:
        num_gpus (int): GPUs available.
        rank (int): Process rank.
        group_name (str): Distributed group.
        output_directory (str): Root for artifacts.
        epochs (int): Max epochs.
        learning_rate (float): Initial LR.
        sigma (float): Prior standard deviation.
        iters_per_checkpoint (int): Save frequency.
        batch_size (int): Samples per step.
        seed (int): Random seed.
        fp16_run (bool): Mixed precision flag.
        checkpoint_path (str): Resume path.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initialize distributed
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    criterion: WaveGlowLoss = WaveGlowLoss(sigma)
    model: WaveGlow = WaveGlow(**waveglow_config).cuda()

    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    iteration: int = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1

    trainset: Mel2Samp = Mel2Samp(**data_config)
    train_sampler: Optional[DistributedSampler] = DistributedSampler(trainset) if num_gpus > 1 else None

    # Set shuffle depending on sampler presence
    shuffle = (train_sampler is None)

    train_loader: DataLoader = DataLoader(
        trainset, 
        num_workers=1, 
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=True
    )

    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    model.train()
    epoch_offset: int = max(0, int(iteration / len(train_loader)))

    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            mel: Tensor
            audio: Tensor
            mel, audio = batch
            mel = mel.cuda().requires_grad_()
            audio = audio.cuda().requires_grad_()

            outputs: Tuple[Tensor, List[Tensor], List[Tensor]] = model((mel, audio))

            loss: Tensor = criterion(outputs)
            reduced_loss: float
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                     scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            print("{}:\t{:.9f}".format(iteration, reduced_loss))

            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_save_path: Path = Path(args.checkpoints_dir) / f"epoch_{iteration}.pt"
                    save_checkpoint(model, optimizer, learning_rate, iteration, str(checkpoint_save_path))

            iteration += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument('--checkpoints_dir', type=str, default=str(ROOT_DIR / "experiments"/ "waveglow" / "checkpoints" ), help='directory to save checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default="local_weight_models/waveglow/nvidia_waveglowpyt_fp32_20190427", help='checkpoint path')
    args: argparse.Namespace = parser.parse_args()

    with open(str(Path(args.config).resolve())) as f:
        config_data: str = f.read()
    config: Dict[str, Any] = json.loads(config_data)

    train_config: Dict[str, Any] = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    num_gpus_count: int = torch.cuda.device_count()
    if num_gpus_count > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus_count = 1

    if num_gpus_count == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    
    # Ensure args.checkpoint_path takes precedence or is used directly if specified
    if args.checkpoint_path:
        train_config["checkpoint_path"] = args.checkpoint_path
        
    train(num_gpus_count, args.rank, args.group_name, **train_config)