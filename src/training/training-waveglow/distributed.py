"""
Distributed training utilities for WaveGlow.

Responsibilities:
    - Initialize distributed process groups for multi-GPU training.
    - Synchronize gradients across multiple GPUs using all-reduce.
    - Flatten and unflatten tensors for efficient communication.
    - Orchestrate multiple worker processes for parallel training.

Main Functions:
    - init_distributed: Setup the process group.
    - apply_gradient_allreduce: Register hooks for automatic gradient synchronization.
    - reduce_tensor: Average a tensor across all processes.
    - main: Launch multiple training processes per GPU.
"""

import os
from pathlib import Path
import sys
import time
import subprocess
import argparse
from typing import List, Tuple, Iterable, Any, Optional

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch import Tensor

ROOT_DIR: Path = Path(__file__).parent.parent.parent

def reduce_tensor(tensor: Tensor, num_gpus: int) -> Tensor:
    """
    Averages a tensor across all distributed processes.

    Args:
        tensor (Tensor): Input tensor.
        num_gpus (int): Total number of GPUs in the group.

    Returns:
        Tensor: Averaged tensor.
    """
    rt: Tensor = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt

def init_distributed(rank: int, num_gpus: int, group_name: str, dist_backend: str, dist_url: str) -> None:
    """
    Initializes the distributed process group.

    Args:
        rank (int): Rank of the current process.
        num_gpus (int): Total world size.
        group_name (str): Unique name for the process group.
        dist_backend (str): Backend (e.g., 'nccl', 'gloo').
        dist_url (str): Initialization URL.
    """
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(dist_backend, init_method=dist_url,
                            world_size=num_gpus, rank=rank,
                            group_name=group_name)

def _flatten_dense_tensors(tensors: Iterable[Tensor]) -> Tensor:
    """
    Flatten dense tensors into a contiguous 1D buffer.

    Args:
        tensors (Iterable[Tensor]): Tensors of the same type.

    Returns:
        Tensor: Contiguous 1D concatenated buffer.
    """
    tensors_list: List[Tensor] = list(tensors)
    if len(tensors_list) == 1:
        return tensors_list[0].contiguous().view(-1)
    flat: Tensor = torch.cat([t.contiguous().view(-1) for t in tensors_list], dim=0)
    return flat

def _unflatten_dense_tensors(flat: Tensor, tensors: Iterable[Tensor]) -> Tuple[Tensor, ...]:
    """
    View a flat buffer using the sizes of target tensors.

    Args:
        flat (Tensor): 1D buffer.
        tensors (Iterable[Tensor]): Templates for unflattening.

    Returns:
        Tuple[Tensor, ...]: Unflattened views into the buffer.
    """
    outputs: List[Tensor] = []
    offset: int = 0
    for tensor in tensors:
        numel: int = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def apply_gradient_allreduce(module: torch.nn.Module) -> torch.nn.Module:
    """
    Injects gradient synchronization logic into a model.

    Architecture:
        Registers backward hooks that queue all-reduce operations.

    Args:
        module (nn.Module): Model instance.

    Returns:
        nn.Module: The modified model.
    """
    if not hasattr(dist, '_backend'):
        module.warn_on_half = True
    else:
        # In newer torch dist.dist_backend might be different
        module.warn_on_half = False 

    for p in module.state_dict().values():
        if not torch.is_tensor(p):
            continue
        dist.broadcast(p, 0)

    def allreduce_params() -> None:
        if hasattr(module, 'needs_reduction') and module.needs_reduction:
            module.needs_reduction = False
            buckets: Dict[Any, List[torch.nn.Parameter]] = {}
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)

            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                dist.all_reduce(coalesced)
                coalesced /= dist.get_world_size()
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    for param in list(module.parameters()):
        def allreduce_hook(*unused: Any) -> None:
            Variable._execution_engine.queue_callback(allreduce_params)
        if param.requires_grad:
            param.register_hook(allreduce_hook)

    def set_needs_reduction(self: Any, input: Any, output: Any) -> None:
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module


def main(config: str, stdout_dir: str, args_str: str) -> None:
    """
    Process launcher for distributed training.

    Args:
        config (str): Config file path.
        stdout_dir (str): Logging directory.
        args_str (str): Extra arguments.
    """
    args_list: List[str] = ['train.py']
    args_list += args_str.split(' ') if len(args_str) > 0 else []

    args_list.append('--config={}'.format(config))

    num_gpus: int = torch.cuda.device_count()
    args_list.append('--num_gpus={}'.format(num_gpus))
    args_list.append("--group_name=group_{}".format(time.strftime("%Y_%m_%d-%H%M%S")))

    if not os.path.isdir(stdout_dir):
        os.makedirs(stdout_dir)
        os.chmod(stdout_dir, 0o775)

    workers: List[subprocess.Popen] = []

    # Placeholder for rank flag
    args_list.append('--rank=0')

    for i in range(num_gpus):
        args_list[-1] = '--rank={}'.format(i)
        stdout = None if i == 0 else open(
            os.path.join(stdout_dir, "GPU_{}.log".format(i)), "w")
        print(args_list)
        p = subprocess.Popen([str(sys.executable)]+args_list, stdout=stdout)
        workers.append(p)

    for p in workers:
        p.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='JSON file for configuration')
    parser.add_argument('-s', '--stdout_dir', type=str, default=".", help='directory to save stdout logs')
    parser.add_argument('-a', '--args_str', type=str, default='', help='double quoted string with space separated key value pairs')

    launch_args: argparse.Namespace = parser.parse_args()
    main(launch_args.config, launch_args.stdout_dir, launch_args.args_str)