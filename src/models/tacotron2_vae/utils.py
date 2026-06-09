import torch


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = ids < lengths.unsqueeze(1)
    return mask


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.contiguous().to(device, non_blocking=True)
