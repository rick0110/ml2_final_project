import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignAndCutL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mapped_mel: torch.Tensor, target_mel: torch.Tensor):
        min_time = min(mapped_mel.size(2), target_mel.size(2))
        
        # Corta o excesso de quem for maior
        mapped_cut = mapped_mel[:, :, :min_time]
        target_cut = target_mel[:, :, :min_time]
        
        # Calcula L1 e retorna
        loss = F.l1_loss(mapped_cut, target_cut)
        return loss, mapped_cut, target_cut