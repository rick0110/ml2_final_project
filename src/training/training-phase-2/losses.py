from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_loss(
    z_post: torch.Tensor,
    post_mean: torch.Tensor,
    post_log_std: torch.Tensor,
    z_prior: torch.Tensor,
    log_det: torch.Tensor,
) -> torch.Tensor:
    log_q = -0.5 * (
        ((z_post - post_mean) ** 2) * torch.exp(-2.0 * post_log_std)
        + 2.0 * post_log_std
        + torch.log(torch.tensor(2.0 * torch.pi, device=z_post.device))
    )
    log_q = log_q.sum(dim=(1, 2))

    log_p = -0.5 * (
        z_prior**2 + torch.log(torch.tensor(2.0 * torch.pi, device=z_prior.device))
    )
    log_p = log_p.sum(dim=(1, 2))

    kl = (log_q - log_p - log_det).mean()
    return kl


def reconstruction_loss(predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(predicted_mel, target_mel)


def info_nce_loss(
    style_ref: torch.Tensor,
    style_gen: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    if style_ref.size() != style_gen.size():
        raise ValueError("style_ref and style_gen must have the same shape")

    ref_norm = F.normalize(style_ref, dim=-1)
    gen_norm = F.normalize(style_gen, dim=-1)
    logits = torch.mm(ref_norm, gen_norm.t()) / max(temperature, 1e-6)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def discriminator_loss(
    real_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    fake_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    loss = 0.0
    for real_pred, _ in real_outputs:
        loss = loss + torch.mean((real_pred - 1.0) ** 2)
    for fake_pred, _ in fake_outputs:
        loss = loss + torch.mean(fake_pred**2)
    return loss


def generator_adv_loss(fake_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]]) -> torch.Tensor:
    loss = 0.0
    for fake_pred, _ in fake_outputs:
        loss = loss + torch.mean((fake_pred - 1.0) ** 2)
    return loss


def feature_matching_loss(
    real_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    fake_outputs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    loss = 0.0
    for (_, real_features), (_, fake_features) in zip(real_outputs, fake_outputs):
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss = loss + F.l1_loss(fake_feat, real_feat.detach())
    return loss
