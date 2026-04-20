from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.numel() == 0:
        return prediction.new_tensor(0.0)
    valid = mask > 0
    if not torch.any(valid):
        return prediction.new_tensor(0.0)
    diff = prediction[valid] - target[valid]
    return (diff ** 2).mean()


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask > 0
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid])


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    valid = available > 0
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[valid], target[valid])
