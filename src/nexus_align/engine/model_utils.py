"""Model utilities for training."""

import torch


@torch.no_grad()
def compute_update_ratio(model: torch.nn.Module, prev_params: list) -> float:
    """
    Compute parameter update ratio ||Δθ|| / ||θ||.
    
    Args:
        model (`torch.nn.Module`): model to compute update ratio.
        prev_params (`list`): parameters saved from the previous step.

    Returns:
        update_ratio (`float`): update ratio.
    """
    param_norm = 0.0
    update_norm = 0.0
    idx = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            prev_p = prev_params[idx]
            param_norm += torch.sum(p.data.float() ** 2)
            update_norm += torch.sum((p.data.float() - prev_p) ** 2)
            idx += 1

    param_norm = torch.sqrt(param_norm)
    update_norm = torch.sqrt(update_norm)

    update_ratio = (update_norm / (param_norm + 1e-12)).item()
    return update_ratio


def clone_params(model):
    """Clone the model's trainable parameters for later update ratio calculation."""
    return [p.data.clone().float() for p in model.parameters() if p.requires_grad]
