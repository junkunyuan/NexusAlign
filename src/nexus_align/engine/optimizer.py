"""Optimizers and learning rate schedulers."""

import torch
from torch.optim.lr_scheduler import LambdaLR


# --------------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------------
def adamw(
    params_train: torch.nn.ParameterList, 
    kwargs: dict
) -> torch.optim.AdamW:
    """AdamW optimizer."""
    lr = kwargs["algorithm"]["optimizer"]["lr"]
    betas = kwargs["algorithm"]["optimizer"].get("betas", (0.9, 0.999))
    weight_decay = kwargs["algorithm"]["optimizer"].get("weight_decay", 0.0)

    optimizer = torch.optim.AdamW(
        params=params_train, 
        betas=betas, 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    info = f"AdamW(lr={lr}, betas={betas}, weight_decay={weight_decay})"
    print(f"✅ Prepared optimizer: {info}")

    return optimizer


get_optimizer = {
    "adamw": adamw,
}


# --------------------------------------------------------------------------------
# Learning Rate Schedulers
# --------------------------------------------------------------------------------
def constant_with_warmup(
    optimizer: torch.optim.Optimizer, 
    kwargs: dict
) -> LambdaLR:
    """Constant with warmup learning rate scheduler."""
    warmup_steps = kwargs["algorithm"]["optimizer"].get("warmup_steps", 0)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1.0, warmup_steps))
        return 1.0

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    
    info = f"ConstantWithWarmup(warmup_steps={warmup_steps})"
    print(f"✅ Prepared LR scheduler: {info}")

    return lr_scheduler


get_lr_scheduler = {
    "constant_with_warmup": constant_with_warmup,
}
