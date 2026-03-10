"""Random seed and deterministic utilities."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set the random seed for random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"✅ Set seed to {seed}")


def set_deterministic(deterministic: bool = True) -> None:
    """Enable or disable deterministic behavior for reproducibility."""
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)

    enable = "Enabled" if deterministic else "Disabled"
    print(f"✅ Deterministic mode: {enable}")
