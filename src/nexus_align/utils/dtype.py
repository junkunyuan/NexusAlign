"""Dtype mapping: config string names to torch dtypes."""

import torch

DTYPE_MAP = {
    "no": None,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
