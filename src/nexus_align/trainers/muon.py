"""
Muon optimizer, as optax 0.2.5 implements it.

References:
    - Muon:
        - Code: https://github.com/KellerJordan/modded-nanogpt
    - optax 0.2.5:
        - Official code: https://github.com/google-deepmind/optax
            - optax/contrib/_muon.py: muon, scale_by_muon,
              orthogonalize_via_newton_schulz
            - optax/_src/transform.py: scale_by_adam with nesterov=True
"""

import math

import torch
from torch.distributed.tensor import DTensor

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # quintic Newton-Schulz coefficients of optax


def orthogonalize_via_newton_schulz(
    x: torch.Tensor, steps: int, eps: float
) -> torch.Tensor:
    """Map a matrix approximately to its nearest semi-orthogonal matrix."""
    a, b, c = NS_COEFFS
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.norm(x) + eps)  # make the spectral norm at most 1
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """
    Muon for the param groups with use_muon=True, and Adam for the other groups.

    - Muon orthogonalizes the Nesterov momentum of each matrix, then scales it by
      sqrt(max(1, fan_out / fan_in)).
    - Adam uses Nesterov momentum, as the Adam part of optax.contrib.muon does.
    - Both use the bias-corrected moments of optax, the same learning rate, and no
      weight decay.
    - Muon needs the full matrices, so FSDP must not shard them (fsdp.shard: 1).
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr, betas=betas, eps=eps, momentum=momentum, ns_steps=ns_steps,
            use_muon=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Work on the local tensors. Without sharding, they are the full ones.
                param = p.to_local() if isinstance(p, DTensor) else p
                grad = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                state = self.state[p]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p)
                    if not group["use_muon"]:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"].item()
                exp_avg = state["exp_avg"]
                if isinstance(exp_avg, DTensor):
                    exp_avg = exp_avg.to_local()

                beta1 = group["momentum"] if group["use_muon"] else group["betas"][0]
                exp_avg.lerp_(grad, 1 - beta1)
                nesterov = (
                    beta1 * exp_avg / (1 - beta1 ** (step + 1))
                    + (1 - beta1) * grad / (1 - beta1 ** step)
                )
                if group["use_muon"]:
                    if param.shape != p.shape:
                        raise ValueError("❌ Muon needs full matrices; set fsdp.shard: 1")
                    update = orthogonalize_via_newton_schulz(
                        nesterov.float(), group["ns_steps"], group["eps"]
                    )
                    update *= math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    if isinstance(exp_avg_sq, DTensor):
                        exp_avg_sq = exp_avg_sq.to_local()
                    beta2 = group["betas"][1]
                    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                    denom = (exp_avg_sq / (1 - beta2 ** step)).sqrt() + group["eps"]
                    update = nesterov / denom
                param.add_(update.to(param.dtype), alpha=-group["lr"])
