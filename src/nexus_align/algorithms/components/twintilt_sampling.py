# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Portions adapted from the local finite-difference-flow-optimization source.
# Licensed under the NVIDIA Source Code License v1 (Non-Commercial).
# See licenses/FDFO-LICENSE.txt for the full license.
"""
Euler and AWM 2/2 PEC SA rollouts with source arithmetic and RNG ordering.

Adapted from fdfo/sampling.py and fdfo/sa_solver.py. The SA implementation
is derived from advantage_weighted_matching (Apache-2.0). See
licenses/FDFO-LICENSE.txt and licenses/AWM-LICENSE.txt.
"""

from dataclasses import dataclass

import torch


@dataclass
class RolloutOutput:
    sample: torch.Tensor
    model_inputs: list[torch.Tensor]
    velocities: list[torch.Tensor]
    model_timesteps: torch.Tensor


@torch.no_grad()
def sample_euler(predict_velocity, sample, timesteps, sigmas, churn_weight):
    """Record pre-step model states; consume no step noise at zero churn or sigma."""
    if timesteps.ndim != 1 or len(timesteps) < 1:
        raise ValueError("Euler requires at least one model timestep")
    if sigmas.ndim != 1 or len(sigmas) != len(timesteps) + 1:
        raise ValueError("Euler requires one terminal sigma after the model steps")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("Euler sigmas must be finite")
    if not bool((sigmas[:-1] > sigmas[1:]).all()) or sigmas[-1] != 0:
        raise ValueError("Euler sigmas must decrease to zero")
    sigma64 = sigmas.to(device=sample.device, dtype=torch.float64)
    gamma = torch.full((sample.shape[0],), churn_weight, device=sample.device)
    states, velocities = [], []
    for index, timestep in enumerate(timesteps):
        velocity = predict_velocity(sample, timestep)
        if velocity.shape != sample.shape:
            raise ValueError("Euler velocity and state shapes must match")
        states.append(sample.detach())
        velocities.append(velocity.detach())
        current, following = sigmas[index].item(), sigmas[index + 1].item()
        value, velocity = sample.float(), velocity.float()
        if churn_weight > 0 and following != 0:
            overshoot = following / (-gamma * following + gamma + 1)
            broadcast = lambda x: x[:, None, None, None]
            value = value + broadcast(overshoot - current) * velocity
            noise_scale = (
                overshoot * ((gamma + 1) ** 2 - 1) ** 0.5
                / (gamma * overshoot + 1)
            )
            scale = 1 / (gamma * overshoot + 1)
            before = broadcast(scale) * value
            sample = before + broadcast(noise_scale) * torch.randn_like(value)
        else:
            dt = (sigma64[index + 1] - sigma64[index]).float()
            sample = value + dt * velocity
    return RolloutOutput(sample, states, velocities, timesteps.detach())


def _as_floating_tensor(value: torch.Tensor | float) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor


def _validate_churn(churn: torch.Tensor) -> None:
    if not bool(torch.isfinite(churn).all()):
        raise ValueError("SA-Solver churn values must be finite")
    if bool((churn < 0).any()):
        raise ValueError("SA-Solver churn values must be non-negative")


def edm_matched_noise_std(
    next_sigma: torch.Tensor | float,
    churn: torch.Tensor | float,
) -> torch.Tensor:
    """Return EDM's absolute fresh-noise standard deviation."""

    churn_tensor = _as_floating_tensor(churn)
    _validate_churn(churn_tensor)
    next_sigma_tensor = torch.as_tensor(
        next_sigma,
        device=churn_tensor.device,
        dtype=churn_tensor.dtype,
    )
    variance = -torch.expm1(-2.0 * torch.log1p(churn_tensor))
    return next_sigma_tensor * torch.sqrt(torch.clamp(variance, min=0.0))


def sa_tau_from_noise_std(
    current_sigma: torch.Tensor | float,
    next_sigma: torch.Tensor | float,
    noise_std: torch.Tensor | float,
    *,
    representability_tolerance: float = 1e-10,
) -> torch.Tensor:
    """
    Solve for SA ``tau`` that realizes an absolute target noise std.

    For ``h = lambda(next_sigma) - lambda(current_sigma)``, SA adds noise
    with standard deviation ``next_sigma * sqrt(1 - exp(-2*tau**2*h))``.
    The final transition to zero has no fresh noise and returns ``tau=0``.
    """

    target_std = _as_floating_tensor(noise_std)
    current = torch.as_tensor(
        current_sigma,
        device=target_std.device,
        dtype=target_std.dtype,
    )
    target = torch.as_tensor(
        next_sigma,
        device=target_std.device,
        dtype=target_std.dtype,
    )
    current, target, target_std = torch.broadcast_tensors(
        current,
        target,
        target_std,
    )
    if not bool(torch.isfinite(target_std).all()):
        raise ValueError("SA-Solver target noise standard deviations must be finite")
    if bool((target_std < 0).any()):
        raise ValueError(
            "SA-Solver target noise standard deviations must be non-negative"
        )
    if bool(((current <= target) | (current >= 1) | (target < 0)).any()):
        raise ValueError(
            "SA-Solver requires 1 > current_sigma > next_sigma >= 0"
        )

    final = target == 0
    final_has_noise = final & (
        target_std > torch.as_tensor(
            representability_tolerance,
            device=target_std.device,
            dtype=target_std.dtype,
        )
    )
    if bool(final_has_noise.any()):
        raise ValueError(
            "A transition to sigma=0 cannot represent non-zero fresh noise"
        )

    safe_target = torch.where(final, torch.ones_like(target), target)
    noise_ratio_squared = (target_std / safe_target).square()
    impossible = (~final) & (
        noise_ratio_squared > 1.0 + representability_tolerance
    )
    if bool(impossible.any()):
        maximum_ratio = torch.sqrt(noise_ratio_squared[impossible]).max().item()
        raise ValueError(
            "EDM fresh noise cannot be represented on the AWM sigma grid: "
            f"max noise_std / next_sigma = {maximum_ratio:.8g} > 1"
        )
    singular = (~final) & (noise_ratio_squared >= 1.0)
    if bool(singular.any()):
        raise ValueError(
            "EDM fresh noise would require an infinite SA-Solver tau"
        )

    safe_current = torch.where(final, torch.full_like(current, 0.75), current)
    safe_target = torch.where(final, torch.full_like(target, 0.25), target)
    flow_lambda = lambda sigma: torch.log1p(-sigma) - torch.log(sigma)
    h = flow_lambda(safe_target) - flow_lambda(safe_current)
    tau_squared = -torch.log1p(-noise_ratio_squared.clamp(min=0.0)) / (2.0 * h)
    tau_squared = torch.where(final, torch.zeros_like(tau_squared), tau_squared)
    return torch.sqrt(torch.clamp(tau_squared, min=0.0))


def build_awm_sigma_schedule(sigmas: torch.Tensor) -> torch.Tensor:
    """Clone a scheduler grid and apply AWM's exact endpoint replacement."""

    if sigmas.ndim != 1 or sigmas.numel() < 3:
        raise ValueError("AWM SA-Solver requires at least two solver steps")
    awm_sigmas = sigmas.detach().clone().float()
    # Preserve the reference implementation's assignment order.  This matters
    # for the degenerate two-step grid, where -3 and 0 are the same element.
    awm_sigmas[-2] = awm_sigmas[-3] / 2.0
    awm_sigmas[0] = 1.0 - 1e-3
    awm_sigmas[-1] = 0.0
    if bool((awm_sigmas[:-1] <= awm_sigmas[1:]).any()):
        raise ValueError("AWM endpoint replacement produced a non-decreasing grid")
    return awm_sigmas


def _flow_lambda(t):
    return torch.log(1.0 - t) - torch.log(t)


class _SASolver:
    """Fixed data-prediction 2/2 PEC updates on a scalar churn schedule."""

    def __init__(self, model_fn):
        self.model = model_fn

    def model_fn(self, x, t):
        return x - t * self.model(x, t)

    def get_coefficients_exponential_positive(
        self,
        order: int,
        interval_start: torch.Tensor,
        interval_end: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        if order not in (0, 1):
            raise ValueError("2/2 PEC only requires polynomial orders 0 and 1")
        interval_end_cov = (1.0 + tau.square()) * interval_end
        interval_start_cov = (1.0 + tau.square()) * interval_start
        if order == 0:
            return torch.exp(interval_end_cov) * (
                1.0 - torch.exp(-(interval_end_cov - interval_start_cov))
            ) / (1.0 + tau.square())
        return torch.exp(interval_end_cov) * (
            (interval_end_cov - 1.0)
            - (interval_start_cov - 1.0)
            * torch.exp(-(interval_end_cov - interval_start_cov))
        ) / (1.0 + tau.square()).square()

    def lagrange_polynomial_coefficient(
        self,
        order: int,
        lambda_list: list[torch.Tensor],
    ) -> list[list[torch.Tensor | int]]:
        if order == 0:
            return [[1]]
        if order != 1 or len(lambda_list) != 2:
            raise ValueError(
                "TwinTilt SA-Solver only supports predictor orders 1 and 2"
            )
        denominator = lambda_list[0] - lambda_list[1]
        return [
            [1.0 / denominator, -lambda_list[1] / denominator],
            [-1.0 / denominator, lambda_list[0] / denominator],
        ]

    def get_coefficients_fn(
        self,
        order: int,
        interval_start: torch.Tensor,
        interval_end: torch.Tensor,
        lambda_list: list[torch.Tensor],
        tau: torch.Tensor,
    ) -> list[torch.Tensor]:
        if order not in (1, 2) or order != len(lambda_list):
            raise ValueError(
                "TwinTilt SA-Solver only supports integration orders 1 and 2"
            )
        lagrange = self.lagrange_polynomial_coefficient(order - 1, lambda_list)
        coefficients = []
        for i in range(order):
            coefficient = sum(
                lagrange[i][j]
                * self.get_coefficients_exponential_positive(
                    order - 1 - j,
                    interval_start,
                    interval_end,
                    tau,
                )
                for j in range(order)
            )
            coefficients.append(coefficient)
        return coefficients

    def adams_bashforth_update_few_steps(
        self,
        order: int,
        x: torch.Tensor,
        tau: torch.Tensor | float,
        model_prev_list: list[torch.Tensor],
        t_prev_list: list[torch.Tensor],
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if order not in (1, 2):
            raise ValueError("TwinTilt SA-Solver predictor order must be 1 or 2")
        tau = torch.as_tensor(tau, device=x.device, dtype=t.dtype)
        sigma_t = t
        lambda_t = _flow_lambda(t)
        sigma_prev = t_prev_list[-1]
        h = lambda_t - _flow_lambda(t_prev_list[-1])
        lambda_list = [
            _flow_lambda(t_prev_list[-(i + 1)]) for i in range(order)
        ]
        gradient_coefficients = self.get_coefficients_fn(
            order,
            _flow_lambda(t_prev_list[-1]),
            lambda_t,
            lambda_list,
            tau,
        )

        if order == 2:
            denominator = (
                _flow_lambda(t_prev_list[-1])
                - _flow_lambda(t_prev_list[-2])
            )
            rescale = torch.exp((1.0 + tau.square()) * lambda_t) * (
                h.square() / 2.0
                - (
                    h * (1.0 + tau.square())
                    - 1.0
                    + torch.exp((1.0 + tau.square()) * (-h))
                )
                / (1.0 + tau.square()).square()
            ) / denominator
            gradient_coefficients[0] += rescale
            gradient_coefficients[1] -= rescale

        gradient_part = torch.zeros_like(x)
        for index in range(order):
            gradient_part += (
                (1.0 + tau.square())
                * sigma_t
                * torch.exp(-tau.square() * lambda_t)
                * gradient_coefficients[index]
                * model_prev_list[-(index + 1)]
            )
        noise_part = (
            sigma_t
            * torch.sqrt(1.0 - torch.exp(-2.0 * tau.square() * h))
            * noise
        )
        return (
            torch.exp(-tau.square() * h) * (sigma_t / sigma_prev) * x
            + gradient_part
            + noise_part
        ).to(dtype=x.dtype)

    def adams_moulton_update_few_steps(
        self,
        order: int,
        x: torch.Tensor,
        tau: torch.Tensor | float,
        model_prev_list: list[torch.Tensor],
        t_prev_list: list[torch.Tensor],
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if order not in (1, 2):
            raise ValueError("TwinTilt SA-Solver corrector order must be 1 or 2")
        tau = torch.as_tensor(tau, device=x.device, dtype=t.dtype)
        sigma_t = t
        lambda_t = _flow_lambda(t)
        sigma_prev = t_prev_list[-1]
        h = lambda_t - _flow_lambda(t_prev_list[-1])
        t_list = [*t_prev_list, t]
        lambda_list = [
            _flow_lambda(t_list[-(i + 1)]) for i in range(order)
        ]
        gradient_coefficients = self.get_coefficients_fn(
            order,
            _flow_lambda(t_prev_list[-1]),
            lambda_t,
            lambda_list,
            tau,
        )

        if order == 2:
            rescale = torch.exp((1.0 + tau.square()) * lambda_t) * (
                h / 2.0
                - (
                    h * (1.0 + tau.square())
                    - 1.0
                    + torch.exp((1.0 + tau.square()) * (-h))
                )
                / ((1.0 + tau.square()).square() * h)
            )
            gradient_coefficients[0] += rescale
            gradient_coefficients[1] -= rescale

        gradient_part = torch.zeros_like(x)
        for index in range(order):
            gradient_part += (
                (1.0 + tau.square())
                * sigma_t
                * torch.exp(-tau.square() * lambda_t)
                * gradient_coefficients[index]
                * model_prev_list[-(index + 1)]
            )
        noise_part = (
            sigma_t
            * torch.sqrt(1.0 - torch.exp(-2.0 * tau.square() * h))
            * noise
        )
        return (
            torch.exp(-tau.square() * h) * (sigma_t / sigma_prev) * x
            + gradient_part
            + noise_part
        ).to(dtype=x.dtype)

    @torch.no_grad()
    def sample(self, x, sigmas, taus):
        # Preserve AWM's unused initial/final draws, including at zero churn.
        torch.randn_like(x)
        times = [sigmas[0]]
        predictions = [self.model_fn(x, sigmas[0])]
        for step in range(1, len(sigmas)):
            t = sigmas[step]
            noise = torch.randn_like(x)
            if step == len(sigmas) - 1:
                return predictions[-1]
            tau = taus[step - 1]
            predicted = self.adams_bashforth_update_few_steps(
                min(2, step), x, tau, predictions, times, noise, t,
            )
            predictions.append(self.model_fn(predicted, t))
            # AWM disables the first corrector during warm-up.
            x = predicted if step == 1 else self.adams_moulton_update_few_steps(
                2, x, tau, predictions, times, noise, t,
            )
            times.append(t)
            if step >= 2:
                del predictions[0]


def _validate_solver_inputs(
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    churn_scales: torch.Tensor,
) -> None:
    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("AWM SA-Solver requires at least two timesteps")
    if sigmas.ndim != 1 or sigmas.numel() != timesteps.numel() + 1:
        raise ValueError(
            "SA-Solver sigmas must be one-dimensional with one more entry "
            "than timesteps"
        )
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("SA-Solver sigmas must be finite")
    if bool(((sigmas < 0) | (sigmas > 1)).any()):
        raise ValueError("SA-Solver sigmas must lie in [0, 1]")
    if bool((sigmas[:-1] <= sigmas[1:]).any()):
        raise ValueError("SA-Solver sigmas must be strictly decreasing")
    if not torch.isclose(sigmas[-1], sigmas.new_zeros(()), atol=1e-12, rtol=0):
        raise ValueError("SA-Solver requires a final sigma of zero")
    scheduler_timesteps = sigmas[:-1].to(
        device=timesteps.device,
        dtype=timesteps.dtype,
    ) * 1000.0
    if not torch.allclose(timesteps, scheduler_timesteps, atol=1e-4, rtol=1e-5):
        raise ValueError(
            "SA-Solver timesteps must use the scheduler's original sigma grid"
        )
    _validate_churn(churn_scales)


@torch.no_grad()
def sample_sa(predict_velocity, sample, timesteps, sigmas, churn_weight):
    """Record actual predictor inputs while matching original-grid churn noise."""
    original_sigmas = sigmas.detach().clone().to(
        device=sample.device, dtype=torch.float64,
    )
    # Match the source's FP32 churn schedule before its FP64 tau calculation.
    churn = torch.full(
        (len(timesteps),), churn_weight, device=sample.device,
        dtype=torch.float32,
    ).double()
    _validate_solver_inputs(timesteps, original_sigmas, churn)
    awm_sigmas = build_awm_sigma_schedule(original_sigmas).to(sample.device)
    noise_stds = edm_matched_noise_std(original_sigmas[1:], churn)
    taus = sa_tau_from_noise_std(
        awm_sigmas[:-1].double(), awm_sigmas[1:].double(), noise_stds,
    ).to(awm_sigmas.dtype)
    states, velocities, times = [], [], []

    def record(model_input, normalized_timestep):
        timestep = normalized_timestep * 1000.0
        velocity = predict_velocity(model_input, timestep)
        if velocity.shape != model_input.shape:
            raise ValueError("SA velocity and state shapes must match")
        states.append(model_input.detach())
        velocities.append(velocity.detach())
        times.append(timestep)
        return velocity

    result = _SASolver(record).sample(sample, awm_sigmas, taus)
    if len(times) != len(timesteps):
        raise RuntimeError("SA model-evaluation count must match num_steps")
    return RolloutOutput(result, states, velocities, torch.stack(times))
