# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Portions adapted from the local finite-difference-flow-optimization source.
# Licensed under the NVIDIA Source Code License v1 (Non-Commercial).
# See licenses/FDFO-LICENSE.txt for the full license.
"""Fixed-budget nonlinear preference projection in the candidate subspace."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch


def _validate_solver_parameters(eta: float, iterations: int) -> None:
    if (not isinstance(iterations, int) or isinstance(iterations, bool)
            or iterations <= 0):
        raise ValueError("exact_projection_iterations must be a positive integer")
    if not math.isfinite(float(eta)) or float(eta) <= 0:
        raise ValueError("eta must be finite and > 0")


def _validate_ridge_parameters(ridge_scale: float, ridge_eps: float) -> None:
    if not math.isfinite(float(ridge_scale)) or float(ridge_scale) < 0:
        raise ValueError("exact_projection requires finite ridge_scale >= 0")
    if not math.isfinite(float(ridge_eps)) or float(ridge_eps) <= 0:
        raise ValueError("exact_projection requires finite ridge_eps > 0")


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"Exact projection produced nonfinite {name}")


@lru_cache(maxsize=32)
def _centered_basis(k: int, device: torch.device) -> torch.Tensor:
    # Cache only a shape/device constant, never candidate-dependent quantities.
    basis = torch.zeros(k, k - 1, device=device, dtype=torch.float64)
    for j in range(k - 1):
        scale = math.sqrt((j + 1) * (j + 2))
        basis[:j + 1, j] = 1 / scale
        basis[j + 1, j] = -(j + 1) / scale
    return basis


@dataclass(frozen=True)
class PreparedExactProjection:
    """Frozen group geometry for the preference projection."""

    centered_points: torch.Tensor
    ridge: torch.Tensor
    logits_map: torch.Tensor
    coefficient_map: torch.Tensor
    feature_shape: tuple[int, ...]


@torch.no_grad()
def prepare_exact_projection(
    points: torch.Tensor,
    *,
    ridge_scale: float = 0.001,
    ridge_eps: float = 1e-6,
) -> PreparedExactProjection:
    """Prepare FP64 endpoint geometry once per group."""
    _validate_ridge_parameters(ridge_scale, ridge_eps)
    ridge_scale, ridge_eps = float(ridge_scale), float(ridge_eps)
    if (
        points.ndim < 3
        or points.shape[0] == 0 or points.shape[1] < 2 or points.numel() == 0
    ):
        raise ValueError("Projection expects nonempty points [groups,K,...], K >= 2")
    if not points.is_floating_point():
        raise ValueError("Exact projection endpoints must be floating-point")

    with torch.autocast(device_type=points.device.type, enabled=False):
        _finite("input endpoints", points)
        d = points.double().flatten(2)
        mean = d.mean(1, keepdim=True)
        d = d - mean
        _finite("centered endpoints", d)
        groups, k, features = d.shape
        gram = torch.bmm(d, d.transpose(1, 2)) / features
        ridge = ridge_scale * gram.diagonal(dim1=1, dim2=2).mean(1) + ridge_eps
        _finite("Gram matrix", gram)
        _finite("damping", ridge)
        center = _centered_basis(k, d.device)
        reduced_gram = center.T @ gram @ center
        _finite("reduced Gram matrix", reduced_gram)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(
                0.5 * (reduced_gram + reduced_gram.transpose(1, 2))
            )
        except RuntimeError as error:
            raise RuntimeError("Projection Gram eigendecomposition failed") from error
        _finite("Gram eigenvalues", eigenvalues)
        _finite("Gram eigenvectors", eigenvectors)
        cutoff = 64 * torch.finfo(d.dtype).eps * eigenvalues.abs().amax(1, keepdim=True)
        if bool((eigenvalues < -cutoff).any()):
            raise RuntimeError(
                "Projection Gram has a negative eigenvalue beyond rounding tolerance"
            )
        keep = eigenvalues > cutoff
        roots = torch.where(
            keep, eigenvalues.clamp_min(0).sqrt(), torch.ones_like(eigenvalues)
        )
        vectors = center @ eigenvectors
        logits_map = vectors * (roots * keep).unsqueeze(1)
        coefficient_map = vectors * (keep / roots).unsqueeze(1)
        _finite("logits map", logits_map)
        _finite("coefficient map", coefficient_map)
        return PreparedExactProjection(
            d, ridge, logits_map, coefficient_map, tuple(points.shape[2:]),
        )


@torch.no_grad()
def solve_prepared_exact_projection(
    prepared: PreparedExactProjection,
    advantages: torch.Tensor,
    eta: float,
    *,
    iterations: int = 5,
) -> torch.Tensor:
    """Fit one FP64 correction per group to the regularized preference KL."""
    _validate_solver_parameters(eta, iterations)
    eta = float(eta)
    d = prepared.centered_points
    groups, k, _ = d.shape
    if advantages.shape != (groups, k):
        raise ValueError("Exact projection advantages must have shape [groups,K]")
    if advantages.device != d.device or not advantages.is_floating_point():
        raise ValueError(
            "Projection rewards must be floating-point on the endpoint device"
        )
    with torch.autocast(device_type=d.device.type, enabled=False):
        _finite("input rewards", advantages)
        a = advantages.double() - advantages.double().mean(1, keepdim=True)
        _finite("centered rewards", a)
        logits_map = prepared.logits_map
        coefficient_map, ridge = prepared.coefficient_map, prepared.ridge
        batch = groups
        y = d.new_zeros(batch, k - 1)
        tilted = eta * a
        logq = tilted.log_softmax(1)
        _finite("tilted log probabilities", logq)
        q = logq.exp()
        eye = torch.eye(k - 1, device=d.device, dtype=d.dtype).expand(batch, -1, -1)
        lt = logits_map.transpose(1, 2)

        def logits(value: torch.Tensor) -> torch.Tensor:
            result = torch.bmm(logits_map, value.unsqueeze(2)).squeeze(2)
            return result

        def objective(value: torch.Tensor) -> torch.Tensor:
            logp = logits(value).log_softmax(1)
            return (q * (logq - logp)).sum(1) + ridge / (2 * k) * value.square().sum(1)

        for _ in range(iterations):
            p = logits(y).softmax(1)
            gradient = (
                k * torch.bmm(lt, (p - q).unsqueeze(2)).squeeze(2) + ridge[:, None] * y
            )
            _finite("Newton gradient", gradient)
            mean = torch.bmm(lt, p.unsqueeze(2)).squeeze(2)
            covariance = (
                torch.bmm(lt, p[:, :, None] * logits_map)
                - mean[:, :, None] * mean[:, None, :]
            )
            hessian = k * covariance + ridge[:, None, None] * eye
            _finite("Newton Hessian", hessian)
            try:
                delta = torch.linalg.solve(hessian, gradient.unsqueeze(2)).squeeze(2)
            except RuntimeError as error:
                raise RuntimeError("Projection Newton matrix solve failed") from error
            _finite("Newton update", delta)
            slope = (gradient * delta).sum(1) / k
            _finite("Newton slope", slope)
            if bool(((slope < 0) | ((slope == 0) & (gradient != 0).any(dim=1))).any()):
                raise RuntimeError("Exact projection Newton direction is not descent")
            before = objective(y)
            _finite("current objective", before)
            slack = 8 * torch.finfo(d.dtype).eps * before.abs().clamp_min(1.)
            rate = d.new_ones(batch)
            accepted = torch.zeros(batch, device=d.device, dtype=torch.bool)
            for _ in range(50):
                candidate = y - rate[:, None] * delta
                _finite("trial update", candidate)
                value = objective(candidate)
                accepted = accepted | (
                    torch.isfinite(value)
                    & (value <= before - 1e-4 * rate * slope + slack)
                )
                if bool(accepted.all()):
                    break
                rate = torch.where(accepted, rate, rate * 0.5)
            else:
                raise RuntimeError("Exact projection Newton line search failed")
            y = y - rate[:, None] * delta
            _finite("accepted update", y)

        coefficients = torch.bmm(coefficient_map, y.unsqueeze(2))
        _finite("final coefficients", coefficients)
        u = torch.bmm(d.transpose(1, 2), coefficients).squeeze(2)
        _finite("correction", u)
        return u.reshape(groups, *prepared.feature_shape)


@torch.no_grad()
def solve_fixed_exact_projection(
    points: torch.Tensor,
    advantages: torch.Tensor,
    eta: float,
    *,
    iterations: int = 5,
    ridge_scale: float = 0.001,
    ridge_eps: float = 1e-6,
) -> torch.Tensor:
    """Return FP64 group corrections after exactly N Newton updates."""
    prepared = prepare_exact_projection(
        points, ridge_scale=ridge_scale, ridge_eps=ridge_eps
    )
    return solve_prepared_exact_projection(
        prepared, advantages, eta, iterations=iterations,
    )
