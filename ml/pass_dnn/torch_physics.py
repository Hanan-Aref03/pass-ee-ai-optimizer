"""Differentiable PASS physics helpers for training-time loss terms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .physics import thermal_noise_power_w
from .schema import PASS_POSITION_COLUMNS, SystemConfig


@dataclass(frozen=True)
class PassPhysicsTorchBatch:
    """Torch-native PASS simulation outputs."""

    rates: torch.Tensor
    sum_rate: torch.Tensor
    total_power: torch.Tensor
    energy_efficiency: torch.Tensor
    qos_margin: torch.Tensor
    qos_satisfied: torch.Tensor


def build_waveguide_y_positions_torch(config: SystemConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Match the MATLAB waveguide-centre placement exactly."""

    indices = torch.arange(config.num_waveguides, device=device, dtype=dtype)
    return (
        -config.area_side_m / 2.0
        + indices * config.area_side_m / config.num_waveguides
        + config.area_side_m / (2.0 * config.num_waveguides)
    )


def _reshape_inputs(inputs: torch.Tensor, config: SystemConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 2 or inputs.shape[1] < config.num_users * 2 + 1:
        raise ValueError("PASS inputs must be a 2D tensor with 7 columns.")
    users = inputs[:, : config.num_users * 2].reshape(-1, config.num_users, 2)
    qos = inputs[:, config.num_users * 2]
    return users, qos


def _reshape_outputs(outputs: torch.Tensor, config: SystemConfig) -> tuple[torch.Tensor, torch.Tensor]:
    position_dim = config.num_waveguides * config.num_pinchers
    power_dim = config.num_waveguides
    if outputs.ndim != 2 or outputs.shape[1] != position_dim + power_dim:
        raise ValueError("PASS outputs must be a 2D tensor with 12 columns.")
    positions = outputs[:, :position_dim].reshape(-1, config.num_waveguides, config.num_pinchers)
    powers = outputs[:, position_dim:]
    return positions, powers


def to_physical_pass_outputs(outputs: torch.Tensor, config: SystemConfig) -> torch.Tensor:
    """Convert normalized PASS outputs into physical coordinates and powers."""

    position_dim = config.num_waveguides * config.num_pinchers
    if outputs.ndim != 2 or outputs.shape[1] != position_dim + config.num_waveguides:
        raise ValueError("PASS outputs must be a 2D tensor with 12 columns.")

    positions = outputs[:, :position_dim] * config.position_bound_m
    powers = outputs[:, position_dim:] * config.power_budget_w
    return torch.cat([positions, powers], dim=1)


def sample_pass_feasibility_candidates(
    reference_outputs: torch.Tensor,
    config: SystemConfig,
    num_candidates: int = 1,
    position_jitter: float = 0.12,
    power_scale_min: float = 0.10,
    power_scale_max: float = 0.55,
) -> torch.Tensor:
    """Build intentionally harder PASS candidates for feasibility-head training."""

    position_dim = config.num_waveguides * config.num_pinchers
    if reference_outputs.ndim != 2 or reference_outputs.shape[1] != position_dim + config.num_waveguides:
        raise ValueError("PASS outputs must be a 2D tensor with 12 columns.")
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    repeated = reference_outputs.repeat_interleave(num_candidates, dim=0)
    positions = repeated[:, :position_dim]
    powers = repeated[:, position_dim:]

    jitter = torch.empty_like(positions).uniform_(-position_jitter, position_jitter)
    candidate_positions = torch.clamp(positions + jitter, -1.0, 1.0)

    scale = torch.empty(
        (powers.shape[0], 1),
        device=reference_outputs.device,
        dtype=reference_outputs.dtype,
    ).uniform_(power_scale_min, power_scale_max)
    candidate_powers = torch.clamp(powers * scale, 0.0, 1.0)

    return torch.cat([candidate_positions, candidate_powers], dim=1)


def evaluate_pass_batch_torch(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    config: SystemConfig,
) -> PassPhysicsTorchBatch:
    """Vectorized PASS evaluation — fully batched, no per-sample loop.

    Replaces the original per-sample for-loop that caused OOM on large batches
    by accumulating thousands of small autograd nodes during backward().
    """

    users, qos = _reshape_inputs(inputs, config)
    positions, powers = _reshape_outputs(outputs, config)
    # positions: (B, W, P)  powers: (B, W)  users: (B, U, 2)  qos: (B,)

    device = outputs.device
    dtype = outputs.dtype

    betay = build_waveguide_y_positions_torch(config, device, dtype)
    loc0_x = torch.tensor(-config.area_side_m / 2.0, device=device, dtype=dtype)
    ple = torch.tensor(
        config.speed_of_light_m_s / (4.0 * math.pi * config.carrier_frequency_thz * 1e12),
        device=device, dtype=dtype,
    )
    lambda_m = torch.tensor(config.wavelength_m, device=device, dtype=dtype)
    noise = torch.tensor(thermal_noise_power_w(), device=device, dtype=dtype)

    # ── Waveguide phase weights ──────────────────────────────────────────────
    # w_tx[b, w, p] = exp(j * -2π * |positions[b,w,p] - loc0_x| / λ)
    dist_w = torch.abs(positions - loc0_x)               # (B, W, P)
    phase_w = -2.0 * math.pi * dist_w / lambda_m         # (B, W, P)
    w_tx = torch.exp(1j * phase_w.to(torch.complex64))   # (B, W, P)

    # ── Air-channel matrix ───────────────────────────────────────────────────
    # tx_u[b, w, p, u] = ple/dist * exp(j * -2π * dist / λ)
    # dx[b,w,p,u] = positions[b,w,p] - users[b,u,0]
    pos_x = positions.unsqueeze(3)                         # (B, W, P, 1)
    user_x = users[:, :, 0].unsqueeze(1).unsqueeze(1)     # (B, 1, 1, U)
    user_y = users[:, :, 1].unsqueeze(1).unsqueeze(1)     # (B, 1, 1, U)
    betay_b = betay.reshape(1, -1, 1, 1)                   # (1, W, 1, 1)

    dx = pos_x - user_x                                    # (B, W, P, U)
    dy = betay_b - user_y                                  # (B, W, P, U)
    dist = torch.sqrt(dx * dx + dy * dy + config.transmitter_height_m ** 2)
    phase_tx = -2.0 * math.pi * dist / lambda_m
    tx_u = (ple / dist) * torch.exp(1j * phase_tx.to(torch.complex64))  # (B, W, P, U)

    # ── Desired signal: user u served by waveguide u (diagonal) ─────────────
    # desired_signal[b, u] = Σ_p  tx_u[b, u, p, u] * w_tx[b, u, p]
    # tx_u[b, u, p, u] = diagonal over (W, U) dims after permuting P to last.
    # tx_u.permute(0,1,3,2): (B, W, U, P)
    # diagonal(dim1=1, dim2=2) → (B, P, U)  where result[b,p,u]=tx_u[b,u,p,u]
    # permute(0,2,1) → (B, U, P)
    tx_u_self = torch.diagonal(
        tx_u.permute(0, 1, 3, 2), dim1=1, dim2=2
    ).permute(0, 2, 1).contiguous()                        # (B, U, P) complex
    # w_tx for waveguide u matches user u (W == U, same ordering)
    desired_signal = (tx_u_self * w_tx).sum(dim=2)         # (B, U) complex

    # ── Interference signal ──────────────────────────────────────────────────
    # interf[b, w, u] = Σ_p tx_u[b, w, p, u] * w_tx[b, w, p]
    interf_signal = (tx_u * w_tx.unsqueeze(3)).sum(dim=2)  # (B, W, U) complex
    # interference_power[b, wg, u] = powers[b,wg] * |interf[b,wg,u]|²
    interf_power = powers.unsqueeze(2) * torch.abs(interf_signal) ** 2   # (B, W, U)
    # Zero out diagonal (serving waveguide wg==u is desired, not interference)
    diag = torch.eye(config.num_users, device=device, dtype=torch.bool)
    interference_power = interf_power.masked_fill(diag.unsqueeze(0), 0.0).sum(dim=1)  # (B, U)

    # ── SINR and per-user rates ──────────────────────────────────────────────
    desired_power = powers * torch.abs(desired_signal) ** 2   # (B, U), powers (B,W=U)
    sinr = desired_power / (interference_power + noise)        # (B, U)
    rates = torch.log2(1.0 + sinr)                            # (B, U)

    total_power = powers.sum(dim=1)                            # (B,)
    sum_rate = rates.sum(dim=1)                                # (B,)
    energy_efficiency = sum_rate / (total_power + config.circuit_power_w)
    qos_margin = rates - qos.unsqueeze(1)                      # (B, U)
    qos_satisfied = (qos_margin >= 0.0).all(dim=1)             # (B,)

    return PassPhysicsTorchBatch(
        rates=rates.to(dtype),
        sum_rate=sum_rate.to(dtype),
        total_power=total_power.to(dtype),
        energy_efficiency=energy_efficiency.to(dtype),
        qos_margin=qos_margin.to(dtype),
        qos_satisfied=qos_satisfied,
    )


def physics_penalty_from_outputs(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    config: SystemConfig,
    qos_weight: float = 1.0,
    power_weight: float = 0.25,
) -> torch.Tensor:
    """Return a differentiable penalty for QoS and power-budget violations."""

    batch = evaluate_pass_batch_torch(inputs, outputs, config)
    qos = inputs[:, config.num_users * 2]
    qos_violation = F.relu(qos.unsqueeze(1) - batch.rates)
    qos_penalty = qos_violation.pow(2).mean(dim=1)
    power_penalty = F.relu(batch.total_power - config.power_budget_w).pow(2)
    return qos_weight * qos_penalty + power_weight * power_penalty
