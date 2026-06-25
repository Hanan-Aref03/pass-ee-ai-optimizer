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
    """Evaluate PASS configs with differentiable torch operations."""

    users, qos = _reshape_inputs(inputs, config)
    positions, powers = _reshape_outputs(outputs, config)

    n_samples = outputs.shape[0]
    rates = torch.zeros((n_samples, config.num_users), device=outputs.device, dtype=outputs.dtype)
    sum_rate = torch.zeros(n_samples, device=outputs.device, dtype=outputs.dtype)
    total_power = powers.sum(dim=1).to(outputs.dtype)
    energy_efficiency = torch.zeros(n_samples, device=outputs.device, dtype=outputs.dtype)
    qos_margin = torch.zeros((n_samples, config.num_users), device=outputs.device, dtype=outputs.dtype)
    qos_satisfied = torch.zeros(n_samples, device=outputs.device, dtype=torch.bool)

    betay = build_waveguide_y_positions_torch(config, outputs.device, outputs.dtype)
    loc0_x = torch.tensor(-config.area_side_m / 2.0, device=outputs.device, dtype=outputs.dtype)
    ple = torch.tensor(
        config.speed_of_light_m_s / (4.0 * math.pi * config.carrier_frequency_thz * 1e12),
        device=outputs.device,
        dtype=outputs.dtype,
    )
    lambda_m = torch.tensor(config.wavelength_m, device=outputs.device, dtype=outputs.dtype)
    noise = torch.tensor(thermal_noise_power_w(), device=outputs.device, dtype=outputs.dtype)

    for sample_idx in range(n_samples):
        tx_u = torch.zeros(
            (config.num_pinchers, config.num_waveguides, config.num_users),
            dtype=torch.complex64,
            device=outputs.device,
        )
        w_tx = torch.zeros(
            (config.num_pinchers, config.num_waveguides),
            dtype=torch.complex64,
            device=outputs.device,
        )

        for wg_idx in range(config.num_waveguides):
            for pincher_idx in range(config.num_pinchers):
                beta = positions[sample_idx, wg_idx, pincher_idx]
                dist_w = torch.abs(beta - loc0_x)
                phase_w = -2.0 * math.pi * dist_w / lambda_m
                w_tx[pincher_idx, wg_idx] = torch.exp(1j * phase_w.to(torch.complex64))

                for user_idx in range(config.num_users):
                    dx = beta - users[sample_idx, user_idx, 0]
                    dy = betay[wg_idx] - users[sample_idx, user_idx, 1]
                    dist = torch.sqrt(dx * dx + dy * dy + config.transmitter_height_m * config.transmitter_height_m)
                    phase = -2.0 * math.pi * dist / lambda_m
                    tx_u[pincher_idx, wg_idx, user_idx] = (
                        ple / dist * torch.exp(1j * phase.to(torch.complex64))
                    )

        for user_idx in range(config.num_users):
            desired_signal = torch.zeros((), dtype=torch.complex64, device=outputs.device)
            for pincher_idx in range(config.num_pinchers):
                desired_signal = desired_signal + tx_u[pincher_idx, user_idx, user_idx] * w_tx[pincher_idx, user_idx]

            desired_power = powers[sample_idx, user_idx] * (torch.abs(desired_signal) ** 2)

            interference_power = torch.zeros((), dtype=outputs.dtype, device=outputs.device)
            for wg_idx in range(config.num_waveguides):
                if wg_idx == user_idx:
                    continue
                interf_signal = torch.zeros((), dtype=torch.complex64, device=outputs.device)
                for pincher_idx in range(config.num_pinchers):
                    interf_signal = interf_signal + tx_u[pincher_idx, wg_idx, user_idx] * w_tx[pincher_idx, wg_idx]
                interference_power = interference_power + powers[sample_idx, wg_idx] * (torch.abs(interf_signal) ** 2)

            sinr = desired_power / (interference_power + noise)
            rates[sample_idx, user_idx] = torch.log2(1.0 + sinr)

        sum_rate[sample_idx] = rates[sample_idx].sum()
        energy_efficiency[sample_idx] = sum_rate[sample_idx] / (total_power[sample_idx] + config.circuit_power_w)
        qos_margin[sample_idx] = rates[sample_idx] - qos[sample_idx]
        qos_satisfied[sample_idx] = bool(torch.all(rates[sample_idx] >= qos[sample_idx]))

    return PassPhysicsTorchBatch(
        rates=rates,
        sum_rate=sum_rate,
        total_power=total_power,
        energy_efficiency=energy_efficiency,
        qos_margin=qos_margin,
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
