"""Physics-aware PASS simulation helpers used for evaluation and UI previews."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import PASS_POSITION_COLUMNS, PASS_POWER_COLUMNS, SystemConfig


@dataclass(frozen=True)
class PassPhysicsBatch:
    """End-to-end PASS simulation outputs for a batch of samples."""

    rates: np.ndarray
    sum_rate: np.ndarray
    total_power: np.ndarray
    energy_efficiency: np.ndarray
    qos_margin: np.ndarray
    qos_satisfied: np.ndarray


def build_waveguide_y_positions(config: SystemConfig) -> np.ndarray:
    """Match the MATLAB waveguide-centre placement exactly."""

    indices = np.arange(config.num_waveguides, dtype=np.float32)
    return -config.area_side_m / 2.0 + indices * config.area_side_m / config.num_waveguides + config.area_side_m / (2.0 * config.num_waveguides)


def thermal_noise_power_w() -> float:
    """Match the MATLAB noise constant used in the legacy scripts."""

    return float(10 ** ((-174 - 30) / 10.0) * 1e9)


def _reshape_inputs(inputs: np.ndarray, config: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    if inputs.ndim != 2 or inputs.shape[1] < config.num_users * 2 + 1:
        raise ValueError("PASS inputs must be a 2D array with 7 columns.")
    users = inputs[:, : config.num_users * 2].reshape(-1, config.num_users, 2)
    qos = inputs[:, config.num_users * 2]
    return users.astype(np.float32, copy=False), qos.astype(np.float32, copy=False)


def _reshape_outputs(outputs: np.ndarray, config: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    position_dim = config.num_waveguides * config.num_pinchers
    power_dim = config.num_waveguides
    if outputs.ndim != 2 or outputs.shape[1] != position_dim + power_dim:
        raise ValueError("PASS outputs must be a 2D array with 12 columns.")
    positions = outputs[:, :position_dim].reshape(-1, config.num_waveguides, config.num_pinchers)
    powers = outputs[:, position_dim:]
    return positions.astype(np.float32, copy=False), powers.astype(np.float32, copy=False)


def evaluate_pass_batch(
    inputs: np.ndarray,
    outputs: np.ndarray,
    config: SystemConfig,
) -> PassPhysicsBatch:
    """Evaluate PASS configs with the same channel and rate equations as MATLAB."""

    users, qos = _reshape_inputs(inputs, config)
    positions, powers = _reshape_outputs(outputs, config)

    n_samples = outputs.shape[0]
    rates = np.zeros((n_samples, config.num_users), dtype=np.float32)
    sum_rate = np.zeros(n_samples, dtype=np.float32)
    total_power = powers.sum(axis=1).astype(np.float32)
    energy_efficiency = np.zeros(n_samples, dtype=np.float32)
    qos_margin = np.zeros((n_samples, config.num_users), dtype=np.float32)
    qos_satisfied = np.zeros(n_samples, dtype=bool)

    betay = build_waveguide_y_positions(config)
    loc0_x = -config.area_side_m / 2.0
    ple = config.speed_of_light_m_s / (4.0 * np.pi * config.carrier_frequency_thz * 1e12)
    lambda_m = config.wavelength_m
    noise = thermal_noise_power_w()

    for sample_idx in range(n_samples):
        tx_u = np.zeros((config.num_pinchers, config.num_waveguides, config.num_users), dtype=np.complex128)
        w_tx = np.zeros((config.num_pinchers, config.num_waveguides), dtype=np.complex128)

        for wg_idx in range(config.num_waveguides):
            for pincher_idx in range(config.num_pinchers):
                beta = float(positions[sample_idx, wg_idx, pincher_idx])
                dist_w = abs(beta - loc0_x)
                w_tx[pincher_idx, wg_idx] = np.exp(-1j * 2.0 * np.pi * dist_w / lambda_m)

                for user_idx in range(config.num_users):
                    dx = beta - float(users[sample_idx, user_idx, 0])
                    dy = betay[wg_idx] - float(users[sample_idx, user_idx, 1])
                    dist = np.sqrt(dx * dx + dy * dy + config.transmitter_height_m * config.transmitter_height_m)
                    tx_u[pincher_idx, wg_idx, user_idx] = ple / dist * np.exp(-1j * 2.0 * np.pi * dist / lambda_m)

        for user_idx in range(config.num_users):
            desired_signal = 0.0 + 0.0j
            for pincher_idx in range(config.num_pinchers):
                desired_signal += tx_u[pincher_idx, user_idx, user_idx] * w_tx[pincher_idx, user_idx]

            desired_power = float(powers[sample_idx, user_idx] * (abs(desired_signal) ** 2))

            interference_power = 0.0
            for wg_idx in range(config.num_waveguides):
                if wg_idx == user_idx:
                    continue
                interf_signal = 0.0 + 0.0j
                for pincher_idx in range(config.num_pinchers):
                    interf_signal += tx_u[pincher_idx, wg_idx, user_idx] * w_tx[pincher_idx, wg_idx]
                interference_power += float(powers[sample_idx, wg_idx] * (abs(interf_signal) ** 2))

            sinr = desired_power / (interference_power + noise)
            rates[sample_idx, user_idx] = np.float32(np.log2(1.0 + sinr))

        sum_rate[sample_idx] = np.float32(rates[sample_idx].sum())
        energy_efficiency[sample_idx] = np.float32(sum_rate[sample_idx] / (total_power[sample_idx] + config.circuit_power_w))
        qos_margin[sample_idx] = rates[sample_idx] - qos[sample_idx]
        qos_satisfied[sample_idx] = bool(np.all(rates[sample_idx] >= qos[sample_idx]))

    return PassPhysicsBatch(
        rates=rates,
        sum_rate=sum_rate,
        total_power=total_power,
        energy_efficiency=energy_efficiency,
        qos_margin=qos_margin,
        qos_satisfied=qos_satisfied,
    )

