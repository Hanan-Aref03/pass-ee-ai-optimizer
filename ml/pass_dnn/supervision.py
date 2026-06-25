"""Physics-derived supervision helpers for PASS training."""

from __future__ import annotations

import numpy as np

from .data import canonicalize_pass_positions
from .physics import evaluate_pass_batch
from .schema import PASS_POSITION_COLUMNS, SystemConfig


def build_pass_feasibility_labels(
    inputs: np.ndarray,
    outputs: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    """Derive binary QoS feasibility labels from the physics simulator."""

    if inputs.ndim != 2 or outputs.ndim != 2:
        raise ValueError("PASS supervision arrays must be 2D matrices.")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("PASS inputs and outputs must contain the same number of rows.")
    if outputs.shape[1] != len(PASS_POSITION_COLUMNS) + config.num_waveguides:
        raise ValueError("Unexpected PASS output dimension.")

    physical_outputs = outputs.astype(np.float32, copy=True)
    physical_outputs[:, : len(PASS_POSITION_COLUMNS)] = canonicalize_pass_positions(
        physical_outputs[:, : len(PASS_POSITION_COLUMNS)]
    )
    evaluation = evaluate_pass_batch(inputs.astype(np.float32, copy=False), physical_outputs, config)
    return evaluation.qos_satisfied.astype(np.float32).reshape(-1, 1)


def build_pass_sample_weights(
    inputs: np.ndarray,
    feasibility: np.ndarray,
    qos_weight: float = 1.5,
    infeasible_weight: float = 4.0,
    feasible_boost: float = 1.0,
) -> np.ndarray:
    """Assign higher loss weight to harder QoS rows and infeasible examples."""

    if inputs.ndim != 2:
        raise ValueError("PASS inputs must be a 2D matrix.")
    if feasibility.ndim == 1:
        feasibility = feasibility.reshape(-1, 1)
    if inputs.shape[0] != feasibility.shape[0]:
        raise ValueError("Inputs and feasibility labels must have the same number of rows.")

    qos = np.asarray(inputs[:, -1], dtype=np.float32)
    qos_min = float(qos.min())
    qos_max = float(qos.max())
    qos_span = max(qos_max - qos_min, 1e-8)
    qos_norm = (qos - qos_min) / qos_span
    feasibility = np.asarray(feasibility, dtype=np.float32).reshape(-1)

    weights = (
        1.0
        + qos_weight * qos_norm
        + infeasible_weight * (1.0 - feasibility)
        + feasible_boost * feasibility
    )
    return weights.astype(np.float32).reshape(-1, 1)


def build_pass_training_supervision(
    inputs: np.ndarray,
    outputs: np.ndarray,
    config: SystemConfig,
    qos_weight: float = 1.5,
    infeasible_weight: float = 4.0,
    feasible_boost: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return feasibility labels and sample weights for PASS training."""

    feasibility = build_pass_feasibility_labels(inputs, outputs, config)
    sample_weights = build_pass_sample_weights(
        inputs=inputs,
        feasibility=feasibility,
        qos_weight=qos_weight,
        infeasible_weight=infeasible_weight,
        feasible_boost=feasible_boost,
    )
    return feasibility, sample_weights
