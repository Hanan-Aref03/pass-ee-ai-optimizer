"""Train a PASS or conventional tabular DNN from MATLAB-generated CSV pairs."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import (
    DatasetBundle,
    RegressionDataset,
    SupervisedDataset,
    Standardizer,
    augment_pass_dataset,
    canonicalize_pass_positions,
    denormalize_powers,
    denormalize_positions,
    fit_standardizer,
    infer_outputs_from_model,
    load_dataset_pair,
    load_dataset_corpus,
    load_latest_pass_dataset_pair,
    normalize_powers,
    normalize_positions,
    project_pass_outputs,
    project_pass_powers,
    project_pass_positions,
    split_indices,
)
from .model import ModelConfig, PassDnnRegressor
from .metrics import classification_like_report, regression_report
from .physics import evaluate_pass_batch
from .supervision import build_pass_training_supervision
from .torch_physics import (
    evaluate_pass_batch_torch,
    physics_penalty_from_outputs,
    sample_pass_feasibility_candidates,
    to_physical_pass_outputs,
)
from .schema import (
    INPUT_COLUMNS,
    CONVENTIONAL_OUTPUT_COLUMNS,
    PASS_OUTPUT_COLUMNS,
    PASS_POSITION_COLUMNS,
    PASS_POWER_COLUMNS,
    SystemConfig,
    DatasetSchema,
)


@dataclass
class TrainResult:
    artifact_dir: Path
    best_epoch: int
    best_val_loss: float
    test_loss: float
    test_metrics: dict


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_paths(
    input_csv: str | None,
    output_csv: str | None,
    search_roots: Sequence[str | Path],
) -> tuple[Path, Path]:
    if input_csv and output_csv:
        return Path(input_csv), Path(output_csv)

    if input_csv and not output_csv:
        input_path = Path(input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        if "dataset_input_" in input_path.name:
            output_name = input_path.name.replace("dataset_input_", "dataset_output_")
        else:
            output_name = input_path.with_suffix(".csv").name
        output_path = input_path.with_name(output_name)
        if not output_path.exists():
            raise FileNotFoundError(
                f"Could not infer matching output CSV from {input_path}. Provide --output-csv."
            )
        return input_path, output_path

    if output_csv and not input_csv:
        output_path = Path(output_csv)
        if not output_path.exists():
            raise FileNotFoundError(f"Output CSV not found: {output_path}")
        if "dataset_output_" in output_path.name:
            input_name = output_path.name.replace("dataset_output_", "dataset_input_")
        else:
            input_name = output_path.with_suffix(".csv").name
        input_path = output_path.with_name(input_name)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Could not infer matching input CSV from {output_path}. Provide --input-csv."
            )
        return input_path, output_path

    return load_latest_pass_dataset_pair(search_roots)


def build_system_config(args: argparse.Namespace) -> SystemConfig:
    return SystemConfig(
        area_side_m=args.area_side_m,
        transmitter_height_m=args.transmitter_height_m,
        carrier_frequency_thz=args.carrier_frequency_thz,
        power_budget_w=args.power_budget_w,
        circuit_power_w=args.circuit_power_w,
    )


def build_model(
    schema: DatasetSchema,
    model_cfg: ModelConfig,
) -> PassDnnRegressor:
    return PassDnnRegressor(
        input_dim=schema.input_dim,
        mode=schema.mode,
        position_dim=schema.position_dim,
        power_dim=schema.power_dim,
        config=model_cfg,
    )


def prepare_targets(
    bundle: DatasetBundle,
    config: SystemConfig,
) -> np.ndarray:
    outputs = bundle.outputs.copy()
    if bundle.schema.mode == "pass":
        outputs[:, : len(PASS_POSITION_COLUMNS)] = canonicalize_pass_positions(
            outputs[:, : len(PASS_POSITION_COLUMNS)]
        )
        positions = normalize_positions(outputs[:, : len(PASS_POSITION_COLUMNS)], config)
        powers = normalize_powers(outputs[:, len(PASS_POSITION_COLUMNS) :], config)
        return np.concatenate([positions, powers], axis=1).astype(np.float32)

    powers = normalize_powers(outputs, config)
    return powers.astype(np.float32)


def prepare_pass_supervision(
    bundle: DatasetBundle,
    config: SystemConfig,
    train_idx: np.ndarray,
    augment_user_permutations: bool,
    qos_weight: float,
    infeasible_weight: float,
    feasible_boost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feasibility labels and sample weights for PASS training rows."""

    if bundle.schema.mode != "pass":
        ones = np.ones((len(train_idx), 1), dtype=np.float32)
        return ones, ones

    raw_inputs = bundle.inputs[train_idx]
    raw_outputs = bundle.outputs[train_idx]
    feasibility, sample_weights = build_pass_training_supervision(
        inputs=raw_inputs,
        outputs=raw_outputs,
        config=config,
        qos_weight=qos_weight,
        infeasible_weight=infeasible_weight,
        feasible_boost=feasible_boost,
    )

    if augment_user_permutations:
        multiplier = len(list(permutations(range(3))))
        feasibility = np.tile(feasibility, (multiplier, 1))
        sample_weights = np.tile(sample_weights, (multiplier, 1))

    return feasibility.astype(np.float32), sample_weights.astype(np.float32)


def split_bundle(bundle: DatasetBundle, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = split_indices(len(bundle.inputs), seed=seed)
    return split.train_idx, split.val_idx, split.test_idx


def make_loaders(
    inputs: np.ndarray,
    targets: np.ndarray,
    schema: DatasetSchema,
    seed: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    split = split_indices(len(inputs), seed=seed)
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx

    train_inputs = inputs[train_idx]
    val_inputs = inputs[val_idx]
    test_inputs = inputs[test_idx]
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]
    test_targets = targets[test_idx]

    train_ds = RegressionDataset(train_inputs, train_targets, schema)
    val_ds = RegressionDataset(val_inputs, val_targets, schema)
    test_ds = RegressionDataset(test_inputs, test_targets, schema)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


def build_qos_balanced_sampler(inputs: np.ndarray) -> WeightedRandomSampler:
    """Create a sampler that balances PASS training rows by QoS value."""

    if inputs.ndim != 2 or inputs.shape[1] < len(INPUT_COLUMNS):
        raise ValueError("PASS inputs must be a 2D matrix with QoS in the last column.")

    qos_values = np.asarray(inputs[:, -1], dtype=np.float64)
    qos_keys, qos_counts = np.unique(np.round(qos_values, 6), return_counts=True)
    count_lookup = {float(key): float(count) for key, count in zip(qos_keys, qos_counts)}
    sample_weights = np.asarray(
        [1.0 / count_lookup[float(np.round(value, 6))] for value in qos_values],
        dtype=np.float64,
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _unpack_supervised_batch(batch):
    """Accept either the legacy two-tuple or the new four-tuple batch format."""

    if len(batch) == 2:
        inputs, targets = batch
        aux_targets = None
        sample_weights = None
        physics_inputs = None
    elif len(batch) == 4:
        inputs, targets, aux_targets, sample_weights = batch
        physics_inputs = None
    elif len(batch) == 5:
        inputs, targets, aux_targets, sample_weights, physics_inputs = batch
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} items.")
    return inputs, targets, aux_targets, sample_weights, physics_inputs


def _reduce_per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
    """Collapse feature dimensions into one loss value per sample."""

    if loss_tensor.ndim == 0:
        return loss_tensor.reshape(1)
    if loss_tensor.ndim == 1:
        return loss_tensor
    return loss_tensor.mean(dim=tuple(range(1, loss_tensor.ndim)))


def _weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    """Compute a numerically stable weighted mean over batch elements."""

    if sample_weights is None:
        return values.mean()

    weights = sample_weights.reshape(-1)
    numerator = torch.sum(values * weights)
    denominator = torch.clamp(weights.sum(), min=1e-8)
    return numerator / denominator


def run_epoch(
    model: PassDnnRegressor,
    loader: DataLoader,
    device: torch.device,
    schema: DatasetSchema,
    config: SystemConfig,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
    regression_loss_weight: float = 1.0,
    feasibility_loss_weight: float = 1.0,
    feasibility_negative_weight: float = 1.0,
    feasibility_negative_candidates: int = 1,
    feasibility_negative_jitter: float = 0.12,
    feasibility_negative_power_scale_min: float = 0.10,
    feasibility_negative_power_scale_max: float = 0.55,
    physics_loss_weight: float = 1.0,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_pos_loss = 0.0
    total_pow_loss = 0.0
    total_count = 0

    for batch in loader:
        inputs, targets, aux_targets, sample_weights, physics_inputs = _unpack_supervised_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)
        if aux_targets is not None:
            aux_targets = aux_targets.to(device)
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)
        if physics_inputs is not None:
            physics_inputs = physics_inputs.to(device)
        else:
            physics_inputs = inputs

        if training:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        if schema.mode == "pass":
            pos_target = targets[:, : schema.position_dim]
            pow_target = targets[:, schema.position_dim :]
            pos_loss = _reduce_per_sample(criterion(outputs["positions"], pos_target))
            pow_loss = _reduce_per_sample(criterion(outputs["powers"], pow_target))
            batch_pos_loss = _weighted_mean(pos_loss, sample_weights)
            batch_pow_loss = _weighted_mean(pow_loss, sample_weights)
            regression_loss = _weighted_mean(pos_loss + pow_loss, sample_weights)

            feasibility_loss = torch.tensor(0.0, device=device)
            if "feasibility_logit" in outputs and aux_targets is not None:
                feasibility_target = aux_targets.view(-1)
                feasibility_pred = outputs["feasibility_logit"].view(-1)
                positive_feasibility_loss = _weighted_mean(
                    F.binary_cross_entropy_with_logits(
                        feasibility_pred,
                        feasibility_target,
                        reduction="none",
                    ),
                    sample_weights,
                )
                feasibility_loss = positive_feasibility_loss

                if (
                    training
                    and model.config.feasibility_head
                    and model.config.feasibility_conditioning == "input_output"
                    and feasibility_negative_candidates > 0
                ):
                    negative_candidates = sample_pass_feasibility_candidates(
                        targets.detach(),
                        config=config,
                        num_candidates=feasibility_negative_candidates,
                        position_jitter=feasibility_negative_jitter,
                        power_scale_min=feasibility_negative_power_scale_min,
                        power_scale_max=feasibility_negative_power_scale_max,
                    )
                    negative_physical = to_physical_pass_outputs(negative_candidates, config)
                    repeated_physics_inputs = physics_inputs.repeat_interleave(
                        feasibility_negative_candidates, dim=0
                    )
                    with torch.no_grad():
                        negative_labels = evaluate_pass_batch_torch(
                            repeated_physics_inputs,
                            negative_physical,
                            config,
                        ).qos_satisfied.to(dtype=targets.dtype).view(-1)
                    repeated_inputs = inputs.repeat_interleave(
                        feasibility_negative_candidates, dim=0
                    )
                    negative_positions = negative_candidates[:, : schema.position_dim]
                    negative_powers = negative_candidates[:, schema.position_dim :]
                    negative_logits = model.feasibility_from_candidate(
                        repeated_inputs,
                        negative_positions,
                        negative_powers,
                    ).view(-1)
                    negative_sample_weights = sample_weights.repeat_interleave(
                        feasibility_negative_candidates, dim=0
                    ).view(-1)
                    negative_loss = _weighted_mean(
                        F.binary_cross_entropy_with_logits(
                            negative_logits,
                            negative_labels,
                            reduction="none",
                        ),
                        negative_sample_weights,
                    )
                    feasibility_loss = feasibility_loss + feasibility_negative_weight * negative_loss

            physics_loss = torch.tensor(0.0, device=device)
            if physics_loss_weight > 0:
                physical_predictions = to_physical_pass_outputs(
                    torch.cat([outputs["positions"], outputs["powers"]], dim=1),
                    config,
                )
                physics_loss = _weighted_mean(
                    physics_penalty_from_outputs(physics_inputs, physical_predictions, config),
                    sample_weights,
                )

            loss = (
                regression_loss_weight * regression_loss
                + feasibility_loss_weight * feasibility_loss
                + physics_loss_weight * physics_loss
            )
        else:
            pow_target = targets
            pos_loss = torch.tensor(0.0, device=device)
            pow_loss = _reduce_per_sample(criterion(outputs["powers"], pow_target))
            batch_pos_loss = pos_loss
            batch_pow_loss = _weighted_mean(pow_loss, sample_weights)
            loss = regression_loss_weight * _weighted_mean(pow_loss, sample_weights)

        if training:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = inputs.shape[0]
        total_count += batch_size
        total_loss += loss.item() * batch_size
        total_pos_loss += batch_pos_loss.item() * batch_size
        total_pow_loss += batch_pow_loss.item() * batch_size

    denom = max(total_count, 1)
    return total_loss / denom, total_pos_loss / denom, total_pow_loss / denom


@torch.no_grad()
def predict_normalized(
    model: PassDnnRegressor,
    loader: DataLoader,
    device: torch.device,
    schema: DatasetSchema,
    include_aux: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    aux_predictions: list[np.ndarray] = []

    for batch in loader:
        inputs, target, _, _, _ = _unpack_supervised_batch(batch)
        inputs = inputs.to(device)
        outputs = model(inputs)
        if schema.mode == "pass":
            pred = torch.cat([outputs["positions"], outputs["powers"]], dim=1)
            if include_aux:
                if "feasibility_prob" in outputs:
                    aux_predictions.append(outputs["feasibility_prob"].detach().cpu().numpy())
                elif "feasibility_logit" in outputs:
                    aux_predictions.append(torch.sigmoid(outputs["feasibility_logit"]).detach().cpu().numpy())
        else:
            pred = outputs["powers"]
        predictions.append(pred.cpu().numpy())
        targets.append(target.numpy())

    pred_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    if include_aux:
        aux_array = np.concatenate(aux_predictions, axis=0) if aux_predictions else None
        return pred_array, target_array, aux_array
    return pred_array, target_array


def compute_physical_metrics(
    inputs: np.ndarray,
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    schema: DatasetSchema,
    config: SystemConfig,
) -> dict:
    metrics: dict[str, float] = {}

    if schema.mode == "pass":
        pos_pred_m = denormalize_positions(pred_norm[:, : schema.position_dim], config)
        pos_target_m = denormalize_positions(
            target_norm[:, : schema.position_dim], config
        )
        pow_pred_w = denormalize_powers(pred_norm[:, schema.position_dim :], config)
        pow_target_w = denormalize_powers(
            target_norm[:, schema.position_dim :], config
        )

        pos_projected_m, pow_projected_w = project_pass_outputs(
            pos_pred_m, pow_pred_w, config
        )
        position_tolerance_m = 0.10
        power_tolerance_w = max(config.power_budget_w * 0.10, 1e-6)

        position_raw_report = regression_report(
            pos_target_m,
            pos_pred_m,
            PASS_POSITION_COLUMNS,
            tolerance=position_tolerance_m,
        )
        position_projected_report = regression_report(
            pos_target_m,
            pos_projected_m,
            PASS_POSITION_COLUMNS,
            tolerance=position_tolerance_m,
        )
        power_raw_report = regression_report(
            pow_target_w,
            pow_pred_w,
            PASS_POWER_COLUMNS,
            tolerance=power_tolerance_w,
        )
        power_projected_report = regression_report(
            pow_target_w,
            pow_projected_w,
            PASS_POWER_COLUMNS,
            tolerance=power_tolerance_w,
        )

        metrics["position_tolerance_m"] = position_tolerance_m
        metrics["power_tolerance_w"] = power_tolerance_w
        metrics["position_mae_m_raw"] = float(position_raw_report.overall["mae"])
        metrics["position_rmse_m_raw"] = float(position_raw_report.overall["rmse"])
        metrics["position_r2_raw"] = float(position_raw_report.overall["r2"])
        metrics["position_max_abs_error_m_raw"] = float(
            position_raw_report.overall["max_abs_error"]
        )
        metrics["position_within_tolerance_rate_raw"] = float(
            position_raw_report.overall["within_tolerance_rate"]
        )
        metrics["position_mae_m_projected"] = float(
            position_projected_report.overall["mae"]
        )
        metrics["position_rmse_m_projected"] = float(
            position_projected_report.overall["rmse"]
        )
        metrics["position_r2_projected"] = float(
            position_projected_report.overall["r2"]
        )
        metrics["position_max_abs_error_m_projected"] = float(
            position_projected_report.overall["max_abs_error"]
        )
        metrics["position_within_tolerance_rate_projected"] = float(
            position_projected_report.overall["within_tolerance_rate"]
        )
        metrics["power_mae_w_raw"] = float(power_raw_report.overall["mae"])
        metrics["power_rmse_w_raw"] = float(power_raw_report.overall["rmse"])
        metrics["power_r2_raw"] = float(power_raw_report.overall["r2"])
        metrics["power_max_abs_error_w_raw"] = float(
            power_raw_report.overall["max_abs_error"]
        )
        metrics["power_within_tolerance_rate_raw"] = float(
            power_raw_report.overall["within_tolerance_rate"]
        )
        metrics["power_mae_w_projected"] = float(power_projected_report.overall["mae"])
        metrics["power_rmse_w_projected"] = float(
            power_projected_report.overall["rmse"]
        )
        metrics["power_r2_projected"] = float(power_projected_report.overall["r2"])
        metrics["power_max_abs_error_w_projected"] = float(
            power_projected_report.overall["max_abs_error"]
        )
        metrics["power_within_tolerance_rate_projected"] = float(
            power_projected_report.overall["within_tolerance_rate"]
        )
        metrics["mean_total_power_w_projected"] = float(pow_projected_w.sum(axis=1).mean())
        metrics["feasible_power_rate_projected"] = float(
            np.mean(pow_projected_w.sum(axis=1) <= config.power_budget_w + 1e-9)
        )

        true_outputs = np.concatenate([pos_target_m, pow_target_w], axis=1)
        projected_outputs = np.concatenate([pos_projected_m, pow_projected_w], axis=1)
        true_eval = evaluate_pass_batch(inputs, true_outputs, config)
        pred_eval = evaluate_pass_batch(inputs, projected_outputs, config)
        qos_report = classification_like_report(true_eval.qos_satisfied, pred_eval.qos_satisfied)

        metrics["true_sum_rate_mean"] = float(true_eval.sum_rate.mean())
        metrics["pred_sum_rate_mean"] = float(pred_eval.sum_rate.mean())
        metrics["sum_rate_mae"] = float(np.abs(pred_eval.sum_rate - true_eval.sum_rate).mean())
        metrics["true_ee_mean"] = float(true_eval.energy_efficiency.mean())
        metrics["pred_ee_mean"] = float(pred_eval.energy_efficiency.mean())
        metrics["ee_mae"] = float(np.abs(pred_eval.energy_efficiency - true_eval.energy_efficiency).mean())
        metrics["qos_satisfaction_rate_pred"] = float(pred_eval.qos_satisfied.mean())
        metrics["qos_satisfaction_rate_true"] = float(true_eval.qos_satisfied.mean())
        metrics["mean_qos_margin_pred"] = float(pred_eval.qos_margin.mean())
        metrics["mean_qos_margin_true"] = float(true_eval.qos_margin.mean())
        metrics["qos_accuracy"] = float(qos_report["accuracy"])
        metrics["qos_precision"] = float(qos_report["precision"])
        metrics["qos_recall"] = float(qos_report["recall"])
        metrics["qos_f1"] = float(qos_report["f1"])
        metrics["qos_balanced_accuracy"] = float(qos_report["balanced_accuracy"])
        metrics["qos_tp"] = float(qos_report["tp"])
        metrics["qos_tn"] = float(qos_report["tn"])
        metrics["qos_fp"] = float(qos_report["fp"])
        metrics["qos_fn"] = float(qos_report["fn"])
        for user_idx in range(config.num_users):
            user_prefix = f"user{user_idx + 1}"
            metrics[f"{user_prefix}_true_rate_mean"] = float(true_eval.rates[:, user_idx].mean())
            metrics[f"{user_prefix}_pred_rate_mean"] = float(pred_eval.rates[:, user_idx].mean())
            metrics[f"{user_prefix}_true_margin_mean"] = float(true_eval.qos_margin[:, user_idx].mean())
            metrics[f"{user_prefix}_pred_margin_mean"] = float(pred_eval.qos_margin[:, user_idx].mean())
            metrics[f"{user_prefix}_pred_rate_mae"] = float(
                np.abs(pred_eval.rates[:, user_idx] - true_eval.rates[:, user_idx]).mean()
            )

        per_qos: dict[str, dict[str, float]] = {}
        for qos_value in sorted(np.unique(np.round(inputs[:, -1], 6))):
            mask = np.isclose(inputs[:, -1], qos_value, atol=1e-6)
            if not np.any(mask):
                continue
            group_report = classification_like_report(
                true_eval.qos_satisfied[mask],
                pred_eval.qos_satisfied[mask],
            )
            per_qos[f"{float(qos_value):.1f}"] = {
                "count": float(mask.sum()),
                "true_sat_rate": float(true_eval.qos_satisfied[mask].mean()),
                "pred_sat_rate": float(pred_eval.qos_satisfied[mask].mean()),
                "qos_accuracy": float(group_report["accuracy"]),
                "qos_precision": float(group_report["precision"]),
                "qos_recall": float(group_report["recall"]),
                "qos_f1": float(group_report["f1"]),
                "qos_balanced_accuracy": float(group_report["balanced_accuracy"]),
                "sum_rate_mae": float(
                    np.abs(pred_eval.sum_rate[mask] - true_eval.sum_rate[mask]).mean()
                ),
                "ee_mae": float(
                    np.abs(
                        pred_eval.energy_efficiency[mask]
                        - true_eval.energy_efficiency[mask]
                    ).mean()
                ),
            }

        metrics["per_qos"] = per_qos
    else:
        pow_pred_w = denormalize_powers(pred_norm, config)
        pow_target_w = denormalize_powers(target_norm, config)
        pow_projected_w = project_pass_powers(pow_pred_w, config)
        power_tolerance_w = max(config.power_budget_w * 0.10, 1e-6)
        power_raw_report = regression_report(
            pow_target_w,
            pow_pred_w,
            CONVENTIONAL_OUTPUT_COLUMNS,
            tolerance=power_tolerance_w,
        )
        power_projected_report = regression_report(
            pow_target_w,
            pow_projected_w,
            CONVENTIONAL_OUTPUT_COLUMNS,
            tolerance=power_tolerance_w,
        )
        metrics["power_tolerance_w"] = power_tolerance_w
        metrics["power_mae_w_raw"] = float(power_raw_report.overall["mae"])
        metrics["power_rmse_w_raw"] = float(power_raw_report.overall["rmse"])
        metrics["power_r2_raw"] = float(power_raw_report.overall["r2"])
        metrics["power_max_abs_error_w_raw"] = float(
            power_raw_report.overall["max_abs_error"]
        )
        metrics["power_within_tolerance_rate_raw"] = float(
            power_raw_report.overall["within_tolerance_rate"]
        )
        metrics["power_mae_w_projected"] = float(power_projected_report.overall["mae"])
        metrics["power_rmse_w_projected"] = float(
            power_projected_report.overall["rmse"]
        )
        metrics["power_r2_projected"] = float(power_projected_report.overall["r2"])
        metrics["power_max_abs_error_w_projected"] = float(
            power_projected_report.overall["max_abs_error"]
        )
        metrics["power_within_tolerance_rate_projected"] = float(
            power_projected_report.overall["within_tolerance_rate"]
        )
        metrics["mean_total_power_w_projected"] = float(pow_projected_w.sum(axis=1).mean())
        metrics["feasible_power_rate_projected"] = float(
            np.mean(pow_projected_w.sum(axis=1) <= config.power_budget_w + 1e-9)
        )

    return metrics


def save_artifacts(
    artifact_dir: Path,
    model: PassDnnRegressor,
    input_scaler: Standardizer,
    bundle: DatasetBundle,
    config: SystemConfig,
    model_config: ModelConfig,
    training_config: dict,
    history: list[dict],
    best_epoch: int,
    best_val_loss: float,
    test_loss: float,
    test_metrics: dict,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    np.savez(artifact_dir / "input_scaler.npz", mean=input_scaler.mean, std=input_scaler.std)

    source_paths = bundle.input_frame.attrs.get("source_paths")
    if source_paths:
        source_input_csv = [pair[0] for pair in source_paths]
        source_output_csv = [pair[1] for pair in source_paths]
    else:
        source_input_csv = str(bundle.input_frame.attrs.get("source_path", ""))
        source_output_csv = str(bundle.output_frame.attrs.get("source_path", ""))

    metadata = {
        "mode": bundle.schema.mode,
        "schema": {
            "input_columns": list(bundle.schema.input_columns),
            "output_columns": list(bundle.schema.output_columns),
            "position_columns": list(bundle.schema.position_columns),
            "power_columns": list(bundle.schema.power_columns),
        },
        "system_config": config.to_dict(),
        "model_config": asdict(model_config),
        "training_config": training_config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "test_rows": int(len(test_indices)),
        "source_input_csv": source_input_csv,
        "source_output_csv": source_output_csv,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(artifact_dir / "history.csv", index=False)


def load_artifact_bundle(
    artifact_dir: str | Path,
    device: str | torch.device = "cpu",
) -> dict:
    artifact_path = Path(artifact_dir)
    metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
    scaler = np.load(artifact_path / "input_scaler.npz")
    input_scaler = Standardizer.from_arrays(scaler["mean"], scaler["std"])

    schema = DatasetSchema(
        mode=metadata["mode"],
        input_columns=tuple(metadata["schema"]["input_columns"]),
        output_columns=tuple(metadata["schema"]["output_columns"]),
        position_columns=tuple(metadata["schema"]["position_columns"]),
        power_columns=tuple(metadata["schema"]["power_columns"]),
    )
    config_payload = metadata["system_config"]
    config_fields = SystemConfig.__dataclass_fields__.keys()
    filtered_config = {key: config_payload[key] for key in config_fields if key in config_payload}
    config = SystemConfig(**filtered_config)
    model_config = ModelConfig(**metadata["model_config"])
    model = build_model(schema, model_config)
    state_dict = torch.load(artifact_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "artifact_dir": artifact_path,
        "metadata": metadata,
        "schema": schema,
        "config": config,
        "model_config": model_config,
        "input_scaler": input_scaler,
        "model": model,
        "device": torch.device(device),
    }


@torch.no_grad()
def predict_frame(
    frame: pd.DataFrame,
    bundle: dict,
    project: bool = True,
    include_aux: bool = False,
) -> pd.DataFrame:
    schema: DatasetSchema = bundle["schema"]
    config: SystemConfig = bundle["config"]
    input_scaler: Standardizer = bundle["input_scaler"]
    model: PassDnnRegressor = bundle["model"]
    device: torch.device = bundle["device"]

    if tuple(frame.columns.tolist()) != schema.input_columns:
        raise ValueError(
            f"Unexpected input columns.\nExpected: {schema.input_columns}\nActual:   {tuple(frame.columns.tolist())}"
        )

    inputs = frame.to_numpy(dtype=np.float32, copy=True)
    inputs_scaled = input_scaler.transform(inputs)
    tensor_inputs = torch.tensor(inputs_scaled, dtype=torch.float32, device=device)

    outputs = model(tensor_inputs)
    feasibility_prob = None
    if schema.mode == "pass" and include_aux:
        if "feasibility_prob" in outputs:
            feasibility_prob = outputs["feasibility_prob"].detach().cpu().numpy()
        elif "feasibility_logit" in outputs:
            feasibility_prob = torch.sigmoid(outputs["feasibility_logit"]).detach().cpu().numpy()

    if schema.mode == "pass":
        preds = torch.cat([outputs["positions"], outputs["powers"]], dim=1).cpu().numpy()
    else:
        preds = outputs["powers"].cpu().numpy()

    if project:
        if schema.mode == "pass":
            physical = infer_outputs_from_model(
                preds[:, : schema.position_dim],
                preds[:, schema.position_dim :],
                schema,
                config,
            )
        else:
            physical = infer_outputs_from_model(None, preds, schema, config)
    else:
        if schema.mode == "pass":
            physical = np.concatenate(
                [
                    denormalize_positions(preds[:, : schema.position_dim], config),
                    denormalize_powers(preds[:, schema.position_dim :], config),
                ],
                axis=1,
            )
        else:
            physical = denormalize_powers(preds, config)

    output_frame = pd.DataFrame(physical, columns=schema.output_columns)
    if include_aux and schema.mode == "pass":
        if feasibility_prob is None:
            output_frame["feasibility_prob"] = np.nan
            output_frame["feasibility_pred"] = np.nan
        else:
            feasibility_prob = feasibility_prob.reshape(-1)
            output_frame["feasibility_prob"] = feasibility_prob
            output_frame["feasibility_pred"] = (feasibility_prob >= 0.5).astype(np.int32)

    return output_frame


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DNN surrogate for the PASS energy-efficiency optimizer."
    )
    parser.add_argument("--input-csv", type=str, default=None, help="PASS input CSV")
    parser.add_argument("--output-csv", type=str, default=None, help="PASS output CSV")
    parser.add_argument(
        "--search-root",
        action="append",
        default=["data/raw", "matlab/legacy/csv_data", "matlab/legacy"],
        help="Directories to search when --input-csv/--output-csv are omitted.",
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default="ml/pass_dnn/artifacts",
        help="Directory where training artifacts will be written.",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--area-side-m", type=float, default=10.0)
    parser.add_argument("--transmitter-height-m", type=float, default=5.0)
    parser.add_argument("--carrier-frequency-thz", type=float, default=0.3)
    parser.add_argument("--power-budget-w", type=float, default=0.1)
    parser.add_argument("--circuit-power-w", type=float, default=0.01)
    parser.add_argument(
        "--criterion",
        choices=("mse", "huber"),
        default="huber",
        help="Loss function for noisy regression targets.",
    )
    parser.add_argument(
        "--augment-user-permutations",
        dest="augment_user_permutations",
        action="store_true",
        default=True,
        help="Augment PASS training rows by permuting the 3-user symmetry classes.",
    )
    parser.add_argument(
        "--no-augment-user-permutations",
        dest="augment_user_permutations",
        action="store_false",
        help="Disable symmetry augmentation.",
    )
    parser.add_argument(
        "--balance-by-qos",
        dest="balance_by_qos",
        action="store_true",
        default=True,
        help="Balance PASS training batches by inverse QoS frequency.",
    )
    parser.add_argument(
        "--no-balance-by-qos",
        dest="balance_by_qos",
        action="store_false",
        help="Use the raw QoS distribution during training.",
    )
    parser.add_argument(
        "--feasibility-head",
        dest="feasibility_head",
        action="store_true",
        default=True,
        help="Train a separate PASS feasibility head.",
    )
    parser.add_argument(
        "--no-feasibility-head",
        dest="feasibility_head",
        action="store_false",
        help="Disable the feasibility head and train the older regression-only model.",
    )
    parser.add_argument(
        "--feasibility-conditioning",
        choices=("hidden", "input_output"),
        default="input_output",
        help="How the feasibility head is conditioned. input_output uses the input row and candidate outputs.",
    )
    parser.add_argument(
        "--feasibility-loss-weight",
        type=float,
        default=2.5,
        help="Relative weight for the feasibility-head BCE loss.",
    )
    parser.add_argument(
        "--feasibility-negative-weight",
        type=float,
        default=1.5,
        help="Relative weight for synthetic negative feasibility examples.",
    )
    parser.add_argument(
        "--feasibility-negative-candidates",
        type=int,
        default=1,
        help="How many synthetic feasibility negatives to generate per training sample.",
    )
    parser.add_argument(
        "--feasibility-negative-jitter",
        type=float,
        default=0.12,
        help="Normalized position jitter applied when generating synthetic feasibility negatives.",
    )
    parser.add_argument(
        "--feasibility-negative-power-scale-min",
        type=float,
        default=0.10,
        help="Lower bound of the random power scale used for synthetic feasibility negatives.",
    )
    parser.add_argument(
        "--feasibility-negative-power-scale-max",
        type=float,
        default=0.55,
        help="Upper bound of the random power scale used for synthetic feasibility negatives.",
    )
    parser.add_argument(
        "--regression-loss-weight",
        type=float,
        default=1.0,
        help="Relative weight for the regression heads.",
    )
    parser.add_argument(
        "--qos-sample-weight",
        type=float,
        default=1.5,
        help="How strongly to weight harder QoS rows in the physics-aware loss.",
    )
    parser.add_argument(
        "--infeasible-sample-weight",
        type=float,
        default=4.0,
        help="Extra weight for infeasible examples in the physics-aware loss.",
    )
    parser.add_argument(
        "--feasible-sample-boost",
        type=float,
        default=1.0,
        help="Extra weight for feasible examples to keep the head calibrated.",
    )
    parser.add_argument(
        "--physics-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the differentiable PASS QoS violation penalty.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Clip gradient norm to stabilize training.",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=8,
        help="ReduceLROnPlateau patience for validation loss.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Learning-rate reduction factor when validation loss plateaus.",
    )
    parser.add_argument(
        "--save-test-predictions",
        action="store_true",
        help="Write projected test predictions to CSV for inspection.",
    )
    return parser


def train_main(args: argparse.Namespace) -> TrainResult:
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.input_csv or args.output_csv:
        input_csv, output_csv = resolve_dataset_paths(
            args.input_csv, args.output_csv, args.search_root
        )
        bundle = load_dataset_pair(input_csv, output_csv)
    else:
        bundle = load_dataset_corpus(args.search_root)

    config = build_system_config(args)
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        feasibility_head=bool(args.feasibility_head),
        feasibility_conditioning=str(args.feasibility_conditioning),
    )

    targets = prepare_targets(bundle, config)
    split = split_indices(len(bundle.inputs), seed=args.seed)
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx

    x_train_raw = bundle.inputs[train_idx]
    x_val_raw = bundle.inputs[val_idx]
    x_test_raw = bundle.inputs[test_idx]

    y_train = targets[train_idx].astype(np.float32)
    y_val = targets[val_idx].astype(np.float32)
    y_test = targets[test_idx].astype(np.float32)

    if args.augment_user_permutations and bundle.schema.mode == "pass":
        x_train_raw, y_train = augment_pass_dataset(x_train_raw, y_train)

    train_feasibility, train_sample_weights = prepare_pass_supervision(
        bundle=bundle,
        config=config,
        train_idx=train_idx,
        augment_user_permutations=bool(args.augment_user_permutations and bundle.schema.mode == "pass"),
        qos_weight=args.qos_sample_weight,
        infeasible_weight=args.infeasible_sample_weight,
        feasible_boost=args.feasible_sample_boost,
    )
    val_feasibility, val_sample_weights = prepare_pass_supervision(
        bundle=bundle,
        config=config,
        train_idx=val_idx,
        augment_user_permutations=False,
        qos_weight=args.qos_sample_weight,
        infeasible_weight=args.infeasible_sample_weight,
        feasible_boost=args.feasible_sample_boost,
    )
    test_feasibility, test_sample_weights = prepare_pass_supervision(
        bundle=bundle,
        config=config,
        train_idx=test_idx,
        augment_user_permutations=False,
        qos_weight=args.qos_sample_weight,
        infeasible_weight=args.infeasible_sample_weight,
        feasible_boost=args.feasible_sample_boost,
    )

    input_scaler = fit_standardizer(x_train_raw)
    x_train = input_scaler.transform(x_train_raw).astype(np.float32)
    x_val = input_scaler.transform(x_val_raw).astype(np.float32)
    x_test = input_scaler.transform(x_test_raw).astype(np.float32)

    train_ds = SupervisedDataset(
        x_train,
        y_train,
        aux_targets=train_feasibility,
        sample_weights=train_sample_weights,
        physical_inputs=x_train_raw,
    )
    val_ds = SupervisedDataset(
        x_val,
        y_val,
        aux_targets=val_feasibility,
        sample_weights=val_sample_weights,
        physical_inputs=x_val_raw,
    )
    test_ds = SupervisedDataset(
        x_test,
        y_test,
        aux_targets=test_feasibility,
        sample_weights=test_sample_weights,
        physical_inputs=x_test_raw,
    )

    train_sampler = None
    if args.balance_by_qos and bundle.schema.mode == "pass":
        train_sampler = build_qos_balanced_sampler(x_train_raw)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(bundle.schema, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.SmoothL1Loss(reduction="none") if args.criterion == "huber" else nn.MSELoss(reduction="none")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=1e-6,
    )
    training_config = {
        "criterion": args.criterion,
        "augment_user_permutations": bool(args.augment_user_permutations),
        "balance_by_qos": bool(args.balance_by_qos),
        "feasibility_head": bool(args.feasibility_head),
        "feasibility_conditioning": str(args.feasibility_conditioning),
        "regression_loss_weight": float(args.regression_loss_weight),
        "feasibility_loss_weight": float(args.feasibility_loss_weight),
        "feasibility_negative_weight": float(args.feasibility_negative_weight),
        "feasibility_negative_candidates": int(args.feasibility_negative_candidates),
        "feasibility_negative_jitter": float(args.feasibility_negative_jitter),
        "feasibility_negative_power_scale_min": float(args.feasibility_negative_power_scale_min),
        "feasibility_negative_power_scale_max": float(args.feasibility_negative_power_scale_max),
        "qos_sample_weight": float(args.qos_sample_weight),
        "infeasible_sample_weight": float(args.infeasible_sample_weight),
        "feasible_sample_boost": float(args.feasible_sample_boost),
        "physics_loss_weight": float(args.physics_loss_weight),
        "grad_clip": float(args.grad_clip),
        "lr_scheduler_patience": int(args.lr_scheduler_patience),
        "lr_scheduler_factor": float(args.lr_scheduler_factor),
    }

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pos_loss, train_pow_loss = run_epoch(
            model,
            train_loader,
            device,
            bundle.schema,
            config,
            criterion,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            regression_loss_weight=args.regression_loss_weight,
            feasibility_loss_weight=args.feasibility_loss_weight,
            feasibility_negative_weight=args.feasibility_negative_weight,
            feasibility_negative_candidates=args.feasibility_negative_candidates,
            feasibility_negative_jitter=args.feasibility_negative_jitter,
            feasibility_negative_power_scale_min=args.feasibility_negative_power_scale_min,
            feasibility_negative_power_scale_max=args.feasibility_negative_power_scale_max,
            physics_loss_weight=args.physics_loss_weight,
        )
        val_loss, val_pos_loss, val_pow_loss = run_epoch(
            model,
            val_loader,
            device,
            bundle.schema,
            config,
            criterion,
            optimizer=None,
            regression_loss_weight=args.regression_loss_weight,
            feasibility_loss_weight=args.feasibility_loss_weight,
            feasibility_negative_weight=args.feasibility_negative_weight,
            feasibility_negative_candidates=0,
            feasibility_negative_jitter=args.feasibility_negative_jitter,
            feasibility_negative_power_scale_min=args.feasibility_negative_power_scale_min,
            feasibility_negative_power_scale_max=args.feasibility_negative_power_scale_max,
            physics_loss_weight=args.physics_loss_weight,
        )
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pos_loss": train_pos_loss,
                "train_pow_loss": train_pow_loss,
                "val_loss": val_loss,
                "val_pos_loss": val_pos_loss,
                "val_pow_loss": val_pow_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    test_loss, test_pos_loss, test_pow_loss = run_epoch(
        model,
        test_loader,
        device,
        bundle.schema,
        config,
        criterion,
        optimizer=None,
        regression_loss_weight=args.regression_loss_weight,
        feasibility_loss_weight=args.feasibility_loss_weight,
        feasibility_negative_weight=args.feasibility_negative_weight,
        feasibility_negative_candidates=0,
        feasibility_negative_jitter=args.feasibility_negative_jitter,
        feasibility_negative_power_scale_min=args.feasibility_negative_power_scale_min,
        feasibility_negative_power_scale_max=args.feasibility_negative_power_scale_max,
        physics_loss_weight=args.physics_loss_weight,
    )
    pred_result = predict_normalized(
        model,
        test_loader,
        device,
        bundle.schema,
        include_aux=True,
    )
    pred_norm, target_norm, feasibility_prob = pred_result
    test_metrics = compute_physical_metrics(
        inputs=bundle.inputs[test_idx],
        pred_norm=pred_norm,
        target_norm=target_norm,
        schema=bundle.schema,
        config=config,
    )
    test_metrics["test_loss_norm"] = test_loss
    test_metrics["test_pos_loss_norm"] = test_pos_loss
    test_metrics["test_pow_loss_norm"] = test_pow_loss
    if feasibility_prob is not None and bundle.schema.mode == "pass":
        feasibility_true = test_feasibility.reshape(-1)
        feasibility_pred = (feasibility_prob.reshape(-1) >= 0.5).astype(np.float32)
        feasibility_report = classification_like_report(feasibility_true, feasibility_pred)
        test_metrics["feasibility_accuracy"] = float(feasibility_report["accuracy"])
        test_metrics["feasibility_precision"] = float(feasibility_report["precision"])
        test_metrics["feasibility_recall"] = float(feasibility_report["recall"])
        test_metrics["feasibility_f1"] = float(feasibility_report["f1"])
        test_metrics["feasibility_balanced_accuracy"] = float(feasibility_report["balanced_accuracy"])
        test_metrics["feasibility_tp"] = float(feasibility_report["tp"])
        test_metrics["feasibility_tn"] = float(feasibility_report["tn"])
        test_metrics["feasibility_fp"] = float(feasibility_report["fp"])
        test_metrics["feasibility_fn"] = float(feasibility_report["fn"])
        test_metrics["feasibility_prob_mean"] = float(np.mean(feasibility_prob))
        test_metrics["feasibility_prob_min"] = float(np.min(feasibility_prob))
        test_metrics["feasibility_prob_max"] = float(np.max(feasibility_prob))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(args.artifact_root) / f"{bundle.schema.mode}_{timestamp}"
    save_artifacts(
        artifact_dir=artifact_dir,
        model=model,
        input_scaler=input_scaler,
        bundle=bundle,
        config=config,
        model_config=model_config,
        training_config=training_config,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
        test_metrics=test_metrics,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )

    if args.save_test_predictions:
        physical_preds = infer_outputs_from_model(
            pred_norm[:, : bundle.schema.position_dim] if bundle.schema.mode == "pass" else None,
            pred_norm[:, bundle.schema.position_dim :] if bundle.schema.mode == "pass" else pred_norm,
            bundle.schema,
            config,
        )
        test_frame = bundle.input_frame.iloc[test_idx].reset_index(drop=True).copy()
        pred_frame = pd.DataFrame(
            physical_preds, columns=[f"pred_{name}" for name in bundle.schema.output_columns]
        )
        target_physical = bundle.output_frame.iloc[test_idx].reset_index(drop=True).copy()
        if bundle.schema.mode == "pass":
            target_positions = canonicalize_pass_positions(
                target_physical.iloc[:, : bundle.schema.position_dim].to_numpy(dtype=np.float32)
            )
            target_physical.iloc[:, : bundle.schema.position_dim] = target_positions
        target_frame = target_physical.rename(columns={c: f"target_{c}" for c in target_physical.columns})
        pd.concat([test_frame, pred_frame, target_frame], axis=1).to_csv(
            artifact_dir / "test_predictions.csv", index=False
        )

    return TrainResult(
        artifact_dir=artifact_dir,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
        test_metrics=test_metrics,
    )


def main(argv: Sequence[str] | None = None) -> TrainResult:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = train_main(args)

    print("\nTraining complete.")
    print(f"Artifact directory: {result.artifact_dir}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best validation loss: {result.best_val_loss:.6f}")
    print(f"Test loss: {result.test_loss:.6f}")
    print(f"Test metrics: {json.dumps(result.test_metrics, indent=2)}")
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
