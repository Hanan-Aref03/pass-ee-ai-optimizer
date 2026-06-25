"""Explainability helpers for the PASS DNN surrogate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .schema import DatasetSchema


@dataclass(frozen=True)
class ExplainabilityResult:
    """Container for global explainability artifacts."""

    permutation_importance: pd.DataFrame
    gradient_saliency: pd.DataFrame
    baseline_loss: float


def _combined_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    schema: DatasetSchema,
    criterion: nn.Module,
) -> torch.Tensor:
    outputs = model(inputs)
    if schema.mode == "pass":
        pos_target = targets[:, : schema.position_dim]
        pow_target = targets[:, schema.position_dim :]
        return criterion(outputs["positions"], pos_target) + criterion(outputs["powers"], pow_target)
    return criterion(outputs["powers"], targets)


@torch.no_grad()
def _baseline_loss(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    schema: DatasetSchema,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    loss = _combined_loss(model, x, y, schema, criterion)
    return float(loss.item())


def permutation_importance(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    schema: DatasetSchema,
    feature_names: Sequence[str],
    device: torch.device,
    criterion: nn.Module | None = None,
    repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate global feature importance via permutation on the held-out set."""

    criterion = criterion or nn.SmoothL1Loss(reduction="mean")
    baseline = _baseline_loss(model, inputs, targets, schema, device, criterion)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for feature_idx, feature_name in enumerate(feature_names):
        deltas: list[float] = []
        for _ in range(repeats):
            permuted = inputs.copy()
            rng.shuffle(permuted[:, feature_idx])
            loss = _baseline_loss(model, permuted, targets, schema, device, criterion)
            deltas.append(loss - baseline)

        rows.append(
            {
                "feature": feature_name,
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas)),
                "importance_min": float(np.min(deltas)),
                "importance_max": float(np.max(deltas)),
            }
        )

    importance = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return importance


def gradient_saliency(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    schema: DatasetSchema,
    feature_names: Sequence[str],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> pd.DataFrame:
    """Compute mean absolute gradient and gradient*x attributions."""

    criterion = criterion or nn.SmoothL1Loss(reduction="mean")
    model.eval()

    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    x.requires_grad_(True)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    loss = _combined_loss(model, x, y, schema, criterion)
    loss.backward()

    grads = x.grad.detach().cpu().numpy()
    values = x.detach().cpu().numpy()
    rows = []
    for idx, feature_name in enumerate(feature_names):
        rows.append(
            {
                "feature": feature_name,
                "mean_abs_grad": float(np.mean(np.abs(grads[:, idx]))),
                "mean_abs_grad_x_input": float(np.mean(np.abs(grads[:, idx] * values[:, idx]))),
            }
        )

    saliency = pd.DataFrame(rows).sort_values("mean_abs_grad_x_input", ascending=False).reset_index(drop=True)
    return saliency


def explain_model(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    schema: DatasetSchema,
    feature_names: Sequence[str],
    device: torch.device,
    repeats: int = 5,
    seed: int = 42,
    criterion: nn.Module | None = None,
) -> ExplainabilityResult:
    """Generate both permutation-importance and gradient-based explanations."""

    criterion = criterion or nn.SmoothL1Loss(reduction="mean")
    perm = permutation_importance(
        model=model,
        inputs=inputs,
        targets=targets,
        schema=schema,
        feature_names=feature_names,
        device=device,
        criterion=criterion,
        repeats=repeats,
        seed=seed,
    )
    sal = gradient_saliency(
        model=model,
        inputs=inputs,
        targets=targets,
        schema=schema,
        feature_names=feature_names,
        device=device,
        criterion=criterion,
    )
    baseline = _baseline_loss(model, inputs, targets, schema, device, criterion)
    return ExplainabilityResult(permutation_importance=perm, gradient_saliency=sal, baseline_loss=baseline)


def save_explainability_report(
    result: ExplainabilityResult,
    output_dir: str | Path,
) -> None:
    """Persist explainability outputs to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.permutation_importance.to_csv(output_path / "permutation_importance.csv", index=False)
    result.gradient_saliency.to_csv(output_path / "gradient_saliency.csv", index=False)

    summary = {
        "baseline_loss": result.baseline_loss,
        "top_permutation_features": result.permutation_importance.head(3)["feature"].tolist(),
        "top_gradient_features": result.gradient_saliency.head(3)["feature"].tolist(),
    }
    (output_path / "xai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
