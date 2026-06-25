"""Dataset loading, canonicalization, scaling, and feasibility projection."""

from __future__ import annotations

from itertools import permutations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import (
    CONVENTIONAL_SCHEMA,
    INPUT_COLUMNS,
    PASS_POSITION_COLUMNS,
    PASS_POWER_COLUMNS,
    PASS_SCHEMA,
    DatasetSchema,
    SystemConfig,
    detect_schema,
    get_schema,
)


@dataclass(frozen=True)
class Standardizer:
    """Simple mean/std scaler for tabular features."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_arrays(cls, mean: np.ndarray, std: np.ndarray) -> "Standardizer":
        safe_std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=safe_std.astype(np.float32))


@dataclass(frozen=True)
class DatasetBundle:
    """Container for one fully loaded CSV pair."""

    schema: DatasetSchema
    inputs: np.ndarray
    outputs: np.ndarray
    input_frame: pd.DataFrame
    output_frame: pd.DataFrame


@dataclass(frozen=True)
class DatasetSplit:
    """Index split for train/validation/test partitions."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class RegressionDataset(Dataset):
    """Torch dataset returning inputs and output heads."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        schema: DatasetSchema,
    ) -> None:
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.schema = schema

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


class SupervisedDataset(Dataset):
    """Torch dataset returning regression and auxiliary PASS supervision."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        aux_targets: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        physical_inputs: np.ndarray | None = None,
    ) -> None:
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

        if physical_inputs is not None:
            physical_inputs = np.asarray(physical_inputs, dtype=np.float32)
            if physical_inputs.shape[0] != len(inputs):
                raise ValueError("Physical inputs must contain the same number of rows as inputs.")
            self.physical_inputs = torch.as_tensor(physical_inputs, dtype=torch.float32)
        else:
            self.physical_inputs = None

        if aux_targets is None:
            aux_targets = np.zeros((len(inputs), 1), dtype=np.float32)
        aux_targets = np.asarray(aux_targets, dtype=np.float32)
        if aux_targets.ndim == 1:
            aux_targets = aux_targets.reshape(-1, 1)
        self.aux_targets = torch.as_tensor(aux_targets, dtype=torch.float32)

        if sample_weights is None:
            sample_weights = np.ones((len(inputs), 1), dtype=np.float32)
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if sample_weights.ndim == 1:
            sample_weights = sample_weights.reshape(-1, 1)
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int):
        if self.physical_inputs is None:
            return (
                self.inputs[index],
                self.targets[index],
                self.aux_targets[index],
                self.sample_weights[index],
            )

        return (
            self.inputs[index],
            self.targets[index],
            self.aux_targets[index],
            self.sample_weights[index],
            self.physical_inputs[index],
        )


def _validate_columns(frame: pd.DataFrame, expected: Sequence[str], kind: str) -> None:
    actual = tuple(frame.columns.tolist())
    if actual != tuple(expected):
        raise ValueError(
            f"Unexpected {kind} columns.\nExpected: {tuple(expected)}\nActual:   {actual}"
        )


def load_dataset_pair(input_csv: str | Path, output_csv: str | Path) -> DatasetBundle:
    """Load a MATLAB-generated CSV pair and validate the schema."""

    input_path = Path(input_csv)
    output_path = Path(output_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not output_path.exists():
        raise FileNotFoundError(f"Output CSV not found: {output_path}")

    input_frame = pd.read_csv(input_path)
    output_frame = pd.read_csv(output_path)

    _validate_columns(input_frame, INPUT_COLUMNS, "input")
    schema = detect_schema(output_frame.columns.tolist())
    _validate_columns(output_frame, schema.output_columns, "output")

    if len(input_frame) != len(output_frame):
        raise ValueError(
            f"Row-count mismatch between input ({len(input_frame)}) and output ({len(output_frame)}) CSVs."
        )

    inputs = input_frame.to_numpy(dtype=np.float32, copy=True)
    outputs = output_frame.to_numpy(dtype=np.float32, copy=True)
    if np.isnan(inputs).any() or np.isnan(outputs).any():
        raise ValueError("NaN values detected in the dataset pair.")

    input_frame.attrs["source_path"] = str(input_path)
    output_frame.attrs["source_path"] = str(output_path)

    return DatasetBundle(
        schema=schema,
        inputs=inputs,
        outputs=outputs,
        input_frame=input_frame,
        output_frame=output_frame,
    )


def load_latest_pass_dataset_pair(
    search_roots: Sequence[str | Path],
) -> tuple[Path, Path]:
    """Find the most recent PASS dataset pair on disk."""

    candidates: list[tuple[float, Path, Path]] = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for input_path in root_path.rglob("dataset_input_pass*.csv"):
            output_name = input_path.name.replace("dataset_input_", "dataset_output_")
            output_path = input_path.with_name(output_name)
            if output_path.exists():
                stamp = max(input_path.stat().st_mtime, output_path.stat().st_mtime)
                candidates.append((stamp, input_path, output_path))

    if not candidates:
        raise FileNotFoundError(
            "No PASS dataset pair found. Generate one with generate_pass_dataset.m first."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, input_path, output_path = candidates[0]
    return input_path, output_path


def _index_pass_dataset_files(
    search_roots: Sequence[str | Path],
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Index PASS dataset input and output files by timestamp stamp."""

    input_files: dict[str, Path] = {}
    output_files: dict[str, Path] = {}
    seen_inputs: set[Path] = set()
    seen_outputs: set[Path] = set()

    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        for input_path in root_path.rglob("dataset_input_pass*.csv"):
            resolved = input_path.resolve()
            if resolved in seen_inputs:
                continue
            seen_inputs.add(resolved)
            stamp = input_path.name.replace("dataset_input_", "").replace(".csv", "")
            input_files[stamp] = input_path

        for output_path in root_path.rglob("dataset_output_pass*.csv"):
            resolved = output_path.resolve()
            if resolved in seen_outputs:
                continue
            seen_outputs.add(resolved)
            stamp = output_path.name.replace("dataset_output_", "").replace(".csv", "")
            output_files[stamp] = output_path

    return input_files, output_files


def find_complete_pass_dataset_pairs(
    search_roots: Sequence[str | Path],
) -> list[tuple[Path, Path]]:
    """Return all complete PASS dataset pairs found under the search roots."""

    input_files, output_files = _index_pass_dataset_files(search_roots)
    complete_stamps = sorted(set(input_files) & set(output_files))
    return [(input_files[stamp], output_files[stamp]) for stamp in complete_stamps]


def combine_dataset_bundles(
    bundles: Sequence[DatasetBundle],
) -> DatasetBundle:
    """Combine multiple homogeneous bundles into one corpus bundle."""

    bundles = list(bundles)
    if not bundles:
        raise ValueError("No bundles were provided.")

    schema = bundles[0].schema
    for bundle in bundles[1:]:
        if bundle.schema != schema:
            raise ValueError("Mixed PASS and conventional datasets are not supported in one corpus.")

    inputs = np.concatenate([bundle.inputs for bundle in bundles], axis=0)
    outputs = np.concatenate([bundle.outputs for bundle in bundles], axis=0)
    input_frame = pd.concat([bundle.input_frame for bundle in bundles], ignore_index=True)
    output_frame = pd.concat([bundle.output_frame for bundle in bundles], ignore_index=True)

    source_pairs: list[tuple[str, str]] = []
    for bundle in bundles:
        source_paths = bundle.input_frame.attrs.get("source_paths")
        if source_paths:
            source_pairs.extend(list(source_paths))
        else:
            source_input = bundle.input_frame.attrs.get("source_path", "")
            source_output = bundle.output_frame.attrs.get("source_path", "")
            if source_input and source_output:
                source_pairs.append((str(source_input), str(source_output)))

    if source_pairs:
        source_pairs_tuple = tuple(source_pairs)
        input_frame.attrs["source_paths"] = source_pairs_tuple
        output_frame.attrs["source_paths"] = source_pairs_tuple

    return DatasetBundle(
        schema=schema,
        inputs=inputs,
        outputs=outputs,
        input_frame=input_frame,
        output_frame=output_frame,
    )


def load_dataset_corpus(
    search_roots: Sequence[str | Path],
) -> DatasetBundle:
    """Load and concatenate every complete PASS dataset pair found on disk."""

    pairs = find_complete_pass_dataset_pairs(search_roots)
    if not pairs:
        raise FileNotFoundError(
            "No complete PASS dataset pairs found. Generate data with generate_pass_dataset.m first."
        )

    bundles = [load_dataset_pair(input_path, output_path) for input_path, output_path in pairs]
    return combine_dataset_bundles(bundles)


def augment_pass_dataset(
    inputs: np.ndarray,
    outputs: np.ndarray,
    canonicalize_outputs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment PASS rows using user/per-waveguide permutation symmetry."""

    if inputs.ndim != 2 or outputs.ndim != 2:
        raise ValueError("inputs and outputs must be 2D matrices.")
    if inputs.shape[1] != len(INPUT_COLUMNS):
        raise ValueError("Unexpected PASS input dimension.")
    if outputs.shape[1] != len(PASS_POSITION_COLUMNS) + len(PASS_POWER_COLUMNS):
        raise ValueError("Unexpected PASS output dimension.")

    augmented_inputs: list[np.ndarray] = []
    augmented_outputs: list[np.ndarray] = []
    user_perms = list(permutations(range(3)))

    for perm in user_perms:
        input_idx = [idx for user in perm for idx in (2 * user, 2 * user + 1)] + [6]
        pos_idx = [idx for waveguide in perm for idx in range(3 * waveguide, 3 * waveguide + 3)]
        pow_idx = [9 + waveguide for waveguide in perm]

        permuted_inputs = inputs[:, input_idx]
        permuted_outputs = np.concatenate(
            [outputs[:, pos_idx], outputs[:, pow_idx]],
            axis=1,
        )

        if canonicalize_outputs:
            permuted_outputs = permuted_outputs.copy()
            permuted_outputs[:, : len(PASS_POSITION_COLUMNS)] = canonicalize_pass_positions(
                permuted_outputs[:, : len(PASS_POSITION_COLUMNS)]
            )

        augmented_inputs.append(permuted_inputs.astype(np.float32, copy=False))
        augmented_outputs.append(permuted_outputs.astype(np.float32, copy=False))

    return (
        np.concatenate(augmented_inputs, axis=0),
        np.concatenate(augmented_outputs, axis=0),
    )


def audit_raw_pass_dataset_corpus(
    search_roots: Sequence[str | Path],
    config: SystemConfig | None = None,
) -> dict:
    """Summarize dataset completeness, validity, and training readiness."""

    config = config or SystemConfig()
    input_files, output_files = _index_pass_dataset_files(search_roots)
    complete_pairs = find_complete_pass_dataset_pairs(search_roots)
    complete_stamps = sorted(set(input_files) & set(output_files))
    missing_inputs = sorted(set(output_files) - set(input_files))
    missing_outputs = sorted(set(input_files) - set(output_files))

    pair_summaries: list[dict] = []
    combined_frames: list[pd.DataFrame] = []

    for stamp in complete_stamps:
        input_path = input_files[stamp]
        output_path = output_files[stamp]
        bundle = load_dataset_pair(input_path, output_path)
        pair_frame = pd.concat([bundle.input_frame, bundle.output_frame], axis=1)
        combined_frames.append(pair_frame)

        pos = bundle.outputs[:, : len(PASS_POSITION_COLUMNS)]
        poww = bundle.outputs[:, len(PASS_POSITION_COLUMNS) :]
        power_sum = poww.sum(axis=1)
        spacing_ok = True
        for wg in range(config.num_waveguides):
            block = np.sort(pos[:, wg * config.num_pinchers : (wg + 1) * config.num_pinchers], axis=1)
            spacing_ok = spacing_ok and bool(
                np.all(np.diff(block, axis=1) >= config.min_spacing_m - 1e-12)
            )

        pair_summaries.append(
            {
                "stamp": stamp,
                "input_rows": int(len(bundle.input_frame)),
                "output_rows": int(len(bundle.output_frame)),
                "qos_values": bundle.input_frame["QoS_R"].value_counts().to_dict(),
                "input_source": str(input_path),
                "output_source": str(output_path),
                "position_min": float(pos.min()),
                "position_max": float(pos.max()),
                "power_sum_min": float(power_sum.min()),
                "power_sum_max": float(power_sum.max()),
                "spacing_ok": spacing_ok,
            }
        )

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        duplicate_rows = int(combined.duplicated().sum())
        nan_count = int(combined.isna().sum().sum())
        qos_values = combined["QoS_R"].value_counts().sort_index().to_dict()
        user_variance = combined[list(INPUT_COLUMNS)].nunique().to_dict()
    else:
        combined = pd.DataFrame()
        duplicate_rows = 0
        nan_count = 0
        qos_values = {}
        user_variance = {}

    recommendations = []
    total_rows = int(len(combined))
    if len(complete_pairs) == 0:
        recommendations.append("No complete dataset pairs were found; generate a matching input/output CSV pair first.")
    if len(qos_values) <= 1:
        recommendations.append(
            "All complete data appear to use a single QoS value, so the DNN will learn one operating point well but will not generalize across QoS yet."
        )
    if duplicate_rows > 0:
        recommendations.append("Duplicates were detected; deduplicate before training if you want a cleaner effective dataset size.")
    if total_rows < 1000:
        recommendations.append("The corpus is small for a high-capacity DNN; use augmentation and consider generating more samples.")
    if total_rows >= 1000 and len(qos_values) == 1:
        recommendations.append("The sample count is reasonable for a proof-of-concept, but you should generate more QoS diversity for a production-grade surrogate.")

    return {
        "complete_pairs": [
            {
                "stamp": stamp,
                "input_path": str(input_files[stamp]),
                "output_path": str(output_files[stamp]),
                "input_rows": int(len(load_dataset_pair(input_files[stamp], output_files[stamp]).input_frame)),
            }
            for stamp in complete_stamps
        ],
        "missing_inputs": missing_inputs,
        "missing_outputs": missing_outputs,
        "pair_summaries": pair_summaries,
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "nan_count": nan_count,
        "qos_values": qos_values,
        "user_cardinality": user_variance,
        "recommendations": recommendations,
    }


def split_indices(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Create a reproducible random split."""

    if n_samples <= 0:
        raise ValueError("Cannot split an empty dataset.")

    ratios_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratios_sum, 1.0, atol=1e-6):
        raise ValueError("Split ratios must sum to 1.0.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    train_end = max(1, int(round(n_samples * train_ratio)))
    val_end = max(train_end + 1, int(round(n_samples * (train_ratio + val_ratio))))
    train_end = min(train_end, n_samples - 2) if n_samples >= 3 else max(1, n_samples - 2)
    val_end = min(val_end, n_samples - 1) if n_samples >= 2 else n_samples

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if len(val_idx) == 0 and len(test_idx) > 1:
        val_idx = test_idx[:1]
        test_idx = test_idx[1:]
    if len(test_idx) == 0 and len(val_idx) > 1:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]

    return DatasetSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def fit_standardizer(values: np.ndarray) -> Standardizer:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return Standardizer.from_arrays(mean=mean, std=std)


def canonicalize_pass_positions(
    positions_flat: np.ndarray,
    num_waveguides: int = 3,
    num_pinchers: int = 3,
) -> np.ndarray:
    """Sort the PA positions within each waveguide to remove label permutation noise."""

    if positions_flat.ndim == 1:
        positions_flat = positions_flat.reshape(1, -1)

    reshaped = positions_flat.reshape(-1, num_waveguides, num_pinchers)
    reshaped = np.sort(reshaped, axis=2)
    return reshaped.reshape(positions_flat.shape[0], -1)


def normalize_positions(positions_m: np.ndarray, config: SystemConfig) -> np.ndarray:
    return positions_m / config.position_bound_m


def denormalize_positions(positions_norm: np.ndarray, config: SystemConfig) -> np.ndarray:
    return positions_norm * config.position_bound_m


def normalize_powers(powers_w: np.ndarray, config: SystemConfig) -> np.ndarray:
    return powers_w / config.power_budget_w


def denormalize_powers(powers_norm: np.ndarray, config: SystemConfig) -> np.ndarray:
    return powers_norm * config.power_budget_w


def project_pass_positions(
    positions_m: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    """Project each waveguide's PA positions back into the feasible spacing interval."""

    if positions_m.ndim == 1:
        positions_m = positions_m.reshape(1, -1)

    if positions_m.shape[1] != len(PASS_POSITION_COLUMNS):
        raise ValueError("Unexpected PASS position dimension.")

    lower = -config.position_bound_m
    upper = config.position_bound_m
    spacing = config.min_spacing_m
    n_waveguides = config.num_waveguides
    n_pinchers = config.num_pinchers
    span = (n_pinchers - 1) * spacing
    if span > (upper - lower) + 1e-12:
        raise ValueError("No feasible interval exists for the requested spacing.")

    projected = np.zeros_like(positions_m, dtype=np.float32)
    grouped = np.sort(positions_m.reshape(-1, n_waveguides, n_pinchers), axis=2)

    for row_idx in range(grouped.shape[0]):
        for wg_idx in range(n_waveguides):
            x = grouped[row_idx, wg_idx].astype(np.float32).copy()
            for _ in range(3):
                x[0] = max(x[0], lower)
                for pincher_idx in range(1, n_pinchers):
                    x[pincher_idx] = max(x[pincher_idx], x[pincher_idx - 1] + spacing)

                x[-1] = min(x[-1], upper)
                for pincher_idx in range(n_pinchers - 2, -1, -1):
                    x[pincher_idx] = min(x[pincher_idx], x[pincher_idx + 1] - spacing)

                if x[0] < lower:
                    x += lower - x[0]
                if x[-1] > upper:
                    x -= x[-1] - upper

            projected[row_idx, wg_idx * n_pinchers : (wg_idx + 1) * n_pinchers] = x

    return projected


def project_pass_powers(powers_w: np.ndarray, config: SystemConfig) -> np.ndarray:
    """Clip and renormalize waveguide powers to respect the BS budget."""

    if powers_w.ndim == 1:
        powers_w = powers_w.reshape(1, -1)

    projected = np.clip(powers_w.astype(np.float32), 0.0, config.power_budget_w)
    totals = projected.sum(axis=1, keepdims=True)
    over_budget = totals[:, 0] > config.power_budget_w + 1e-12
    if np.any(over_budget):
        scale = config.power_budget_w / np.maximum(totals[over_budget], 1e-12)
        projected[over_budget] *= scale
    return projected


def project_pass_outputs(
    positions_m: np.ndarray,
    powers_w: np.ndarray,
    config: SystemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Project both PASS output heads into the feasible region."""

    return project_pass_positions(positions_m, config), project_pass_powers(powers_w, config)


def split_targets(
    outputs: np.ndarray,
    schema: DatasetSchema,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Split a raw output matrix into position and power targets."""

    if schema.mode == "pass":
        positions = outputs[:, : len(PASS_POSITION_COLUMNS)]
        powers = outputs[:, len(PASS_POSITION_COLUMNS) :]
        return positions, powers

    return None, outputs


def prepare_targets(
    outputs: np.ndarray,
    schema: DatasetSchema,
    config: SystemConfig,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Canonicalize and normalize targets for model training."""

    positions, powers = split_targets(outputs, schema)
    if schema.mode == "pass" and positions is not None:
        positions = canonicalize_pass_positions(positions)
        positions = normalize_positions(positions, config)
        powers = normalize_powers(powers, config)
        return positions.astype(np.float32), powers.astype(np.float32)

    return None, normalize_powers(powers, config).astype(np.float32)


def infer_outputs_from_model(
    positions_norm: np.ndarray | None,
    powers_norm: np.ndarray,
    schema: DatasetSchema,
    config: SystemConfig,
) -> np.ndarray:
    """Convert normalized model outputs back to the physical CSV layout."""

    powers_w = denormalize_powers(powers_norm, config)
    if schema.mode == "pass":
        if positions_norm is None:
            raise ValueError("PASS mode requires position outputs.")
        positions_m = denormalize_positions(positions_norm, config)
        positions_m, powers_w = project_pass_outputs(positions_m, powers_w, config)
        return np.concatenate([positions_m, powers_w], axis=1).astype(np.float32)

    powers_w = project_pass_powers(powers_w, config)
    return powers_w.astype(np.float32)
