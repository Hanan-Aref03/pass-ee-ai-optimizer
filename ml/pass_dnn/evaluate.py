"""Evaluate a saved PASS DNN artifact on a dataset pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SupervisedDataset, combine_dataset_bundles, load_dataset_pair, split_indices
from .schema import SystemConfig
from .train import (
    compute_physical_metrics,
    load_artifact_bundle,
    predict_normalized,
    prepare_targets,
    run_epoch,
)
from .supervision import build_pass_training_supervision


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved PASS DNN model.")
    parser.add_argument("--artifact-dir", required=True, type=str)
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    bundle = load_artifact_bundle(args.artifact_dir, device=args.device)

    if args.input_csv and args.output_csv:
        dataset = load_dataset_pair(args.input_csv, args.output_csv)
    else:
        metadata = bundle["metadata"]
        source_input = metadata.get("source_input_csv")
        source_output = metadata.get("source_output_csv")
        if isinstance(source_input, list) and isinstance(source_output, list):
            try:
                paired_bundles = [
                    load_dataset_pair(input_path, output_path)
                    for input_path, output_path in zip(source_input, source_output)
                ]
                dataset = combine_dataset_bundles(paired_bundles)
            except Exception:
                dataset = None
        elif source_input and source_output:
            try:
                dataset = load_dataset_pair(source_input, source_output)
            except Exception:
                dataset = None
        else:
            dataset = None

        if dataset is None:
            raise ValueError(
                "Provide --input-csv and --output-csv when the artifact metadata does not contain usable source paths."
            )

    config: SystemConfig = bundle["config"]
    schema = bundle["schema"]
    inputs = dataset.inputs.astype(np.float32)
    targets = prepare_targets(dataset, config)
    split = split_indices(len(inputs), seed=args.seed)
    test_idx = split.test_idx
    input_scaler = bundle["input_scaler"]
    model = bundle["model"]
    device = bundle["device"]

    x_test = input_scaler.transform(inputs[test_idx]).astype(np.float32)
    y_test = targets[test_idx].astype(np.float32)
    if schema.mode == "pass":
        test_feasibility, test_sample_weights = build_pass_training_supervision(
            inputs=dataset.inputs[test_idx],
            outputs=dataset.outputs[test_idx],
            config=config,
        )
    else:
        test_feasibility = np.ones((len(test_idx), 1), dtype=np.float32)
        test_sample_weights = np.ones((len(test_idx), 1), dtype=np.float32)

    test_ds = SupervisedDataset(
        x_test,
        y_test,
        aux_targets=test_feasibility,
        sample_weights=test_sample_weights,
        physical_inputs=dataset.inputs[test_idx],
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    criterion_name = bundle["metadata"].get("training_config", {}).get("criterion", "huber")
    criterion = nn.SmoothL1Loss(reduction="none") if criterion_name == "huber" else nn.MSELoss(reduction="none")
    training_config = bundle["metadata"].get("training_config", {})
    test_loss, _, _ = run_epoch(
        model,
        test_loader,
        device,
        schema,
        config,
        criterion,
        optimizer=None,
        regression_loss_weight=float(training_config.get("regression_loss_weight", 1.0)),
        feasibility_loss_weight=float(training_config.get("feasibility_loss_weight", 1.0)),
        feasibility_negative_weight=float(training_config.get("feasibility_negative_weight", 1.0)),
        feasibility_negative_candidates=0,
        feasibility_negative_jitter=float(training_config.get("feasibility_negative_jitter", 0.12)),
        feasibility_negative_power_scale_min=float(
            training_config.get("feasibility_negative_power_scale_min", 0.10)
        ),
        feasibility_negative_power_scale_max=float(
            training_config.get("feasibility_negative_power_scale_max", 0.55)
        ),
        physics_loss_weight=float(training_config.get("physics_loss_weight", 1.0)),
    )
    pred_result = predict_normalized(model, test_loader, device, schema, include_aux=True)
    pred_norm, target_norm, feasibility_prob = pred_result
    metrics = compute_physical_metrics(
        inputs=dataset.inputs[test_idx],
        pred_norm=pred_norm,
        target_norm=target_norm,
        schema=schema,
        config=config,
    )
    metrics["test_loss_norm"] = test_loss
    if feasibility_prob is not None and schema.mode == "pass":
        from .metrics import classification_like_report

        feasibility_pred = (feasibility_prob.reshape(-1) >= 0.5).astype(np.float32)
        feasibility_true = test_feasibility.reshape(-1)
        feasibility_report = classification_like_report(feasibility_true, feasibility_pred)
        metrics["feasibility_accuracy"] = float(feasibility_report["accuracy"])
        metrics["feasibility_precision"] = float(feasibility_report["precision"])
        metrics["feasibility_recall"] = float(feasibility_report["recall"])
        metrics["feasibility_f1"] = float(feasibility_report["f1"])
        metrics["feasibility_balanced_accuracy"] = float(feasibility_report["balanced_accuracy"])
        metrics["feasibility_tp"] = float(feasibility_report["tp"])
        metrics["feasibility_tn"] = float(feasibility_report["tn"])
        metrics["feasibility_fp"] = float(feasibility_report["fp"])
        metrics["feasibility_fn"] = float(feasibility_report["fn"])
        metrics["feasibility_prob_mean"] = float(np.mean(feasibility_prob))
        metrics["feasibility_prob_min"] = float(np.min(feasibility_prob))
        metrics["feasibility_prob_max"] = float(np.max(feasibility_prob))

    report_path = Path(args.artifact_dir) / "evaluation_report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(Path(args.artifact_dir) / "evaluation_report.csv", index=False)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":  # pragma: no cover
    main()
