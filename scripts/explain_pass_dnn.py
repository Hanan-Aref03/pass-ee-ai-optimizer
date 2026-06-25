"""Generate explainability reports for a trained PASS DNN artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pass_dnn.data import (
    combine_dataset_bundles,
    load_dataset_corpus,
    load_dataset_pair,
    split_indices,
)
from ml.pass_dnn.train import load_artifact_bundle, prepare_targets
from ml.pass_dnn.xai import explain_model, save_explainability_report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain a trained PASS DNN model.")
    parser.add_argument("--artifact-dir", required=True, type=str)
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument(
        "--search-root",
        action="append",
        default=["data/raw", "matlab/legacy/csv_data", "matlab/legacy"],
        help="Directories to search when --input-csv/--output-csv are omitted.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def load_dataset_from_args(args: argparse.Namespace, metadata: dict):
    if args.input_csv and args.output_csv:
        return load_dataset_pair(args.input_csv, args.output_csv)

    source_input = metadata.get("source_input_csv")
    source_output = metadata.get("source_output_csv")

    if isinstance(source_input, list) and isinstance(source_output, list):
        try:
            bundles = [load_dataset_pair(inp, out) for inp, out in zip(source_input, source_output)]
            return combine_dataset_bundles(bundles)
        except Exception:
            pass

    if source_input and source_output:
        try:
            return load_dataset_pair(source_input, source_output)
        except Exception:
            pass

    return load_dataset_corpus(args.search_root)


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    bundle = load_artifact_bundle(args.artifact_dir, device=args.device)
    dataset = load_dataset_from_args(args, bundle["metadata"])

    config = bundle["config"]
    schema = bundle["schema"]
    targets = prepare_targets(dataset, config)
    split = split_indices(len(dataset.inputs), seed=args.seed)
    test_idx = split.test_idx

    x_test = bundle["input_scaler"].transform(dataset.inputs[test_idx]).astype(np.float32)
    y_test = targets[test_idx].astype(np.float32)
    feature_names = schema.input_columns

    result = explain_model(
        model=bundle["model"],
        inputs=x_test,
        targets=y_test,
        schema=schema,
        feature_names=feature_names,
        device=bundle["device"],
        repeats=args.repeats,
    )

    output_dir = Path(args.artifact_dir) / "xai"
    save_explainability_report(result, output_dir)
    print(f"Explainability reports saved to: {output_dir}")
    print(json.dumps(result.permutation_importance.head(5).to_dict(orient="records"), indent=2))
    return output_dir


if __name__ == "__main__":  # pragma: no cover
    main()
