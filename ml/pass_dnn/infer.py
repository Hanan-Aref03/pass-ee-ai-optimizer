"""Run inference with a saved PASS DNN artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .schema import INPUT_COLUMNS
from .train import load_artifact_bundle, predict_frame


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer PASS configurations from user layouts.")
    parser.add_argument("--artifact-dir", required=True, type=str)
    parser.add_argument("--input-csv", required=True, type=str)
    parser.add_argument("--output-csv", required=True, type=str)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    bundle = load_artifact_bundle(args.artifact_dir, device=args.device)

    input_frame = pd.read_csv(args.input_csv)
    if tuple(input_frame.columns.tolist()) != INPUT_COLUMNS:
        raise ValueError(
            f"Unexpected input columns.\nExpected: {INPUT_COLUMNS}\nActual:   {tuple(input_frame.columns.tolist())}"
        )

    predictions = predict_frame(input_frame, bundle, project=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predictions written to: {output_path}")
    return output_path


if __name__ == "__main__":  # pragma: no cover
    main()

