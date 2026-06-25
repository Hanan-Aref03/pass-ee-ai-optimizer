"""Rename raw PASS dataset files into QoS-aware, representative names."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pass_dnn.data import find_complete_pass_dataset_pairs, load_dataset_pair


def qos_tag(value: float) -> str:
    """Format QoS values into compact filenames such as ``0p5``."""

    return format(float(value), "g").replace(".", "p")


def representative_name(prefix: str, qos_value: float, row_count: int, run_index: int) -> str:
    """Build a descriptive filename for one PASS dataset pair."""

    return f"{prefix}_pass_qos_{qos_tag(qos_value)}_n{row_count:04d}_run{run_index:02d}.csv"


def rename_pairs(dry_run: bool = False) -> list[dict]:
    """Rename all complete PASS dataset pairs by QoS and row count."""

    pairs = []
    for input_path, output_path in find_complete_pass_dataset_pairs([RAW_DIR]):
        bundle = load_dataset_pair(input_path, output_path)
        qos_values = bundle.input_frame["QoS_R"].astype(float).round(6).unique()
        if len(qos_values) != 1:
            raise ValueError(f"Expected a single QoS value in {input_path}, found {qos_values.tolist()}")
        pairs.append(
            {
                "input_path": input_path,
                "output_path": output_path,
                "qos": float(qos_values[0]),
                "rows": int(len(bundle.inputs)),
                "stamp": input_path.stat().st_mtime,
            }
        )

    pairs.sort(key=lambda item: (item["qos"], -item["rows"], item["stamp"], item["input_path"].name))

    plan: list[dict] = []
    grouped: dict[float, list[dict]] = {}
    for item in pairs:
        grouped.setdefault(item["qos"], []).append(item)

    for qos_value in sorted(grouped):
        for run_index, item in enumerate(grouped[qos_value], start=1):
            target_input = RAW_DIR / representative_name("dataset_input", qos_value, item["rows"], run_index)
            target_output = RAW_DIR / representative_name("dataset_output", qos_value, item["rows"], run_index)
            plan.append(
                {
                    "source_input": item["input_path"],
                    "source_output": item["output_path"],
                    "target_input": target_input,
                    "target_output": target_output,
                }
            )

    renames: list[dict] = []
    for item in plan:
        source_input = item["source_input"]
        source_output = item["source_output"]
        target_input = item["target_input"]
        target_output = item["target_output"]

        if source_input.name != target_input.name:
            if target_input.exists() and target_input.resolve() != source_input.resolve():
                raise FileExistsError(f"Target input already exists: {target_input}")
            if not dry_run:
                source_input.rename(target_input)

        if source_output.name != target_output.name:
            if target_output.exists() and target_output.resolve() != source_output.resolve():
                raise FileExistsError(f"Target output already exists: {target_output}")
            if not dry_run:
                source_output.rename(target_output)

        renames.append(
            {
                "from_input": str(source_input),
                "to_input": str(target_input),
                "from_output": str(source_output),
                "to_output": str(target_output),
            }
        )

    return renames


def rename_orphan_outputs(dry_run: bool = False) -> list[dict]:
    """Rename output-only orphan CSVs so they are easy to spot."""

    renames: list[dict] = []
    for output_path in sorted(RAW_DIR.glob("dataset_output_pass*.csv")):
        input_name = output_path.name.replace("dataset_output_", "dataset_input_")
        input_path = output_path.with_name(input_name)
        if input_path.exists():
            continue

        if "orphan" in output_path.stem:
            continue

        suffix = output_path.stem.replace("dataset_output_pass_", "")
        if suffix.startswith("ml_"):
            suffix = suffix[3:]
        target_path = output_path.with_name(f"dataset_output_pass_orphan_{suffix}.csv")
        if target_path.exists() and target_path.resolve() != output_path.resolve():
            raise FileExistsError(f"Target orphan output already exists: {target_path}")

        if not dry_run:
            output_path.rename(target_path)

        renames.append({"from_output": str(output_path), "to_output": str(target_path)})

    return renames


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rename raw PASS CSVs to representative names.")
    parser.add_argument("--dry-run", action="store_true", help="Print the rename plan without changing files.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    pair_renames = rename_pairs(dry_run=args.dry_run)
    orphan_renames = rename_orphan_outputs(dry_run=args.dry_run)

    print(f"Pair renames: {len(pair_renames)}")
    for entry in pair_renames:
        print(f"{entry['from_input']} -> {entry['to_input']}")
        print(f"{entry['from_output']} -> {entry['to_output']}")

    print(f"Orphan renames: {len(orphan_renames)}")
    for entry in orphan_renames:
        print(f"{entry['from_output']} -> {entry['to_output']}")


if __name__ == "__main__":  # pragma: no cover
    main()
