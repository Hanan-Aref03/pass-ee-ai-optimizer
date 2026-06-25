"""Merge all complete raw PASS dataset pairs into one corpus export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pass_dnn.data import audit_raw_pass_dataset_corpus, load_dataset_corpus


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge PASS raw CSV pairs into a single corpus.")
    parser.add_argument(
        "--search-root",
        action="append",
        default=["data/raw"],
        help="Directories to search for complete PASS dataset pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory where the merged corpus will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    bundle = load_dataset_corpus(args.search_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / "pass_merged_corpus_input.csv"
    output_path = output_dir / "pass_merged_corpus_output.csv"
    manifest_path = output_dir / "pass_merged_corpus_manifest.json"

    bundle.input_frame.to_csv(input_path, index=False)
    bundle.output_frame.to_csv(output_path, index=False)

    audit = audit_raw_pass_dataset_corpus(args.search_root)
    manifest = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "total_rows": int(len(bundle.inputs)),
        "qos_values": audit.get("qos_values", {}),
        "source_pairs": audit.get("complete_pairs", []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
