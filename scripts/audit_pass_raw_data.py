"""Inspect the raw PASS dataset folder and report training readiness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pass_dnn.data import audit_raw_pass_dataset_corpus


def main() -> None:
    report = audit_raw_pass_dataset_corpus(["data/raw"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
