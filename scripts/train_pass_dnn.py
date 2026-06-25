"""Convenience wrapper for the PASS DNN trainer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pass_dnn.train import main


if __name__ == "__main__":  # pragma: no cover
    main()
