"""Compute per-QoS-tier F1 for DNN-raw and DNN+refinement from saved outputs.

Requires refinement_results.json and refined_predictions.npy to already exist.

Usage:
    python -u scripts/per_tier_f1.py --artifact-dir <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ml.pass_dnn.train import load_artifact_bundle, load_dataset_corpus
from ml.pass_dnn.data import split_indices
from ml.pass_dnn.physics import evaluate_pass_batch


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1_val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--search-root", action="append",
        default=["data/raw", "matlab/legacy/csv_data", "matlab/legacy"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    refined_path = artifact_dir / "refined_predictions.npy"
    if not refined_path.exists():
        sys.exit(f"refined_predictions.npy not found in {artifact_dir}. Run refine_predictions.py first.")

    bundle = load_artifact_bundle(str(artifact_dir))
    model  = bundle["model"]
    config = bundle["config"]
    scaler = bundle["input_scaler"]

    corpus = load_dataset_corpus(args.search_root)
    split  = split_indices(len(corpus.inputs), seed=args.seed)
    test_idx = split.test_idx

    x_test_raw  = corpus.inputs[test_idx].astype(np.float32)
    x_test_scaled = scaler.transform(x_test_raw).astype(np.float32)
    gamma_col   = x_test_raw[:, -1]           # QoS γ is last input column

    # Ground-truth QoS
    true_phys   = corpus.outputs[test_idx].astype(np.float32)
    true_eval   = evaluate_pass_batch(x_test_raw, true_phys, config)
    true_qos    = true_eval.qos_satisfied.astype(int)

    # DNN raw QoS
    import torch
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_test_scaled)
        out = model(x_t)
        pred_norm = torch.cat([out["positions"], out["powers"]], dim=1).numpy()
    raw_phys = pred_norm.copy()
    raw_phys[:, :9] *= config.position_bound_m
    raw_phys[:, 9:] *= config.power_budget_w
    raw_eval  = evaluate_pass_batch(x_test_raw, raw_phys, config)
    raw_qos   = raw_eval.qos_satisfied.astype(int)

    # Refined QoS
    refined_phys = np.load(refined_path)
    ref_eval  = evaluate_pass_batch(x_test_raw, refined_phys, config)
    ref_qos   = ref_eval.qos_satisfied.astype(int)

    tiers = sorted(np.unique(np.round(gamma_col, 6)))

    header = f"{'gamma':>6} {'N':>5} {'TrueQoS':>8} {'RawSat':>8} {'RawF1':>8} {'RefSat':>8} {'RefF1':>8}"
    print(header)
    print("-" * len(header))

    tier_results = {}
    for g in tiers:
        mask = np.isclose(gamma_col, g, atol=1e-5)
        n = mask.sum()
        true_sat = true_qos[mask].mean()
        raw_sat  = raw_qos[mask].mean()
        ref_sat  = ref_qos[mask].mean()
        _, _, raw_f1 = f1(true_qos[mask], raw_qos[mask])
        _, _, ref_f1 = f1(true_qos[mask], ref_qos[mask])
        print(f"{g:>6.1f} {n:>5d} {true_sat:>8.3f} {raw_sat:>8.3f} {raw_f1:>8.3f} {ref_sat:>8.3f} {ref_f1:>3.3f}")
        tier_results[str(round(g, 6))] = {
            "n": int(n), "true_sat": round(true_sat, 4),
            "raw_sat": round(raw_sat, 4), "raw_f1": round(raw_f1, 4),
            "ref_sat": round(ref_sat, 4), "ref_f1": round(ref_f1, 4),
        }

    # Overall
    _, _, ov_raw_f1 = f1(true_qos, raw_qos)
    _, _, ov_ref_f1 = f1(true_qos, ref_qos)
    print("-" * len(header))
    print(f"{'ALL':>6} {len(true_qos):>5d} {true_qos.mean():>8.3f} {raw_qos.mean():>8.3f} {ov_raw_f1:>8.3f} {ref_qos.mean():>8.3f} {ov_ref_f1:>8.3f}")

    out_path = artifact_dir / "pertier_f1.json"
    out_path.write_text(json.dumps(tier_results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()