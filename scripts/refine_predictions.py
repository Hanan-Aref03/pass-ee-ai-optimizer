"""Test-time gradient refinement to boost QoS F1.

Usage:
    python -u scripts/refine_predictions.py --artifact-dir <path> [--refine-steps 150]

For each test sample, starts from the DNN prediction and runs Adam on the
differentiable physics penalty until QoS violation is minimised.  Prints
before/after F1 comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ml.pass_dnn.train import load_artifact_bundle, load_dataset_corpus
from ml.pass_dnn.data import split_indices, fit_standardizer
from ml.pass_dnn.physics import evaluate_pass_batch
from ml.pass_dnn.torch_physics import (
    evaluate_pass_batch_torch,
    physics_penalty_from_outputs,
    to_physical_pass_outputs,
)


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def dnn_predict(model, x_scaled: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Return normalised DNN predictions (positions in Tanh, powers in [0,1])."""
    model.eval()
    all_preds = []
    x_t = torch.tensor(x_scaled, dtype=torch.float32)
    for start in range(0, len(x_t), batch_size):
        batch = x_t[start:start + batch_size]
        out = model(batch)
        preds = torch.cat([out["positions"], out["powers"]], dim=1)
        all_preds.append(preds.cpu())
    return torch.cat(all_preds, dim=0).numpy()


def _run_adam(
    inputs_raw: torch.Tensor,
    init_positions: torch.Tensor,
    init_powers: torch.Tensor,
    config,
    n_steps: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single Adam run; returns (positions, powers, per-sample penalty)."""
    positions = init_positions.clone().detach().requires_grad_(True)
    powers = init_powers.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([positions, powers], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)

    for _ in range(n_steps):
        opt.zero_grad()
        outputs_cat = torch.cat([positions, powers], dim=1)
        phys = to_physical_pass_outputs(outputs_cat, config)
        penalties = physics_penalty_from_outputs(inputs_raw, phys, config)
        penalties.sum().backward()
        opt.step()
        scheduler.step()
        with torch.no_grad():
            positions.data.clamp_(-1.0, 1.0)
            powers.data.clamp_(0.0, 1.0)

    with torch.no_grad():
        outputs_cat = torch.cat([positions, powers], dim=1)
        phys = to_physical_pass_outputs(outputs_cat, config)
        final_penalties = physics_penalty_from_outputs(inputs_raw, phys, config)
    return positions.detach(), powers.detach(), final_penalties.detach()


def refine_batch(
    inputs_raw: torch.Tensor,
    init_positions: torch.Tensor,
    init_powers: torch.Tensor,
    config,
    n_steps: int = 300,
    lr: float = 1e-2,
    n_restarts: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-of-N Adam restarts per sample.

    Restart 0 = DNN prediction; restarts 1..N-1 = random in feasible box.
    Keeps whichever restart has the lowest physics penalty per sample.
    """
    B = inputs_raw.shape[0]
    best_pos = init_positions.clone()
    best_pow = init_powers.clone()

    with torch.no_grad():
        outputs_cat = torch.cat([best_pos, best_pow], dim=1)
        phys = to_physical_pass_outputs(outputs_cat, config)
        best_pen = physics_penalty_from_outputs(inputs_raw, phys, config)

    for k in range(n_restarts):
        if k == 0:
            pos_init = init_positions.clone()
            pow_init = init_powers.clone()
        else:
            pos_init = torch.rand(B, init_positions.shape[1]) * 2 - 1
            pow_init = torch.rand(B, init_powers.shape[1])

        pos_r, pow_r, pen_r = _run_adam(inputs_raw, pos_init, pow_init, config, n_steps, lr)

        improved = pen_r < best_pen
        best_pos[improved] = pos_r[improved]
        best_pow[improved] = pow_r[improved]
        best_pen[improved] = pen_r[improved]

    return best_pos, best_pow


def evaluate_qos(inputs_raw: np.ndarray, pred_norm: np.ndarray, config) -> np.ndarray:
    """Return bool array: True where all-user QoS is satisfied."""
    pos_bound = config.position_bound_m
    pwr_budget = config.power_budget_w
    pred_phys = pred_norm.copy()
    pred_phys[:, :9] *= pos_bound
    pred_phys[:, 9:] *= pwr_budget
    result = evaluate_pass_batch(inputs_raw, pred_phys, config)
    return result.qos_satisfied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--search-root",
        action="append",
        default=["data/raw", "matlab/legacy/csv_data", "matlab/legacy"],
    )
    parser.add_argument("--refine-steps", type=int, default=300)
    parser.add_argument("--refine-lr", type=float, default=1e-2)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-only", action="store_true",
                        help="All restarts use random init (no DNN warm-start) — ablation baseline")
    args = parser.parse_args()

    print(f"Loading artifact: {args.artifact_dir}")
    bundle = load_artifact_bundle(args.artifact_dir)
    model = bundle["model"]
    config = bundle["config"]
    input_scaler = bundle["input_scaler"]

    corpus = load_dataset_corpus(args.search_root)
    split = split_indices(len(corpus.inputs), seed=args.seed)
    test_idx = split.test_idx

    x_test_raw = corpus.inputs[test_idx].astype(np.float32)
    x_test_scaled = input_scaler.transform(x_test_raw).astype(np.float32)

    # Ground-truth QoS satisfaction from MATLAB targets (raw physical outputs)
    raw_outputs_test = corpus.outputs[test_idx].astype(np.float32)
    true_eval = evaluate_pass_batch(x_test_raw, raw_outputs_test, config)
    true_qos = true_eval.qos_satisfied

    print(f"Test samples : {len(x_test_raw)}")
    print(f"True QoS sat : {true_qos.mean():.3f}")

    # --- Raw DNN prediction ---
    print("\nRunning DNN forward pass...")
    pred_norm = dnn_predict(model, x_test_scaled, batch_size=args.batch_size)
    pred_qos_raw = evaluate_qos(x_test_raw, pred_norm, config)
    raw_report = _classification_report(true_qos.astype(int), pred_qos_raw.astype(int))
    print(f"  QoS sat (DNN raw) : {pred_qos_raw.mean():.3f}")
    print(f"  F1  (DNN raw)     : {raw_report['f1']:.4f}")
    print(f"  P/R               : {raw_report['precision']:.4f} / {raw_report['recall']:.4f}")

    # --- DNN raw rate/EE baseline ---
    raw_phys = pred_norm.copy()
    raw_phys[:, :9] *= config.position_bound_m
    raw_phys[:, 9:] *= config.power_budget_w
    raw_eval = evaluate_pass_batch(x_test_raw, raw_phys, config)

    # --- Refinement ---
    mode_label = "random-only (ablation)" if args.random_only else "DNN warm-start"
    print(f"\nRefining with Adam ({args.refine_steps} steps, lr={args.refine_lr}, "
          f"{args.n_restarts} restarts, {mode_label})...")
    import time as _time
    t_refine_start = _time.perf_counter()
    x_raw_t = torch.tensor(x_test_raw, dtype=torch.float32)
    # For --random-only: DNN prediction is still computed above but we will
    # override init_pos/init_pow to random in refine_batch via n_restarts_dnn=0.
    init_pos = torch.tensor(pred_norm[:, :9], dtype=torch.float32)
    init_pow = torch.tensor(pred_norm[:, 9:], dtype=torch.float32)

    refined_pos_parts = []
    refined_pow_parts = []
    B = args.batch_size
    n = len(x_raw_t)
    for start in range(0, n, B):
        end = min(start + B, n)
        b_pos = torch.rand(end - start, init_pos.shape[1]) * 2 - 1 if args.random_only else init_pos[start:end]
        b_pow = torch.rand(end - start, init_pow.shape[1]) if args.random_only else init_pow[start:end]
        r_pos, r_pow = refine_batch(
            x_raw_t[start:end],
            b_pos,
            b_pow,
            config,
            n_steps=args.refine_steps,
            lr=args.refine_lr,
            n_restarts=args.n_restarts,
        )
        refined_pos_parts.append(r_pos)
        refined_pow_parts.append(r_pow)
        pct = 100 * end / n
        print(f"  [{pct:5.1f}%] batch {start//B + 1} done", flush=True)

    t_refine_end = _time.perf_counter()
    refine_total_s = t_refine_end - t_refine_start
    refine_per_sample_ms = 1000 * refine_total_s / n

    refined_norm = np.concatenate(
        [
            torch.cat(refined_pos_parts).numpy(),
            torch.cat(refined_pow_parts).numpy(),
        ],
        axis=1,
    )
    pred_qos_refined = evaluate_qos(x_test_raw, refined_norm, config)
    ref_report = _classification_report(true_qos.astype(int), pred_qos_refined.astype(int))

    # --- Rate/EE metrics after refinement ---
    refined_phys = refined_norm.copy()
    refined_phys[:, :9] *= config.position_bound_m
    refined_phys[:, 9:] *= config.power_budget_w
    refined_eval = evaluate_pass_batch(x_test_raw, refined_phys, config)

    # Ground truth metrics
    true_sr_mean  = float(true_eval.sum_rate.mean())
    true_ee_mean  = float(true_eval.energy_efficiency.mean())
    # DNN raw metrics
    raw_sr_mean   = float(raw_eval.sum_rate.mean())
    raw_sr_mae    = float(np.abs(raw_eval.sum_rate - true_eval.sum_rate).mean())
    raw_ee_mean   = float(raw_eval.energy_efficiency.mean())
    # Refined metrics
    ref_sr_mean   = float(refined_eval.sum_rate.mean())
    ref_sr_mae    = float(np.abs(refined_eval.sum_rate - true_eval.sum_rate).mean())
    ref_ee_mean   = float(refined_eval.energy_efficiency.mean())

    print(f"\n{'='*60}")
    print(f"  QoS sat (refined)          : {pred_qos_refined.mean():.3f}")
    print(f"  F1  (refined)              : {ref_report['f1']:.4f}")
    print(f"  P/R                        : {ref_report['precision']:.4f} / {ref_report['recall']:.4f}")
    print(f"  Delta F1 vs raw DNN        : {ref_report['f1'] - raw_report['f1']:+.4f}")
    print(f"  Refinement time            : {refine_total_s:.1f}s total  |  {refine_per_sample_ms:.1f} ms/sample")
    print(f"{'='*60}")
    print(f"\n  Rate/EE comparison:")
    print(f"  {'Metric':<28} {'MATLAB':>10} {'DNN raw':>10} {'Refined':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Mean sum-rate (b/s/Hz)':<28} {true_sr_mean:>10.3f} {raw_sr_mean:>10.3f} {ref_sr_mean:>10.3f}")
    print(f"  {'Sum-rate MAE (b/s/Hz)':<28} {'—':>10} {raw_sr_mae:>10.3f} {ref_sr_mae:>10.3f}")
    print(f"  {'Mean EE (bits/J)':<28} {true_ee_mean:>10.2f} {raw_ee_mean:>10.2f} {ref_ee_mean:>10.2f}")
    print(f"{'='*60}")

    suffix = "_random_only" if args.random_only else ""
    pred_npy_path = Path(args.artifact_dir) / f"refined_predictions{suffix}.npy"
    np.save(pred_npy_path, refined_phys)

    results = {
        "artifact_dir": str(args.artifact_dir),
        "mode": "random_only" if args.random_only else "dnn_warmstart",
        "refine_steps": args.refine_steps,
        "refine_lr": args.refine_lr,
        "n_restarts": args.n_restarts,
        "n_test": int(len(x_test_raw)),
        "refine_total_s": round(refine_total_s, 2),
        "refine_per_sample_ms": round(refine_per_sample_ms, 2),
        "true_sat_rate": float(true_qos.mean()),
        "raw": raw_report,
        "refined": ref_report,
        "rate_ee": {
            "true_sr_mean":  true_sr_mean,
            "true_ee_mean":  true_ee_mean,
            "raw_sr_mean":   raw_sr_mean,
            "raw_sr_mae":    raw_sr_mae,
            "raw_ee_mean":   raw_ee_mean,
            "ref_sr_mean":   ref_sr_mean,
            "ref_sr_mae":    ref_sr_mae,
            "ref_ee_mean":   ref_ee_mean,
        },
    }
    out_path = Path(args.artifact_dir) / f"refinement_results{suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    print(f"Refined predictions saved to {pred_npy_path}")


if __name__ == "__main__":
    main()