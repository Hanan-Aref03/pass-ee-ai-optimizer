"""Experiments 1, 2, and 4 for the PASS DNN paper.

Exp 1 -EE-augmented refinement: J = QoS_pen + 0.25*P_pen - lambda_ee * EE
         Grid-search lambda_ee on val set; eval best on test.
Exp 2 -Restart-budget sweep: K x T x {dnn, random} seeding.
Exp 4 -EE-optimal random baseline: J = -EE + 10*QoS_pen + 0.25*P_pen,
         select restart with highest EE that satisfies QoS.

Usage:
    python -u scripts/run_experiments.py --artifact-dir <path>
    python -u scripts/run_experiments.py --artifact-dir <path> --exps 1,4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ml.pass_dnn.train import load_artifact_bundle, load_dataset_corpus
from ml.pass_dnn.data import split_indices
from ml.pass_dnn.physics import evaluate_pass_batch
from ml.pass_dnn.torch_physics import (
    evaluate_pass_batch_torch,
    to_physical_pass_outputs,
)

BATCH_SIZE = 128


# ── Core Adam loop (parameterized for all experiments) ───────────────────────

def _run_adam(
    inputs_raw: torch.Tensor,
    init_pos: torch.Tensor,
    init_pow: torch.Tensor,
    config,
    n_steps: int,
    lr: float,
    lambda_ee: float = 0.0,
    qos_weight: float = 1.0,
    power_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single Adam restart.

    Returns (positions, powers, per-sample QoS+power penalty, per-sample EE).
    """
    pos = init_pos.clone().detach().requires_grad_(True)
    pow_ = init_pow.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([pos, pow_], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)

    for _ in range(n_steps):
        opt.zero_grad()
        out = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        batch = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qos_viol = F.relu(qos_thresh.unsqueeze(1) - batch.rates).pow(2).mean(dim=1)
        pow_viol = F.relu(batch.total_power - config.power_budget_w).pow(2)
        loss = qos_weight * qos_viol + power_weight * pow_viol
        if lambda_ee > 0.0:
            loss = loss - lambda_ee * batch.energy_efficiency
        loss.sum().backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            pos.data.clamp_(-1.0, 1.0)
            pow_.data.clamp_(0.0, 1.0)

    with torch.no_grad():
        out = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        batch = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qos_viol = F.relu(qos_thresh.unsqueeze(1) - batch.rates).pow(2).mean(dim=1)
        pow_viol = F.relu(batch.total_power - config.power_budget_w).pow(2)
        penalty = qos_weight * qos_viol + power_weight * pow_viol
        ee = batch.energy_efficiency
        sat = batch.qos_satisfied

    return pos.detach(), pow_.detach(), penalty.detach(), ee.detach(), sat.detach()


def refine_batch(
    inputs_raw: torch.Tensor,
    init_pos: torch.Tensor,
    init_pow: torch.Tensor,
    config,
    n_steps: int,
    lr: float,
    n_restarts: int,
    dnn_warmstart: bool = False,
    lambda_ee: float = 0.0,
    qos_weight: float = 1.0,
    power_weight: float = 0.25,
    ee_select: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-of-N Adam restarts with configurable selection criterion.

    dnn_warmstart=True  ->restart 0 uses DNN prediction, rest random.
    ee_select=True      ->select restart with highest EE among QoS-satisfied
                          (Exp 4); fall back to lowest penalty if none satisfied.
    ee_select=False     ->select restart with lowest QoS+power penalty (default).
    """
    B = inputs_raw.shape[0]
    best_pos  = init_pos.clone()
    best_pow  = init_pow.clone()

    with torch.no_grad():
        out = torch.cat([best_pos, best_pow], dim=1)
        phys = to_physical_pass_outputs(out, config)
        batch0 = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qv = F.relu(qos_thresh.unsqueeze(1) - batch0.rates).pow(2).mean(dim=1)
        pv = F.relu(batch0.total_power - config.power_budget_w).pow(2)
        best_pen = qos_weight * qv + power_weight * pv
        best_ee  = batch0.energy_efficiency.clone()
        best_sat = batch0.qos_satisfied.clone()

    for k in range(n_restarts):
        if k == 0 and dnn_warmstart:
            p0, w0 = init_pos.clone(), init_pow.clone()
        else:
            p0 = torch.rand(B, init_pos.shape[1]) * 2 - 1
            w0 = torch.rand(B, init_pow.shape[1])

        pos_r, pow_r, pen_r, ee_r, sat_r = _run_adam(
            inputs_raw, p0, w0, config, n_steps, lr,
            lambda_ee=lambda_ee, qos_weight=qos_weight, power_weight=power_weight,
        )

        if ee_select:
            both_sat     = best_sat & sat_r
            only_new_sat = (~best_sat) & sat_r
            neither_sat  = (~best_sat) & (~sat_r)
            take = (both_sat & (ee_r > best_ee)) | only_new_sat | (neither_sat & (pen_r < best_pen))
        else:
            take = pen_r < best_pen

        best_pos[take] = pos_r[take]
        best_pow[take] = pow_r[take]
        best_pen[take] = pen_r[take]
        best_ee[take]  = ee_r[take]
        best_sat[take] = sat_r[take]

    return best_pos, best_pow


# ── Metric helpers ────────────────────────────────────────────────────────────

def _f1(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def evaluate_metrics(x_raw, true_eval, refined_norm, config):
    """Return dict: F1, sat_rate, sr_mean, sr_mae, ee_mean from normalised outputs."""
    phys = refined_norm.copy()
    phys[:, :9] *= config.position_bound_m
    phys[:, 9:] *= config.power_budget_w
    ev = evaluate_pass_batch(x_raw, phys, config)
    true_qos = true_eval.qos_satisfied.astype(int)
    pred_qos = ev.qos_satisfied.astype(int)
    return {
        "f1":      round(_f1(true_qos, pred_qos), 4),
        "sat":     round(float(ev.qos_satisfied.mean()), 4),
        "sr_mean": round(float(ev.sum_rate.mean()), 4),
        "sr_mae":  round(float(np.abs(ev.sum_rate - true_eval.sum_rate).mean()), 4),
        "ee_mean": round(float(ev.energy_efficiency.mean()), 4),
    }


@torch.no_grad()
def dnn_predict(model, x_scaled):
    model.eval()
    x_t = torch.tensor(x_scaled, dtype=torch.float32)
    parts = []
    for i in range(0, len(x_t), BATCH_SIZE):
        out = model(x_t[i:i + BATCH_SIZE])
        parts.append(torch.cat([out["positions"], out["powers"]], dim=1).cpu())
    return torch.cat(parts).numpy()


def run_refinement_dataset(
    x_raw, pred_norm, config, n_steps, lr, n_restarts,
    dnn_warmstart, lambda_ee=0.0, qos_weight=1.0, power_weight=0.25,
    ee_select=False, label="",
):
    """Refine all samples in batches; return (refined_norm, wall_s)."""
    x_t = torch.tensor(x_raw, dtype=torch.float32)
    ip  = torch.tensor(pred_norm[:, :9], dtype=torch.float32)
    iw  = torch.tensor(pred_norm[:, 9:], dtype=torch.float32)
    pos_parts, pow_parts = [], []
    n = len(x_raw)
    t0 = time.perf_counter()
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        rp, rw = refine_batch(
            x_t[s:e], ip[s:e], iw[s:e], config,
            n_steps=n_steps, lr=lr, n_restarts=n_restarts,
            dnn_warmstart=dnn_warmstart, lambda_ee=lambda_ee,
            qos_weight=qos_weight, power_weight=power_weight,
            ee_select=ee_select,
        )
        pos_parts.append(rp)
        pow_parts.append(rw)
        if label:
            print(f"    {label}  [{100*e/n:5.1f}%]", flush=True)
    wall = time.perf_counter() - t0
    refined_norm = np.concatenate(
        [torch.cat(pos_parts).numpy(), torch.cat(pow_parts).numpy()], axis=1
    )
    return refined_norm, wall


# ── Experiment 1 -EE-augmented refinement ───────────────────────────────────

def run_exp1(model, corpus, split, config, scaler, artifact_dir, lr=1e-2):
    print("\n" + "="*60)
    print("EXPERIMENT 1 -EE-augmented refinement (lambda_ee grid search (lam_ee))")
    print("="*60)

    lambdas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    baseline_f1 = 0.929   # random-only F1 from prior run

    # --- Validation set ---
    val_idx = split.val_idx
    xv_raw   = corpus.inputs[val_idx].astype(np.float32)
    xv_sc    = scaler.transform(xv_raw).astype(np.float32)
    yv_raw   = corpus.outputs[val_idx].astype(np.float32)
    true_eval_v = evaluate_pass_batch(xv_raw, yv_raw, config)
    pred_norm_v  = dnn_predict(model, xv_sc)

    print(f"\nVal set: {len(xv_raw)} samples")
    print(f"Grid: lambda_ee in {lambdas}")
    print(f"Constraint: F1 >= {baseline_f1 - 0.03:.3f} (baseline {baseline_f1:.3f} - 3 pp)\n")

    val_results = {}
    for lam in lambdas:
        tag = f"lam={lam}"
        print(f"  Val run {tag} (3 restarts x 300 steps)...", flush=True)
        ref_norm, _ = run_refinement_dataset(
            xv_raw, pred_norm_v, config,
            n_steps=300, lr=lr, n_restarts=3,
            dnn_warmstart=False, lambda_ee=lam,
        )
        m = evaluate_metrics(xv_raw, true_eval_v, ref_norm, config)
        val_results[lam] = m
        print(f"    F1={m['f1']:.4f}  EE={m['ee_mean']:.2f}  SR={m['sr_mean']:.3f}")

    # Select best lambda: maximize EE subject to F1 >= baseline - 0.03
    f1_floor = baseline_f1 - 0.03
    eligible = {lam: r for lam, r in val_results.items() if r["f1"] >= f1_floor}
    if eligible:
        best_lam = max(eligible, key=lambda l: eligible[l]["ee_mean"])
    else:
        print("  WARNING: no lambda met F1 floor; picking best EE regardless")
        best_lam = max(val_results, key=lambda l: val_results[l]["ee_mean"])

    print(f"\n  Best lambda_ee = {best_lam}  (val EE={val_results[best_lam]['ee_mean']:.2f}, F1={val_results[best_lam]['f1']:.4f})")

    # --- Test set ---
    test_idx = split.test_idx
    xt_raw  = corpus.inputs[test_idx].astype(np.float32)
    xt_sc   = scaler.transform(xt_raw).astype(np.float32)
    yt_raw  = corpus.outputs[test_idx].astype(np.float32)
    true_eval_t = evaluate_pass_batch(xt_raw, yt_raw, config)
    pred_norm_t  = dnn_predict(model, xt_sc)

    test_rows = {}
    for lam in [0.0, best_lam]:
        tag = f"lam={lam}"
        print(f"\n  Test run {tag} (5 restarts x 300 steps)...", flush=True)
        ref_norm, wall = run_refinement_dataset(
            xt_raw, pred_norm_t, config,
            n_steps=300, lr=lr, n_restarts=5,
            dnn_warmstart=False, lambda_ee=lam,
            label=tag,
        )
        m = evaluate_metrics(xt_raw, true_eval_t, ref_norm, config)
        m["latency_ms"] = round(1000 * wall / len(xt_raw), 1)
        test_rows[lam] = m

    print("\n  EXP 1 TEST RESULTS:")
    print(f"  {'Method':<30} {'F1':>7} {'Sat':>7} {'SR':>7} {'SR-MAE':>8} {'EE':>8} {'ms/s':>8}")
    print(f"  {'-'*60}")
    baseline_m = test_rows[0.0]
    print(f"  {'Baseline (lambda_ee=0)':<30} {baseline_m['f1']:>7.4f} {baseline_m['sat']:>7.3f} {baseline_m['sr_mean']:>7.3f} {baseline_m['sr_mae']:>8.3f} {baseline_m['ee_mean']:>8.2f} {baseline_m['latency_ms']:>8.1f}")
    if best_lam != 0.0:
        ee_m = test_rows[best_lam]
        print(f"  {f'EE-augmented (lambda_ee={best_lam})':<30} {ee_m['f1']:>7.4f} {ee_m['sat']:>7.3f} {ee_m['sr_mean']:>7.3f} {ee_m['sr_mae']:>8.3f} {ee_m['ee_mean']:>8.2f} {ee_m['latency_ms']:>8.1f}")

    result = {
        "best_lambda_ee": best_lam,
        "val_results": {str(k): v for k, v in val_results.items()},
        "test_results": {str(k): v for k, v in test_rows.items()},
    }
    (artifact_dir / "exp1_ee_refinement.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Saved exp1_ee_refinement.json")
    return result


# ── Experiment 2 -Restart budget sweep ──────────────────────────────────────

def run_exp2(model, corpus, split, config, scaler, artifact_dir, lr=1e-2):
    print("\n" + "="*60)
    print("EXPERIMENT 2 -Restart-budget sweep K x T x {dnn, random}")
    print("="*60)

    K_vals = [1, 2, 3, 5]
    T_vals = [100, 200, 300]

    test_idx = split.test_idx
    xt_raw  = corpus.inputs[test_idx].astype(np.float32)
    xt_sc   = scaler.transform(xt_raw).astype(np.float32)
    yt_raw  = corpus.outputs[test_idx].astype(np.float32)
    true_eval = evaluate_pass_batch(xt_raw, yt_raw, config)
    pred_norm  = dnn_predict(model, xt_sc)
    n = len(xt_raw)

    results = {"dnn": {}, "random": {}}
    for seeding in ["dnn", "random"]:
        warmstart = (seeding == "dnn")
        print(f"\n  Seeding: {seeding.upper()}")
        for K in K_vals:
            for T in T_vals:
                key = f"K{K}_T{T}"
                print(f"    K={K} T={T}...", end=" ", flush=True)
                ref_norm, wall = run_refinement_dataset(
                    xt_raw, pred_norm, config,
                    n_steps=T, lr=lr, n_restarts=K,
                    dnn_warmstart=warmstart,
                )
                m = evaluate_metrics(xt_raw, true_eval, ref_norm, config)
                m["latency_ms"] = round(1000 * wall / n, 1)
                results[seeding][key] = m
                print(f"F1={m['f1']:.4f}  EE={m['ee_mean']:.2f}  {m['latency_ms']:.1f}ms/s", flush=True)

    # Print 2D tables
    for seeding in ["dnn", "random"]:
        print(f"\n  F1 table -{seeding.upper()} seeding:")
        header = f"  {'K\\T':>5}" + "".join(f"  T={T:>4}" for T in T_vals)
        print(header)
        for K in K_vals:
            row = f"  K={K:>3}"
            for T in T_vals:
                row += f"  {results[seeding][f'K{K}_T{T}']['f1']:>7.4f}"
            print(row)

        print(f"\n  Latency (ms/sample) -{seeding.upper()} seeding:")
        print(header)
        for K in K_vals:
            row = f"  K={K:>3}"
            for T in T_vals:
                row += f"  {results[seeding][f'K{K}_T{T}']['latency_ms']:>7.1f}"
            print(row)

    (artifact_dir / "exp2_restart_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Saved exp2_restart_sweep.json")
    return results


# ── Experiment 4 -EE-optimal random baseline ────────────────────────────────

def run_exp4(model, corpus, split, config, scaler, artifact_dir, lr=1e-2):
    print("\n" + "="*60)
    print("EXPERIMENT 4 -EE-optimal random-search baseline")
    print("="*60)

    test_idx = split.test_idx
    xt_raw  = corpus.inputs[test_idx].astype(np.float32)
    xt_sc   = scaler.transform(xt_raw).astype(np.float32)
    yt_raw  = corpus.outputs[test_idx].astype(np.float32)
    true_eval = evaluate_pass_batch(xt_raw, yt_raw, config)
    pred_norm  = dnn_predict(model, xt_sc)

    print("\n  Running 5 restarts x 300 steps, J = -EE + 10*QoS_pen + 0.25*P_pen")
    print("  Selection: highest EE among QoS-satisfied restarts\n")

    ref_norm, wall = run_refinement_dataset(
        xt_raw, pred_norm, config,
        n_steps=300, lr=lr, n_restarts=5,
        dnn_warmstart=False,
        lambda_ee=1.0, qos_weight=10.0, power_weight=0.25,
        ee_select=True,
        label="Exp4",
    )
    m = evaluate_metrics(xt_raw, true_eval, ref_norm, config)
    m["latency_ms"] = round(1000 * wall / len(xt_raw), 1)

    print(f"\n  EXP 4 RESULTS:")
    print(f"  F1={m['f1']:.4f}  Sat={m['sat']:.3f}  SR={m['sr_mean']:.3f}  EE={m['ee_mean']:.2f}  {m['latency_ms']:.1f}ms/s")

    (artifact_dir / "exp4_ee_random_baseline.json").write_text(json.dumps(m, indent=2))
    print(f"\n  Saved exp4_ee_random_baseline.json")
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--search-root", action="append",
        default=["data/raw", "matlab/legacy/csv_data", "matlab/legacy"],
    )
    parser.add_argument("--exps", default="1,2,4",
                        help="Comma-separated list of experiments to run (default: 1,2,4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exps = {int(x) for x in args.exps.split(",")}
    artifact_dir = Path(args.artifact_dir)

    bundle = load_artifact_bundle(str(artifact_dir))
    model  = bundle["model"]
    config = bundle["config"]
    scaler = bundle["input_scaler"]

    corpus = load_dataset_corpus(args.search_root)
    split  = split_indices(len(corpus.inputs), seed=args.seed)

    print(f"Corpus: {len(corpus.inputs)} samples")
    print(f"Test  : {len(split.test_idx)}  Val: {len(split.val_idx)}")
    print(f"Running experiments: {sorted(exps)}")

    results = {}
    if 1 in exps:
        results[1] = run_exp1(model, corpus, split, config, scaler, artifact_dir)
    if 2 in exps:
        results[2] = run_exp2(model, corpus, split, config, scaler, artifact_dir)
    if 4 in exps:
        results[4] = run_exp4(model, corpus, split, config, scaler, artifact_dir)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()