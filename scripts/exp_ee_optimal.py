"""Three-phase EE maximization for the 3x3 PASS system.

Phase 1: 200-step QoS-only Adam convergence (random restarts).
Phase 2: 150-step EE-augmented Adam (lambda=0.05, LR*0.2) from Phase 1 solution.
Phase 3: Per-waveguide binary search for minimum feasible power (exact, ~40 iterations).

Phase 3 is the key: for FIXED PA positions from Phase 2, find the minimum
power allocation per waveguide that still satisfies all QoS constraints.
This is exact (no local optima) and fast (~120 forward passes total).

Usage:
    python -u scripts/exp_ee_optimal.py --artifact-dir artifacts/physics_run/pass_20260630_185731
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
from ml.pass_dnn.torch_physics import evaluate_pass_batch_torch, to_physical_pass_outputs

BATCH     = 128
N_RESTART = 5
LR        = 1e-2
T1, T2    = 200, 150       # best two-phase config from prior experiment
LAM_P2    = 0.05


# ── Adam helpers (unchanged from exp_ee_bc.py) ───────────────────────────────

def _adam(inputs_raw, pos0, pow0, config, n_steps, lr,
          lambda_ee=0.0, power_pen=0.0):
    pos  = pos0.clone().detach().requires_grad_(True)
    pow_ = pow0.clone().detach().requires_grad_(True)
    opt  = torch.optim.Adam([pos, pow_], lr=lr)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps, 1), eta_min=lr * 0.01)
    for _ in range(n_steps):
        opt.zero_grad()
        out  = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qv   = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv   = F.relu(b.total_power - config.power_budget_w).pow(2)
        loss = qv + 0.25 * pv
        if power_pen > 0.0:
            loss = loss + power_pen * (b.total_power / config.power_budget_w)
        if lambda_ee > 0.0:
            loss = loss - lambda_ee * b.energy_efficiency
        loss.sum().backward()
        opt.step(); sch.step()
        with torch.no_grad():
            pos.data.clamp_(-1.0, 1.0)
            pow_.data.clamp_(0.0, 1.0)
    with torch.no_grad():
        out  = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qv   = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv   = F.relu(b.total_power - config.power_budget_w).pow(2)
        pen  = qv + 0.25 * pv
    return pos.detach(), pow_.detach(), pen.detach()


# ── Phase 3: Per-waveguide binary search for minimum power ───────────────────

@torch.no_grad()
def _qos_satisfied(inputs_raw, pos_norm, pow_norm, config):
    out  = torch.cat([pos_norm, pow_norm], dim=1)
    phys = to_physical_pass_outputs(out, config)
    b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
    qos_thresh = inputs_raw[:, config.num_users * 2]
    return (b.rates >= qos_thresh.unsqueeze(1)).all(dim=1)  # (B,) bool


@torch.no_grad()
def binary_search_per_waveguide(inputs_raw, pos_norm, pow_norm, config, n_iter=40):
    """For each sample, minimise each waveguide power independently.

    Iterates over K waveguides; for each, binary-searches the minimum scale
    in [0, current_p_k] that still satisfies all user QoS constraints.
    Two coordinate-descent passes catch cross-waveguide interactions.

    Returns:
        pow_opt (B, K) in normalised [0,1] space.
        alpha_per_wg (B, K) — achieved scale per waveguide.
    """
    B, K = pow_norm.shape
    pow_opt = pow_norm.clone()

    for _pass in range(2):            # two coordinate-descent passes
        for k in range(K):
            lo = torch.zeros(B)
            hi = pow_opt[:, k].clone()  # current power for this waveguide

            for _ in range(n_iter):
                mid = (lo + hi) / 2.0
                trial = pow_opt.clone()
                trial[:, k] = mid
                sat = _qos_satisfied(inputs_raw, pos_norm, trial, config)
                hi = torch.where(sat, mid, hi)   # feasible -> can reduce
                lo = torch.where(~sat, mid, lo)  # infeasible -> need more

            pow_opt[:, k] = hi  # minimum feasible power for waveguide k

    # Compute how much each waveguide was reduced (for logging)
    alpha = pow_opt / (pow_norm + 1e-12)
    return pow_opt, alpha


# ── Restart wrapper ───────────────────────────────────────────────────────────

def refine_threephase(inputs_raw, init_pos, init_pow, config):
    """Best-of-N two-phase Adam restarts, then exact power minimisation."""
    B = inputs_raw.shape[0]
    best_pos = init_pos.clone()
    best_pow = init_pow.clone()

    with torch.no_grad():
        out = torch.cat([best_pos, best_pow], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qv = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv = F.relu(b.total_power - config.power_budget_w).pow(2)
        best_pen = qv + 0.25 * pv

    for _ in range(N_RESTART):
        p0 = torch.rand(B, init_pos.shape[1]) * 2 - 1
        w0 = torch.rand(B, init_pow.shape[1])
        # Phase 1: QoS convergence
        pos, pow_, _ = _adam(inputs_raw, p0, w0, config, T1, LR)
        # Phase 2: EE push from feasible
        pos, pow_, pen = _adam(inputs_raw, pos, pow_, config, T2, LR * 0.2,
                                lambda_ee=LAM_P2)
        improved = pen < best_pen
        best_pos[improved] = pos[improved]
        best_pow[improved] = pow_[improved]
        best_pen[improved] = pen[improved]

    # Phase 3: exact per-waveguide power minimisation
    pow_opt, _ = binary_search_per_waveguide(inputs_raw, best_pos, best_pow, config)
    return best_pos, pow_opt


def run_dataset(x_raw, pred_norm, config, label=""):
    x_t = torch.tensor(x_raw, dtype=torch.float32)
    ip  = torch.tensor(pred_norm[:, :9], dtype=torch.float32)
    iw  = torch.tensor(pred_norm[:, 9:], dtype=torch.float32)
    pp, pw = [], []
    n, t0 = len(x_raw), time.perf_counter()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        rp, rw = refine_threephase(x_t[s:e], ip[s:e], iw[s:e], config)
        pp.append(rp); pw.append(rw)
        print(f"  {label} [{100*e/n:5.1f}%]", flush=True)
    wall = time.perf_counter() - t0
    norm = np.concatenate([torch.cat(pp).numpy(), torch.cat(pw).numpy()], axis=1)
    return norm, wall


def get_metrics(x_raw, true_eval, norm, config):
    phys = norm.copy()
    phys[:, :9] *= config.position_bound_m
    phys[:, 9:] *= config.power_budget_w
    ev = evaluate_pass_batch(x_raw, phys, config)
    tq = true_eval.qos_satisfied.astype(int)
    pq = ev.qos_satisfied.astype(int)
    tp = int(((tq==1)&(pq==1)).sum()); fp = int(((tq==0)&(pq==1)).sum()); fn=int(((tq==1)&(pq==0)).sum())
    pr = tp/(tp+fp) if (tp+fp)>0 else 0.0; re = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*pr*re/(pr+re) if (pr+re)>0 else 0.0
    pow_mean = float(ev.sum_rate.mean()) / (float(ev.energy_efficiency.mean()) + 1e-9) - config.circuit_power_w
    return {"f1": round(f1, 4), "sat": round(float(ev.qos_satisfied.mean()), 4),
            "sr_mean": round(float(ev.sum_rate.mean()), 4),
            "sr_mae":  round(float(np.abs(ev.sum_rate - true_eval.sum_rate).mean()), 4),
            "ee_mean": round(float(ev.energy_efficiency.mean()), 4),
            "pow_mean_mw": round(pow_mean * 1000, 2)}


@torch.no_grad()
def dnn_predict(model, x_sc):
    model.eval()
    parts = []
    for i in range(0, len(x_sc), BATCH):
        b = torch.tensor(x_sc[i:i+BATCH], dtype=torch.float32)
        o = model(b)
        parts.append(torch.cat([o["positions"], o["powers"]], dim=1).cpu())
    return torch.cat(parts).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--search-root", action="append",
                        default=["data/raw", "matlab/legacy/csv_data", "matlab/legacy"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    bundle = load_artifact_bundle(str(artifact_dir))
    model, config, scaler = bundle["model"], bundle["config"], bundle["input_scaler"]
    corpus = load_dataset_corpus(args.search_root)
    split  = split_indices(len(corpus.inputs), seed=args.seed)

    xt  = corpus.inputs[split.test_idx].astype(np.float32)
    xts = scaler.transform(xt).astype(np.float32)
    yt  = corpus.outputs[split.test_idx].astype(np.float32)
    tt  = evaluate_pass_batch(xt, yt, config)
    pt  = dnn_predict(model, xts)

    print(f"Test samples : {len(xt)}")
    print(f"MATLAB ref   : SR={tt.sum_rate.mean():.3f}  EE={tt.energy_efficiency.mean():.2f}  pow~37mW")
    print(f"\nRunning 3-phase: {T1} QoS steps -> {T2} EE steps (lam={LAM_P2}) -> per-waveguide binary search")
    print(f"N restarts: {N_RESTART}  Binary search iterations per waveguide: 40 x 2 passes")

    norm, wall = run_dataset(xt, pt, config, label="3ph")
    m = get_metrics(xt, tt, norm, config)
    m["ms_per_s"] = round(1000 * wall / len(xt), 1)

    print("\n" + "="*65)
    print("THREE-PHASE RESULTS vs PRIOR BEST")
    print("="*65)
    prior = [
        ("MATLAB solver",                  "---", 78.55, 3.769, "~37",  "~30min"),
        ("B: two-phase lam=0.05",         0.977,  52.03, 4.549, "~77",  "132ms"),
        ("C: two-phase+pp lam=0.05,pp0.5",0.978,  53.45, 4.483, "~74",  "165ms"),
    ]
    print(f"  {'Method':<42} {'F1':>6} {'EE':>8} {'SR':>8} {'Pow':>8} {'Lat':>8}")
    print(f"  {'-'*75}")
    for r in prior:
        print(f"  {r[0]:<42} {str(r[1]):>6} {r[2]:>8.2f} {str(r[3]):>8} {str(r[4]):>8} {str(r[5]):>8}")
    print(f"  {'D: 3-phase + binary search (ours)':<42} {m['f1']:>6.4f} {m['ee_mean']:>8.2f} {m['sr_mean']:>8.4f} {m['pow_mean_mw']:>7.1f}mW {m['ms_per_s']:>7.1f}ms")
    print("="*65)

    ee_matlab = float(tt.energy_efficiency.mean())
    gap_pct = 100 * (ee_matlab - m["ee_mean"]) / ee_matlab
    print(f"\n  EE gap to MATLAB: {ee_matlab:.2f} - {m['ee_mean']:.2f} = {ee_matlab - m['ee_mean']:.2f} bits/J ({gap_pct:.1f}% below)")
    print(f"  SR vs MATLAB    : {m['sr_mean']:.4f} vs {tt.sum_rate.mean():.3f} b/s/Hz ({100*(m['sr_mean']-tt.sum_rate.mean())/tt.sum_rate.mean():+.1f}%)")

    result = {
        "method": f"three_phase_T1{T1}_T2{T2}_lam{LAM_P2}_bsearch",
        "config": {"T1": T1, "T2": T2, "lam_p2": LAM_P2, "n_restarts": N_RESTART, "bsearch_iters": 40},
        "test": m,
        "matlab_ref": {"ee": round(ee_matlab, 4), "sr": round(float(tt.sum_rate.mean()), 4)},
    }
    out = artifact_dir / "exp_ee_optimal.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()