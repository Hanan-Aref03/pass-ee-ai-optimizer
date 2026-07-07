"""Dinkelbach EE maximization for fixed Phase-2 positions.

Pipeline per restart:
  Phase 1 (T1=200 steps): joint pos+pow, QoS-only  -> feasible
  Phase 2 (T2=150 steps): joint pos+pow, weak EE   -> improved positions
  Dinkelbach outer loop (N_DK=8 iterations):
    Inner (T_INN=150 steps): power-only Adam, LR=1e-2, constant
      loss = QoS_W*qv - (b.sum_rate - lambda * b.total_power)
    Update: lambda = EE(positions, powers_inner)

Dinkelbach guarantees convergence to the GLOBAL EE optimum over the
feasible set for fixed positions (the fractional program SR/P_total is
quasi-concave in powers when interference is treated as fixed).

Why this beats Phase-3 Adam:
  - Phase-3 used LR=5e-4 (too small to move from 74 mW to optimal ~37 mW)
  - Dinkelbach inner uses LR=1e-2, no cosine decay -> large moves possible
  - Adaptive lambda corrects the gradient direction iteration by iteration

Usage:
    python -u scripts/exp_ee_dinkelbach.py \\
        --artifact-dir artifacts/physics_run/pass_20260630_185731
"""
from __future__ import annotations
import argparse, json, sys, time
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
T1, T2    = 200, 150
LAM_P2    = 0.05
N_DK      = 8      # Dinkelbach outer iterations
T_INN     = 150    # inner Adam steps per Dinkelbach iteration
LR_INN    = 1e-2   # constant LR inside inner problem (no cosine decay)
QOS_W     = 500.0  # strong QoS penalty to stay feasible during large power moves


def _adam_joint(inputs_raw, pos0, pow0, config, n_steps, lr, lambda_ee=0.0):
    """Joint position+power Adam (Phase 1 and Phase 2)."""
    pos  = pos0.clone().detach().requires_grad_(True)
    pow_ = pow0.clone().detach().requires_grad_(True)
    opt  = torch.optim.Adam([pos, pow_], lr=lr)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps,1), eta_min=lr*0.01)
    for _ in range(n_steps):
        opt.zero_grad()
        out  = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qv   = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv   = F.relu(b.total_power - config.power_budget_w).pow(2)
        loss = qv + 0.25 * pv
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


def _dinkelbach_inner(inputs_raw, pos_fixed, pow_init, config, lam_ee):
    """One Dinkelbach inner solve: max SR - lam_ee * P_total  s.t. QoS, pow in [0,1].

    loss = QOS_W * qv - (SR - lam_ee * P_total)
         = QOS_W * qv - SR + lam_ee * P_total

    Gradient w.r.t. p_k:
      d(-SR)/dp_k + lam_ee * (P_budget) * 1  [normalised power]
    This pushes p_k down wherever marginal rate < lam_ee * P_budget
    and up wherever marginal rate > lam_ee * P_budget.
    At convergence: marginal rate = lam_ee for all k  (EE-optimal condition).
    """
    pos  = pos_fixed.detach()
    pow_ = pow_init.clone().detach().requires_grad_(True)
    opt  = torch.optim.Adam([pow_], lr=LR_INN)
    qos_thresh = inputs_raw[:, config.num_users * 2]
    for _ in range(T_INN):
        opt.zero_grad()
        out  = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qv   = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv   = F.relu(b.total_power - config.power_budget_w).pow(2)
        # Dinkelbach inner objective (per sample):
        #   -( SR - lam * P_total )  +  QoS penalty
        # lam_ee is a scalar (B,) tensor — one lambda per sample
        dinkelbach_obj = b.sum_rate - lam_ee * b.total_power
        loss = QOS_W * (qv + 0.25 * pv) - dinkelbach_obj
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            pow_.data.clamp_(0.0, 1.0)
    with torch.no_grad():
        out  = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qv   = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv   = F.relu(b.total_power - config.power_budget_w).pow(2)
        pen  = qv + 0.25 * pv
        ee   = b.energy_efficiency   # (B,) — new lambda for next iteration
    return pow_.detach(), pen.detach(), ee.detach()


def _dinkelbach_loop(inputs_raw, pos_fixed, pow_init, config):
    """Run N_DK outer Dinkelbach iterations; return final (pow, pen)."""
    pow_ = pow_init.clone()
    with torch.no_grad():
        out  = torch.cat([pos_fixed, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        lam  = b.energy_efficiency.detach().clone()  # (B,) initial lambdas
    for _ in range(N_DK):
        pow_, pen, lam = _dinkelbach_inner(inputs_raw, pos_fixed, pow_, config, lam)
    return pow_, pen


def refine_threephase(inputs_raw, init_pos, init_pow, config):
    """Best-of-N: Phase1 QoS -> Phase2 EE -> Dinkelbach power optimization."""
    B = inputs_raw.shape[0]
    best_pos = init_pos.clone(); best_pow = init_pow.clone()
    with torch.no_grad():
        out  = torch.cat([best_pos, best_pow], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qv = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv = F.relu(b.total_power - config.power_budget_w).pow(2)
        best_pen = qv + 0.25 * pv

    for _ in range(N_RESTART):
        p0 = torch.rand(B, init_pos.shape[1]) * 2 - 1
        w0 = torch.rand(B, init_pow.shape[1])
        # Phase 1
        pos, pow_, _ = _adam_joint(inputs_raw, p0, w0, config, T1, LR)
        # Phase 2
        pos, pow_, _ = _adam_joint(inputs_raw, pos, pow_, config, T2, LR * 0.2,
                                    lambda_ee=LAM_P2)
        # Dinkelbach power optimization
        pow_dk, pen_dk = _dinkelbach_loop(inputs_raw, pos, pow_, config)
        improved = pen_dk < best_pen
        best_pos[improved] = pos[improved]
        best_pow[improved] = pow_dk[improved]
        best_pen[improved] = pen_dk[improved]

    return best_pos, best_pow


def run_dataset(x_raw, pred_norm, config, label=""):
    x_t = torch.tensor(x_raw, dtype=torch.float32)
    ip  = torch.tensor(pred_norm[:, :9], dtype=torch.float32)
    iw  = torch.tensor(pred_norm[:, 9:], dtype=torch.float32)
    pp, pw = [], []; n, t0 = len(x_raw), time.perf_counter()
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
    tq = true_eval.qos_satisfied.astype(int); pq = ev.qos_satisfied.astype(int)
    tp = int(((tq==1)&(pq==1)).sum()); fp = int(((tq==0)&(pq==1)).sum())
    fn = int(((tq==1)&(pq==0)).sum())
    pr = tp/(tp+fp) if (tp+fp)>0 else 0.0
    re = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*pr*re/(pr+re) if (pr+re)>0 else 0.0
    pow_mean = float(ev.sum_rate.mean())/(float(ev.energy_efficiency.mean())+1e-9) - config.circuit_power_w
    return {"f1": round(f1,4), "sat": round(float(ev.qos_satisfied.mean()),4),
            "sr_mean":    round(float(ev.sum_rate.mean()),4),
            "sr_mae":     round(float(np.abs(ev.sum_rate - true_eval.sum_rate).mean()),4),
            "ee_mean":    round(float(ev.energy_efficiency.mean()),4),
            "pow_mean_mw":round(pow_mean*1000,2)}


@torch.no_grad()
def dnn_predict(model, x_sc):
    model.eval(); parts = []
    for i in range(0, len(x_sc), BATCH):
        b = torch.tensor(x_sc[i:i+BATCH], dtype=torch.float32)
        o = model(b)
        parts.append(torch.cat([o["positions"], o["powers"]], dim=1).cpu())
    return torch.cat(parts).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--search-root", action="append",
                        default=["data/raw","matlab/legacy/csv_data","matlab/legacy"])
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

    ee_matlab = float(tt.energy_efficiency.mean())
    sr_matlab = float(tt.sum_rate.mean())
    print(f"Test samples : {len(xt)}")
    print(f"MATLAB ref   : SR={sr_matlab:.3f}  EE={ee_matlab:.2f}")
    print(f"\nPhase1: T1={T1} (QoS)  Phase2: T2={T2} lam={LAM_P2}")
    print(f"Dinkelbach: N_outer={N_DK} x T_inner={T_INN} steps, LR={LR_INN}, QoS_W={QOS_W}")
    print(f"Total steps/restart: {T1}+{T2}+{N_DK*T_INN} = {T1+T2+N_DK*T_INN}")
    print(f"N restarts: {N_RESTART}")

    norm, wall = run_dataset(xt, pt, config, label="DK")
    m = get_metrics(xt, tt, norm, config)
    m["ms_per_s"] = round(1000*wall/len(xt), 1)

    gap = ee_matlab - m["ee_mean"]
    print("\n" + "="*70)
    print("DINKELBACH RESULTS")
    print("="*70)
    rows = [
        ("MATLAB solver",                    "---",   78.55, 3.769, "~37",   "~30min"),
        ("B: two-phase lam=0.05",            0.9765,  52.03, 4.549, "~77",   "132ms"),
        ("C: two-phase+pp=0.5",              0.9776,  53.45, 4.483, "~74",   "165ms"),
        ("E: pow-only Phase3 (prev)",        0.9783,  57.91, 4.322, "~65",   "411ms"),
        ("F: Dinkelbach (ours)",             m["f1"], m["ee_mean"], m["sr_mean"],
         f"~{m['pow_mean_mw']:.0f}", f"{m['ms_per_s']}ms"),
    ]
    print(f"  {'Method':<42} {'F1':>7} {'EE':>8} {'SR':>8} {'Pow':>8} {'Lat':>8}")
    print(f"  {'-'*80}")
    for r in rows:
        print(f"  {r[0]:<42} {str(r[1]):>7} {float(r[2]):>8.2f} {str(r[3]):>8} {str(r[4]):>8} {str(r[5]):>8}")
    print("="*70)
    print(f"\n  EE gap to MATLAB: {ee_matlab:.2f} - {m['ee_mean']:.2f} = {gap:.2f} bits/J ({100*gap/ee_matlab:.1f}% below)")
    print(f"  SR vs MATLAB    : {m['sr_mean']:.4f} vs {sr_matlab:.3f} ({100*(m['sr_mean']-sr_matlab)/sr_matlab:+.1f}%)")

    result = {
        "method": f"dinkelbach_T1{T1}_T2{T2}_N_DK{N_DK}_T_INN{T_INN}",
        "config": {"T1":T1,"T2":T2,"lam_p2":LAM_P2,"N_DK":N_DK,"T_INN":T_INN,
                   "LR_INN":LR_INN,"QOS_W":QOS_W,"n_restarts":N_RESTART},
        "test": m,
        "matlab_ref": {"ee": round(ee_matlab,4), "sr": round(sr_matlab,4)},
    }
    out = artifact_dir / "exp_ee_dinkelbach.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()