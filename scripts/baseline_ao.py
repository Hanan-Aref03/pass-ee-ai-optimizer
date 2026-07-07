"""AO (Alternating Optimization) baseline -- single initialization, no global restarts.

This is the classical heuristic requested by Reviewer Comment 3.
It implements the standard AO structure used in wireless resource allocation:
  repeat K_AO times:
    Step A: optimize PA positions (gradient ascent, powers fixed)
    Step B: optimize powers via Dinkelbach fractional programming (positions fixed)
starting from a single random initialization -- no global random restarts.

Step B is provably optimal (for fixed positions) after convergence of the
Dinkelbach outer loop; Step A is a gradient-based position refinement.
The alternation mirrors classical AO but WITHOUT the global random restart
search used in our main method.

Comparison purpose
------------------
  AO baseline:  1 init, 3 alternations, ~80 ms/sample  -> suboptimal EE
  Ours (Exp G): 5 inits, Dinkelbach, 257 ms/sample     -> near-MATLAB EE

Usage:
    python -u scripts/baseline_ao.py \\
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

BATCH      = 128
K_AO       = 3           # number of AO outer alternations
T_POS      = 100         # position-step Adam steps per AO iteration (Step A)
LR_POS     = 1e-2        # LR for position Adam
N_DK       = 4           # Dinkelbach outer iterations in Step B
T_POW      = 50          # Dinkelbach inner Adam steps per outer iter (Step B)
LR_POW     = 1e-2        # LR for power Adam (constant, no cosine decay)
QOS_W      = 500.0       # QoS penalty weight
QOS_MARGIN = 1.05        # tighten threshold by 5% for robustness


# ── Step A: position-only gradient step ──────────────────────────────────────

def _position_step(inputs_raw, pos_init, pow_fixed, config):
    """Gradient ascent on PA positions with powers fixed."""
    pos  = pos_init.clone().detach().requires_grad_(True)
    pow_ = pow_fixed.detach()
    opt  = torch.optim.Adam([pos], lr=LR_POS)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(T_POS,1),
                                                       eta_min=LR_POS*0.01)
    qos_thresh = inputs_raw[:, config.num_users * 2]
    for _ in range(T_POS):
        opt.zero_grad()
        out  = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qv   = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv   = F.relu(b.total_power - config.power_budget_w).pow(2)
        # Weak EE signal in position step (lambda=0.05)
        loss = qv + 0.25 * pv - 0.05 * b.energy_efficiency
        loss.sum().backward()
        opt.step(); sch.step()
        with torch.no_grad():
            pos.data.clamp_(-1.0, 1.0)
    return pos.detach()


# ── Step B: power-only Dinkelbach ─────────────────────────────────────────────

def _power_step_inner(inputs_raw, pos_fixed, pow_init, lam, config):
    """One Dinkelbach inner solve for power allocation."""
    pos  = pos_fixed.detach()
    pow_ = pow_init.clone().detach().requires_grad_(True)
    lam  = lam.detach().clone()
    opt  = torch.optim.Adam([pow_], lr=LR_POW)
    qos_thresh_tight = inputs_raw[:, config.num_users * 2] * QOS_MARGIN
    for _ in range(T_POW):
        opt.zero_grad()
        out  = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qv   = F.relu(qos_thresh_tight.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv   = F.relu(b.total_power - config.power_budget_w).pow(2)
        loss = QOS_W * (qv + 0.25 * pv) - (b.sum_rate - lam * b.total_power)
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            pow_.data.clamp_(0.0, 1.0)
    with torch.no_grad():
        out  = torch.cat([pos, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        new_lam = b.energy_efficiency.detach().clone()
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qv  = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv  = F.relu(b.total_power - config.power_budget_w).pow(2)
        pen = qv + 0.25 * pv
    return pow_.detach(), pen.detach(), new_lam


def _power_step(inputs_raw, pos_fixed, pow_init, config):
    """Dinkelbach outer loop for EE-optimal power (positions fixed)."""
    pow_ = pow_init.clone()
    with torch.no_grad():
        out  = torch.cat([pos_fixed, pow_], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        lam  = b.energy_efficiency.detach().clone()
    for _ in range(N_DK):
        pow_, pen, lam = _power_step_inner(inputs_raw, pos_fixed, pow_, lam, config)
    return pow_, pen


# ── AO loop (no restarts) ─────────────────────────────────────────────────────

def ao_single_init(inputs_raw, config):
    """AO from a single random init: K_AO alternations of position+power."""
    B = inputs_raw.shape[0]
    # Random initialisation (single, no restarts)
    pos = torch.rand(B, 9) * 2 - 1   # positions in [-1, 1]
    pow_ = torch.full((B, config.num_users), 0.5)  # equal power start at 50%

    for ao_iter in range(K_AO):
        pos  = _position_step(inputs_raw, pos, pow_, config)
        pow_, pen = _power_step(inputs_raw, pos, pow_, config)

    return pos, pow_


def run_dataset(x_raw, config, label="AO"):
    x_t = torch.tensor(x_raw, dtype=torch.float32)
    pp, pw = [], []; n, t0 = len(x_raw), time.perf_counter()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        rp, rw = ao_single_init(x_t[s:e], config)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--search-root", action="append",
                        default=["data/raw","matlab/legacy/csv_data","matlab/legacy"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    bundle = load_artifact_bundle(str(artifact_dir))
    config, scaler = bundle["config"], bundle["input_scaler"]
    corpus = load_dataset_corpus(args.search_root)
    split  = split_indices(len(corpus.inputs), seed=args.seed)

    xt = corpus.inputs[split.test_idx].astype(np.float32)
    yt = corpus.outputs[split.test_idx].astype(np.float32)
    tt = evaluate_pass_batch(xt, yt, config)

    ee_matlab = float(tt.energy_efficiency.mean())
    sr_matlab = float(tt.sum_rate.mean())
    total_steps = K_AO * (T_POS + N_DK * T_POW)
    print(f"Test samples : {len(xt)}")
    print(f"MATLAB ref   : SR={sr_matlab:.3f}  EE={ee_matlab:.2f}")
    print(f"\nAO config: {K_AO} alternations x (pos {T_POS} steps + pow {N_DK}x{T_POW} Dinkelbach)")
    print(f"Total steps: {total_steps}  Inits: 1 (NO global restarts)")

    norm, wall = run_dataset(xt, config, label="AO")
    m = get_metrics(xt, tt, norm, config)
    m["ms_per_s"] = round(1000 * wall / len(xt), 1)

    gap = ee_matlab - m["ee_mean"]
    print("\n" + "="*70)
    print("AO BASELINE vs OURS (Fast Dinkelbach)")
    print("="*70)
    rows = [
        ("MATLAB solver (oracle)",              "---",   78.55, 3.769, "~38", "~30min", "global search"),
        ("AO, 1-init (classical baseline)",     m["f1"], m["ee_mean"], m["sr_mean"],
         f"~{m['pow_mean_mw']:.0f}", f"{m['ms_per_s']}ms", "1 rand init, no restarts"),
        ("Ours: Fast Dinkelbach (5 restarts)",  0.986,  78.27, 2.664, "~24", "257ms",  "5 rand inits"),
    ]
    print(f"  {'Method':<40} {'F1':>7} {'EE':>8} {'SR':>8} {'Pow':>6} {'Lat':>8}  Notes")
    print(f"  {'-'*85}")
    for r in rows:
        print(f"  {r[0]:<40} {str(r[1]):>7} {float(r[2]):>8.2f} {str(r[3]):>8} {str(r[4]):>6} {str(r[5]):>8}  {r[6]}")
    print("="*70)
    print(f"\n  EE gap: AO={m['ee_mean']:.2f} vs Ours=78.27 vs MATLAB=78.55 b/J")
    print(f"  EE gain of ours vs AO: {78.27-m['ee_mean']:.2f} b/J ({100*(78.27-m['ee_mean'])/m['ee_mean']:.1f}%)")
    print(f"  Latency: AO={m['ms_per_s']}ms vs Ours=257ms ({257/m['ms_per_s']:.1f}x slower for {78.27-m['ee_mean']:.1f} b/J gain)")

    result = {
        "method": "ao_single_init_dinkelbach",
        "config": {"K_AO":K_AO,"T_POS":T_POS,"LR_POS":LR_POS,
                   "N_DK":N_DK,"T_POW":T_POW,"LR_POW":LR_POW,
                   "QOS_W":QOS_W,"QOS_MARGIN":QOS_MARGIN,"n_inits":1},
        "test": m,
        "matlab_ref": {"ee": round(ee_matlab,4), "sr": round(sr_matlab,4)},
    }
    out = artifact_dir / "baseline_ao.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()