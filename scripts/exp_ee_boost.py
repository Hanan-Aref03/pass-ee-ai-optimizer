"""Maximize EE on the existing 3x3 PASS system without retraining.

Three approaches (all use existing model checkpoint, no new data needed):
  A - Power-penalty: J = QoS + 0.25*P_budget_viol + alpha*(total_power/P_budget)
  B - Two-phase:     Phase1 (T1 steps, lambda=0) -> Phase2 (T2 steps, lambda_ee high)
  C - Combined:      Two-phase with power penalty in both phases

Grid-searches alpha / lambda_ee on val set; full eval on test set.

Usage:
    python -u scripts/exp_ee_boost.py --artifact-dir artifacts/physics_run/pass_20260630_185731
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

BATCH = 128
N_RESTARTS = 5
LR = 1e-2


# ── Core Adam loop ────────────────────────────────────────────────────────────

def _adam(inputs_raw, pos0, pow0, config, n_steps, lr,
          lambda_ee=0.0, power_pen=0.0, qos_w=1.0, pbud_w=0.25):
    """Single Adam restart.

    power_pen: weight on (total_power / P_budget) — pushes toward lower power use.
    lambda_ee: weight on -EE — rewards energy efficiency.
    Returns (pos, pow, per-sample penalty, per-sample EE, per-sample qos_satisfied).
    """
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
        loss = qos_w * qv + pbud_w * pv
        if power_pen > 0.0:
            loss = loss + power_pen * (b.total_power / config.power_budget_w)
        if lambda_ee > 0.0:
            loss = loss - lambda_ee * b.energy_efficiency
        loss.sum().backward()
        opt.step()
        sch.step()
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
        pen  = qos_w * qv + pbud_w * pv

    return pos.detach(), pow_.detach(), pen.detach(), b.energy_efficiency.detach(), b.qos_satisfied.detach()


def _adam_twophase(inputs_raw, pos0, pow0, config,
                   t1, t2, lr=LR,
                   lambda_ee_p2=0.01, power_pen_p2=0.0):
    """Two-phase Adam: Phase 1 = QoS convergence, Phase 2 = EE push from feasible."""
    # Phase 1: converge to QoS-feasible
    pos, pow_, _, _, _ = _adam(inputs_raw, pos0, pow0, config,
                                n_steps=t1, lr=lr,
                                lambda_ee=0.0, power_pen=0.0)
    # Phase 2: EE/power improvement, starting from feasible point, lower LR
    lr2 = lr * 0.2
    pos, pow_, pen, ee, sat = _adam(inputs_raw, pos, pow_, config,
                                     n_steps=t2, lr=lr2,
                                     lambda_ee=lambda_ee_p2, power_pen=power_pen_p2)
    return pos, pow_, pen, ee, sat


# ── Best-of-N restart wrapper ─────────────────────────────────────────────────

def refine(inputs_raw, init_pos, init_pow, config,
           mode="single", **kwargs):
    """Run N random-init restarts; select by lowest QoS penalty."""
    B = inputs_raw.shape[0]
    best_pos = init_pos.clone()
    best_pow = init_pow.clone()

    with torch.no_grad():
        out  = torch.cat([best_pos, best_pow], dim=1)
        phys = to_physical_pass_outputs(out, config)
        b    = evaluate_pass_batch_torch(inputs_raw, phys, config)
        qos_thresh = inputs_raw[:, config.num_users * 2]
        qv = F.relu(qos_thresh.unsqueeze(1) - b.rates).pow(2).mean(dim=1)
        pv = F.relu(b.total_power - config.power_budget_w).pow(2)
        best_pen = qv + 0.25 * pv

    for _ in range(N_RESTARTS):
        p0 = torch.rand(B, init_pos.shape[1]) * 2 - 1
        w0 = torch.rand(B, init_pow.shape[1])
        if mode == "twophase":
            pos_r, pow_r, pen_r, _, _ = _adam_twophase(inputs_raw, p0, w0, config, **kwargs)
        else:
            pos_r, pow_r, pen_r, _, _ = _adam(inputs_raw, p0, w0, config, **kwargs)
        improved = pen_r < best_pen
        best_pos[improved] = pos_r[improved]
        best_pow[improved] = pow_r[improved]
        best_pen[improved] = pen_r[improved]

    return best_pos, best_pow


def run_dataset(x_raw, pred_norm, config, label="", mode="single", **kwargs):
    x_t  = torch.tensor(x_raw, dtype=torch.float32)
    ip   = torch.tensor(pred_norm[:, :9], dtype=torch.float32)
    iw   = torch.tensor(pred_norm[:, 9:], dtype=torch.float32)
    pos_p, pow_p = [], []
    n  = len(x_raw)
    t0 = time.perf_counter()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        rp, rw = refine(x_t[s:e], ip[s:e], iw[s:e], config, mode=mode, **kwargs)
        pos_p.append(rp); pow_p.append(rw)
        if label:
            print(f"  {label} [{100*e/n:5.1f}%]", flush=True)
    wall = time.perf_counter() - t0
    norm = np.concatenate([torch.cat(pos_p).numpy(), torch.cat(pow_p).numpy()], axis=1)
    return norm, wall


def metrics(x_raw, true_eval, norm, config):
    phys = norm.copy()
    phys[:, :9] *= config.position_bound_m
    phys[:, 9:] *= config.power_budget_w
    ev = evaluate_pass_batch(x_raw, phys, config)
    tq = true_eval.qos_satisfied.astype(int)
    pq = ev.qos_satisfied.astype(int)
    tp = int(((tq == 1) & (pq == 1)).sum())
    fp = int(((tq == 0) & (pq == 1)).sum())
    fn = int(((tq == 1) & (pq == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return {
        "f1":      round(f1, 4),
        "sat":     round(float(ev.qos_satisfied.mean()), 4),
        "sr_mean": round(float(ev.sum_rate.mean()), 4),
        "sr_mae":  round(float(np.abs(ev.sum_rate - true_eval.sum_rate).mean()), 4),
        "ee_mean": round(float(ev.energy_efficiency.mean()), 4),
        "pow_mean_mw": round(float(ev.sum_rate.mean() / (ev.energy_efficiency.mean() + 1e-9) * 1000 -
                                    config.circuit_power_w * 1000), 2),
    }


@torch.no_grad()
def dnn_predict(model, x_sc):
    model.eval()
    parts = []
    for i in range(0, len(x_sc), BATCH):
        b = torch.tensor(x_sc[i:i+BATCH], dtype=torch.float32)
        o = model(b)
        parts.append(torch.cat([o["positions"], o["powers"]], dim=1).cpu())
    return torch.cat(parts).numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

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

    # Prepare splits
    def _prep(idx):
        xr = corpus.inputs[idx].astype(np.float32)
        xs = scaler.transform(xr).astype(np.float32)
        yr = corpus.outputs[idx].astype(np.float32)
        te = evaluate_pass_batch(xr, yr, config)
        pn = dnn_predict(model, xs)
        return xr, pn, te

    xv, pv, tv = _prep(split.val_idx)
    xt, pt, tt = _prep(split.test_idx)

    print(f"Val: {len(xv)}  Test: {len(xt)}")
    print(f"MATLAB reference - SR:{tt.sum_rate.mean():.3f}  EE:{tt.energy_efficiency.mean():.2f}")

    # ── Reference row: best from Exp 1 (single-phase, lambda_ee=0.001) ──────
    print("\nRef: single-phase lambda_ee=0.001 (5x300, random) ...")
    ref_norm, ref_wall = run_dataset(xt, pt, config, label="ref",
                                      mode="single", n_steps=300, lr=LR, lambda_ee=0.001)
    ref_m = metrics(xt, tt, ref_norm, config)
    ref_m["ms_per_s"] = round(1000*ref_wall/len(xt), 1)
    print(f"  Ref: F1={ref_m['f1']} EE={ref_m['ee_mean']} SR={ref_m['sr_mean']}")

    results = {"reference_lam001": ref_m}

    # ─────────────────────────────────────────────────────────────────────────
    # APPROACH A: Power penalty grid search on val
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Approach A: Power penalty (alpha * total_power/P_budget) ---")
    alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
    val_A = {}
    for alpha in alphas:
        print(f"  Val alpha={alpha}...", end=" ", flush=True)
        nm, _ = run_dataset(xv, pv, config, mode="single",
                             n_steps=300, lr=LR, power_pen=alpha)
        m = metrics(xv, tv, nm, config)
        val_A[alpha] = m
        print(f"F1={m['f1']:.4f}  EE={m['ee_mean']:.2f}  SR={m['sr_mean']:.3f}")

    eligible_A = {a: m for a, m in val_A.items() if m["f1"] >= ref_m["f1"] - 0.03}
    best_alpha = max(eligible_A, key=lambda a: eligible_A[a]["ee_mean"]) if eligible_A else max(val_A, key=lambda a: val_A[a]["ee_mean"])
    print(f"\n  Best alpha={best_alpha}  (val EE={val_A[best_alpha]['ee_mean']:.2f}, F1={val_A[best_alpha]['f1']:.4f})")

    print(f"  Test alpha={best_alpha}...")
    nm_A, wall_A = run_dataset(xt, pt, config, label=f"A-a{best_alpha}",
                                mode="single", n_steps=300, lr=LR, power_pen=best_alpha)
    m_A = metrics(xt, tt, nm_A, config)
    m_A["ms_per_s"] = round(1000*wall_A/len(xt), 1)
    m_A["best_alpha"] = best_alpha
    results["approach_A_powpen"] = m_A
    print(f"  TEST A: F1={m_A['f1']}  EE={m_A['ee_mean']}  SR={m_A['sr_mean']}  pow={m_A['pow_mean_mw']}mW")

    # ─────────────────────────────────────────────────────────────────────────
    # APPROACH B: Two-phase grid search on val
    # T_total = 350 (extra steps to compensate for lower Phase 2 LR)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Approach B: Two-phase (Phase1 QoS, Phase2 EE from feasible) ---")
    tp_configs = [
        (250, 100, 0.005, 0.0),
        (250, 100, 0.01,  0.0),
        (250, 100, 0.05,  0.0),
        (200, 150, 0.01,  0.0),
        (200, 150, 0.05,  0.0),
    ]
    val_B = {}
    for t1, t2, lam, pp in tp_configs:
        tag = f"t1={t1},lam={lam}"
        print(f"  Val {tag}...", end=" ", flush=True)
        nm, _ = run_dataset(xv, pv, config, mode="twophase",
                             t1=t1, t2=t2, lambda_ee_p2=lam, power_pen_p2=pp)
        m = metrics(xv, tv, nm, config)
        val_B[tag] = {"config": (t1, t2, lam, pp), "metrics": m}
        print(f"F1={m['f1']:.4f}  EE={m['ee_mean']:.2f}  SR={m['sr_mean']:.3f}")

    eligible_B = {k: v for k, v in val_B.items() if v["metrics"]["f1"] >= ref_m["f1"] - 0.03}
    best_B_key = max(eligible_B, key=lambda k: eligible_B[k]["metrics"]["ee_mean"]) if eligible_B else max(val_B, key=lambda k: val_B[k]["metrics"]["ee_mean"])
    best_B_cfg = val_B[best_B_key]["config"]
    print(f"\n  Best two-phase: {best_B_key}")
    print(f"  val EE={val_B[best_B_key]['metrics']['ee_mean']:.2f}, F1={val_B[best_B_key]['metrics']['f1']:.4f}")

    t1b, t2b, lamb, ppb = best_B_cfg
    print(f"  Test two-phase t1={t1b},t2={t2b},lam={lamb}...")
    nm_B, wall_B = run_dataset(xt, pt, config, label=f"B-{best_B_key}",
                                mode="twophase", t1=t1b, t2=t2b,
                                lambda_ee_p2=lamb, power_pen_p2=ppb)
    m_B = metrics(xt, tt, nm_B, config)
    m_B["ms_per_s"] = round(1000*wall_B/len(xt), 1)
    m_B["best_config"] = {"t1": t1b, "t2": t2b, "lambda_ee_p2": lamb, "power_pen_p2": ppb}
    results["approach_B_twophase"] = m_B
    print(f"  TEST B: F1={m_B['f1']}  EE={m_B['ee_mean']}  SR={m_B['sr_mean']}  pow={m_B['pow_mean_mw']}mW")

    # ─────────────────────────────────────────────────────────────────────────
    # APPROACH C: Two-phase + power penalty combined
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Approach C: Two-phase + power penalty combined ---")
    tp_pp_configs = [
        (t1b, t2b, lamb, 0.3),
        (t1b, t2b, lamb, 0.5),
        (t1b, t2b, lamb, 1.0),
    ]
    val_C = {}
    for t1, t2, lam, pp in tp_pp_configs:
        tag = f"lam={lam},pp={pp}"
        print(f"  Val {tag}...", end=" ", flush=True)
        nm, _ = run_dataset(xv, pv, config, mode="twophase",
                             t1=t1, t2=t2, lambda_ee_p2=lam, power_pen_p2=pp)
        m = metrics(xv, tv, nm, config)
        val_C[tag] = {"config": (t1, t2, lam, pp), "metrics": m}
        print(f"F1={m['f1']:.4f}  EE={m['ee_mean']:.2f}  SR={m['sr_mean']:.3f}")

    eligible_C = {k: v for k, v in val_C.items() if v["metrics"]["f1"] >= ref_m["f1"] - 0.03}
    best_C_key = max(eligible_C, key=lambda k: eligible_C[k]["metrics"]["ee_mean"]) if eligible_C else max(val_C, key=lambda k: val_C[k]["metrics"]["ee_mean"])
    best_C_cfg = val_C[best_C_key]["config"]
    print(f"\n  Best combined: {best_C_key}")

    t1c, t2c, lamc, ppc = best_C_cfg
    print(f"  Test combined t1={t1c},t2={t2c},lam={lamc},pp={ppc}...")
    nm_C, wall_C = run_dataset(xt, pt, config, label=f"C-{best_C_key}",
                                mode="twophase", t1=t1c, t2=t2c,
                                lambda_ee_p2=lamc, power_pen_p2=ppc)
    m_C = metrics(xt, tt, nm_C, config)
    m_C["ms_per_s"] = round(1000*wall_C/len(xt), 1)
    m_C["best_config"] = {"t1": t1c, "t2": t2c, "lambda_ee_p2": lamc, "power_pen_p2": ppc}
    results["approach_C_combined"] = m_C
    print(f"  TEST C: F1={m_C['f1']}  EE={m_C['ee_mean']}  SR={m_C['sr_mean']}  pow={m_C['pow_mean_mw']}mW")

    # ── Summary ──────────────────────────────────────────────────────────────
    matlab_ee  = float(tt.energy_efficiency.mean())
    matlab_sr  = float(tt.sum_rate.mean())
    print("\n" + "="*70)
    print("EE BOOST SUMMARY")
    print("="*70)
    print(f"  {'Method':<40} {'F1':>6} {'EE':>7} {'SR':>7} {'ms/s':>7}")
    print(f"  {'-'*65}")
    print(f"  {'MATLAB reference':<40} {'--':>6} {matlab_ee:>7.2f} {matlab_sr:>7.3f} {'--':>7}")
    print(f"  {'Ref: lam=0.001 (Exp1)':<40} {ref_m['f1']:>6.4f} {ref_m['ee_mean']:>7.2f} {ref_m['sr_mean']:>7.3f} {ref_m['ms_per_s']:>7.1f}")
    print(f"  {f'A: power-pen alpha={best_alpha}':<40} {m_A['f1']:>6.4f} {m_A['ee_mean']:>7.2f} {m_A['sr_mean']:>7.3f} {m_A['ms_per_s']:>7.1f}")
    print(f"  {f'B: two-phase lam={lamb}':<40} {m_B['f1']:>6.4f} {m_B['ee_mean']:>7.2f} {m_B['sr_mean']:>7.3f} {m_B['ms_per_s']:>7.1f}")
    print(f"  {f'C: combined lam={lamc},pp={ppc}':<40} {m_C['f1']:>6.4f} {m_C['ee_mean']:>7.2f} {m_C['sr_mean']:>7.3f} {m_C['ms_per_s']:>7.1f}")
    print("="*70)

    out = artifact_dir / "exp_ee_boost.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()