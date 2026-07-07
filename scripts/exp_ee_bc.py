"""Run Approaches B and C from exp_ee_boost — two-phase and combined.
Approach A (power penalty) already completed: F1=0.9406, EE=56.92, SR=2.244, pow=29.4mW.

Usage:
    python -u scripts/exp_ee_bc.py --artifact-dir artifacts/physics_run/pass_20260630_185731
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


def _adam(inputs_raw, pos0, pow0, config, n_steps, lr,
          lambda_ee=0.0, power_pen=0.0):
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


def _twophase(inputs_raw, pos0, pow0, config, t1, t2,
              lambda_ee_p2=0.01, power_pen_p2=0.0):
    pos, pow_, _ = _adam(inputs_raw, pos0, pow0, config, n_steps=t1, lr=LR)
    pos, pow_, pen = _adam(inputs_raw, pos, pow_, config, n_steps=t2, lr=LR*0.2,
                            lambda_ee=lambda_ee_p2, power_pen=power_pen_p2)
    return pos, pow_, pen


def refine_all(inputs_raw, init_pos, init_pow, config, fn, **kw):
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
        pos_r, pow_r, pen_r = fn(inputs_raw, p0, w0, config, **kw)
        improved = pen_r < best_pen
        best_pos[improved] = pos_r[improved]
        best_pow[improved] = pow_r[improved]
        best_pen[improved] = pen_r[improved]
    return best_pos, best_pow


def run_dataset(x_raw, pred_norm, config, fn, label="", **kw):
    x_t = torch.tensor(x_raw, dtype=torch.float32)
    ip  = torch.tensor(pred_norm[:, :9], dtype=torch.float32)
    iw  = torch.tensor(pred_norm[:, 9:], dtype=torch.float32)
    pp, pw = [], []
    n  = len(x_raw)
    t0 = time.perf_counter()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        rp, rw = refine_all(x_t[s:e], ip[s:e], iw[s:e], config, fn, **kw)
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
    tp = int(((tq==1)&(pq==1)).sum()); fp = int(((tq==0)&(pq==1)).sum()); fn_ = int(((tq==1)&(pq==0)).sum())
    pr = tp/(tp+fp) if (tp+fp)>0 else 0.0; re = tp/(tp+fn_) if (tp+fn_)>0 else 0.0
    f1 = 2*pr*re/(pr+re) if (pr+re)>0 else 0.0
    pow_mean = float(ev.sum_rate.mean())/(float(ev.energy_efficiency.mean())+1e-9) - config.circuit_power_w
    return {"f1": round(f1,4), "sat": round(float(ev.qos_satisfied.mean()),4),
            "sr_mean": round(float(ev.sum_rate.mean()),4),
            "sr_mae":  round(float(np.abs(ev.sum_rate-true_eval.sum_rate).mean()),4),
            "ee_mean": round(float(ev.energy_efficiency.mean()),4),
            "pow_mean_mw": round(pow_mean*1000,2)}


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
                        default=["data/raw","matlab/legacy/csv_data","matlab/legacy"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    bundle = load_artifact_bundle(str(artifact_dir))
    model, config, scaler = bundle["model"], bundle["config"], bundle["input_scaler"]
    corpus = load_dataset_corpus(args.search_root)
    split  = split_indices(len(corpus.inputs), seed=args.seed)

    def prep(idx):
        xr = corpus.inputs[idx].astype(np.float32)
        xs = scaler.transform(xr).astype(np.float32)
        yr = corpus.outputs[idx].astype(np.float32)
        return xr, dnn_predict(model, xs), evaluate_pass_batch(xr, yr, config)

    xv, pv, tv = prep(split.val_idx)
    xt, pt, tt = prep(split.test_idx)
    baseline_f1 = 0.942   # Exp1 reference F1

    # ── Approach B: Two-phase ────────────────────────────────────────────────
    print("\n--- Approach B: Two-phase (T1 QoS -> T2 EE from feasible) ---")
    tp_grid = [
        (250, 100, 0.005, 0.0),
        (250, 100, 0.01,  0.0),
        (250, 100, 0.05,  0.0),
        (200, 150, 0.01,  0.0),
        (200, 150, 0.05,  0.0),
    ]
    val_B = {}
    for t1, t2, lam, pp in tp_grid:
        tag = f"t1={t1},lam={lam}"
        print(f"  Val {tag}...", end=" ", flush=True)
        nm, _ = run_dataset(xv, pv, config, _twophase, label="",
                             t1=t1, t2=t2, lambda_ee_p2=lam, power_pen_p2=pp)
        m = get_metrics(xv, tv, nm, config)
        val_B[tag] = {"cfg": (t1,t2,lam,pp), "m": m}
        print(f"F1={m['f1']:.4f}  EE={m['ee_mean']:.2f}  SR={m['sr_mean']:.3f}  pow={m['pow_mean_mw']:.1f}mW")

    eligible_B = {k:v for k,v in val_B.items() if v["m"]["f1"] >= baseline_f1 - 0.03}
    best_B = max(eligible_B or val_B, key=lambda k: (eligible_B or val_B)[k]["m"]["ee_mean"])
    t1b,t2b,lamb,ppb = val_B[best_B]["cfg"]
    print(f"\n  Best B: {best_B}  val EE={val_B[best_B]['m']['ee_mean']:.2f}  F1={val_B[best_B]['m']['f1']:.4f}")

    print(f"  Test B: t1={t1b} t2={t2b} lam={lamb}...")
    nm_B, wall_B = run_dataset(xt, pt, config, _twophase, label=f"B",
                                t1=t1b, t2=t2b, lambda_ee_p2=lamb, power_pen_p2=ppb)
    m_B = get_metrics(xt, tt, nm_B, config); m_B["ms_per_s"] = round(1000*wall_B/len(xt),1)
    print(f"  TEST B: F1={m_B['f1']}  EE={m_B['ee_mean']}  SR={m_B['sr_mean']}  pow={m_B['pow_mean_mw']}mW  {m_B['ms_per_s']}ms/s")

    # ── Approach C: Two-phase + power penalty ────────────────────────────────
    print("\n--- Approach C: Two-phase + power penalty in Phase 2 ---")
    c_grid = [
        (t1b, t2b, lamb, 0.1),
        (t1b, t2b, lamb, 0.3),
        (t1b, t2b, lamb, 0.5),
    ]
    val_C = {}
    for t1, t2, lam, pp in c_grid:
        tag = f"lam={lam},pp={pp}"
        print(f"  Val {tag}...", end=" ", flush=True)
        nm, _ = run_dataset(xv, pv, config, _twophase, label="",
                             t1=t1, t2=t2, lambda_ee_p2=lam, power_pen_p2=pp)
        m = get_metrics(xv, tv, nm, config)
        val_C[tag] = {"cfg": (t1,t2,lam,pp), "m": m}
        print(f"F1={m['f1']:.4f}  EE={m['ee_mean']:.2f}  SR={m['sr_mean']:.3f}  pow={m['pow_mean_mw']:.1f}mW")

    eligible_C = {k:v for k,v in val_C.items() if v["m"]["f1"] >= baseline_f1 - 0.03}
    best_C = max(eligible_C or val_C, key=lambda k: (eligible_C or val_C)[k]["m"]["ee_mean"])
    t1c,t2c,lamc,ppc = val_C[best_C]["cfg"]
    print(f"\n  Best C: {best_C}  val EE={val_C[best_C]['m']['ee_mean']:.2f}  F1={val_C[best_C]['m']['f1']:.4f}")

    print(f"  Test C: t1={t1c} t2={t2c} lam={lamc} pp={ppc}...")
    nm_C, wall_C = run_dataset(xt, pt, config, _twophase, label=f"C",
                                t1=t1c, t2=t2c, lambda_ee_p2=lamc, power_pen_p2=ppc)
    m_C = get_metrics(xt, tt, nm_C, config); m_C["ms_per_s"] = round(1000*wall_C/len(xt),1)
    print(f"  TEST C: F1={m_C['f1']}  EE={m_C['ee_mean']}  SR={m_C['sr_mean']}  pow={m_C['pow_mean_mw']}mW  {m_C['ms_per_s']}ms/s")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("EE BOOST SUMMARY (including previously completed Approach A)")
    print("="*72)
    ref = {"f1":0.942,"ee_mean":49.49,"sr_mean":3.106,"ms_per_s":98.5}
    A   = {"f1":0.9406,"ee_mean":56.92,"sr_mean":2.244,"ms_per_s":"~99","pow_mean_mw":29.4}
    rows = [
        ("MATLAB reference",          "--",          78.55, 3.769, "--"),
        ("Ref: single-phase lam=0.001", ref["f1"],   ref["ee_mean"], ref["sr_mean"], ref["ms_per_s"]),
        ("A: power-pen alpha=0.1",     A["f1"],       A["ee_mean"],  A["sr_mean"],  A["ms_per_s"]),
        (f"B: two-phase lam={lamb}",   m_B["f1"],     m_B["ee_mean"],m_B["sr_mean"],m_B["ms_per_s"]),
        (f"C: combined lam={lamc},pp={ppc}", m_C["f1"], m_C["ee_mean"],m_C["sr_mean"],m_C["ms_per_s"]),
    ]
    print(f"  {'Method':<40} {'F1':>7} {'EE':>8} {'SR':>8} {'ms/s':>7}")
    print(f"  {'-'*70}")
    for row in rows:
        print(f"  {row[0]:<40} {str(row[1]):>7} {row[2]:>8.2f} {str(row[3]):>8} {str(row[4]):>7}")

    result = {
        "approach_A_from_log": A,
        "approach_B": {"best_cfg": {"t1":t1b,"t2":t2b,"lam":lamb}, "test": m_B,
                       "val_grid": {k:v["m"] for k,v in val_B.items()}},
        "approach_C": {"best_cfg": {"t1":t1c,"t2":t2c,"lam":lamc,"pp":ppc}, "test": m_C,
                       "val_grid": {k:v["m"] for k,v in val_C.items()}},
    }
    out = artifact_dir / "exp_ee_bc.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()