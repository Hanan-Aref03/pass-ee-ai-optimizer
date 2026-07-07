"""
Generate publication-quality figures and an interactive HTML report from a PASS DNN artifact directory.

Usage:
    python scripts/visualize_results.py                          # auto-picks latest artifact
    python scripts/visualize_results.py --artifact-dir artifacts/feasibility_push/pass_20260612_191835
    python scripts/visualize_results.py --artifact-dir artifacts/pass_20260629_XXXXXX

Outputs (saved to <artifact_dir>/figures/):
    training_curves.pdf/png
    pred_vs_actual.pdf/png
    residuals_positions.pdf/png
    per_qos_metrics.pdf/png
    feasibility_metrics.pdf/png
    data_eda.pdf/png
    report.html          -- standalone interactive HTML with all figures + metric tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── colour palette (colour-blind-friendly) ──────────────────────────────────
BLUE   = "#2196F3"
GREEN  = "#4CAF50"
RED    = "#F44336"
PURPLE = "#9C27B0"
ORANGE = "#FF9800"
TEAL   = "#00BCD4"
PALETTE = [BLUE, GREEN, RED, PURPLE, ORANGE, TEAL]

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":       120,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ── helpers ──────────────────────────────────────────────────────────────────

def latest_artifact(root: Path) -> Path:
    candidates = sorted(root.rglob("metadata.json"))
    if not candidates:
        raise FileNotFoundError(f"No metadata.json found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def load_artifact(artifact_dir: Path) -> dict:
    meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    history = pd.read_csv(artifact_dir / "history.csv")
    pred_path = artifact_dir / "test_predictions.csv"
    preds = pd.read_csv(pred_path) if pred_path.exists() else None

    # Older artifacts store per_qos only in evaluation_report.json — merge it in
    if "per_qos" not in meta.get("test_metrics", {}):
        eval_path = artifact_dir / "evaluation_report.json"
        if eval_path.exists():
            eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
            if "per_qos" in eval_report:
                meta["test_metrics"]["per_qos"] = eval_report["per_qos"]

    return {"meta": meta, "history": history, "preds": preds, "dir": artifact_dir}


def savefig(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{stem}.{ext}")
    print(f"  saved {stem}.pdf / .png")
    plt.close(fig)


# ── figure 1: data EDA ───────────────────────────────────────────────────────

def fig_eda(corpus_input: Path, corpus_output: Path, figures_dir: Path) -> str:
    if not corpus_input.exists():
        return ""
    df_in  = pd.read_csv(corpus_input)
    df_out = pd.read_csv(corpus_output)

    pos_cols   = [c for c in df_out.columns if c.startswith("PA")]
    power_cols = [c for c in df_out.columns if c.startswith("power")]
    qos_counts = df_in["QoS_R"].value_counts().sort_index()

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── QoS bar ──────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    clrs = PALETTE[: len(qos_counts)]
    bars = ax.bar([f"{q:.1f}" for q in qos_counts.index], qos_counts.values,
                  color=clrs, width=0.6, edgecolor="white")
    for bar, val in zip(bars, qos_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}", ha="center", fontsize=8)
    ax.set_xlabel("QoS Threshold R (bits/s/Hz)")
    ax.set_ylabel("Samples")
    ax.set_title(f"Corpus QoS Distribution  (N={len(df_in):,} total)")

    # ── User position scatter ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    for i, c in enumerate(PALETTE[:3], start=1):
        ax2.scatter(df_in[f"user{i}_x"], df_in[f"user{i}_y"],
                    s=1, alpha=0.15, color=c, label=f"User {i}")
    rect = plt.Rectangle((-5, -5), 10, 10, linewidth=1.2,
                          edgecolor="gray", facecolor="none", linestyle="--")
    ax2.add_patch(rect)
    ax2.set_xlim(-6, 6); ax2.set_ylim(-6, 6)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    ax2.set_title("User Spatial Distribution")
    ax2.legend(markerscale=5, loc="upper right")

    # ── PA position histograms (one per waveguide) ────────────────────────────
    wg_labels = ["WG1 (PA1x1–PA3x1)", "WG2 (PA1x2–PA3x2)", "WG3 (PA1x3–PA3x3)"]
    for wg, (label, color) in enumerate(zip(wg_labels, PALETTE)):
        ax3 = fig.add_subplot(gs[1, wg + (1 if wg == 2 else wg)])  # skip middle col
        cols = pos_cols[wg * 3: wg * 3 + 3]
        for col in cols:
            ax3.hist(df_out[col], bins=40, alpha=0.55, color=color, edgecolor="none")
        ax3.set_xlabel("Position (m)")
        ax3.set_title(label, fontsize=9)

    ax_wg0 = fig.add_subplot(gs[1, 0])
    for col in pos_cols[:3]:
        ax_wg0.hist(df_out[col], bins=40, alpha=0.55, color=BLUE, edgecolor="none")
    ax_wg0.set_xlabel("Position (m)"); ax_wg0.set_title("WG1 PA Positions")

    ax_wg1 = fig.add_subplot(gs[1, 1])
    for col in pos_cols[3:6]:
        ax_wg1.hist(df_out[col], bins=40, alpha=0.55, color=GREEN, edgecolor="none")
    ax_wg1.set_xlabel("Position (m)"); ax_wg1.set_title("WG2 PA Positions")

    ax_wg2 = fig.add_subplot(gs[1, 2])
    for col in pos_cols[6:9]:
        ax_wg2.hist(df_out[col], bins=40, alpha=0.55, color=RED, edgecolor="none")
    ax_wg2.set_xlabel("Position (m)"); ax_wg2.set_title("WG3 PA Positions")

    # ── Power violin per QoS ─────────────────────────────────────────────────
    ax_pow = fig.add_subplot(gs[1, 3])
    df_pow = df_out[power_cols].copy() * 1000
    df_pow["QoS"] = df_in["QoS_R"].values
    total_pw = df_pow[power_cols].sum(axis=1)
    qos_vals = sorted(df_in["QoS_R"].unique())
    data_vp = [total_pw[df_in["QoS_R"] == qv].values for qv in qos_vals]
    vp = ax_pow.violinplot(data_vp, positions=range(len(qos_vals)),
                           showmedians=True, showextrema=False)
    for body, c in zip(vp["bodies"], PALETTE):
        body.set_facecolor(c); body.set_alpha(0.6)
    ax_pow.axhline(100, color="red", lw=1.2, ls="--", label="Budget")
    ax_pow.set_xticks(range(len(qos_vals)))
    ax_pow.set_xticklabels([f"{q:.1f}" for q in qos_vals], fontsize=8)
    ax_pow.set_xlabel("QoS"); ax_pow.set_ylabel("Total Power (mW)")
    ax_pow.set_title("Power vs QoS"); ax_pow.legend(fontsize=8)

    # ── Power histograms ──────────────────────────────────────────────────────
    for idx, (col, c) in enumerate(zip(power_cols, [BLUE, GREEN, RED])):
        ax_ph = fig.add_subplot(gs[2, idx])
        ax_ph.hist(df_out[col] * 1000, bins=40, color=c, alpha=0.75, edgecolor="none")
        ax_ph.set_xlabel("Power (mW)"); ax_ph.set_title(col)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    ax_corr = fig.add_subplot(gs[2, 3])
    short_in  = ["u1x","u1y","u2x","u2y","u3x","u3y","QoS"]
    short_out = [f"P{i+1}" for i in range(len(pos_cols))] + ["pw1","pw2","pw3"]
    df_corr = pd.concat([df_in, df_out], axis=1)
    df_corr.columns = short_in + short_out
    corr = df_corr.corr().values
    im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_corr.set_xticks(range(len(short_in + short_out)))
    ax_corr.set_xticklabels(short_in + short_out, rotation=90, fontsize=6)
    ax_corr.set_yticks(range(len(short_in + short_out)))
    ax_corr.set_yticklabels(short_in + short_out, fontsize=6)
    ax_corr.set_title("Correlation Matrix", fontsize=9)
    ax_corr.axhline(len(short_in) - 0.5, color="black", lw=1.5)
    ax_corr.axvline(len(short_in) - 0.5, color="black", lw=1.5)
    fig.colorbar(im, ax=ax_corr, fraction=0.04, pad=0.02)

    fig.suptitle("PASS Corpus — Exploratory Data Analysis", fontsize=13, fontweight="bold", y=1.01)
    savefig(fig, figures_dir, "data_eda")
    return _fig_to_b64(figures_dir / "data_eda.png")


# ── figure 2: training curves ────────────────────────────────────────────────

def fig_training(history: pd.DataFrame, best_epoch: int, figures_dir: Path) -> str:
    ep = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(ep, history["train_loss"], color=BLUE,   lw=1.5, label="Train")
    ax.plot(ep, history["val_loss"],   color=RED,    lw=1.5, label="Validation")
    ax.axvline(best_epoch, color=GREEN, lw=1.2, ls="--", label=f"Best (ep {best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("Total Loss"); ax.legend()

    ax = axes[1]
    ax.plot(ep, history["train_pos_loss"], color=BLUE,   lw=1.5, label="Train")
    ax.plot(ep, history["val_pos_loss"],   color=RED,    lw=1.5, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Position Head Loss"); ax.legend()

    ax = axes[2]
    ax.plot(ep, history["train_pow_loss"], color=GREEN, lw=1.5, label="Train")
    ax.plot(ep, history["val_pow_loss"],   color=ORANGE, lw=1.5, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Power Head Loss"); ax.legend()

    plt.suptitle("Training Curves", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, figures_dir, "training_curves")
    return _fig_to_b64(figures_dir / "training_curves.png")


# ── figure 3: predicted vs actual ────────────────────────────────────────────

def fig_pred_vs_actual(preds: pd.DataFrame, figures_dir: Path) -> str:
    pos_pred = [c for c in preds.columns if c.startswith("pred_PA")]
    pos_tgt  = [c for c in preds.columns if c.startswith("target_PA")]
    pw_pred  = [c for c in preds.columns if c.startswith("pred_power")]
    pw_tgt   = [c for c in preds.columns if c.startswith("target_power")]

    p_pred_all = preds[pos_pred].values.ravel()
    p_tgt_all  = preds[pos_tgt].values.ravel()
    pw_pred_all = preds[pw_pred].values.ravel() * 1000
    pw_tgt_all  = preds[pw_tgt].values.ravel()  * 1000

    r2_pos = 1 - np.var(p_pred_all - p_tgt_all) / np.var(p_tgt_all)
    mae_pos = np.mean(np.abs(p_pred_all - p_tgt_all))
    r2_pow = 1 - np.var(pw_pred_all - pw_tgt_all) / np.var(pw_tgt_all)
    mae_pow = np.mean(np.abs(pw_pred_all - pw_tgt_all))

    rng = np.random.default_rng(0)
    idx_s = rng.choice(len(p_pred_all), size=min(3000, len(p_pred_all)), replace=False)
    idx_p = rng.choice(len(pw_pred_all), size=min(3000, len(pw_pred_all)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.scatter(p_tgt_all[idx_s], p_pred_all[idx_s], s=3, alpha=0.3, color=BLUE, label="Test samples")
    lim = max(np.abs(p_tgt_all).max(), np.abs(p_pred_all).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.2, label="Ideal")
    ax.set_xlabel("Actual Position (m)"); ax.set_ylabel("Predicted Position (m)")
    ax.set_title(f"PA Positions\nR²={r2_pos:.4f}  MAE={mae_pos:.3f} m")
    ax.legend(markerscale=4)

    ax = axes[1]
    ax.scatter(pw_tgt_all[idx_p], pw_pred_all[idx_p], s=4, alpha=0.35, color=GREEN, label="Test samples")
    lim_p = max(np.abs(pw_tgt_all).max(), np.abs(pw_pred_all).max()) * 1.05
    ax.plot([0, lim_p], [0, lim_p], "r--", lw=1.2, label="Ideal")
    ax.set_xlabel("Actual Power (mW)"); ax.set_ylabel("Predicted Power (mW)")
    ax.set_title(f"Waveguide Powers\nR²={r2_pow:.4f}  MAE={mae_pow:.3f} mW")
    ax.legend(markerscale=4)

    plt.suptitle("Test Set: Predicted vs Actual", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, figures_dir, "pred_vs_actual")
    return _fig_to_b64(figures_dir / "pred_vs_actual.png")


# ── figure 4: residual distributions ─────────────────────────────────────────

def fig_residuals(preds: pd.DataFrame, figures_dir: Path) -> str:
    pos_pred = [c for c in preds.columns if c.startswith("pred_PA")]
    pos_tgt  = [c for c in preds.columns if c.startswith("target_PA")]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for i, (pc, tc) in enumerate(zip(pos_pred, pos_tgt)):
        ax = axes[i // 3][i % 3]
        res = preds[pc].values - preds[tc].values
        ax.hist(res, bins=40, color=BLUE, alpha=0.75, edgecolor="white", lw=0.3)
        ax.axvline(0, color="red", lw=1.2, ls="--")
        label = pc.replace("pred_", "")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Residual (m)", fontsize=8)
        ax.text(0.97, 0.95, f"μ={res.mean():.3f}\nσ={res.std():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.suptitle("PA Position Residuals  (Predicted − Actual)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, figures_dir, "residuals_positions")
    return _fig_to_b64(figures_dir / "residuals_positions.png")


# ── figure 5: per-QoS metrics ─────────────────────────────────────────────────

def fig_per_qos(per_qos: dict, figures_dir: Path) -> str:
    if not per_qos:
        return ""
    rows = [{"QoS": float(k), **v} for k, v in per_qos.items()]
    df = pd.DataFrame(rows).sort_values("QoS")
    x  = np.arange(len(df))
    labels = [f"{q:.1f}" for q in df["QoS"]]
    clrs   = PALETTE[: len(df)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    def bar_ax(ax, col, title, ylabel):
        bars = ax.bar(x, df[col], color=clrs, width=0.6, edgecolor="white")
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + df[col].max() * 0.02,
                    f"{val:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("QoS Threshold (bits/s/Hz)")
        ax.set_ylabel(ylabel); ax.set_title(title)

    bar_ax(axes[0, 0], "qos_accuracy",         "QoS Accuracy",         "Accuracy")
    bar_ax(axes[0, 1], "qos_f1",                "QoS F1 Score",          "F1")
    bar_ax(axes[0, 2], "qos_balanced_accuracy",  "Balanced Accuracy",    "Bal. Acc.")
    bar_ax(axes[1, 0], "sum_rate_mae",           "Sum-Rate MAE",         "MAE (bits/s/Hz)")
    bar_ax(axes[1, 1], "ee_mae",                 "Energy Efficiency MAE","MAE")

    # true vs pred sat rate
    ax = axes[1, 2]
    w = 0.35
    ax.bar(x - w / 2, df["true_sat_rate"], w, color=BLUE,  alpha=0.85, label="Ground Truth", edgecolor="white")
    ax.bar(x + w / 2, df["pred_sat_rate"], w, color=RED,   alpha=0.85, label="DNN Predicted", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("QoS Threshold"); ax.set_ylabel("Rate"); ax.set_ylim(0, 1.2)
    ax.set_title("QoS Satisfaction Rate\nTrue vs Predicted")
    ax.legend()

    plt.suptitle("Per-QoS Tier Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, figures_dir, "per_qos_metrics")
    return _fig_to_b64(figures_dir / "per_qos_metrics.png")


# ── figure 6: feasibility head ────────────────────────────────────────────────

def fig_feasibility(tm: dict, figures_dir: Path) -> str:
    keys = ["feasibility_accuracy", "feasibility_precision", "feasibility_recall",
            "feasibility_f1", "feasibility_balanced_accuracy"]
    if not all(k in tm for k in keys):
        return ""

    tp = tm.get("feasibility_tp", 0)
    tn = tm.get("feasibility_tn", 0)
    fp = tm.get("feasibility_fp", 0)
    fn = tm.get("feasibility_fn", 0)
    cm = np.array([[tn, fp], [fn, tp]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{int(val)}", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Infeasible", "Pred Feasible"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True Infeasible", "True Feasible"])
    ax.set_title("Feasibility Head — Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    ax = axes[1]
    names  = ["Accuracy", "Precision", "Recall", "F1", "Bal. Acc."]
    values = [tm[k] for k in keys]
    colors = [GREEN if v >= 0.85 else (ORANGE if v >= 0.70 else RED) for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.set_xlim(0, 1.18)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.set_xlabel("Score")
    ax.set_title("Feasibility Head — Classification Metrics")
    ax.axvline(0.5, color="gray", lw=0.8, ls=":")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")

    plt.suptitle("Feasibility Head Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, figures_dir, "feasibility_metrics")
    return _fig_to_b64(figures_dir / "feasibility_metrics.png")


# ── HTML report ───────────────────────────────────────────────────────────────

def _fig_to_b64(path: Path) -> str:
    import base64
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


def _metric_table_html(tm: dict, best_epoch: int, train_rows: int, val_rows: int, test_rows: int) -> str:
    per_qos = tm.get("per_qos", {})

    rows_global = [
        ("Best epoch",            best_epoch,                           ""),
        ("Train / Val / Test",    f"{train_rows:,} / {val_rows:,} / {test_rows:,}", "rows"),
        ("Position MAE (raw)",    f"{tm.get('position_mae_m_raw',float('nan')):.4f}",    "m"),
        ("Position MAE (proj.)",  f"{tm.get('position_mae_m_projected',float('nan')):.4f}", "m"),
        ("Position RMSE (proj.)", f"{tm.get('position_rmse_m_projected',float('nan')):.4f}", "m"),
        ("Position R² (proj.)",   f"{tm.get('position_r2_projected',float('nan')):.4f}", ""),
        ("±10 cm rate (proj.)",   f"{tm.get('position_within_tolerance_rate_projected',float('nan'))*100:.1f}", "%"),
        ("Power MAE (proj.)",     f"{tm.get('power_mae_w_projected',float('nan'))*1000:.4f}", "mW"),
        ("Power R² (proj.)",      f"{tm.get('power_r2_projected',float('nan')):.4f}", ""),
        ("Mean total power",      f"{tm.get('mean_total_power_w_projected',float('nan'))*1000:.2f}", "mW"),
        ("Feasible power rate",   f"{tm.get('feasible_power_rate_projected',float('nan'))*100:.1f}", "%"),
        ("Sum-Rate MAE",          f"{tm.get('sum_rate_mae',float('nan')):.4f}", "bps/Hz"),
        ("True SR mean",          f"{tm.get('true_sum_rate_mean',float('nan')):.4f}", "bps/Hz"),
        ("Pred SR mean",          f"{tm.get('pred_sum_rate_mean',float('nan')):.4f}", "bps/Hz"),
        ("EE MAE",                f"{tm.get('ee_mae',float('nan')):.4f}", ""),
        ("True EE mean",          f"{tm.get('true_ee_mean',float('nan')):.4f}", ""),
        ("Pred EE mean",          f"{tm.get('pred_ee_mean',float('nan')):.4f}", ""),
        ("QoS Accuracy",          f"{tm.get('qos_accuracy',float('nan'))*100:.2f}", "%"),
        ("QoS Precision",         f"{tm.get('qos_precision',float('nan'))*100:.2f}", "%"),
        ("QoS Recall",            f"{tm.get('qos_recall',float('nan'))*100:.2f}", "%"),
        ("QoS F1",                f"{tm.get('qos_f1',float('nan')):.4f}", ""),
        ("QoS Balanced Acc.",     f"{tm.get('qos_balanced_accuracy',float('nan'))*100:.2f}", "%"),
        ("Feasibility Accuracy",  f"{tm.get('feasibility_accuracy',float('nan'))*100:.2f}", "%"),
        ("Feasibility F1",        f"{tm.get('feasibility_f1',float('nan')):.4f}", ""),
        ("Feasibility Bal. Acc.", f"{tm.get('feasibility_balanced_accuracy',float('nan'))*100:.2f}", "%"),
    ]

    def _row_color(metric, value):
        good_metrics = {"Position R²", "Power R²", "QoS Accuracy", "QoS F1",
                        "Feasibility Accuracy", "Feasibility F1"}
        if metric in good_metrics:
            try:
                v = float(str(value).replace("%",""))
            except Exception:
                return ""
            if "%" in str(value):
                v /= 100
            if v >= 0.85:
                return "background:#e8f5e9"
            elif v >= 0.60:
                return "background:#fff9c4"
            else:
                return "background:#ffebee"
        return ""

    tr_html = "\n".join(
        f'<tr style="{_row_color(n,v)}">'
        f'<td style="padding:5px 12px;border-bottom:1px solid #eee">{n}</td>'
        f'<td style="padding:5px 12px;border-bottom:1px solid #eee;text-align:right"><b>{v}</b></td>'
        f'<td style="padding:5px 12px;border-bottom:1px solid #eee;color:#888">{u}</td>'
        f'</tr>'
        for n, v, u in rows_global
    )

    per_qos_rows = ""
    if per_qos:
        headers = ["QoS", "N", "True Sat.", "Pred Sat.", "Accuracy", "F1", "SR MAE", "EE MAE"]
        per_qos_rows = (
            f'<h3 style="margin-top:2em">Per-QoS Breakdown</h3>'
            f'<table style="border-collapse:collapse;font-size:13px;width:100%">'
            f'<tr>{"".join(f"<th style=padding:6px>{h}</th>" for h in headers)}</tr>'
        )
        for qv_str, qd in sorted(per_qos.items(), key=lambda x: float(x[0])):
            cells = [
                f"{float(qv_str):.1f}",
                f"{int(qd['count'])}",
                f"{qd['true_sat_rate']*100:.1f}%",
                f"{qd['pred_sat_rate']*100:.1f}%",
                f"{qd['qos_accuracy']*100:.1f}%",
                f"{qd['qos_f1']:.3f}",
                f"{qd['sum_rate_mae']:.4f}",
                f"{qd['ee_mae']:.4f}",
            ]
            per_qos_rows += (
                f'<tr>{"".join(f"<td style=padding:5px;border-bottom:1px solid #eee;text-align:center>{c}</td>" for c in cells)}</tr>'
            )
        per_qos_rows += "</table>"

    return (
        f'<table style="border-collapse:collapse;font-size:13px">'
        f'<tr><th style="padding:6px 12px;text-align:left;background:#f5f5f5">Metric</th>'
        f'<th style="padding:6px 12px;text-align:right;background:#f5f5f5">Value</th>'
        f'<th style="padding:6px 12px;background:#f5f5f5">Unit</th></tr>'
        f'{tr_html}</table>'
        f'{per_qos_rows}'
    )


def write_html_report(
    artifact_dir: Path,
    figures_dir: Path,
    meta: dict,
    b64_images: dict,
) -> None:
    tm = meta["test_metrics"]
    title = f"PASS DNN Results — {artifact_dir.name}"

    def img_block(b64: str, caption: str) -> str:
        if not b64:
            return ""
        return (
            f'<figure style="margin:2em 0">'
            f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:4px">'
            f'<figcaption style="text-align:center;color:#555;margin-top:6px">{caption}</figcaption>'
            f'</figure>'
        )

    table_html = _metric_table_html(
        tm,
        best_epoch=meta["best_epoch"],
        train_rows=meta["train_rows"],
        val_rows=meta["val_rows"],
        test_rows=meta["test_rows"],
    )

    mc = meta["model_config"]
    tc = meta["training_config"]
    config_html = (
        f"<b>Architecture:</b> hidden={mc['hidden_dim']}, blocks={mc['num_blocks']}, "
        f"dropout={mc['dropout']}, feasibility_head={mc.get('feasibility_head',True)}<br>"
        f"<b>Training:</b> criterion={tc['criterion']}, "
        f"augment={tc['augment_user_permutations']}, "
        f"balance_qos={tc['balance_by_qos']}, "
        f"feasibility_loss_w={tc['feasibility_loss_weight']}, "
        f"physics_loss_w={tc.get('physics_loss_weight',1.0)}, "
        f"grad_clip={tc['grad_clip']}<br>"
        f"<b>Data:</b> train={meta['train_rows']:,} / val={meta['val_rows']:,} / test={meta['test_rows']:,}"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  body{{font-family:Georgia,serif;max-width:1100px;margin:2em auto;padding:0 1.5em;color:#222;line-height:1.6}}
  h1{{color:#1565C0;border-bottom:2px solid #1565C0;padding-bottom:.4em}}
  h2{{color:#37474F;margin-top:2em}}
  .config{{background:#f8f9fa;padding:1em;border-radius:6px;font-size:13px;margin-bottom:1.5em}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:bold}}
  .good{{background:#c8e6c9;color:#1b5e20}}
  .warn{{background:#fff9c4;color:#f57f17}}
  .poor{{background:#ffcdd2;color:#b71c1c}}
</style>
</head>
<body>
<h1>PASS DNN Surrogate — Results Report</h1>
<p style="color:#555">Artifact: <code>{artifact_dir}</code></p>

<div class="config">{config_html}</div>

<h2>1. Data EDA</h2>
{img_block(b64_images.get('eda',''), 'Corpus exploratory data analysis')}

<h2>2. Training Curves</h2>
{img_block(b64_images.get('training',''), f'Loss curves — best epoch {meta["best_epoch"]}')}

<h2>3. Predicted vs Actual (Test Set)</h2>
{img_block(b64_images.get('pred',''), 'Scatter: predicted vs actual positions and powers')}

<h2>4. PA Position Residuals</h2>
{img_block(b64_images.get('residuals',''), 'Residual distributions per PA output')}

<h2>5. Per-QoS Metrics</h2>
{img_block(b64_images.get('per_qos',''), 'Per-QoS-tier breakdown of key metrics')}

<h2>6. Feasibility Head</h2>
{img_block(b64_images.get('feasibility',''), 'Confusion matrix and classification metrics')}

<h2>7. Full Metrics Table</h2>
{table_html}

<hr style="margin-top:3em">
<p style="color:#aaa;font-size:11px">Generated by scripts/visualize_results.py</p>
</body>
</html>"""

    out_path = figures_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved report.html")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate PASS DNN result figures.")
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Path to artifact directory (default: auto-detect latest).")
    parser.add_argument("--no-eda", action="store_true",
                        help="Skip EDA figure (faster if corpus is large).")
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else latest_artifact(ROOT / "artifacts")
    print(f"\nArtifact dir: {artifact_dir}")

    bundle = load_artifact(artifact_dir)
    meta, history, preds = bundle["meta"], bundle["history"], bundle["preds"]
    tm = meta["test_metrics"]

    figures_dir = artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures dir : {figures_dir}\n")

    b64 = {}

    # ── EDA ──────────────────────────────────────────────────────────────────
    if not args.no_eda:
        ci = ROOT / "data" / "processed" / "pass_merged_corpus_input.csv"
        co = ROOT / "data" / "processed" / "pass_merged_corpus_output.csv"
        print("Generating EDA figure...")
        b64["eda"] = fig_eda(ci, co, figures_dir)

    # ── Training ──────────────────────────────────────────────────────────────
    print("Generating training curves...")
    b64["training"] = fig_training(history, meta["best_epoch"], figures_dir)

    # ── Pred vs actual ────────────────────────────────────────────────────────
    if preds is not None:
        print("Generating pred-vs-actual scatter...")
        b64["pred"] = fig_pred_vs_actual(preds, figures_dir)

        print("Generating residual distributions...")
        b64["residuals"] = fig_residuals(preds, figures_dir)
    else:
        print("  (test_predictions.csv not found — skipping scatter & residuals)")

    # ── Per-QoS ───────────────────────────────────────────────────────────────
    per_qos = tm.get("per_qos", {})
    if per_qos:
        print("Generating per-QoS metrics...")
        b64["per_qos"] = fig_per_qos(per_qos, figures_dir)

    # ── Feasibility ───────────────────────────────────────────────────────────
    print("Generating feasibility head figure...")
    b64["feasibility"] = fig_feasibility(tm, figures_dir)

    # ── HTML report ───────────────────────────────────────────────────────────
    print("Writing HTML report...")
    write_html_report(artifact_dir, figures_dir, meta, b64)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PASS DNN TEST METRICS SUMMARY")
    print("="*60)
    print(f"  Artifact     : {artifact_dir.name}")
    print(f"  Best epoch   : {meta['best_epoch']}")
    print(f"  Test rows    : {meta['test_rows']:,}")
    print()
    print(f"  Position MAE (proj.)  : {tm.get('position_mae_m_projected', float('nan')):.4f} m")
    print(f"  Position RMSE (proj.) : {tm.get('position_rmse_m_projected', float('nan')):.4f} m")
    print(f"  Position R²  (proj.)  : {tm.get('position_r2_projected', float('nan')):.4f}")
    print(f"  ±10 cm rate           : {tm.get('position_within_tolerance_rate_projected', float('nan'))*100:.1f}%")
    print()
    print(f"  Power MAE (proj.)     : {tm.get('power_mae_w_projected', float('nan'))*1000:.4f} mW")
    print(f"  Power R²  (proj.)     : {tm.get('power_r2_projected', float('nan')):.4f}")
    print()
    print(f"  QoS Accuracy          : {tm.get('qos_accuracy', float('nan'))*100:.2f}%")
    print(f"  QoS F1                : {tm.get('qos_f1', float('nan')):.4f}")
    print(f"  QoS Balanced Acc.     : {tm.get('qos_balanced_accuracy', float('nan'))*100:.2f}%")
    print(f"  Sum-Rate MAE          : {tm.get('sum_rate_mae', float('nan')):.4f} bps/Hz")
    print(f"  EE MAE                : {tm.get('ee_mae', float('nan')):.4f}")
    print()
    print(f"  Feasibility Accuracy  : {tm.get('feasibility_accuracy', float('nan'))*100:.2f}%")
    print(f"  Feasibility F1        : {tm.get('feasibility_f1', float('nan')):.4f}")
    print(f"  Feasibility Bal. Acc. : {tm.get('feasibility_balanced_accuracy', float('nan'))*100:.2f}%")
    print("="*60)
    print(f"\n  Open report: {figures_dir / 'report.html'}")
    print()

    if per_qos:
        print("  Per-QoS breakdown:")
        print(f"  {'QoS':>5}  {'N':>5}  {'TrueSat':>8}  {'PredSat':>8}  {'Acc':>6}  {'F1':>6}  {'SR-MAE':>8}")
        for qv_str, qd in sorted(per_qos.items(), key=lambda x: float(x[0])):
            print(f"  {float(qv_str):5.1f}  {int(qd['count']):5d}  "
                  f"{qd['true_sat_rate']*100:7.1f}%  {qd['pred_sat_rate']*100:7.1f}%  "
                  f"{qd['qos_accuracy']*100:5.1f}%  {qd['qos_f1']:6.3f}  {qd['sum_rate_mae']:8.4f}")
        print()


if __name__ == "__main__":
    main()