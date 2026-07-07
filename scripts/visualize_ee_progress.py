"""Publication figure: EE progression from DNN raw through to Fast Dinkelbach.

Produces:
  fig_ee_progress.pdf/png  -- horizontal bar chart of all methods' EE
  fig_ee_power.pdf/png     -- scatter: power vs EE for all methods + MATLAB
  fig_method_summary.pdf/png -- combined F1 + EE + power three-panel summary

Usage:
    python scripts/visualize_ee_progress.py \\
        --artifact-dir artifacts/physics_run/pass_20260630_185731
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLUE   = "#1565C0"
GREEN  = "#2E7D32"
ORANGE = "#E65100"
PURPLE = "#6A1B9A"
TEAL   = "#00695C"
RED    = "#B71C1C"
GRAY   = "#616161"
GOLD   = "#F9A825"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":    8.5,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def savefig(fig, figures_dir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/png")


# ── Master data ────────────────────────────────────────────────────────────────

METHODS = [
    # (label,                    F1,      EE,    SR,    pow_mw, color,  hatch)
    ("MATLAB solver",            None,   78.55, 3.769,  38.0,  BLUE,   "//"),
    ("DNN raw (no refinement)",  0.449,  25.56, 1.644,  None,  GRAY,   ""),
    ("Single-phase λ=0.001",     0.942,  49.49, 3.106,  None,  ORANGE, ""),
    ("Two-phase λ=0.05 (B)",     0.977,  52.03, 4.549,  77.0,  PURPLE, ""),
    ("Two-phase+pp (C)",         0.978,  53.45, 4.483,  74.0,  TEAL,   ""),
    ("Power-only Adam (E)",      0.978,  57.91, 4.322,  65.0,  GOLD,   ""),
    ("Fast Dinkelbach (G, ours)",0.986,  78.27, 2.664,  24.0,  GREEN,  ""),
]

EE_MATLAB = 78.55


def fig_ee_progress(figures_dir):
    """Horizontal bar chart: EE for each method vs MATLAB."""
    labels = [m[0] for m in METHODS]
    ees    = [m[2] for m in METHODS]
    colors = [m[5] for m in METHODS]
    hatches= [m[6] for m in METHODS]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    h = 0.6

    bars = []
    for i, (yi, ee, col, hatch) in enumerate(zip(y, ees, colors, hatches)):
        alpha = 0.55 if i == 0 else 0.85
        b = ax.barh(yi, ee, h, color=col, alpha=alpha,
                    hatch=hatch, edgecolor="white", zorder=3)
        bars.append(b[0])

    # MATLAB reference line
    ax.axvline(EE_MATLAB, color=BLUE, linestyle="--", linewidth=1.5,
               label=f"MATLAB EE = {EE_MATLAB:.2f} b/J", zorder=4)

    # Annotate bars
    for i, (bar, v) in enumerate(zip(bars, ees)):
        pct = f" ({100*(EE_MATLAB-v)/EE_MATLAB:.1f}% gap)" if i != 0 else ""
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}{pct}", va="center", ha="left", fontsize=8,
                color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Energy Efficiency (bits/J)")
    ax.set_xlim(0, EE_MATLAB * 1.25)
    ax.set_title("Energy Efficiency Progression Across Methods\n"
                 "(PASS 3×3, 0.3 THz, P_budget=100 mW)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, figures_dir, "fig_ee_progress")


def fig_ee_power(figures_dir):
    """Scatter: average power (mW) vs EE (bits/J) for methods with known power."""
    # Filter to methods with known power
    pts = [(m[0], m[2], m[4], m[5]) for m in METHODS if m[4] is not None]
    labels = [p[0] for p in pts]
    ees    = np.array([p[1] for p in pts])
    pows   = np.array([p[2] for p in pts])
    colors = [p[3] for p in pts]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (lab, ee, pw, col) in enumerate(zip(labels, ees, pows, colors)):
        marker = "*" if lab == "MATLAB solver" else ("D" if "Dinkelbach" in lab else "o")
        size   = 220 if lab == "MATLAB solver" else (200 if "Dinkelbach" in lab else 100)
        alpha  = 0.7 if lab == "MATLAB solver" else 0.9
        ax.scatter(pw, ee, s=size, color=col, marker=marker, alpha=alpha, zorder=4,
                   edgecolors="black", linewidths=0.6)
        # Label offset
        dx = 1.5 if pw > 40 else -2.0
        dy = 1.5 if "Dinkelbach" in lab else -2.5
        ax.annotate(lab, (pw, ee), xytext=(pw + dx, ee + dy),
                    fontsize=7.5, ha="left" if dx > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))

    # Draw iso-EE curves at key values
    p_range = np.linspace(5, 100, 200)
    Pc = 10.0  # circuit power mW
    for iso_ee, ls in [(50, ":"), (70, "--"), (78.55, "-")]:
        sr_curve = iso_ee * (p_range + Pc) / 1000
        ax.plot(p_range, iso_ee * np.ones_like(p_range), color=BLUE,
                linestyle=ls, linewidth=0.8, alpha=0.5,
                label=f"EE = {iso_ee:.0f} b/J" if iso_ee in (50, 78.55) else None)

    ax.set_xlabel("Average Power (mW)")
    ax.set_ylabel("Energy Efficiency (bits/J)")
    ax.set_title("Power vs EE Trade-off for Each Method\n"
                 "(P_circuit = 10 mW)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    savefig(fig, figures_dir, "fig_ee_power")


def fig_method_summary(figures_dir):
    """Three-panel figure: F1, EE, and SR side by side."""
    # exclude MATLAB (no F1) and DNN raw (no power) from the bar comparison
    methods_show = [m for m in METHODS if m[0] != "MATLAB solver"]
    labels = [m[0] for m in methods_show]
    f1s    = [m[1] for m in methods_show]
    ees    = [m[2] for m in methods_show]
    srs    = [m[3] for m in methods_show]
    colors = [m[5] for m in methods_show]

    x = np.arange(len(labels))
    w = 0.6

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # F1
    bars1 = ax1.bar(x, f1s, w, color=colors, alpha=0.85, zorder=3)
    ax1.axhline(1.0, color=BLUE, linestyle="--", linewidth=1.2, label="Ideal F1=1")
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=7.5, rotation=20, ha="right")
    ax1.set_ylabel("QoS F1 Score")
    ax1.set_title("QoS Satisfaction F1")
    for bar, v in zip(bars1, f1s):
        if v is not None:
            ax1.text(bar.get_x()+bar.get_width()/2, v+0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    # EE
    bars2 = ax2.bar(x, ees, w, color=colors, alpha=0.85, zorder=3)
    ax2.axhline(EE_MATLAB, color=BLUE, linestyle="--", linewidth=1.2,
                label=f"MATLAB EE={EE_MATLAB:.1f}")
    ax2.set_ylim(0, EE_MATLAB * 1.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=7.5, rotation=20, ha="right")
    ax2.set_ylabel("Energy Efficiency (bits/J)")
    ax2.set_title("Energy Efficiency")
    ax2.legend(loc="upper left", fontsize=8)
    for bar, v in zip(bars2, ees):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.4,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    # SR
    bars3 = ax3.bar(x, srs, w, color=colors, alpha=0.85, zorder=3)
    ax3.axhline(3.769, color=BLUE, linestyle="--", linewidth=1.2, label="MATLAB SR=3.769")
    ax3.set_ylim(0, max(srs) * 1.3)
    ax3.set_xticks(x)
    ax3.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=7.5, rotation=20, ha="right")
    ax3.set_ylabel("Mean Sum-Rate (b/s/Hz)")
    ax3.set_title("Sum-Rate")
    ax3.legend(loc="upper left", fontsize=8)
    for bar, v in zip(bars3, srs):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.04,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Method Comparison — PASS 3×3 Test Set (2901 samples)", fontsize=11)
    fig.tight_layout()
    savefig(fig, figures_dir, "fig_method_summary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    figures_dir  = artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating EE-progress figures in {figures_dir}/")
    fig_ee_progress(figures_dir)
    fig_ee_power(figures_dir)
    fig_method_summary(figures_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()