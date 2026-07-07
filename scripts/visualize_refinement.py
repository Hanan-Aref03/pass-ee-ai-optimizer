"""Generate publication figures for the refinement results section.

Produces three figures into <artifact_dir>/figures/:
  fig_pertier_f1.pdf/png   -- per-QoS-tier F1 grouped bar chart (Table II visual)
  fig_ablation.pdf/png     -- method comparison: sat rate + F1 (Table III visual)
  fig_rate_ee.pdf/png      -- sum-rate and EE comparison across methods

Usage:
    python scripts/visualize_refinement.py --artifact-dir artifacts/physics_run/pass_20260630_185731
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Colour-blind-friendly palette
BLUE   = "#2196F3"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
GRAY   = "#9E9E9E"
RED    = "#F44336"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":    9,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.grid.axis":    "y",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def savefig(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{stem}.{ext}")
    plt.close(fig)


def fig_pertier_f1(pertier: dict, figures_dir: Path) -> None:
    tiers = sorted(pertier.keys(), key=float)
    x = np.arange(len(tiers))
    w = 0.28

    raw_f1 = [pertier[g]["raw_f1"] for g in tiers]
    ref_f1 = [pertier[g]["ref_f1"] for g in tiers]
    true_sat = [pertier[g]["true_sat"] for g in tiers]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x - w, raw_f1,  w, label="DNN raw",           color=ORANGE, alpha=0.85, zorder=3)
    ax.bar(x,     ref_f1,  w, label="DNN + 5×300 Adam",  color=GREEN,  alpha=0.85, zorder=3)
    ax.bar(x + w, true_sat, w, label="MATLAB sat. (ref)", color=BLUE,  alpha=0.40,
           hatch="//", edgecolor=BLUE, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"γ={g}" for g in tiers], rotation=30, ha="right")
    ax.set_ylabel("F1 / QoS Satisfaction Rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-QoS-Tier F1: DNN Raw vs. DNN + Refinement")
    ax.legend(loc="upper right")

    # Annotate refined F1 on top of each green bar
    for xi, v in zip(x, ref_f1):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#2E7D32")

    fig.tight_layout()
    savefig(fig, figures_dir, "fig_pertier_f1")
    print(f"  Saved fig_pertier_f1.pdf/png")


def fig_ablation(res_dnn: dict, res_rand: dict, figures_dir: Path) -> None:
    methods = ["MATLAB\nsolver", "DNN\nraw", "Random-only\n5×300 Adam", "DNN warm-start\n5×300 Adam"]
    sat  = [res_dnn["true_sat_rate"],
            res_dnn["raw"]["recall"],         # raw sat ≈ TP/(TP+FN) = recall
            res_rand["refined"]["tp"] / (res_rand["refined"]["tp"] + res_rand["refined"]["fn"]),
            res_dnn["refined"]["recall"]]
    # use qos_sat directly: true_sat_rate, raw sat=0.283, rand=0.881, dnn=0.873
    sat = [res_dnn["true_sat_rate"], 0.283, res_rand["refined"]["tp"] / res_rand["n_test"],
           res_dnn["refined"]["tp"] / res_dnn["n_test"]]
    f1s = [float("nan"), res_dnn["raw"]["f1"], res_rand["refined"]["f1"], res_dnn["refined"]["f1"]]

    x = np.arange(len(methods))
    w = 0.32
    colors_sat = [BLUE, ORANGE, GREEN, GREEN]
    colors_f1  = [GRAY, ORANGE, GREEN, GREEN]
    alphas = [0.5, 0.85, 0.65, 0.95]

    fig, ax = plt.subplots(figsize=(8, 4.2))

    bars1 = ax.bar(x - w/2, sat, w, color=colors_sat, alpha=0.7, label="QoS sat. rate", zorder=3)
    bars2 = ax.bar(x + w/2, [v if not np.isnan(v) else 0 for v in f1s], w,
                   color=colors_f1, alpha=0.95, hatch=["", "", "//", ""], label="F1", zorder=3)

    for bar, v in zip(bars1, sat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars2, f1s):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Rate / F1")
    ax.set_ylim(0, 1.12)
    ax.set_title("QoS Satisfaction Rate and F1 by Method")
    sat_patch = mpatches.Patch(color=BLUE, alpha=0.7, label="QoS sat. rate")
    f1_patch  = mpatches.Patch(color=GREEN, alpha=0.95, label="F1 score")
    ax.legend(handles=[sat_patch, f1_patch], loc="lower right")

    fig.tight_layout()
    savefig(fig, figures_dir, "fig_ablation")
    print(f"  Saved fig_ablation.pdf/png")


def fig_rate_ee(res_dnn: dict, res_rand: dict, figures_dir: Path) -> None:
    methods   = ["MATLAB", "DNN raw", "Random-only\n+refine", "DNN warm-start\n+refine"]
    colors    = [BLUE, ORANGE, GREEN, GREEN]
    alphas_sr = [0.55, 0.85, 0.65, 0.95]

    re = res_dnn["rate_ee"]
    re_r = res_rand["rate_ee"]
    sum_rates = [re["true_sr_mean"], re["raw_sr_mean"], re_r["ref_sr_mean"], re["ref_sr_mean"]]
    ee_vals   = [re["true_ee_mean"], re["raw_ee_mean"], re_r["ref_ee_mean"], re["ref_ee_mean"]]

    x = np.arange(len(methods))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Sum-rate
    for i, (xi, v) in enumerate(zip(x, sum_rates)):
        ax1.bar(xi, v, w, color=colors[i], alpha=alphas_sr[i], zorder=3)
        ax1.text(xi, v + 0.04, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8.5)
    ax1.set_ylabel("Mean Sum-Rate (b/s/Hz)")
    ax1.set_ylim(0, max(sum_rates) * 1.2)
    ax1.set_title("Mean Sum-Rate Comparison")

    # EE
    for i, (xi, v) in enumerate(zip(x, ee_vals)):
        ax2.bar(xi, v, w, color=colors[i], alpha=alphas_sr[i], zorder=3)
        ax2.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8.5)
    ax2.set_ylabel("Mean Energy Efficiency (bits/J)")
    ax2.set_ylim(0, max(ee_vals) * 1.25)
    ax2.set_title("Energy Efficiency Comparison")

    fig.suptitle("Rate and Energy Efficiency by Method", fontsize=11)
    fig.tight_layout()
    savefig(fig, figures_dir, "fig_rate_ee")
    print(f"  Saved fig_rate_ee.pdf/png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    figures_dir  = artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pertier_path  = artifact_dir / "pertier_f1.json"
    res_dnn_path  = artifact_dir / "refinement_results.json"
    res_rand_path = artifact_dir / "refinement_results_random_only.json"

    for p in (pertier_path, res_dnn_path, res_rand_path):
        if not p.exists():
            sys.exit(f"Missing: {p}\nRun refine_predictions.py and per_tier_f1.py first.")

    pertier  = json.loads(pertier_path.read_text())
    res_dnn  = json.loads(res_dnn_path.read_text())
    res_rand = json.loads(res_rand_path.read_text())

    print(f"Generating refinement figures in {figures_dir}/")
    fig_pertier_f1(pertier, figures_dir)
    fig_ablation(res_dnn, res_rand, figures_dir)
    fig_rate_ee(res_dnn, res_rand, figures_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()