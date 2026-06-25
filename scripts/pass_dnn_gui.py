"""Tkinter interface for trying a PASS DNN artifact locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pass_dnn.data import combine_dataset_bundles, load_dataset_corpus, load_dataset_pair
from ml.pass_dnn.physics import evaluate_pass_batch
from ml.pass_dnn.schema import INPUT_COLUMNS
from ml.pass_dnn.train import load_artifact_bundle, predict_frame


DEFAULT_SEARCH_ROOTS = ("data/raw", "matlab/legacy/csv_data", "matlab/legacy")


def discover_latest_artifact_dir(search_roots: Sequence[str | Path] = ("artifacts", "ml/pass_dnn/artifacts")) -> Path | None:
    """Find the most recently modified artifact directory containing metadata."""

    candidates: list[tuple[float, Path]] = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for metadata_path in root_path.rglob("metadata.json"):
            candidates.append((metadata_path.stat().st_mtime, metadata_path.parent))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_dataset_from_metadata(metadata: dict, search_roots: Sequence[str | Path]) -> object | None:
    """Load the source PASS dataset referenced by an artifact, if available."""

    source_input = metadata.get("source_input_csv")
    source_output = metadata.get("source_output_csv")

    if isinstance(source_input, list) and isinstance(source_output, list):
        try:
            bundles = [load_dataset_pair(inp, out) for inp, out in zip(source_input, source_output)]
            return combine_dataset_bundles(bundles)
        except Exception:
            pass

    if source_input and source_output:
        try:
            return load_dataset_pair(source_input, source_output)
        except Exception:
            pass

    try:
        return load_dataset_corpus(search_roots)
    except Exception:
        return None


def _format_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


class PassDnnSimulatorApp:
    """Small desktop UI for loading an artifact and running a single simulation."""

    def __init__(
        self,
        root: tk.Tk,
        artifact_dir: str | None = None,
        device: str = "cpu",
        search_roots: Sequence[str | Path] = DEFAULT_SEARCH_ROOTS,
    ) -> None:
        self.root = root
        self.device = device
        self.search_roots = search_roots
        self.bundle: dict | None = None
        self.source_dataset = None
        self.evaluation_report: dict | None = None
        self.rng = np.random.default_rng(42)

        self.artifact_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready.")
        self.input_vars = {column: tk.StringVar(value="0.0") for column in INPUT_COLUMNS}

        self._build_ui()

        initial_artifact = artifact_dir or discover_latest_artifact_dir()
        if initial_artifact is not None:
            self.artifact_var.set(str(initial_artifact))
            self.root.after(100, self.load_artifact)

    def _build_ui(self) -> None:
        self.root.title("PASS DNN Simulator")
        self.root.geometry("1300x860")
        self.root.minsize(1100, 760)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        artifact_frame = ttk.LabelFrame(outer, text="Artifact")
        artifact_frame.pack(fill=tk.X)

        artifact_entry = ttk.Entry(artifact_frame, textvariable=self.artifact_var)
        artifact_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8), pady=8)

        ttk.Button(artifact_frame, text="Browse", command=self.browse_artifact).pack(side=tk.LEFT, padx=(0, 6), pady=8)
        ttk.Button(artifact_frame, text="Load", command=self.load_artifact).pack(side=tk.LEFT, padx=(0, 8), pady=8)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(12, 8))

        self.sim_tab = ttk.Frame(self.notebook, padding=10)
        self.metrics_tab = ttk.Frame(self.notebook, padding=10)
        self.xai_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.sim_tab, text="Simulation")
        self.notebook.add(self.metrics_tab, text="Metrics")
        self.notebook.add(self.xai_tab, text="XAI")

        self._build_simulation_tab()
        self._build_metrics_tab()
        self._build_xai_tab()

        status_bar = ttk.Label(outer, textvariable=self.status_var, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(6, 0))

    def _build_simulation_tab(self) -> None:
        paned = ttk.Panedwindow(self.sim_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        inputs_box = ttk.LabelFrame(left, text="User Layout and QoS", padding=10)
        inputs_box.pack(fill=tk.BOTH, expand=True)

        for row, column in enumerate(INPUT_COLUMNS):
            ttk.Label(inputs_box, text=column).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            entry = ttk.Entry(inputs_box, textvariable=self.input_vars[column], width=18)
            entry.grid(row=row, column=1, sticky="ew", pady=4)

        inputs_box.columnconfigure(1, weight=1)

        button_row = ttk.Frame(left)
        button_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_row, text="Run Simulation", command=self.run_simulation).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Load Random Sample", command=self.load_random_sample).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Clear", command=self.clear_inputs).pack(side=tk.LEFT)

        outputs_box = ttk.LabelFrame(right, text="Predicted PASS Configuration and Physics Simulation", padding=10)
        outputs_box.pack(fill=tk.BOTH, expand=True)

        self.simulation_text = scrolledtext.ScrolledText(outputs_box, wrap=tk.WORD, height=20, font=("Consolas", 10))
        self.simulation_text.pack(fill=tk.BOTH, expand=True)
        self.simulation_text.insert(tk.END, "Load an artifact, then run a simulation.\n")
        self.simulation_text.configure(state=tk.DISABLED)

    def _build_metrics_tab(self) -> None:
        box = ttk.LabelFrame(self.metrics_tab, text="Artifact Summary", padding=10)
        box.pack(fill=tk.BOTH, expand=True)
        self.metrics_text = scrolledtext.ScrolledText(box, wrap=tk.WORD, height=24, font=("Consolas", 10))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.insert(tk.END, "Load an artifact to view metrics.\n")
        self.metrics_text.configure(state=tk.DISABLED)

    def _build_xai_tab(self) -> None:
        box = ttk.LabelFrame(self.xai_tab, text="Explainability Summary", padding=10)
        box.pack(fill=tk.BOTH, expand=True)
        self.xai_text = scrolledtext.ScrolledText(box, wrap=tk.WORD, height=24, font=("Consolas", 10))
        self.xai_text.pack(fill=tk.BOTH, expand=True)
        self.xai_text.insert(tk.END, "Run `python scripts/explain_pass_dnn.py --artifact-dir <dir>` to generate XAI outputs.\n")
        self.xai_text.configure(state=tk.DISABLED)

    def browse_artifact(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.artifact_var.get() or str(ROOT))
        if selected:
            self.artifact_var.set(selected)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.root.update_idletasks()

    def _set_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def clear_inputs(self) -> None:
        for var in self.input_vars.values():
            var.set("0.0")
        self.set_status("Inputs cleared.")

    def load_artifact(self) -> None:
        artifact_text = self.artifact_var.get().strip()
        if not artifact_text:
            messagebox.showwarning("PASS DNN Simulator", "Choose an artifact directory first.")
            return

        artifact_dir = Path(artifact_text).expanduser()
        if not artifact_dir.exists():
            messagebox.showerror("PASS DNN Simulator", f"Artifact directory not found:\n{artifact_dir}")
            return

        try:
            self.bundle = load_artifact_bundle(artifact_dir, device=self.device)
            self.source_dataset = load_dataset_from_metadata(self.bundle["metadata"], self.search_roots)
            evaluation_report_path = artifact_dir / "evaluation_report.json"
            if evaluation_report_path.exists():
                self.evaluation_report = json.loads(evaluation_report_path.read_text(encoding="utf-8"))
            else:
                self.evaluation_report = None
            self.refresh_metrics_panel()
            self.refresh_xai_panel()
            self.set_status(f"Loaded artifact: {artifact_dir}")
        except Exception as exc:
            messagebox.showerror("PASS DNN Simulator", f"Failed to load artifact:\n{exc}")
            self.set_status("Artifact load failed.")

    def load_random_sample(self) -> None:
        if self.source_dataset is None:
            messagebox.showwarning("PASS DNN Simulator", "No source dataset is available for sampling.")
            return

        row_idx = int(self.rng.integers(0, len(self.source_dataset.inputs)))
        values = self.source_dataset.inputs[row_idx]
        for column, value in zip(INPUT_COLUMNS, values, strict=True):
            self.input_vars[column].set(f"{float(value):.6f}")
        self.set_status(f"Loaded sample row {row_idx} from the source dataset.")

    def refresh_metrics_panel(self) -> None:
        if self.bundle is None:
            self._set_text(self.metrics_text, "Load an artifact to view metrics.\n")
            return

        metadata = self.bundle["metadata"]
        lines: list[str] = []
        lines.append(f"Artifact: {self.bundle['artifact_dir']}")
        lines.append(f"Mode: {metadata.get('mode', 'unknown')}")
        lines.append("")
        lines.append("System config:")
        lines.append(json.dumps(metadata.get("system_config", {}), indent=2))
        lines.append("")
        lines.append("Model config:")
        lines.append(json.dumps(metadata.get("model_config", {}), indent=2))
        lines.append("")
        lines.append("Training config:")
        lines.append(json.dumps(metadata.get("training_config", {}), indent=2))
        lines.append("")
        lines.append(f"Best epoch: {metadata.get('best_epoch')}")
        lines.append(f"Best validation loss: {metadata.get('best_val_loss')}")
        lines.append(f"Test loss: {metadata.get('test_loss')}")
        lines.append("")
        lines.append(f"Train rows: {metadata.get('train_rows')}")
        lines.append(f"Validation rows: {metadata.get('val_rows')}")
        lines.append(f"Test rows: {metadata.get('test_rows')}")
        lines.append("")
        report_metrics = self.evaluation_report or metadata.get("test_metrics", {})
        lines.append("Test metrics:")
        lines.append(json.dumps(report_metrics, indent=2))

        if self.source_dataset is not None:
            qos_counts = self.source_dataset.input_frame["QoS_R"].value_counts().sort_index().to_dict()
            lines.append("")
            lines.append("Source dataset QoS counts:")
            lines.append(json.dumps(qos_counts, indent=2))

        self._set_text(self.metrics_text, "\n".join(lines) + "\n")

    def refresh_xai_panel(self) -> None:
        if self.bundle is None:
            self._set_text(self.xai_text, "Load an artifact to inspect explainability outputs.\n")
            return

        artifact_dir = self.bundle["artifact_dir"]
        xai_dir = artifact_dir / "xai"
        summary_path = xai_dir / "xai_summary.json"
        perm_path = xai_dir / "permutation_importance.csv"
        grad_path = xai_dir / "gradient_saliency.csv"

        if not summary_path.exists():
            self._set_text(
                self.xai_text,
                f"No XAI files found under {xai_dir}.\nRun `python scripts/explain_pass_dnn.py --artifact-dir {artifact_dir}` first.\n",
            )
            return

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines: list[str] = []
        lines.append(f"XAI directory: {xai_dir}")
        lines.append("")
        lines.append("Summary:")
        lines.append(json.dumps(summary, indent=2))

        if perm_path.exists():
            perm = pd.read_csv(perm_path).head(5)
            lines.append("")
            lines.append("Top permutation importance features:")
            lines.append(perm.to_string(index=False))

        if grad_path.exists():
            grad = pd.read_csv(grad_path).head(5)
            lines.append("")
            lines.append("Top gradient saliency features:")
            lines.append(grad.to_string(index=False))

        self._set_text(self.xai_text, "\n".join(lines) + "\n")

    def run_simulation(self) -> None:
        if self.bundle is None:
            messagebox.showwarning("PASS DNN Simulator", "Load an artifact before running a simulation.")
            return

        try:
            values = [float(self.input_vars[column].get()) for column in INPUT_COLUMNS]
        except ValueError as exc:
            messagebox.showerror("PASS DNN Simulator", f"Invalid numeric input:\n{exc}")
            return

        input_frame = pd.DataFrame([values], columns=INPUT_COLUMNS)
        try:
            predicted_frame = predict_frame(input_frame, self.bundle, project=True, include_aux=True)
            prediction_cols = list(predicted_frame.columns)
            physical_prediction = predicted_frame[[col for col in prediction_cols if col in self.bundle["schema"].output_columns]]
            system_eval = evaluate_pass_batch(
                input_frame.to_numpy(dtype=np.float32),
                physical_prediction.to_numpy(dtype=np.float32),
                self.bundle["config"],
            )
        except Exception as exc:
            messagebox.showerror("PASS DNN Simulator", f"Simulation failed:\n{exc}")
            return

        rates = system_eval.rates[0]
        qos_margin = system_eval.qos_margin[0]
        text_lines = [
            "Input row:",
            input_frame.to_string(index=False),
            "",
            "Predicted PASS configuration:",
            physical_prediction.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
            "",
            "Physics-aware simulation:",
        ]
        if "feasibility_prob" in predicted_frame.columns:
            feasibility_value = predicted_frame["feasibility_prob"].iloc[0]
            feasibility_pred_value = predicted_frame["feasibility_pred"].iloc[0]
            if pd.notna(feasibility_value):
                text_lines.append(f"  feasibility probability: {_format_float(float(feasibility_value))}")
            else:
                text_lines.append("  feasibility probability: n/a")
            if pd.notna(feasibility_pred_value):
                text_lines.append(f"  feasibility prediction: {bool(int(feasibility_pred_value))}")
            else:
                text_lines.append("  feasibility prediction: n/a")
        for idx, rate in enumerate(rates, start=1):
            text_lines.append(f"  user{idx} rate: {_format_float(rate)} bps/Hz")
        text_lines.extend(
            [
                f"  sum rate: {_format_float(system_eval.sum_rate[0])} bps/Hz",
                f"  total power: {_format_float(system_eval.total_power[0])} W",
                f"  energy efficiency: {_format_float(system_eval.energy_efficiency[0])} bps/Hz/W",
                f"  QoS satisfied: {bool(system_eval.qos_satisfied[0])}",
                f"  QoS margin: {', '.join(_format_float(value) for value in qos_margin)}",
            ]
        )

        self._set_text(self.simulation_text, "\n".join(text_lines) + "\n")
        self.set_status("Simulation complete.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the PASS DNN simulator UI.")
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    root = tk.Tk()
    PassDnnSimulatorApp(root, artifact_dir=args.artifact_dir, device=args.device)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
