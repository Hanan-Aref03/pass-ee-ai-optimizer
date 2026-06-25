# PASS DNN Starter

This package trains a tabular neural network surrogate for the PASS optimizer.

## Data contract

The MATLAB dataset generator writes:

- `dataset_input_pass_ml_<timestamp>.csv`
- `dataset_output_pass_ml_<timestamp>.csv`

Those files must have the same row count and the exact column order defined in `data/templates/`.

The trainer can also load the full raw corpus from `data/raw/`, ignore incomplete file pairs, and augment the training split using user/waveguide symmetry permutations.

Use the audit script first if you want a quick health check:

```bash
python scripts/audit_pass_raw_data.py
```

## Training

Run from the repository root:

```bash
python -m ml.pass_dnn --help
python -m ml.pass_dnn --input-csv matlab/legacy/csv_data/dataset_input_pass_ml_YYYY_MM_DD_HH_MM_SS.csv --output-csv matlab/legacy/csv_data/dataset_output_pass_ml_YYYY_MM_DD_HH_MM_SS.csv
```

If you omit the dataset paths, the trainer searches:

1. `data/raw/`
2. `matlab/legacy/csv_data/`
3. `matlab/legacy/`

and combines every complete PASS pair it can find.

## Inference

Use `infer.py` to produce predicted PASS outputs for a CSV of user locations:

```bash
python -m ml.pass_dnn.infer --artifact-dir ml/pass_dnn/artifacts/pass_YYYYMMDD_HHMMSS --input-csv sample_inputs.csv --output-csv predicted_pass.csv
```

The inference path projects outputs back into the feasible region:

- PA positions stay inside the service area
- adjacent PAs keep the required spacing
- waveguide powers respect the BS power budget

## Explainability

The project includes XAI helpers for:

- permutation importance
- gradient-based saliency
- a compact JSON summary of the most influential input features

Run:

```bash
python scripts/explain_pass_dnn.py --artifact-dir ml/pass_dnn/artifacts/<run-folder>
```
