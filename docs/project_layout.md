# Project Layout

This repository is organized to keep the MATLAB physics/optimization code separate from the DNN surrogate work.

## Top-Level Folders

- `matlab/legacy/`
  - The original corrected MATLAB bundle, kept runnable without rewriting the algorithms.
  - Contains the optimization scripts, channel model, visualizer, and dataset export helpers.
- `data/templates/`
  - Canonical CSV header files for the ML pipeline.
  - These define the exact column order expected by the DNN training code.
- `ml/pass_dnn/`
  - The PASS surrogate implementation.
  - Includes data loading, augmentation, model definition, training, evaluation, inference, and explainability.
- `docs/`
  - Architecture notes and implementation rationale.
- `data/raw/`, `data/processed/`, `data/checkpoints/`
  - Reserved for ML datasets and intermediate artifacts.
- `artifacts/`
  - Reserved for experiments and one-off outputs.

## Data Strategy

- `data/raw/` is the landing zone for every MATLAB export you want to train on.
- The trainer ignores incomplete file pairs and can combine all complete pairs into one corpus.
- Symmetry augmentation is applied only to the training split so validation and test metrics stay honest.

## Why This Split Works

- The MATLAB bundle stays close to the original physics model, so simulation results remain trustworthy.
- The ML code can evolve independently without touching the validated simulator.
- The CSV templates act as the interface contract between MATLAB dataset generation and DNN training.

## DNN Scope

The first DNN pass focuses on PASS mode only:

- `7` inputs: `user1_x`, `user1_y`, `user2_x`, `user2_y`, `user3_x`, `user3_y`, `QoS_R`
- `12` outputs: `PA1x1..PA3x3` plus `power_wg1..power_wg3`

That matches the MATLAB dataset generator exactly.
