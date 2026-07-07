"""Quick timing test: run 1 batch (128 samples) for exp_ee_fast to estimate total time."""
import sys, time
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

# import everything from the fast script
sys.path.insert(0, str(ROOT / "scripts"))
import exp_ee_fast as F_

ARTIFACT = "artifacts/physics_run/pass_20260630_185731"
SEARCH   = ["data/raw","matlab/legacy/csv_data","matlab/legacy"]

bundle = load_artifact_bundle(ARTIFACT)
model, config, scaler = bundle["model"], bundle["config"], bundle["input_scaler"]
corpus = load_dataset_corpus(SEARCH)
split  = split_indices(len(corpus.inputs), seed=42)

xt  = corpus.inputs[split.test_idx[:128]].astype(np.float32)
xts = scaler.transform(xt).astype(np.float32)
pt  = F_.dnn_predict(model, xts)

x_t = torch.tensor(xt, dtype=torch.float32)
ip  = torch.tensor(pt[:, :9], dtype=torch.float32)
iw  = torch.tensor(pt[:, 9:], dtype=torch.float32)

print("Running 1 batch (128 samples) of Fast Dinkelbach...")
t0 = time.perf_counter()
rp, rw = F_.refine_threephase(x_t, ip, iw, config)
elapsed = time.perf_counter() - t0

ms_per_sample = 1000 * elapsed / 128
total_est_min = ms_per_sample * 2901 / 60000
print(f"1 batch: {elapsed:.1f}s  ->  {ms_per_sample:.1f} ms/sample")
print(f"Estimated full test time: {total_est_min:.1f} minutes")
print("Config: N_DK=%d  T_INN=%d  N_RESTART=%d  T1=%d  T2=%d" % (
    F_.N_DK, F_.T_INN, F_.N_RESTART, F_.T1, F_.T2))