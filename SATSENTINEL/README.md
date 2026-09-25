# SATSENTINEL — Hybrid AI Framework for Fault Detection in Spacecraft Telemetry

LSTM Autoencoder → residual Isolation Forest → channel diagnosis pipeline for
satellite telemetry anomaly detection, trained and evaluated on the real
NASA SMAP/MSL anomaly-detection dataset (Hundman et al. 2018 /
`patrickfleith/nasa-anomaly-detection-dataset-smap-msl` on Kaggle).

## 1. Setup

```bash
cd SATSENTINEL
pip install -r requirements.txt
```

The real dataset is already included under `data/raw/`:
```
data/raw/train/<channel_id>.npy         # 82 channels
data/raw/test/<channel_id>.npy          # 82 channels
data/raw/labeled_anomalies.csv          # ground-truth anomaly windows
```
`data/load_data.py` reads from here automatically. If these files are ever
missing (e.g. a teammate clones the repo without the data), it transparently
falls back to synthetic sine-wave data so the code still runs — check the
console output for `source=SYNTHETIC` vs `source=real` to know which you're
getting.

**Note:** `data/raw/train/` and `data/raw/test/` are ~52MB and ~115MB. Add
them to `.gitignore` and have each teammate copy this folder locally rather
than committing it — GitHub will reject pushes with files this large in bulk.

## 2. Run everything end-to-end

```bash
python run_eval.py                      # trains all 8 default channels (~4 min on CPU)
python run_eval.py --channels P-1 S-1   # just a couple of channels, faster
python run_eval.py --epochs 30          # train longer for a better-fit AE
```

This does, in order:
1. Loads the 8 default channels (`P-1, S-1, E-1, A-1, M-1, T-1, C-1, D-1`) —
   a mix of SMAP/MSL, point/contextual anomaly types.
2. Trains one LSTM Autoencoder per channel, saves weights to
   `models/weights/<channel_id>.pt`.
3. Fits an Isolation Forest on each channel's reconstruction-error series.
4. Runs the fixed-threshold baseline for comparison.
5. Writes `results/evaluation_report.csv` (SATSENTINEL vs baselines,
   precision/recall/F1/false-alarm-rate per channel).
6. Writes `results/ablation_report.csv` (No-IF vs With-IF, isolating IF's
   contribution to false-alarm rate — this is the Week 3 experiment).

## 3. Run the API + dashboard

```bash
python run_eval.py                          # make sure weights exist first
uvicorn api.main:app --reload --port 8000   # backend
# then open dashboard/index.html in a browser
```

`POST /detect {"channel_id": "P-1"}` runs the full pipeline and returns
per-window decisions + the reconstruction-error series the dashboard plots.
`POST /rank {"channel_ids": [...]}` runs several channels and ranks them by
anomaly severity (the "which channel is responsible" view).

## 4. What's real right now (verified working end-to-end)

- ✅ Real Kaggle SMAP/MSL data loading (`data/load_data.py`), including the
  `ast.literal_eval` parse of `anomaly_sequences` from the CSV.
- ✅ Preprocessing: NaN interpolation, train-only-fit min-max scaling,
  sliding-window segmentation.
- ✅ LSTM Autoencoder trains on real per-channel telemetry (not just the
  synthetic fallback) — verified for all 8 default channels.
- ✅ Isolation Forest stage runs on real reconstruction-error output.
- ✅ Fixed-threshold baseline.
- ✅ `pipeline.py` and the FastAPI `/detect` + `/rank` endpoints tested live
  against a trained channel (`P-1`) — confirmed working.
- ✅ `evaluation_report.csv` and `ablation_report.csv` generated from a real
  8-channel run (see `results/`).

## 5. Known issue found while wiring this up (worth mentioning to your guide)

**Bug fixed:** `models/baselines.py::threshold_baseline` used to flatten
`train`/`test` with `.reshape(-1)`. Real SMAP/MSL channels are multivariate
(e.g. 25 columns — one true sensor value plus one-hot command features), so
flattening multiplied the array length by the feature count and crashed the
metrics step with a shape mismatch. Fixed to use column 0 (the real sensor
reading) only, which also matches how `anomaly_sequences` indices in
`labeled_anomalies.csv` are defined.

**Tuning needed, not yet done:** with `IsolationForest(contamination="auto")`,
the current run shows very high recall but poor precision (e.g. channel P-1:
recall 1.00, precision 0.09, false-alarm rate 0.99) — the IF stage is
flagging far too many windows as anomalous. This is a real, common failure
mode: `"auto"` assumes IsolationForest's internal heuristic, which doesn't
suit every channel's actual anomaly rate. Two workable interventions to try
next work session:
1. Set `contamination` explicitly per channel from the training-set anomaly
   rate you'd expect (or sweep a few fixed values like 0.01, 0.05, 0.1 and
   compare F1 in `results/evaluation_report.csv`).
2. Add a minimum-consecutive-anomalous-windows rule before flagging (reduces
   isolated false positives without hurting recall on genuine multi-window
   anomaly events).

This is exactly the kind of result a "Week 3 ablation study" section should
report — it's a finding, not a failure, and doing the contamination sweep is
good, concrete next-milestone work.

## 6. Project layout

```
SATSENTINEL/
├── data/
│   ├── load_data.py         # real + synthetic-fallback data loading
│   └── raw/                 # real Kaggle SMAP/MSL dataset (train/test/labels)
├── preprocessing/
│   └── preprocess.py        # interpolation, scaling, windowing
├── models/
│   ├── lstm_ae.py           # LSTM Autoencoder (encoder/decoder)
│   ├── train.py             # per-channel AE training loop
│   ├── isolation_forest_stage.py  # Stage 2: IF over AE error
│   ├── baselines.py         # fixed-threshold + LSTM-predictor baselines
│   └── weights/             # trained .pt files (generated by run_eval.py)
├── evaluation/
│   └── metrics.py           # precision/recall/F1/FAR + ablation study
├── api/
│   └── main.py              # FastAPI /detect and /rank endpoints
├── dashboard/
│   └── index.html           # Chart.js dashboard, calls the API
├── results/
│   ├── evaluation_report.csv
│   └── ablation_report.csv
├── pipeline.py               # single-channel integration entry point
├── run_eval.py                # trains + evaluates all channels, writes results/
└── requirements.txt
```
