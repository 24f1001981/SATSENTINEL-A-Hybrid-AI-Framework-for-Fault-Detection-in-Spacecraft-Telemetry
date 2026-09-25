"""
evaluation/metrics.py

Precision, Recall, F1, False-Alarm Rate for SATSENTINEL (AE+IF) vs baselines,
computed per channel and aggregated. Also runs the Week 3 ablation study:
same pipeline with IF removed (raw thresholded AE error) vs IF included,
comparing false-alarm rate specifically (that's IF's whole justification).

=========================== HOW TO RUN THIS FILE ===========================
This is a library module, not a standalone script -- it's meant to be
called after models/train.py has produced results. From your project root:

    python -c "
from data.load_data import load_channels
from models.train import train_all
from models.isolation_forest_stage import run_if_all
from models.baselines import threshold_baseline
from evaluation.metrics import evaluate_all_methods

channels = load_channels()
ae_results = train_all()
if_results = run_if_all(ae_results)

baseline1 = {}
for cid, (train, test, _) in channels.items():
    flags, _ = threshold_baseline(train, test)
    baseline1[cid] = flags

# baseline2 (LSTM predictor) is slower to train; see models/baselines.py
df = evaluate_all_methods(channels, if_results, baseline1, baseline1)
print(df)
"

See results/README or run pipeline.py for the simpler single-channel path.
=============================================================================
"""

import numpy as np
import pandas as pd


def flags_from_windows(window_decisions: np.ndarray, window_size: int, series_len: int):
    """Expands per-window anomaly decisions (1 window = 1 label) into a
    per-timestep boolean flag array of length series_len, by marking every
    timestep covered by an anomalous window as anomalous (OR over windows).
    window_decisions: array where True/-1 means anomalous per window.
    """
    flags = np.zeros(series_len, dtype=bool)
    is_anom = window_decisions == -1 if set(np.unique(window_decisions)) <= {-1, 1} else window_decisions.astype(bool)
    for i, anom in enumerate(is_anom):
        if anom:
            flags[i:i + window_size] = True
    return flags[:series_len]


def ground_truth_flags(anomaly_sequences, series_len: int):
    """anomaly_sequences: list of [start, end] index pairs -> boolean array."""
    gt = np.zeros(series_len, dtype=bool)
    for start, end in anomaly_sequences:
        gt[start:end] = True
    return gt


def confusion_counts(pred_flags: np.ndarray, gt_flags: np.ndarray):
    """Standard confusion-matrix counts:
    tp = correctly flagged anomalies, fp = false alarms,
    fn = missed anomalies, tn = correctly-ignored normal points.
    """
    tp = np.sum(pred_flags & gt_flags)
    fp = np.sum(pred_flags & ~gt_flags)
    fn = np.sum(~pred_flags & gt_flags)
    tn = np.sum(~pred_flags & ~gt_flags)
    return tp, fp, fn, tn


def precision_recall_f1_far(pred_flags: np.ndarray, gt_flags: np.ndarray):
    tp, fp, fn, tn = confusion_counts(pred_flags, gt_flags)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # False-alarm rate: fraction of NORMAL timesteps incorrectly flagged
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm_rate": far,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def evaluate_all_methods(channels: dict, ae_if_results: dict, baseline1_flags: dict,
                          baseline2_flags: dict, window_size: int = 100) -> pd.DataFrame:
    """Builds the metrics table: SATSENTINEL (AE+IF) vs both baselines, per channel.

    channels: {chan_id: (train, test, anomalies)} raw data
    ae_if_results: {chan_id: {"decisions", "scores", "anomalies"}} from IF stage
    baseline1_flags / baseline2_flags: {chan_id: boolean_flags_over_test}
    """
    rows = []
    for cid, (train, test, anomalies) in channels.items():
        series_len = test.shape[0]
        gt = ground_truth_flags(anomalies, series_len)

        # SATSENTINEL: expand window-level IF decisions to timestep flags
        sat_flags = flags_from_windows(ae_if_results[cid]["decisions"], window_size, series_len)
        sat_metrics = precision_recall_f1_far(sat_flags, gt)
        rows.append({"channel": cid, "method": "SATSENTINEL (AE+IF)", **sat_metrics})

        b1_metrics = precision_recall_f1_far(baseline1_flags[cid], gt)
        rows.append({"channel": cid, "method": "Baseline: Fixed Threshold", **b1_metrics})

        b2_metrics = precision_recall_f1_far(baseline2_flags[cid], gt)
        rows.append({"channel": cid, "method": "Baseline: LSTM Predictor", **b2_metrics})

    return pd.DataFrame(rows)


def ablation_study(channels: dict, ae_results: dict, window_size: int = 100,
                    error_threshold_k: float = 3.0) -> pd.DataFrame:
    """Compares: raw AE error thresholded directly (no IF) vs AE+IF, per channel,
    focused on false-alarm rate (IF's whole justification).

    ae_results: {chan_id: {"train_err", "test_err", "anomalies"}} from train.py.
    """
    from models.isolation_forest_stage import fit_isolation_forest, score_channel

    rows = []
    for cid, d in ae_results.items():
        series_len = channels[cid][1].shape[0]
        gt = ground_truth_flags(d["anomalies"], series_len)

        # No-IF: threshold the raw AE error directly (mean + k*std on train error)
        mu, sigma = d["train_err"].mean(), d["train_err"].std()
        sigma = sigma if sigma > 0 else 1e-8
        threshold = mu + error_threshold_k * sigma
        raw_decisions = d["test_err"] > threshold  # True = anomalous window
        raw_flags = flags_from_windows(raw_decisions, window_size, series_len)
        raw_metrics = precision_recall_f1_far(raw_flags, gt)

        # With IF
        iso = fit_isolation_forest(d["train_err"])
        if_decisions, _ = score_channel(iso, d["test_err"])
        if_flags = flags_from_windows(if_decisions, window_size, series_len)
        if_metrics = precision_recall_f1_far(if_flags, gt)

        rows.append({"channel": cid, "variant": "No IF (raw thresholded error)", **raw_metrics})
        rows.append({"channel": cid, "variant": "With IF", **if_metrics})

    return pd.DataFrame(rows)
