"""
pipeline.py

Single integration entry point: raw telemetry in -> preprocessed -> AE ->
error series out -> IF -> decision + score, for any channel by ID.
This is what api/main.py calls, and what the full-team Day-10/Day-14
end-to-end test runs against.

=========================== HOW TO RUN THIS FILE ===========================
    cd SATSENTINEL
    pip install -r requirements.txt
    python models/train.py            # train weights first, otherwise you'll
                                       # just get an untrained-model warning
    python pipeline.py                # runs the full pipeline on channel P-1
=============================================================================
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
from data.load_data import get_channel
from preprocessing.preprocess import preprocess_channel
from models.lstm_ae import LSTMAutoencoder
from models.isolation_forest_stage import fit_isolation_forest, score_channel, rank_channels

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "models", "weights")
WINDOW_SIZE = 100


def _load_model(chan_id: str, n_features: int, window_size: int = WINDOW_SIZE):
    """Builds a fresh model with the right shape, then loads trained weights
    for this specific channel if they exist (each channel gets its own AE,
    since normal telemetry patterns differ per sensor).
    """
    model = LSTMAutoencoder(n_features=n_features, window_size=window_size)
    path = os.path.join(WEIGHTS_DIR, f"{chan_id}.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        # No trained weights yet for this channel -- caller should train first.
        # Left un-trained here only so the API doesn't hard-crash during dev.
        print(f"WARNING: no trained weights found for {chan_id} at {path}; "
              f"using an untrained model (run models/train.py first).")
    model.eval()
    return model


def run_pipeline(chan_id: str = None, raw_array: np.ndarray = None,
                  window_size: int = WINDOW_SIZE):
    """Runs the full two-stage pipeline for one channel.

    Provide either chan_id (loads real/synthetic data + trained weights)
    or raw_array (a 1D/2D array of new telemetry) plus chan_id for weights.

    Returns dict with: decision ("normal"/"anomaly" per window), scores,
    ranking-relevant max/mean score, and the underlying error series.
    """
    if raw_array is not None:
        train, _, anomalies = get_channel(chan_id)
        test = np.asarray(raw_array)
        if test.ndim == 1:
            test = test.reshape(-1, 1)
    else:
        train, test, anomalies = get_channel(chan_id)

    train_windows, test_windows, _ = preprocess_channel(train, test, window_size)
    n_features = test_windows.shape[-1]

    model = _load_model(chan_id, n_features, window_size)
    with torch.no_grad():
        train_err = model.reconstruction_error(
            torch.tensor(train_windows, dtype=torch.float32)).numpy()
        test_err = model.reconstruction_error(
            torch.tensor(test_windows, dtype=torch.float32)).numpy()

    iso = fit_isolation_forest(train_err)
    decisions, scores = score_channel(iso, test_err)

    return {
        "channel": chan_id,
        "decision": ["anomaly" if d == -1 else "normal" for d in decisions],
        "scores": scores.tolist(),
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "test_error_series": test_err.tolist(),
        "ground_truth_anomalies": anomalies,
    }


def run_pipeline_multi(chan_ids: list):
    """Runs the pipeline across multiple channels and ranks them by
    contribution, for the 'which channel is responsible' view.
    """
    if_results = {}
    all_out = {}
    for cid in chan_ids:
        out = run_pipeline(cid)
        all_out[cid] = out
        if_results[cid] = {"scores": np.array(out["scores"])}
    ranking = sorted(
        ((cid, d["scores"].max(), d["scores"].mean()) for cid, d in if_results.items()),
        key=lambda r: r[1], reverse=True,
    )
    return {"per_channel": all_out, "ranking": ranking}


if __name__ == "__main__":
    result = run_pipeline("P-1")
    print(f"channel={result['channel']}  max_score={result['max_score']:.3f}  "
          f"mean_score={result['mean_score']:.3f}")
    print("ground truth anomalies (test-index ranges):", result["ground_truth_anomalies"])
