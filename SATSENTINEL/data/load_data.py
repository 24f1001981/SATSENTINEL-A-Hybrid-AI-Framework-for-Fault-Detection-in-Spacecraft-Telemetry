"""
data/load_data.py

Loads NASA SMAP/MSL telemetry channels for SATSENTINEL.

"""

import ast
import os
import numpy as np
import pandas as pd

# __file__ is .../SATSENTINEL/data/load_data.py, so this resolves to
# .../SATSENTINEL/data/raw regardless of which directory you run python from.
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
LABELS_CSV = os.path.join(RAW_DIR, "labeled_anomalies.csv")

# The 8 candidate channels the team should confirm/replace on Day 1.
# Mix of SMAP + MSL, point + contextual anomaly types. All 8 of these exist
# in the real dataset, so training will use real telemetry, not synthetic.
DEFAULT_CHANNELS = [
    "P-1", "S-1", "E-1", "A-1",   # SMAP examples
    "M-1", "T-1", "C-1", "D-1",   # MSL examples
]


def _synthetic_channel(chan_id: str, n_train=2000, n_test=800, seed=None):
    """Generates a plausible single-sensor telemetry series with a few
    injected anomalous segments in the test split, so preprocessing/
    AE/IF/eval code can be exercised end-to-end without the real data.
    This is ONLY a placeholder/dev fallback -- it is never used once the
    real .npy files exist under data/raw/.
    """
    rng = np.random.default_rng(seed or abs(hash(chan_id)) % (2**32))
    t_train = np.linspace(0, 40 * np.pi, n_train)
    train = np.sin(t_train) + 0.05 * rng.standard_normal(n_train)

    t_test = np.linspace(0, 16 * np.pi, n_test)
    test = np.sin(t_test) + 0.05 * rng.standard_normal(n_test)

    # inject 1-2 anomalous windows into the test series
    anomalies = []
    n_anom = rng.integers(1, 3)
    for _ in range(n_anom):
        start = int(rng.integers(50, n_test - 100))
        length = int(rng.integers(15, 40))
        end = start + length
        kind = rng.choice(["point_spike", "contextual_drift"])
        if kind == "point_spike":
            test[start:end] += rng.choice([-1, 1]) * rng.uniform(2.5, 4.0)
        else:
            drift = np.linspace(0, rng.uniform(1.5, 3.0), length)
            test[start:end] += drift
        anomalies.append([start, end])

    return train.reshape(-1, 1), test.reshape(-1, 1), anomalies


def get_channel(chan_id: str):
    """Returns (train_array, test_array, anomaly_sequences) for one channel.

    anomaly_sequences: list of [start, end] index pairs into the TEST array.
    Falls back to synthetic data if raw files/labels aren't found.
    """
    train_path = os.path.join(RAW_DIR, "train", f"{chan_id}.npy")
    test_path = os.path.join(RAW_DIR, "test", f"{chan_id}.npy")

    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(LABELS_CSV):
        # --- real-data path ---
        train = np.load(train_path)
        test = np.load(test_path)
        labels = pd.read_csv(LABELS_CSV)
        row = labels[labels["chan_id"] == chan_id]
        if row.empty:
            anomalies = []
        else:
            # anomaly_sequences is stored as a Python-literal string in the CSV
            # (e.g. "[[2149, 2349], [4536, 4844]]"), so ast.literal_eval turns
            # it back into a real Python list of [start, end] pairs.
            anomalies = ast.literal_eval(row.iloc[0]["anomaly_sequences"])
        # SMAP/MSL channels can be 1D (single sensor) or already 2D
        # (multiple telemetry features per channel) -- normalize to 2D
        # (T, n_features) so downstream code never has to branch on shape.
        if train.ndim == 1:
            train = train.reshape(-1, 1)
        if test.ndim == 1:
            test = test.reshape(-1, 1)
        return train, test, anomalies

    # --- fallback: synthetic stand-in so the pipeline is fully runnable today ---
    return _synthetic_channel(chan_id)


def load_channels(channel_ids=None):
    """Returns {chan_id: (train, test, anomaly_sequences)} for a list of channels."""
    channel_ids = channel_ids or DEFAULT_CHANNELS
    return {cid: get_channel(cid) for cid in channel_ids}


if __name__ == "__main__":
    data = load_channels()
    for cid, (train, test, anomalies) in data.items():
        real = os.path.exists(os.path.join(RAW_DIR, "train", f"{cid}.npy"))
        print(f"{cid}: train={train.shape} test={test.shape} "
              f"anomalies={anomalies} source={'real' if real else 'SYNTHETIC'}")
