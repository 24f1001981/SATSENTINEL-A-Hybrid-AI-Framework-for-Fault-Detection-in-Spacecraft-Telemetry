"""
preprocessing/preprocess.py

- Missing-value interpolation
- Min-max normalization (scaler FIT ON TRAIN ONLY, applied to test -> no leakage)
- Sliding-window segmentation (parametrized window size, default 100)

=========================== HOW TO RUN THIS FILE ===========================
    cd SATSENTINEL
    python -m preprocessing.preprocess
This loads the 8 default channels, preprocesses each one, saves the windowed
train/test arrays as .npy files under data/processed/, and prints the
resulting shapes so you can sanity-check window counts.
=============================================================================
"""

import numpy as np


def interpolate_missing(arr: np.ndarray) -> np.ndarray:
    """Linear-interpolates NaNs column-wise. Edge NaNs are filled with the
    nearest valid value (forward/back-fill) since np.interp can't extrapolate.
    """
    arr = arr.astype(float).copy()
    for c in range(arr.shape[1]):
        col = arr[:, c]
        nans = np.isnan(col)
        if not nans.any():
            continue  # nothing to fix in this column
        if nans.all():
            col[:] = 0.0  # whole column missing -> can't interpolate, zero it
            continue
        idx = np.arange(len(col))
        # np.interp(x_to_fill, x_known, y_known): estimates the missing
        # values from the surrounding known points.
        col[nans] = np.interp(idx[nans], idx[~nans], col[~nans])
        arr[:, c] = col
    return arr


class MinMaxScaler:
    """Simple column-wise min-max scaler, fit on train only.

    Fitting only on train (never on test) is what prevents "data leakage":
    if we let the scaler see test-set min/max, the model would implicitly
    know something about the test distribution before evaluation, which
    would make reported metrics optimistic.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, arr: np.ndarray):
        self.min_ = arr.min(axis=0)
        self.max_ = arr.max(axis=0)
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return (arr - self.min_) / denom

    def fit_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.fit(arr).transform(arr)


def make_windows(arr: np.ndarray, window_size: int = 100, stride: int = 1) -> np.ndarray:
    """Sliding-window segmentation. arr: (T, n_features) -> (n_windows, window_size, n_features).

    stride=1 means every possible overlapping window is produced (max data,
    slower training); raise stride to speed up training at the cost of
    fewer, less-overlapping windows.
    """
    T = arr.shape[0]
    if T < window_size:
        # pad by repeating the last row so very short channels don't crash the pipeline
        pad = np.repeat(arr[-1:], window_size - T, axis=0)
        arr = np.vstack([arr, pad])
        T = arr.shape[0]
    n_windows = (T - window_size) // stride + 1
    windows = np.stack([arr[i:i + window_size] for i in range(0, n_windows * stride, stride)])
    return windows


def preprocess_channel(train: np.ndarray, test: np.ndarray, window_size: int = 100):
    """Full pipeline for one channel: interpolate -> scale (fit on train) -> window.

    Returns: train_windows, test_windows, scaler
    """
    train = interpolate_missing(train)
    test = interpolate_missing(test)

    scaler = MinMaxScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    train_windows = make_windows(train_scaled, window_size)
    test_windows = make_windows(test_scaled, window_size)
    return train_windows, test_windows, scaler


def preprocess_all(channels: dict, window_size: int = 100, save_dir: str = None):
    """channels: {chan_id: (train, test, anomalies)} from data/load_data.py.
    Returns {chan_id: {"train_windows", "test_windows", "anomalies", "scaler"}}.
    Optionally saves windowed .npy outputs to save_dir.
    """
    import os
    out = {}
    for cid, (train, test, anomalies) in channels.items():
        train_w, test_w, scaler = preprocess_channel(train, test, window_size)
        out[cid] = {
            "train_windows": train_w,
            "test_windows": test_w,
            "anomalies": anomalies,
            "scaler": scaler,
        }
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{cid}_train_windows.npy"), train_w)
            np.save(os.path.join(save_dir, f"{cid}_test_windows.npy"), test_w)
    return out


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.load_data import load_channels

    channels = load_channels()
    save_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    processed = preprocess_all(channels, window_size=100, save_dir=save_dir)
    for cid, d in processed.items():
        print(f"{cid}: train_windows={d['train_windows'].shape} "
              f"test_windows={d['test_windows'].shape}")
