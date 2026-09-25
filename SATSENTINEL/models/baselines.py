"""
models/baselines.py

Two baselines SATSENTINEL (AE + IF) is compared against:

1. Fixed-threshold detector on raw telemetry: mean +/- k*std ("traditional method").
2. Plain LSTM predictor (no AE, no IF): trained to predict the next value,
   error = prediction residual, thresholded the same way. This matches the
   Hundman et al. (2018) NASA telemanom baseline referenced in the lit review.

Both baselines are threshold-based by design (that's the point of comparison
against the AE+IF pipeline, whose whole justification is reducing false
alarms relative to naive thresholding).

=========================== HOW TO RUN THIS FILE ===========================
    cd SATSENTINEL
    pip install torch
    python -m models.baselines
This is a library module (no real __main__ demo needed) -- it gets called
from evaluation/metrics.py and train.py. Import-check it with:
    python -c "import models.baselines"
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def threshold_baseline(train: np.ndarray, test: np.ndarray, k: float = 3.0):
    """mean +/- k*std computed on TRAIN, applied to flag TEST points.
    Returns per-timestep boolean anomaly flags for the test series.

    NOTE: real SMAP/MSL channels are multivariate (e.g. 25 columns per
    channel -- one true telemetry value plus one-hot command features), so
    a naive train.reshape(-1) would flatten across features and produce a
    flags array LONGER than the ground-truth per-timestep labels. We use
    only column 0 (the actual sensor reading) to keep this per-timestep,
    which also matches how the anomaly_sequences indices in
    labeled_anomalies.csv are defined (against series length, not
    series_length * n_features).
    """
    train = train[:, 0] if train.ndim > 1 else train.reshape(-1)
    test = test[:, 0] if test.ndim > 1 else test.reshape(-1)
    mu, sigma = train.mean(), train.std()
    sigma = sigma if sigma > 0 else 1e-8
    lower, upper = mu - k * sigma, mu + k * sigma
    flags = (test < lower) | (test > upper)
    return flags, (lower, upper)


class LSTMPredictor(nn.Module):
    """Predicts x[t+1] from x[t-lookback:t]. Single-layer LSTM + linear head.
    Unlike the autoencoder, this never reconstructs a whole window -- it
    only ever predicts one step ahead, which is the classic telemanom-style
    baseline architecture.
    """

    def __init__(self, n_features=1, hidden_size=32, lookback=10):
        super().__init__()
        self.lookback = lookback
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # only the last timestep's prediction is used


def _make_sequences(series: np.ndarray, lookback: int):
    """Turns a 1D/2D time series into (X, y) supervised pairs:
    X[i] = series[i : i+lookback], y[i] = series[i+lookback] (the next value).
    """
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def train_lstm_predictor(train: np.ndarray, lookback=10, hidden_size=32,
                          epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = train.shape[-1]
    X, y = _make_sequences(train, lookback)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    model = LSTMPredictor(n_features, hidden_size, lookback).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    predictor epoch {epoch + 1}/{epochs}  MSE={loss.item():.6f}")
    return model, device


def lstm_predictor_residuals(model, series: np.ndarray, lookback: int, device):
    X, y = _make_sequences(series, lookback)
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(X_t).cpu().numpy()
    residual = np.abs(y - pred).reshape(-1)
    # pad the first `lookback` steps (no prediction available) with 0 residual
    return np.concatenate([np.zeros(lookback), residual])


def lstm_predictor_baseline(train: np.ndarray, test: np.ndarray, lookback=10,
                             hidden_size=32, epochs=20, k: float = 3.0):
    """Full baseline #2: train predictor on train, get residuals on test,
    threshold residuals the same mean+/-k*std way as baseline #1.
    """
    model, device = train_lstm_predictor(train, lookback, hidden_size, epochs)
    train_resid = lstm_predictor_residuals(model, train, lookback, device)
    test_resid = lstm_predictor_residuals(model, test, lookback, device)

    mu, sigma = train_resid.mean(), train_resid.std()
    sigma = sigma if sigma > 0 else 1e-8
    threshold = mu + k * sigma
    flags = test_resid > threshold
    return flags, test_resid, threshold
