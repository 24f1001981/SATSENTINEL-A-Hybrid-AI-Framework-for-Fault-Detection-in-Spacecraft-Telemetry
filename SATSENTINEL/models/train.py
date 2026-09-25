"""
models/train.py

Trains one LSTM Autoencoder per channel (Adam + MSE loss), saves weights (.pt)
so nobody has to retrain from scratch, and returns train/test reconstruction
error series for the Isolation Forest stage.

=========================== HOW TO RUN THIS FILE ===========================
    cd SATSENTINEL
    pip install -r requirements.txt
    python -m models.train                        # trains all 8 default channels
    python -m models.train --channels P-1 S-1      # trains only these channels
    python -m models.train --epochs 5              # fewer epochs, faster/rougher

Trained weights are saved to models/weights/<channel_id>.pt, which is what
pipeline.py and api/main.py load at inference time.
=============================================================================
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.load_data import load_channels
from preprocessing.preprocess import preprocess_all
from models.lstm_ae import LSTMAutoencoder

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def train_one_channel(train_windows: np.ndarray, window_size: int, n_features: int,
                       hidden_size=64, latent_size=16, epochs=20, batch_size=32, lr=1e-3):
    """Standard PyTorch training loop: batches the windows, runs forward +
    backward passes, and reports the average MSE every 5 epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(n_features, window_size, hidden_size, latent_size).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x = torch.tensor(train_windows, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()          # clear gradients from the previous step
            recon = model(batch)     # forward pass: reconstruct the window
            loss = loss_fn(recon, batch)  # MSE between reconstruction and original
            loss.backward()          # backprop
            opt.step()               # update weights
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch + 1}/{epochs}  train_MSE={total_loss / len(x):.6f}")

    return model, device


def error_series(model, windows: np.ndarray, device) -> np.ndarray:
    """Runs the trained model in eval mode (no gradient tracking) and
    returns one reconstruction-error value per window.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32).to(device)
        err = model.reconstruction_error(x).cpu().numpy()
    return err


def train_all(channel_ids=None, window_size=100, hidden_size=64, latent_size=16,
              epochs=20, save=True):
    """Loads data -> preprocesses -> trains one AE per channel -> collects
    train/test error series (input to the Isolation Forest stage).
    """
    channels = load_channels(channel_ids)
    processed = preprocess_all(channels, window_size=window_size)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    results = {}
    for cid, d in processed.items():
        print(f"Training channel {cid} ...")
        n_features = d["train_windows"].shape[-1]
        model, device = train_one_channel(
            d["train_windows"], window_size, n_features,
            hidden_size=hidden_size, latent_size=latent_size, epochs=epochs,
        )
        train_err = error_series(model, d["train_windows"], device)
        test_err = error_series(model, d["test_windows"], device)

        if save:
            path = os.path.join(WEIGHTS_DIR, f"{cid}.pt")
            torch.save(model.state_dict(), path)
            print(f"  saved weights -> {path}")

        results[cid] = {
            "train_err": train_err,
            "test_err": test_err,
            "anomalies": d["anomalies"],
            "n_features": n_features,
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="*", default=None,
                         help="channel IDs to train, e.g. --channels P-1 S-1 (default: the 8 DEFAULT_CHANNELS)")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_all(channel_ids=args.channels, window_size=args.window_size, epochs=args.epochs)
