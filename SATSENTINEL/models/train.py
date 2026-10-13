"""
models/train.py

train.py takes normal training data, teaches an LSTM Autoencoder what “normal” looks like, checks how badly the model reconstructs the data, and saves the trained model.

Trains one LSTM Autoencoder per channel (Adam + MSE loss), saves weights (.pt)
so nobody has to retrain from scratch, and returns train/test reconstruction
error series for the Isolation Forest stage.

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
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            bar_len = int((epoch + 1) / epochs * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"    [{bar}] epoch {epoch + 1:>2}/{epochs}  train_MSE={total_loss / len(x):.6f}")

    return model, device


def error_series(model, windows: np.ndarray, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32).to(device)
        err = model.reconstruction_error(x).cpu().numpy()
    return err


def train_all(channel_ids=None, window_size=100, hidden_size=64, latent_size=16,
              epochs=20, save=True):
    channels = load_channels(channel_ids)
    processed = preprocess_all(channels, window_size=window_size)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    results = {}
    for cid, d in processed.items():
        print(f"\n{'='*50}\nTraining channel: {cid}\n{'='*50}")
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
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_all(channel_ids=args.channels, window_size=args.window_size, epochs=args.epochs)