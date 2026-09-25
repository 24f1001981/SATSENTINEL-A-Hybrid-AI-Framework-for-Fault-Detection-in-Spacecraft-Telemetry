"""
models/lstm_ae.py

LSTM Autoencoder: encoder LSTM -> latent vector -> decoder LSTM -> reconstruction.
Reconstruction error (per-timestep MSE) is what feeds the Isolation Forest stage.

Requires torch (pip install torch). Run this file directly to sanity-check
shapes on a random tensor before wiring in real data.

=========================== HOW TO RUN THIS FILE ===========================
    cd SATSENTINEL
    pip install torch
    python -m models.lstm_ae
This does NOT need real data -- it builds one autoencoder and feeds it random
noise, just to prove the encoder/decoder shapes line up before you spend time
training on real telemetry.
=============================================================================
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Compresses a (window_size, n_features) sequence down to a single
    fixed-size latent vector, the same way a normal (non-recurrent)
    autoencoder compresses an image into a bottleneck layer.
    """

    def __init__(self, n_features, hidden_size=64, latent_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # so input/output shape is (batch, time, features)
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        # x: (batch, window_size, n_features)
        _, (h_n, _) = self.lstm(x)  # h_n: final hidden state of every layer
        h_last = h_n[-1]  # (batch, hidden_size) -- hidden state of the last layer
        z = self.to_latent(h_last)  # (batch, latent_size) -- the bottleneck
        return z


class LSTMDecoder(nn.Module):
    """Expands the latent vector back out into a full (window_size, n_features)
    reconstruction. The latent vector is repeated at every timestep as the
    decoder LSTM's input, which is the standard "seq2seq without attention"
    way to turn one vector back into a sequence.
    """

    def __init__(self, n_features, window_size, hidden_size=64, latent_size=16, num_layers=1):
        super().__init__()
        self.window_size = window_size
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_output = nn.Linear(hidden_size, n_features)

    def forward(self, z):
        # z: (batch, latent_size) -> repeat across time, decode
        h0 = self.from_latent(z)  # (batch, hidden_size)
        h0_seq = h0.unsqueeze(1).repeat(1, self.window_size, 1)  # (batch, window_size, hidden_size)
        out, _ = self.lstm(h0_seq)
        recon = self.to_output(out)  # (batch, window_size, n_features)
        return recon


class LSTMAutoencoder(nn.Module):
    """Wires encoder + decoder together. Trained with MSE loss so the
    reconstruction should match the original input; on unseen ANOMALOUS
    data, the reconstruction error spikes because the model only ever
    learned to reproduce NORMAL patterns during training.
    """

    def __init__(self, n_features=1, window_size=100, hidden_size=64, latent_size=16, num_layers=1):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_size, latent_size, num_layers)
        self.decoder = LSTMDecoder(n_features, window_size, hidden_size, latent_size, num_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def reconstruction_error(self, x):
        """Per-window MSE reconstruction error, shape (batch,).
        This is the scalar (well, one-per-window) signal that gets handed
        to the Isolation Forest stage in models/isolation_forest_stage.py.
        """
        recon = self.forward(x)
        err = torch.mean((recon - x) ** 2, dim=(1, 2))
        return err


if __name__ == "__main__":
    # Shape sanity check on random tensors (Day 2 task).
    batch, window_size, n_features = 8, 100, 1
    model = LSTMAutoencoder(n_features=n_features, window_size=window_size,
                             hidden_size=64, latent_size=16)
    dummy = torch.randn(batch, window_size, n_features)
    recon = model(dummy)
    err = model.reconstruction_error(dummy)
    print("input:", dummy.shape, "recon:", recon.shape, "error:", err.shape)
    assert recon.shape == dummy.shape
    print("OK: forward pass shapes match.")
