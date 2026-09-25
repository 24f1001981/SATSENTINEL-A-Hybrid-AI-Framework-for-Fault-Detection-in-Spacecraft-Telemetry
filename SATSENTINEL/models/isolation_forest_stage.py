"""
models/isolation_forest_stage.py

Stage 2: Isolation Forest over AE reconstruction-error series.

Why IF on top of the AE error, instead of just thresholding the error
directly? A fixed threshold on reconstruction error assumes a clean global
cutoff; IF instead isolates points that are *structurally* easy to separate
in error-feature space (here: error value + local error stats), which is
more robust when normal error itself has bursts of higher variance. This
is what the ablation study in Week 3 is meant to demonstrate quantitatively
(IF vs raw thresholded error, compared on false-alarm rate).

Anomaly score is derived from the AE's per-window reconstruction error:
IsolationForest.fit_predict() gives -1/1 per window; decision_function()
gives a continuous anomaly score (lower = more anomalous), which we flip
and min-max normalize to [0, 1] so scores are comparable across channels
for the ranking step.

=========================== HOW TO RUN THIS FILE ===========================
    cd SATSENTINEL
    pip install scikit-learn
    python -m models.isolation_forest_stage
This runs a self-contained demo on a synthetic error array (no AE or real
data needed) so you can confirm IF correctly flags an injected error burst.
=============================================================================
"""

import numpy as np
from sklearn.ensemble import IsolationForest


def _error_features(err: np.ndarray, roll: int = 5) -> np.ndarray:
    """Turns a 1D error series into a small feature matrix for IF:
    [error, rolling_mean, rolling_std]. Gives IF local context instead of
    a single scalar, without hand-picking a threshold.
    """
    err = np.asarray(err).reshape(-1)
    s = np.zeros_like(err)
    m = np.zeros_like(err)
    for i in range(len(err)):
        lo = max(0, i - roll)
        window = err[lo:i + 1]
        m[i] = window.mean()
        s[i] = window.std()
    return np.column_stack([err, m, s])


def fit_isolation_forest(train_err: np.ndarray, contamination: float = "auto",
                          random_state: int = 42) -> IsolationForest:
    """Fit IF on the TRAIN error series only (assumed anomaly-free)."""
    X_train = _error_features(train_err)
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X_train)
    return iso


def score_channel(iso: IsolationForest, test_err: np.ndarray):
    """Returns (decisions, normalized_scores).
    decisions: 1 = normal, -1 = anomaly (raw IF convention preserved for clarity).
    normalized_scores: higher = more anomalous, min-max scaled to [0, 1]
    so scores are comparable across channels for ranking.
    """
    X_test = _error_features(test_err)
    decisions = iso.predict(X_test)          # 1 normal, -1 anomaly
    raw_scores = -iso.decision_function(X_test)  # flip: higher = more anomalous
    lo, hi = raw_scores.min(), raw_scores.max()
    normalized = (raw_scores - lo) / (hi - lo) if hi > lo else np.zeros_like(raw_scores)
    return decisions, normalized


def run_if_all(ae_results: dict, contamination: float = "auto"):
    """ae_results: {chan_id: {"train_err", "test_err", "anomalies", ...}} from train.py.
    Returns {chan_id: {"decisions", "scores", "anomalies"}}.
    """
    out = {}
    for cid, d in ae_results.items():
        iso = fit_isolation_forest(d["train_err"], contamination=contamination)
        decisions, scores = score_channel(iso, d["test_err"])
        out[cid] = {"decisions": decisions, "scores": scores, "anomalies": d["anomalies"]}
    return out


def rank_channels(if_results: dict, top_k: int = None):
    """Channel-wise ranking: rank channels by their MAX anomaly score
    contribution, so "which channel is responsible" can be surfaced when
    multiple channels are monitored together. Returns a sorted list of
    (chan_id, max_score, mean_score).
    """
    ranking = []
    for cid, d in if_results.items():
        ranking.append((cid, float(d["scores"].max()), float(d["scores"].mean())))
    ranking.sort(key=lambda r: r[1], reverse=True)
    return ranking[:top_k] if top_k else ranking


if __name__ == "__main__":
    # Prototype on a synthetic error array (Day 4 task) before real AE output exists.
    rng = np.random.default_rng(0)
    fake_train_err = np.abs(rng.normal(0.01, 0.005, 1900))
    fake_test_err = np.abs(rng.normal(0.01, 0.005, 700))
    fake_test_err[300:330] += 0.08  # inject an obvious anomalous burst

    iso = fit_isolation_forest(fake_train_err)
    decisions, scores = score_channel(iso, fake_test_err)
    print("decisions in injected window:", decisions[300:330])
    print("mean score in injected window:", scores[300:330].mean().round(3))
    print("mean score elsewhere:", np.delete(scores, np.arange(300, 330)).mean().round(3))
