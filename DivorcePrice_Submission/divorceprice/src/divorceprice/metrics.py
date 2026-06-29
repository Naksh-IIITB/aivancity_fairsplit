"""Metrics — RMSLE is the Kaggle-standard scoring function for this task."""
from __future__ import annotations

import numpy as np


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared log error. Both inputs in raw price space."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), a_min=1.0, a_max=None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def rmsle_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Both inputs are already in log1p space — saves a round-trip."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    return float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2)))
