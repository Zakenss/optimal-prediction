"""Error and variance helpers used by the experiments."""
from __future__ import annotations

import numpy as np


def rpe(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Relative error at each time step: ||pred - target|| / ||target||."""
    num = np.linalg.norm(pred - target, axis=1)
    den = np.linalg.norm(target, axis=1)
    return num / np.maximum(den, eps)


def mean_rpe(pred: np.ndarray, target: np.ndarray) -> float:
    """Average relative error over the whole time window."""
    return float(np.mean(rpe(pred, target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root-mean-square error over all times and components."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def error_growth_rate(pred: np.ndarray, target: np.ndarray, ts: np.ndarray) -> float:
    """Fit the slope of log(RPE) against time. Positive means growing error."""
    e = rpe(pred, target)
    mask = e > 0
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(ts[mask], np.log(e[mask]), 1)[0]
    return float(slope)


def variance_vt(trajs: list[np.ndarray]) -> np.ndarray:
    """
    Ensemble variance across M predicted trajectories:
    v(t) = (1/M) sum_m || x_m(t) - mean(t) ||^2.
    """
    stack = np.stack(trajs, axis=0)
    mean = stack.mean(axis=0, keepdims=True)
    return np.mean(np.sum((stack - mean) ** 2, axis=2), axis=0)
