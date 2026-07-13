"""
Standard DMD and delay-based HODMD helpers.

Trajectory convention: traj[k] is the state at time index k.
Shape of traj is (number_of_times, state_dimension).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _snapshots(traj: np.ndarray):
    """Build snapshot matrices X (past) and Xp (future) as columns."""
    X = traj[:-1].T
    Xp = traj[1:].T
    return X, Xp


def fit_dmd(traj: np.ndarray, rank: int | None = None):
    """
    Fit A so that x_{k+1} ≈ A x_k using SVD.

    Returns a dict with:
      A         - fitted matrix
      residual  - Frobenius norm of Xp - A X
      cond      - condition number of A
    """
    X, Xp = _snapshots(traj)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    if rank is not None:
        r = min(rank, len(s))
        U = U[:, :r]
        s = s[:r]
        Vt = Vt[:r]

    A = Xp @ (Vt.T * (1.0 / s)) @ U.T
    residual = np.linalg.norm(Xp - A @ X, ord="fro")
    cond = np.linalg.cond(A)
    return {"A": A, "residual": float(residual), "cond": float(cond)}


def predict_linear(A: np.ndarray, x0: np.ndarray, n_steps: int) -> np.ndarray:
    """Roll forward x_{k+1} = A x_k for n_steps steps. Returns (n_steps, n)."""
    n = A.shape[0]
    out = np.empty((n_steps, n))
    x = np.asarray(x0, dtype=float).copy()
    out[0] = x
    for k in range(1, n_steps):
        x = A @ x
        out[k] = x
    return out


def hankel(traj: np.ndarray, d: int) -> np.ndarray:
    """
    Build delay-embedded states of depth d.

    Each row is [x_k, x_{k-1}, ..., x_{k-d+1}].
    Returns shape (T - d + 1, n * d).
    """
    T, n = traj.shape
    rows = T - d + 1
    S = np.empty((rows, n * d))
    for i in range(rows):
        k = i + d - 1
        blocks = [traj[k - j] for j in range(d)]
        S[i] = np.concatenate(blocks)
    return S


def fit_hodmd(traj: np.ndarray, d: int, rank: int | None = None):
    """
    Fit DMD on delay-embedded states of depth d.

    Returns the augmented operator, residual, condition number,
    and a reconstruction of the original observed trajectory.
    """
    T, n = traj.shape
    S = hankel(traj, d)
    fit = fit_dmd(S, rank=rank)
    A_aug = fit["A"]

    s0 = S[0]
    aug_pred = predict_linear(A_aug, s0, S.shape[0])

    recon = np.empty((T, n))
    recon[: d - 1] = traj[: d - 1]
    recon[d - 1 :] = aug_pred[:, :n]

    return {
        "A": A_aug,
        "residual": fit["residual"],
        "cond": fit["cond"],
        "recon": recon,
        "d": d,
    }


def continuous_eigs(A: np.ndarray, dt: float):
    """
    Convert discrete eigenvalues mu of A into continuous rates:
    lam = log(mu) / dt.
    Sort by descending imaginary part.
    """
    mu, V = np.linalg.eig(A)
    lam = np.log(mu.astype(complex)) / dt
    order = np.argsort(-lam.imag)
    return lam[order], V[:, order]


def match_modes(lam_model, V_model, lam_true, V_true):
    """
    Pair model modes to true modes by maximising eigenvector overlap.
    Use the Hungarian algorithm on the cost matrix (1 - overlap).
    """
    Vm = V_model[:2, :].copy()
    Vm = Vm / np.maximum(np.linalg.norm(Vm, axis=0), 1e-12)

    Vt = V_true.copy()
    Vt = Vt / np.maximum(np.linalg.norm(Vt, axis=0), 1e-12)

    overlap = np.abs(Vm.conj().T @ Vt)
    cost = 1.0 - overlap
    rows, cols = linear_sum_assignment(cost)

    matched_model = lam_model[rows]
    matched_true = lam_true[cols]
    return matched_model, matched_true
