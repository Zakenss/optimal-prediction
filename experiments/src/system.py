"""
Build the 4D test system used in the preliminary experiments.

State layout:
  y = [q1, p1, q2, p2]
  observed (resolved):   x = [q1, p1]
  hidden (unresolved):   z = [q2, p2]

This is a linear coupled-oscillator stand-in. The paper uses a nonlinear
Hamiltonian system (Eq. 21). Replacing this file with that system is the
next implementation step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class SystemConfig:
    omega1: float = 1.0
    omega2: float = 1.7
    gamma1: float = 0.05
    gamma2: float = 0.10
    kappa: float = 0.45
    x0: tuple = (1.0, 0.0)
    z0: tuple = (0.0, 0.0)


def build_L(cfg: SystemConfig) -> np.ndarray:
    """Build the 4x4 continuous-time matrix L for dy/dt = L y."""
    w1 = cfg.omega1
    w2 = cfg.omega2
    g1 = cfg.gamma1
    g2 = cfg.gamma2
    k = cfg.kappa
    L = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-(w1 ** 2), -2.0 * g1, 0.0, k],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -k, -(w2 ** 2), -2.0 * g2],
        ]
    )
    return L


def partition(L: np.ndarray):
    """Split L into the four 2x2 blocks (xx, xz, zx, zz)."""
    L_xx = L[:2, :2]
    L_xz = L[:2, 2:]
    L_zx = L[2:, :2]
    L_zz = L[2:, 2:]
    return L_xx, L_xz, L_zx, L_zz


def memory_kernel_zero(L: np.ndarray) -> np.ndarray:
    """Compute K0 = L_xz @ L_zx (leading memory contribution at s = 0)."""
    _, L_xz, L_zx, _ = partition(L)
    return L_xz @ L_zx


def full_trajectory(L: np.ndarray, y0: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Integrate the linear system exactly: y(t) = expm(L * t) @ y0."""
    out = np.empty((len(ts), 4))
    for i, t in enumerate(ts):
        out[i] = expm(L * t) @ y0
    return out


def true_projection(cfg: SystemConfig, L: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """
    Return the observed part of the exact trajectory when the hidden
    initial state is set to its mean (z0). Shape: (len(ts), 2).
    """
    y0 = np.array([cfg.x0[0], cfg.x0[1], cfg.z0[0], cfg.z0[1]])
    full = full_trajectory(L, y0, ts)
    return full[:, :2]


def true_koopman_eigs(L: np.ndarray):
    """
    Return eigenvalues of L and the first two rows of the eigenvectors
    (observed components), each column normalised.
    """
    lam, V = np.linalg.eig(L)
    Vr = V[:2, :].copy()
    norms = np.linalg.norm(Vr, axis=0)
    norms[norms == 0.0] = 1.0
    Vr = Vr / norms
    return lam, Vr
