"""Shared setup used by every experiment script."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import matplotlib

matplotlib.use("Agg")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import dmd, system  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class Context:
    cfg: system.SystemConfig
    L: np.ndarray
    blocks: tuple
    K0: np.ndarray
    dt: float
    t_end: float
    ts: np.ndarray
    x0: np.ndarray
    proj: np.ndarray
    dmd_fit: dict
    dmd_recon: np.ndarray
    A_init: np.ndarray
    extras: dict = field(default_factory=dict)


def build_context(dt: float = 0.1, t_end: float = 12.0,
                  cfg: system.SystemConfig | None = None) -> Context:
    """Build the shared time grid, projection target, and DMD baseline."""
    cfg = cfg or system.SystemConfig()
    L = system.build_L(cfg)
    blocks = system.partition(L)
    K0 = system.memory_kernel_zero(L)
    ts = np.arange(0.0, t_end + 0.5 * dt, dt)
    x0 = np.array(cfg.x0, dtype=float)

    proj = system.true_projection(cfg, L, ts)
    dmd_fit = dmd.fit_dmd(proj, rank=2)
    dmd_recon = dmd.predict_linear(dmd_fit["A"], proj[0], len(ts))

    # Start the fitted t-model from the observed-block L_xx.
    A_init = blocks[0]

    return Context(
        cfg=cfg, L=L, blocks=blocks, K0=K0, dt=dt, t_end=t_end, ts=ts, x0=x0,
        proj=proj, dmd_fit=dmd_fit, dmd_recon=dmd_recon, A_init=A_init,
    )


def savefig(fig, name: str) -> str:
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path
