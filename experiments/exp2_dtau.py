"""
Experiment 2: sweep the time step and measure RMSE of the discrete
and analytic t-model against the projection.
"""
from __future__ import annotations

import numpy as np

from experiments.common import build_context
from src import metrics, tmodel


def run(dtaus=(0.05, 0.1, 0.5, 1.0), t_end: float = 30.0) -> dict:
    rmse_discrete = []
    rmse_model = []

    for dt in dtaus:
        ctx = build_context(dt=dt, t_end=t_end)
        L_xx = ctx.blocks[0]

        pred_disc = np.array(tmodel.rollout((L_xx, ctx.K0), ctx.x0, dt, len(ctx.ts)))
        rmse_discrete.append(metrics.rmse(pred_disc, ctx.proj))

        pred_rk4 = tmodel.analytic_tmodel(L_xx, ctx.K0, ctx.x0, ctx.ts)
        rmse_model.append(metrics.rmse(pred_rk4, ctx.proj))

    dt_arr = np.array(dtaus)
    disc = np.array(rmse_discrete)
    slope = float(np.polyfit(np.log(dt_arr), np.log(np.maximum(disc, 1e-15)), 1)[0])

    return {
        "dtaus": list(dtaus),
        "rmse_discrete": rmse_discrete,
        "rmse_model": rmse_model,
        "empirical_order_slope": slope,
    }


if __name__ == "__main__":
    out = run()
    for dt, rd, rm in zip(out["dtaus"], out["rmse_discrete"], out["rmse_model"]):
        print(f"dt={dt:<5} rmse_discrete={rd:.4e}  rmse_model={rm:.4e}")
    print("log-log slope =", out["empirical_order_slope"])
