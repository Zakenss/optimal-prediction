"""
Experiment 3: fit the t-model at several Adam budgets and check
sensitivity to the random memory initialisation.
"""
from __future__ import annotations

import numpy as np

from experiments.common import Context, build_context
from src import metrics, tmodel


def run(ctx: Context | None = None,
        budgets=(5, 50, 200, 1000),
        n_seeds: int = 8,
        var_iters: int = 200) -> dict:
    ctx = ctx or build_context()

    long_run = tmodel.fit_tmodel(
        ctx.proj,
        ctx.x0,
        ctx.dt,
        n_iters=max(budgets),
        lr=1e-2,
        seed=0,
        A_init=ctx.A_init,
        checkpoints=tuple(budgets),
    )
    final_loss = [long_run["ckpt_loss"][n] for n in budgets]
    rpe_budget = [metrics.mean_rpe(long_run["ckpt_pred"][n], ctx.proj) for n in budgets]

    params = (long_run["A"], long_run["M"])
    alphas, betas, Z, hess_evals = tmodel.loss_landscape(
        params, ctx.proj, ctx.x0, ctx.dt, span=2.0, n=31
    )

    seed_preds = []
    for s in range(n_seeds):
        fit_s = tmodel.fit_tmodel(
            ctx.proj, ctx.x0, ctx.dt, n_iters=var_iters,
            lr=1e-2, seed=s, A_init=ctx.A_init,
        )
        seed_preds.append(fit_s["pred"])

    vt = metrics.variance_vt(seed_preds)
    mean_rpe_seeds = float(np.mean([metrics.mean_rpe(p, ctx.proj) for p in seed_preds]))

    return {
        "budgets": list(budgets),
        "final_loss": final_loss,
        "rpe_budget": rpe_budget,
        "losses": long_run["losses"],
        "grad_norms": long_run["grad_norms"],
        "landscape": (alphas, betas, Z, hess_evals),
        "vt": vt,
        "ts": ctx.ts,
        "mean_rpe_seeds": mean_rpe_seeds,
    }


if __name__ == "__main__":
    out = run()
    for n, fl, rp in zip(out["budgets"], out["final_loss"], out["rpe_budget"]):
        print(f"iters={n:<5} final_loss={fl:.4e}  mean_RPE={rp:.4e}")
    print("mean RPE across seeds =", out["mean_rpe_seeds"])
