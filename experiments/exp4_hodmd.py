"""
Experiment 4: fit HODMD (delay embedding) on the observed trajectory
and compare against standard DMD and the fitted t-model.
"""
from __future__ import annotations

from experiments.common import Context, build_context
from src import dmd, metrics, tmodel


def run(ctx: Context | None = None, depths=(1, 2, 3, 4)) -> dict:
    ctx = ctx or build_context()

    rows = []
    recon_d2 = None
    for d in depths:
        fit = dmd.fit_hodmd(ctx.proj, d=d, rank=None)
        rpe = metrics.mean_rpe(fit["recon"], ctx.proj)
        rows.append({
            "d": d,
            "rpe": rpe,
            "rmse": metrics.rmse(fit["recon"], ctx.proj),
            "residual": fit["residual"],
            "cond": fit["cond"],
        })
        if d == 2:
            recon_d2 = fit["recon"]

    tfit = tmodel.fit_tmodel(
        ctx.proj, ctx.x0, ctx.dt, n_iters=1000, lr=1e-2, seed=0, A_init=ctx.A_init
    )
    rpe_tmodel = metrics.mean_rpe(tfit["pred"], ctx.proj)
    rpe_dmd = metrics.mean_rpe(ctx.dmd_recon, ctx.proj)
    best = min(rows, key=lambda r: r["rpe"])

    return {
        "rows": rows,
        "recon_d2": recon_d2,
        "rpe_tmodel": rpe_tmodel,
        "rpe_dmd": rpe_dmd,
        "best_hodmd": best,
        "ts": ctx.ts,
        "proj": ctx.proj,
    }


if __name__ == "__main__":
    out = run()
    for r in out["rows"]:
        print(f"d={r['d']}  RPE={r['rpe']:.4e}  resid={r['residual']:.3e}  cond={r['cond']:.3e}")
    print("DMD RPE    =", out["rpe_dmd"])
    print("t-model RPE =", out["rpe_tmodel"])
