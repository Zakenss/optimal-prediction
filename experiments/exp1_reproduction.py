"""
Experiment 1: compare projection, standard DMD, analytic t-model,
and fitted t-model on the observed variables.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from experiments.common import Context, build_context, savefig
from src import metrics, tmodel


def run(ctx: Context | None = None, n_iters: int = 1000, make_plot: bool = True) -> dict:
    ctx = ctx or build_context()
    L_xx = ctx.blocks[0]

    paper = tmodel.analytic_tmodel(L_xx, ctx.K0, ctx.x0, ctx.ts)
    fit = tmodel.fit_tmodel(
        ctx.proj, ctx.x0, ctx.dt, n_iters=n_iters, lr=1e-2, seed=0, A_init=ctx.A_init
    )
    ours = fit["pred"]

    res = {
        "dmd_residual_fro": ctx.dmd_fit["residual"],
        "dmd_cond_A": ctx.dmd_fit["cond"],
        "rpe_dmd": metrics.mean_rpe(ctx.dmd_recon, ctx.proj),
        "rpe_paper": metrics.mean_rpe(paper, ctx.proj),
        "rpe_ours": metrics.mean_rpe(ours, ctx.proj),
        "rmse_dmd": metrics.rmse(ctx.dmd_recon, ctx.proj),
        "rmse_paper": metrics.rmse(paper, ctx.proj),
        "rmse_ours": metrics.rmse(ours, ctx.proj),
        "tmodel_final_loss": fit["final_loss"],
        "A_fit": fit["A"],
        "M_fit": fit["M"],
        "paper": paper,
        "ours": ours,
    }

    if make_plot:
        t = ctx.ts
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        for i, name in enumerate([r"$y_1(t)$", r"$y_2(t)$"]):
            ax[i].plot(t, ctx.proj[:, i], "k-", lw=2.2, label="Projection")
            ax[i].plot(t, ctx.dmd_recon[:, i], "b--", lw=1.4, label="Standard DMD")
            ax[i].plot(t, paper[:, i], "r-.", lw=1.4, label="Analytic t-model")
            ax[i].plot(t, ours[:, i], "g:", lw=2.0, label="Fitted t-model")
            ax[i].set_ylabel(name)
            ax[i].grid(alpha=0.3)
        ax[0].legend(ncol=2, fontsize=8, loc="upper right")
        ax[-1].set_xlabel("t")
        fig.suptitle("Observed trajectories: projection vs DMD vs t-model")
        res["figure"] = savefig(fig, "visual1_trajectories.png")
        plt.close(fig)

    return res


if __name__ == "__main__":
    out = run()
    print({k: v for k, v in out.items() if isinstance(v, (int, float, str))})
