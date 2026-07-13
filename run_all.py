"""
Run all preliminary experiments and write figures plus metrics.json
into the local results/ folder.

Usage (from the project root):
  python run_all.py
"""
from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from experiments import exp1_reproduction, exp2_dtau, exp3_optdepth, exp4_hodmd
from experiments.common import RESULTS_DIR, build_context, savefig
from src import dmd, metrics, system, tmodel


def spectral_comparison(ctx, A_fit, M_fit):
    """Compare continuous eigenvalues of DMD and the fitted t-model to the true ones."""
    t_mid = float(ctx.ts[len(ctx.ts) // 2])

    lam_true, Vr_true = system.true_koopman_eigs(ctx.L)
    lam_dmd, V_dmd = dmd.continuous_eigs(ctx.dmd_fit["A"], ctx.dt)

    G = tmodel.effective_generator(A_fit, M_fit, t_mid)
    mu_tm, V_tm = np.linalg.eig(G)
    order = np.argsort(-mu_tm.imag)
    lam_tm = mu_tm[order]
    V_tm = V_tm[:, order]

    md_dmd, mt_dmd = dmd.match_modes(lam_dmd, V_dmd, lam_true, Vr_true)
    md_tm, mt_tm = dmd.match_modes(lam_tm, V_tm, lam_true, Vr_true)

    def err(model, true):
        return {
            "freq_err": float(np.mean(np.abs(model.imag - true.imag))),
            "damp_err": float(np.mean(np.abs(model.real - true.real))),
        }

    return {
        "t_mid": t_mid,
        "lam_true": lam_true,
        "lam_dmd": lam_dmd,
        "lam_tm": lam_tm,
        "dmd_vs_true": err(md_dmd, mt_dmd),
        "tm_vs_true": err(md_tm, mt_tm),
    }


def wallclock_pareto(ctx, budgets=(5, 50, 200, 1000, 2000)):
    """Record wall-clock time and mean RPE for several Adam budgets."""
    pts = []
    for n in budgets:
        t0 = time.perf_counter()
        fit = tmodel.fit_tmodel(
            ctx.proj, ctx.x0, ctx.dt, n_iters=n, lr=1e-2, seed=0, A_init=ctx.A_init
        )
        wall = time.perf_counter() - t0
        pts.append((n, wall, metrics.mean_rpe(fit["pred"], ctx.proj)))
    return pts


def plot_sensitivity(e2, e3):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    a.bar([str(d) for d in e2["dtaus"]], e2["rmse_discrete"], color="steelblue")
    a.set_xlabel(r"$\Delta\tau$")
    a.set_ylabel("RMSE (discrete t-model vs projection)")
    a.set_yscale("log")
    a.set_title(r"RMSE vs $\Delta\tau$")
    a.grid(alpha=0.3, axis="y")

    b.bar([str(n) for n in e3["budgets"]], e3["final_loss"], color="indianred")
    b.set_xlabel("Adam iterations")
    b.set_ylabel("final training loss")
    b.set_yscale("log")
    b.set_title("Loss vs optimisation depth")
    b.grid(alpha=0.3, axis="y")
    fig.suptitle("Sensitivity to time step and optimisation depth")
    return savefig(fig, "visual2_sensitivity.png")


def plot_eigenvalues(spec):
    fig, ax = plt.subplots(figsize=(6.6, 6))
    lt = spec["lam_true"]
    ld = spec["lam_dmd"]
    lm = spec["lam_tm"]

    ax.scatter(lt.real, lt.imag, marker="*", s=220, c="black", label="True system", zorder=3)
    ax.scatter(
        ld.real, ld.imag, marker="o", s=80, facecolors="none",
        edgecolors="blue", label="Standard DMD", zorder=3,
    )
    ax.scatter(lm.real, lm.imag, marker="^", s=90, c="green", label="Fitted t-model", zorder=3)

    for d_val, t_val in zip(ld, lm):
        ax.annotate(
            "",
            xy=(t_val.real, t_val.imag),
            xytext=(d_val.real, d_val.imag),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
        )

    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$\Re(\lambda)$ (damping)")
    ax.set_ylabel(r"$\Im(\lambda)$ (frequency)")
    ax.set_title("Eigenvalues in the complex plane")
    ax.legend()
    ax.grid(alpha=0.3)
    return savefig(fig, "visual3_eigenvalues.png")


def plot_convergence(e3):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    it = np.arange(1, len(e3["losses"]) + 1)

    a.semilogy(it, e3["losses"], "r-", label="loss")
    a.set_xlabel("Adam iteration")
    a.set_ylabel("loss", color="r")
    a.axvline(5, color="gray", ls="--", alpha=0.7, label="5 iterations")
    a2 = a.twinx()
    a2.semilogy(it, e3["grad_norms"], "b-", alpha=0.5)
    a2.set_ylabel("grad norm", color="b")
    a.set_title("Convergence curve")
    a.legend(loc="upper right", fontsize=8)

    alphas, betas, Z, _ = e3["landscape"]
    cs = b.contourf(alphas, betas, np.log10(Z.T + 1e-12), levels=30, cmap="viridis")
    b.plot(0, 0, "r*", ms=14, label="fitted point")
    b.set_xlabel("Hessian direction 1")
    b.set_ylabel("Hessian direction 2")
    b.set_title(r"Loss landscape ($\log_{10}$)")
    b.legend()
    fig.colorbar(cs, ax=b)
    return savefig(fig, "exp3_convergence_landscape.png")


def plot_variance(e3):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(e3["ts"], e3["vt"] + 1e-18, "purple")
    ax.set_xlabel("t")
    ax.set_ylabel("v(t)")
    ax.set_title("Variance across memory initialisations")
    ax.grid(alpha=0.3)
    return savefig(fig, "exp3_variance_vt.png")


def plot_pareto(pareto):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    walls = [p[1] for p in pareto]
    rpes = [p[2] for p in pareto]
    ns = [p[0] for p in pareto]
    ax.plot(walls, rpes, "o-", color="darkorange")
    for n, w, r in zip(ns, walls, rpes):
        ax.annotate(f"{n}", (w, r), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("mean RPE")
    ax.set_yscale("log")
    ax.set_title("Wall-clock time vs accuracy")
    ax.grid(alpha=0.3)
    return savefig(fig, "exp_pareto.png")


def write_summary(ctx, e1, e2, e3, e4, spec, pareto):
    """Write a short plain-text summary of the run."""
    lines = []
    a = lines.append
    a("Preliminary experiment summary")
    a("==============================")
    a(f"dt = {ctx.dt}, horizon = {ctx.t_end}")
    a("")
    a("Experiment 1 (reproduction)")
    a(f"  DMD residual Frobenius = {e1['dmd_residual_fro']:.4e}")
    a(f"  DMD cond(A)            = {e1['dmd_cond_A']:.4e}")
    a(f"  mean RPE DMD           = {e1['rpe_dmd']:.4e}")
    a(f"  mean RPE analytic t    = {e1['rpe_paper']:.4e}")
    a(f"  mean RPE fitted t      = {e1['rpe_ours']:.4e}")
    a("")
    a("Experiment 2 (Delta-tau)")
    for dt, rd, rm in zip(e2["dtaus"], e2["rmse_discrete"], e2["rmse_model"]):
        a(f"  dt={dt:<4}  RMSE discrete={rd:.4e}  RMSE analytic={rm:.4e}")
    a("")
    a("Experiment 3 (optimisation depth)")
    for n, fl, rp in zip(e3["budgets"], e3["final_loss"], e3["rpe_budget"]):
        a(f"  iters={n:<5}  final_loss={fl:.4e}  mean_RPE={rp:.4e}")
    a("")
    a("Experiment 4 (HODMD)")
    for r in e4["rows"]:
        a(f"  d={r['d']}  RPE={r['rpe']:.4e}  residual={r['residual']:.3e}  cond={r['cond']:.3e}")
    a(f"  best delay d = {e4['best_hodmd']['d']}")
    a("")
    a("Spectra (mode-matched)")
    a(f"  DMD freq err / damp err     = {spec['dmd_vs_true']['freq_err']:.4e} / {spec['dmd_vs_true']['damp_err']:.4e}")
    a(f"  t-model freq err / damp err = {spec['tm_vs_true']['freq_err']:.4e} / {spec['tm_vs_true']['damp_err']:.4e}")
    a("")
    a("Wall-clock vs accuracy (iters, seconds, mean RPE)")
    for n, w, r in pareto:
        a(f"  {n:<5}  {w:.3f}s  {r:.4e}")

    path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    t_start = time.perf_counter()
    ctx = build_context(dt=0.1, t_end=12.0)

    print("Running experiment 1: reproduction")
    e1 = exp1_reproduction.run(ctx, n_iters=1000, make_plot=True)

    print("Running experiment 2: Delta-tau sweep")
    e2 = exp2_dtau.run()

    print("Running experiment 3: optimisation depth")
    e3 = exp3_optdepth.run(ctx)

    print("Running experiment 4: HODMD baseline")
    e4 = exp4_hodmd.run(ctx)

    print("Spectral comparison and wall-clock sweep")
    spec = spectral_comparison(ctx, e1["A_fit"], e1["M_fit"])
    pareto = wallclock_pareto(ctx)

    plot_sensitivity(e2, e3)
    plot_eigenvalues(spec)
    plot_convergence(e3)
    plot_variance(e3)
    plot_pareto(pareto)

    metrics_out = {
        "exp1": {
            k: (v if isinstance(v, (int, float, str)) else None) for k, v in e1.items()
        },
        "exp2": e2,
        "exp3": {
            "budgets": e3["budgets"],
            "final_loss": e3["final_loss"],
            "rpe_budget": e3["rpe_budget"],
            "mean_rpe_seeds": e3["mean_rpe_seeds"],
        },
        "exp4": {
            "rows": e4["rows"],
            "rpe_dmd": e4["rpe_dmd"],
            "rpe_tmodel": e4["rpe_tmodel"],
            "best": e4["best_hodmd"],
        },
        "spectral": {
            "dmd_vs_true": spec["dmd_vs_true"],
            "tm_vs_true": spec["tm_vs_true"],
        },
        "pareto": pareto,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    summary_path = write_summary(ctx, e1, e2, e3, e4, spec, pareto)

    print(f"Finished in {time.perf_counter() - t_start:.1f}s")
    print(f"Summary: {summary_path}")
    print(f"Figures and metrics are in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
