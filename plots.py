"""Generate the experiment figures from the rows exported to CSV."""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


COLORS = dict(PGD="#2769a0", APG="#c85200", IPM="#19805e", ADMM="#8c4b99")


def filtered(rows, **filters):
    return [r for r in rows if all(r[key] == value for key, value in filters.items())]


def series(rows, x, y, aggregate=np.median):
    groups = defaultdict(list)
    for row in rows:
        groups[row[x]].append(row[y])
    keys = sorted(groups)
    return keys, [aggregate(groups[key]) for key in keys]


def plot_results(out, fixed, history, tracking, paths, warm):
    def save(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.75))
    for ax, kind in zip(axes, ("circle", "s", "eight")):
        ref = filtered(paths, trajectory=kind, method="PGD")
        ax.plot([r["ref_x"] for r in ref], [r["ref_y"] for r in ref],
                "--", color="black", label="Reference")
        for method, color in COLORS.items():
            rows = filtered(paths, trajectory=kind, method=method)
            ax.plot([r["x"] for r in rows], [r["y"] for r in rows], color=color,
                    label=method, alpha=0.8)
        ax.set(title={"circle": "Circle", "s": "S-shaped", "eight": "Figure-eight"}[kind],
               xlabel="x (m)", ylabel="y (m)", aspect="equal")
    axes[-1].legend(fontsize=7)
    save(fig, "tracking")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    for method, color in COLORS.items():
        rows = filtered(history, experiment="convergence", method=method)
        for ax, x, y, xlabel in zip(axes, ("iteration", "elapsed", "iteration"),
                                   ("gap", "gap", "kkt"),
                                   ("Iteration", "Solver time (ms)", "Iteration")):
            values = [r[x] * (1000 if x == "elapsed" else 1) for r in rows]
            ax.semilogy(values, np.maximum(1e-14, [r[y] for r in rows]),
                        color=color, label=method)
            ax.set(xlabel=xlabel, ylabel="Relative gap" if y == "gap" else "KKT residual")
    axes[0].legend(fontsize=7)
    save(fig, "convergence")

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for rho in (0.1, 1., 10.):
        rows = filtered(history, experiment="rho", value=rho)[1:]
        for ax, field in zip(axes, ("primal", "dual")):
            ax.semilogy([r["iteration"] for r in rows],
                        np.maximum(1e-14, [r[field] for r in rows]), label=f"{rho:g}")
            ax.set(xlabel="Iteration", ylabel=f"ADMM {field} residual")
    axes[0].legend(title="Penalty", fontsize=7)
    save(fig, "admm_residuals")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    scale = filtered(fixed, experiment="scaling")
    n, build = series(scale, "horizon", "build_ms")
    axes[0].loglog(n, build, "k--o", label="QP build", markersize=3)
    for method, color in COLORS.items():
        rows = filtered(scale, method=method)
        for ax, metric in zip(axes, ("solve_ms", "setup_ms", "iterations")):
            n, y = series(rows, "horizon", metric)
            ax.loglog(n, np.maximum(y, 1e-6), "-o", color=color, label=method, markersize=3)
    for ax, label in zip(axes, ("Total solver (ms)", "Solver setup (ms)", "Iterations")):
        ax.set(xlabel="Horizon", ylabel=label)
    axes[0].legend(fontsize=7)
    save(fig, "scaling")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.65))
    conditions = filtered(fixed, experiment="condition")
    for method, color in COLORS.items():
        rows = sorted(filtered(conditions, method=method), key=lambda r: r["kappa"])
        axes[0].loglog([r["kappa"] for r in rows], [r["iterations"] for r in rows],
                       "-o", color=color, label=method, markersize=3)
        censored = [r for r in rows if not r["converged"]]
        axes[0].scatter([r["kappa"] for r in censored], [r["iterations"] for r in censored],
                        color=color, marker="x", s=65)
    for row in filtered(conditions, method="PGD"):
        trace = filtered(history, experiment="condition", method="PGD", value=row["value"])
        axes[1].loglog([r["iteration"] + 1 for r in trace],
                       np.maximum(1e-14, [r["gap"] for r in trace]),
                       label=f'{row["kappa"]:.1f}')
    axes[0].set(xlabel="Hessian condition number", ylabel="Iterations")
    axes[1].set(xlabel="PGD iteration + 1", ylabel="Relative gap")
    axes[0].legend(fontsize=7)
    axes[1].legend(title="Condition number", fontsize=7)
    save(fig, "conditioning")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    for method, color in COLORS.items():
        for ax, experiment, x, y in zip(axes, ("iterations", "time", "time"),
                                       ("cap", "budget_ms", "budget_ms"),
                                       ("position_rmse", "position_rmse", "overrun_rate")):
            rows = filtered(tracking, experiment=experiment, method=method)
            keys, values = series(rows, x, y, np.mean)
            ax.semilogx(keys, values, "-o", color=color, label=method, markersize=3)
            ax.set(xlabel="Iteration cap" if x == "cap" else "Time budget (ms)",
                   ylabel="Overrun fraction" if y == "overrun_rate" else "Position RMSE (m)")
    axes[0].legend(fontsize=7)
    save(fig, "budgets")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    names = ("PGD cold", "PGD warm", "ADMM cold", "ADMM warm")
    groups = [filtered(warm, method=method, warm=flag) for method in ("PGD", "ADMM")
              for flag in (0, 1)]
    for ax, metric, title in zip(axes, ("initial_gap", "iterations", "elapsed_ms"),
                                 ("Initial relative gap", "Iterations", "Cycle time (ms)")):
        ax.boxplot([[r[metric] for r in rows] for rows in groups], showfliers=False)
        ax.set_xticks(range(1, 5), names, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel=title)
    save(fig, "warm_start")
