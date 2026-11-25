"""Run the six experiment families and save their measured data."""
import argparse
import csv
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import numpy as np
import osqp
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from config import Config
from mpc import condense, reference, shifted_start, step, wrap
from plots import plot_results
from solvers import METHODS, osqp_reference, solve


TRAJECTORIES = ("circle", "s", "eight")
HISTORY_FIELDS = ("elapsed", "objective", "kkt", "primal", "dual")


def gap(objective, reference_objective):
    return abs(objective - reference_objective) / max(1., abs(reference_objective))


def simulate(cfg, kind, offset, method, steps, cap, warm=False, ms=None,
             compare_initial=False):
    xr, ur = reference(kind, steps + cfg.horizon, cfg.dt)
    if not (np.all(ur >= cfg.lower) and np.all(ur <= cfg.upper)):
        raise ValueError("Reference command outside input bounds")
    x = xr[0] + offset
    path, applied, cycles = [x], [], []
    previous = None
    for k in range(steps):
        begin = perf_counter()
        window = ur[k:k + cfg.horizon]
        qp = condense(x, xr[k:k + cfg.horizon + 1], window, cfg)
        z0 = shifted_start(previous, window, qp) if previous is not None and warm else None
        built = perf_counter()
        result = solve(qp, method, cap, cfg.tolerance, z0,
                       time_limit=None if ms is None else ms / 1000, started=begin)
        previous = window + result.z.reshape(-1, 2)
        command = previous[0]
        elapsed = perf_counter() - begin
        initial_gap = np.nan
        if compare_initial:
            target = qp.diagnostics(osqp_reference(qp))[0]
            initial_gap = gap(result.history[0][1], target)
        cycles.append(dict(cycle=k, elapsed_ms=1000 * elapsed,
                           build_ms=1000 * (built - begin), iterations=result.iterations,
                           kkt=result.history[-1][2], initial_gap=initial_gap,
                           overrun=int(ms is not None and elapsed * 1000 > ms)))
        x = step(x, command, cfg.dt)
        applied.append(command)
        path.append(x)
    path, applied = np.array(path), np.array(applied)
    position = np.linalg.norm((path - xr[:steps+1])[:, :2], axis=1)
    heading = wrap(path[:, 2] - xr[:steps+1, 2])
    slew = np.diff(applied, axis=0)
    metrics = dict(position_rmse=np.sqrt(np.mean(position**2)),
                   heading_rmse=np.sqrt(np.mean(heading**2)),
                   max_error=np.max(position), final_error=position[-1],
                   linear_slew_rms=np.sqrt(np.mean(slew[:, 0]**2)),
                   angular_slew_rms=np.sqrt(np.mean(slew[:, 1]**2)),
                   mean_ms=np.mean([row["elapsed_ms"] for row in cycles]),
                   p95_ms=np.percentile([row["elapsed_ms"] for row in cycles], 95),
                   violation=max(0., np.max(np.asarray(cfg.lower) - applied),
                                 np.max(applied - np.asarray(cfg.upper))),
                   mean_iterations=np.mean([row["iterations"] for row in cycles]),
                   overrun_rate=np.mean([row["overrun"] for row in cycles]))
    return metrics, path, xr, cycles


def fixed(cfg, cap, offsets, quick):
    def case(kind, label, trial, local, offset, trace=False):
        xr, ur = reference("s", local.horizon + 17, local.dt)
        x = xr[17] + offset
        begin = perf_counter()
        qp = condense(x, xr[17:], ur[17:], local)
        build_ms = 1000 * (perf_counter() - begin)
        spectrum = scipy.linalg.eigvalsh(qp.H)
        kappa = spectrum[-1] / spectrum[0]
        target = qp.diagnostics(osqp_reference(qp))[0]
        for method in METHODS:
            result = solve(qp, method, cap, cfg.tolerance)
            end = result.history[-1]
            experiments.append(dict(experiment=kind, value=label, trial=trial, method=method,
                                    horizon=local.horizon, variables=2 * local.horizon,
                                    inequalities=4 * local.horizon, kappa=kappa,
                                    build_ms=build_ms, setup_ms=1000 * result.setup,
                                    solve_ms=1000 * result.elapsed,
                                    iteration_ms=1000 * (result.elapsed - result.setup),
                                    iterations=result.iterations, gap=gap(end[1], target), kkt=end[2],
                                    converged=int(end[2] <= cfg.tolerance)))
            if trace:
                for iteration, row in enumerate(result.history):
                    history.append(dict(experiment=kind, value=label, method=method,
                                        iteration=iteration, gap=gap(row[1], target),
                                        **dict(zip(HISTORY_FIELDS, row))))

    experiments, history = [], []
    offset = np.array([0.28, -0.36, 0.32])
    case("convergence", 0, 0, cfg, offset, trace=True)
    xr, ur = reference("s", cfg.horizon + 17, cfg.dt)
    qp = condense(xr[17] + offset, xr[17:], ur[17:], cfg)
    target = qp.diagnostics(osqp_reference(qp))[0]
    for rho in (0.1, 1., 10.):
        result = solve(qp, "ADMM", cap, cfg.tolerance, rho=rho)
        for iteration, row in enumerate(result.history):
            history.append(dict(experiment="rho", value=rho, method="ADMM",
                                iteration=iteration, gap=gap(row[1], target),
                                **dict(zip(HISTORY_FIELDS, row))))
    for n in ((5, 20, 50) if quick else (5, 10, 20, 50, 100, 200)):
        for trial, sample in enumerate(offsets[:(1 if quick else 5)]):
            case("scaling", n, trial, replace(cfg, horizon=n), sample)
    for eta in (0.001, 0.01, 0.1, 1, 10):
        case("condition", eta, 0, replace(cfg, eta=eta), offset, trace=True)
    return experiments, history


def closed_loop(cfg, offsets, steps, quick):
    summary, paths, warm_cycles = [], [], []

    def run(experiment, kind, trial, method, cap, warm=False, ms=None):
        metrics, path, xr, cycles = simulate(
            cfg, kind, offsets[trial], method, steps, cap, warm, ms,
            compare_initial=experiment == "warm")
        label = dict(experiment=experiment, trajectory=kind, method=method, trial=trial,
                     cap=cap, warm=int(warm), budget_ms=ms)
        summary.append(dict(**label, **metrics))
        if experiment == "tracking" and trial == 0:
            for k, (x, ref) in enumerate(zip(path, xr)):
                paths.append(dict(**label, time=k * cfg.dt, x=x[0], y=x[1],
                                  ref_x=ref[0], ref_y=ref[1]))
        if experiment == "warm":
            warm_cycles.extend(dict(**label, **cycle) for cycle in cycles)

    for kind in TRAJECTORIES:
        for trial in range(len(offsets)):
            for method in METHODS:
                run("tracking", kind, trial, method, 100)
    print("Tracking complete; running budget and warm-start sweeps.", flush=True)
    # Use paired initial offsets across all methods and configurations.
    for kind in TRAJECTORIES:
        for trial in range(min(len(offsets), 1 if quick else 3)):
            for method in METHODS:
                for cap in ((1, 10) if quick else (1, 3, 5, 10, 20, 50, 100)):
                    run("iterations", kind, trial, method, cap)
                for ms in ((1,) if quick else (1, 5, 10)):
                    run("time", kind, trial, method, 2000, ms=ms)
            for method in ("PGD", "ADMM"):
                for warm in (False, True):
                    run("warm", kind, trial, method, 100, warm)
    return summary, paths, warm_cycles


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=6704)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.steps < 2:
        parser.error("expected at least one trial and two steps")
    trials = 2 if args.quick else args.trials
    steps = 24 if args.quick else args.steps
    offsets = np.random.default_rng(args.seed).uniform(-1, 1, (trials, 3))
    offsets *= [0.35, 0.35, 0.4]
    args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1, user_api="blas"):
        print("Running fixed-QP convergence, scaling and conditioning.", flush=True)
        fixed_rows, history = fixed(Config(), 300 if args.quick else 2000, offsets, args.quick)
        print("Fixed-QP experiments complete; running closed-loop tracking.", flush=True)
        tracking, paths, warm = closed_loop(Config(), offsets, steps, args.quick)
        for filename, rows in (("fixed", fixed_rows), ("convergence", history),
                               ("closed_loop", tracking), ("paths", paths),
                               ("warm_cycles", warm)):
            write_csv(args.out / (filename + ".csv"), rows)
        metadata = dict(seed=args.seed, trials=trials, steps=steps, quick=args.quick,
                        python=platform.python_version(), platform=platform.platform(),
                        numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__,
                        matplotlib=matplotlib.__version__,
                        blas=[{k: v for k, v in entry.items() if k != "filepath"}
                              for entry in threadpool_info()])
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        plot_results(args.out, fixed_rows, history, tracking, paths, warm)
    print(f"Saved data and figures in {args.out}", flush=True)


if __name__ == "__main__":
    main()
