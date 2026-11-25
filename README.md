# Mobile Robot MPC Solver Benchmark

Compare projected gradient descent (PGD), accelerated projected gradient (APG), a logarithmic barrier interior-point method (IPM), and the alternating direction method of multipliers (ADMM) on the same box-constrained quadratic programs. The benchmark uses linearized model predictive control (MPC) to track circular, S-shaped, and figure-eight trajectories in a nonlinear unicycle simulation. High-accuracy OSQP solutions provide a numerical reference for the optimization experiments.

## Quick start

Use Python 3.12 and run these commands from the folder containing `main.py`. Create and activate a virtual environment:

| Platform | Create | Activate |
| --- | --- | --- |
| Windows PowerShell | `python -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |
| macOS or Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |

Install the dependencies, run the numerical tests, and try a short benchmark:

```sh
python -m pip install "numpy>=1.26" "scipy>=1.12" "osqp>=1.0" "matplotlib>=3.8" "threadpoolctl>=3.2"
python -m unittest discover -s tests -v
python main.py --quick --out quick-results
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in these three commands.

## Run the full benchmark

```sh
python main.py --out results
```

The full run covers trajectory tracking, fixed-QP convergence, horizon scaling, Hessian conditioning, computation budgets, and shifted warm starts. The default seed is 6704. The tracking experiment uses 50 paired initial offsets per trajectory and 250 control steps per episode. The full run can take several minutes. Run `python main.py --help` for options, including the seed, trial count, episode length, and output directory.

Each run creates its output directory. Choose a different `--out` path to keep runs separate.

| Output | What it contains |
| --- | --- |
| `fixed.csv` | Solver times, iteration counts, objective gaps, and final KKT residuals |
| `convergence.csv` | Iteration histories and ADMM split residuals |
| `closed_loop.csv` | Episode-level tracking error, command smoothness, input violations, and latency |
| `paths.csv` | Representative simulated and reference trajectories |
| `warm_cycles.csv` | Cycle-level initial gaps, iteration counts, and timing for warm and cold starts |
| `metadata.json` | Run settings, dependency versions, and platform information |
| `*.pdf` | Figures for the six experiments and the ADMM penalty sweep |

The time budget includes QP construction and solver setup. The solver checks the deadline between updates, so an update can finish after the deadline. Timing measurements vary by machine and system load.

## Code layout

| File | Role |
| --- | --- |
| [`main.py`](main.py) | Command-line entry point, experiments, and closed-loop simulation |
| [`config.py`](config.py) | Model and default benchmark parameters |
| [`mpc.py`](mpc.py) | Unicycle dynamics, reference paths, condensed QP, and warm-start shift |
| [`solvers.py`](solvers.py) | Four solvers, stopping diagnostics, and OSQP reference |
| [`plots.py`](plots.py) | Experiment figure generation |
| [`tests/test_numerics.py`](tests/test_numerics.py) | Tests for reference feasibility, QP construction, solver accuracy, and warm starts |
