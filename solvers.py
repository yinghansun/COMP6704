from dataclasses import dataclass
from time import perf_counter

import numpy as np
import osqp
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from scipy.sparse import csc_matrix, eye


METHODS = ("PGD", "APG", "IPM", "ADMM")


@dataclass
class Result:
    z: np.ndarray
    setup: float
    elapsed: float
    history: list

    @property
    def iterations(self):
        return len(self.history) - 1


def solve(qp, method, max_iter=500, tol=1e-5, initial=None, rho=1.,
          time_limit=None, started=None):
    """Return feasible iterates. The optional time limit is checked between updates."""
    if method not in METHODS or rho <= 0:
        raise ValueError("Expected a supported method and positive rho")
    start = perf_counter()
    deadline = (start if started is None else started) + time_limit if time_limit is not None else np.inf
    H, q, lo, hi = qp.H, qp.q, qp.lo, qp.hi
    z = qp.project(np.zeros_like(q) if initial is None else initial)
    # IPM's initialization must be strictly feasible.
    if method == "IPM":
        margin = 0.05 * (hi - lo)
        z = np.clip(z, lo + margin, hi - margin)
        tau = 0.1
    if method in ("PGD", "APG") and perf_counter() < deadline:
        mu, L = eigvalsh(H)[[0, -1]]
        beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
        y = z
    if method == "ADMM" and perf_counter() < deadline:
        factor = cho_factor(H + rho * np.eye(len(q)), check_finite=False)
        dual = np.zeros_like(q)
    setup = perf_counter() - start
    history = []

    def record(primal=np.nan, dual_res=np.nan):
        objective, kkt = qp.diagnostics(z)
        history.append((perf_counter() - start, objective, kkt, primal, dual_res))

    record()
    for _ in range(max_iter):
        if perf_counter() >= deadline or history[-1][2] <= tol:
            break
        if method in ("PGD", "APG"):
            next_z = qp.project(y - (H @ y + q) / L)
            y = next_z + beta * (next_z - z) if method == "APG" else next_z
            z = next_z
            record()
        elif method == "ADMM":
            primal_z = cho_solve(factor, rho * (z - dual) - q, check_finite=False)
            next_z = qp.project(primal_z + dual)
            delta = primal_z - next_z
            dual += delta
            dual_res = rho * np.linalg.norm(next_z - z, ord=np.inf)
            z = next_z
            record(np.linalg.norm(delta, ord=np.inf), dual_res)
        else:
            z, tau = barrier_step(qp, z, tau)
            record()
    return Result(z, setup, perf_counter() - start, history)


def barrier_step(qp, z, tau):
    H, q, lo, hi = qp.H, qp.q, qp.lo, qp.hi
    objective = z @ (0.5 * (H @ z) + q)
    while True:
        left, right = z - lo, hi - z
        grad = H @ z + q + tau * (1 / right - 1 / left)
        diagonal = tau * (left**-2 + right**-2)
        factor = cho_factor(H + np.diag(diagonal), check_finite=False)
        direction = cho_solve(factor, -grad, check_finite=False)
        slope = grad @ direction
        # Follow the central path after approximately centering.
        if -slope / 2 > 0.05 * tau or tau <= 1e-14:
            break
        tau = max(tau * 0.2, 1e-14)
    value = objective - tau * np.sum(np.log(left) + np.log(right))
    alpha = 1.
    for _ in range(60):
        candidate = z + alpha * direction
        if np.all(candidate > lo) and np.all(candidate < hi):
            cv = candidate @ (0.5 * (H @ candidate) + q)
            cv -= tau * np.sum(np.log(candidate - lo) + np.log(hi - candidate))
            if cv <= value + 1e-4 * alpha * slope:
                return candidate, tau
        alpha *= 0.5
    return z, tau


def osqp_reference(qp):
    solver = osqp.OSQP()
    solver.setup(P=csc_matrix(np.triu(qp.H)), q=qp.q, A=eye(len(qp.q), format="csc"),
                 l=qp.lo, u=qp.hi, verbose=False, eps_abs=1e-10, eps_rel=1e-10,
                 max_iter=100000)
    result = solver.solve(raise_error=True)
    z = qp.project(result.x)
    if qp.diagnostics(z)[1] > 1e-7:
        raise RuntimeError("OSQP reference did not meet the KKT verification threshold")
    return z
