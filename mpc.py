from dataclasses import dataclass

import numpy as np


def step(x, u, dt):
    theta = x[2]
    v, omega = u
    return x + dt * np.array([v * np.cos(theta), v * np.sin(theta), omega])


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reference(kind, steps, dt):
    """Integrate bounded commands so the reference is Euler feasible."""
    time = dt * np.arange(steps)
    if kind == "circle":
        omega = np.full(steps, 0.45)
    elif kind == "s":
        omega = 0.65 * np.cos(0.65 * time)
    elif kind == "eight":
        # Sample the tangent heading of a Gerono lemniscate, then
        # construct Euler-feasible commands from its sampled chords.
        phase = 0.52 * dt * np.arange(steps + 2)
        points = np.column_stack((np.sin(phase), 0.5 * np.sin(2 * phase)))
        chords = np.diff(points, axis=0)
        headings = np.unwrap(np.arctan2(chords[:, 1], chords[:, 0]))
        commands = np.column_stack((np.linalg.norm(chords[:-1], axis=1) / dt,
                                    np.diff(headings) / dt))
        return np.column_stack((points[:-1], headings)), commands
    else:
        raise ValueError(f"Unknown reference: {kind}")
    commands = np.column_stack((np.full(steps, 0.5), omega))
    states = np.zeros((steps + 1, 3))
    for k, u in enumerate(commands):
        states[k + 1] = step(states[k], u, dt)
    return states, commands


@dataclass
class QP:
    H: np.ndarray
    q: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    def project(self, z):
        return np.clip(z, self.lo, self.hi)

    def diagnostics(self, z):
        hz = self.H @ z
        grad = hz + self.q
        residual = np.linalg.norm(z - self.project(z - grad), ord=np.inf)
        return z @ (0.5 * hz + self.q), residual


def condense(x, states, commands, cfg):
    """Stack predicted errors at stages 1 through N."""
    n = len(commands)
    sx = np.empty((3 * n, 3))
    su = np.zeros((3 * n, 2 * n))
    previous_x = np.eye(3)
    previous_u = np.zeros((3, 2 * n))
    for k, (xr, ur) in enumerate(zip(states, commands)):
        c, s = np.cos(xr[2]), np.sin(xr[2])
        A = np.eye(3)
        A[:2, 2] = cfg.dt * ur[0] * np.array([-s, c])
        B = cfg.dt * np.array([[c, 0], [s, 0], [0, 1]])
        previous_x = A @ previous_x
        prefix = slice(0, 2 * k)
        previous_u[:, prefix] = A @ previous_u[:, prefix]
        previous_u[:, 2*k:2*k+2] = B
        sx[3*k:3*k+3] = previous_x
        su[3*k:3*k+3] = previous_u
    weights = np.tile(cfg.q, n)
    weights[-3:] *= cfg.terminal
    weighted = su.T * weights
    H = 2 * (weighted @ su + cfg.eta * np.eye(2 * n))
    error = x - states[0]
    error[2] = wrap(error[2])
    q = 2 * weighted @ (sx @ error)
    return QP((H + H.T) / 2, q,
              (np.asarray(cfg.lower) - commands).ravel(),
              (np.asarray(cfg.upper) - commands).ravel())


def shifted_start(previous, commands, qp):
    shifted = np.vstack((previous[1:], previous[-1]))
    return qp.project((shifted - commands).ravel())
