import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from config import Config
from mpc import QP, condense, reference, shifted_start, step
from solvers import METHODS, osqp_reference, solve


class NumericsTest(unittest.TestCase):
    def test_discrete_reference_and_zero_error(self):
        cfg = Config()
        for kind in ("circle", "s", "eight"):
            states, commands = reference(kind, 300, cfg.dt)
            propagated = [step(x, u, cfg.dt) for x, u in zip(states, commands)]
            np.testing.assert_allclose(propagated, states[1:], atol=1e-12)
            self.assertTrue(np.all(commands >= cfg.lower))
            self.assertTrue(np.all(commands <= cfg.upper))
            qp = condense(states[0], states[:21], commands[:20], cfg)
            np.testing.assert_allclose(qp.q, 0.)
            self.assertGreaterEqual(np.linalg.eigvalsh(qp.H)[0], 2 * cfg.eta - 1e-12)

    def test_condensed_cost_matches_rollout(self):
        cfg = Config()
        states, commands = reference("s", cfg.horizon, cfg.dt)
        error = np.array([0.2, -0.3, 0.1])
        qp = condense(states[0] + error, states, commands, cfg)
        controls = np.random.default_rng(2).uniform(-0.2, 0.2, commands.shape)

        def cost(corrections):
            predicted = error.copy()
            total = 0.
            for x, u, correction in zip(states, commands, corrections):
                epsilon = 1e-5
                nominal = step(x, u, cfg.dt)
                perturbed = step(x + epsilon * predicted, u + epsilon * correction, cfg.dt)
                total += np.dot(cfg.q, predicted**2) + cfg.eta * (correction @ correction)
                predicted = (perturbed - nominal) / epsilon
            return total + cfg.terminal * np.dot(cfg.q, predicted**2)

        difference = cost(controls) - cost(np.zeros_like(controls))
        self.assertAlmostEqual(qp.diagnostics(controls.ravel())[0], difference, places=4)

    def test_solvers_match_reference_with_active_constraints(self):
        qp = QP(np.array([[4., 1.], [1., 3.]]), np.array([-5., 0.]),
                np.array([-1., -1.]), np.array([1., 1.]))
        with threadpool_limits(limits=1, user_api="blas"):
            expected = osqp_reference(qp)
            np.testing.assert_allclose(expected, [1., -1. / 3], atol=1e-8)
            for method in METHODS:
                result = solve(qp, method)
                self.assertLessEqual(result.history[-1][2], 1e-5, method)
                np.testing.assert_allclose(result.z, expected, atol=1e-5)
                np.testing.assert_array_equal(qp.project(result.z), result.z)

    def test_shift_uses_absolute_commands_and_projects(self):
        previous = np.array([[0.5, 0.], [0.7, 1.], [0.3, 0.]])
        commands = np.array([[0.6, 0.7], [0.4, 0.1], [0.2, 0.1]])
        qp = QP(np.eye(6), np.zeros(6), np.full(6, -0.15), np.full(6, 0.15))
        np.testing.assert_allclose(shifted_start(previous, commands, qp),
                                   [0.1, 0.15, -0.1, -0.1, 0.1, -0.1])


if __name__ == "__main__":
    unittest.main()
