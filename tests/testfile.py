import unittest
import numpy as np
from src.ez_diffusion import compute_forward_stats, simulate_summary_stats
from src.recovery import recover_parameters

class TestEZDiffusion(unittest.TestCase):

    def setUp(self):
        # Define true parameters for testing
        self.a_true = 1.0   # True boundary separation
        self.v_true = 1.0   # True drift rate
        self.t_true = 0.3   # True nondecision time
        self.N = 100        # Sample size for simulation

    def test_compute_forward_stats(self):
        # Test the forward simulation using a known set of intermediate parameters.
        a = 1.0
        v = 1.0
        t = 0.3
        R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
        y = np.exp(-a * v)
        expected_R = 1 / (1 + y)
        expected_M = t + (a / (2 * v)) * ((1 - y) / (1 + y))
        expected_V = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((1 + y)**2))
        self.assertAlmostEqual(R_pred, expected_R, places=5)
        self.assertAlmostEqual(M_pred, expected_M, places=5)
        self.assertAlmostEqual(V_pred, expected_V, places=5)

    def test_simulate_summary_stats(self):
        # For a fixed seed, check that simulate_summary_stats returns values in expected ranges.
        np.random.seed(42)
        a = 1.0
        v = 1.0
        t = 0.3
        N = 100
        R_obs, M_obs, V_obs = simulate_summary_stats(a, v, t, N)
        self.assertTrue(0 <= R_obs <= 1)
        self.assertGreater(V_obs, 0)
        # Also check that the simulated mean is reasonably close to the predicted mean.
        _, M_pred, _ = compute_forward_stats(a, v, t)
        self.assertAlmostEqual(M_obs, M_pred, delta=0.1)

    def test_recovery_no_noise(self):
        # Test that using the noise-free forward predictions returns the original parameters.
        R_pred, M_pred, V_pred = compute_forward_stats(self.a_true, self.v_true, self.t_true)
        nu_est, a_est, t_est = recover_parameters(R_pred, M_pred, V_pred)
        self.assertAlmostEqual(nu_est, self.v_true, places=5)
        self.assertAlmostEqual(a_est, self.a_true, places=5)
        self.assertAlmostEqual(t_est, self.t_true, places=5)

    # Tests using lower-bound parameters: a=0.5, v=0.5, t=0.1
    def test_forward_stats_lower_bounds(self):
        a = 0.5
        v = 0.5
        t = 0.1
        R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
        y = np.exp(-a * v)
        expected_R = 1 / (1 + y)
        expected_M = t + (a / (2 * v)) * ((1 - y) / (1 + y))
        expected_V = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((1 + y)**2))
        self.assertAlmostEqual(R_pred, expected_R, places=5)
        self.assertAlmostEqual(M_pred, expected_M, places=5)
        self.assertAlmostEqual(V_pred, expected_V, places=5)

    def test_recovery_lower_bounds(self):
        a = 0.5
        v = 0.5
        t = 0.1
        R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
        nu_est, a_est, t_est = recover_parameters(R_pred, M_pred, V_pred)
        self.assertAlmostEqual(nu_est, v, places=5)
        self.assertAlmostEqual(a_est, a, places=5)
        self.assertAlmostEqual(t_est, t, places=5)

    # Tests using upper-bound parameters: a=2.0, v=2.0, t=0.5
    def test_forward_stats_upper_bounds(self):
        a = 2.0
        v = 2.0
        t = 0.5
        R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
        y = np.exp(-a * v)
        expected_R = 1 / (1 + y)
        expected_M = t + (a / (2 * v)) * ((1 - y) / (1 + y))
        expected_V = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((1 + y)**2))
        self.assertAlmostEqual(R_pred, expected_R, places=5)
        self.assertAlmostEqual(M_pred, expected_M, places=5)
        self.assertAlmostEqual(V_pred, expected_V, places=5)

    def test_recovery_upper_bounds(self):
        a = 2.0
        v = 2.0
        t = 0.5
        R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
        nu_est, a_est, t_est = recover_parameters(R_pred, M_pred, V_pred)
        self.assertAlmostEqual(nu_est, v, places=5)
        self.assertAlmostEqual(a_est, a, places=5)
        self.assertAlmostEqual(t_est, t, places=5)

    def test_full_simulation(self):
        # Run several iterations of the simulation and recovery,
        # and check that the average bias is near zero.
        iterations = 100
        biases = []
        for _ in range(iterations):
            R_obs, M_obs, V_obs = simulate_summary_stats(self.a_true, self.v_true, self.t_true, self.N)
            nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
            bias = np.array([self.v_true, self.a_true, self.t_true]) - np.array([nu_est, a_est, t_est])
            biases.append(bias)
        biases = np.array(biases)
        avg_bias = np.nanmean(biases, axis=0)
        for b in avg_bias:
            self.assertAlmostEqual(b, 0, delta=0.06)

    def test_parameter_stability(self):
        # Run the recovery multiple times on simulated data with a larger sample size for stability.
        N_large = 4500
        params = []
        for _ in range(3):
            R_obs, M_obs, V_obs = simulate_summary_stats(self.a_true, self.v_true, self.t_true, N_large)
            nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
            params.append((nu_est, a_est, t_est))
        for i in range(1, len(params)):
            for p in range(3):
                self.assertAlmostEqual(params[0][p], params[i][p], delta=0.15)

    def test_edge_case_near_chance(self):
        # Test recovery when R_obs is extremely close to 0.5.
        R_obs = 0.500001
        M_obs = 0.3
        V_obs = 0.034
        nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
        self.assertFalse(np.isnan(nu_est))
        self.assertFalse(np.isnan(a_est))
        self.assertFalse(np.isnan(t_est))

    def test_random_seed_consistency(self):
        # Test that setting a fixed random seed produces consistent simulated summary stats.
        np.random.seed(123)
        result1 = simulate_summary_stats(self.a_true, self.v_true, self.t_true, self.N)
        np.random.seed(123)
        result2 = simulate_summary_stats(self.a_true, self.v_true, self.t_true, self.N)
        for r1, r2 in zip(result1, result2):
            self.assertAlmostEqual(r1, r2, places=5)

    def test_large_iterations_performance(self):
        # Stress test: Run many iterations with a larger sample size to ensure the simulation completes.
        iterations = 50
        for _ in range(iterations):
            simulate_summary_stats(self.a_true, self.v_true, self.t_true, 1000)
        # If no error occurs, the test passes.

if __name__ == '__main__':
    unittest.main()