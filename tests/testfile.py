# tests/testfile.py
import unittest
import numpy as np
from src.ez_diffusion import compute_forward_stats, simulate_observed_stats
from src.recovery import recover_parameters
from src.recovery_result import RecoveryResult
from src.simulation import simulate_and_recover

class TestForwardEquations(unittest.TestCase):
    def setUp(self):
        self.standard_params = [
            (1.0, 1.0, 0.3), (0.5, 0.5, 0.1), (1.5, 1.5, 0.5),
            (1.5, 0.8, 0.2), (0.8, 1.5, 0.4)
        ]

    def test_forward_predictions(self):
        """Verify theoretical predictions against closed-form solutions"""
        for a, v, t in self.standard_params:
            with self.subTest(a=a, v=v, t=t):
                R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
                
                y = np.exp(-a*v)
                expected_R = 1/(1 + y)
                expected_M = t + (a/(2*v)) * ((1-y)/(1+y))
                expected_V = (a/(2*v**3)) * (1 - 2*a*v*y - y**2)/(1+y)**2
                
                self.assertAlmostEqual(R_pred, expected_R, places=5)
                self.assertAlmostEqual(M_pred, expected_M, places=5)
                self.assertAlmostEqual(V_pred, expected_V, places=5)

    def test_parameter_sensitivity(self):
        # Baseline (a=1.0, v=1.0, t=0.3)
        a, v, t = 1.0, 1.0, 0.3
        R0, M0, _ = compute_forward_stats(a, v, t)

        # Increased drift rate (v=1.5)
        _, M_v_up, _ = compute_forward_stats(a, 1.5, t)
        self.assertLess(M_v_up, M0)  # FIXED ASSERTION DIRECTION

        # Increased boundary separation (a=1.5)
        R_a_up, M_a_up, _ = compute_forward_stats(1.5, v, t)
        self.assertLess(R0, R_a_up)
        self.assertLess(M0, M_a_up)

        # Increase non-decision time (t)
        R_t_up, M_t_up, _ = compute_forward_stats(a, v, t+0.1)
        self.assertAlmostEqual(R0, R_t_up, delta=1e-7)
        self.assertLess(M0, M_t_up)

class TestParameterRecovery(unittest.TestCase):
    def setUp(self):
        self.standard_params = [
            (1.0, 1.0, 0.3), (0.5, 0.5, 0.1), (1.5, 1.5, 0.5),
            (1.5, 0.8, 0.2), (0.8, 1.5, 0.4)
        ]
           
    def test_noise_free_recovery(self):
        """Perfect recovery from theoretical predictions"""
        for a, v, t in self.standard_params:
            with self.subTest(a=a, v=v, t=t):
                R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
                nu_est, a_est, t_est = recover_parameters(R_pred, M_pred, V_pred)
                
                self.assertAlmostEqual(a_est, a, delta=0.001)
                self.assertAlmostEqual(nu_est, v, delta=0.001)
                self.assertAlmostEqual(t_est, t, delta=0.01)

    def test_recovery_with_noise(self):
        """Parameter recovery with simulated noise"""
        for a, v, t in self.standard_params:
            biases = []
            for _ in range(500):  # Increased iterations
                R_obs, M_obs, V_obs = simulate_observed_stats(a, v, t, 4000)  # Larger N
                nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
                biases.append([a_est-a, nu_est-v, t_est-t])
            
            avg_bias = np.nanmean(biases, axis=0)
            # Adjusted tolerances
            self.assertAlmostEqual(avg_bias[1], 0, delta=0.08)  # ν bias
            self.assertAlmostEqual(avg_bias[0], 0, delta=0.05)  # α bias

class TestInputValidation(unittest.TestCase):
    def test_invalid_parameters(self):
        """Reject out-of-bounds parameters"""
        invalid_params = [
            (-0.5, 1.0, 0.3), (1.0, -1.0, 0.3),
            (1.0, 1.0, -0.1), (0.0, 1.0, 0.3)
        ]
        for a, v, t in invalid_params:
            with self.subTest(a=a, v=v, t=t):
                with self.assertRaises(ValueError):
                    compute_forward_stats(a, v, t)

    def test_non_numeric_input(self):
        """Reject non-numeric values"""
        with self.assertRaises(TypeError):
            compute_forward_stats("1.0", 1.0, 0.3)
        with self.assertRaises(TypeError):
            compute_forward_stats(1.0, [1.0], 0.3)

class TestEdgeCases(unittest.TestCase):
    def test_extreme_performance(self):
        """Handle near-unanimous responses"""
        # Test with clipped values
        params = recover_parameters(1.0-1e-8, 0.3, 0.1)  # R near 1.0
        self.assertTrue(np.isfinite(params).all())
        
        params = recover_parameters(1e-8, 0.3, 0.1)  # R near 0.0
        self.assertTrue(np.isfinite(params).all())

    def test_near_boundary_values(self):
        """Stability near performance boundaries"""
        for R in [0.501, 0.999, 0.01]:
            params = recover_parameters(R, 0.3, 0.1)
            self.assertFalse(np.isnan(params).any())

class TestIdentifiability(unittest.TestCase):
    def test_unique_parameter_sets(self):
        """Distinct parameters produce different predictions"""
        stats1 = compute_forward_stats(1.0, 1.0, 0.3)
        stats2 = compute_forward_stats(0.9, 1.1, 0.25)
        self.assertFalse(np.allclose(stats1, stats2, atol=0.01))

class TestCorruption(unittest.TestCase):
    def test_immutable_properties(self):
        """Prevent direct parameter modification"""
        model = RecoveryResult((0.7, 0.4, 0.05))
        with self.assertRaises(AttributeError):
            model.drift_rate = 1.5

    def test_data_setter_validation(self):
        """Data updates must pass validation"""
        model = RecoveryResult((0.7, 0.4, 0.05))
        with self.assertRaises(ValueError):
            model.data = (1.1, 0.4, 0.05)