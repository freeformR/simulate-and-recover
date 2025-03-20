import unittest
import numpy as np
from src.ez_diffusion import compute_forward_stats, simulate_observed_stats
from src.recovery import recover_parameters
from src.recovery_result import RecoveryResult
from src.simulation import simulate_and_recover

class TestForwardEquations(unittest.TestCase):
    """Validate core EZ equation implementations"""
    def setUp(self):
        # Test parameters cover edge cases and operational ranges
        self.standard_params = [
            (1.0, 1.0, 0.3), (0.5, 0.5, 0.1), (1.5, 1.5, 0.5),
            (1.5, 0.8, 0.2), (0.8, 1.5, 0.4)  # Edge case combinations
        ]

    def test_forward_predictions(self):
        """Validation against original EZ equations"""
        for a, v, t in self.standard_params:
            with self.subTest(a=a, v=v, t=t):
                R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
                
                # Manual derivation verification
                y = np.exp(-a*v)
                expected_R = 1/(1 + y)
                expected_M = t + (a/(2*v)) * ((1-y)/(1+y))
                expected_V = (a/(2*v**3)) * (1 - 2*a*v*y - y**2)/(1+y)**2
                
                self.assertAlmostEqual(R_pred, expected_R, places=5) #Places = 5 suggested by copilot 
                self.assertAlmostEqual(M_pred, expected_M, places=5)
                self.assertAlmostEqual(V_pred, expected_V, places=5)

    def test_parameter_sensitivity(self):
        """Verify expected parameter effects"""
        # Baseline parameters chosen for maximal effect visibility
        a, v, t = 1.0, 1.0, 0.3  
        R0, M0, _ = compute_forward_stats(a, v, t)

        # Drift rate (ν) changes checking for exponential relationships
        _, M_v_up, _ = compute_forward_stats(a, 1.5, t)
        self.assertLess(M_v_up, M0)  

        # Boundary separation (α) changes - checking for multiplicative relationships
        R_a_up, M_a_up, _ = compute_forward_stats(1.5, v, t)
        self.assertLess(R0, R_a_up)  
        self.assertLess(M0, M_a_up)

        # Non-decision time (τ) - checking for additive relationships
        R_t_up, M_t_up, _ = compute_forward_stats(a, v, t+0.1)
        self.assertAlmostEqual(R0, R_t_up, delta=1e-7)  # Strict tolerance suggested by copilot
        self.assertLess(M0, M_t_up)

class TestParameterRecovery(unittest.TestCase):
    """Validate parameter estimation accuracy. Designed with support from copilot"""
    def setUp(self):
        # Shared parameter set across recovery tests
        self.standard_params = [
            (1.0, 1.0, 0.3), (0.5, 0.5, 0.1), (1.5, 1.5, 0.5),
            (1.5, 0.8, 0.2), (0.8, 1.5, 0.4)  
        ]
           
    def test_noise_free_recovery(self):
        """Ideal case validation"""
        # 0.001 delta = 0.1% tolerance for core parameters. - Suggested by copilot
        for a, v, t in self.standard_params:
            with self.subTest(a=a, v=v, t=t):
                R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
                nu_est, a_est, t_est = recover_parameters(R_pred, M_pred, V_pred)
                
                self.assertAlmostEqual(a_est, a, delta=0.001)
                self.assertAlmostEqual(nu_est, v, delta=0.001)
                self.assertAlmostEqual(t_est, t, delta=0.01)  # Looser τ tolerance - suggested by copilot

    def test_recovery_with_noise(self):
        """Real-world simulation"""
        for a, v, t in self.standard_params:
            biases = []
            for _ in range(500):  # Balance between runtime and precision - suggested by copilot
                R_obs, M_obs, V_obs = simulate_observed_stats(a, v, t, 4000)
                nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
                biases.append([a_est-a, nu_est-v, t_est-t])
            
            avg_bias = np.nanmean(biases, axis=0)
            self.assertAlmostEqual(avg_bias[1], 0, delta=0.08)  # ν tolerance
            self.assertAlmostEqual(avg_bias[0], 0, delta=0.05)  # α tolerance

class TestInputValidation(unittest.TestCase):
    """Robustness checks"""
    def test_invalid_parameters(self):
        """Boundary enforcement"""
        invalid_params = [
            (-0.5, 1.0, 0.3), (1.0, -1.0, 0.3),  # Negative values
            (1.0, 1.0, -0.1), (0.0, 1.0, 0.3)    # Boundary violations
        ]
        #Originally drafted by Copilot
        for a, v, t in invalid_params:
            with self.subTest(a=a, v=v, t=t):
                with self.assertRaises(ValueError):
                    compute_forward_stats(a, v, t)

    def test_non_numeric_input(self):
        """Type safety - suggested by copilot"""
        # Copilot suggested type checking
        with self.assertRaises(TypeError):
            compute_forward_stats("1.0", 1.0, 0.3)  # String input
        with self.assertRaises(TypeError):
            compute_forward_stats(1.0, [1.0], 0.3)  # List input

class TestEdgeCases(unittest.TestCase):
    """Boundary condition handling"""
    def test_extreme_performance(self):
        """Numerical stability"""
        #Copilot assisted clipping implementation
        params = recover_parameters(1.0-1e-8, 0.3, 0.1)  # Near-perfect
        self.assertTrue(np.isfinite(params).all())  # Full validation
        
        params = recover_parameters(1e-8, 0.3, 0.1)  # Near-chance
        self.assertTrue(np.isfinite(params).all())

    def test_near_boundary_values(self):
        """degradation checks - Originally drafted by Copilot"""
        for R in [0.501, 0.999, 0.01]:  # Adjacent to decision boundaries
            params = recover_parameters(R, 0.3, 0.1)
            self.assertFalse(np.isnan(params).any())  # Essential check

class TestIdentifiability(unittest.TestCase):
    """Model uniqueness verification (author-driven)"""
    def test_unique_parameter_sets(self):
        """Prevent parameter ambiguity"""
        # Author-defined distinct parameter sets
        # AI-implemented numerical comparison
        stats1 = compute_forward_stats(1.0, 1.0, 0.3)
        stats2 = compute_forward_stats(0.9, 1.1, 0.25)  # Small deltas
        self.assertFalse(np.allclose(stats1, stats2, atol=0.01))  # Author tolerance

class TestCorruption(unittest.TestCase):
    """Data integrity protection - code drafted by Copilot"""
    def test_immutable_properties(self):
        """Enforce"""
        model = RecoveryResult((0.7, 0.4, 0.05))
        with self.assertRaises(AttributeError):  # Copilot suggested pattern
            model.drift_rate = 1.5 

    def test_data_setter_validation(self):
        """State consistency"""
        model = RecoveryResult((0.7, 0.4, 0.05))
        with self.assertRaises(ValueError):  # Copilot suggested implemented
            model.data = (1.1, 0.4, 0.05)  