from .ez_diffusion import simulate_observed_stats
from .recovery_result import RecoveryResult
import numpy as np

def simulate_and_recover(N: int, iterations: int = 1000) -> tuple:
    """
    Simulate and recover parameters, returning aggregate statistics.
    Returns: (avg_bias, avg_mse, valid_count, invalid_count)
    """
    biases = []
    mses = []
    invalid = 0
    
    for _ in range(iterations):
        try:
            # Random true parameters
            a_true = np.random.uniform(0.5, 2)
            v_true = np.random.uniform(0.5, 2)
            t_true = np.random.uniform(0.1, 0.5)
            
            # Simulate observed data
            R_obs, M_obs, V_obs = simulate_observed_stats(a_true, v_true, t_true, N)
            
            # Recover parameters using stateful class
            result = RecoveryResult((R_obs, M_obs, V_obs))
            nu_est = result.drift_rate
            a_est = result.boundary_separation
            t_est = result.non_decision_time
            
            # Calculate metrics
            bias = np.array([v_true - nu_est, a_true - a_est, t_true - t_est])
            biases.append(bias)
            mses.append(bias**2)
            
        except:
            invalid += 1
            biases.append(np.full(3, np.nan))
            mses.append(np.full(3, np.nan))
    
    return (
        np.nanmean(biases, axis=0),
        np.nanmean(mses, axis=0),
        iterations - invalid,
        invalid
    )
