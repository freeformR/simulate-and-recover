import numpy as np
from scipy.optimize import minimize
from .ez_diffusion import compute_forward_stats

def recover_parameters(R_obs: float, M_obs: float, V_obs: float) -> tuple[float, float, float]:
    """Hybrid parameter recovery with numerical safeguards"""
    """Original Recovery utilized analytical appraoch only. Copilot recommended the usage of optimization"""
    # Numerical stabilization
    eps = np.finfo(float).eps
    R_obs = np.clip(R_obs, 0.501, 0.999)  # Keep away from boundaries
    V_obs = max(V_obs, eps**2)  # Prevent tiny variances. Suggested change by Copilot 
    
    # Initialize defaults
    nu_est = 1.0
    a_est = 1.0
    t_est = 0.3

    # Analytical initialization with safeguards
    try:
        L = np.log(R_obs/(1 - R_obs + eps))
        sign = np.sign(R_obs - 0.5)
        
        # Stabilized calculations
        numerator = L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)
        denominator = max(V_obs, eps)
        nu_est = sign * (abs(numerator)/denominator)**0.25
        
        # Clip estimates to valid ranges. The need for clipping became evident, Copilot helped draft these changes.
        nu_est = np.clip(nu_est, 0.5 + eps, 2 - eps)
        a_est = np.clip(L/(nu_est + eps), 0.5 + eps, 2 - eps)
        
        # Non-decision time estimate
        exp_term = np.exp(-a_est * nu_est)
        t_est = M_obs - (a_est/(2 * (nu_est + eps))) * ((1 - exp_term)/(1 + exp_term))
        t_est = np.clip(t_est, 0.1 + eps, 0.5 - eps)
        
    except Exception as e:
        # Fallback to defaults if analytical initialization fails. This fallback change was suggested by copilot 
        pass

    # Optimization with proper parameter ordering - Drafted by Copilot
    def loss(params):
        try:
            a, v, t = params
            R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
            return ((R_obs - R_pred)**2 + 
                    (M_obs - M_pred)**2 + 
                    (V_obs - V_pred)**2)
        except:
            return float('inf')

    # Parameter order: [a, v, t]
    result = minimize(
        loss,
        x0=[a_est, nu_est, t_est],
        bounds=[(0.5, 2), (0.5, 2), (0.1, 0.5)],
        method='L-BFGS-B'
    )
    
    # Return (ν, a, τ) - match optimization parameter order
    return result.x[1], result.x[0], result.x[2]