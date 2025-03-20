import numpy as np
from scipy.stats import gamma

def validate_observed_stats(R_obs: float, M_obs: float, V_obs: float, allow_zero_v=False):
    """Validate observed statistics with optional V_obs=0 allowance"""
    if not (0 < R_obs < 1):
        raise ValueError(f"R_obs must be 0 < R_obs < 1. Got {R_obs}")
    if M_obs <= 0 or (V_obs <= 0 and not allow_zero_v):
        raise ValueError(f"M_obs/V_obs must be >0. Got {M_obs}, {V_obs}")

def compute_forward_stats(a: float, v: float, t: float) -> tuple[float, float, float]:
    """Core EZ equations with parameter validation"""
    if not (0.5 <= a <= 2):
        raise ValueError(f"a must be 0.5 ≤ a ≤ 2. Got {a}")
    if not (0.5 <= v <= 2):
        raise ValueError(f"v must be 0.5 ≤ v ≤ 2. Got {v}")
    if not (0.1 <= t <= 0.5):
        raise ValueError(f"t must be 0.1 ≤ t ≤ 0.5. Got {t}")

    y = np.exp(-a * v)
    R_pred = 1 / (1 + y)
    M_pred = t + (a / (2*v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2*v**3)) * (1 - 2*a*v*y - y**2) / (1 + y)**2
    return R_pred, M_pred, V_pred

def simulate_observed_stats(a: float, v: float, t: float, N: int) -> tuple[float, float, float]:
    """Robust simulation with numerical safeguards"""
    if N <= 0:
        raise ValueError(f"N must be >0. Got {N}")

    R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
    
    # Simulate with stability
    correct = np.random.binomial(N, R_pred)
    eps = np.finfo(float).eps
    R_obs = np.clip(correct/N, eps, 1-eps)
    
    # Simulate M_obs with variance protection
    M_obs = np.random.normal(M_pred, np.sqrt(max(V_pred/N, eps)))
    
    # Simulate V_obs with different handling for N=1
    if N == 1:
        V_obs = 0.0
    else:
        shape = (N-1)/2
        scale = (2*V_pred)/(N-1)
        V_obs = np.random.gamma(shape, scale)
        V_obs = max(V_obs, 1e-8)  # Variance floor for N > 1
    
    # Validate with context-aware rules
    try:
        validate_observed_stats(R_obs, M_obs, V_obs, allow_zero_v=(N == 1))
    except ValueError as e:
        if "V_obs" in str(e) and N == 1:
            pass  # Allow V_obs=0 for N=1
        else:
            raise
    
    return R_obs, M_obs, V_obs