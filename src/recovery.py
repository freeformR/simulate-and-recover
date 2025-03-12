import numpy as np

def recover_parameters(R_obs, M_obs, V_obs):
    """
    Recover EZ diffusion model parameters from observed summary statistics.

    Parameters:
        R_obs (float): Observed accuracy rate (must be between 0 and 1).
        M_obs (float): Observed mean response time.
        V_obs (float): Observed variance of response times.

    Returns:
        nu_est (float): Estimated drift rate.
        a_est (float): Estimated boundary separation.
        t_est (float): Estimated nondecision time.
    """
    # Input validation
    if not (0 < R_obs < 1):
        raise ValueError("R_obs must be between 0 and 1.")
    if M_obs <= 0 or V_obs <= 0:
        raise ValueError("M_obs and V_obs must be positive.")

    # Clip R_obs to avoid boundary values
    epsilon = 1e-3  # Increase the lower bound from 1e-5 to 1e-4
    R_obs = np.clip(R_obs, epsilon, 1 - epsilon)


    # Adjust R_obs if too close to 0.5
    threshold = 5e-3  # Increase from 1e-3 to 5e-3
    if np.abs(R_obs - 0.5) < threshold:
        R_obs = 0.5 + threshold if R_obs >= 0.5 else 0.5 - threshold


    # Compute L = ln(R_obs / (1 - R_obs))
    L = np.log(R_obs / (1 - R_obs))
    
    # Clamp V_obs to avoid small values causing instability
    V_obs = max(V_obs, 1e-5)

    # Inverse equation for drift rate (nu)
    sign_factor = np.sign(R_obs - 0.5)
    inside = L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)
    nu_est = sign_factor * (np.abs(inside) / V_obs)**0.25

    # Compute boundary separation (a)
    a_est = L / nu_est

    # Compute nondecision time (t)
    t_est = M_obs - (a_est / (2 * nu_est)) * ((1 - np.exp(-a_est * nu_est)) / (1 + np.exp(-a_est * nu_est)))

    return nu_est, a_est, t_est

if __name__ == "__main__":
    # Example usage
    R_obs = 0.73
    M_obs = 0.56
    V_obs = 0.034
    try:
        nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
        print(f"Recovered parameters: nu_est={nu_est:.3f}, a_est={a_est:.3f}, t_est={t_est:.3f}")
    except ValueError as e:
        print(f"Error: {e}")
