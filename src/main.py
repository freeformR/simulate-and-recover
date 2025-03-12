import numpy as np
from ez_diffusion import simulate_summary_stats
from recovery import recover_parameters

def simulate_and_recover(a_true, v_true, t_true, N, iterations=1000):
    """
    Perform the simulate-and-recover loop for a given sample size N for a fixed number
    of iterations. All iterations are recorded; if an iteration yields invalid recovered
    parameters, it is recorded as NaN.
    """
    if a_true <= 0 or v_true <= 0 or t_true <= 0:
        raise ValueError("True parameters 'a', 'v', and 't' must be positive.")
    if N <= 0 or iterations <= 0:
        raise ValueError("Sample size 'N' and 'iterations' must be positive integers.")

    biases = []
    squared_errors = []
    invalid_count = 0
    
    for _ in range(iterations):
        R_obs, M_obs, V_obs = simulate_summary_stats(a_true, v_true, t_true, N)
        nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
        
        if np.isnan(nu_est) or np.isnan(a_est) or np.isnan(t_est):
            biases.append(np.array([np.nan, np.nan, np.nan]))
            squared_errors.append(np.array([np.nan, np.nan, np.nan]))
            invalid_count += 1
        else:
            bias = np.array([v_true, a_true, t_true]) - np.array([nu_est, a_est, t_est])
            biases.append(bias)
            squared_errors.append(bias**2)
    
    biases = np.array(biases)
    squared_errors = np.array(squared_errors)
    
    avg_bias = np.nanmean(biases, axis=0)
    avg_squared_error = np.nanmean(squared_errors, axis=0)
    valid_iterations = iterations - invalid_count
    
    return avg_bias, avg_squared_error, valid_iterations, invalid_count

def print_results(N, avg_bias, avg_squared_error, valid_iters, invalid_iters):
    """
    Print the results of the simulate-and-recover analysis.
    """
    print(f"Sample size N = {N}:")
    print("Valid iterations:", valid_iters)
    print("Invalid iterations:", invalid_iters)
    print("Average Bias [v, a, t]:", avg_bias)
    print("Average Squared Error [v, a, t]:", avg_squared_error)
    print("-----\n")

def main():
    """
    Main function to run the simulate-and-recover analysis for various sample sizes.
    """
    a_true = 1.0  # True boundary separation
    v_true = 1.0  # True drift rate
    t_true = 0.3  # True nondecision time
    sample_sizes = [10, 40, 4000]  # Different sample sizes to test
    
    for N in sample_sizes:
        avg_bias, avg_squared_error, valid_iters, invalid_iters = simulate_and_recover(a_true, v_true, t_true, N)
        print_results(N, avg_bias, avg_squared_error, valid_iters, invalid_iters)

if __name__ == "__main__":
    main()
