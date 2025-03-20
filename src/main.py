from .simulation import simulate_and_recover  # Critical change
import numpy as np

"""main.py acts as an entry point for running this code"""

def print_results(results: dict):
    """The formatting for the output was drafted by copilot"""
    for N in [10, 40, 4000]:
        bias, mse, valid, invalid = results[N]
        print(f"\nN = {N} (Valid: {valid}, Invalid: {invalid})")
        print(f"Average Bias [ν, a, τ]: {np.round(bias, 4)}")
        print(f"Average MSE  [ν, a, τ]: {np.round(mse, 4)}")
        print("─" * 50)

def main():
    """Execute analysis across sample sizes"""
    results = {}
    for N in [10, 40, 4000]:
        results[N] = simulate_and_recover(N, 1000)
    print_results(results)

if __name__ == "__main__":
    main()