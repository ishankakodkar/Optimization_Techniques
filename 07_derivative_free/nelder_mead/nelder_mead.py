import numpy as np
from scipy.optimize import minimize

# Define a sample objective function (e.g., Rosenbrock function)
def rosenbrock(x):
    """The Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Initial guess
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

# Perform optimization using the Nelder-Mead method
# This is available directly in SciPy's minimize function
result = minimize(
    fun=rosenbrock,
    x0=x0,
    method='Nelder-Mead',
    options={
        'disp': True,       # Display convergence messages
        'maxiter': 1000,    # Maximum number of iterations
        'xatol': 1e-8,      # Absolute error in xopt between iterations that is acceptable for convergence
    }
)

# Print the results
print("\n--- Nelder-Mead Optimization Results ---")
print(f"Successful: {result.success}")
print(f"Message: {result.message}")
print(f"Optimal solution (x): {result.x}")
print(f"Function value at solution: {result.fun:.6f}")
print(f"Number of iterations: {result.nit}")
print(f"Number of function evaluations: {result.nfev}")
