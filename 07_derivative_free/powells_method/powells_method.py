import numpy as np
from scipy.optimize import minimize

# Define a sample objective function (e.g., a simple quadratic function)
def quadratic_function(x):
    """A simple quadratic bowl shape."""
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

# Initial guess
x0 = np.array([0.0, 0.0])

# Perform optimization using Powell's method
# This is available directly in SciPy's minimize function
result = minimize(
    fun=quadratic_function,
    x0=x0,
    method='Powell',
    options={
        'disp': True,       # Display convergence messages
        'maxiter': 1000,    # Maximum number of iterations
        'ftol': 1e-6,       # Tolerance for convergence
    }
)

# Print the results
print("\n--- Powell's Method Optimization Results ---")
print(f"Successful: {result.success}")
print(f"Message: {result.message}")
print(f"Optimal solution (x): {result.x}")
print(f"Function value at solution: {result.fun:.6f}")
print(f"Number of iterations: {result.nit}")
print(f"Number of function evaluations: {result.nfev}")
