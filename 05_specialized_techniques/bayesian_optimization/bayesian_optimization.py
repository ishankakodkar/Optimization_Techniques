import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
import matplotlib.pyplot as plt

# Define a sample black-box function to optimize (e.g., with multiple local minima)
def black_box_function(x):
    """A sample function to demonstrate Bayesian Optimization."""
    x = np.array(x)
    return -np.sin(5 * x[0]) * (1 - np.tanh(x[0]**2)) - np.random.randn() * 0.1

# Define the search space (bounds for the variable)
search_space = [(-2.0, 2.0)]

# Perform Bayesian Optimization
# gp_minimize uses a Gaussian Process surrogate and Expected Improvement by default
result = gp_minimize(
    func=black_box_function,
    dimensions=search_space,
    n_calls=20,  # Number of function evaluations
    n_initial_points=5, # Number of random points to start with
    random_state=42,
    verbose=True
)

# Print the results
print(f"Best parameters found: {result.x}")
print(f"Best function value: {result.fun:.4f}")

# Plot the convergence
plot_convergence(result)
plt.title('Convergence Plot')
plt.xlabel('Number of calls')
plt.ylabel('Objective value (minimum)')
plt.grid(True)
plt.show()

# Plot the objective function and the points sampled
plot_objective(result, n_points=100)
plt.title('Objective Function and Sampled Points')
plt.xlabel('Parameter value')
plt.ylabel('Objective value')
plt.grid(True)
plt.show()
