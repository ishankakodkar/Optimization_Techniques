import numpy as np
from scipy.optimize import minimize

# Example: Solve a simple Quadratic Program
# Minimize f(x) = 2*x1^2 + x2^2 + x1*x2 + x1 + x2
# Subject to:
# 1. x1 >= 0
# 2. x2 >= 0
# 3. x1 + x2 = 1

# The objective function in matrix form: 1/2 * x.T * Q * x + c.T * x
# f(x) = 1/2 * [x1, x2] * [[4, 1], [1, 2]] * [x1, x2] + [1, 1] * [x1, x2]
Q = np.array([[4., 1.], [1., 2.]])
c = np.array([1., 1.])

def objective_function(x, Q, c):
    """The quadratic objective function to be minimized."""
    return 0.5 * x.T @ Q @ x + c.T @ x

# Define the constraints
# Equality constraint: x1 + x2 - 1 = 0
cons = ({'type': 'eq', 'fun': lambda x: np.array([x[0] + x[1] - 1])})

# Bounds for each variable (x1 >= 0, x2 >= 0)
bounds = ((0, None), (0, None))

# Initial guess
x0 = np.array([0.5, 0.5])

# Solve the Quadratic Program
result = minimize(
    fun=objective_function,
    x0=x0,
    args=(Q, c),
    method='SLSQP',  # Sequential Least Squares Programming is suitable for QPs
    bounds=bounds,
    constraints=cons
)

# Print the results
if result.success:
    print("Quadratic Program solved successfully!")
    print(f"Optimal solution (x1, x2): {result.x}")
    print(f"Minimum objective value: {result.fun:.4f}")
else:
    print("Quadratic Program failed to solve.")
    print(f"Status: {result.message}")
