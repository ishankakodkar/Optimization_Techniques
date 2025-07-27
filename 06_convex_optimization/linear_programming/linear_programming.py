import numpy as np
from scipy.optimize import linprog

# Example: Solve a simple production problem
# Maximize profit P = 3x + 5y
# Subject to:
# 1. x <= 4 (Material A constraint)
# 2. 2y <= 12 (Material B constraint)
# 3. 3x + 2y <= 18 (Labor constraint)
# 4. x >= 0, y >= 0

# The `linprog` function minimizes, so we minimize -P = -3x - 5y
c = [-3, -5]

# Constraints are in the form A_ub @ x <= b_ub
A_ub = [
    [1, 0],  # x <= 4
    [0, 2],  # 2y <= 12
    [3, 2]   # 3x + 2y <= 18
]
b_ub = [4, 12, 18]

# Bounds for x and y (x >= 0, y >= 0)
x_bounds = (0, None)
y_bounds = (0, None)

# Solve the linear program
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[x_bounds, y_bounds], method='highs')

# Print the results
if result.success:
    print("Linear Program solved successfully!")
    print(f"Optimal solution (x, y): {result.x}")
    # The objective function value is the minimum of -P, so max P is -result.fun
    print(f"Maximum profit: {-result.fun:.2f}")
else:
    print("Linear Program failed to solve.")
    print(f"Status: {result.message}")
