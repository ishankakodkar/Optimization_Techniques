import numpy as np
from scipy.optimize import milp

# Example: Knapsack Problem
# We want to maximize the value of items in a knapsack with a weight limit.
# max sum(v_i * x_i) s.t. sum(w_i * x_i) <= W, where x_i are binary (0 or 1)

# Problem data
values = np.array([60, 100, 120, 80, 90])  # Value of each item
weights = np.array([10, 20, 30, 15, 25]) # Weight of each item
max_weight = 50  # Maximum weight capacity of the knapsack

# Number of items
n_items = len(values)

# The objective function to MINIMIZE is the negative of the total value
c = -values

# Constraints: one constraint for the weight limit
# w'x <= max_weight
A = np.array([weights])
b_u = np.array([max_weight]) # Upper bound for the constraint

# Integrality constraints: all variables are binary (integer and between 0 and 1)
integrality = np.ones(n_items, dtype=int)

# Bounds for each variable (0 to 1)
bounds = [(0, 1)] * n_items

# Solve the Mixed-Integer Linear Program (MILP)
# In this case, it's a Binary Integer Program (BIP)
result = milp(c=c, constraints=(A, None, b_u), integrality=integrality, bounds=bounds)

# Print the results
if result.success:
    selected_items = np.round(result.x).astype(bool)
    total_value = -result.fun
    total_weight = weights @ result.x

    print("MILP solved successfully!")
    print(f"Selected items (1 for yes, 0 for no): {result.x.astype(int)}")
    print(f"Total value: {total_value:.2f}")
    print(f"Total weight: {total_weight:.2f}")
else:
    print("MILP failed to solve.")
    print(f"Status: {result.message}")
