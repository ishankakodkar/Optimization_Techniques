import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 1. Define the problem
# We will use a standard test problem, ZDT1, which has a convex Pareto front.
problem = get_problem("zdt1")

# 2. Instantiate the algorithm
# We'll use the NSGA-II algorithm.
algorithm = NSGA2(
    pop_size=100,       # Population size
    eliminate_duplicates=True
)

# 3. Perform the optimization
# The 'minimize' function runs the algorithm on the problem.
res = minimize(
    problem,
    algorithm,
    ('n_gen', 200),     # Termination criterion: 200 generations
    seed=1,
    verbose=True
)

# 4. Visualize the results
# The result 'res.F' contains the objective space values for the Pareto front.
print("\n--- NSGA-II Optimization Results ---")
print(f"Found {len(res.F)} solutions on the Pareto front.")

# Plot the true Pareto front (if known) and the solutions found
plot = Scatter(title="ZDT1 Problem - Pareto Front")
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7, label="True Pareto Front")
plot.add(res.F, color="red", s=30, label="Solutions Found")
plot.show()
