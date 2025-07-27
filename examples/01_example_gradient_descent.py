import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow for module imports
# This is necessary for running examples from the 'examples' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Optimization_Techniques.basic_techniques.gradient_descent.gradient_descent import GradientDescent

def run_gradient_descent_example():
    """
    Demonstrates the use of the GradientDescent optimizer on a simple
    quadratic objective function.
    """
    print("--- Running Gradient Descent Example ---")

    # 1. Define a simple objective function (e.g., a quadratic function)
    def quadratic_function(x):
        """A simple quadratic bowl shape: f(x) = x[0]^2 + x[1]^2"""
        return x[0]**2 + x[1]**2

    # 2. Define the gradient of the objective function
    def quadratic_gradient(x):
        """The gradient of the quadratic function: grad(f) = [2*x[0], 2*x[1]]"""
        return 2 * np.array(x)

    # 3. Set the initial point
    initial_point = np.array([4.0, -3.0])

    # 4. Instantiate the optimizer
    optimizer = GradientDescent(learning_rate=0.1, max_iterations=50, verbose=True)

    # 5. Run the minimization
    result = optimizer.minimize(quadratic_function, quadratic_gradient, initial_point)

    # 6. Print the results
    print("\n--- Gradient Descent Optimization Results ---")
    print(f"Optimal solution (x): {result['solution']}")
    print(f"Function value at solution: {result['function_value']:.6f}")
    print(f"Number of iterations: {result['iterations']}")

    # 7. Plot the convergence
    optimizer.plot_convergence()
    print("\nGradient Descent example finished.")

if __name__ == "__main__":
    run_gradient_descent_example()
