import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt

class ProjectedGradientDescent:
    """
    Projected Gradient Descent for constrained convex optimization.
    """
    def __init__(self, step_size: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = False):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.history = {
            'x': [],
            'f_x': [],
            'grad_norm': [],
        }

    def minimize(self, objective: Callable, gradient: Callable, projection: Callable, initial_point: np.ndarray) -> dict:
        x = np.array(initial_point, dtype=float)

        for k in range(self.max_iterations):
            x_prev = x.copy()
            grad = gradient(x)

            # Perform the gradient and projection step
            x = projection(x - self.step_size * grad)

            # Record history for analysis
            f_x = objective(x)
            self.history['x'].append(x.copy())
            self.history['f_x'].append(f_x)
            # Note: grad_norm is of the previous point, but useful for monitoring
            self.history['grad_norm'].append(np.linalg.norm(grad))

            # Check for convergence
            change = np.linalg.norm(x - x_prev)
            if self.verbose and k % 100 == 0:
                print(f"Iter {k}: f(x)={f_x:.6f}, ||x_k+1 - x_k||={change:.4e}")

            if change < self.tolerance:
                if self.verbose:
                    print(f"\nConvergence achieved after {k+1} iterations.")
                break

        return {
            'solution': x,
            'function_value': objective(x),
            'iterations': len(self.history['x']),
            'history': self.history
        }

    def plot_convergence(self):
        if not self.history['f_x']:
            print("No optimization history to plot. Run minimize() first.")
            return
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['f_x'], 'b-', label='Objective')
        plt.xlabel('Iteration')
        plt.ylabel('f(x)')
        plt.title('Objective Value')
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.semilogy(self.history['grad_norm'], 'r-', label='Grad Norm')
        plt.xlabel('Iteration')
        plt.ylabel('||grad||')
        plt.title('Gradient Norm')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
