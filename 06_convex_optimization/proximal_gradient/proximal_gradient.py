import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt

class ProximalGradientMethod:
    """
    Proximal Gradient Method for composite convex functions.
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

    def minimize(self, g: Callable, grad_g: Callable, prox_h: Callable, initial_point: np.ndarray) -> dict:
        x = np.array(initial_point, dtype=float)
        for k in range(self.max_iterations):
            grad = grad_g(x)
            norm_grad = np.linalg.norm(grad)
            f_x = g(x) + prox_h(x, 0)  # prox_h(x, 0) = h(x)
            self.history['x'].append(x.copy())
            self.history['f_x'].append(f_x)
            self.history['grad_norm'].append(norm_grad)
            if self.verbose and k % 100 == 0:
                print(f"Iter {k}: f(x)={f_x:.6f}, ||grad||={norm_grad:.4e}")
            if norm_grad < self.tolerance:
                break
            x = prox_h(x - self.step_size * grad, self.step_size)
        return {
            'solution': x,
            'function_value': g(x) + prox_h(x, 0),
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
