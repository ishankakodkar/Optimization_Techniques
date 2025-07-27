import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt

class SubgradientMethod:
    """
    Subgradient method for convex, possibly non-differentiable functions.
    """
    def __init__(self, step_size: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = False):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.history = {
            'x': [],
            'f_x': [],
            'subgrad_norm': [],
        }

    def minimize(self, objective: Callable, subgradient: Callable, initial_point: np.ndarray) -> dict:
        x = np.array(initial_point, dtype=float)
        for k in range(self.max_iterations):
            g = subgradient(x)
            norm_g = np.linalg.norm(g)
            f_x = objective(x)
            self.history['x'].append(x.copy())
            self.history['f_x'].append(f_x)
            self.history['subgrad_norm'].append(norm_g)
            if self.verbose and k % 100 == 0:
                print(f"Iter {k}: f(x)={f_x:.6f}, ||g||={norm_g:.4e}")
            if norm_g < self.tolerance:
                break
            alpha_k = self.step_size / np.sqrt(k + 1)
            x = x - alpha_k * g
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
        plt.semilogy(self.history['subgrad_norm'], 'r-', label='Subgrad Norm')
        plt.xlabel('Iteration')
        plt.ylabel('||g||')
        plt.title('Subgradient Norm')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
