import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt

class BarrierInteriorPoint:
    """
    Barrier (log-barrier) interior point method for convex inequality-constrained problems.
    """
    def __init__(self, mu: float = 10.0, t_init: float = 1.0, tol: float = 1e-6, max_iter: int = 50, verbose: bool = False):
        self.mu = mu  # barrier parameter increase factor
        self.t_init = t_init
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.history = {
            'x': [],
            'f_x': [],
            't': [],
        }

    def minimize(self, f: Callable, grad_f: Callable, constraints: list, grad_constraints: list, x0: np.ndarray) -> dict:
        # constraints: list of functions g_i(x) < 0
        # grad_constraints: list of gradients for g_i
        x = np.array(x0, dtype=float)
        t = self.t_init
        for outer in range(self.max_iter):
            def phi(x):
                penalty = 0.0
                for g in constraints:
                    val = g(x)
                    if val >= 0:
                        return np.inf
                    penalty += -np.log(-val)
                return t * f(x) + penalty
            def grad_phi(x):
                grad = t * grad_f(x)
                for g, grad_g in zip(constraints, grad_constraints):
                    val = g(x)
                    if val >= 0:
                        return np.full_like(x, np.nan)
                    grad += -1.0 / val * grad_g(x)
                return grad
            # Use gradient descent as inner solver for simplicity
            for _ in range(100):
                grad = grad_phi(x)
                if np.any(np.isnan(grad)) or np.linalg.norm(grad) < self.tol:
                    break
                x = x - 0.01 * grad  # fixed step size for demo
            self.history['x'].append(x.copy())
            self.history['f_x'].append(f(x))
            self.history['t'].append(t)
            if self.verbose:
                print(f"Outer iter {outer}: t={t:.2e}, f(x)={f(x):.6f}")
            if len(constraints) == 0 or all(g(x) < -self.tol for g in constraints):
                if 1.0 / t < self.tol:
                    break
            t *= self.mu
        return {
            'solution': x,
            'function_value': f(x),
            'iterations': len(self.history['x']),
            'history': self.history
        }

    def plot_convergence(self):
        if not self.history['f_x']:
            print("No optimization history to plot. Run minimize() first.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.history['f_x'], 'b-', label='Objective')
        plt.xlabel('Outer Iteration')
        plt.ylabel('f(x)')
        plt.title('Objective Value (Barrier Method)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
