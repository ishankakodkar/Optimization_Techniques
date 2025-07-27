import numpy as np
from typing import Callable
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class BarrierInteriorPoint:
    """
    Barrier (log-barrier) interior point method for convex inequality-constrained problems.
    min f(x) s.t. g_i(x) <= 0
    """
    def __init__(self, mu: float = 10.0, t_init: float = 1.0, tol: float = 1e-6, max_iter: int = 50, verbose: bool = False):
        self.mu = mu  # Barrier parameter increase factor
        self.t_init = t_init
        self.tol = tol  # Tolerance for the duality gap m/t
        self.max_iter = max_iter
        self.verbose = verbose
        self.history = {
            'x': [],
            'f_x': [],
            't': [],
        }

    def minimize(self, f: Callable, grad_f: Callable, constraints: list, grad_constraints: list, x0: np.ndarray) -> dict:
        x = np.array(x0, dtype=float)
        t = self.t_init
        m = len(constraints)

        for outer in range(self.max_iter):
            # Define the barrier objective and its gradient for the current t
            def barrier_objective(x_inner):
                penalty = sum(-np.log(-g(x_inner)) for g in constraints)
                return t * f(x_inner) + penalty

            def barrier_gradient(x_inner):
                grad_penalty = sum(-1.0 / g(x_inner) * grad_g(x_inner) for g, grad_g in zip(constraints, grad_constraints))
                return t * grad_f(x_inner) + grad_penalty

            # Use a robust solver for the inner unconstrained problem (centering step)
            inner_result = minimize(
                fun=barrier_objective,
                x0=x,
                jac=barrier_gradient,
                method='BFGS',
                tol=1e-5 # Inner problem doesn't need to be solved to high accuracy
            )
            x = inner_result.x

            self.history['x'].append(x.copy())
            self.history['f_x'].append(f(x))
            self.history['t'].append(t)

            if self.verbose:
                print(f"Outer iter {outer}: t={t:.2e}, f(x)={f(x):.6f}, duality_gap_approx={m/t:.2e}")

            # Check for convergence based on the duality gap
            if m / t < self.tol:
                break

            # Increase the barrier parameter t
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
