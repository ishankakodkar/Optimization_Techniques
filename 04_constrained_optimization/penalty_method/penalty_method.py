import numpy as np
from typing import Callable, List
from scipy.optimize import minimize as scipy_minimize

class PenaltyMethod:
    """
    Quadratic Penalty Method for inequality-constrained optimization.
    """
    def __init__(self, rho_init: float = 1.0, rho_factor: float = 10.0, max_outer_iter: int = 10, max_inner_iter: int = 100, tol: float = 1e-6, verbose: bool = False):
        self.rho_init = rho_init
        self.rho_factor = rho_factor
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.verbose = verbose
        self.history = {
            'x': [],
            'f_x': [],
            'rho': []
        }

    def minimize(self, objective: Callable, grad_objective: Callable, constraints: List[Callable], grad_constraints: List[Callable], x0: np.ndarray) -> dict:
        x = np.array(x0, dtype=float)
        rho = self.rho_init

        for outer_iter in range(self.max_outer_iter):
            def penalized_obj(x_inner):
                penalty = 0.5 * sum(np.maximum(0, c(x_inner))**2 for c in constraints)
                return objective(x_inner) + rho * penalty

            def grad_penalized_obj(x_inner):
                grad_penalty = sum(np.maximum(0, c(x_inner)) * gc(x_inner) for c, gc in zip(constraints, grad_constraints))
                return grad_objective(x_inner) + rho * grad_penalty

            # Inner loop: Unconstrained minimization using a robust solver
            res = scipy_minimize(
                fun=penalized_obj,
                x0=x,
                jac=grad_penalized_obj,
                method='BFGS',
                tol=self.tol,
                options={'maxiter': self.max_inner_iter}
            )
            x = res.x
            
            self.history['x'].append(x.copy())
            self.history['f_x'].append(objective(x))
            self.history['rho'].append(rho)

            if self.verbose:
                print(f"Outer iter {outer_iter}: rho={rho:.2e}, f(x)={objective(x):.6f}")

            # Check for convergence (e.g., change in x)
            if outer_iter > 0 and np.linalg.norm(self.history['x'][-1] - self.history['x'][-2]) < self.tol:
                break

            rho *= self.rho_factor

        return {
            'solution': x,
            'function_value': objective(x),
            'iterations': len(self.history['x']),
            'history': self.history
        }
