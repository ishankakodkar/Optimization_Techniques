import numpy as np
from typing import Callable, List
from scipy.optimize import minimize as scipy_minimize

class AugmentedLagrangianMethod:
    """
    Augmented Lagrangian Method for equality-constrained optimization.
    min f(x) s.t. h(x) = 0
    """
    def __init__(self, rho_init: float = 1.0, rho_factor: float = 2.0, max_outer_iter: int = 10, max_inner_iter: int = 100, tol: float = 1e-6, verbose: bool = False):
        self.rho_init = rho_init
        self.rho_factor = rho_factor
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.verbose = verbose
        self.history = {
            'x': [],
            'f_x': [],
            'lambda': [],
            'rho': []
        }

    def minimize(self, objective: Callable, grad_objective: Callable, constraints: List[Callable], grad_constraints: List[Callable], x0: np.ndarray) -> dict:
        x = np.array(x0, dtype=float)
        rho = self.rho_init
        lambdas = np.zeros(len(constraints))

        for outer_iter in range(self.max_outer_iter):
            def augmented_lagrangian(x_inner):
                constraint_vals = np.array([c(x_inner) for c in constraints])
                return objective(x_inner) + lambdas @ constraint_vals + 0.5 * rho * np.sum(constraint_vals**2)

            def grad_augmented_lagrangian(x_inner):
                constraint_vals = np.array([c(x_inner) for c in constraints])
                grad_constraints_vals = np.array([gc(x_inner) for gc in grad_constraints])
                return grad_objective(x_inner) + grad_constraints_vals.T @ (lambdas + rho * constraint_vals)

            # Inner loop: Unconstrained minimization using a robust solver
            x_prev_outer = x.copy()
            res = scipy_minimize(
                fun=augmented_lagrangian,
                x0=x,
                jac=grad_augmented_lagrangian,
                method='BFGS',
                tol=self.tol,
                options={'maxiter': self.max_inner_iter}
            )
            x = res.x
            
            self.history['x'].append(x.copy())
            self.history['f_x'].append(objective(x))
            self.history['lambda'].append(lambdas.copy())
            self.history['rho'].append(rho)

            # Update multipliers
            constraint_vals = np.array([c(x) for c in constraints])
            lambdas += rho * constraint_vals

            if self.verbose:
                print(f"Outer iter {outer_iter}: rho={rho:.2e}, f(x)={objective(x):.6f}, |h(x)|={np.linalg.norm(constraint_vals):.4e}")

            # Check for convergence
            if np.linalg.norm(x - x_prev_outer) < self.tol and np.linalg.norm(constraint_vals) < self.tol:
                break

            rho *= self.rho_factor

        return {
            'solution': x,
            'function_value': objective(x),
            'iterations': len(self.history['x']),
            'history': self.history
        }
