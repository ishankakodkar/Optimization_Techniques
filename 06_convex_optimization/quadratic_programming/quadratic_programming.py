import numpy as np
from typing import Optional
import scipy.linalg

class QuadraticProgramming:
    """
    Simple active-set solver for convex quadratic programming problems:
    min 1/2 x^T Q x + c^T x s.t. Ax <= b
    """
    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter
        self.history = {
            'x': [],
            'obj': [],
        }

    def solve(self, Q: np.ndarray, c: np.ndarray, A: Optional[np.ndarray], b: Optional[np.ndarray], x0: Optional[np.ndarray] = None) -> dict:
        n = Q.shape[0]
        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.array(x0, dtype=float)
        # For demonstration: projected gradient descent for QP
        for k in range(self.max_iter):
            grad = Q @ x + c
            x_new = x - 0.01 * grad
            if A is not None and b is not None:
                # Project onto feasible set (Ax <= b)
                for i in range(A.shape[0]):
                    if A[i] @ x_new > b[i]:
                        x_new -= (A[i] @ x_new - b[i]) / (np.linalg.norm(A[i]) ** 2) * A[i]
            obj = 0.5 * x_new @ Q @ x_new + c @ x_new
            self.history['x'].append(x_new.copy())
            self.history['obj'].append(obj)
            if np.linalg.norm(x_new - x) < self.tol:
                break
            x = x_new
        return {
            'solution': x,
            'objective': 0.5 * x @ Q @ x + c @ x,
            'iterations': len(self.history['x']),
            'history': self.history
        }
