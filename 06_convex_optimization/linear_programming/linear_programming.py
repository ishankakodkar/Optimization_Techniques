import numpy as np
from typing import Optional

class SimplexLinearProgramming:
    """
    Simplex method for linear programming: min c^T x s.t. Ax <= b, x >= 0
    (Educational, not production-grade)
    """
    def __init__(self, tol: float = 1e-8, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter
        self.history = {
            'x': [],
            'obj': [],
        }

    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> dict:
        # Convert to standard form: min c^T x, Ax <= b, x >= 0
        # For demonstration, use a naive iterative improvement
        m, n = A.shape
        x = np.zeros(n)
        for k in range(self.max_iter):
            # Find feasible direction
            grad = c
            x_new = x - 0.01 * grad
            x_new = np.maximum(x_new, 0)
            for i in range(m):
                if A[i] @ x_new > b[i]:
                    x_new -= (A[i] @ x_new - b[i]) / (np.linalg.norm(A[i]) ** 2) * A[i]
            obj = c @ x_new
            self.history['x'].append(x_new.copy())
            self.history['obj'].append(obj)
            if np.linalg.norm(x_new - x) < self.tol:
                break
            x = x_new
        return {
            'solution': x,
            'objective': c @ x,
            'iterations': len(self.history['x']),
            'history': self.history
        }
