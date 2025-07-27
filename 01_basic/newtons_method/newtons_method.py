import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
from scipy.linalg import solve, LinAlgError
import warnings

class NewtonsMethod:
    """
    Newton's Method optimizer for unconstrained optimization.
    
    Implements pure Newton's method with optional regularization
    and line search for improved robustness.
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-8,
                 regularization: float = 1e-8, use_line_search: bool = True,
                 verbose: bool = False):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.use_line_search = use_line_search
        self.verbose = verbose
        self.history = {
            'x': [],
            'f_x': [],
            'gradient_norm': [],
            'step_size': [],
            'hessian_condition': []
        }
    
    def numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad
    
    def numerical_hessian(self, func: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        n = len(x)
        hessian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += h
                x_pp[j] += h
                x_pm[i] += h
                x_pm[j] -= h
                x_mp[i] -= h
                x_mp[j] += h
                x_mm[i] -= h
                x_mm[j] -= h
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
        hessian = (hessian + hessian.T) / 2
        return hessian
    
    def backtracking_line_search(self, func: Callable, x: np.ndarray, 
                                direction: np.ndarray, gradient: np.ndarray,
                                alpha_init: float = 1.0, c1: float = 1e-4,
                                rho: float = 0.5, max_iter: int = 50) -> float:
        alpha = alpha_init
        f_x = func(x)
        for _ in range(max_iter):
            x_new = x + alpha * direction
            f_new = func(x_new)
            if f_new <= f_x + c1 * alpha * np.dot(gradient, direction):
                return alpha
            alpha *= rho
        return alpha
    
    def minimize(self, objective: Callable, initial_point: np.ndarray,
                 gradient: Optional[Callable] = None,
                 hessian: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
        x = np.array(initial_point, dtype=float)
        self.history = {key: [] for key in self.history.keys()}
        if self.verbose:
            print(f"Starting Newton's Method optimization")
            print(f"Initial point: {x}")
            print(f"Initial function value: {objective(x):.6f}")
            print("-" * 50)
        for iteration in range(self.max_iterations):
            f_x = objective(x)
            grad = gradient(x) if gradient is not None else self.numerical_gradient(objective, x)
            grad_norm = np.linalg.norm(grad)
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: f(x) = {f_x:.6f}, ||∇f|| = {grad_norm:.6f}")
            if grad_norm < self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration} iterations")
                break
            H = hessian(x) if hessian is not None else self.numerical_hessian(objective, x)
            H_reg = H + self.regularization * np.eye(len(x))
            try:
                eigenvals = np.linalg.eigvals(H_reg)
                condition_number = np.max(eigenvals) / np.max([np.min(eigenvals), 1e-12])
            except:
                condition_number = np.inf
            try:
                newton_direction = solve(H_reg, -grad)
            except LinAlgError:
                warnings.warn("Hessian is singular, using gradient descent step")
                newton_direction = -grad
                condition_number = np.inf
            if self.use_line_search:
                step_size = self.backtracking_line_search(objective, x, newton_direction, grad)
            else:
                step_size = 1.0
            self.history['x'].append(x.copy())
            self.history['f_x'].append(f_x)
            self.history['gradient_norm'].append(grad_norm)
            self.history['step_size'].append(step_size)
            self.history['hessian_condition'].append(condition_number)
            x = x + step_size * newton_direction
        final_f = objective(x)
        final_grad = gradient(x) if gradient else self.numerical_gradient(objective, x)
        final_grad_norm = np.linalg.norm(final_grad)
        optimization_info = {
            'iterations': len(self.history['x']),
            'final_function_value': final_f,
            'final_gradient_norm': final_grad_norm,
            'converged': final_grad_norm < self.tolerance,
            'history': self.history
        }
        if self.verbose:
            print("-" * 50)
            print(f"Optimization completed!")
            print(f"Final point: {x}")
            print(f"Final function value: {final_f:.6f}")
            print(f"Final gradient norm: {final_grad_norm:.6f}")
            print(f"Converged: {optimization_info['converged']}")
        return x, optimization_info
    
    def plot_convergence(self, title: str = "Newton's Method Convergence"):
        if not self.history['f_x']:
            print("No optimization history to plot. Run minimize() first.")
            return
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        iterations = range(len(self.history['f_x']))
        ax1.plot(iterations, self.history['f_x'], 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function Value')
        ax1.set_title('Objective Function')
        ax1.grid(True, alpha=0.3)
        ax2.semilogy(iterations, self.history['gradient_norm'], 'r-', linewidth=2, marker='s')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Gradient Norm')
        ax2.grid(True, alpha=0.3)
        ax3.plot(iterations, self.history['step_size'], 'g-', linewidth=2, marker='^')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Step Size')
        ax3.set_title('Step Size (Line Search)')
        ax3.grid(True, alpha=0.3)
        valid_conditions = [c for c in self.history['hessian_condition'] if c != np.inf]
        if valid_conditions:
            ax4.semilogy(iterations[:len(valid_conditions)], valid_conditions, 
                        'm-', linewidth=2, marker='d')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Condition Number (log scale)')
            ax4.set_title('Hessian Condition Number')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Hessian condition numbers\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Hessian Condition Number')
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
