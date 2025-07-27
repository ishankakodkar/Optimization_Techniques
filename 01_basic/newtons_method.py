"""
Newton's Method for Optimization

Mathematical Background:
========================
Newton's method is a second-order optimization algorithm that uses both the gradient 
(first derivative) and the Hessian matrix (second derivatives) to find the minimum 
of a function. It has quadratic convergence near the optimum.

Algorithm:
x_{k+1} = x_k - H^{-1}(x_k) ∇f(x_k)

Where:
- x_k: current point
- ∇f(x_k): gradient (first derivatives)
- H(x_k): Hessian matrix (second derivatives)
- H^{-1}(x_k): inverse of Hessian matrix

Convergence:
- Quadratic convergence rate: O(1/k²) near optimum
- Requires positive definite Hessian for guaranteed convergence
- Much faster than gradient descent when close to optimum

Advantages:
- Very fast convergence (quadratic)
- Scale-invariant (automatically adapts to problem conditioning)
- No learning rate to tune

Disadvantages:
- Requires computing and inverting Hessian (expensive for large problems)
- May not converge if Hessian is not positive definite
- Can be unstable far from optimum

Applications:
- Small to medium-sized optimization problems
- Problems where Hessian can be computed efficiently
- Fine-tuning near optimum after coarse optimization
"""

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
        """
        Initialize Newton's Method optimizer.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for gradient norm
            regularization: Regularization parameter for Hessian
            use_line_search: Whether to use backtracking line search
            verbose: Whether to print iteration details
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.use_line_search = use_line_search
        self.verbose = verbose
        
        # History tracking
        self.history = {
            'x': [],
            'f_x': [],
            'gradient_norm': [],
            'step_size': [],
            'hessian_condition': []
        }
    
    def numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """
        Compute numerical gradient using central differences.
        
        Args:
            func: Function to differentiate
            x: Point at which to compute gradient
            h: Step size for finite differences
            
        Returns:
            Numerical gradient vector
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad
    
    def numerical_hessian(self, func: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Compute numerical Hessian using finite differences.
        
        Args:
            func: Function to compute Hessian for
            x: Point at which to compute Hessian
            h: Step size for finite differences
            
        Returns:
            Numerical Hessian matrix
        """
        n = len(x)
        hessian = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Compute second partial derivative ∂²f/∂x_i∂x_j
                x_pp = x.copy()  # x + h_i + h_j
                x_pm = x.copy()  # x + h_i - h_j
                x_mp = x.copy()  # x - h_i + h_j
                x_mm = x.copy()  # x - h_i - h_j
                
                x_pp[i] += h
                x_pp[j] += h
                
                x_pm[i] += h
                x_pm[j] -= h
                
                x_mp[i] -= h
                x_mp[j] += h
                
                x_mm[i] -= h
                x_mm[j] -= h
                
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
        
        # Ensure symmetry
        hessian = (hessian + hessian.T) / 2
        return hessian
    
    def backtracking_line_search(self, func: Callable, x: np.ndarray, 
                                direction: np.ndarray, gradient: np.ndarray,
                                alpha_init: float = 1.0, c1: float = 1e-4,
                                rho: float = 0.5, max_iter: int = 50) -> float:
        """
        Backtracking line search to find appropriate step size.
        
        Args:
            func: Objective function
            x: Current point
            direction: Search direction
            gradient: Gradient at current point
            alpha_init: Initial step size
            c1: Armijo condition parameter
            rho: Step size reduction factor
            max_iter: Maximum line search iterations
            
        Returns:
            Step size that satisfies Armijo condition
        """
        alpha = alpha_init
        f_x = func(x)
        
        for _ in range(max_iter):
            x_new = x + alpha * direction
            f_new = func(x_new)
            
            # Armijo condition: f(x + α*d) ≤ f(x) + c1*α*∇f^T*d
            if f_new <= f_x + c1 * alpha * np.dot(gradient, direction):
                return alpha
            
            alpha *= rho
        
        return alpha
    
    def minimize(self, objective: Callable, initial_point: np.ndarray,
                 gradient: Optional[Callable] = None,
                 hessian: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
        """
        Minimize the objective function using Newton's method.
        
        Args:
            objective: Function to minimize f(x)
            initial_point: Starting point for optimization
            gradient: Gradient function ∇f(x). If None, uses numerical gradient
            hessian: Hessian function H(x). If None, uses numerical Hessian
            
        Returns:
            Tuple of (optimal_point, optimization_info)
        """
        x = np.array(initial_point, dtype=float)
        
        # Clear history
        self.history = {key: [] for key in self.history.keys()}
        
        if self.verbose:
            print(f"Starting Newton's Method optimization")
            print(f"Initial point: {x}")
            print(f"Initial function value: {objective(x):.6f}")
            print("-" * 50)
        
        for iteration in range(self.max_iterations):
            # Compute function value
            f_x = objective(x)
            
            # Compute gradient
            if gradient is not None:
                grad = gradient(x)
            else:
                grad = self.numerical_gradient(objective, x)
            
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: f(x) = {f_x:.6f}, ||∇f|| = {grad_norm:.6f}")
            
            # Convergence check
            if grad_norm < self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration} iterations")
                break
            
            # Compute Hessian
            if hessian is not None:
                H = hessian(x)
            else:
                H = self.numerical_hessian(objective, x)
            
            # Regularize Hessian to ensure positive definiteness
            H_reg = H + self.regularization * np.eye(len(x))
            
            # Compute condition number
            try:
                eigenvals = np.linalg.eigvals(H_reg)
                condition_number = np.max(eigenvals) / np.max([np.min(eigenvals), 1e-12])
            except:
                condition_number = np.inf
            
            # Solve Newton system: H * d = -∇f
            try:
                newton_direction = solve(H_reg, -grad)
            except LinAlgError:
                warnings.warn("Hessian is singular, using gradient descent step")
                newton_direction = -grad
                condition_number = np.inf
            
            # Determine step size
            if self.use_line_search:
                step_size = self.backtracking_line_search(objective, x, newton_direction, grad)
            else:
                step_size = 1.0
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f_x'].append(f_x)
            self.history['gradient_norm'].append(grad_norm)
            self.history['step_size'].append(step_size)
            self.history['hessian_condition'].append(condition_number)
            
            # Newton update
            x = x + step_size * newton_direction
        
        # Final results
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
        """
        Plot the convergence history.
        
        Args:
            title: Title for the plot
        """
        if not self.history['f_x']:
            print("No optimization history to plot. Run minimize() first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        iterations = range(len(self.history['f_x']))
        
        # Function value
        ax1.plot(iterations, self.history['f_x'], 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function Value')
        ax1.set_title('Objective Function')
        ax1.grid(True, alpha=0.3)
        
        # Gradient norm (log scale)
        ax2.semilogy(iterations, self.history['gradient_norm'], 'r-', linewidth=2, marker='s')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Gradient Norm')
        ax2.grid(True, alpha=0.3)
        
        # Step size
        ax3.plot(iterations, self.history['step_size'], 'g-', linewidth=2, marker='^')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Step Size')
        ax3.set_title('Step Size (Line Search)')
        ax3.grid(True, alpha=0.3)
        
        # Hessian condition number
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
