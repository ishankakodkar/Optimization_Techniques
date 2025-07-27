"""
Adam (Adaptive Moment Estimation)

Mathematical Background:
========================
Adam is a popular optimization algorithm that combines the ideas of Momentum
and RMSprop. It uses both a moving average of the gradient (first moment) and
a moving average of the squared gradient (second moment) to adapt the learning
rate for each parameter.

Algorithm:
m_k = β₁ * m_{k-1} + (1-β₁) * ∇f(x_k)
v_k = β₂ * v_{k-1} + (1-β₂) * ∇f(x_k)²

m_hat_k = m_k / (1 - β₁^k)
v_hat_k = v_k / (1 - β₂^k)

x_{k+1} = x_k - (α / (√v_hat_k + ε)) * m_hat_k

Where:
- m_k, v_k: first and second moment estimates
- β₁, β₂: decay rates for moments (typically 0.9 and 0.999)
- α: learning rate
- ε: small constant for stability
- m_hat_k, v_hat_k: bias-corrected moment estimates

Key Features:
- Combines advantages of Momentum and RMSprop
- Bias correction for early iterations
- Robust to choice of hyperparameters
- Computationally efficient and low memory requirements

Applications:
- Default optimizer for most deep learning tasks
- Large-scale machine learning
- Non-stationary and noisy optimization problems
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
import time


class Adam:
    """
    Adam optimizer with adaptive moment estimation.
    
    Implements the Adam algorithm with bias correction, comprehensive
    tracking, and visualization capabilities.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 verbose: bool = False):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate (α)
            beta1: Decay rate for first moment (m)
            beta2: Decay rate for second moment (v)
            epsilon: Small constant for stability
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Whether to print progress
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # History tracking
        self.history = {
            'x': [],
            'f_x': [],
            'gradient_norm': [],
            'effective_lr': [],
            'first_moment': [],
            'second_moment': []
        }
    
    def numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """Compute numerical gradient using finite differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad
    
    def minimize(self, objective: Callable, initial_point: np.ndarray,
                 gradient: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
        """
        Minimize objective function using Adam.
        
        Args:
            objective: Function to minimize
            initial_point: Starting point
            gradient: Gradient function (optional)
            
        Returns:
            Tuple of (optimal_point, optimization_info)
        """
        x = np.array(initial_point, dtype=float)
        m = np.zeros_like(x)  # First moment (mean)
        v = np.zeros_like(x)  # Second moment (variance)
        
        # Clear history
        self.history = {key: [] for key in self.history.keys()}
        
        if self.verbose:
            print(f"Starting Adam optimization")
            print(f"LR: {self.learning_rate}, β₁: {self.beta1}, β₂: {self.beta2}")
            print(f"Initial point: {x}")
            print(f"Initial function value: {objective(x):.6f}")
            print("-" * 50)
        
        start_time = time.time()
        
        for k in range(1, self.max_iterations + 1):
            # Compute function value
            f_x = objective(x)
            
            # Compute gradient
            if gradient is not None:
                grad = gradient(x)
            else:
                grad = self.numerical_gradient(objective, x)
            
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            
            # Update moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = m / (1 - self.beta1**k)
            v_hat = v / (1 - self.beta2**k)
            
            # Compute effective learning rate
            effective_lr = self.learning_rate / (np.sqrt(v_hat) + self.epsilon)
            avg_effective_lr = np.mean(effective_lr)
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f_x'].append(f_x)
            self.history['gradient_norm'].append(grad_norm)
            self.history['effective_lr'].append(avg_effective_lr)
            self.history['first_moment'].append(np.mean(m_hat))
            self.history['second_moment'].append(np.mean(v_hat))
            
            if self.verbose and k % 100 == 0:
                print(f"Iter {k:4d}: f(x) = {f_x:.6f}, ||∇f|| = {grad_norm:.6f}, "
                      f"avg_lr = {avg_effective_lr:.6f}")
            
            # Convergence check
            if grad_norm < self.tolerance:
                if self.verbose:
                    print(f"Converged after {k} iterations")
                break
            
            # Adam update
            x = x - effective_lr * m_hat
        
        end_time = time.time()
        
        # Final results
        final_f = objective(x)
        final_grad = gradient(x) if gradient else self.numerical_gradient(objective, x)
        final_grad_norm = np.linalg.norm(final_grad)
        
        optimization_info = {
            'iterations': len(self.history['x']),
            'final_function_value': final_f,
            'final_gradient_norm': final_grad_norm,
            'converged': final_grad_norm < self.tolerance,
            'optimization_time': end_time - start_time,
            'history': self.history
        }
        
        if self.verbose:
            print("-" * 50)
            print(f"Optimization completed in {end_time - start_time:.3f} seconds!")
            print(f"Final point: {x}")
            print(f"Final function value: {final_f:.6f}")
            print(f"Final gradient norm: {final_grad_norm:.6f}")
            print(f"Converged: {optimization_info['converged']}")
        
        return x, optimization_info
    
    def plot_convergence(self, title: str = "Adam Convergence"):
        """Plot convergence history."""
        if not self.history['f_x']:
            print("No optimization history to plot. Run minimize() first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        iterations = range(len(self.history['f_x']))
        
        # Function value
        ax1.plot(iterations, self.history['f_x'], 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function Value')
        ax1.set_title('Objective Function')
        ax1.grid(True, alpha=0.3)
        
        # Gradient norm
        ax2.semilogy(iterations, self.history['gradient_norm'], 'r-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Gradient Norm')
        ax2.grid(True, alpha=0.3)
        
        # First moment
        ax3.plot(iterations, self.history['first_moment'], 'g-', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Mean First Moment')
        ax3.set_title('First Moment (Momentum)')
        ax3.grid(True, alpha=0.3)
        
        # Second moment
        ax4.plot(iterations, self.history['second_moment'], 'm-', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Mean Second Moment')
        ax4.set_title('Second Moment (Variance)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def example_adam_vs_others():
    """
    Example 1: Compare Adam with other optimizers
    """
    print("=" * 60)
    print("EXAMPLE 1: Adam vs Other Optimizers")
    print("=" * 60)
    
    # Rosenbrock function
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        grad = np.zeros(2)
        grad[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        grad[1] = 200*(x[1] - x[0]**2)
        return grad
    
    # Optimizers
    from gradient_descent import GradientDescent
    from momentum import MomentumOptimizer
    from rmsprop import RMSprop
    
    optimizers = {
        'Gradient Descent': GradientDescent(learning_rate=0.001, max_iterations=2000),
        'Momentum': MomentumOptimizer(learning_rate=0.001, max_iterations=2000),
        'RMSprop': RMSprop(learning_rate=0.001, max_iterations=2000),
        'Adam': Adam(learning_rate=0.01, max_iterations=2000, verbose=True)
    }
    
    results = {}
    
    plt.figure(figsize=(12, 4))
    
    for name, opt in optimizers.items():
        result, info = opt.minimize(rosenbrock,
                                    initial_point=np.array([-1.0, 1.0]),
                                    gradient=rosenbrock_grad)
        results[name] = info
        
        plt.subplot(1, 2, 1)
        plt.semilogy(range(len(info['history']['f_x'])), 
                    info['history']['f_x'], linewidth=2, label=name)
        
        plt.subplot(1, 2, 2)
        x_path = [x[0] for x in info['history']['x']]
        y_path = [x[1] for x in info['history']['x']]
        plt.plot(x_path, y_path, linewidth=1, alpha=0.7, label=name)
    
    # Finalize plots
    plt.subplot(1, 2, 1)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value (log scale)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(1, 1, 'go', markersize=10, label='True minimum')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Optimization Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Adam vs Other Optimizers on Rosenbrock Function', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    for name, info in results.items():
        print(f"\n{name}:")
        print(f"Converged in {info['iterations']} iterations")
        print(f"Final function value: {info['final_function_value']:.6f}")


def example_hyperparameter_sensitivity():
    """
    Example 2: Sensitivity to hyperparameters (β₁, β₂)
    """
    print("=" * 60)
    print("EXAMPLE 2: Hyperparameter Sensitivity")
    print("=" * 60)
    
    # Beale function
    def beale(x):
        term1 = (1.5 - x[0] + x[0]*x[1])**2
        term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
        term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
        return term1 + term2 + term3
    
    def beale_grad(x):
        grad = np.zeros(2)
        term1 = 1.5 - x[0] + x[0]*x[1]
        term2 = 2.25 - x[0] + x[0]*x[1]**2
        term3 = 2.625 - x[0] + x[0]*x[1]**3
        grad[0] = 2*term1*(-1 + x[1]) + 2*term2*(-1 + x[1]**2) + 2*term3*(-1 + x[1]**3)
        grad[1] = 2*term1*x[0] + 2*term2*x[0]*2*x[1] + 2*term3*x[0]*3*x[1]**2
        return grad
    
    betas = [(0.9, 0.999), (0.8, 0.9), (0.95, 0.99)]
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(15, 5))
    
    for i, (beta1, beta2) in enumerate(betas):
        optimizer = Adam(
            learning_rate=0.01,
            beta1=beta1,
            beta2=beta2,
            max_iterations=1000,
            verbose=(i == 0)
        )
        
        result, info = optimizer.minimize(beale,
                                        initial_point=np.array([1.0, 1.0]),
                                        gradient=beale_grad)
        
        plt.subplot(1, 3, i+1)
        plt.semilogy(range(len(info['history']['f_x'])), 
                    info['history']['f_x'], color=colors[i], linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Function Value (log scale)')
        plt.title(f'β₁={beta1}, β₂={beta2}')
        plt.grid(True, alpha=0.3)
        
        print(f"\nβ₁={beta1}, β₂={beta2}:")
        print(f"Final function value: {info['final_function_value']:.6f}")
        print(f"Converged: {info['converged']}")
    
    plt.suptitle('Hyperparameter Sensitivity in Adam', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def example_bias_correction():
    """
    Example 3: Effect of bias correction
    """
    print("=" * 60)
    print("EXAMPLE 3: Bias Correction Effect")
    print("=" * 60)
    
    # Simple quadratic
    def quadratic(x):
        return x[0]**2 + x[1]**2
    
    def quadratic_grad(x):
        return 2 * x
    
    # Adam with and without bias correction
    adam_opt = Adam(learning_rate=0.1, max_iterations=50, verbose=False)
    result, info = adam_opt.minimize(quadratic,
                                   initial_point=np.array([5.0, 5.0]),
                                   gradient=quadratic_grad)
    
    # Manually compute moments without bias correction
    m_raw = []
    v_raw = []
    m_val = np.zeros(2)
    v_val = np.zeros(2)
    x_val = np.array([5.0, 5.0])
    for k in range(1, 51):
        grad = quadratic_grad(x_val)
        m_val = 0.9 * m_val + 0.1 * grad
        v_val = 0.999 * v_val + 0.01 * grad**2
        m_raw.append(np.mean(m_val))
        v_raw.append(np.mean(v_val))
        x_val = x_val - 0.1 * m_val / (np.sqrt(v_val) + 1e-8)
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(info['history']['first_moment'])), 
            info['history']['first_moment'], 'b-', linewidth=2, label='With Bias Correction')
    plt.plot(range(len(m_raw)), m_raw, 'r--', linewidth=2, label='Without Bias Correction')
    plt.xlabel('Iteration')
    plt.ylabel('Mean First Moment')
    plt.title('First Moment Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(info['history']['second_moment'])), 
            info['history']['second_moment'], 'b-', linewidth=2, label='With Bias Correction')
    plt.plot(range(len(v_raw)), v_raw, 'r--', linewidth=2, label='Without Bias Correction')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Second Moment')
    plt.title('Second Moment Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Bias Correction in Adam', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("Bias correction helps stabilize moments in early iterations")


if __name__ == "__main__":
    # Run examples
    example_adam_vs_others()
    example_hyperparameter_sensitivity()
    example_bias_correction()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("="*60)
    print("1. Adam combines Momentum and RMSprop for robust optimization")
    print("2. Bias correction is important for early iterations")
    print("3. Default hyperparameters (β₁=0.9, β₂=0.999) work well in practice")
    print("4. Often the best choice for deep learning and large-scale problems")
    print("5. Computationally efficient and low memory requirements")
    print("6. Robust to noisy gradients and non-stationary objectives")
