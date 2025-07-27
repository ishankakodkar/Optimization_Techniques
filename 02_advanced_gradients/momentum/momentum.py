import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
import time

class MomentumOptimizer:
    """
    Implements both classical and Nesterov momentum-based optimizers.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, max_iterations: int = 1000, tolerance: float = 1e-6, nesterov: bool = False, verbose: bool = False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.nesterov = nesterov
        self.verbose = verbose
        self.history = {
            'x': [],
            'f_x': [],
            'gradient_norm': [],
            'step_size': []
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

    def minimize(self, objective: Callable, initial_point: np.ndarray, gradient: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
        x = np.array(initial_point, dtype=float)
        v = np.zeros_like(x)
        self.history = {key: [] for key in self.history.keys()}
        if self.verbose:
            print(f"Starting {'Nesterov' if self.nesterov else 'Classical'} Momentum optimization")
            print(f"Learning rate: {self.learning_rate}, Momentum: {self.momentum}")
            print(f"Initial point: {x}")
            print(f"Initial function value: {objective(x):.6f}")
            print("-" * 50)
        start_time = time.time()
        for iteration in range(self.max_iterations):
            if self.nesterov:
                grad = gradient(x + self.momentum * v) if gradient is not None else self.numerical_gradient(lambda z: objective(z + self.momentum * v), x)
            else:
                grad = gradient(x) if gradient is not None else self.numerical_gradient(objective, x)
            grad_norm = np.linalg.norm(grad)
            f_x = objective(x)
            self.history['x'].append(x.copy())
            self.history['f_x'].append(f_x)
            self.history['gradient_norm'].append(grad_norm)
            self.history['step_size'].append(self.learning_rate)
            if self.verbose and iteration % 100 == 0:
                print(f"Iter {iteration:4d}: f(x) = {f_x:.6f}, ||∇f|| = {grad_norm:.6f}")
            if grad_norm < self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration} iterations")
                break
            v = self.momentum * v - self.learning_rate * grad
            x = x + v
        end_time = time.time()
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

    def plot_convergence(self, title: str = "Momentum Convergence"):
        if not self.history['f_x']:
            print("No optimization history to plot. Run minimize() first.")
            return
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        iterations = range(len(self.history['f_x']))
        ax1.plot(iterations, self.history['f_x'], 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function Value')
        ax1.set_title('Objective Function')
        ax1.grid(True, alpha=0.3)
        ax2.semilogy(iterations, self.history['gradient_norm'], 'r-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Gradient Norm')
        ax2.grid(True, alpha=0.3)
        ax3.plot(iterations, self.history['step_size'], 'g-', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Step Size')
        ax3.set_title('Learning Rate')
        ax3.grid(True, alpha=0.3)
        if len(self.history['x']) > 0 and len(self.history['x'][0]) == 2:
            x_path = [x[0] for x in self.history['x']]
            y_path = [x[1] for x in self.history['x']]
            ax4.plot(x_path, y_path, 'mo-', markersize=3, linewidth=1, alpha=0.7)
            ax4.plot(x_path[0], y_path[0], 'go', markersize=8, label='Start')
            ax4.plot(x_path[-1], y_path[-1], 'ro', markersize=8, label='End')
            ax4.set_xlabel('x₁')
            ax4.set_ylabel('x₂')
            ax4.set_title('Optimization Path')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Path visualization\navailable for 2D problems only', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Optimization Path')
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
