"""
Stochastic Gradient Descent (SGD)

Mathematical Background:
========================
SGD is a variant of gradient descent that uses a randomly selected subset (mini-batch) 
of the training data to compute the gradient at each iteration. This makes it much more 
efficient for large datasets.

Algorithm:
x_{k+1} = x_k - α ∇f_i(x_k)

Where:
- x_k: current point
- α: learning rate
- ∇f_i(x_k): gradient of function f_i (single sample or mini-batch)
- i: randomly selected sample/batch index

Key Differences from Batch GD:
- Uses subset of data → faster iterations
- Noisy gradient estimates → can escape local minima
- Requires learning rate scheduling for convergence

Applications:
- Deep learning (neural networks)
- Large-scale machine learning
- Online learning scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Union
import random


class StochasticGradientDescent:
    """
    Stochastic Gradient Descent optimizer with mini-batch support.
    
    Supports various learning rate schedules and mini-batch strategies
    for efficient optimization on large datasets.
    """
    
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 32,
                 max_epochs: int = 100, tolerance: float = 1e-6,
                 lr_schedule: str = 'constant', verbose: bool = False,
                 random_seed: int = 42):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Initial learning rate
            batch_size: Size of mini-batches
            max_epochs: Maximum number of epochs
            tolerance: Convergence tolerance
            lr_schedule: Learning rate schedule ('constant', 'decay', 'step')
            verbose: Whether to print progress
            random_seed: Random seed for reproducibility
        """
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.lr_schedule = lr_schedule
        self.verbose = verbose
        
        # Set random seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # History tracking
        self.history = {
            'x': [],
            'f_x': [],
            'gradient_norm': [],
            'learning_rate': [],
            'epoch': []
        }
    
    def update_learning_rate(self, epoch: int):
        """
        Update learning rate based on schedule.
        
        Args:
            epoch: Current epoch number
        """
        if self.lr_schedule == 'constant':
            pass  # Keep initial learning rate
        elif self.lr_schedule == 'decay':
            # Exponential decay: lr = lr_0 * exp(-decay_rate * epoch)
            decay_rate = 0.01
            self.learning_rate = self.initial_lr * np.exp(-decay_rate * epoch)
        elif self.lr_schedule == 'step':
            # Step decay: reduce by factor every few epochs
            if epoch > 0 and epoch % 20 == 0:
                self.learning_rate *= 0.5
        elif self.lr_schedule == 'inverse':
            # Inverse scaling: lr = lr_0 / (1 + decay_rate * epoch)
            decay_rate = 0.01
            self.learning_rate = self.initial_lr / (1 + decay_rate * epoch)
    
    def create_mini_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create mini-batches from dataset.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            List of (X_batch, y_batch) tuples
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        mini_batches = []
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            mini_batches.append((X_batch, y_batch))
        
        return mini_batches
    
    def minimize_ml(self, X: np.ndarray, y: np.ndarray, 
                   loss_function: Callable, gradient_function: Callable,
                   initial_params: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Minimize a machine learning objective using SGD.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            loss_function: Loss function f(params, X_batch, y_batch)
            gradient_function: Gradient function ∇f(params, X_batch, y_batch)
            initial_params: Initial parameter values
            
        Returns:
            Tuple of (optimal_params, optimization_info)
        """
        params = np.array(initial_params, dtype=float)
        n_samples = X.shape[0]
        
        # Clear history
        self.history = {key: [] for key in self.history.keys()}
        
        if self.verbose:
            print(f"Starting SGD optimization")
            print(f"Dataset size: {n_samples} samples")
            print(f"Batch size: {self.batch_size}")
            print(f"Initial loss: {loss_function(params, X, y):.6f}")
            print("-" * 50)
        
        for epoch in range(self.max_epochs):
            # Update learning rate
            self.update_learning_rate(epoch)
            
            # Create mini-batches
            mini_batches = self.create_mini_batches(X, y)
            
            epoch_loss = 0
            epoch_grad_norm = 0
            
            for batch_idx, (X_batch, y_batch) in enumerate(mini_batches):
                # Compute gradient on mini-batch
                grad = gradient_function(params, X_batch, y_batch)
                grad_norm = np.linalg.norm(grad)
                
                # SGD update
                params = params - self.learning_rate * grad
                
                # Accumulate metrics
                batch_loss = loss_function(params, X_batch, y_batch)
                epoch_loss += batch_loss * len(X_batch)
                epoch_grad_norm += grad_norm
            
            # Average metrics over epoch
            epoch_loss /= n_samples
            epoch_grad_norm /= len(mini_batches)
            
            # Store history
            self.history['x'].append(params.copy())
            self.history['f_x'].append(epoch_loss)
            self.history['gradient_norm'].append(epoch_grad_norm)
            self.history['learning_rate'].append(self.learning_rate)
            self.history['epoch'].append(epoch)
            
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss = {epoch_loss:.6f}, "
                      f"||∇f|| = {epoch_grad_norm:.6f}, lr = {self.learning_rate:.6f}")
            
            # Convergence check
            if epoch_grad_norm < self.tolerance:
                if self.verbose:
                    print(f"Converged after {epoch} epochs")
                break
        
        # Final evaluation
        final_loss = loss_function(params, X, y)
        
        optimization_info = {
            'epochs': len(self.history['x']),
            'final_loss': final_loss,
            'final_gradient_norm': epoch_grad_norm,
            'converged': epoch_grad_norm < self.tolerance,
            'history': self.history
        }
        
        if self.verbose:
            print("-" * 50)
            print(f"Optimization completed!")
            print(f"Final parameters: {params}")
            print(f"Final loss: {final_loss:.6f}")
            print(f"Final gradient norm: {epoch_grad_norm:.6f}")
        
        return params, optimization_info
    
    def plot_convergence(self, title: str = "SGD Convergence"):
        """
        Plot the convergence history.
        
        Args:
            title: Title for the plot
        """
        if not self.history['f_x']:
            print("No optimization history to plot. Run minimize_ml() first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = self.history['epoch']
        
        # Loss function
        ax1.plot(epochs, self.history['f_x'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Gradient norm
        ax2.semilogy(epochs, self.history['gradient_norm'], 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Gradient Norm')
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
        
        # Parameter evolution (for 2D problems)
        if len(self.history['x']) > 0 and len(self.history['x'][0]) == 2:
            param1 = [x[0] for x in self.history['x']]
            param2 = [x[1] for x in self.history['x']]
            ax4.plot(param1, param2, 'mo-', markersize=3, linewidth=1, alpha=0.7)
            ax4.plot(param1[0], param2[0], 'go', markersize=8, label='Start')
            ax4.plot(param1[-1], param2[-1], 'ro', markersize=8, label='End')
            ax4.set_xlabel('Parameter 1')
            ax4.set_ylabel('Parameter 2')
            ax4.set_title('Parameter Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Parameter visualization\navailable for 2D problems only', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Parameter Evolution')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
