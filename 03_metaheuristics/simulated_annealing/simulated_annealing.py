import numpy as np
import random
from typing import Callable

class SimulatedAnnealing:
    """
    Simulated Annealing for combinatorial and continuous optimization.
    """
    def __init__(self, initial_temp: float = 1.0, cooling_rate: float = 0.99, max_iter: int = 1000, verbose: bool = False):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.history = {
            'best_fitness': [],
            'temp': []
        }

    def minimize(self, objective: Callable, neighbor: Callable, initial_solution: np.ndarray) -> dict:
        x = np.array(initial_solution, dtype=float)
        f_x = objective(x)
        best_x = x.copy()
        best_f = f_x
        temp = self.initial_temp
        for k in range(self.max_iter):
            x_new = neighbor(x)
            f_new = objective(x_new)
            if f_new < f_x or random.random() < np.exp(-(f_new - f_x) / (temp + 1e-12)):
                x = x_new
                f_x = f_new
                if f_new < best_f:
                    best_x = x_new.copy()
                    best_f = f_new
            self.history['best_fitness'].append(best_f)
            self.history['temp'].append(temp)
            temp *= self.cooling_rate
            if self.verbose and k % 100 == 0:
                print(f"Iter {k}: Best={best_f:.4f}, Temp={temp:.4f}")
        return {
            'solution': best_x,
            'fitness': best_f,
            'history': self.history
        }
