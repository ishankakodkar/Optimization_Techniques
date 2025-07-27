import numpy as np
from typing import Callable

class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization for continuous optimization problems.
    """
    def __init__(self, num_particles: int = 30, inertia: float = 0.7, cognitive: float = 1.5, social: float = 1.5, max_iter: int = 100, verbose: bool = False):
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_iter = max_iter
        self.verbose = verbose
        self.history = {
            'best_fitness': [],
            'mean_fitness': []
        }

    def minimize(self, objective: Callable, bounds: tuple) -> dict:
        dim = len(bounds[0])
        X = np.random.uniform(bounds[0], bounds[1], (self.num_particles, dim))
        V = np.zeros_like(X)
        pbest = X.copy()
        pbest_val = np.array([objective(x) for x in X])
        gbest = pbest[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        for it in range(self.max_iter):
            r1, r2 = np.random.rand(self.num_particles, dim), np.random.rand(self.num_particles, dim)
            V = self.inertia * V + self.cognitive * r1 * (pbest - X) + self.social * r2 * (gbest - X)
            X = np.clip(X + V, bounds[0], bounds[1])
            vals = np.array([objective(x) for x in X])
            improved = vals < pbest_val
            pbest[improved] = X[improved]
            pbest_val[improved] = vals[improved]
            if np.min(pbest_val) < gbest_val:
                gbest = pbest[np.argmin(pbest_val)]
                gbest_val = np.min(pbest_val)
            self.history['best_fitness'].append(gbest_val)
            self.history['mean_fitness'].append(np.mean(vals))
            if self.verbose and it % 10 == 0:
                print(f"Iter {it}: Best={gbest_val:.4f}, Mean={np.mean(vals):.4f}")
        return {
            'solution': gbest,
            'fitness': gbest_val,
            'history': self.history
        }
