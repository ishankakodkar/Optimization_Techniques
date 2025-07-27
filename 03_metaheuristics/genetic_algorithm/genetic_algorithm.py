import numpy as np
import random
from typing import Callable, Tuple

class GeneticAlgorithm:
    """
    Simple Genetic Algorithm for combinatorial and continuous optimization.
    """
    def __init__(self, population_size: int = 50, crossover_rate: float = 0.8, mutation_rate: float = 0.1, generations: int = 100, verbose: bool = False):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.verbose = verbose
        self.history = {
            'best_fitness': [],
            'mean_fitness': []
        }

    def minimize(self, objective: Callable, bounds: Tuple[np.ndarray, np.ndarray]) -> dict:
        dim = len(bounds[0])
        pop = np.random.uniform(bounds[0], bounds[1], (self.population_size, dim))
        fitness = np.array([objective(ind) for ind in pop])
        for gen in range(self.generations):
            # Store the best individual (elitism)
            best_idx = np.argmin(fitness)
            elite = pop[best_idx].copy()

            # Selection (Tournament Selection)
            selected_parents = []
            for _ in range(self.population_size):
                tournament_indices = np.random.choice(self.population_size, 2, replace=False)
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                selected_parents.append(pop[winner_idx])
            selected_parents = np.array(selected_parents)

            # Crossover
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_parents[i], selected_parents[(i + 1) % self.population_size]
                if random.random() < self.crossover_rate:
                    alpha = np.random.rand(dim)
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = (1 - alpha) * parent1 + alpha * parent2
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                offspring.extend([child1, child2])
            offspring = np.array(offspring)

            # Mutation
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
            mutation_values = np.random.uniform(-(bounds[1] - bounds[0]) * 0.1, (bounds[1] - bounds[0]) * 0.1, offspring.shape)
            offspring += mutation_mask * mutation_values
            offspring = np.clip(offspring, bounds[0], bounds[1])

            # Replace worst individual with the elite one from the previous generation
            offspring[np.argmax(fitness)] = elite

            # Evaluate new population
            pop = offspring
            fitness = np.array([objective(ind) for ind in pop])
            self.history['best_fitness'].append(np.min(fitness))
            self.history['mean_fitness'].append(np.mean(fitness))
            if self.verbose and gen % 10 == 0:
                print(f"Gen {gen}: Best={np.min(fitness):.4f}, Mean={np.mean(fitness):.4f}")
        best_idx = np.argmin(fitness)
        return {
            'solution': pop[best_idx],
            'fitness': fitness[best_idx],
            'history': self.history
        }
