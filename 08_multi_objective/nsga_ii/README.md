# NSGA-II (Non-dominated Sorting Genetic Algorithm II)

## Mathematical Background
NSGA-II is a powerful multi-objective optimization algorithm that finds a set of Pareto optimal solutions. Instead of a single optimal solution, it seeks to find a trade-off surface (Pareto front) of solutions where improving one objective necessarily degrades another.

### Core Concepts
1.  **Non-dominated Sorting**: The population is sorted into several fronts. The first front contains non-dominated solutions (no other solution is better in all objectives). The second front is non-dominated with respect to the rest of the population, and so on.
2.  **Crowding Distance**: Within each front, a crowding distance is calculated for each solution. This metric measures the density of solutions surrounding a particular point. It is used as a tie-breaker to favor solutions in less crowded regions, promoting diversity in the Pareto front.
3.  **Elitism**: The algorithm uses elitism by combining the parent and offspring populations and selecting the best individuals to survive to the next generation, ensuring that good solutions are not lost.

## Relevance
-   The de facto standard for many multi-objective optimization problems.
-   Effectively balances convergence to the true Pareto front with maintaining a diverse set of solutions.
-   Its principles are foundational to many modern multi-objective evolutionary algorithms (MOEAs).

## Applications
-   **Engineering Design**: Finding the best trade-off between cost, strength, and weight.
-   **Finance**: Balancing risk and return in portfolio optimization.
-   **Logistics**: Minimizing both cost and delivery time.
-   **Machine Learning**: Tuning models with conflicting objectives, like accuracy and inference speed.

## References and Further Reading
-   K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, 2002. [[Paper](https://ieeexplore.ieee.org/document/996017)]
-   [Wikipedia: NSGA-II](https://en.wikipedia.org/wiki/NSGA-II)
