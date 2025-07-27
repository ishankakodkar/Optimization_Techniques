# Simulated Annealing (SA)

## Mathematical Background
Simulated Annealing is a probabilistic metaheuristic inspired by the annealing process in metallurgy. It explores the solution space by accepting worse solutions with a probability that decreases over time (temperature).

### Algorithm Steps
1. Initialize with a random solution and temperature $T$
2. Repeat:
   - Propose a new solution $x'$ near current $x$
   - If $f(x') < f(x)$, accept $x'$
   - Else, accept $x'$ with probability $\exp(-(f(x')-f(x))/T)$
   - Decrease $T$ according to a schedule
3. Stop when $T$ is low or after max iterations

## Relevance
- Good for escaping local minima in non-convex and combinatorial problems
- Simple and widely applicable

## Applications
- Traveling Salesman Problem (TSP)
- VLSI design
- Scheduling

## References and Further Reading
- S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by Simulated Annealing," *Science*, 1983. [[PDF](https://www.cs.brandeis.edu/~marshall/courses/cs159/SimulatedAnnealing.pdf)]
- E. Aarts and J. Korst, "Simulated Annealing and Boltzmann Machines," Wiley, 1988. [[Book](https://www.wiley.com/en-us/Simulated+Annealing+and+Boltzmann+Machines-p-9780471921462)]
- [Wikipedia: Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
