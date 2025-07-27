# Particle Swarm Optimization (PSO)

## Mathematical Background
PSO is a population-based stochastic optimization technique inspired by the social behavior of birds flocking or fish schooling. Each particle adjusts its position based on its own experience and that of its neighbors.

### Update Rule
$$
v_{i}^{k+1} = \omega v_{i}^k + c_1 r_1 (p_{i}^k - x_{i}^k) + c_2 r_2 (g^k - x_{i}^k) \\
x_{i}^{k+1} = x_{i}^k + v_{i}^{k+1}
$$
where $x_i$ is the position, $v_i$ is the velocity, $p_i$ is the personal best, $g$ is the global best, $\omega$ is inertia, $c_1$, $c_2$ are acceleration coefficients, and $r_1$, $r_2$ are random numbers in [0,1].

## Relevance
- Suitable for continuous, non-convex, and multi-modal problems
- Simple to implement and parallelize

## Applications
- Neural network training
- Hyperparameter tuning
- Engineering design

## References and Further Reading
- J. Kennedy and R. Eberhart, "Particle Swarm Optimization," *Proceedings of ICNN'95 - International Conference on Neural Networks*, 1995. [[PDF](https://ieeexplore.ieee.org/document/488968)]
- M. Clerc, "Particle Swarm Optimization," ISTE Press, 2010. [[Book](https://www.elsevier.com/books/particle-swarm-optimization/clerc/978-1-84821-120-8)]
- [Wikipedia: Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization)
