# Penalty Method

## Mathematical Background
The penalty method converts a constrained optimization problem into a series of unconstrained problems by adding a penalty term to the objective function for constraint violations.

### Algorithm
For a problem `min f(x) s.t. g(x) <= 0`, the penalized objective is:
$$ \phi(x, \rho) = f(x) + \rho \sum_i \max(0, g_i(x))^2 $$
where $\rho > 0$ is a penalty parameter. The algorithm involves solving a sequence of unconstrained problems for increasing values of $\rho$.

## Relevance
- A simple and intuitive way to handle constraints.
- Can be used with any unconstrained optimization algorithm.
- Can lead to numerical issues if $\rho$ is too large.

## Applications
- Structural optimization
- Trajectory planning

## References and Further Reading
- J. Nocedal and S. Wright, "Numerical Optimization," Springer, 2006. (Chapter 17)
- [Wikipedia: Penalty Method](https://en.wikipedia.org/wiki/Penalty_method)
