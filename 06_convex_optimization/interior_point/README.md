# Interior Point Methods

## Mathematical Background
Interior point methods are a class of algorithms for solving large-scale convex optimization problems, especially linear and quadratic programs. They approach the solution from the interior of the feasible region by iteratively minimizing a barrier-augmented objective.

### Update Rule (Barrier Method)
Minimize:
$$
f(x) + \mu \sum_{i} -\log(-g_i(x))
$$
where $g_i(x) < 0$ are inequality constraints and $\mu > 0$ is the barrier parameter.

## Relevance
- Highly efficient for large, sparse convex problems.
- Polynomial-time complexity for linear and convex quadratic programming.
- Foundation for modern solvers (e.g., CVXOPT, MOSEK).

## Applications
- Linear programming (LP)
- Quadratic programming (QP)
- Semidefinite programming (SDP)

## References and Further Reading
- S. Boyd and L. Vandenberghe, "Convex Optimization," Cambridge University Press, 2004. [[Book](https://web.stanford.edu/~boyd/cvxbook/)]
- S. Wright, "Primal-Dual Interior-Point Methods," SIAM, 1997. [[Book](https://epubs.siam.org/doi/book/10.1137/1.9781611970791)]
- [Wikipedia: Interior Point Method](https://en.wikipedia.org/wiki/Interior-point_method)
