# Projected Gradient Descent

## Mathematical Background
Projected Gradient Descent (PGD) is used for constrained convex optimization. After a standard gradient step, the result is projected back onto the feasible set.

### Update Rule
$$
x_{k+1} = \Pi_{C}(x_k - \alpha \nabla f(x_k))
$$
where $\Pi_{C}$ is the projection operator onto the convex set $C$.

## Relevance
- Solves problems with simple convex constraints (e.g., box constraints, simplex constraints).
- Used in machine learning, signal processing, and robust optimization.

## Applications
- Constrained regression (e.g., non-negative least squares)
- Sparse recovery
- Adversarial machine learning (generating adversarial examples)

## References and Further Reading
- S. Boyd and L. Vandenberghe, "Convex Optimization," Cambridge University Press, 2004. [[Book](https://web.stanford.edu/~boyd/cvxbook/)]
- D. Bertsekas, "Convex Optimization Algorithms," Athena Scientific, 2015. [[Book](https://web.mit.edu/dimitrib/www/ConvexOptimizationAlgorithms.html)]
- [Wikipedia: Projected Gradient Descent](https://en.wikipedia.org/wiki/Projected_gradient_descent)
