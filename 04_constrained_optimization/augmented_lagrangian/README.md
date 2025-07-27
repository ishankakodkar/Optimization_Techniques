# Augmented Lagrangian Method (Method of Multipliers)

## Mathematical Background
The Augmented Lagrangian method combines the ideas of the penalty method and Lagrange multipliers to solve constrained optimization problems. It adds a penalty term to the Lagrangian, which improves numerical conditioning and often leads to better convergence.

### Algorithm
For a problem `min f(x) s.t. g(x) = 0`, the augmented Lagrangian is:
$$ L_\rho(x, \lambda) = f(x) + \lambda^T g(x) + \frac{\rho}{2} \|g(x)\|^2 $$

The algorithm alternates between:
1. Minimizing $L_\rho(x, \lambda_k)$ with respect to $x$ to find $x_{k+1}$.
2. Updating the Lagrange multipliers: $\lambda_{k+1} = \lambda_k + \rho g(x_{k+1})$.
3. Increasing the penalty parameter $\rho$ if convergence is slow.

## Relevance
- More robust than the standard penalty method.
- Avoids the ill-conditioning associated with very large penalty parameters.
- Forms the basis for the Alternating Direction Method of Multipliers (ADMM).

## Applications
- Large-scale nonlinear programming.
- Optimal control problems.
- Machine learning (e.g., training constrained models).

## References and Further Reading
- J. Nocedal and S. Wright, "Numerical Optimization," Springer, 2006. (Chapter 17)
- D. P. Bertsekas, "Nonlinear Programming," Athena Scientific, 1999.
- [Wikipedia: Augmented Lagrangian method](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method)
