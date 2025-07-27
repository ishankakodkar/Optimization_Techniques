# Lagrange Multipliers & KKT Conditions

## Mathematical Background
Lagrange multipliers are a strategy for finding the local maxima and minima of a function subject to equality constraints. The Karush-Kuhn-Tucker (KKT) conditions extend this to inequality constraints.

### Equality Constraints
For a problem `min f(x) s.t. g(x) = 0`, the Lagrangian is:
$$ L(x, \lambda) = f(x) + \lambda g(x) $$
At an optimal point, $\nabla L(x, \lambda) = 0$, which means $\nabla f(x) = -\lambda \nabla g(x)$.

### KKT Conditions (Inequality Constraints)
For `min f(x) s.t. g(x) <= 0`, the KKT conditions for optimality are:
1. **Stationarity**: $\nabla f(x) + \mu \nabla g(x) = 0$
2. **Primal Feasibility**: $g(x) \le 0$
3. **Dual Feasibility**: $\mu \ge 0$
4. **Complementary Slackness**: $\mu g(x) = 0$

## Relevance
- Forms the theoretical foundation for many constrained optimization algorithms.
- Provides a check for optimality.

## Applications
- Support Vector Machines (SVMs)
- Optimal control
- Resource allocation

## References and Further Reading
- S. Boyd and L. Vandenberghe, "Convex Optimization," Cambridge University Press, 2004. (Chapter 5)
- [Wikipedia: Lagrange Multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier)
- [Wikipedia: Karush–Kuhn–Tucker conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)
