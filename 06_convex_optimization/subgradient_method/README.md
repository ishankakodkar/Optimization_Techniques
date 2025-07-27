# Subgradient Method

## Mathematical Background
The subgradient method is a generalization of gradient descent for convex but non-differentiable functions. Instead of the gradient, it uses a subgradient, which always exists for convex functions.

### Update Rule
$$
x_{k+1} = x_k - \alpha_k g_k
$$
where $g_k$ is a subgradient of $f$ at $x_k$ and $\alpha_k$ is the step size.

## Relevance
- Enables optimization of non-smooth convex functions (e.g., $\ell_1$-regularized problems, hinge loss in SVMs).
- Simplicity and broad applicability, but slower convergence than gradient descent for smooth problems.

## Applications
- LASSO regression
- Support Vector Machines (SVM)
- Robust optimization

## References and Further Reading
- S. Boyd and L. Vandenberghe, "Convex Optimization," Cambridge University Press, 2004. [[Book](https://web.stanford.edu/~boyd/cvxbook/)]
- D. Bertsekas, "Convex Optimization Algorithms," Athena Scientific, 2015. [[Book](https://web.mit.edu/dimitrib/www/ConvexOptimizationAlgorithms.html)]
- [Wikipedia: Subgradient Method](https://en.wikipedia.org/wiki/Subgradient_method)
