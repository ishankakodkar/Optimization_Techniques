# Newton's Method

## Mathematical Background
Newton's method is a second-order optimization algorithm that uses both the gradient (first derivative) and the Hessian matrix (second derivatives) to find the minimum of a function. It has quadratic convergence near the optimum.

### Update Rule
$$
x_{k+1} = x_k - H^{-1}(x_k) \nabla f(x_k)
$$

- $x_k$: current point
- $\nabla f(x_k)$: gradient
- $H(x_k)$: Hessian matrix

## Convergence
- Quadratic convergence rate: $O(1/k^2)$ near optimum
- Requires positive definite Hessian
- Fast near optimum, may be unstable far from it

## Applications
- Small to medium-sized optimization problems
- Fine-tuning near optimum

## References and Further Reading
- S. Boyd and L. Vandenberghe, "Convex Optimization," Cambridge University Press, 2004. [[Book](https://web.stanford.edu/~boyd/cvxbook/)]
- J. Nocedal and S. Wright, "Numerical Optimization," Springer, 2006. [[Book](https://link.springer.com/book/10.1007/978-0-387-40065-5)]
- [Wikipedia: Newton's Method in Optimization](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)
