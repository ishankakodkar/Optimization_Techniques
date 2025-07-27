# Proximal Gradient Method

## Mathematical Background
The proximal gradient method extends gradient descent to composite convex functions of the form $f(x) = g(x) + h(x)$, where $g$ is smooth and $h$ is possibly non-smooth but has an easy-to-compute proximal operator.

### Update Rule
$$
x_{k+1} = \text{prox}_{\alpha h}(x_k - \alpha \nabla g(x_k))
$$
where $\text{prox}_{\alpha h}(v) = \arg\min_x \left( h(x) + \frac{1}{2\alpha} \|x - v\|^2 \right)$

## Relevance
- Efficient for problems with non-smooth regularization (e.g., LASSO, group lasso).
- Enables splitting of smooth and non-smooth parts.

## Applications
- Sparse regression (LASSO)
- Signal/image denoising (total variation)
- Compressed sensing

## References and Further Reading
- N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends in Optimization, 2014. [[PDF](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)]
- S. Boyd and L. Vandenberghe, "Convex Optimization," Cambridge University Press, 2004. [[Book](https://web.stanford.edu/~boyd/cvxbook/)]
- [Wikipedia: Proximal Gradient Method](https://en.wikipedia.org/wiki/Proximal_gradient_method)
