# Gradient Descent

## Mathematical Background
Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. At each step, it moves in the direction of the negative gradient of the function at the current point.

### Update Rule
$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

- $x_k$: current point
- $\alpha$: learning rate (step size)
- $\nabla f(x_k)$: gradient at $x_k$

## Convergence
- For convex functions, GD converges to the global minimum.
- Convergence rate: $O(1/k)$ for constant step size.
- Step size $\alpha$ must be chosen carefully.

## Applications
- Machine learning model training (linear/logistic regression)
- Neural networks (base for more advanced optimizers)
- General unconstrained minimization

## References and Further Reading
- L. Bottou, "Stochastic Gradient Descent Tricks," in *Neural Networks: Tricks of the Trade*, Springer, 2012. [[PDF](https://leon.bottou.org/papers/bottou-2012)]
- S. Boyd and L. Vandenberghe, "Convex Optimization," Cambridge University Press, 2004. [[Book](https://web.stanford.edu/~boyd/cvxbook/)]
- [Wikipedia: Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
