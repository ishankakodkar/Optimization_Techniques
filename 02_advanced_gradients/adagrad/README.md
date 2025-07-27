# AdaGrad Optimizer

## Mathematical Background
AdaGrad is an adaptive learning rate optimization algorithm that adjusts the learning rate for each parameter based on the sum of the squares of past gradients. This is especially useful for dealing with sparse data and parameters with different frequencies of updates.

### Update Rule
$$
g_{t+1} = g_t + (\nabla f(x_t))^2 \\
x_{t+1} = x_t - \frac{\alpha}{\sqrt{g_{t+1}} + \epsilon} \nabla f(x_t)
$$

- $x_t$: current parameter vector
- $\alpha$: initial learning rate
- $g_t$: sum of squares of gradients (element-wise)
- $\epsilon$: small constant to avoid division by zero

## Convergence
- Works well for sparse data
- Learning rate decreases over time (can become too small)

## Applications
- Natural language processing (sparse features)
- Training shallow neural networks

## References and Further Reading
- J. Duchi, E. Hazan, and Y. Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization," *Journal of Machine Learning Research*, 2011. [[PDF](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)]
- [Stanford CS231n: Optimization Algorithms](https://cs231n.github.io/neural-networks-3/#adagrad)
