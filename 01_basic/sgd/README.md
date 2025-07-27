# Stochastic Gradient Descent (SGD)

## Mathematical Background
SGD is a variant of gradient descent that uses a randomly selected subset (mini-batch) of the training data to compute the gradient at each iteration. This makes it much more efficient for large datasets.

### Update Rule
$$
x_{k+1} = x_k - \alpha \nabla f_i(x_k)
$$

- $x_k$: current point
- $\alpha$: learning rate
- $\nabla f_i(x_k)$: gradient using a single sample or mini-batch

## Differences from Batch GD
- Uses subset of data → faster iterations
- Noisy gradient estimates → can escape local minima
- Requires learning rate scheduling for convergence

## Applications
- Deep learning (neural networks)
- Large-scale machine learning
- Online learning scenarios

## References and Further Reading
- L. Bottou, "Stochastic Gradient Descent Tricks," in *Neural Networks: Tricks of the Trade*, Springer, 2012. [[PDF](https://leon.bottou.org/papers/bottou-2012)]
- R. Johnson and T. Zhang, "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction," *NIPS*, 2013. [[PDF](https://proceedings.neurips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf)]
- [Stanford CS231n: Optimization Algorithms](https://cs231n.github.io/neural-networks-3/#sgd)
