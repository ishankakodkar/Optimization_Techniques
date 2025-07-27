# RMSprop Optimizer

## Mathematical Background
RMSprop (Root Mean Square Propagation) is an adaptive learning rate method that maintains an exponentially decaying average of squared gradients. It addresses AdaGrad's rapid learning rate decay, enabling better long-term training.

### Update Rule
$$
E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) (\nabla f(x_t))^2 \\
x_{t+1} = x_t - \frac{\alpha}{\sqrt{E[g^2]_t} + \epsilon} \nabla f(x_t)
$$

- $x_t$: current parameter vector
- $\alpha$: learning rate
- $\rho$: decay rate (typically 0.9)
- $E[g^2]_t$: running average of squared gradients
- $\epsilon$: small constant to avoid division by zero

## Convergence
- Fixes AdaGrad's vanishing learning rate issue
- Effective for non-stationary objectives

## Applications
- Deep learning (RNNs, CNNs)
- Large-scale optimization

## References and Further Reading
- T. Tieleman and G. Hinton, "Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude," *COURSERA: Neural Networks for Machine Learning*, 2012. [[PDF](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)]
- [Stanford CS231n: Optimization Algorithms](https://cs231n.github.io/neural-networks-3/#rmsprop)
