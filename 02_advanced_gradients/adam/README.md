# Adam Optimizer

## Mathematical Background
Adam (Adaptive Moment Estimation) combines the benefits of Momentum and RMSprop. It maintains exponentially decaying averages of past gradients (first moment) and squared gradients (second moment), and includes bias correction for both.

### Update Rule
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla f(x_t) \\
v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla f(x_t))^2 \\
\hat{m}_t = m_t / (1-\beta_1^t) \\
\hat{v}_t = v_t / (1-\beta_2^t) \\
x_{t+1} = x_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

- $x_t$: current parameter vector
- $\alpha$: learning rate
- $\beta_1$: decay rate for first moment (typically 0.9)
- $\beta_2$: decay rate for second moment (typically 0.999)
- $\epsilon$: small constant to avoid division by zero

## Convergence
- Fast and robust for noisy and sparse gradients
- Works well for non-stationary objectives

## Applications
- Deep learning (standard optimizer for neural networks)
- Large-scale optimization

## References and Further Reading
- D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *International Conference on Learning Representations (ICLR)*, 2015. [[PDF](https://arxiv.org/abs/1412.6980)]
- [Stanford CS231n: Optimization Algorithms](https://cs231n.github.io/neural-networks-3/#adam)
- [Distill.pub: Why Adam Works](https://distill.pub/2017/momentum/)
