# Momentum Optimizer

## Mathematical Background
Momentum accelerates gradient descent by accumulating a velocity vector in directions of persistent reduction in the objective function. This helps smooth out oscillations and can significantly speed up convergence, especially on ravine-shaped surfaces.

### Update Rule (Classical)
$$
v_{k+1} = \beta v_k - \alpha \nabla f(x_k) \\
x_{k+1} = x_k + v_{k+1}
$$

### Nesterov Accelerated Gradient (NAG)
$$
v_{k+1} = \beta v_k - \alpha \nabla f(x_k + \beta v_k) \\
x_{k+1} = x_k + v_{k+1}
$$

- $x_k$: current point
- $v_k$: velocity
- $\alpha$: learning rate
- $\beta$: momentum parameter (typically 0.9)

## Convergence
- Faster than vanilla GD on ill-conditioned problems
- NAG can provide even faster convergence

## Applications
- Deep learning (standard optimizer for neural networks)
- Convex and non-convex optimization

## References and Further Reading
- Y. Nesterov, "A Method for Unconstrained Convex Minimization Problem with the Rate of Convergence O(1/k^2)," *Doklady AN USSR*, 1983. [[PDF](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=46990&option_lang=eng)]
- Sutskever, Ilya, et al. "On the importance of initialization and momentum in deep learning." *ICML*, 2013. [[PDF](https://proceedings.mlr.press/v28/sutskever13.pdf)]
- [Stanford CS231n: Optimization Algorithms](https://cs231n.github.io/neural-networks-3/#sgd)
