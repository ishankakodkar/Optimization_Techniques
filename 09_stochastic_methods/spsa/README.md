# SPSA (Simultaneous Perturbation Stochastic Approximation)

## Mathematical Background
SPSA is a stochastic optimization algorithm that is highly effective for high-dimensional problems, especially when the gradient is not available or is too expensive to compute. It requires only two function evaluations per iteration to approximate the gradient, regardless of the problem's dimension.

### Gradient Approximation
Instead of calculating the full gradient (which requires $2n$ function evaluations for dimension $n$ using finite differences), SPSA approximates it using a random perturbation:

1.  Generate a random perturbation vector $\Delta_k$, where each component is typically drawn from a Bernoulli distribution (e.g., -1, 1).
2.  Evaluate the function at two points: $f(\theta_k + c_k \Delta_k)$ and $f(\theta_k - c_k \Delta_k)$.
3.  Approximate the gradient $g_k(\theta_k)$ as:
    $$ \hat{g}_k(\theta_k) = \frac{f(\theta_k + c_k \Delta_k) - f(\theta_k - c_k \Delta_k)}{2 c_k} \Delta_k^{-1} $$

### Update Rule
The parameter update follows a standard stochastic approximation form:
$$ \theta_{k+1} = \theta_k - a_k \hat{g}_k(\theta_k) $$
where $a_k$ and $c_k$ are gain sequences that must satisfy certain conditions to ensure convergence.

## Relevance
-   Extremely efficient for high-dimensional optimization problems where gradient computation is a bottleneck.
-   Its performance is independent of the problem dimension, making it scalable.
-   Robust to noise in the function evaluations.

## Applications
-   Real-time control and system identification.
-   Training large-scale machine learning models.
-   Simulation-based optimization.

## References and Further Reading
-   J. C. Spall, "A Stochastic Approximation Algorithm for Large-Dimensional Systems in the Kiefer-Wolfowitz Setting," *Proceedings of the 30th IEEE Conference on Decision and Control*, 1991. [[Paper](https://ieeexplore.ieee.org/document/261323/)]
-   J. C. Spall, "An Overview of the Simultaneous Perturbation Method for Efficient Optimization," *Johns Hopkins APL Technical Digest*, 1998. [[PDF](https://www.jhuapl.edu/Content/techdigest/pdf/V19-N04/19-04-Spall.pdf)]
