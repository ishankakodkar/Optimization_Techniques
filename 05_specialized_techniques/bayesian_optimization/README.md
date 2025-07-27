# Bayesian Optimization

## Mathematical Background
Bayesian Optimization is a sequential, model-based approach for finding the maximum of an expensive, black-box function $f(x)$. It works by building a probabilistic model (a surrogate) of the objective function and using an acquisition function to decide where to sample next.

### Core Components
1.  **Surrogate Model**: Typically a Gaussian Process (GP) that models the distribution over functions $P(f|\text{data})$. The GP provides a mean prediction and uncertainty for any point $x$.
2.  **Acquisition Function**: A function that guides the search for the optimum. It balances exploration (sampling in areas of high uncertainty) and exploitation (sampling where the surrogate predicts a high objective value). Common choices include:
    -   **Expected Improvement (EI)**: $EI(x) = E[\max(0, f(x) - f(x^+))]$, where $x^+$ is the best point seen so far.
    -   **Upper Confidence Bound (UCB)**: $UCB(x) = \mu(x) + \kappa \sigma(x)$.

## Relevance
- Highly effective for problems where function evaluations are expensive (e.g., training a deep neural network, running a complex physics simulation).
- Does not require gradient information.
- Provides a principled way to trade off exploration and exploitation.

## Applications
- Hyperparameter tuning for machine learning models.
- A/B testing and experimental design.
- Robotics and reinforcement learning.
- Drug discovery and material science.

## References and Further Reading
- J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," *Advances in Neural Information Processing Systems (NIPS)*, 2012. [[PDF](https://papers.nips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf)]
- B. Shahriari et al., "Taking the Human Out of the Loop: A Review of Bayesian Optimization," *Proceedings of the IEEE*, 2016. [[PDF](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)]
- [Distill.pub: A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
