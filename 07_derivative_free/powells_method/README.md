# Powell's Method

## Mathematical Background
Powell's method is a derivative-free optimization algorithm that finds a local minimum of a function. It belongs to the class of conjugate direction methods. The core idea is to perform a sequence of line searches along a set of directions that are updated to become mutually conjugate with respect to the Hessian of the function.

### Algorithm Steps
1.  Initialize with a set of $n$ linearly independent directions (e.g., the standard basis vectors).
2.  Iteratively perform line searches along each direction to find the minimum.
3.  After a full cycle, create a new direction vector from the initial point to the final point of the cycle.
4.  Replace one of the old directions with this new direction.
5.  Repeat until convergence.

By doing this, the set of directions progressively aligns with the principal axes of the quadratic form of the function near the minimum, leading to faster convergence.

## Relevance
-   Does not require derivatives, making it suitable for black-box functions.
-   Generally more efficient than simple coordinate descent due to its use of conjugate directions.
-   It is designed for unconstrained, continuous optimization.

## Applications
-   Calibration of physical models.
-   Parameter estimation in complex systems.

## References and Further Reading
-   M. J. D. Powell, "An efficient method for finding the minimum of a function of several variables without calculating derivatives," *The Computer Journal*, 1964. [[Paper](https://academic.oup.com/comjnl/article/7/2/155/339994)]
-   J. Nocedal and S. Wright, "Numerical Optimization," Springer, 2006. (Chapter 9 discusses derivative-free methods).
