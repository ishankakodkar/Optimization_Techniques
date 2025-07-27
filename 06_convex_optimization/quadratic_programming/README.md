# Quadratic Programming (QP)

## Mathematical Background
Quadratic programming involves minimizing a convex quadratic objective subject to linear constraints. The general form is:
$$
\min_x \ \frac{1}{2} x^T Q x + c^T x \quad \text{subject to} \ Ax \leq b
$$
where $Q$ is positive semidefinite.

## Relevance
- Many machine learning and signal processing problems reduce to QP (e.g., SVMs, portfolio optimization).
- Efficiently solvable using interior point or active set methods.

## Applications
- Support Vector Machines (SVM)
- Portfolio optimization
- Control systems

## References and Further Reading
- S. Boyd and L. Vandenberghe, "Convex Optimization," Cambridge University Press, 2004. [[Book](https://web.stanford.edu/~boyd/cvxbook/)]
- J. Nocedal and S. Wright, "Numerical Optimization," Springer, 2006. [[Book](https://link.springer.com/book/10.1007/978-0-387-40065-5)]
- [Wikipedia: Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming)
