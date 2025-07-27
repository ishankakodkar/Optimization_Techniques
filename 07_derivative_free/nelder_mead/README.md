# Nelder-Mead Method

## Mathematical Background
The Nelder-Mead method, also known as the downhill simplex method, is a widely used derivative-free optimization algorithm. It works by maintaining a simplex, a geometric figure in n dimensions consisting of n+1 vertices. The algorithm iteratively modifies the simplex to find a minimum of the objective function.

### Core Operations
The algorithm uses a set of core operations to move and reshape the simplex:
1.  **Ordering**: Sort the vertices by their function values, from best ($x_1$) to worst ($x_{n+1}$).
2.  **Reflection**: Reflect the worst vertex through the centroid of the remaining vertices.
3.  **Expansion**: If the reflected point is better than the current best, expand it further in that direction.
4.  **Contraction**: If the reflected point is not an improvement, contract the simplex towards the best vertex.
5.  **Shrink**: If contraction fails, shrink the entire simplex towards the best vertex.

## Relevance
-   Extremely popular for problems where the derivative is unknown, noisy, or expensive to compute.
-   Does not require the function to be smooth.
-   Can sometimes get stuck in local minima, and convergence for high-dimensional problems can be slow.

## Applications
-   Parameter tuning in scientific models.
-   Image processing and registration.
-   Engineering design.

## References and Further Reading
-   J. A. Nelder and R. Mead, "A simplex method for function minimization," *The Computer Journal*, 1965. [[Paper](https://academic.oup.com/comjnl/article/7/4/308/346947)]
-   [Wikipedia: Nelder–Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
