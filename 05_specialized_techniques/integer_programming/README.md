# Integer Programming

## Mathematical Background
Integer Programming (IP) deals with optimization problems where some or all of the variables are restricted to be integers. This restriction makes the problems significantly harder to solve than their continuous (Linear Programming) counterparts, often rendering them NP-hard.

### Problem Formulation
A typical Mixed-Integer Linear Program (MILP) is formulated as:
$$ \min_{x, y} \ c^T x + d^T y \\
\text{s.t.} \ Ax + By \le b \\
\qquad x \in \mathbb{Z}^n, y \in \mathbb{R}^m $$

### Core Algorithm: Branch and Bound
Most modern solvers use a method called **Branch and Bound**:
1.  **Relaxation**: Solve the problem as a continuous LP (ignoring integer constraints). If the solution is integer-feasible, it is optimal.
2.  **Branching**: If a variable $x_i$ is fractional (e.g., $x_i = 2.5$), create two new subproblems: one with the constraint $x_i \le 2$ and another with $x_i \ge 3$. This creates a tree of problems.
3.  **Bounding**: Keep track of the best integer solution found so far (the incumbent). If a subproblem's relaxed solution is worse than the incumbent, that entire branch of the tree can be pruned (fathomed).

## Relevance
- Essential for problems involving discrete choices, counts, or logical constraints.
- Widely used in operations research, logistics, and finance.

## Applications
- **Supply Chain Management**: Facility location, vehicle routing.
- **Scheduling**: Crew scheduling, job-shop scheduling.
- **Finance**: Portfolio optimization with minimum transaction sizes.
- **Telecommunications**: Network design.

## References and Further Reading
- L. A. Wolsey, "Integer Programming," Wiley, 1998.
- H. P. Williams, "Model Building in Mathematical Programming," Wiley, 2013.
- [Wikipedia: Integer Programming](https://en.wikipedia.org/wiki/Integer_programming)
- [Wikipedia: Branch and Bound](https://en.wikipedia.org/wiki/Branch_and_bound)
