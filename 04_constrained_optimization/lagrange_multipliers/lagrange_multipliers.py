import numpy as np

def solve_kkt_example(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> dict:
    """
    Solves a simple QP with equality constraints using the KKT conditions directly.
    min 1/2 x'Qx + c'x  s.t. Ax = b

    The KKT system is:
    [ Q  A' ] [ x ] = [ -c ]
    [ A  0  ] [ v ]   [  b ]
    """
    n = Q.shape[0]
    m = A.shape[0]
    
    # Assemble the KKT matrix
    kkt_matrix = np.block([
        [Q, A.T],
        [A, np.zeros((m, m))]
    ])
    
    # Assemble the right-hand side vector
    rhs = np.concatenate([-c, b])
    
    try:
        # Solve the linear system
        solution = np.linalg.solve(kkt_matrix, rhs)
        x = solution[:n]
        nu = solution[n:] # Lagrange multipliers
        
        return {
            'solution': x,
            'multipliers': nu,
            'objective_value': 0.5 * x.T @ Q @ x + c.T @ x
        }
    except np.linalg.LinAlgError:
        return {
            'error': 'KKT matrix is singular. Problem may be ill-posed.'
        }

if __name__ == '__main__':
    # Example: min x1^2 + x2^2 s.t. x1 + x2 = 1
    Q = 2 * np.array([[1, 0], [0, 1]])
    c = np.array([0, 0])
    A = np.array([[1, 1]])
    b = np.array([1])
    
    result = solve_kkt_example(Q, c, A, b)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Optimal solution (x): {result['solution']}")
        print(f"Lagrange multiplier (nu): {result['multipliers']}")
        print(f"Objective value: {result['objective_value']:.4f}")
