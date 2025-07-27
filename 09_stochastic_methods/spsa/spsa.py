import numpy as np
import matplotlib.pyplot as plt

class SPSA:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.
    """
    def __init__(self, a=0.01, c=0.01, A=100, alpha=0.602, gamma=0.101):
        """
        Initializes the SPSA optimizer with recommended gain parameters.
        - a, c: Scaling factors for the gain sequences.
        - A: Stability constant.
        - alpha, gamma: Decay rate exponents.
        """
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.history = {'loss': []}

    def minimize(self, objective_func, theta_init, n_iter=1000):
        """
        Minimizes the objective function using SPSA.
        """
        theta = np.array(theta_init, dtype=float)
        p = len(theta)

        for k in range(n_iter):
            # Update gain sequences
            ak = self.a / (k + 1 + self.A)**self.alpha
            ck = self.c / (k + 1)**self.gamma

            # 1. Generate random perturbation vector (Bernoulli distribution)
            delta = np.random.randint(0, 2, p) * 2 - 1

            # 2. Evaluate objective function at two perturbed points
            theta_plus = theta + ck * delta
            theta_minus = theta - ck * delta
            y_plus = objective_func(theta_plus)
            y_minus = objective_func(theta_minus)

            # 3. Approximate the gradient
            ghat = (y_plus - y_minus) / (2 * ck * delta)

            # 4. Update theta
            theta = theta - ak * ghat

            self.history['loss'].append(objective_func(theta))
        
        return {'solution': theta, 'loss': self.history['loss'][-1], 'history': self.history}

if __name__ == '__main__':
    # Define a simple quadratic objective function to test SPSA
    def quadratic_loss(theta):
        return np.sum((theta - np.array([1, 2, 3]))**2)

    # Initial guess
    initial_theta = np.array([0.0, 0.0, 0.0])

    # Initialize and run the optimizer
    spsa_optimizer = SPSA(a=0.1, c=0.1)
    result = spsa_optimizer.minimize(quadratic_loss, initial_theta, n_iter=2000)

    print("--- SPSA Optimization Results ---")
    print(f"Optimal solution (theta): {result['solution']}")
    print(f"Final loss: {result['loss']:.6f}")

    # Plot the convergence
    plt.figure(figsize=(10, 6))
    plt.plot(result['history']['loss'])
    plt.title('SPSA Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
