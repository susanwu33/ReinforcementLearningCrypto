import numpy as np
from scipy.stats import beta as beta_dist

class BetaPosterior:
    """
    Beta posterior for one Bernoulli arm.
    """
    def __init__(self, a: float = 3, b: float = 3):
        self.a = a
        self.b = b
        # 用于 UCB1 这种频率派算法，记录纯粹的观测值
        self.n_success = 0
        self.n_total = 0

    def update(self, y: int) -> None:
        """
        Beta-Bernoulli conjugate update.
        """
        if y == 1:
            self.a += 1
            self.n_success += 1
        else:
            self.b += 1
        
        self.n_total += 1

    def sample(self) -> float:
        """Draw one sample from the posterior. Used by Thompson Sampling."""
        return np.random.beta(self.a, self.b)

    def mean(self) -> float:
        """Posterior mean (Bayesian)."""
        return self.a / (self.a + self.b)

    def empirical_mean(self) -> float:
        """Empirical mean (Frequentist). Used by UCB1."""
        if self.n_total == 0:
            return 0.0
        return self.n_success / self.n_total

    def quantile(self, q: float) -> float:
        """Posterior quantile. Used by Bayes-UCB."""
        return beta_dist.ppf(q, self.a, self.b)