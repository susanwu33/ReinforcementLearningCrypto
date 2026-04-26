import numpy as np

def choose_arm(algorithm: str, posteriors: list, counts: np.ndarray, t: int) -> int:
    """
    Select one arm according to the chosen bandit algorithm.
    Note: t is the current time step (1-indexed).
    """
    K = len(posteriors)

    # 1. Greedy: based on posterior mean
    if algorithm == "greedy":
        values = [p.mean() for p in posteriors]
        return int(np.argmax(values))

    # 2. Thompson Sampling
    if algorithm == "thompson":
        samples = [p.sample() for p in posteriors]
        return int(np.argmax(samples))

    # 3. Bayes-UCB: use the 1 - 1/t quantile
    if algorithm == "bayes_ucb":
        # Avoid q=0 when t=1.
        q = 1 - 1.0 / max(t, 2)
        values = [p.quantile(q) for p in posteriors]
        return int(np.argmax(values))

    # 4. UCB1: standard frequentist implementation
    if algorithm == "ucb1":
        values = []
        for k in range(K):
            if counts[k] == 0:
                values.append(np.inf) # Ensure each arm is tried at least once.
            else:
                # Use observed empirical mean, not a prior-influenced Bayesian mean.
                mu_hat = posteriors[k].empirical_mean()
                bonus = np.sqrt(2 * np.log(t) / counts[k])
                values.append(mu_hat + bonus)
        return int(np.argmax(values))

    raise ValueError(f"Unknown algorithm: {algorithm}")