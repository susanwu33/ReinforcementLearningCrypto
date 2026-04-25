import numpy as np

def choose_arm(algorithm: str, posteriors: list, counts: np.ndarray, t: int) -> int:
    """
    Select one arm according to the chosen bandit algorithm.
    Note: t is the current time step (1-indexed).
    """
    K = len(posteriors)

    # 1. Greedy: 基于后验均值
    if algorithm == "greedy":
        values = [p.mean() for p in posteriors]
        return int(np.argmax(values))

    # 2. Thompson Sampling
    if algorithm == "thompson":
        samples = [p.sample() for p in posteriors]
        return int(np.argmax(samples))

    # 3. Bayes-UCB: 使用 1 - 1/t 分位数
    if algorithm == "bayes_ucb":
        # 避免 t=1 时 q=0
        q = 1 - 1.0 / max(t, 2)
        values = [p.quantile(q) for p in posteriors]
        return int(np.argmax(values))

    # 4. UCB1: 典型的频率派实现
    if algorithm == "ucb1":
        values = []
        for k in range(K):
            if counts[k] == 0:
                values.append(np.inf) # 确保每个臂至少被试一次
            else:
                # 使用观测到的频率均值，而非带先验的贝叶斯均值
                mu_hat = posteriors[k].empirical_mean()
                bonus = np.sqrt(2 * np.log(t) / counts[k])
                values.append(mu_hat + bonus)
        return int(np.argmax(values))

    raise ValueError(f"Unknown algorithm: {algorithm}")