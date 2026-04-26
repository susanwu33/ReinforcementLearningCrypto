import numpy as np
from src.posterior import BetaPosterior
from src.algorithms import choose_arm

def estimate_ts_action_probs(posteriors, n_samples: int = 2000) -> np.ndarray:
    """
    Vectorized Monte Carlo estimation of Thompson Sampling action probabilities.
    """
    K = len(posteriors)
    # Generate all samples at once: shape (n_samples, K).
    # Draw n_samples from each posterior distribution.
    all_samples = np.zeros((n_samples, K))
    
    for k, p in enumerate(posteriors):
        # Use numpy.random.beta with size for vectorized sampling.
        all_samples[:, k] = np.random.beta(p.a, p.b, size=n_samples)  # Posterior parameters are a and b.
    
    # Find the winning arm index for each sample row.
    winners = np.argmax(all_samples, axis=1)
    
    # Count winner frequencies per arm.
    # bincount counts occurrences for indices 0..K-1.
    counts = np.bincount(winners, minlength=K)
    
    return counts / n_samples

def sample_time_window(Y_full, T=500):
    """
    Provide a different data path for each replication via random time windows.
    """
    if len(Y_full) <= T:
        return Y_full.copy()
    
    # Randomly choose a start index.
    start = np.random.randint(0, len(Y_full) - T + 1)
    return Y_full.iloc[start : start + T].reset_index(drop=True)

def run_one_simulation(
    Y_window, 
    algorithm: str, 
    prior_alpha: float = 3, 
    prior_beta: float = 3, 
    snapshot_times=None
):
    if snapshot_times is None:
        snapshot_times = [50, 100, 200, 500]

    T, K = Y_window.shape
    
    posteriors = [BetaPosterior(prior_alpha, prior_beta) for _ in range(K)]
    counts = np.zeros(K)
    
    cumulative_regret = []
    regret = 0
    gamma_values = []
    posterior_snapshots = {}

    for t in range(1, T + 1):
        y_t = Y_window.iloc[t - 1].values
        
        # Select arm.
        chosen_arm = choose_arm(algorithm, posteriors, counts, t)
        
        # At each time step, regret is computed against the best realized reward
        # among all arms in the observed data window.
        regret_increment = y_t.max() - y_t[chosen_arm]
        regret += regret_increment
        cumulative_regret.append(regret)

        # Update posterior and counts.
        posteriors[chosen_arm].update(y_t[chosen_arm])
        counts[chosen_arm] += 1

        # Information-ratio numerator approximation for Thompson Sampling:
        # We estimate E[Delta_t] under posterior-induced action probabilities
        # and store (E[Delta_t])^2 as an empirical approximation of Eq. 3.1 numerator.
        if algorithm == "thompson":
            ts_probs = estimate_ts_action_probs(posteriors)
            current_means = np.array([p.mean() for p in posteriors])
            delta_t = np.sum(ts_probs * (np.max(current_means) - current_means))
            gamma_values.append(delta_t ** 2)

        if t in snapshot_times:
            posterior_snapshots[t] = [(p.a, p.b) for p in posteriors]

    return {
        "regret": np.array(cumulative_regret),
        "snapshots": posterior_snapshots,
        "gamma": np.array(gamma_values),
    }

def run_all_experiments(Y_full, n_replications: int = 100, T=500):
    """
    Run all experiments with randomized window sampling.
    """
    algorithms = ["ucb1", "bayes_ucb", "thompson", "greedy"]
    results = {}

    for algo in algorithms:
        print(f"  Running {algo}...")
        all_regrets = []
        all_gammas = []
        last_snapshots = None

        for i in range(n_replications):
            # Sample a different 500-step window for each replication.
            Y_rep = sample_time_window(Y_full, T=T)
            
            sim = run_one_simulation(Y_rep, algo)
            all_regrets.append(sim["regret"])
            
            if algo == "thompson":
                all_gammas.append(sim["gamma"])
                last_snapshots = sim["snapshots"]

        results[algo] = {
            "regrets": np.array(all_regrets),
            "snapshots": last_snapshots,
        }
        if algo == "thompson":
            results[algo]["gammas"] = np.array(all_gammas)

    return results