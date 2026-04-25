import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

def plot_cumulative_regret(results, save_path=None):
    plt.figure(figsize=(10, 6))
    for algorithm, result in results.items():
        regrets = result["regrets"]
        mean_regret = regrets.mean(axis=0)
        # 计算标准误 SE 并生成 95% 置信区间
        se = regrets.std(axis=0, ddof=1) / np.sqrt(regrets.shape[0])
        t = np.arange(1, len(mean_regret) + 1)

        plt.plot(t, mean_regret, label=algorithm.upper())
        plt.fill_between(t, mean_regret - 1.96 * se, mean_regret + 1.96 * se, alpha=0.15)

    plt.xlabel("Time (8h periods)")
    plt.ylabel("Cumulative Regret")
    plt.title("Figure 1: Cumulative Regret vs Time (Mean ± 95% CI over 100 Replications)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()

def plot_posterior_distributions(results, arm_names, save_prefix=None):
    snapshots = results["thompson"]["snapshots"]
    x = np.linspace(0, 1, 500)

    for t, params in snapshots.items():
        plt.figure(figsize=(10, 5))
        # 这里的 params 结构是 [(a1, b1), (a2, b2), ...]
        for i, (a, b) in enumerate(params):
            y = beta_dist.pdf(x, a, b)
            plt.plot(x, y, label=f"{arm_names[i]} (α={a}, β={b})")
        
        plt.title(f"Figure 2: Posterior Distributions at t={t}")
        plt.xlabel("Success Probability θ")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_prefix: plt.savefig(f"{save_prefix}_t{t}.png")
        plt.show()

def plot_information_ratio(results, save_path=None):
    """
    Figure 3: Approximate information ratio numerator for Thompson Sampling.

    Note:
    We plot (E[Δ_t])^2, which corresponds to the numerator
    of the information ratio in Eq. 3.1 (Ghavamzadeh et al.).
    """

    gammas = results["thompson"]["gammas"]

    mean_gamma = gammas.mean(axis=0)
    se = gammas.std(axis=0, ddof=1) / np.sqrt(gammas.shape[0])

    lower = mean_gamma - 1.96 * se
    upper = mean_gamma + 1.96 * se

    t = np.arange(1, len(mean_gamma) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(t, mean_gamma, label="TS squared expected regret $(E[\\Delta_t])^2$")
    plt.fill_between(t, lower, upper, alpha=0.2)

    plt.xlabel("Time t")
    plt.ylabel(r"$(E[\Delta_t])^2$")

    plt.title("Figure 3: Approximate Information Ratio Numerator for Thompson Sampling")

    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()