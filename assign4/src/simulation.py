import numpy as np
from src.posterior import BetaPosterior
from src.algorithms import choose_arm

def estimate_ts_action_probs(posteriors, n_samples: int = 2000) -> np.ndarray:
    """
    向量化蒙特卡洛模拟，大幅提升 Thompson Sampling 动作概率的估算速度。
    """
    K = len(posteriors)
    # 一次性生成所有 samples: 形状为 (n_samples, K)
    # 我们直接从每个 posterior 的分布里抽 n_samples 个点
    all_samples = np.zeros((n_samples, K))
    
    for k, p in enumerate(posteriors):
        # 利用 numpy.random.beta 的 size 参数进行向量化采样
        all_samples[:, k] = np.random.beta(p.a, p.b, size=n_samples) # 修正变量名为 a, b
    
    # 找到每一行（每一次采样）中值最大的 arm 索引
    winners = np.argmax(all_samples, axis=1)
    
    # 统计每个 arm 获胜的频率
    # bincount 会统计 0 到 K-1 每个索引出现的次数
    counts = np.bincount(winners, minlength=K)
    
    return counts / n_samples

def sample_time_window(Y_full, T=500):
    """
    通过随机时间窗口为每次 Replication 提供不同的数据路径。
    """
    if len(Y_full) <= T:
        return Y_full.copy()
    
    # 随机选择起始点
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
    
    # --- 修正 2: 根据题目公式 Regret(t) = Σ [r(Y_i(a*)) - r(Y_i(a_i))] ---
    # 计算当前窗口下表现最好的 arm (a*)
    window_means = Y_window.mean(axis=0).values
    best_arm_idx = np.argmax(window_means)
    
    posteriors = [BetaPosterior(prior_alpha, prior_beta) for _ in range(K)]
    counts = np.zeros(K)
    
    cumulative_regret = []
    regret = 0
    gamma_values = []
    posterior_snapshots = {}

    for t in range(1, T + 1):
        y_t = Y_window.iloc[t - 1].values
        
        # 选择臂
        chosen_arm = choose_arm(algorithm, posteriors, counts, t)
        
        # --- 修正 2: 使用 a* 的观测收益减去被选臂的观测收益 ---
        # 这里的 y_t[best_arm_idx] 即 r(Y_i(a*))
        regret += (y_t[best_arm_idx] - y_t[chosen_arm])
        cumulative_regret.append(regret)

        # 更新
        posteriors[chosen_arm].update(y_t[chosen_arm])
        counts[chosen_arm] += 1

        # Information Ratio (针对 TS)
        if algorithm == "thompson":
            ts_probs = estimate_ts_action_probs(posteriors)
            current_means = np.array([p.mean() for p in posteriors])
            delta_t = np.sum(ts_probs * (np.max(current_means) - current_means))
            gamma_values.append(delta_t)

        if t in snapshot_times:
            posterior_snapshots[t] = [(p.a, p.b) for p in posteriors]

    return {
        "regret": np.array(cumulative_regret),
        "snapshots": posterior_snapshots,
        "gamma": np.array(gamma_values),
    }

def run_all_experiments(Y_full, n_replications: int = 100, T=500):
    """
    运行实验，包含随机采样逻辑。
    """
    algorithms = ["ucb1", "bayes_ucb", "thompson", "greedy"]
    results = {}

    for algo in algorithms:
        print(f"  Running {algo}...")
        all_regrets = []
        all_gammas = []
        last_snapshots = None

        for i in range(n_replications):
            # --- 修正 1: 每次 replication 采样不同的 500 时间步窗口 ---
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