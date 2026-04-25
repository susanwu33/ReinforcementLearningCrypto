import pandas as pd
from src.data_loader import build_outcome_matrix
from src.simulation import run_all_experiments
from src.plots import (
    plot_cumulative_regret,
    plot_posterior_distributions,
    plot_information_ratio,
)
from src.table import make_final_regret_table

def main():
    # 1. 资产文件路径
    asset_files = {
        "BTC": "data/BTC-USD_DataHr.csv",
        "ETH": "data/ETH-USD_DataHr.csv",
        "SOL": "data/SOL-USD_DataHr.csv",
        "BNB": "data/BNB-USD_DataHr.csv",
        "USDC": "data/USDC-USD_DataHr.csv",
    }

    # 实验参数
    T_WINDOW = 500         # 每次实验持续的时间步
    N_REPLICATIONS = 100   # 独立重复次数
    T_TOTAL = 1000         # 从原始数据中提取的总长度（应大于 T_WINDOW 以允许随机采样）

    print(f"--- 步骤 1: 加载数据 (预留采样空间 T={T_TOTAL}) ---")
    # 修改：这里获取比 500 长的矩阵，以便 simulation 里的 sample_time_window 有切片余量
    Y_full = build_outcome_matrix(asset_files, T=T_TOTAL)
    
    print(f"原始数据矩阵形状: {Y_full.shape}")
    print("前 5 行数据示例:")
    print(Y_full.head())

    print(f"\n--- 步骤 2: 运行实验 ({N_REPLICATIONS} 次随机采样路径) ---")
    # run_all_experiments 现在内部会调用 sample_time_window
    results = run_all_experiments(
        Y_full=Y_full,
        n_replications=N_REPLICATIONS,
        T=T_WINDOW
    )

    print("\n--- 步骤 3: 生成结果汇总表 ---")
    # 由于引入了随机采样，确定性算法（UCB1等）现在也会有非零的 SE
    table = make_final_regret_table(results)
    print(table)
    table.to_csv("final_regret_table.csv", index=False)

    print("\n--- 步骤 4: 绘制图表 ---")
    
    print("正在生成 Figure 1: Cumulative Regret (含 95% CI)...")
    plot_cumulative_regret(
        results,
        save_path="figure1_cumulative_regret.png",
    )

    print("正在生成 Figure 2: Posterior Distributions (Thompson Sampling Snapshots)...")
    # 传入臂名称以便在图例中显示
    arm_names = list(Y_full.columns)
    plot_posterior_distributions(
        results,
        arm_names=arm_names,
        save_prefix="figure2_posterior"
    )

    print("正在生成 Figure 3: Information Ratio Trajectory...")
    plot_information_ratio(
        results,
        save_path="figure3_information_ratio.png",
    )

    print("\n所有任务已完成。请检查生成的 CSV 和 PNG 文件。")

if __name__ == "__main__":
    main()