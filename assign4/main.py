import os
import importlib.util
import pandas as pd
from src.data_loader import build_outcome_matrix


def run_preflight_checks(asset_files: dict) -> None:
    """
    Validate required runtime dependency and input data files before simulation.
    """
    if importlib.util.find_spec("scipy") is None:
        raise ModuleNotFoundError(
            "Missing required package: scipy. Install it with: "
            "python3 -m pip install scipy"
        )

    missing_files = [path for path in asset_files.values() if not os.path.isfile(path)]
    if missing_files:
        missing_list = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing required data file(s). Expected files:\n"
            f"{missing_list}\n"
            "Please place all required CSVs under the data/ directory."
        )

def main():
    # 1. Asset file paths
    asset_files = {
        "BTC": "data/BTC-USD_DataHr.csv",
        "ETH": "data/ETH-USD_DataHr.csv",
        "SOL": "data/SOL-USD_DataHr.csv",
        "BNB": "data/BNB-USD_DataHr.csv",
        "USDC": "data/USDC-USD_DataHr.csv",
    }

    # Experiment parameters
    T_WINDOW = 500         # Time horizon per experiment run
    N_REPLICATIONS = 100   # Number of independent replications
    T_TOTAL = 1000         # Total rows loaded for random window sampling (must exceed T_WINDOW)

    print("--- Preflight: Checking dependencies and data files ---")
    run_preflight_checks(asset_files)

    from src.simulation import run_all_experiments
    from src.plots import (
        plot_cumulative_regret,
        plot_posterior_distributions,
        plot_information_ratio,
    )
    from src.table import make_final_regret_table

    print(f"--- Step 1: Loading data (sampling buffer T={T_TOTAL}) ---")
    # Load more than 500 rows so simulation.sample_time_window has slicing headroom.
    Y_full = build_outcome_matrix(asset_files, T=T_TOTAL)
    
    print(f"Raw outcome matrix shape: {Y_full.shape}")
    print("First 5 rows:")
    print(Y_full.head())

    print(f"\n--- Step 2: Running experiments ({N_REPLICATIONS} random sampled paths) ---")
    # run_all_experiments internally calls sample_time_window.
    results = run_all_experiments(
        Y_full=Y_full,
        n_replications=N_REPLICATIONS,
        T=T_WINDOW
    )

    print("\n--- Step 3: Generating summary table ---")
    # With randomized window sampling, deterministic algorithms (e.g., UCB1) also have nonzero SE.
    table = make_final_regret_table(results)
    print(table)
    table.to_csv("final_regret_table.csv", index=False)

    print("\n--- Step 4: Generating figures ---")
    
    print("Generating Figure 1: Cumulative Regret (with 95% CI)...")
    plot_cumulative_regret(
        results,
        save_path="figure1_cumulative_regret.png",
    )

    print("Generating Figure 2: Posterior Distributions (Thompson Sampling Snapshots)...")
    # Pass arm names to show them in the legend.
    arm_names = list(Y_full.columns)
    plot_posterior_distributions(
        results,
        arm_names=arm_names,
        save_prefix="figure2_posterior"
    )

    print("Generating Figure 3: Information Ratio Trajectory...")
    plot_information_ratio(
        results,
        save_path="figure3_information_ratio.png",
    )

    print("\nAll tasks completed. Please check the generated CSV and PNG files.")

if __name__ == "__main__":
    main()