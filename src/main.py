import argparse
import pandas as pd
from pathlib import Path
import json
import torch
import warnings
import os

# Enable MPS fallback for unsupported ops (like multinomial)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress gym warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.env.crypto_env import make_env
from src.agents.dqn import train_dqn, run_policy_dqn
from src.agents.reinforce import train_reinforce, run_policy_reinforce
from src.evaluation.shapley import calculate_shap_for_agent
from src.evaluation.plotting import generate_all_plots

def load_split(data_dir: Path, sym: str):
    train = pd.read_csv(data_dir / f"{sym}_train.csv", parse_dates=["datetime"])
    val   = pd.read_csv(data_dir / f"{sym}_val.csv",   parse_dates=["datetime"])
    test  = pd.read_csv(data_dir / f"{sym}_test.csv",  parse_dates=["datetime"])
    return train, val, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed", help="Directory with split CSVs")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save logs/models/plots")
    parser.add_argument("--sym", type=str, default="AAVE", help="Which crypto to train on")
    parser.add_argument("--trader_types", nargs="+", default=["rational", "manipulator", "retail"])
    parser.add_argument("--episodes", type=int, default=10000, help="Episodes for REINFORCE")
    parser.add_argument("--dqn_steps", type=int, default=10000, help="Steps for DQN")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {args.sym}...")
    train_df, val_df, test_df = load_split(data_dir, args.sym)

    device = "cpu"  # Using CPU for small 2-layer NN to avoid MPS fallback issues
    print(f"Using device: {device}")

    all_logs = {}
    metrics_summary = []

    for t_type in args.trader_types:
        print(f"\n=============================================")
        print(f"Training traders of type: {t_type}")
        print(f"=============================================")
        
        # Build environments
        env_train, obs_cols = make_env(args.sym, train_df, trader_type=t_type, random_start=True, horizon=168)
        env_val, _     = make_env(args.sym, val_df,   trader_type=t_type, random_start=False)
        env_test, _    = make_env(args.sym, test_df,  trader_type=t_type, random_start=False)

        # Append position column if env adds it
        feature_names = list(obs_cols)
        if env_train.add_position_to_obs:
            feature_names.append("position")

        # 1. Train REINFORCE (Policy Gradient)
        print(f"--- Training REINFORCE [{t_type}] ---")
        pg_net = train_reinforce(env_train, env_val, episodes=args.episodes, device=device)
        pg_test_log = run_policy_reinforce(env_test, pg_net, device=device)
        all_logs[f"{t_type}_pg"] = pg_test_log

        # 2. Train DQN
        print(f"--- Training DQN [{t_type}] ---")
        dqn_net = train_dqn(env_train, env_val, steps=args.dqn_steps, warmup=1000, device=device)
        dqn_test_log = run_policy_dqn(env_test, dqn_net, device=device)
        all_logs[f"{t_type}_dqn"] = dqn_test_log

        # 3. SHAP Values
        print(f"--- Calculating SHAP values for REINFORCE [{t_type}] ---")
        top3_pg, _ = calculate_shap_for_agent(pg_net, env_test, device, feature_names, n_background=500, n_test=50)
        
        print(f"--- Calculating SHAP values for DQN [{t_type}] ---")
        top3_dqn, _ = calculate_shap_for_agent(dqn_net, env_test, device, feature_names, n_background=500, n_test=50)

        # Summarize test results
        metrics_summary.append({
            "trader_type": t_type,
            "pg_return": pg_test_log["net_port_r"].sum(),
            "dqn_return": dqn_test_log["net_port_r"].sum(),
            "pg_turnover": pd.to_numeric(pg_test_log["delta_pos"]).mean(),
            "dqn_turnover": pd.to_numeric(dqn_test_log["delta_pos"]).mean(),
            "pg_top3_features": top3_pg,
            "dqn_top3_features": top3_dqn,
        })
        
        # Save logs
        pg_test_log.to_csv(out_dir / f"{args.sym}_{t_type}_pg_log.csv", index=False)
        dqn_test_log.to_csv(out_dir / f"{args.sym}_{t_type}_dqn_log.csv", index=False)

    # 4. Regulator Analysis Plots
    print("\nGenerating Macro Regulator Plots...")
    generate_all_plots(all_logs, out_dir=out_dir)

    # Save summary 
    summary_df = pd.DataFrame(metrics_summary)
    summary_df.to_csv(out_dir / f"{args.sym}_summary.csv", index=False)
    
    print("\n--- Final Evaluation Summary ---")
    print(summary_df.to_string())

if __name__ == "__main__":
    main()
