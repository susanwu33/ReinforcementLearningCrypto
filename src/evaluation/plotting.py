import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_crisis_frequency(logs_dict, out_dir: Path):
    """
    Plots the frequency of "crashes" or extreme drawdowns 
    based on the action logs from different populations.
    """
    plt.figure(figsize=(10, 6))
    for name, log_df in logs_dict.items():
        if "regime" in log_df.columns:
            # crisis defined as regime == crash
            crashes = (log_df["regime"] == "crash").rolling(24).sum()
            plt.plot(crashes.values, label=f"{name} Crash Frequency (24h roll)", alpha=0.7)
        else:
            # proxy for crisis: heavily negative portfolio return
            net_r = log_df["net_port_r"] if "net_port_r" in log_df.columns else log_df["port_r"]
            crashes = (net_r < -0.05).astype(int).rolling(24).sum()
            plt.plot(crashes.values, label=f"{name} Crisis Events (<-5% ret/24h)", alpha=0.7)
            
    plt.title("Crisis Frequency Over Time by Trader Population")
    plt.xlabel("Hours (Timestep)")
    plt.ylabel("Number of Crisis Events (24h Rolling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "crisis_frequency.png")
    plt.close()

def plot_market_efficiency(logs_dict, out_dir: Path):
    """
    Proxies market efficiency by looking at Sharpe Ratio or 
    Volatility over time as strategy execution progresses.
    """
    plt.figure(figsize=(10, 6))
    for name, log_df in logs_dict.items():
        net_r = log_df["net_port_r"] if "net_port_r" in log_df.columns else log_df["port_r"]
        # Rolling sharpe ratio
        roll_mean = net_r.rolling(24*7).mean()
        roll_std = net_r.rolling(24*7).std() + 1e-9
        sharpe = (roll_mean / roll_std) * np.sqrt(24*365)
        
        plt.plot(sharpe.values, label=f"{name} Market Efficiency (Sharpe)", alpha=0.7)
        
    plt.title("Proxy for Market Efficiency (Rolling 7d Sharpe)")
    plt.xlabel("Hours (Timestep)")
    plt.ylabel("Annualized Sharpe Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "market_efficiency.png")
    plt.close()

def plot_stablecoin_stability(logs_dict, out_dir: Path):
    # Depending on the dataset, if stablecoins aren't present directly, 
    # we can use portfolio volatility as a proxy for systemic stability.
    plt.figure(figsize=(10, 6))
    for name, log_df in logs_dict.items():
        if "vol" in log_df.columns:
            plt.plot(log_df["vol"].rolling(24).mean().values, label=f"{name} 24h systemic volatility", alpha=0.7)
            
    plt.title("Systemic Stability Prototype (Portfolio/Market Volatility)")
    plt.xlabel("Hours (Timestep)")
    plt.ylabel("Rolling 24h Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "stablecoin_stability.png")
    plt.close()

def generate_all_plots(logs_dict, out_dir: str):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    plot_crisis_frequency(logs_dict, p)
    plot_market_efficiency(logs_dict, p)
    plot_stablecoin_stability(logs_dict, p)
    print(f"Saved plots to {p}")
