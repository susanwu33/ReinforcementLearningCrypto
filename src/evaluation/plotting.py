import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_crisis_frequency(logs_dict, out_dir: Path, prefix: str = ""):
    """
    Per-trader crisis stress from portfolio outcomes (net_port_r), so curves differ by policy.

    Note: ``regime`` / raw market ``vol`` in the log are properties of the price path at each
    timestep; every rollout walks the same test index, so plotting those per trader duplicates
    the same series six times. Optional reference: one dashed line for market crash regime.
    """
    plt.figure(figsize=(10, 6))
    # Hourly log portfolio return threshold for a "stressed" step (tune to your asset scale).
    stress_thresh = -0.02

    for name, log_df in logs_dict.items():
        net_r = log_df["net_port_r"] if "net_port_r" in log_df.columns else log_df["port_r"]
        stressed_hours = (net_r < stress_thresh).astype(int).rolling(24, min_periods=1).sum()
        plt.plot(
            stressed_hours.values,
            label=f"{name} portfolio stress hours (24h roll, r<{stress_thresh})",
            alpha=0.85,
        )

    first = next(iter(logs_dict.values()), None)
    if first is not None and "regime" in first.columns:
        mkt = (first["regime"] == "crash").astype(int).rolling(24, min_periods=1).sum()
        plt.plot(
            mkt.values,
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            label="market: crash regime hours (24h roll, same for all)",
            alpha=0.9,
        )

    plt.title("Crisis Frequency Over Time by Trader Population")
    plt.xlabel("Hours (Timestep)")
    plt.ylabel("Stressed portfolio hours (24h rolling count)")
    plt.legend()
    plt.tight_layout()
    filename = f"{prefix}_" + "crisis_frequency.png" if prefix else "crisis_frequency.png"
    plt.savefig(out_dir / filename)
    plt.close()

def plot_market_efficiency(logs_dict, out_dir: Path, prefix: str = ""):
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
    filename = f"{prefix}_" + "market_efficiency.png" if prefix else "market_efficiency.png"
    plt.savefig(out_dir / filename)
    plt.close()

def plot_stablecoin_stability(logs_dict, out_dir: Path, prefix: str = ""):
    """
    Rolling volatility of realized net portfolio returns per strategy.

    The ``vol`` column in env logs is the asset's precomputed rolling vol from the dataframe,
    identical across traders on the same test split; use per-step net returns instead.
    """
    plt.figure(figsize=(10, 6))
    for name, log_df in logs_dict.items():
        net_r = log_df["net_port_r"] if "net_port_r" in log_df.columns else log_df["port_r"]
        roll_vol = net_r.rolling(24, min_periods=1).std()
        plt.plot(roll_vol.values, label=f"{name} rolling σ(net port r), 24h", alpha=0.85)

    plt.title("Strategy return volatility (not asset vol from CSV)")
    plt.xlabel("Hours (Timestep)")
    plt.ylabel("Rolling 24h σ of net portfolio log return")
    plt.legend()
    plt.tight_layout()
    filename = f"{prefix}_" + "stablecoin_stability.png" if prefix else "stablecoin_stability.png"
    plt.savefig(out_dir / filename)
    plt.close()

def generate_all_plots(logs_dict, out_dir: str, prefix: str = ""):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    plot_crisis_frequency(logs_dict, p, prefix)
    plot_market_efficiency(logs_dict, p, prefix)
    plot_stablecoin_stability(logs_dict, p, prefix)
    print(f"Saved plots to {p}")
