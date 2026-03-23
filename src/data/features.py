import pandas as pd
import numpy as np
from pathlib import Path
import argparse

FILES = {
    "SUI": "SUI20947-USD_DataHr.csv",
    "AAVE": "AAVE-USD_DataHr.csv",
    "ETC": "ETC-USD_DataHr.csv",
    "BTC": "BTC-USD_DataHr.csv",
    "ES":  "ES=F_DataHr.csv",
    "GC":  "GC=F_DataHr.csv",
}

def read_market_csv(path: Path) -> pd.DataFrame:
    """
    Reads the course CSV with 2 header rows (group label + ticker),
    returns a dataframe indexed by Datetime with columns:
    ['Close','High','Low','Open','Volume'] as floats.
    """
    df = pd.read_csv(path, header=[0,1], index_col=0)
    # index name is likely 'Datetime'
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    # Flatten multiindex columns, keep only the first level names (Close/High/Low/Open/Volume)
    # Example: ('Close','ETC-USD') -> 'Close'
    df.columns = [c[0] for c in df.columns]

    # Keep only expected columns (some files may have 'Price' col label weirdness)
    keep = [c for c in ["Close","High","Low","Open","Volume"] if c in df.columns]
    df = df[keep].copy()

    # Coerce to numeric
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with no Close
    df = df.dropna(subset=["Close"])

    return df

def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    # if there are duplicates or irregularities, take last within hour
    return df.resample("h").last()

def safe_ffill(series: pd.Series, limit: int = 6) -> pd.Series:
    return series.ffill(limit=limit)

def log_return(close: pd.Series) -> pd.Series:
    return np.log(close).diff()

def rolling_sum(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).sum()

def rolling_std(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).std()

def drawdown(close: pd.Series, window: int) -> pd.Series:
    roll_max = close.rolling(window=window, min_periods=window).max()
    return close / roll_max - 1.0

def build_features(target: str, aligned: dict, rets: dict, full_index: pd.DatetimeIndex, ES_has_trade: pd.Series, GC_has_trade: pd.Series) -> pd.DataFrame:
    assert target in ["SUI", "AAVE", "ETC"]

    feat = pd.DataFrame(index=full_index)

    # target series aligned
    close_t = aligned[target]["Close"].reindex(full_index)
    r_t = rets[target].reindex(full_index)

    feat[f"{target}_r1h"] = r_t
    feat[f"{target}_mom24h"] = rolling_sum(r_t, 24)
    feat[f"{target}_mom4h"] = rolling_sum(r_t, 4)
    feat[f"{target}_vol4h"] = rolling_std(r_t, 4)
    feat[f"{target}_vol24h"] = rolling_std(r_t, 24)
    feat[f"{target}_dd7d"] = drawdown(close_t, 24 * 7)

    # cross-crypto
    others = [x for x in ["SUI", "AAVE", "ETC"] if x != target]
    feat[f"{others[0]}_r1h"] = rets[others[0]].reindex(full_index)
    feat[f"{others[1]}_r1h"] = rets[others[1]].reindex(full_index)

    # macro returns aligned
    feat["BTC_r1h"] = rets["BTC"].reindex(full_index)
    feat["ES_r1h"]  = rets["ES"].reindex(full_index)
    feat["GC_r1h"]  = rets["GC"].reindex(full_index)

    feat["ES_has_trade"] = ES_has_trade.reindex(full_index)
    feat["GC_has_trade"] = GC_has_trade.reindex(full_index)

    # time-of-day
    hours = feat.index.hour.values
    feat["tod_sin"] = np.sin(2*np.pi*hours/24.0)
    feat["tod_cos"] = np.cos(2*np.pi*hours/24.0)

    needed = [
        f"{target}_r1h", f"{target}_mom24h", f"{target}_vol24h", f"{target}_dd7d", f"{target}_mom4h", f"{target}_vol4h",
        "BTC_r1h", "ES_r1h", "GC_r1h",
        f"{others[0]}_r1h", f"{others[1]}_r1h"
    ]
    feat = feat.dropna(subset=needed)

    return feat

def define_regime(feat: pd.DataFrame, target: str, q: float = 0.10):
    """
    Define regimes based on 24h momentum percentiles.
    """
    mom = feat[f"{target}_mom24h"]

    low_thr = mom.quantile(q)
    high_thr = mom.quantile(1 - q)

    regime = pd.Series("steady", index=feat.index)

    regime[mom <= low_thr] = "crash"
    regime[mom >= high_thr] = "surge"

    return regime, low_thr, high_thr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing raw data CSVs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data = {}
    for k, fname in FILES.items():
        path = data_dir / fname
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
        data[k] = read_market_csv(path)
        print(k, data[k].index.min(), "->", data[k].index.max(), "rows:", len(data[k]))

    if not data:
        print("No data found. Exiting.")
        return

    # Use crypto union time range
    start = min(data[x].index.min() for x in ["SUI","AAVE","ETC","BTC"] if x in data)
    end   = max(data[x].index.max() for x in ["SUI","AAVE","ETC","BTC"] if x in data)

    full_index = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    print("Full hourly index:", full_index[0], "->", full_index[-1], "len:", len(full_index))

    for k in data:
        data[k] = to_hourly(data[k])

    # reindex
    aligned = {k: data[k].reindex(full_index) for k in data}

    # Save raw availability BEFORE forward fill
    ES_has_trade = aligned["ES"]["Close"].notna().astype(int) if "ES" in aligned else pd.Series(1, index=full_index)
    GC_has_trade = aligned["GC"]["Close"].notna().astype(int) if "GC" in aligned else pd.Series(1, index=full_index)

    # Forward-fill macro signals safely
    for k in ["ES", "GC"]:
        if k in aligned:
            for col in ["Close", "High", "Low", "Open", "Volume"]:
                if col in aligned[k].columns:
                    aligned[k][col] = safe_ffill(aligned[k][col], limit=6)

    rets = {}
    for k in data.keys():
        rets[k] = log_return(aligned[k]["Close"])

    for target in ["SUI", "AAVE", "ETC"]:
        if target not in data:
            continue
        feat = build_features(target, aligned, rets, full_index, ES_has_trade, GC_has_trade)
        feat["regime"], low, high = define_regime(feat, target)
        
        out_path = data_dir / f"features_{target}.csv"
        feat.to_csv(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
