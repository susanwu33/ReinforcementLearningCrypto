!pip -q install pandas numpy

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/content/drive/MyDrive/ReinforcementLearningCrypto")
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

data = {}
for k, fname in FILES.items():
    path = DATA_DIR / fname
    data[k] = read_market_csv(path)
    print(k, data[k].index.min(), "->", data[k].index.max(), "rows:", len(data[k]))

# Use crypto union time range
start = min(data[x].index.min() for x in ["SUI","AAVE","ETC","BTC"])
end   = max(data[x].index.max() for x in ["SUI","AAVE","ETC","BTC"])

full_index = pd.date_range(start=start, end=end, freq="H", tz="UTC")
print("Full hourly index:", full_index[0], "->", full_index[-1], "len:", len(full_index))

def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    # if there are duplicates or irregularities, take last within hour
    return df.resample("H").last()

for k in data:
    data[k] = to_hourly(data[k])

# reindex
aligned = {k: data[k].reindex(full_index) for k in data}

# Save raw availability BEFORE forward fill
ES_has_trade = aligned["ES"]["Close"].notna().astype(int)
GC_has_trade = aligned["GC"]["Close"].notna().astype(int)

'''
Forward-fill ES and GC safely:
forward fill up to 6 hours (overnight gaps)
do not fill entire weekends
'''
def safe_ffill(series: pd.Series, limit: int = 6) -> pd.Series:
    return series.ffill(limit=limit)

for k in ["ES", "GC"]:
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in aligned[k].columns:
            aligned[k][col] = safe_ffill(aligned[k][col], limit=6)

# Create core returns series (Close → log return)
def log_return(close: pd.Series) -> pd.Series:
    return np.log(close).diff()

rets = {}
for k in ["SUI","AAVE","ETC","BTC","ES","GC"]:
    rets[k] = log_return(aligned[k]["Close"])

def rolling_sum(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).sum()

def rolling_std(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).std()

def drawdown(close: pd.Series, window: int) -> pd.Series:
    roll_max = close.rolling(window=window, min_periods=window).max()
    return close / roll_max - 1.0

def build_features(target: str) -> pd.DataFrame:
    assert target in ["SUI", "AAVE", "ETC"]

    feat = pd.DataFrame(index=full_index)

    # target series aligned
    close_t = aligned[target]["Close"].reindex(full_index)
    r_t = rets[target].reindex(full_index)

    feat[f"{target}_r1h"] = r_t
    feat[f"{target}_mom24h"] = rolling_sum(r_t, 24)
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
        f"{target}_r1h", f"{target}_mom24h", f"{target}_vol24h", f"{target}_dd7d",
        "BTC_r1h", "ES_r1h", "GC_r1h",
        f"{others[0]}_r1h", f"{others[1]}_r1h"
    ]
    feat = feat.dropna(subset=needed)

    return feat

feat_SUI  = build_features("SUI")
feat_AAVE = build_features("AAVE")
feat_ETC  = build_features("ETC")

print(feat_SUI.shape, feat_AAVE.shape, feat_ETC.shape)
feat_SUI.head()

f = build_features("ETC")
print(f.index.min(), f.index.max(), f.shape)
print(f.isna().sum().sort_values(ascending=False).head(10))
print(f[[ "ETC_r1h","ETC_mom24h","ETC_vol24h","ETC_dd7d","ES_r1h","GC_r1h"]].describe())

f = build_features("ETC")

print("ES trade rate:", f["ES_has_trade"].mean())
print("GC trade rate:", f["GC_has_trade"].mean())

print("ES zero return fraction:", (f["ES_r1h"] == 0).mean())

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

feat_ETC["regime"], low_ETC, high_ETC = define_regime(feat_ETC, "ETC")
feat_SUI["regime"], low_SUI, high_SUI = define_regime(feat_SUI, "SUI")
feat_AAVE["regime"], low_AAVE, high_AAVE = define_regime(feat_AAVE, "AAVE")

print(feat_ETC["regime"].value_counts())

feat_ETC.groupby("regime")[f"ETC_mom24h"].describe()

feat_SUI.to_csv("features_SUI.csv")
feat_AAVE.to_csv("features_AAVE.csv")
feat_ETC.to_csv("features_ETC.csv")

print("Saved:", "features_SUI.csv", "features_AAVE.csv", "features_ETC.csv")

feat_AAVE.head()

