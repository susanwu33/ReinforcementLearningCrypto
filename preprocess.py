!pip -q install pandas numpy

import pandas as pd
import numpy as np
from pathlib import Path
import json

from google.colab import drive
drive.mount('/content/drive')

# -----------------------
# Config
# -----------------------
DATA_DIR = Path("/content/drive/MyDrive/ReinforcementLearningCrypto")  # where features_*.csv live
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_FILES = {
    "SUI":  DATA_DIR / "features_SUI.csv",
    "AAVE": DATA_DIR / "features_AAVE.csv",
    "ETC":  DATA_DIR / "features_ETC.csv",
}

DATETIME_COL = "datetime"   # your csv index is datetime; we'll load it properly
SPLIT_COL = "split"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

# If you want all 3 cryptos to share the same split boundaries:
REFERENCE_ASSET = "AAVE"   # compute cutoffs on this one, apply to all

def load_feature_csv(path: Path) -> pd.DataFrame:
    """
    Your features csv was saved from a DataFrame with datetime index.
    So datetime becomes the first column in CSV (often unnamed) OR named 'datetime'.
    We'll handle both.
    """
    df = pd.read_csv(path)

    # If datetime column exists, use it; otherwise use the first column as datetime.
    if DATETIME_COL in df.columns:
        dt = df[DATETIME_COL]
    else:
        first_col = df.columns[0]
        dt = df[first_col]
        df = df.rename(columns={first_col: DATETIME_COL})

    df[DATETIME_COL] = pd.to_datetime(dt, utc=True, errors="coerce")
    df = df.dropna(subset=[DATETIME_COL]).sort_values(DATETIME_COL).reset_index(drop=True)
    return df


def compute_time_cutoffs(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df)
    i = int(n * train_frac)
    j = int(n * (train_frac + val_frac))

    # Use boundary timestamps
    t_train_end = df.loc[i, DATETIME_COL]
    t_val_end   = df.loc[j, DATETIME_COL]
    return t_train_end, t_val_end

def add_split_column(df: pd.DataFrame, t_train_end, t_val_end) -> pd.DataFrame:
    out = df.copy()
    out[SPLIT_COL] = "test"
    out.loc[out[DATETIME_COL] < t_train_end, SPLIT_COL] = "train"
    out.loc[(out[DATETIME_COL] >= t_train_end) & (out[DATETIME_COL] < t_val_end), SPLIT_COL] = "val"
    return out

def infer_numeric_cols(df: pd.DataFrame):
    exclude = {DATETIME_COL, SPLIT_COL, "regime"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def fit_scaler(train_df: pd.DataFrame, feature_cols):
    mu = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)
    return mu, std

def apply_scaler(df: pd.DataFrame, cols, mu, std, prefix="z_"):
    """
    Create scaled *copies* instead of overwriting raw columns.
    """
    out = df.copy()
    for c in cols:
        out[f"{prefix}{c}"] = (out[c] - mu[c]) / std[c]
    return out

def env_critical_cols(sym: str):
    """
    Columns that must stay RAW because the environment uses them for dynamics/reward.
    """
    return {f"{sym}_r1h", f"{sym}_vol24h"}  # keep raw; do NOT z-score

# 1) Load all feature files
dfs = {sym: load_feature_csv(path) for sym, path in FEATURE_FILES.items()}

# 2) Compute split cutoffs using reference asset (consistent splits across assets)
ref_df = dfs[REFERENCE_ASSET]
t_train_end, t_val_end = compute_time_cutoffs(ref_df, TRAIN_FRAC, VAL_FRAC)

split_marks = {
    "reference_asset": REFERENCE_ASSET,
    "train_frac": TRAIN_FRAC,
    "val_frac": VAL_FRAC,
    "train_end_time": str(t_train_end),
    "val_end_time": str(t_val_end),
}
with open(OUT_DIR / "split_marks.json", "w") as f:
    json.dump(split_marks, f, indent=2)

# 3) Apply split + create scaled feature copies (train-only)
for sym, df in dfs.items():
    df = add_split_column(df, t_train_end, t_val_end)

    # Determine numeric columns
    numeric_cols = infer_numeric_cols(df)

    # EXCLUDE env-critical columns from scaling
    critical = env_critical_cols(sym)
    scale_cols = [c for c in numeric_cols if c not in critical]

    # Split
    train_df = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
    val_df   = df[df[SPLIT_COL] == "val"].reset_index(drop=True)
    test_df  = df[df[SPLIT_COL] == "test"].reset_index(drop=True)

    # Fit scaler on train only (for scale_cols)
    mu, std = fit_scaler(train_df, scale_cols)

    # Apply scaler by creating z_ columns
    train_out = apply_scaler(train_df, scale_cols, mu, std, prefix="z_")
    val_out   = apply_scaler(val_df,   scale_cols, mu, std, prefix="z_")
    test_out  = apply_scaler(test_df,  scale_cols, mu, std, prefix="z_")

    # Save scaler params
    scaler = {
        "scaled_cols": scale_cols,
        "critical_raw_cols": sorted(list(critical)),
        "mu": mu.to_dict(),
        "std": std.to_dict(),
        "prefix": "z_",
    }
    with open(OUT_DIR / f"{sym}_scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)

    # Save outputs
    df.to_csv(OUT_DIR / f"{sym}_features_with_split.csv", index=False)
    train_out.to_csv(OUT_DIR / f"{sym}_train.csv", index=False)
    val_out.to_csv(OUT_DIR / f"{sym}_val.csv", index=False)
    test_out.to_csv(OUT_DIR / f"{sym}_test.csv", index=False)

    print(f"{sym}: train={len(train_out)}, val={len(val_out)}, test={len(test_out)} | "
          f"{train_out[DATETIME_COL].min()} -> {test_out[DATETIME_COL].max()}")

print("\nDone. Outputs saved to:", OUT_DIR)
print("Split marks saved to:", OUT_DIR / "split_marks.json")