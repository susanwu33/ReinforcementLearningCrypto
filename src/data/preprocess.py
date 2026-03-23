import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse

DATETIME_COL = "datetime"
SPLIT_COL = "split"
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
REFERENCE_ASSET = "AAVE"

def load_feature_csv(path: Path) -> pd.DataFrame:
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
    out = df.copy()
    for c in cols:
        out[f"{prefix}{c}"] = (out[c] - mu[c]) / std[c]
    return out

def env_critical_cols(sym: str):
    return {f"{sym}_r1h", f"{sym}_vol24h"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing feature CSVs")
    parser.add_argument("--out_dir", type=str, default="processed", help="Directory to save processed data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(data_dir / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_files = {
        "SUI":  data_dir / "features_SUI.csv",
        "AAVE": data_dir / "features_AAVE.csv",
        "ETC":  data_dir / "features_ETC.csv",
    }

    dfs = {}
    for sym, path in feature_files.items():
        if path.exists():
            dfs[sym] = load_feature_csv(path)
        else:
            print(f"Warning: {path} not found.")
            
    if not dfs:
        print("No features found. Exiting.")
        return

    # Use reference asset to compute cutoffs
    ref_asset = REFERENCE_ASSET if REFERENCE_ASSET in dfs else list(dfs.keys())[0]
    ref_df = dfs[ref_asset]
    t_train_end, t_val_end = compute_time_cutoffs(ref_df, TRAIN_FRAC, VAL_FRAC)

    split_marks = {
        "reference_asset": ref_asset,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "train_end_time": str(t_train_end),
        "val_end_time": str(t_val_end),
    }
    with open(out_dir / "split_marks.json", "w") as f:
        json.dump(split_marks, f, indent=2)

    for sym, df in dfs.items():
        df = add_split_column(df, t_train_end, t_val_end)

        numeric_cols = infer_numeric_cols(df)
        critical = env_critical_cols(sym)
        scale_cols = [c for c in numeric_cols if c not in critical]

        train_df = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
        val_df   = df[df[SPLIT_COL] == "val"].reset_index(drop=True)
        test_df  = df[df[SPLIT_COL] == "test"].reset_index(drop=True)

        mu, std = fit_scaler(train_df, scale_cols)

        train_out = apply_scaler(train_df, scale_cols, mu, std, prefix="z_")
        val_out   = apply_scaler(val_df,   scale_cols, mu, std, prefix="z_")
        test_out  = apply_scaler(test_df,  scale_cols, mu, std, prefix="z_")

        scaler = {
            "scaled_cols": scale_cols,
            "critical_raw_cols": sorted(list(critical)),
            "mu": mu.to_dict(),
            "std": std.to_dict(),
            "prefix": "z_",
        }
        with open(out_dir / f"{sym}_scaler.json", "w") as f:
            json.dump(scaler, f, indent=2)

        df.to_csv(out_dir / f"{sym}_features_with_split.csv", index=False)
        train_out.to_csv(out_dir / f"{sym}_train.csv", index=False)
        val_out.to_csv(out_dir / f"{sym}_val.csv", index=False)
        test_out.to_csv(out_dir / f"{sym}_test.csv", index=False)

        print(f"{sym}: train={len(train_out)}, val={len(val_out)}, test={len(test_out)} | "
              f"{train_out[DATETIME_COL].min()} -> {test_out[DATETIME_COL].max()}")

    print("\nDone. Outputs saved to:", out_dir)

if __name__ == "__main__":
    main()
