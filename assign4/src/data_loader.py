import numpy as np
import pandas as pd

def load_crypto_csv(path: str, asset_name: str) -> pd.Series:
    """
    Load one hourly crypto CSV and convert it into 8-hour Bernoulli outcomes.
    """
    # Common Yahoo-exported hourly CSV format:
    # Header: Price,Close,High,Low,Open,Volume
    # Row 1 : Ticker,BTC-USD,...
    # Row 2 : Datetime,,,,,
    # Rename first column to Datetime and remove the first two metadata rows.
    df = pd.read_csv(path)

    if "Price" in df.columns:
        first_col = "Price"
        df = df.rename(columns={first_col: "Datetime"})
        df = df[~df["Datetime"].astype(str).isin(["Ticker", "Datetime"])]

    if "Datetime" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})

    required_cols = {"Datetime", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{asset_name}: missing required columns {required_cols}, got {set(df.columns)}"
        )

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df[["Datetime", "Close"]].dropna()
    df = df.set_index("Datetime").sort_index()

    # Resample to 8-hour frequency.
    close_8h = df["Close"].resample("8h").last().ffill()

    # Compute log returns.
    # Note: diff() produces NaN at the first index.
    log_return = np.log(close_8h).diff()

    # Convert to Bernoulli outcomes (Y=1 if log_return > 0).
    # Use .iloc[1:] to remove the first NaN from diff().
    outcome = (log_return > 0).astype(int).iloc[1:]
    outcome.name = asset_name

    return outcome

def build_outcome_matrix(asset_files: dict, T: int = 500) -> pd.DataFrame:
    """
    Combine all crypto arms into one matrix.
    """
    series_list = []

    for asset_name, path in asset_files.items():
        series = load_crypto_csv(path, asset_name)
        series_list.append(series)

    # Align by timestamp intersection and drop rows containing any NaN.
    Y = pd.concat(series_list, axis=1).dropna()

    if len(Y) < T:
        print(f"Warning: Only {len(Y)} time periods available, requested {T}.")
    
    return Y.iloc[:T]