import numpy as np
import pandas as pd

def load_crypto_csv(path: str, asset_name: str) -> pd.Series:
    """
    Load one hourly crypto CSV and convert it into 8-hour Bernoulli outcomes.
    """
    # Yahoo 导出的小时级 CSV 常见格式:
    # Header: Price,Close,High,Low,Open,Volume
    # Row 1 : Ticker,BTC-USD,...
    # Row 2 : Datetime,,,,,
    # 读取后需要将首列重命名为 Datetime，并移除前两行 metadata。
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

    # 采样 8h
    close_8h = df["Close"].resample("8h").last().ffill()

    # 计算对数收益率
    # 注意: diff() 会导致第一个值为 NaN
    log_return = np.log(close_8h).diff()

    # 转换为 Bernoulli (Y=1 if log_return > 0)
    # 使用 .iloc[1:] 剔除因 diff() 产生的第一个 NaN
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

    # 按照时间戳对齐，取交集并剔除任何包含 NaN 的行
    Y = pd.concat(series_list, axis=1).dropna()

    if len(Y) < T:
        print(f"Warning: Only {len(Y)} time periods available, requested {T}.")
    
    return Y.iloc[:T]