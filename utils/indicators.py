import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to OHLCV dataframe.
    Expects columns: Open, High, Low, Close, Volume
    """
    df = df.copy()

    # --- Trend Indicators ---
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # --- Momentum Indicators ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- Volatility Indicators ---
    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

    # --- Volume Indicators ---
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-9)

    # --- Price-derived Features ---
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]

    df.dropna(inplace=True)
    return df


def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale-free versions of the indicators (ratios to the close price, returns),
    so a model trained on 2015-2022 prices still works when prices move into a
    range it has never seen.
    """
    df = df.copy()
    close = df["Close"]
    df["Open_Rel"] = df["Open"] / close - 1
    df["High_Rel"] = df["High"] / close - 1
    df["Low_Rel"] = df["Low"] / close - 1
    df["EMA12_Gap"] = df["EMA_12"] / close - 1
    df["EMA26_Gap"] = df["EMA_26"] / close - 1
    df["MACD_Norm"] = df["MACD"] / close
    df["MACD_Signal_Norm"] = df["MACD_Signal"] / close
    df["MACD_Hist_Norm"] = df["MACD_Hist"] / close
    df["RSI_Scaled"] = df["RSI"] / 100
    df["BB_Position"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)
    df["BB_Width_Norm"] = df["BB_Width"] / close
    df["Return_5d"] = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)
    df["Volatility_20d"] = df["Log_Return"].rolling(20).std()
    df.dropna(inplace=True)
    return df


STATIONARY_FEATURE_COLS = [
    "Open_Rel", "High_Rel", "Low_Rel", "EMA12_Gap", "EMA26_Gap",
    "MACD_Norm", "MACD_Signal_Norm", "MACD_Hist_Norm", "RSI_Scaled",
    "BB_Position", "BB_Width_Norm", "Volume_Ratio", "Daily_Return",
    "Log_Return", "HL_Range", "Return_5d", "Return_20d", "Volatility_20d",
]


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA_12", "EMA_26", "MACD", "MACD_Signal", "MACD_Hist",
    "RSI", "BB_Mid", "BB_Upper", "BB_Lower", "BB_Width",
    "Volume_Ratio", "Daily_Return", "Log_Return", "HL_Range",
]
