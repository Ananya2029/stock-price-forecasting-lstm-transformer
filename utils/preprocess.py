import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_scale(df: pd.DataFrame, feature_cols: list, target_col: str = "Close", fit_rows: int = None):
    """Scale features and target using MinMaxScaler.

    The scalers are fitted only on the first `fit_rows` rows (the training period),
    so no information from the validation/test period leaks into training.
    """
    fit_rows = len(df) if fit_rows is None else fit_rows
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_x.fit(df[feature_cols].iloc[:fit_rows])
    scaler_y.fit(df[[target_col]].iloc[:fit_rows])
    X_scaled = scaler_x.transform(df[feature_cols])
    y_scaled = scaler_y.transform(df[[target_col]])

    return X_scaled, y_scaled, scaler_x, scaler_y


def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 60, horizon: int = 7):
    """
    Create sliding window sequences for multi-step forecasting.

    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples, 1)
        lookback: Number of past days as input
        horizon: Number of future days to predict

    Returns:
        X_seq: (n_sequences, lookback, n_features)
        y_seq: (n_sequences, horizon)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback - horizon + 1):
        X_seq.append(X[i : i + lookback])
        y_seq.append(y[i + lookback : i + lookback + horizon, 0])
    return np.array(X_seq), np.array(y_seq)


def split_sizes(n, val_ratio=0.1, test_ratio=0.1):
    """Number of train / val / test sequences for a chronological split."""
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    return n - val_size - test_size, val_size, test_size


def train_val_test_split(X_seq, y_seq, val_ratio=0.1, test_ratio=0.1):
    """Chronological split — no shuffling for time series."""
    train_size, val_size, test_size = split_sizes(len(X_seq), val_ratio, test_ratio)

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size : train_size + val_size]
    y_val = y_seq[train_size : train_size + val_size]
    X_test = X_seq[train_size + val_size :]
    y_test = y_seq[train_size + val_size :]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
