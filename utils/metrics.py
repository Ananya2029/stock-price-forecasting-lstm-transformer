import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100


def directional_accuracy(y_true, y_pred, last_close):
    """
    % of forecasts that get the direction right: will the price on each future
    day be above or below today's close? A flat forecast (e.g. the naive
    baseline) makes no directional call, so it returns NaN.
    """
    last_close = np.asarray(last_close)[:, None]
    pred_move = np.sign(y_pred - last_close)
    if np.all(pred_move == 0):
        return float("nan")
    true_move = np.sign(y_true - last_close)
    return (pred_move == true_move).mean() * 100


def evaluate_all(y_true, y_pred, label="Model", last_close=None):
    """Print all metrics for a model."""
    direction = directional_accuracy(y_true, y_pred, last_close) if last_close is not None else float("nan")
    print(f"\n{'='*40}")
    print(f"  {label} Evaluation")
    print(f"{'='*40}")
    print(f"  RMSE  : {rmse(y_true, y_pred):.4f}")
    print(f"  MAE   : {mae(y_true, y_pred):.4f}")
    print(f"  MAPE  : {mape(y_true, y_pred):.2f}%")
    print(f"  Dir.  : {direction:.2f}%")
    print(f"{'='*40}")

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_accuracy": direction,
    }
