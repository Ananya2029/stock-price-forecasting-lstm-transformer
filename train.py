import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf

from utils.indicators import add_technical_indicators, add_stationary_features, STATIONARY_FEATURE_COLS
from utils.preprocess import split_sizes
from utils.metrics import evaluate_all
from models.lstm_model import LSTMForecaster
from models.transformer_model import TransformerForecaster


# ── Config ──────────────────────────────────────────────────────────────
TICKER = "AAPL"
START = "2015-01-01"
END = "2024-12-31"
LOOKBACK = 60
HORIZON = 7          # change to 30 for monthly forecasting
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_data():
    # Cache the download so results are reproducible and training works offline
    cache = f"data/{TICKER}_{START}_{END}.csv"
    if os.path.exists(cache):
        print(f"Loading cached {TICKER} data from {cache}")
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {TICKER} data ...")
        df = yf.download(TICKER, start=START, end=END, auto_adjust=True)

        # Newer yfinance returns MultiIndex columns like ('Close', 'AAPL') — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        os.makedirs("data", exist_ok=True)
        df.to_csv(cache)

    df = add_technical_indicators(df)
    df = add_stationary_features(df)
    return df


def build_dataset(df):
    """
    Sliding windows of scale-free features -> the next HORIZON daily returns
    relative to the last observed close. Predicting returns (not price levels)
    matters because test-period prices ($160-$257) lie above anything seen in
    training (max ~$178): a price-level model would have to extrapolate.
    """
    features = df[STATIONARY_FEATURE_COLS].to_numpy()
    close = df["Close"].to_numpy()

    n_seq = len(df) - LOOKBACK - HORIZON + 1
    X = np.stack([features[i : i + LOOKBACK] for i in range(n_seq)])
    last_close = close[LOOKBACK - 1 : LOOKBACK - 1 + n_seq]
    future = np.stack([close[i + LOOKBACK : i + LOOKBACK + HORIZON] for i in range(n_seq)])
    y = future / last_close[:, None] - 1

    train_size, val_size, _ = split_sizes(n_seq)
    tr, va = slice(0, train_size), slice(train_size, train_size + val_size)
    te = slice(train_size + val_size, n_seq)

    # Scalers are fitted on training windows only (no look-ahead into val/test)
    scaler_x = StandardScaler().fit(X[tr].reshape(-1, X.shape[2]))
    scaler_y = StandardScaler().fit(y[tr])
    X_scaled = scaler_x.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    y_scaled = scaler_y.transform(y)

    def to_loader(sl, shuffle=False):
        ds = TensorDataset(torch.tensor(X_scaled[sl], dtype=torch.float32),
                           torch.tensor(y_scaled[sl], dtype=torch.float32))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return {
        "train": to_loader(tr, shuffle=True),
        "val": to_loader(va),
        "X_test": X_scaled[te],
        "last_close_test": last_close[te],
        "truth_test": future[te],          # actual prices, for evaluation in $
        "scaler_y": scaler_y,
        "n_features": X.shape[2],
    }


def plot_loss(train_losses, val_losses, model_name):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name.lower()}_loss_curve.png")
    plt.close()


def plot_predictions(actual, predicted, model_name):
    plt.figure(figsize=(12,6))

    plt.plot(actual[:,0], label="Actual")
    plt.plot(predicted[:,0], label="Predicted")

    plt.title(f"{model_name} Actual vs Predicted (next-day close, test period)")
    plt.xlabel("Test Samples")
    plt.ylabel("Stock Price ($)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"outputs/actual_vs_predicted_{model_name.lower()}.png")
    plt.close()


def plot_metrics(results):

    models=list(results.keys())

    rmse=[results[m]["rmse"] for m in models]
    mae=[results[m]["mae"] for m in models]
    mape=[results[m]["mape"] for m in models]
    direction=[results[m]["directional_accuracy"] for m in models]

    def save_bar(values,title,ylabel,file):

        plt.figure(figsize=(7,4))
        plt.bar(models,values)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(file)
        plt.close()

    save_bar(rmse,"RMSE Comparison","RMSE ($)","outputs/model_comparison_rmse.png")
    save_bar(mae,"MAE Comparison","MAE ($)","outputs/model_comparison_mae.png")
    save_bar(mape,"MAPE Comparison","MAPE (%)","outputs/model_comparison_mape.png")
    save_bar(direction,"Directional Accuracy","Accuracy (%)","outputs/model_comparison_direction.png")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        total_loss += criterion(model(X_batch), y_batch).item()
    return total_loss / len(loader)


def run(model, model_name, data):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, data["train"], optimizer, criterion)
        val_loss = eval_epoch(model, data["val"], criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"outputs/{model_name}_best.pt")

        if epoch % 10 == 0:
            print(f"[{model_name}] Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    plot_loss(train_losses, val_losses, model_name)

    # ── Test evaluation ──────────────────────────────────────────────────
    model.load_state_dict(torch.load(f"outputs/{model_name}_best.pt"))
    model.eval()
    X_tensor = torch.tensor(data["X_test"], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds_scaled = model(X_tensor).cpu().numpy()

    # Predicted returns -> predicted prices
    pred_returns = data["scaler_y"].inverse_transform(preds_scaled)
    preds = data["last_close_test"][:, None] * (1 + pred_returns)
    truth = data["truth_test"]

    metrics = evaluate_all(truth, preds, label=model_name, last_close=data["last_close_test"])

    plot_predictions(truth, preds, model_name)

    return preds, truth, metrics

results = {}
all_predictions = {}

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)   # create outputs/ if it doesn't exist
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "both"], default="both")
    args = parser.parse_args()

    df = get_data()
    data = build_dataset(df)

    # Naive baseline: assume the price stays at the last observed close for the whole horizon.
    # A deep model is only useful if it beats this.
    naive = np.repeat(data["last_close_test"][:, None], HORIZON, axis=1)
    results["Naive"] = evaluate_all(data["truth_test"], naive, label="Naive (last close)",
                                    last_close=data["last_close_test"])
    all_predictions["Actual_day1"] = data["truth_test"][:, 0]
    all_predictions["Naive_day1"] = naive[:, 0]

    if args.model in ("lstm", "both"):
        lstm = LSTMForecaster(input_size=data["n_features"], horizon=HORIZON)
        preds, truth, metrics = run(lstm, "LSTM", data)
        results["LSTM"] = metrics
        all_predictions["LSTM_day1"] = preds[:, 0]

    if args.model in ("transformer", "both"):
        transformer = TransformerForecaster(input_size=data["n_features"], horizon=HORIZON)
        preds, truth, metrics = run(transformer, "Transformer", data)
        results["Transformer"] = metrics
        all_predictions["Transformer_day1"] = preds[:, 0]

    plot_metrics(results)
    pd.DataFrame(results).T.to_csv("outputs/evaluation.csv", index=True)
    pd.DataFrame(all_predictions).to_csv("outputs/predictions.csv", index=False)

    print("\nSaved:")
    print("✓ Evaluation CSV")
    print("✓ Predictions CSV")
    print("✓ Model comparison charts")
