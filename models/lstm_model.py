import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Stacked LSTM for multi-step stock price forecasting.

    Args:
        input_size:  Number of input features
        hidden_size: LSTM hidden state dimension
        num_layers:  Number of stacked LSTM layers
        horizon:     Number of future steps to predict
        dropout:     Dropout probability between LSTM layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon: int = 7,
        dropout: float = 0.2,
    ):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, lookback, input_size)
        Returns:
            out: (batch_size, horizon)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))      # (batch, lookback, hidden)
        last_hidden = lstm_out[:, -1, :]           # take last time step
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)                 # (batch, horizon)
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = LSTMForecaster(input_size=19, hidden_size=128, num_layers=2, horizon=7)
    dummy = torch.randn(32, 60, 19)               # batch=32, lookback=60, features=19
    print("LSTM output shape:", model(dummy).shape)  # Expected: (32, 7)
