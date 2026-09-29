import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                        # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    Transformer encoder for multi-step stock price forecasting.

    Args:
        input_size:  Number of input features
        d_model:     Embedding / model dimension
        nhead:       Number of attention heads (d_model must be divisible by nhead)
        num_layers:  Number of TransformerEncoder layers
        dim_feedforward: FFN inner dimension
        horizon:     Number of future steps to predict
        dropout:     Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        horizon: int = 7,
        dropout: float = 0.1,
    ):
        super(TransformerForecaster, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, lookback, input_size)
        Returns:
            out: (batch_size, horizon)
        """
        x = self.input_projection(x)               # (batch, lookback, d_model)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)             # (batch, lookback, d_model)
        x = self.norm(x)
        x = x[:, -1, :]                             # take last time step
        x = self.dropout(x)
        out = self.fc(x)                            # (batch, horizon)
        return out


if __name__ == "__main__":
    model = TransformerForecaster(input_size=19, d_model=128, nhead=8, num_layers=3, horizon=7)
    dummy = torch.randn(32, 60, 19)
    print("Transformer output shape:", model(dummy).shape)  # Expected: (32, 7)
