import torch
import torch.nn as nn
from config import Config

class TrajectoryHead(nn.Module):
    """
    Takes the shared TFT encoder output and produces multivariate
    quantile forecasts for Config.TARGET_VITALS over the prediction horizon.

    Output shape: (batch, horizon_steps, num_vitals, num_quantiles)
    """

    def __init__(self, hidden_size, num_vitals, num_quantiles, horizon_steps, dropout=0.1):
        super().__init__()
        self.num_vitals = num_vitals
        self.num_quantiles = num_quantiles
        self.horizon_steps = horizon_steps

        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.horizon_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(
            hidden_size,
            num_vitals * num_quantiles,
        )

    def forward(self, encoder_output):
        """
        Args:
            encoder_output: (batch, seq_len, hidden_size) — from shared TFT encoder

        Returns:
            trajectories: (batch, horizon_steps, num_vitals, num_quantiles)
        """
        context = encoder_output[:, -1:, :]
        context = self.temporal_proj(context)

        context_expanded = context.expand(-1, self.horizon_steps, -1)

        lstm_out, _ = self.horizon_lstm(context_expanded)

        raw = self.output_proj(lstm_out)

        trajectories = raw.view(
            raw.shape[0],
            self.horizon_steps,
            self.num_vitals,
            self.num_quantiles,
        )

        return trajectories


def quantileLoss(predictions, targets, quantiles, mask=None):
    """
    Pinball / quantile loss summed over all vitals and quantiles.
    Only computes loss on real (non-imputed) values via mask.

    Args:
        predictions : (batch, horizon, num_vitals, num_quantiles)
        targets     : (batch, horizon, num_vitals)
        quantiles   : list of floats e.g. [0.1, 0.5, 0.9]
        mask        : (batch, horizon, num_vitals) — 1 = real value, 0 = imputed

    Returns:
        scalar loss
    """
    targets_expanded = targets.unsqueeze(-1).expand_as(predictions)

    errors = targets_expanded - predictions

    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=predictions.device)
    q_tensor = q_tensor.view(1, 1, 1, -1)

    loss_per_element = torch.max(
        q_tensor * errors,
        (q_tensor - 1.0) * errors,
    )

    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand_as(loss_per_element)
        denom = mask_expanded.sum().clamp(min=1)
        loss = (loss_per_element * mask_expanded).sum() / denom
    else:
        loss = loss_per_element.mean()

    return loss