import torch
import torch.nn as nn
import torch.nn.functional as F
from trajectory_head import TrajectoryHead
from config import Config


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.gate_fc = nn.Linear(hidden_size, output_size)
        self.skip_fc = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, context=None):
        skip = x
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        gate = torch.sigmoid(self.gate_fc(h))
        h = h * gate
        if self.skip_fc is not None:
            skip = self.skip_fc(skip)
        return self.layer_norm(h + skip)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout=0.1, context_size=None):
        super().__init__()
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout, context_size)
            for _ in range(num_inputs)
        ])
        self.weight_network = GatedResidualNetwork(
            num_inputs * hidden_size, hidden_size, num_inputs, dropout, context_size
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, variables, context=None):
        processed = [grn(v, context) for grn, v in zip(self.grns, variables)]
        stacked = torch.stack(processed, dim=-2)
        weights = self.softmax(self.weight_network(stacked.flatten(start_dim=-2), context)).unsqueeze(-1)
        return (stacked * weights).sum(dim=-2), weights


class TemporalFusionTransformer(nn.Module):
    def __init__(self, static_size, time_series_size, hidden_size, num_heads, dropout):
        super().__init__()
        self.static_encoder = GatedResidualNetwork(static_size, hidden_size, hidden_size, dropout)
        self.static_context_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.static_enrichment_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.ts_encoder = nn.Linear(time_series_size, hidden_size)
        self.lstm_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        self.post_lstm_gate = nn.Linear(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)
        self.static_enrichment = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.post_attn_gate = nn.Linear(hidden_size, hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)
        self.pos_wise_ff = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, static, time_series, mask=None):
        batch_size, seq_len, _ = time_series.shape

        static_encoded = self.static_encoder(static)
        static_context = self.static_context_grn(static_encoded)

        ts_encoded = self.ts_encoder(time_series)
        lstm_out, _ = self.lstm_encoder(ts_encoded)

        gate = torch.sigmoid(self.post_lstm_gate(lstm_out))
        lstm_out = self.post_lstm_norm(lstm_out * gate + ts_encoded)

        static_context_expanded = static_context.unsqueeze(1).expand(-1, seq_len, -1)
        enriched = self.static_enrichment(lstm_out, static_context_expanded)

        attn_mask = ~mask.bool() if mask is not None else None
        attn_out, attn_weights = self.multihead_attn(
            enriched, enriched, enriched, key_padding_mask=attn_mask
        )

        gate = torch.sigmoid(self.post_attn_gate(attn_out))
        attn_out = self.post_attn_norm(attn_out * gate + enriched)

        output = self.pos_wise_ff(attn_out)
        return self.dropout(output), attn_weights


class FailureAwareModule(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        self.confidence_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mc_dropout = nn.Dropout(dropout)

    def forward(self, x, training=False, mc_samples=1):
        batch_size, seq_len, hidden = x.shape

        if training or mc_samples > 1:
            samples = torch.stack([self.mc_dropout(x) for _ in range(mc_samples)], dim=0)
            mean_pred = samples.mean(dim=0)
        else:
            mean_pred = x

        uncertainty_score = self.uncertainty_head(mean_pred)
        confidence = 1.0 - uncertainty_score
        gated = mean_pred * self.confidence_gate(mean_pred) * confidence

        entropy = -(
            uncertainty_score * torch.log(uncertainty_score + 1e-10)
            + (1 - uncertainty_score) * torch.log(1 - uncertainty_score + 1e-10)
        )
        failure_risk = (uncertainty_score + entropy) / 2

        return gated, uncertainty_score, failure_risk


class FailureAwareTFT(nn.Module):
    """
    Combined Failure-Aware TFT + TFT-multi model.

    The shared encoder feeds two heads:
      1. TrajectoryHead  — predicts future vital trajectories (TFT-multi style)
      2. ClassificationHead — predicts deterioration probability (failure-aware)

    The classification head receives both the encoder output AND the trajectory
    predictions concatenated together, giving it awareness of where vitals are headed.
    """

    def __init__(
        self,
        static_size,
        time_series_size,
        hidden_size,
        num_heads,
        dropout,
        num_quantiles,
        mc_samples=10,
        num_target_vitals=None,
        trajectory_horizon_steps=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mc_samples = mc_samples
        self.num_target_vitals = num_target_vitals or Config.NUM_TARGET_VITALS
        self.trajectory_horizon_steps = trajectory_horizon_steps or Config.TRAJECTORY_HORIZON_STEPS

        self.tft = TemporalFusionTransformer(
            static_size, time_series_size, hidden_size, num_heads, dropout
        )

        self.trajectory_head = TrajectoryHead(
            hidden_size=hidden_size,
            num_vitals=self.num_target_vitals,
            num_quantiles=num_quantiles,
            horizon_steps=self.trajectory_horizon_steps,
            dropout=dropout,
        )

        # trajectory context: flatten median quantile predictions -> project to hidden_size
        trajectory_flat_size = self.trajectory_horizon_steps * self.num_target_vitals
        self.trajectory_proj = nn.Sequential(
            nn.Linear(trajectory_flat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.failure_module = FailureAwareModule(hidden_size, dropout)

        # classification input = encoder last step + projected trajectory context
        self.prediction_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, static, time_series, mask=None, training=False):
        encoder_output, attn_weights = self.tft(static, time_series, mask)

        # trajectory predictions: (batch, horizon, num_vitals, num_quantiles)
        trajectories = self.trajectory_head(encoder_output)

        # use median (q=0.5, index 1) as context signal for classification
        median_idx = 1
        median_traj = trajectories[:, :, :, median_idx]  # (batch, horizon, num_vitals)
        traj_flat = median_traj.flatten(start_dim=1)      # (batch, horizon * num_vitals)
        traj_context = self.trajectory_proj(traj_flat)    # (batch, hidden_size)

        mc_samples = self.mc_samples if training else 1
        gated_output, uncertainty, failure_risk = self.failure_module(
            encoder_output, training=training, mc_samples=mc_samples
        )

        last_step = gated_output[:, -1, :]  # (batch, hidden_size)

        # concatenate encoder context + trajectory context before final classification
        combined = torch.cat([last_step, traj_context], dim=-1)  # (batch, hidden_size * 2)
        prediction = torch.sigmoid(self.prediction_head(combined))  # (batch, 1)

        last_uncertainty = uncertainty[:, -1, :]
        last_failure_risk = failure_risk[:, -1, :]

        return prediction, last_uncertainty, last_failure_risk, trajectories, attn_weights
