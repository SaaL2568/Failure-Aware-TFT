import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        
        self.gate_fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        if input_size != output_size:
            self.skip_fc = nn.Linear(input_size, output_size)
        else:
            self.skip_fc = None
        
        self.layer_norm = nn.LayerNorm(output_size)
    
    def forward(self, x, context=None):
        skip = x
        
        x = self.fc1(x)
        
        if context is not None and self.context_size is not None:
            x = x + self.context_fc(context)
        
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        gate = self.sigmoid(self.gate_fc(x))
        x = x * gate
        
        if self.skip_fc is not None:
            skip = self.skip_fc(skip)
        
        x = x + skip
        x = self.layer_norm(x)
        
        return x

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout=0.1, context_size=None):
        super().__init__()
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout, context_size)
            for _ in range(num_inputs)
        ])
        
        self.softmax = nn.Softmax(dim=-1)
        self.weight_network = GatedResidualNetwork(
            num_inputs * hidden_size, hidden_size, num_inputs, dropout, context_size
        )
    
    def forward(self, variables, context=None):
        flatten_vars = []
        
        for i, grn in enumerate(self.grns):
            processed = grn(variables[i], context)
            flatten_vars.append(processed)
        
        flatten_vars = torch.stack(flatten_vars, dim=-2)
        
        flattened = flatten_vars.flatten(start_dim=-2)
        weights = self.weight_network(flattened, context)
        weights = self.softmax(weights).unsqueeze(-1)
        
        weighted = flatten_vars * weights
        output = weighted.sum(dim=-2)
        
        return output, weights

class TemporalFusionTransformer(nn.Module):
    def __init__(self, static_size, time_series_size, hidden_size, num_heads, dropout, num_quantiles):
        super().__init__()
        self.static_size = static_size
        self.time_series_size = time_series_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_quantiles = num_quantiles
        
        self.static_encoder = GatedResidualNetwork(static_size, hidden_size, hidden_size, dropout)
        
        self.static_context_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.static_enrichment_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        
        self.ts_encoder = nn.Linear(time_series_size, hidden_size)
        
        self.lstm_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        
        self.post_lstm_gate = nn.Linear(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)
        
        self.static_enrichment = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout, hidden_size)
        
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.post_attn_gate = nn.Linear(hidden_size, hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)
        
        self.pos_wise_ff = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        
        self.output_layer = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, static, time_series, mask=None):
        batch_size, seq_len, _ = time_series.shape
        
        static_encoded = self.static_encoder(static)
        static_context = self.static_context_grn(static_encoded)
        static_enrichment = self.static_enrichment_grn(static_encoded)
        
        ts_encoded = self.ts_encoder(time_series)
        
        lstm_out, _ = self.lstm_encoder(ts_encoded)
        
        gate = torch.sigmoid(self.post_lstm_gate(lstm_out))
        lstm_out = lstm_out * gate
        lstm_out = self.post_lstm_norm(lstm_out + ts_encoded)
        
        static_context_expanded = static_context.unsqueeze(1).expand(-1, seq_len, -1)
        enriched = self.static_enrichment(lstm_out, static_context_expanded)
        
        if mask is not None:
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
        
        attn_out, attn_weights = self.multihead_attn(
            enriched, enriched, enriched,
            key_padding_mask=attn_mask
        )
        
        gate = torch.sigmoid(self.post_attn_gate(attn_out))
        attn_out = attn_out * gate
        attn_out = self.post_attn_norm(attn_out + enriched)
        
        output = self.pos_wise_ff(attn_out)
        
        output = self.dropout(output)
        
        return output, attn_weights

class FailureAwareModule(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.confidence_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.mc_dropout = nn.Dropout(dropout)
    
    def forward(self, x, training=False, mc_samples=1):
        batch_size, seq_len, hidden = x.shape
        
        if training or mc_samples > 1:
            predictions = []
            for _ in range(mc_samples):
                dropped = self.mc_dropout(x)
                predictions.append(dropped)
            
            stacked = torch.stack(predictions, dim=0)
            mean_pred = stacked.mean(dim=0)
            uncertainty = stacked.std(dim=0).mean(dim=-1, keepdim=True)
        else:
            mean_pred = x
            uncertainty = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        uncertainty_score = self.uncertainty_head(mean_pred)
        
        confidence = 1 - uncertainty_score
        gated_output = mean_pred * self.confidence_gate(mean_pred) * confidence
        
        entropy = -(uncertainty_score * torch.log(uncertainty_score + 1e-10) + 
                   (1 - uncertainty_score) * torch.log(1 - uncertainty_score + 1e-10))
        
        failure_risk = (uncertainty_score + entropy) / 2
        
        return gated_output, uncertainty_score, failure_risk

class FailureAwareTFT(nn.Module):
    def __init__(self, static_size, time_series_size, hidden_size, num_heads, dropout, num_quantiles, mc_samples=10):
        super().__init__()
        self.static_size = static_size
        self.time_series_size = time_series_size
        self.hidden_size = hidden_size
        self.mc_samples = mc_samples
        
        self.tft = TemporalFusionTransformer(
            static_size, time_series_size, hidden_size, num_heads, dropout, num_quantiles
        )
        
        self.failure_module = FailureAwareModule(hidden_size, dropout)
        
        self.prediction_head = nn.Linear(hidden_size, 1)
    
    def forward(self, static, time_series, mask=None, training=False):
        tft_output, attn_weights = self.tft(static, time_series, mask)
        
        mc_samples = self.mc_samples if training else 1
        gated_output, uncertainty, failure_risk = self.failure_module(
            tft_output, training=training, mc_samples=mc_samples
        )
        
        prediction = self.prediction_head(gated_output)
        prediction = torch.sigmoid(prediction)
        
        last_step_pred = prediction[:, -1, :]
        last_step_uncertainty = uncertainty[:, -1, :]
        last_step_failure = failure_risk[:, -1, :]
        
        return last_step_pred, last_step_uncertainty, last_step_failure, attn_weights
