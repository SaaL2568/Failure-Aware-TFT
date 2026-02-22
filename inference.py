import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from config import Config
from model import FailureAwareTFT
from feature_engineering import FeatureEngineer, VITAL_THRESHOLDS

VITAL_CRITICAL_THRESHOLDS = {
    "heart_rate":  {"low": 40,  "high": 130},
    "sbp":         {"low": 80,  "high": 180},
    "dbp":         {"low": 50,  "high": 110},
    "resp_rate":   {"low": 8,   "high": 30},
    "spo2":        {"low": 88,  "high": 100},
}

class InferencePipeline:
    """
    Wraps the trained FailureAwareTFT model for single-patient real-time inference.
    Handles:
      - Feature construction from raw vital dicts
      - Tensor assembly
      - Model forward pass
      - Time-to-event estimation from trajectory predictions
      - Response formatting
    """

    def __init__(self, checkpoint_path, scaler_path, static_size, time_series_size):
        self.device = Config.DEVICE
        self.scaler = joblib.load(scaler_path)

        self.model = FailureAwareTFT(
            static_size=static_size,
            time_series_size=time_series_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT,
            num_quantiles=Config.NUM_QUANTILES,
            mc_samples=1,
            num_target_vitals=Config.NUM_TARGET_VITALS,
            trajectory_horizon_steps=Config.TRAJECTORY_HORIZON_STEPS,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from epoch {checkpoint['epoch']} | val_loss={checkpoint['val_loss']:.4f}")

    def _preprocessWindow(self, window_df):
        """
        Convert a DataFrame of raw vital readings into a scaled numpy array
        ready for model input.

        Args:
            window_df: DataFrame with columns matching VITAL_ITEMID_MAP vital names
                       and a 'timestamp' column

        Returns:
            numpy array of shape (seq_len, num_features)
        """
        feature_cols = [c for c in window_df.columns if c not in ("timestamp", "patient_id")]
        values = window_df[feature_cols].values.astype(np.float32)
        values = np.nan_to_num(values, nan=0.0)

        if self.scaler is not None:
            values = self.scaler.transform(values)

        return values

    def _estimateTimeToEvent(self, trajectories, quantile_idx=1):
        """
        Estimate time to deterioration from predicted vital trajectories by finding
        the first step where any vital crosses its critical threshold.

        Args:
            trajectories: (batch=1, horizon_steps, num_vitals, num_quantiles) tensor
            quantile_idx: which quantile to use (1 = median)

        Returns:
            float — estimated hours until threshold crossing, or prediction horizon if none
        """
        median_traj = trajectories[0, :, :, quantile_idx].cpu().numpy()
        vital_names = Config.TARGET_VITALS
        step_hours = Config.PREDICTION_STEP_MINUTES / 60

        for step_idx in range(median_traj.shape[0]):
            for vital_idx, vital_name in enumerate(vital_names):
                val = median_traj[step_idx, vital_idx]
                thresholds = VITAL_CRITICAL_THRESHOLDS.get(vital_name)
                if thresholds is None:
                    continue
                if val < thresholds["low"] or val > thresholds["high"]:
                    return round(step_idx * step_hours, 2)

        return float(Config.PREDICTION_HORIZON_HOURS)

    def predict(self, static_features, time_series_window):
        """
        Run inference for one patient.

        Args:
            static_features  : 1D numpy array of static patient features
            time_series_window: 2D numpy array (seq_len, num_ts_features)

        Returns:
            dict with willDeteriorate, timeToEventHours, confidence, riskScore
        """
        static_tensor = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)
        ts_tensor = torch.FloatTensor(time_series_window).unsqueeze(0).to(self.device)
        mask = torch.ones(1, ts_tensor.shape[1]).to(self.device)

        with torch.no_grad():
            prediction, uncertainty, failure_risk, trajectories, _ = self.model(
                static_tensor, ts_tensor, mask, training=False
            )

        risk_score = float(prediction[0, 0].cpu())
        uncertainty_score = float(uncertainty[0, 0].cpu())
        confidence = 1.0 - uncertainty_score

        will_deteriorate = risk_score >= Config.RISK_THRESHOLD
        time_to_event = self._estimateTimeToEvent(trajectories) if will_deteriorate else float(Config.PREDICTION_HORIZON_HOURS)

        return {
            "willDeteriorate": bool(will_deteriorate),
            "timeToEventHours": round(time_to_event, 2),
            "confidence": round(float(confidence), 4),
            "riskScore": round(float(risk_score), 4),
        }

    def predictFromRawWindow(self, window_df, static_dict):
        """
        Convenience wrapper: takes a raw DataFrame window and a static feature dict,
        runs preprocessing and returns the prediction response.

        Args:
            window_df   : DataFrame with raw vitals and timestamps
            static_dict : dict of static features (gender, age, etc)

        Returns:
            prediction response dict
        """
        ts_array = self._preprocessWindow(window_df)

        static_keys = Config.STATIC_FEATURES
        static_array = np.array([static_dict.get(k, 0.0) for k in static_keys], dtype=np.float32)

        return self.predict(static_array, ts_array)
