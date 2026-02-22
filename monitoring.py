import numpy as np
from scipy.stats import ks_2samp
from collections import deque
from config import Config

class DriftDetector:
    """
    Monitors prediction score distribution for data/model drift using a
    Kolmogorov-Smirnov test comparing a reference window to a recent window.

    Alerts when the distribution of recent risk scores diverges significantly
    from the training-time reference distribution.
    """

    def __init__(self, window_size=None, ks_alpha=None):
        self.window_size = window_size or Config.DRIFT_WINDOW_SIZE
        self.ks_alpha = ks_alpha or Config.DRIFT_KS_ALPHA

        self.reference_scores = None
        self.recent_scores = deque(maxlen=self.window_size)
        self.drift_history = []

    def setReference(self, reference_scores):
        """
        Call this after training to set the baseline score distribution.
        Args:
            reference_scores: list or array of risk scores from validation set
        """
        self.reference_scores = np.array(reference_scores)
        print(f"Drift reference set: n={len(self.reference_scores)}, "
              f"mean={self.reference_scores.mean():.3f}, "
              f"std={self.reference_scores.std():.3f}")

    def update(self, risk_score):
        self.recent_scores.append(float(risk_score))

    def isDrifting(self):
        """
        Returns True if recent score distribution significantly differs from reference.
        Requires at least window_size // 2 recent scores before testing.
        """
        if self.reference_scores is None:
            return False
        if len(self.recent_scores) < self.window_size // 2:
            return False

        recent = np.array(self.recent_scores)
        stat, p_value = ks_2samp(self.reference_scores, recent)

        is_drift = p_value < self.ks_alpha
        self.drift_history.append({
            "n_recent": len(recent),
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": is_drift,
            "recent_mean": round(float(recent.mean()), 4),
            "reference_mean": round(float(self.reference_scores.mean()), 4),
        })

        return is_drift

    def getReport(self):
        if not self.drift_history:
            return {"status": "no_checks_run"}
        latest = self.drift_history[-1]
        return {
            "status": "DRIFT_DETECTED" if latest["drift_detected"] else "OK",
            "ks_stat": latest["ks_stat"],
            "p_value": latest["p_value"],
            "recent_mean_score": latest["recent_mean"],
            "reference_mean_score": latest["reference_mean"],
            "num_drift_events": sum(h["drift_detected"] for h in self.drift_history),
            "total_checks": len(self.drift_history),
        }


class PerformanceMonitor:
    """
    Tracks rolling prediction metrics over time.
    Useful for post-deployment monitoring when ground truth becomes available
    (e.g., after a shift when outcomes are known).
    """

    def __init__(self):
        self.predictions = []
        self.labels = []

    def log(self, risk_score, true_label):
        self.predictions.append(float(risk_score))
        self.labels.append(int(true_label))

    def getMetrics(self, threshold=None):
        if len(self.predictions) < 10:
            return {"status": "insufficient_data"}

        threshold = threshold or Config.RISK_THRESHOLD
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        binary = (preds >= threshold).astype(int)

        tp = int(((binary == 1) & (labels == 1)).sum())
        tn = int(((binary == 0) & (labels == 0)).sum())
        fp = int(((binary == 1) & (labels == 0)).sum())
        fn = int(((binary == 0) & (labels == 1)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else None
        except Exception:
            auroc = None

        return {
            "n": len(self.predictions),
            "prevalence": round(float(labels.mean()), 4),
            "mean_risk_score": round(float(preds.mean()), 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "auroc": round(auroc, 4) if auroc is not None else None,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
