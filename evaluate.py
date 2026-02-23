import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve
from config import Config


class Evaluator:
    def __init__(self, model, test_loader, config):
        self.model       = model
        self.test_loader = test_loader
        self.config      = config
        self.device      = config.DEVICE
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self):
        all_preds, all_labels, all_unc, all_risk = [], [], [], []
        all_trajectories   = []
        all_future_vitals  = []
        all_future_masks   = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Dataset now returns 7 values
                static, time_series, labels, masks, _, future_vitals, future_masks = batch

                static        = static.to(self.device)
                time_series   = time_series.to(self.device)
                labels        = labels.to(self.device)
                masks         = masks.to(self.device)
                future_vitals = future_vitals.to(self.device)
                future_masks  = future_masks.to(self.device)

                preds, uncertainty, failure_risk, trajectories, _ = self.model(
                    static, time_series, masks, training=False
                )

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_unc.extend(uncertainty.cpu().numpy())
                all_risk.extend(failure_risk.cpu().numpy())
                all_trajectories.append(trajectories.cpu().numpy())
                all_future_vitals.append(future_vitals.cpu().numpy())
                all_future_masks.append(future_masks.cpu().numpy())

        return (
            np.array(all_preds).flatten(),
            np.array(all_labels).flatten(),
            np.array(all_unc).flatten(),
            np.array(all_risk).flatten(),
            np.concatenate(all_trajectories,  axis=0),
            np.concatenate(all_future_vitals, axis=0),
            np.concatenate(all_future_masks,  axis=0),
        )

    def computeClassificationMetrics(self, predictions, labels, threshold=None):
        threshold = threshold or Config.RISK_THRESHOLD
        binary    = (predictions >= threshold).astype(int)

        auroc = roc_auc_score(labels, predictions)          if len(np.unique(labels)) > 1 else 0.0
        auprc = average_precision_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
        f1    = f1_score(labels, binary, zero_division=0)

        cm = confusion_matrix(labels, binary)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        return {
            "AUROC":       round(auroc, 4),
            "AUPRC":       round(auprc, 4),
            "F1":          round(f1, 4),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "PPV":         round(ppv, 4),
            "NPV":         round(npv, 4),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        }

    def computeTrajectoryMAE(self, trajectories, future_vitals_true, future_masks):
        """
        MAE per vital sign using median (q=0.5) trajectory prediction.
        Only evaluates on steps where a real measurement exists (mask=1).

        Args:
            trajectories      : (N, horizon, num_vitals, num_quantiles)
            future_vitals_true: (N, horizon, num_vitals)
            future_masks      : (N, horizon, num_vitals)  — 1=real, 0=missing
        """
        median     = trajectories[:, :, :, 1]   # 0.5 quantile
        errors     = np.abs(median - future_vitals_true)
        mask_bool  = future_masks.astype(bool)

        mae_per_vital = {}
        for i, vital in enumerate(Config.TARGET_VITALS):
            m = mask_bool[:, :, i]
            mae_per_vital[vital] = round(float(errors[:, :, i][m].mean()), 4) if m.any() else float("nan")

        return mae_per_vital

    def plotRocCurve(self, predictions, labels, save_path=None):
        fpr, tpr, _ = roc_curve(labels, predictions)
        auroc = roc_auc_score(labels, predictions)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(); plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plotPrCurve(self, predictions, labels, save_path=None):
        precision, recall, _ = precision_recall_curve(labels, predictions)
        auprc = average_precision_score(labels, predictions)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(); plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plotCalibrationCurve(self, predictions, labels, n_bins=10, save_path=None):
        prob_true, prob_pred = calibration_curve(labels, predictions, n_bins=n_bins)
        plt.figure(figsize=(7, 5))
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.xlabel("Predicted Probability"); plt.ylabel("True Probability")
        plt.title("Calibration Curve")
        plt.legend(); plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def analyzeUncertainty(self, predictions, labels, uncertainties):
        correct  = (predictions >= Config.RISK_THRESHOLD) == labels.astype(bool)
        high_unc = uncertainties > uncertainties.mean()
        print("\nUncertainty Analysis:")
        print(f"  Mean uncertainty (correct)  : {uncertainties[correct].mean():.4f}")
        print(f"  Mean uncertainty (incorrect): {uncertainties[~correct].mean():.4f}")
        print(f"  Accuracy (high uncertainty) : {correct[high_unc].mean():.4f}")
        print(f"  Accuracy (low uncertainty)  : {correct[~high_unc].mean():.4f}")

    def computeEarlyDetectionLeadTime(self, predictions, labels):
        positive_mask = labels == 1
        if positive_mask.sum() == 0:
            return 0.0
        detected = (predictions[positive_mask] >= Config.RISK_THRESHOLD).mean()
        return round(float(detected * Config.PREDICTION_HORIZON_HOURS), 2)

    def fullEvaluation(self, best_threshold=None):
        print("Evaluating on test set...")
        predictions, labels, uncertainties, failure_risks, trajectories, future_vitals, future_masks = self.evaluate()

        # Use tuned threshold from training if available, else fall back to config
        threshold = best_threshold or Config.RISK_THRESHOLD

        print("\n" + "=" * 55)
        print("CLASSIFICATION METRICS")
        print("=" * 55)
        metrics = self.computeClassificationMetrics(predictions, labels, threshold=threshold)
        for k, v in metrics.items():
            print(f"  {k:<15}: {v}")

        print(f"\n  (threshold used: {threshold:.3f})")

        self.analyzeUncertainty(predictions, labels, uncertainties)

        lead_time = self.computeEarlyDetectionLeadTime(predictions, labels)
        print(f"\n  Early Detection Lead Time: {lead_time:.2f} hours")

        print("\n" + "=" * 55)
        print("TRAJECTORY MAE (scaled units)")
        print("=" * 55)
        mae = self.computeTrajectoryMAE(trajectories, future_vitals, future_masks)
        for vital, val in mae.items():
            print(f"  {vital:<15}: {val}")

        save = Config.SAVE_PATH
        self.plotRocCurve(predictions, labels, f"{save}roc_curve.png")
        self.plotPrCurve(predictions, labels, f"{save}pr_curve.png")
        self.plotCalibrationCurve(predictions, labels, save_path=f"{save}calibration.png")
        print(f"\nPlots saved to {save}")

        return metrics, trajectories


def loadBestModel(model, checkpoint_path):
    # weights_only=False required for PyTorch >= 2.6 with numpy scalars in checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint.get("best_threshold", Config.RISK_THRESHOLD)
    print(f"Loaded epoch {checkpoint['epoch']} | val_loss={checkpoint['val_loss']:.4f} | "
          f"threshold={best_threshold:.3f}")
    return model, best_threshold


if __name__ == "__main__":
    from train import main
    model, test_loader, _ = main()
    model, best_threshold = loadBestModel(model, f"{Config.SAVE_PATH}best_model.pt")
    evaluator = Evaluator(model, test_loader, Config)
    evaluator.fullEvaluation(best_threshold=best_threshold)