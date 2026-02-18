import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from config import Config

class Evaluator:
    def __init__(self, model, test_loader, config):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self):
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        all_failure_risks = []
        
        with torch.no_grad():
            for static, time_series, labels, masks, _ in self.test_loader:
                static = static.to(self.device)
                time_series = time_series.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                
                predictions, uncertainty, failure_risk, _ = self.model(
                    static, time_series, masks, training=False
                )
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())
                all_failure_risks.extend(failure_risk.cpu().numpy())
        
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        all_uncertainties = np.array(all_uncertainties).flatten()
        all_failure_risks = np.array(all_failure_risks).flatten()
        
        return all_predictions, all_labels, all_uncertainties, all_failure_risks
    
    def compute_metrics(self, predictions, labels):
        auroc = roc_auc_score(labels, predictions)
        auprc = average_precision_score(labels, predictions)
        
        pred_binary = (predictions > 0.5).astype(int)
        f1 = f1_score(labels, pred_binary)
        
        cm = confusion_matrix(labels, pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics = {
            'AUROC': auroc,
            'AUPRC': auprc,
            'F1': f1,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
        
        return metrics
    
    def plot_roc_curve(self, predictions, labels, save_path=None):
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        auroc = roc_auc_score(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_pr_curve(self, predictions, labels, save_path=None):
        precision, recall, thresholds = precision_recall_curve(labels, predictions)
        auprc = average_precision_score(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AUPRC = {auprc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_calibration_curve(self, predictions, labels, n_bins=10, save_path=None):
        prob_true, prob_pred = calibration_curve(labels, predictions, n_bins=n_bins)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def analyze_uncertainty(self, predictions, labels, uncertainties):
        correct = (predictions > 0.5) == labels
        
        avg_uncertainty_correct = uncertainties[correct].mean()
        avg_uncertainty_incorrect = uncertainties[~correct].mean()
        
        print("\nUncertainty Analysis:")
        print(f"Average uncertainty (correct predictions): {avg_uncertainty_correct:.4f}")
        print(f"Average uncertainty (incorrect predictions): {avg_uncertainty_incorrect:.4f}")
        
        high_uncertainty_mask = uncertainties > uncertainties.mean()
        acc_high_uncertainty = correct[high_uncertainty_mask].mean()
        acc_low_uncertainty = correct[~high_uncertainty_mask].mean()
        
        print(f"Accuracy (high uncertainty): {acc_high_uncertainty:.4f}")
        print(f"Accuracy (low uncertainty): {acc_low_uncertainty:.4f}")
    
    def analyze_failure_risk(self, predictions, labels, failure_risks):
        correct = (predictions > 0.5) == labels
        
        avg_failure_correct = failure_risks[correct].mean()
        avg_failure_incorrect = failure_risks[~correct].mean()
        
        print("\nFailure Risk Analysis:")
        print(f"Average failure risk (correct predictions): {avg_failure_correct:.4f}")
        print(f"Average failure risk (incorrect predictions): {avg_failure_incorrect:.4f}")
        
        high_risk_mask = failure_risks > failure_risks.mean()
        acc_high_risk = correct[high_risk_mask].mean()
        acc_low_risk = correct[~high_risk_mask].mean()
        
        print(f"Accuracy (high failure risk): {acc_high_risk:.4f}")
        print(f"Accuracy (low failure risk): {acc_low_risk:.4f}")
    
    def compute_early_detection_lead_time(self, predictions, labels):
        positive_indices = np.where(labels == 1)[0]
        
        if len(positive_indices) == 0:
            return 0
        
        detected_early = 0
        for idx in positive_indices:
            if predictions[idx] > 0.5:
                detected_early += 1
        
        lead_time = (detected_early / len(positive_indices)) * Config.PREDICTION_WINDOW
        
        return lead_time
    
    def full_evaluation(self):
        print("Evaluating model on test set...")
        predictions, labels, uncertainties, failure_risks = self.evaluate()
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        metrics = self.compute_metrics(predictions, labels)
        
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        self.analyze_uncertainty(predictions, labels, uncertainties)
        self.analyze_failure_risk(predictions, labels, failure_risks)
        
        lead_time = self.compute_early_detection_lead_time(predictions, labels)
        print(f"\nEstimated Early Detection Lead Time: {lead_time:.2f} hours")
        
        self.plot_roc_curve(predictions, labels, f"{Config.SAVE_PATH}roc_curve.png")
        self.plot_pr_curve(predictions, labels, f"{Config.SAVE_PATH}pr_curve.png")
        self.plot_calibration_curve(predictions, labels, save_path=f"{Config.SAVE_PATH}calibration_curve.png")
        
        print("\nPlots saved to", Config.SAVE_PATH)
        
        return metrics

def load_best_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.4f}")
    return model

if __name__ == "__main__":
    from train import main
    from model import FailureAwareTFT
    
    model, test_loader, preprocessor = main()
    
    model = load_best_model(model, f"{Config.SAVE_PATH}best_model.pt")
    
    evaluator = Evaluator(model, test_loader, Config)
    metrics = evaluator.full_evaluation()
