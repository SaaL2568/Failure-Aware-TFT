import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from config import Config, set_seed
from data_loader import MIMICDataLoader
from preprocessing import MIMICPreprocessor
from model import FailureAwareTFT
from dataset import MIMICDataset, collate_fn
from trajectory_head import quantileLoss

set_seed(Config.SEED)


class MultiTaskLoss(nn.Module):
    """
    Combined loss = BCE (classification) + uncertainty regularization + quantile loss (trajectory)
    """

    def __init__(self, pos_weight, uncertainty_weight, trajectory_weight):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.pos_weight = pos_weight
        self.uncertainty_weight = uncertainty_weight
        self.trajectory_weight = trajectory_weight
        self.quantiles = Config.QUANTILES

    def forward(self, predictions, labels, uncertainty, trajectories, future_vitals, future_mask=None):
        bce_loss = self.bce(predictions, labels)
        weights = torch.where(labels > 0.5,
                              torch.tensor(self.pos_weight, device=labels.device),
                              torch.tensor(1.0, device=labels.device))
        weighted_bce = (bce_loss * weights).mean()

        uncertainty_reg = uncertainty.mean()

        traj_loss = quantileLoss(trajectories, future_vitals, self.quantiles, mask=future_mask)

        total = (
            weighted_bce
            + self.uncertainty_weight * uncertainty_reg
            + self.trajectory_weight * traj_loss
        )
        return total, weighted_bce, uncertainty_reg, traj_loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE

        self.model.to(self.device)

        self.criterion = MultiTaskLoss(
            pos_weight=config.POS_WEIGHT,
            uncertainty_weight=config.UNCERTAINTY_WEIGHT,
            trajectory_weight=config.TRAJECTORY_LOSS_WEIGHT,
        )
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        os.makedirs(config.SAVE_PATH, exist_ok=True)

    def _makeDummyFutureVitals(self, batch_size, device):
        """
        Placeholder for future vital targets when not available in dataset.
        Returns zeros with a mask of zeros (contributes nothing to loss).
        """
        future = torch.zeros(
            batch_size, Config.TRAJECTORY_HORIZON_STEPS, Config.NUM_TARGET_VITALS, device=device
        )
        mask = torch.zeros_like(future)
        return future, mask

    def trainEpoch(self):
        self.model.train()
        totals = {"loss": 0, "bce": 0, "uncertainty": 0, "traj": 0}

        for static, time_series, labels, masks, _ in self.train_loader:
            static = static.to(self.device)
            time_series = time_series.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)
            batch_size = static.shape[0]

            future_vitals, future_mask = self._makeDummyFutureVitals(batch_size, self.device)

            self.optimizer.zero_grad()

            predictions, uncertainty, failure_risk, trajectories, _ = self.model(
                static, time_series, masks, training=True
            )

            loss, bce, unc_reg, traj = self.criterion(
                predictions, labels, uncertainty, trajectories, future_vitals, future_mask
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()

            totals["loss"] += loss.item()
            totals["bce"] += bce.item()
            totals["uncertainty"] += unc_reg.item()
            totals["traj"] += traj.item()

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels, all_unc = [], [], []

        with torch.no_grad():
            for static, time_series, labels, masks, _ in self.val_loader:
                static = static.to(self.device)
                time_series = time_series.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                batch_size = static.shape[0]

                future_vitals, future_mask = self._makeDummyFutureVitals(batch_size, self.device)

                predictions, uncertainty, _, trajectories, _ = self.model(
                    static, time_series, masks, training=False
                )

                loss, _, _, _ = self.criterion(
                    predictions, labels, uncertainty, trajectories, future_vitals, future_mask
                )
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_unc.extend(uncertainty.cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        avg_loss = total_loss / len(self.val_loader)

        auroc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
        auprc = average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)

        return avg_loss, auroc, auprc, f1, all_preds

    def train(self):
        print(f"Training on {self.device}")
        print(f"Train: {len(self.train_loader.dataset)} | Val: {len(self.val_loader.dataset)}")

        val_scores_history = []

        for epoch in range(self.config.EPOCHS):
            train_metrics = self.trainEpoch()
            val_loss, auroc, auprc, f1, val_preds = self.validate()

            self.scheduler.step(val_loss)
            val_scores_history.extend(val_preds.tolist())

            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"  Train — loss:{train_metrics['loss']:.4f} bce:{train_metrics['bce']:.4f} "
                  f"unc:{train_metrics['uncertainty']:.4f} traj:{train_metrics['traj']:.4f}")
            print(f"  Val   — loss:{val_loss:.4f} AUROC:{auroc:.4f} AUPRC:{auprc:.4f} F1:{f1:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_auroc": auroc,
                    "val_auprc": auprc,
                    "val_f1": f1,
                }, f"{self.config.SAVE_PATH}best_model.pt")
                print("  Checkpoint saved.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

        return val_scores_history


def main():
    data_loader = MIMICDataLoader(Config.DATA_PATH_HOSP, Config.DATA_PATH_ICU)
    preprocessor = MIMICPreprocessor(data_loader)

    print("Creating cohort...")
    cohort = preprocessor.create_cohort()
    print(f"Cohort: {len(cohort)} stays")

    print("Creating labels...")
    labels = preprocessor.create_labels(cohort)
    pos = labels["label"].sum()
    print(f"Positive: {int(pos)}/{len(labels)} ({100*pos/len(labels):.2f}%)")

    print("Extracting static features...")
    static_features = preprocessor.extract_static_features(cohort)

    print("Extracting time series...")
    time_series_data = preprocessor.extract_time_series(cohort)

    print("Normalizing features...")
    time_series_data, feature_names = preprocessor.normalize_features(time_series_data, fit=True)

    joblib.dump(preprocessor.scalers["time_series"], f"{Config.SAVE_PATH}scaler.pkl")
    print("Scaler saved.")

    print("Creating sequences...")
    sequences = preprocessor.create_sequences(time_series_data, static_features, labels)
    print(f"Total sequences: {len(sequences)}")

    dataset = MIMICDataset(sequences)
    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    static_size      = sequences[0]["static"].shape[0]
    time_series_size = sequences[0]["time_series"].shape[1]
    print(f"Static size: {static_size} | TS size: {time_series_size}")

    model = FailureAwareTFT(
        static_size=static_size,
        time_series_size=time_series_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        num_quantiles=Config.NUM_QUANTILES,
        mc_samples=Config.MC_SAMPLES,
        num_target_vitals=Config.NUM_TARGET_VITALS,
        trajectory_horizon_steps=Config.TRAJECTORY_HORIZON_STEPS,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    trainer = Trainer(model, train_loader, val_loader, Config)
    val_scores = trainer.train()

    from monitoring import DriftDetector
    detector = DriftDetector()
    detector.setReference(val_scores)
    joblib.dump(detector, f"{Config.SAVE_PATH}drift_detector.pkl")
    print("Drift detector saved.")

    return model, test_loader, preprocessor


if __name__ == "__main__":
    model, test_loader, preprocessor = main()
