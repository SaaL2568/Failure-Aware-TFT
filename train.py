import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

from config import Config, set_seed
from data_loader import MIMICDataLoader
from preprocessing import MIMICPreprocessor
from model import FailureAwareTFT
from dataset import MIMICDataset, collate_fn
from trajectory_head import quantileLoss

set_seed(Config.SEED)


# ======================================================================
#  Loss
# ======================================================================
class MultiTaskLoss(nn.Module):
    def __init__(self, pos_weight, uncertainty_weight, trajectory_weight):
        super().__init__()
        self.bce                = nn.BCELoss(reduction="none")
        self.pos_weight         = pos_weight
        self.uncertainty_weight = uncertainty_weight
        self.trajectory_weight  = trajectory_weight
        self.quantiles          = Config.QUANTILES

    def forward(self, predictions, labels, uncertainty, trajectories,
                future_vitals, future_mask=None):
        bce_loss = self.bce(predictions, labels)
        weights  = torch.where(
            labels > 0.5,
            torch.tensor(self.pos_weight, device=labels.device),
            torch.tensor(1.0,             device=labels.device),
        )
        weighted_bce    = (bce_loss * weights).mean()
        uncertainty_reg = uncertainty.mean()
        traj_loss       = quantileLoss(trajectories, future_vitals,
                                       self.quantiles, mask=future_mask)
        total = (
            weighted_bce
            + self.uncertainty_weight * uncertainty_reg
            + self.trajectory_weight  * traj_loss
        )
        return total, weighted_bce, uncertainty_reg, traj_loss


# ======================================================================
#  Trainer
# ======================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = config.DEVICE

        self.model.to(self.device)

        self.criterion = MultiTaskLoss(
            pos_weight=config.POS_WEIGHT,
            uncertainty_weight=config.UNCERTAINTY_WEIGHT,
            trajectory_weight=config.TRAJECTORY_LOSS_WEIGHT,
        )
        self.optimizer = optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.best_val_loss    = float("inf")
        self.patience_counter = 0
        os.makedirs(config.SAVE_PATH, exist_ok=True)

    # ------------------------------------------------------------------
    #  Checkpoint helpers
    # ------------------------------------------------------------------
    def save_last_checkpoint(self, epoch, val_loss, val_auroc, val_auprc, val_f1):
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss":             val_loss,
            "val_auroc":            val_auroc,
            "val_auprc":            val_auprc,
            "val_f1":               val_f1,
            "best_val_loss":        self.best_val_loss,
            "patience_counter":     self.patience_counter,
        }, Config.LAST_CHECKPOINT)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print("No checkpoint found — starting from scratch.")
            return 0
        print(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_loss    = ckpt.get("best_val_loss",    ckpt["val_loss"])
        self.patience_counter = ckpt.get("patience_counter", 0)
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch} | "
              f"best_val_loss={self.best_val_loss:.4f} | "
              f"patience={self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
        return start_epoch

    # ------------------------------------------------------------------
    #  Epoch loops
    # ------------------------------------------------------------------
    def trainEpoch(self):
        self.model.train()
        totals = {"loss": 0.0, "bce": 0.0, "uncertainty": 0.0, "traj": 0.0}

        for batch in self.train_loader:
            static, time_series, labels, masks, _, future_vitals, future_masks = batch
            static        = static.to(self.device)
            time_series   = time_series.to(self.device)
            labels        = labels.to(self.device)
            masks         = masks.to(self.device)
            future_vitals = future_vitals.to(self.device)
            future_masks  = future_masks.to(self.device)

            self.optimizer.zero_grad()
            predictions, uncertainty, _, trajectories, _ = self.model(
                static, time_series, masks, training=True
            )
            loss, bce, unc_reg, traj = self.criterion(
                predictions, labels, uncertainty, trajectories, future_vitals, future_masks
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()

            totals["loss"]        += loss.item()
            totals["bce"]         += bce.item()
            totals["uncertainty"] += unc_reg.item()
            totals["traj"]        += traj.item()

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                static, time_series, labels, masks, _, future_vitals, future_masks = batch
                static        = static.to(self.device)
                time_series   = time_series.to(self.device)
                labels        = labels.to(self.device)
                masks         = masks.to(self.device)
                future_vitals = future_vitals.to(self.device)
                future_masks  = future_masks.to(self.device)

                predictions, uncertainty, _, trajectories, _ = self.model(
                    static, time_series, masks, training=False
                )
                loss, _, _, _ = self.criterion(
                    predictions, labels, uncertainty, trajectories, future_vitals, future_masks
                )
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds  = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        avg_loss   = total_loss / len(self.val_loader)

        auroc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
        auprc = average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0

        # Tune threshold on val set to maximise F1 (fixes hardcoded 0.5 problem)
        best_threshold = 0.5
        best_f1        = 0.0
        if len(np.unique(all_labels)) > 1:
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
            f1_scores = np.where(
                (precisions + recalls) > 0,
                2 * precisions * recalls / (precisions + recalls),
                0.0,
            )
            best_idx       = np.argmax(f1_scores)
            best_f1        = f1_scores[best_idx]
            # thresholds has one fewer element than precisions/recalls
            best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])

        return avg_loss, auroc, auprc, best_f1, best_threshold, all_preds

    # ------------------------------------------------------------------
    #  Main train loop
    # ------------------------------------------------------------------
    def train(self, resume=True):
        print(f"Training on {self.device}")
        print(f"Train: {len(self.train_loader.dataset)} | Val: {len(self.val_loader.dataset)}")

        # Check CUDA
        if not torch.cuda.is_available() and str(self.device) != "cpu":
            print("WARNING: CUDA not available — training on CPU. "
                  "Install a CUDA-enabled PyTorch build for GPU training.")

        start_epoch = 0
        if resume and Config.RESUME_TRAINING:
            start_epoch = self.load_checkpoint(Config.LAST_CHECKPOINT)

        val_scores_history = []
        best_threshold     = 0.5

        for epoch in range(start_epoch, self.config.EPOCHS):
            train_metrics = self.trainEpoch()
            val_loss, auroc, auprc, f1, best_threshold, val_preds = self.validate()

            self.scheduler.step(val_loss)
            val_scores_history.extend(val_preds.tolist())

            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"  Train — loss:{train_metrics['loss']:.4f}  bce:{train_metrics['bce']:.4f}  "
                  f"unc:{train_metrics['uncertainty']:.4f}  traj:{train_metrics['traj']:.4f}")
            print(f"  Val   — loss:{val_loss:.4f}  AUROC:{auroc:.4f}  AUPRC:{auprc:.4f}  "
                  f"F1:{f1:.4f}  threshold:{best_threshold:.3f}")

            self.save_last_checkpoint(epoch, val_loss, auroc, auprc, f1)

            if val_loss < self.best_val_loss:
                self.best_val_loss    = val_loss
                self.patience_counter = 0
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss":             val_loss,
                    "val_auroc":            auroc,
                    "val_auprc":            auprc,
                    "val_f1":               f1,
                    "best_threshold":       best_threshold,
                }, f"{self.config.SAVE_PATH}best_model.pt")
                print("  ✓ Best model checkpoint saved.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

        # Save the best tuned threshold for inference
        joblib.dump(best_threshold, f"{self.config.SAVE_PATH}best_threshold.pkl")
        print(f"Best inference threshold saved: {best_threshold:.3f}")
        return val_scores_history


# ======================================================================
#  Main pipeline
# ======================================================================
def main():
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SAVE_PATH, exist_ok=True)

    # CUDA check
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_loader  = MIMICDataLoader(Config.DATA_PATH_HOSP, Config.DATA_PATH_ICU)
    preprocessor = MIMICPreprocessor(data_loader)

    sequences = preprocessor.load_sequences()

    if sequences is None:
        print("No sequence cache — running full preprocessing pipeline...\n")

        print("Creating cohort...")
        cohort = preprocessor.create_cohort()
        print(f"Cohort: {len(cohort)} stays")

        print("Creating windowed labels...")
        labels = preprocessor.create_labels(cohort)

        print("Extracting static features...")
        static_features = preprocessor.extract_static_features(cohort)

        print("Normalising static features...")
        static_features = preprocessor.normalize_static_features(static_features, fit=True)

        print("Extracting time series + future vitals...")
        time_series_data = preprocessor.extract_time_series(cohort)

        print("Normalising time-series + future vitals...")
        time_series_data, _ = preprocessor.normalize_features(time_series_data, fit=True)

        # Save all scalers and label encoders
        preprocessor.save_artefacts(Config.SAVE_PATH)

        print("Creating sequences...")
        sequences = preprocessor.create_sequences(time_series_data, static_features, labels)
        print(f"Total sequences: {len(sequences)}")

        preprocessor.save_sequences(sequences)

    else:
        print(f"Loaded {len(sequences)} cached sequences — skipping preprocessing.\n")

    has_future = sum(1 for s in sequences
                     if s.get("future_vitals") is not None and s["future_vitals"].sum() > 0)
    print(f"Sequences with real future vital targets: {has_future}/{len(sequences)}")

    pos_count = sum(1 for s in sequences if s["label"] == 1)
    print(f"Label balance: {pos_count} positive / {len(sequences)-pos_count} negative "
          f"({100*pos_count/len(sequences):.1f}%)\n")

    # Split
    dataset    = MIMICDataset(sequences)
    train_size = int(0.70 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

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
    print(f"Parameters: {total_params:,}\n")

    trainer    = Trainer(model, train_loader, val_loader, Config)
    val_scores = trainer.train(resume=True)

    from monitoring import DriftDetector
    detector = DriftDetector()
    detector.setReference(val_scores)
    joblib.dump(detector, f"{Config.SAVE_PATH}drift_detector.pkl")
    print("Drift detector saved.")

    return model, test_loader, preprocessor


if __name__ == "__main__":
    model, test_loader, preprocessor = main()