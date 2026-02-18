import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from config import Config, set_seed
from data_loader import MIMICDataLoader
from preprocessing import MIMICPreprocessor
from model import FailureAwareTFT
from dataset import MIMICDataset, collate_fn

set_seed(Config.SEED)

class UncertaintyRegularizedLoss(nn.Module):
    def __init__(self, pos_weight, uncertainty_weight):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.pos_weight = pos_weight
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, predictions, labels, uncertainty):
        bce_loss = self.bce(predictions, labels)
        
        weights = torch.where(labels > 0.5, self.pos_weight, 1.0)
        weighted_bce = (bce_loss * weights).mean()
        
        uncertainty_reg = uncertainty.mean()
        
        total_loss = weighted_bce + self.uncertainty_weight * uncertainty_reg
        
        return total_loss, weighted_bce, uncertainty_reg

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        
        self.model.to(self.device)
        
        self.criterion = UncertaintyRegularizedLoss(config.POS_WEIGHT, config.UNCERTAINTY_WEIGHT)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        os.makedirs(config.SAVE_PATH, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_bce = 0
        total_uncertainty = 0
        
        for batch_idx, (static, time_series, labels, masks, _) in enumerate(self.train_loader):
            static = static.to(self.device)
            time_series = time_series.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions, uncertainty, failure_risk, _ = self.model(
                static, time_series, masks, training=True
            )
            
            loss, bce_loss, uncertainty_reg = self.criterion(predictions, labels, uncertainty)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_uncertainty += uncertainty_reg.item()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_bce = total_bce / len(self.train_loader)
        avg_uncertainty = total_uncertainty / len(self.train_loader)
        
        return avg_loss, avg_bce, avg_uncertainty
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        all_failure_risks = []
        
        with torch.no_grad():
            for static, time_series, labels, masks, _ in self.val_loader:
                static = static.to(self.device)
                time_series = time_series.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                
                predictions, uncertainty, failure_risk, _ = self.model(
                    static, time_series, masks, training=False
                )
                
                loss, _, _ = self.criterion(predictions, labels, uncertainty)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())
                all_failure_risks.extend(failure_risk.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        
        auroc = roc_auc_score(all_labels, all_predictions)
        auprc = average_precision_score(all_labels, all_predictions)
        
        pred_binary = (all_predictions > 0.5).astype(int)
        f1 = f1_score(all_labels, pred_binary)
        
        return avg_loss, auroc, auprc, f1
    
    def train(self):
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config.EPOCHS):
            train_loss, train_bce, train_uncertainty = self.train_epoch()
            val_loss, val_auroc, val_auprc, val_f1 = self.validate()
            
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Uncertainty: {train_uncertainty:.4f})")
            print(f"  Val Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}, F1: {val_f1:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auroc': val_auroc,
                    'val_auprc': val_auprc,
                    'val_f1': val_f1
                }, f"{self.config.SAVE_PATH}best_model.pt")
                print(f"  Model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

def main():
    data_loader = MIMICDataLoader(Config.DATA_PATH_HOSP, Config.DATA_PATH_ICU)
    preprocessor = MIMICPreprocessor(data_loader)
    
    print("Creating cohort...")
    cohort = preprocessor.create_cohort()
    print(f"Cohort size: {len(cohort)}")
    
    print("Creating labels...")
    labels = preprocessor.create_labels(cohort)
    print(f"Positive samples: {labels['label'].sum()}/{len(labels)}")
    
    print("Extracting static features...")
    static_features = preprocessor.extract_static_features(cohort)
    
    print("Extracting time series...")
    time_series_data = preprocessor.extract_time_series(cohort)
    
    print("Normalizing features...")
    time_series_data, feature_names = preprocessor.normalize_features(time_series_data, fit=True)
    
    print("Creating sequences...")
    sequences = preprocessor.create_sequences(time_series_data, static_features, labels)
    print(f"Total sequences: {len(sequences)}")
    
    dataset = MIMICDataset(sequences)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    labels_array = np.array([s['label'] for s in sequences])
    
    if labels_array.sum() < 10:
        print(f"WARNING: Only {labels_array.sum()} positive samples found!")
        print("Consider:")
        print("  1. Increasing SAMPLE_SIZE in config.py")
        print("  2. Adjusting label criteria")
        print("  3. Using different prediction task")
    
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(dataset))
    
    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=Config.SEED,
        stratify=labels_array if labels_array.sum() >= 2 else None
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=Config.SEED,
        stratify=labels_array[train_val_idx] if labels_array[train_val_idx].sum() >= 2 else None
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    print(f"Train positive: {labels_array[train_idx].sum()}/{len(train_idx)}")
    print(f"Val positive: {labels_array[val_idx].sum()}/{len(val_idx)}")
    print(f"Test positive: {labels_array[test_idx].sum()}/{len(test_idx)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    static_size = sequences[0]['static'].shape[0]
    time_series_size = sequences[0]['time_series'].shape[1]
    
    print(f"Static size: {static_size}, Time series size: {time_series_size}")
    
    model = FailureAwareTFT(
        static_size=static_size,
        time_series_size=time_series_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        num_quantiles=Config.NUM_QUANTILES,
        mc_samples=Config.MC_SAMPLES
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = Trainer(model, train_loader, val_loader, Config)
    trainer.train()
    
    return model, test_loader, preprocessor

if __name__ == "__main__":
    model, test_loader, preprocessor = main()
