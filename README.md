# Failure-Aware Temporal Fusion Transformers for ICU Critical State Detection

Production-ready implementation of a Failure-Aware Temporal Fusion Transformer (TFT) model for early detection of critical state transitions in Intensive Care Units using MIMIC-IV v3.1 dataset.

## Project Structure

```
failure_aware_tft/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # MIMIC-IV data loading utilities
├── preprocessing.py       # Data preprocessing and feature engineering
├── model.py              # TFT and Failure-Aware module implementation
├── dataset.py            # PyTorch Dataset and DataLoader utilities
├── train.py              # Training loop and model training
├── evaluate.py           # Evaluation metrics and analysis
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

### Model Architecture
- **Temporal Fusion Transformer (TFT)** with:
  - Variable Selection Networks
  - Gated Residual Networks (GRN)
  - LSTM encoder/decoder
  - Multi-Head Attention mechanism
  - Static covariate encoders

### Failure-Aware Module
- Monte Carlo dropout for uncertainty estimation
- Confidence-aware gating mechanism
- Dynamic attention reweighting based on entropy
- Outputs:
  - Prediction probability
  - Uncertainty score
  - Failure risk indicator

### Data Processing
- Automatic merging of MIMIC-IV tables
- Handles missing values and irregular time sampling
- Variable sequence length support with masking
- Sliding window generation for time-series
- Static and time-varying feature extraction

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set:
- Data paths (MIMIC-IV location)
- Prediction and lookback windows
- Model hyperparameters
- Training parameters

Key parameters:
```python
PREDICTION_WINDOW = 6      # Hours ahead to predict
LOOKBACK_WINDOW = 24       # Hours of history to use
HIDDEN_SIZE = 128          # Model hidden dimension
NUM_HEADS = 4              # Attention heads
BATCH_SIZE = 32
EPOCHS = 100
```

## Usage

### Training

```bash
python train.py
```

This will:
1. Load and preprocess MIMIC-IV data
2. Create train/val/test splits
3. Train the model with early stopping
4. Save best model checkpoint to `./checkpoints/`

### Evaluation

```bash
python evaluate.py
```

Outputs:
- AUROC, AUPRC, F1 scores
- Sensitivity, Specificity, PPV, NPV
- Uncertainty analysis
- Failure risk analysis
- Early detection lead time
- ROC curve, PR curve, calibration plots

## Data Requirements

MIMIC-IV v3.1 dataset with the following tables:
- `hosp/patients.csv`
- `hosp/admissions.csv`
- `hosp/labevents.csv`
- `icu/icustays.csv`
- `icu/chartevents.csv`

Update `DATA_PATH_HOSP` and `DATA_PATH_ICU` in `config.py` with your data location.

## Model Output

For each prediction, the model outputs:
1. **Prediction**: Binary probability of critical event
2. **Uncertainty**: Model confidence (lower is better)
3. **Failure Risk**: Combined uncertainty and entropy metric

## Training Features

- Class imbalance handling with weighted loss
- Early stopping with patience
- Gradient clipping
- GPU support (automatic detection)
- Reproducible results (seed setting)
- Checkpoint saving

## Evaluation Metrics

- **Classification**: AUROC, AUPRC, F1, Sensitivity, Specificity
- **Calibration**: Calibration curves
- **Uncertainty**: Accuracy stratified by uncertainty levels
- **Clinical**: Early detection lead time estimation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- MIMIC-IV v3.1 dataset access

## Citation

If you use this code, please cite:
- MIMIC-IV dataset: https://physionet.org/content/mimiciv/
- Temporal Fusion Transformers: Lim et al. (2021)

## License

This code is provided for research purposes. MIMIC-IV data requires credentialed access through PhysioNet.
