import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH_HOSP = "C:/Users/itism/Documents/CHRIST NOTES/Deep Learning/PROJECT/iews/mimic-iv-3.1/hosp/"
    DATA_PATH_ICU  = "C:/Users/itism/Documents/CHRIST NOTES/Deep Learning/PROJECT/iews/mimic-iv-3.1/icu/"

    # --- time windows ---
    PREDICTION_HORIZON_HOURS   = 4
    LOOKBACK_WINDOW_HOURS      = 24
    OBSERVATION_WINDOW_MINUTES = 30
    PREDICTION_STEP_MINUTES    = 5
    TIME_STEP_HOURS            = 1

    # --- static feature columns expected after preprocessing ---
    STATIC_FEATURES = [
        "gender", "anchor_age", "admission_type",
        "insurance", "marital_status", "race",
    ]

    # --- item id lists (kept for backward compat with original preprocessing) ---
    VITAL_ITEMIDS = [220045, 220179, 220180, 220210, 220277, 223761, 220050, 220051]
    LAB_ITEMIDS   = [50912, 50971, 50983, 51006, 50902, 50882, 50893, 51221, 51222, 51265]

    # --- trajectory prediction (TFT-multi head) ---
    TARGET_VITALS          = ["heart_rate", "sbp", "dbp", "resp_rate", "spo2"]
    NUM_TARGET_VITALS      = len(TARGET_VITALS)
    QUANTILES              = [0.1, 0.5, 0.9]
    NUM_QUANTILES          = len(QUANTILES)
    TRAJECTORY_HORIZON_STEPS = 48          # 4 hours at 5-min steps
    TRAJECTORY_LOSS_WEIGHT    = 0.4        # lambda blending classification vs trajectory loss

    # --- model architecture ---
    HIDDEN_SIZE  = 128
    NUM_HEADS    = 4
    DROPOUT      = 0.1
    MC_SAMPLES   = 10

    # --- training ---
    BATCH_SIZE               = 32
    EPOCHS                   = 100
    LEARNING_RATE            = 1e-3
    WEIGHT_DECAY             = 1e-5
    EARLY_STOPPING_PATIENCE  = 10
    POS_WEIGHT               = 3.0
    UNCERTAINTY_WEIGHT       = 0.1
    GRAD_CLIP                = 1.0

    # --- data sampling (for memory-constrained runs) ---
    SAMPLE_SIZE = 5000

    # --- paths ---
    SAVE_PATH        = "./checkpoints/"
    FEATURE_CACHE    = "./cache/features.parquet"
    LABEL_CACHE      = "./cache/labels.parquet"

    # --- inference / API ---
    RISK_THRESHOLD = 0.5       # willDeteriorate = True when riskScore >= this
    API_HOST       = "0.0.0.0"
    API_PORT       = 8000

    # --- drift detection ---
    DRIFT_WINDOW_SIZE = 500    # number of recent predictions to monitor
    DRIFT_KS_ALPHA    = 0.05   # p-value threshold for KS test alert

set_seed(Config.SEED)
