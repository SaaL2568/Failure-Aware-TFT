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

    DATA_PATH_HOSP = "C:/Users/pranay/Projects/Deep Learning/mimic-iv-3.1/hosp/"
    DATA_PATH_ICU  = "C:/Users/pranay/Projects/Deep Learning/mimic-iv-3.1/icu/"

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

    # --- item id lists ---
    VITAL_ITEMIDS = [220045, 220179, 220180, 220210, 220277, 223761, 220050, 220051]
    LAB_ITEMIDS   = [50912, 50971, 50983, 51006, 50902, 50882, 50893, 51221, 51222, 51265]

    # --- trajectory prediction ---
    TARGET_VITALS             = ["heart_rate", "sbp", "dbp", "resp_rate", "spo2"]
    NUM_TARGET_VITALS         = len(TARGET_VITALS)
    QUANTILES                 = [0.1, 0.5, 0.9]
    NUM_QUANTILES             = len(QUANTILES)
    TRAJECTORY_HORIZON_STEPS  = 48          # 4 hours @ 5-min steps
    TRAJECTORY_LOSS_WEIGHT    = 0.4

    # --- vital item id → name mapping (used for trajectory target extraction) ---
    VITAL_ITEMID_TO_TARGET = {
        220045: "heart_rate",
        220179: "sbp",
        220180: "dbp",
        220210: "resp_rate",
        220277: "spo2",
    }

    # --- model architecture ---
    HIDDEN_SIZE  = 64
    NUM_HEADS    = 4
    DROPOUT      = 0.3
    MC_SAMPLES   = 10

    # --- training ---
    BATCH_SIZE               = 32
    EPOCHS                   = 100
    LEARNING_RATE            = 1e-3
    WEIGHT_DECAY             = 1e-3
    EARLY_STOPPING_PATIENCE  = 10
    POS_WEIGHT               = 3.0
    UNCERTAINTY_WEIGHT       = 0.1
    GRAD_CLIP                = 1.0

    # --- data sampling ---
    SAMPLE_SIZE = 5000

    # --- paths ---
    SAVE_PATH     = "./checkpoints/"
    CACHE_DIR     = "./cache/"
    FEATURE_CACHE = "./cache/features.parquet"
    LABEL_CACHE   = "./cache/labels.parquet"
    SEQ_CACHE     = "./cache/sequences.pkl"      # NEW: cached sequences

    # --- checkpoint resumption ---
    # Set RESUME_TRAINING = True to auto-resume from last checkpoint
    RESUME_TRAINING = True
    LAST_CHECKPOINT = "./checkpoints/last_checkpoint.pt"

    # --- inference / API ---
    RISK_THRESHOLD = 0.5
    API_HOST       = "0.0.0.0"
    API_PORT       = 8000

    # --- drift detection ---
    DRIFT_WINDOW_SIZE = 500
    DRIFT_KS_ALPHA    = 0.05

set_seed(Config.SEED)