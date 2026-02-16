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
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_PATH_HOSP = "C:/Users/pranay/Projects/Deep Learning/mimic-iv-3.1/hosp/"
    DATA_PATH_ICU = "C:/Users/pranay/Projects/Deep Learning/mimic-iv-3.1/icu/"
    
    PREDICTION_WINDOW = 6
    LOOKBACK_WINDOW = 24
    TIME_STEP_HOURS = 1
    
    STATIC_FEATURES = ['gender', 'anchor_age', 'admission_type', 'insurance', 'marital_status', 'race']
    
    VITAL_ITEMIDS = [220045, 220179, 220180, 220210, 220277, 223761, 220050, 220051]
    LAB_ITEMIDS = [50912, 50971, 50983, 51006, 50902, 50882, 50893, 51221, 51222, 51265]
    
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10
    
    HIDDEN_SIZE = 128
    NUM_HEADS = 4
    DROPOUT = 0.1
    NUM_QUANTILES = 3
    MC_SAMPLES = 10
    
    POS_WEIGHT = 3.0
    UNCERTAINTY_WEIGHT = 0.1
    
    SAVE_PATH = "./checkpoints/"
    
set_seed(Config.SEED)