import torch
from torch.utils.data import Dataset
import numpy as np

class MIMICDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        static      = torch.FloatTensor(seq["static"])
        time_series = torch.FloatTensor(seq["time_series"])
        label       = torch.FloatTensor([seq["label"]])

        seq_len = time_series.shape[0]
        mask    = torch.ones(seq_len)

        # Future vitals for trajectory supervision.
        # Shape: (TRAJECTORY_HORIZON_STEPS, NUM_TARGET_VITALS)
        # future_mask = 1 where real data exists, 0 where imputed/missing.
        if "future_vitals" in seq and seq["future_vitals"] is not None:
            future_vitals = torch.FloatTensor(seq["future_vitals"])
            future_mask   = torch.FloatTensor(seq.get("future_mask", np.ones_like(seq["future_vitals"])))
        else:
            # No ground-truth future: use zeros with zero mask (no traj loss contribution)
            from config import Config
            future_vitals = torch.zeros(Config.TRAJECTORY_HORIZON_STEPS, Config.NUM_TARGET_VITALS)
            future_mask   = torch.zeros(Config.TRAJECTORY_HORIZON_STEPS, Config.NUM_TARGET_VITALS)

        return static, time_series, label, mask, seq["stay_id"], future_vitals, future_mask


def collate_fn(batch):
    statics, time_series_list, labels, masks, stay_ids, future_vitals_list, future_masks_list = zip(*batch)

    statics       = torch.stack(statics)
    labels        = torch.stack(labels)
    future_vitals = torch.stack(future_vitals_list)
    future_masks  = torch.stack(future_masks_list)

    max_len     = max(ts.shape[0] for ts in time_series_list)
    feature_dim = time_series_list[0].shape[1]

    padded_time_series = torch.zeros(len(batch), max_len, feature_dim)
    padded_masks       = torch.zeros(len(batch), max_len)

    for i, (ts, mask) in enumerate(zip(time_series_list, masks)):
        seq_len = ts.shape[0]
        padded_time_series[i, :seq_len, :] = ts
        padded_masks[i, :seq_len]          = mask

    return statics, padded_time_series, labels, padded_masks, stay_ids, future_vitals, future_masks