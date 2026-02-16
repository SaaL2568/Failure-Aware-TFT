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
        
        static = torch.FloatTensor(seq['static'])
        time_series = torch.FloatTensor(seq['time_series'])
        label = torch.FloatTensor([seq['label']])
        
        seq_len = time_series.shape[0]
        mask = torch.ones(seq_len)
        
        return static, time_series, label, mask, seq['stay_id']

def collate_fn(batch):
    statics, time_series_list, labels, masks, stay_ids = zip(*batch)
    
    statics = torch.stack(statics)
    labels = torch.stack(labels)
    
    max_len = max([ts.shape[0] for ts in time_series_list])
    feature_dim = time_series_list[0].shape[1]
    
    padded_time_series = torch.zeros(len(batch), max_len, feature_dim)
    padded_masks = torch.zeros(len(batch), max_len)
    
    for i, (ts, mask) in enumerate(zip(time_series_list, masks)):
        seq_len = ts.shape[0]
        padded_time_series[i, :seq_len, :] = ts
        padded_masks[i, :seq_len] = mask
    
    return statics, padded_time_series, labels, padded_masks, stay_ids