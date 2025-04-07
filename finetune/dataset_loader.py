import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, path, mode):
        super().__init__()
        self.mode = mode

        with open(path, 'r') as f:
            self.data = [line.strip() for line in f.readlines()] 
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        npy_path = self.data[idx]
        sample = np.load(npy_path, allow_pickle=True).item()
        image = torch.from_numpy(sample['image']).float()
        return image