import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

H, W = 256, 256
C = 3

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
        sample = np.load(npy_path)  # shape: (D, H, W)

        image = torch.from_numpy(sample).float()  # (D, H, W)

        # Add channel dim: (D, 1, H, W)
        image = image.unsqueeze(1)

        # Repeat along channel: (D, 3, H, W)
        image = image.repeat(1, C, 1, 1)

        # Resize H, W if needed
        if image.shape[-2:] != (H, W):
            image = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=False)

        return image  # shape: (D, 3, 256, 256)

def get_dataloader(path_to_txt, mode="train", batch_size=4, shuffle=True, num_workers=4):
    dataset = CustomDataset(path=path_to_txt, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if mode == "train" else False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader