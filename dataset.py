import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BraTSMRIDataset(Dataset):
    def __init__(self, root_dir: str):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        t1 = torch.from_numpy(data["t1n"].astype(np.float32)).unsqueeze(0)
        t2 = torch.from_numpy(data["t2w"].astype(np.float32)).unsqueeze(0)
        return t1, t2

def get_loaders(batch_size=8):
    train_ds = BraTSMRIDataset(r"data\BraTS2023_slices\train")
    val_ds   = BraTSMRIDataset(r"data\BraTS2023_slices\val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
