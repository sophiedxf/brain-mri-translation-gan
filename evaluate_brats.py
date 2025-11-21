import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset_brats import BraTSMRIDataset
from model_brats import UNetGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_ds = BraTSMRIDataset(r"data\BraTS2023_slices\test")
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

G = UNetGenerator().to(device)
G.load_state_dict(torch.load(r"checkpoints_brats\G_epoch50.pth", map_location=device))
G.eval()

psnrs, ssim_vals = [], []

with torch.no_grad():
    for t1, t2 in test_loader:
        t1, t2 = t1.to(device), t2.to(device)
        fake_t2 = G(t1)

        for i in range(t1.size(0)):
            gt = t2[i, 0].cpu().numpy()
            pred = fake_t2[i, 0].cpu().numpy()
            psnrs.append(peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min()))
            ssim_vals.append(structural_similarity(gt, pred, data_range=gt.max() - gt.min()))

print("Mean PSNR:", float(np.mean(psnrs)))
print("Mean SSIM:", float(np.mean(ssim_vals)))
