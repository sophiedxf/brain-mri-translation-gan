import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_brats import BraTSMRIDataset
from model_brats import UNetGenerator

# -----------------------------
# 1. Setup: device, paths, dirs
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Test dataset (npz slices)
test_ds = BraTSMRIDataset(r"data\BraTS2023_slices\test")
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

# Load trained generator
checkpoint_path = r"checkpoints_brats\G_epoch50.pth"  # adjust if needed
print("Loading generator from:", checkpoint_path)

G = UNetGenerator().to(device)
G.load_state_dict(torch.load(checkpoint_path, map_location=device))
G.eval()

# Directory to save plots
out_dir = pathlib.Path("eval_plots")
out_dir.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# 2. Collect real & generated intensities (brain only)
# -------------------------------------------------

all_real = []
all_fake = []

# Threshold to define "background". Real T2w brain voxels are non-zero.
# You can tweak this if needed.
BACKGROUND_EPS = 1e-6

with torch.no_grad():
    for t1, t2 in test_loader:
        t1 = t1.to(device)
        t2 = t2.to(device)

        fake_t2 = G(t1)

        # Move to CPU and convert to numpy: shape (B, 1, H, W)
        real_np = t2.cpu().numpy()
        fake_np = fake_t2.cpu().numpy()

        # Build a brain mask from real T2: non-zero (or > small eps) = brain
        # This assumes background is exactly or very close to zero.
        brain_mask = np.abs(real_np) > BACKGROUND_EPS  # same shape as real_np

        # Apply mask to both real and fake
        real_brain = real_np[brain_mask]
        fake_brain = fake_np[brain_mask]

        # Skip batches with empty mask (shouldn't happen, but just in case)
        if real_brain.size == 0:
            continue

        all_real.append(real_brain.reshape(-1))
        all_fake.append(fake_brain.reshape(-1))

# Concatenate into single long vectors
all_real = np.concatenate(all_real, axis=0)
all_fake = np.concatenate(all_fake, axis=0)

print("Collected brain-only pixels (non-background):")
print("  Real:", all_real.shape)
print("  Fake:", all_fake.shape)


# ---------------------------------
# 3. Intensity histogram comparison
# ---------------------------------

def plot_intensity_histogram(real_vals, fake_vals, out_path, bins=100):
    """
    Compare intensity distributions of real vs generated images (brain only).
    Saves a PNG at out_path.
    """
    plt.figure(figsize=(8, 5))

    plt.hist(real_vals, bins=bins, density=True, alpha=0.5, label="Real T2w (brain)")
    plt.hist(fake_vals, bins=bins, density=True, alpha=0.5, label="Generated T2w (brain)")

    plt.xlabel("Intensity")
    plt.ylabel("Probability density")
    plt.title("Intensity Histogram (Brain Only): Real vs Generated T2w")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved intensity histogram to: {out_path}")


hist_path = out_dir / "histogram_brain_only_real_vs_generated_T2w.png"
plot_intensity_histogram(all_real, all_fake, hist_path)


# -------------------------
# 4. Bland–Altman plot
# -------------------------

def plot_bland_altman(real_vals, fake_vals, out_path, sample_size=100_000):
    """
    Create a Bland–Altman plot between real and fake intensities (brain only).
    Saves a PNG at out_path.

    sample_size: we'll subsample pixels if there are too many,
    to keep the plot readable and fast.
    """

    real_vals = real_vals.reshape(-1)
    fake_vals = fake_vals.reshape(-1)

    # Optional: subsample if there are too many pixels
    n = real_vals.shape[0]
    if n > sample_size:
        idx = np.random.choice(n, size=sample_size, replace=False)
        real_vals = real_vals[idx]
        fake_vals = fake_vals[idx]
        print(f"Subsampled {n} brain pixels to {sample_size} for Bland–Altman plot.")

    # Compute mean and difference per pixel
    mean_vals = (real_vals + fake_vals) / 2.0
    diff_vals = fake_vals - real_vals  # bias: generated - real

    # Compute statistics
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    # Make plot
    plt.figure(figsize=(8, 5))
    plt.scatter(mean_vals, diff_vals, s=1, alpha=0.3)
    plt.axhline(mean_diff, linestyle="--", label=f"Mean diff = {mean_diff:.3f}")
    plt.axhline(upper_limit, linestyle="--", label=f"+1.96 SD = {upper_limit:.3f}")
    plt.axhline(lower_limit, linestyle="--", label=f"-1.96 SD = {lower_limit:.3f}")

    plt.xlabel("Mean intensity (Real & Generated, brain only)")
    plt.ylabel("Difference (Generated - Real)")
    plt.title("Bland–Altman Plot (Brain Only): Real vs Generated T2w")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved Bland–Altman plot to: {out_path}")
    print(f"  Mean diff: {mean_diff:.4f}, Std diff: {std_diff:.4f}")


ba_path = out_dir / "bland_altman_brain_only_real_vs_generated_T2w.png"
plot_bland_altman(all_real, all_fake, ba_path)

print("Done. All brain-only evaluation plots saved in:", out_dir)
