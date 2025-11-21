import os
import glob
import pathlib

import nibabel as nib
import numpy as np
from skimage.transform import resize

# ---------- 1. Find BraTS subjects and pair t1n/t2w ----------

brats_root = r"data\BraTS2023"  # adjust if your folder name differs

# Each BraTS-GLI-* directory is one subject
subjects = sorted(glob.glob(os.path.join(brats_root, "BraTS-GLI-*")))
print("Found", len(subjects), "subject directories")

pairs = []  # list of (t1n_path, t2w_path)

for sub in subjects:
    # We search recursively inside subject folder
    # Expected filenames:
    #   *-t1n.nii.gz
    #   *-t2w.nii.gz
    t1n = glob.glob(os.path.join(sub, "*-t1n.nii.gz"))
    t2w = glob.glob(os.path.join(sub, "*t2w.nii.gz"))

    if len(t1n) == 1 and len(t2w) == 1:
        pairs.append((t1n[0], t2w[0]))
    else:
        print(f"Skipping {sub}: found {len(t1n)} t1n, {len(t2w)} t2w")

print("Found", len(pairs), "T1n/T2w paired subjects")


# ---------- 2. Normalisation & resizing helpers ----------

def normalise_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    low, high = np.percentile(vol, (1, 99))
    vol = np.clip(vol, low, high)
    vol -= vol.mean()
    vol /= (vol.std() + 1e-8)
    return vol


def resize_volume(vol: np.ndarray, new_shape=(256, 256)) -> np.ndarray:
    """
    Assume vol shape is (X, Y, Z). Resize each axial slice (X,Y) -> new_shape.
    """
    X, Y, Z = vol.shape
    out = np.zeros((new_shape[0], new_shape[1], Z), dtype=np.float32)
    for z in range(Z):
        out[:, :, z] = resize(vol[:, :, z], new_shape, anti_aliasing=True)
    return out


# ---------- 3. Convert all volumes into 2D paired slices ----------

out_root = pathlib.Path(r"data\BraTS2023_slices")
(out_root / "all").mkdir(parents=True, exist_ok=True)

slice_id = 0

for t1n_path, t2w_path in pairs:
    print("Processing:", t1n_path, "|", t2w_path)

    t1n_vol = nib.load(t1n_path).get_fdata()
    t2w_vol = nib.load(t2w_path).get_fdata()

    t1n = resize_volume(normalise_volume(t1n_vol), (256, 256))
    t2w = resize_volume(normalise_volume(t2w_vol), (256, 256))

    X, Y, Z = t1n.shape

    # Keep middle 60% of slices (avoid top/bottom where brain is tiny)
    z_start = int(0.2 * Z)
    z_end = int(0.8 * Z)

    for z in range(z_start, z_end):
        t1_slice = t1n[:, :, z]
        t2_slice = t2w[:, :, z]
        np.savez_compressed(
            out_root / "all" / f"slice_{slice_id:06d}.npz",
            t1n=t1_slice,
            t2w=t2_slice,
        )
        slice_id += 1

print("Saved", slice_id, "paired slices to", out_root / "all")
