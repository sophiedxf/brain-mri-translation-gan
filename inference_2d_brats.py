import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from model_brats import UNetGenerator

# ---------------------------------------------------
# Load trained generator
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

G = UNetGenerator().to(device)
G.load_state_dict(torch.load("checkpoints_brats/G_epoch50.pth", map_location=device))
G.eval()


# ---------------------------------------------------
# Function to infer and save triplet image
# ---------------------------------------------------

def infer_triplet(npz_path, out_png_path=None):
    """
    Load .npz (t1n, t2w), generate fake T2,
    and save a PNG with T1 | Real T2 | Generated T2
    using EXACTLY the same plotting code as in train_brats.py.
    """
    data = np.load(npz_path)

    # These should match what dataset_brats returns: t1n / t2w or t1 / t2
    # Adjust keys if needed.
    real_t1 = data["t1n"].astype(np.float32)
    real_t2 = data["t2w"].astype(np.float32)

    # Convert to tensor: (1,1,H,W)
    t1_t = torch.from_numpy(real_t1).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_t2 = G(t1_t)

    gen_t2 = fake_t2[0, 0].cpu().numpy()  # same style as train_brats

    # --------- PLOTTING: COPY-PASTE FROM train_brats.py ----------
    if out_png_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(real_t1, cmap="gray"); axes[0].set_title("T1")
        axes[1].imshow(real_t2, cmap="gray"); axes[1].set_title("Real T2")
        axes[2].imshow(gen_t2,  cmap="gray"); axes[2].set_title("Generated T2")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        # NOTE: no bbox_inches / pad_inches here, to match train_brats.py
        plt.savefig(out_png_path)
        plt.close()
    # -------------------------------------------------------------

    return gen_t2


# ---------------------------------------------------
# OPTION A: run inference on entire folder
# ---------------------------------------------------

def run_folder(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} slices in {in_dir}")

    for npz_path in npz_files:
        out_png = out_dir / (npz_path.stem + "_triplet.png")
        infer_triplet(npz_path, out_png)

    print(f"Saved triplet PNGs to {out_dir}")


# ---------------------------------------------------
# OPTION B: choose MULTIPLE .npz files manually
# ---------------------------------------------------

if __name__ == "__main__":
    print("Enter paths to .npz files, one per line.")
    print("When you're done, press ENTER on an empty line.\n")

    selected_files = []
    while True:
        line = input().strip()
        if line == "":
            break
        selected_files.append(line)

    if not selected_files:
        print("No files entered. Exiting.")
    else:
        out_dir = Path("generated_triplets/selected")
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in selected_files:
            npz_path = Path(f)
            if not npz_path.is_file():
                print(f"WARNING: {f} does not exist or is not a file, skipping.")
                continue

            out_png = out_dir / (npz_path.stem + "_triplet.png")
            infer_triplet(npz_path, out_png)
            print("Saved:", out_png)

        print(f"\nDone. Saved triplet PNGs to {out_dir}")
