# Brain MRI Modality Translation with Pix2Pix GAN

This project trains a Pix2Pix GAN (U-Net generator + PatchGAN discriminator)
to synthesize T2-weighted (T2w) 2D brain MRI images from T1-normalized (T1n) images using
the BraTS 2023 dataset.

## Example Output

Below is an example showing the model’s inference output (T1 → Real T2 → Generated T2):

<p align="center"> <img src="images/example.png" width="800"> </p>

## Project structure

- `preprocess.py` — converts BraTS 2023 3D brain MRI volumes into 2D paired slices (.npz)
- `split_slices.py` — splits slices into train/val/test sets
- `dataset.py` — PyTorch Dataset/DataLoader for BraTS slices
- `model.py` — U-Net generator and PatchGAN discriminator
- `train.py` — training loop (GAN + L1 loss)
- `evaluate.py` — PSNR/SSIM evaluation
- `evaluate_plots.py` — histogram + Bland–Altman plots
- `inference_2d.py` — inference, saves T1 / Real T2 / Generated T2 images in .png format

## Environment

- OS: Windows
- Python: 3.10
- GPU: NVIDIA RTX 3080 Ti
- Framework: PyTorch

## Data

The BraTS 2023 dataset is **not** included in this repository.

To run the code, you must:
1. Request BraTS 2023 access from the organisers.
2. Place the data under `data/BraTS2023/` as described in the code.
