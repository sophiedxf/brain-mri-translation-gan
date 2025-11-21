import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_brats import get_loaders
from model_brats import UNetGenerator, PatchDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader = get_loaders(batch_size=8)

G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()
lambda_l1 = 100.0

os.makedirs("checkpoints_brats", exist_ok=True)
os.makedirs("samples_brats", exist_ok=True)

def train_one_epoch(epoch):
    G.train()
    D.train()
    for i, (t1, t2) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        t1, t2 = t1.to(device), t2.to(device)

        # ---- Train D ----
        optim_D.zero_grad()
        with torch.no_grad():
            fake_t2_detached = G(t1)

        real_pair = torch.cat([t1, t2], dim=1)
        fake_pair = torch.cat([t1, fake_t2_detached], dim=1)

        pred_real = D(real_pair)
        pred_fake = D(fake_pair)

        real_labels = torch.ones_like(pred_real)
        fake_labels = torch.zeros_like(pred_fake)

        loss_D_real = criterion_gan(pred_real, real_labels)
        loss_D_fake = criterion_gan(pred_fake, fake_labels)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        optim_D.step()

        # ---- Train G ----
        optim_G.zero_grad()
        fake_t2 = G(t1)
        fake_pair_for_G = torch.cat([t1, fake_t2], dim=1)
        pred_fake_for_G = D(fake_pair_for_G)

        loss_G_gan = criterion_gan(pred_fake_for_G, real_labels)
        loss_G_l1 = criterion_l1(fake_t2, t2) * lambda_l1
        loss_G = loss_G_gan + loss_G_l1
        loss_G.backward()
        optim_G.step()

        if i % 100 == 0:
            print(f"Iter {i}: Loss_D={loss_D.item():.4f}, Loss_G={loss_G.item():.4f}")

def save_samples(epoch):
    G.eval()
    with torch.no_grad():
        for i, (t1, t2) in enumerate(val_loader):
            t1, t2 = t1.to(device), t2.to(device)
            fake_t2 = G(t1)

            real_t1 = t1[0, 0].cpu().numpy()
            real_t2 = t2[0, 0].cpu().numpy()
            gen_t2  = fake_t2[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(real_t1, cmap="gray"); axes[0].set_title("T1")
            axes[1].imshow(real_t2, cmap="gray"); axes[1].set_title("Real T2")
            axes[2].imshow(gen_t2, cmap="gray"); axes[2].set_title("Generated T2")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"samples_brats/epoch{epoch}_example{i}.png")
            plt.close()
            break  # just one batch

def main():
    num_epochs = 50  # you can start with 10 to test the pipeline
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch)
        save_samples(epoch)
        torch.save(G.state_dict(), f"checkpoints_brats/G_epoch{epoch}.pth")
        torch.save(D.state_dict(), f"checkpoints_brats/D_epoch{epoch}.pth")

if __name__ == "__main__":
    main()
