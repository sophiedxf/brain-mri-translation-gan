import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        self.down1 = self._block(in_channels, features)
        self.down2 = self._block(features, features * 2)
        self.down3 = self._block(features * 2, features * 4)
        self.down4 = self._block(features * 4, features * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._block(features * 8, features * 16)

        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = self._block(features * 8 * 2, features * 8)
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(features * 4 * 2, features * 4)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(features * 2 * 2, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = self._block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        p4 = self.pool(d4)

        bottleneck = self.bottleneck(p4)

        up4 = self.up4(bottleneck)
        cat4 = torch.cat([up4, d4], dim=1)
        dec4 = self.dec4(cat4)

        up3 = self.up3(dec4)
        cat3 = torch.cat([up3, d3], dim=1)
        dec3 = self.dec3(cat3)

        up2 = self.up2(dec3)
        cat2 = torch.cat([up2, d2], dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        cat1 = torch.cat([up1, d1], dim=1)
        dec1 = self.dec1(cat1)

        return self.final_conv(dec1)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = [features, features * 2, features * 4, features * 8]
        for in_f, out_f in zip(channels[:-1], channels[1:]):
            layers += [
                nn.Conv2d(in_f, out_f, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.Conv2d(channels[-1], 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
