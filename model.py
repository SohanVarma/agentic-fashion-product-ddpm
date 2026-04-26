import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.cond = nn.Linear(cond_dim, out_ch)

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.cond(cond)[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h


class ConditionalUNet(nn.Module):
    """Small conditional U-Net for 64x64 RGB DDPM.

    Label `num_classes` is reserved for NULL/unconditional conditioning.
    """
    def __init__(self, num_classes: int, img_channels: int = 3, base_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.null_class = num_classes
        self.time_dim = time_dim
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim)
        )

        c = base_channels
        self.in_conv = nn.Conv2d(img_channels, c, 3, padding=1)
        self.down1 = ConvBlock(c, c, time_dim)
        self.down2 = ConvBlock(c, c * 2, time_dim)
        self.down3 = ConvBlock(c * 2, c * 4, time_dim)
        self.pool = nn.MaxPool2d(2)

        self.mid = ConvBlock(c * 4, c * 4, time_dim)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c, time_dim)
        self.out = nn.Conv2d(c, img_channels, 1)

    def condition(self, t, y):
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        y_emb = self.label_emb(y)
        return t_emb + y_emb

    def forward(self, x, t, y):
        cond = self.condition(t, y)
        x0 = self.in_conv(x)
        d1 = self.down1(x0, cond)       # 64x64
        p1 = self.pool(d1)              # 32x32
        d2 = self.down2(p1, cond)       # 32x32
        p2 = self.pool(d2)              # 16x16
        d3 = self.down3(p2, cond)       # 16x16
        m = self.mid(d3, cond)
        u2 = self.up2(m)                # 32x32
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2, cond)
        u1 = self.up1(u2)               # 64x64
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1, cond)
        return self.out(u1)
