import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# tmp unet for demo
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.in_channels = in_channels
        if inter_channels is None:
            inter_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H*W)
        proj_query = proj_query.permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(B, -1, H*W)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(B, -1, H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class SelfAttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()
        self.in_conv = ConvBlock(in_channels, base_ch)

        self.enc1 = ConvBlock(base_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(base_ch*8, base_ch*16)

        self.attn4 = SelfAttention2d(base_ch*8)

        self.up4 = UpBlock(base_ch*16, base_ch*8)
        self.up3 = UpBlock(base_ch*8, base_ch*4)
        self.up2 = UpBlock(base_ch*4, base_ch*2)
        self.up1 = UpBlock(base_ch*2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):

        x0 = self.in_conv(x)

        e1 = self.enc1(x0)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        e4 = self.attn4(e4)
        p4 = self.pool4(e4)

        center = self.center(p4)

        d4 = self.up4(center, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.out_conv(d1)
        return out

class Dataset2D(Dataset):
    def __init__(self, img, mask):
        self.img = img
        self.mask = mask

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        return self.img[idx], self.mask[idx]