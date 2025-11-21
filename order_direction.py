import os
import glob
import numpy as np
from typing import List, Tuple
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def remove_duplicates_keep_last(arr):
    seen = set()
    output = []

    for x, y in arr[::-1]:
        key = (int(x), int(y))
        if key not in seen:
            seen.add(key)
            output.append([x, y])

    output.reverse()
    return np.array(output, dtype=arr.dtype)

class CurveOrderDataset(Dataset):
    def __init__(self, pts, H, W):
        self.pts = pts
        self.num = len(pts)
        self.mask = np.zeros((self.num, H, W))
        self.order = np.zeros((self.num, H, W))
        for i in range(self.num):
            curve = pts[i]
            xs = curve[:, 0].astype(np.int64)
            ys = curve[:, 1].astype(np.int64)
            self.mask[i, ys, xs] = 1
            self.order[i, ys, xs] = np.arange(len(curve), dtype=np.int32) / len(curve)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        return torch.from_numpy(self.mask[idx]).unsqueeze(0).to(torch.float32), torch.from_numpy(self.order[idx]).unsqueeze(0).to(torch.float32)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()

        self.enc1 = UNetBlock(in_ch, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


class DirectionOrderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_ch=1, out_ch=1)

    def forward(self, x):
        out = self.unet(x)
        order = torch.sigmoid(out)    # [B,1,H,W]
        return order

def compute_gt_tangent(gt_order_norm):
    B, H, W = gt_order_norm.shape

    dy = F.pad(gt_order_norm[:, 1:, :] - gt_order_norm[:, :-1, :], (0,0,0,1))
    dx = F.pad(gt_order_norm[:, :, 1:] - gt_order_norm[:, :, :-1], (0,1,0,0))

    vec = torch.stack([dx, dy], dim=1)  # [B,2,H,W]
    vec = vec / (vec.norm(dim=1, keepdim=True) + 1e-6)
    return vec


def order_loss(pred_order, gt_order_norm, mask):
    return F.smooth_l1_loss(pred_order[mask > 0.5], gt_order_norm[mask.squeeze(1) > 0.5])

def direction_loss(pred_vec, gt_vec, mask):
    dot = (pred_vec * gt_vec).sum(1, keepdim=True)
    return (1 - dot)[mask > 0.5].mean()

def pairwise_order_loss(order_pred, order_gt, mask):
    B, _, H, W = order_pred.shape
    loss = 0
    for b in range(B):
        ys, xs = torch.where(mask[b,0] > 0)
        if len(xs) < 2:
            continue
        pred_vals = order_pred[b,0,ys,xs]
        gt_vals   = order_gt[b,0,ys,xs]
        idx_i, idx_j = torch.triu_indices(len(xs), len(xs), offset=1)
        # 仅对 GT 顺序 i<j 计算约束
        diff_gt   = gt_vals[idx_j] - gt_vals[idx_i]
        diff_pred = pred_vals[idx_j] - pred_vals[idx_i]
        # 只对 GT 顺序有效的 pair
        mask_pair = (diff_gt > 0)
        if mask_pair.sum() == 0:
            continue
        loss += F.relu(-diff_pred[mask_pair]).mean()
    return loss / B

if __name__ == "__main__":
    epochs = 100
    num_train = 120
    num_test = 10
    TRAIN = True
    device = 'cuda:7'

    H, W = 2048, 512

    if TRAIN:

        pts = []
        for i in range(num_train):
            data_i = torch.load(f'./dataset/processed_data/data_{i+1}.pt')
            img = np.zeros((H, W))

            uv1 = np.rint(data_i['uv1']).numpy().astype(np.int32)
            uv1 = remove_duplicates_keep_last(uv1)
            uv2 = np.rint(data_i['uv2']).numpy().astype(np.int32)
            uv2 = remove_duplicates_keep_last(uv2)

            pts.append(uv1)
            pts.append(uv2)

        dataset = CurveOrderDataset(pts, H, W)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = DirectionOrderNet().to(device)
        # model.load_state_dict(torch.load("ordernet_l2.pth"))
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            L_epoch = 0
            for batch in loader:
                mask, order_gt = batch
                mask = mask.to(device)                  # [B,1,H,W]
                order_gt = order_gt.to(device)          # [B,H,W]
                
                order_pred = model(mask)                # [B,1,H,W]

                loss_mse = F.mse_loss(order_pred * mask, order_gt * mask, reduction='sum') / mask.sum()
                # loss_pairwise = pairwise_order_loss(order_pred, order_gt, mask)

                dy = torch.abs(order_pred[:, 0, :-1, :-1] - order_pred[:, 0, :-1, 1:])
                dx = torch.abs(order_pred[:, 0, :-1, :-1] - order_pred[:, 0, 1:, :-1])
                loss_smooth = (dx + dy)[mask[:,0, :-1,:-1]==1].mean()

                loss = loss_mse + loss_smooth
                print(loss_mse.item(), loss_smooth.item())

                optim.zero_grad()
                loss.backward()
                optim.step()

                L_epoch += loss.item()
            print(epoch, L_epoch)

        torch.save(model.state_dict(), "ordernet.pth")