import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np

class SimpleOrderNet(nn.Module):
    def __init__(self, embed_dim=16, max_order=2000):
        super().__init__()
        self.max_order = max_order

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1)
        )

    def forward_single(self, mask, K, R, t):
        out = self.conv(mask)
        out = torch.sigmoid(out) * self.max_order * mask
        return out

    def forward(self, mask1, mask2, K1, R1, t1, K2, R2, t2):
        order1 = self.forward_single(mask1, K1, R1, t1)
        order2 = self.forward_single(mask2, K2, R2, t2)
        return order1, order2

def order_loss(pred, gt, mask):
    diff = torch.norm(pred - gt, p=2)
    return diff

def total_loss(order1_pred, order2_pred, order1_gt, order2_gt, mask1, mask2):
    L1 = order_loss(order1_pred, order1_gt, mask1)
    L2 = order_loss(order2_pred, order2_gt, mask2)
    return (L1 + L2) / 2

if __name__ == "__main__":
    # B, H, W = 2, 64, 64
    # mask_L = torch.rand(B,1,H,W).cuda()
    # mask_R = torch.rand(B,1,H,W).cuda()

    # K = torch.tensor([[500.,0.,32.],[0.,500.,32.],[0.,0.,1.]]).unsqueeze(0).repeat(B,1,1).cuda()
    # R = torch.eye(3).unsqueeze(0).repeat(B,1,1).cuda()
    # t = torch.zeros(B,3,1).cuda()

    # gt_points = torch.rand(B, H*W, 3).cuda()

    # model = MaskStereo3D().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # for step in range(3):
    #     pred = model(mask_L, mask_R, K, R, t, K, R, t)
    #     loss = chamfer_loss(pred, gt_points)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Step {step}: loss={loss.item():.6f}")

    RATE_TRAIN = 0.5
    RATE_VAL = 0.1
    NUM_EPOCHS = 10

    files = os.listdir('dataset/processed_data/')
    n_files = len(files)
    n_train = int(RATE_TRAIN * n_files)
    n_val = int(RATE_VAL * n_files)
    n_test = n_files - n_train - n_val
    data_train = []
    data_val = []
    data_test = []

    for i in range(n_files):
        data_i = torch.load(f'dataset/processed_data/data_{i+1}.pt')
        if i < n_train:
            data_train.append(data_i)
        elif i < n_train + n_val:
            data_val.append(data_i)
        else:
            data_test.append(data_i)
    
    model = SimpleOrderNet().cuda()
    model.load_state_dict(torch.load('model.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.eval()

    for data_i in data_test:
        mask1 = data_i['mask1'].unsqueeze(0).unsqueeze(0).float().cuda()
        mask2 = data_i['mask2'].unsqueeze(0).unsqueeze(0).float().cuda()
        K1 = data_i['K1'].unsqueeze(0).float().cuda()
        RT1 = data_i['RT1'].unsqueeze(0).float().cuda()
        K2 = data_i['K2'].unsqueeze(0).float().cuda()
        RT2 = data_i['RT2'].unsqueeze(0).float().cuda()
        R1, t1 = RT1[:,:,:3], RT1[:,:,3:]
        R2, t2 = RT2[:,:,:3], RT2[:,:,3:]

        order1_gt = data_i['order1'].unsqueeze(0).float().cuda()
        order2_gt = data_i['order2'].unsqueeze(0).float().cuda()

        order1_pred, order2_pred = model(mask1, mask2, K1,R1,t1,K2,R2,t2)
        uv1 = data_i['uv1'].numpy()
        uv2 = data_i['uv2'].numpy()
        for i in range(uv1.shape[0]):
            u, v = uv1[i].astype(int)
            print(order1_pred[0,0,v,u].item(), order1_gt[0,v,u].item())
        quit()