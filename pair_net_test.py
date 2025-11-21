import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
import cv2

def make_sample(dir = ''):
    data_i = torch.load(dir)
    return {'ptsL':data_i['x1'], 'ptsR':data_i['x2'], 'K1':data_i['K1'], 'R1':data_i['RT1'][:,:3], 't1':data_i['RT1'][:,3], 'K2':data_i['K2'], 'R2':data_i['RT2'][:,:3], 't2':data_i['RT2'][:,3], 'M':data_i['M_ij']}

def triangulate_point(pL, pR, K1, R1, t1, K2, R2, t2):
    """
    pL, pR: (2,) 像素坐标
    K1, K2: (3,3) 相机内参
    R1, R2: (3,3) 外参旋转矩阵
    t1, t2: (3,) 外参平移
    return X: (3,) 世界坐标
    """
    # 投影矩阵 P = K [R|t]
    P1 = np.hstack([R1, t1.reshape(3,1)])
    P1 = K1 @ P1
    P2 = np.hstack([R2, t2.reshape(3,1)])
    P2 = K2 @ P2

    uL, vL = pL
    uR, vR = pR

    A = np.zeros((4,4))
    A[0] = uL*P1[2] - P1[0]
    A[1] = vL*P1[2] - P1[1]
    A[2] = uR*P2[2] - P2[0]
    A[3] = vR*P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]   # 齐次归一化
    return X

class PairsDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_fn(batch):
    B = len(batch)
    N1max = max([s['ptsL'].shape[0] for s in batch])
    N2max = max([s['ptsR'].shape[0] for s in batch])
    ptsL = torch.zeros((B, N1max, 2))
    ptsR = torch.zeros((B, N2max, 2))
    maskL = torch.zeros((B, N1max), dtype = torch.bool)
    maskR = torch.zeros((B, N2max), dtype = torch.bool)
    M = torch.zeros((B, N1max, N2max))
    K1 = []
    R1 = []
    t1 = []
    K2 = []
    R2 = []
    t2 = []
    for b, s in enumerate(batch):
        n1 = s['ptsL'].shape[0]
        n2 = s['ptsR'].shape[0]
        if n1 > 0:
            ptsL[b,:n1] = s['ptsL']
            maskL[b,:n1] = True
        if n2 > 0:
            ptsR[b,:n2] = s['ptsR'] 
            maskR[b,:n2] = True
        if 'M' in s:
            M[b,:n1,:n2] = s['M']
        K1.append(s['K1'])
        R1.append(s['R1'])
        t1.append(s['t1'])
        K2.append(s['K2'])
        R2.append(s['R2'])
        t2.append(s['t2'])
    return {'ptsL':ptsL.float(), 'ptsR':ptsR.float(), 'maskL':maskL, 'maskR':maskR, 'M':M.float(),
            'K1':torch.stack(K1), 'R1':torch.stack(R1), 't1':torch.stack(t1),
            'K2':torch.stack(K2), 'R2':torch.stack(R2), 't2':torch.stack(t2)}

class DoubleConv(nn.Module):
    """(conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNetMatcher(nn.Module):
    def __init__(self, in_ch=6, base_ch=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2, ceil_mode=True) 
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)

        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        # 对齐 d2 与 e2
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        # 对齐 d1 与 e1
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out
    
def compute_F_batch(K1, R1, t1, K2, R2, t2):
    Fm = torch.zeros((3,3), device=K1.device, dtype=K1.dtype)
    Rrel = R2 @ R1.T
    trel = (t2 - Rrel @ t1).reshape(3)
    tx = torch.tensor([[0.0, -trel[2], trel[1]],[trel[2], 0.0, -trel[0]],[-trel[1], trel[0], 0.0]], device=K1.device, dtype=K1.dtype)
    Fm = torch.inverse(K2).T @ tx @ Rrel @ torch.inverse(K1)
    return Fm

if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    samples = [make_sample(f'dataset/processed_data/data_{i+1}.pt') for i in range(100,130)]
    dataset = PairsDataset(samples)
    dataloader = DataLoader(dataset, batch_size = 1, collate_fn = collate_fn, shuffle = True)

    model = UNetMatcher(in_ch=5, base_ch=64).to(DEVICE)
    model.load_state_dict(torch.load('pair_net.pth'))

    for data in dataloader:
        ptsL = data['ptsL'][0][:,[1,0]].to(DEVICE) # [N1,2]
        ptsR = data['ptsR'][0][:,[1,0]].to(DEVICE) # [N2,2]
        M = data['M'][0].to(DEVICE)       # [N1,N2]

        K1 = data['K1'][0].to(DEVICE)
        R1 = data['R1'][0].to(DEVICE)
        t1 = data['t1'][0].to(DEVICE)
        K2 = data['K2'][0].to(DEVICE)
        R2 = data['R2'][0].to(DEVICE)
        t2 = data['t2'][0].to(DEVICE)
        N1 = ptsL.shape[0]
        N2 = ptsR.shape[0]
        
        epi = torch.zeros((N1,N2), device=DEVICE)
        Fm = compute_F_batch(K1,R1,t1,K2,R2,t2)
        x1 = ptsL
        x2 = ptsR
        x1h = torch.cat([x1, torch.ones((N1,1), device=DEVICE)], dim = 1)
        x2h = torch.cat([x2, torch.ones((N2,1), device=DEVICE)], dim = 1)
        Fx1 = (Fm @ x1h.T).T
        Mpair = torch.abs(x2h @ Fx1.T)
        epi = Mpair.T
        epi = epi / (epi.mean()+1e-9)

        input = torch.zeros((1,5,N1,N2)).to(DEVICE)
        input[0,0] = epi
        input[0,1] = torch.sin(ptsL[:,0].unsqueeze(1).repeat(1,N2))
        input[0,2] = torch.sin(ptsL[:,1].unsqueeze(1).repeat(1,N2))
        input[0,3] = torch.sin(ptsR[:,0].unsqueeze(0).repeat(N1,1))
        input[0,4] = torch.sin(ptsR[:,1].unsqueeze(0).repeat(N1,1))

        hat_M = model(input)
        P = torch.sigmoid(hat_M[0,0])

        pts = []
        for i in range(N1):
            # print(ptsL[0,i], ptsR[0,torch.argmax(P[0,i])], M[0,i,torch.argmax(P[0,i])])
            idxL = i
            idxR = torch.argwhere(P[i]>0.2)
            # idxR = torch.argwhere(M[0,i]>0.5)
            # print(int(M[0,i].sum().item()), P[0,i, idxR].sum().item() / P[0,i].sum().item())
            # idxR = torch.argmax(P[0,i], dim=0).item()
            for idx in idxR:
                idx = idx.item()
                ptL = ptsL[idxL]
                ptR = ptsR[idx]
                pt3d = triangulate_point(ptL.cpu().numpy(), ptR.cpu().numpy(), K1.cpu().numpy(), R1.cpu().numpy(), t1.cpu().numpy(), K2.cpu().numpy(), R2.cpu().numpy(), t2.cpu().numpy())
                pts.append(pt3d)

        pts = np.array(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud("test1.ply", pcd)
        quit()