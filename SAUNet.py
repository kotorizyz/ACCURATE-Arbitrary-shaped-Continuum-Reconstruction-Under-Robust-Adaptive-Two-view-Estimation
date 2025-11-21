import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import open3d as o3d


def make_sample(dir = ''):
    data_i = torch.load(dir)
    return {'ptsL':data_i['x1'], 'ptsR':data_i['x2'], 'K1':data_i['K1'], 'R1':data_i['RT1'][:,:3], 't1':data_i['RT1'][:,3], 'K2':data_i['K2'], 'R2':data_i['RT2'][:,:3], 't2':data_i['RT2'][:,3], 'M':data_i['M_ij']}

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
    
def compute_F_batch(K1, R1, t1, K2, R2, t2):
    Fm = torch.zeros((3,3), device=K1.device, dtype=K1.dtype)
    Rrel = R2 @ R1.T
    trel = (t2 - Rrel @ t1).reshape(3)
    tx = torch.tensor([[0.0, -trel[2], trel[1]],[trel[2], 0.0, -trel[0]],[-trel[1], trel[0], 0.0]], device=K1.device, dtype=K1.dtype)
    Fm = torch.inverse(K2).T @ tx @ Rrel @ torch.inverse(K1)
    return Fm

def normalize_xy(uv, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (uv[:,0] - cx) / fx
    y = (uv[:,1] - cy) / fy
    return torch.stack([x, y], dim=1)


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

def P_to_PLY(P, K1, R1, t1, K2, R2, t2, ptsL, ptsR, threshold = 0.5, filename = 'test.ply'):
    N1, N2 = P.shape[0], P.shape[1]
    pts = []
    for i in range(N1):
        idxL = i
        idxR = torch.argwhere(P[i] > threshold)
        for idx in idxR:
            idx = idx.item()
            ptL = ptsL[idxL]
            ptR = ptsR[idx]
            pt3d = triangulate_point(ptL.cpu().numpy(), ptR.cpu().numpy(), K1.cpu().numpy(), R1.cpu().numpy(), t1.cpu().numpy(), K2.cpu().numpy(), R2.cpu().numpy(), t2.cpu().numpy())
            pts.append(pt3d)
    print(len(pts))
    pts = np.array(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(filename, pcd)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        ws = self.window_size

        x = x.reshape(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)  # [num_windows*B, ws*ws, C]

        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.num_heads)**0.5)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
        x = self.proj(x)
        x = x.reshape(B, H // ws, W // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, H, W)

        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class SA_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(5, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)
        self.attn_bottleneck = WindowAttention(256, window_size=8)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.attn_dec2 = WindowAttention(128, window_size=8)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.attn_dec1 = WindowAttention(64, window_size=8)

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)         # [B,32,256,256]
        e2 = self.enc2(self.pool(e1))  # [B,64,128,128]
        e3 = self.enc3(self.pool(e2))  # [B,128,64,64]

        b = self.bottleneck(self.pool(e3))  # [B,256,32,32]
        b = self.attn_bottleneck(b)

        d2 = self.up2(b)
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.dec2(d2)
        d2 = self.attn_dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e2), dim=1)
        d1 = self.dec1(d1)
        d1 = self.attn_dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat((d0, e1), dim=1)
        d0 = self.dec0(d0)

        return self.out(d0)  # [B,1,256,256]

if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    TRAIN = False
    img_size = 256

    samples = [make_sample(f'dataset/processed_data/data_{i+1}.pt') for i in range(110, 120)]
    dataset = PairsDataset(samples)
    dataloader = DataLoader(dataset, batch_size = 1, collate_fn = collate_fn, shuffle = True)

    model = SA_UNet().to(DEVICE)

    if TRAIN:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.BCEWithLogitsLoss()
        model.train()

        for epoch in range(1000):
            loss_epoch = 0
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

                ptsL_norm = normalize_xy(ptsL, K1)
                ptsR_norm = normalize_xy(ptsR, K2)

                with torch.no_grad():
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
                input[0,1] = ptsL_norm[:,0].unsqueeze(1).repeat(1,N2)
                input[0,2] = ptsL_norm[:,1].unsqueeze(1).repeat(1,N2)
                input[0,3] = ptsR_norm[:,0].unsqueeze(0).repeat(N1,1)
                input[0,4] = ptsR_norm[:,1].unsqueeze(0).repeat(N1,1)

                mask = input[0,0] < 0.01
                input_simple = input[0,:,mask]
                M_simple = M[mask]

                img_M = torch.zeros((img_size**2)).to(DEVICE)
                img_M[:M_simple.shape[0]] = M_simple
                img_M = img_M.reshape(1,1,img_size,img_size)

                img_input = torch.ones((1,5, img_size**2)).to(DEVICE)
                img_input[0,:,:input_simple.shape[1]] = input_simple
                img_input = img_input.reshape(1,5,img_size,img_size)
                pred = model(img_input)
                loss = criterion(pred, img_M)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            print(f"Epoch {epoch+1}, Loss: {loss_epoch/len(dataloader)}")
        
        torch.save(model.state_dict(), 'SAUNet.pth')
    
    else:
        model.load_state_dict(torch.load('SAUNet.pth', map_location=DEVICE))
        model.eval()
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

                ptsL_norm = normalize_xy(ptsL, K1)
                ptsR_norm = normalize_xy(ptsR, K2)

                with torch.no_grad():
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
                input[0,1] = ptsL_norm[:,0].unsqueeze(1).repeat(1,N2)
                input[0,2] = ptsL_norm[:,1].unsqueeze(1).repeat(1,N2)
                input[0,3] = ptsR_norm[:,0].unsqueeze(0).repeat(N1,1)
                input[0,4] = ptsR_norm[:,1].unsqueeze(0).repeat(N1,1)

                mask = input[0,0] < 0.01
                input_simple = input[0,:,mask]
                M_simple = M[mask]

                img_M = torch.zeros((img_size**2)).to(DEVICE)
                img_M[:M_simple.shape[0]] = M_simple
                img_M = img_M.reshape(1,1,img_size,img_size)

                img_input = torch.ones((1,5, img_size**2)).to(DEVICE)
                img_input[0,:,:input_simple.shape[1]] = input_simple
                img_input = img_input.reshape(1,5,img_size,img_size)

                # img_show = img_input[0,0].cpu().detach().numpy()
                # cv2.imwrite('epi.png', (img_show / img_show.max())*255)
                # quit()

                pred = model(img_input)
                pred = pred.reshape(img_size**2)[:M_simple.shape[0]]
                prob = torch.sigmoid(pred)
                pred_M = torch.zeros((N1, N2), device=DEVICE)
                pred_M[mask] = prob
                print(M.sum().item())
                P_to_PLY(M.cpu(), K1.cpu(), R1.cpu(), t1.cpu(), K2.cpu(), R2.cpu(), t2.cpu(), ptsL.cpu(), ptsR.cpu(), threshold=0.95)
                P_to_PLY(pred_M.cpu(), K1.cpu(), R1.cpu(), t1.cpu(), K2.cpu(), R2.cpu(), t2.cpu(), ptsL.cpu(), ptsR.cpu(), threshold=0.95, filename='test1.ply')
                quit()