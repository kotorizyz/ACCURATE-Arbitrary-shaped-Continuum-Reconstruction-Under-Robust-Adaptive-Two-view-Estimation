# Retry: smaller demo to avoid environment issues.
# Re-run a compact version of the training script on tiny synthetic data (1 epoch, small sizes).
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
import numpy as np
import open3d as o3d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_F_batch(K1, R1, t1, K2, R2, t2):
    B = K1.shape[0]
    Fm = torch.zeros((B,3,3), device=K1.device, dtype=K1.dtype)
    for b in range(B):
        Rrel = R2[b] @ R1[b].T
        trel = (t2[b] - Rrel @ t1[b]).reshape(3)
        tx = torch.tensor([[0.0, -trel[2], trel[1]],[trel[2], 0.0, -trel[0]],[-trel[1], trel[0], 0.0]], device=K1.device, dtype=K1.dtype)
        Fm[b] = torch.inverse(K2[b]).T @ tx @ Rrel @ torch.inverse(K1[b])
    return Fm

class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, nl: nn.Module = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nl(),
            nn.Linear(hidden, out_dim),
            nl()
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.0):
        super().__init__()
        # MultiheadAttention
        self.attn_lr = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.attn_rl = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.ffn_l = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
        self.ffn_r = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
        self.norm1_l = nn.LayerNorm(d_model)
        self.norm2_l = nn.LayerNorm(d_model)
        self.norm1_r = nn.LayerNorm(d_model)
        self.norm2_r = nn.LayerNorm(d_model)

    def forward(self, fL, fR):
        # fL: (1, N1, D), fR: (1, N2, D)
        # Left attends to Right
        attn_out_l, _ = self.attn_lr(query=fL, key=fR, value=fR, need_weights=False)
        fL = self.norm1_l(fL + attn_out_l)
        fL = self.norm2_l(fL + self.ffn_l(fL))
        # Right attends to Left
        attn_out_r, _ = self.attn_rl(query=fR, key=fL, value=fL, need_weights=False)
        fR = self.norm1_r(fR + attn_out_r)
        fR = self.norm2_r(fR + self.ffn_r(fR))
        return fL, fR

class MatchNet(nn.Module):
    def __init__(self, feat_dim: int = 128, mlp_hidden: int = 64, n_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.embedL = SmallMLP(2, mlp_hidden, feat_dim)
        self.embedR = SmallMLP(2, mlp_hidden, feat_dim)
        self.pos_mlp = SmallMLP(1, mlp_hidden, feat_dim)  # for epi bias encoding (takes scalar epi)
        self.cross_blocks = nn.ModuleList([CrossAttentionBlock(feat_dim, nhead) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(feat_dim)
        self.temp = nn.Parameter(torch.tensor(1.0))  # learnable temperature
        self._epi_proj = nn.Linear(self.feat_dim, 1)

    def forward(self, ptsL: torch.Tensor, ptsR: torch.Tensor, epi: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # ptsL: (1, N1, 2), ptsR: (1, N2, 2), epi: (1, N1, N2) optional
        B, N1, _ = ptsL.shape
        _, N2, _ = ptsR.shape
        fL = self.embedL(ptsL)  # (1, N1, D)
        fR = self.embedR(ptsR)  # (1, N2, D)

        # cross-attention fusion
        for block in self.cross_blocks:
            fL, fR = block(fL, fR)

        fL = self.final_ln(fL)
        fR = self.final_ln(fR)

        # similarity logits: scaled dot-product
        # (1, N1, D) @ (1, D, N2) -> (B, N1, N2)
        S = torch.matmul(fL, fR.transpose(1, 2)) / (self.feat_dim ** 0.5)
        # temperature
        S = S / (torch.clamp(self.temp, min=1e-3))

        # incorporate epi bias if provided (we convert epi scalar to a bias via small MLP)

        # epi: (B, N1, N2) -- convert to a bias of same shape
        # we pass epi as a scalar through pos_mlp by flattening and reshaping
        # small trick: pos_mlp expects (B, L, 1) input; we process per-batch unfolded
        B, N1, N2 = epi.shape
        epi_flat = epi.reshape(B, -1, 1)  # (B, N1*N2, 1)
        epi_feat = self.pos_mlp(epi_flat)  # (B, N1*N2, D)
        # reduce epi_feat to a scalar bias per pair using a linear projection (learnable)
        # create projection on the fly
        # proj = getattr(self, "_epi_proj", None)
        # if proj is None:
        #     self._epi_proj = nn.Linear(self.feat_dim, 1).to(S.device)
        #     proj = self._epi_proj

        epi_bias = self._epi_proj(epi_feat).reshape(B, N1, N2)  # (B, N1, N2)
        S = S + epi_bias

        # Row and column softmax probabilities
        P_row = F.softmax(S, dim=-1)  # left -> distribution over right for each left
        P_col = F.softmax(S, dim=1)   # right <- distribution over left for each right (columns)

        out = {"logits": S, "P_row": P_row, "P_col": P_col}
        # optional Sinkhorn (if configured) -- returns doubly-stochastic approx
        return out

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

def make_sample(dir = ''):
    data_i = torch.load(dir)
    return {'ptsL':data_i['x1'], 'ptsR':data_i['x2'], 'K1':data_i['K1'], 'R1':data_i['RT1'][:,:3], 't1':data_i['RT1'][:,3], 'K2':data_i['K2'], 'R2':data_i['RT2'][:,:3], 't2':data_i['RT2'][:,3], 'M':data_i['M_ij']}


samples = [make_sample(f'dataset/processed_data/data_{i+1}.pt') for i in range(100, 135)]
ds = PairsDataset(samples)
dl = DataLoader(ds, batch_size = 1, collate_fn = collate_fn, shuffle = False)

model = MatchNet(feat_dim=128, mlp_hidden=64, n_layers=2, nhead=4).to(DEVICE)

# load
model_path = 'pair.pth'
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    for batch in dl:
        ptsL = batch['ptsL'][:,:,[1,0]].to(DEVICE)                  # (B, N1, 2)
        ptsR = batch['ptsR'][:,:,[1,0]].to(DEVICE)                  # (B, N2, 2)
        M = batch['M'].to(DEVICE)                                   # (B, N1, N2)
        K1 = batch['K1'].to(DEVICE)
        R1 = batch['R1'].to(DEVICE)
        t1 = batch['t1'].to(DEVICE)
        K2 = batch['K2'].to(DEVICE)
        R2 = batch['R2'].to(DEVICE)
        t2 = batch['t2'].to(DEVICE)

        # compute epi_dist per batch element (normalized)
        Fm = compute_F_batch(K1,R1,t1,K2,R2,t2)
        B, N1, _ = ptsL.shape
        _, N2, _ = ptsR.shape
        epi = torch.zeros((B,N1,N2), device=DEVICE)
        for b in range(B):
            x1 = ptsL[b]
            x2 = ptsR[b]
            x1h = torch.cat([x1, torch.ones((N1,1), device=DEVICE)], dim = 1)
            x2h = torch.cat([x2, torch.ones((N2,1), device=DEVICE)], dim = 1)
            Fx1 = (Fm[b] @ x1h.T).T
            Mpair = torch.abs(x2h @ Fx1.T)
            epi[b] = Mpair.T
        epi = epi / (epi.mean(dim = (1,2), keepdim = True)+1e-9)

        out = model(ptsL, ptsR, epi)
        logits = out["logits"]
        P = torch.sigmoid(logits)
        # for i in range(10):
        #     print(torch.sort(P[0,i], descending=True).values[:int(M[0,i].sum().item())+3].cpu().numpy())
        #     idx = torch.sort(P[0,i], descending=True).indices[:int(M[0,i].sum().item())+3].cpu().numpy()
        #     print(M[0,i,idx].cpu().numpy())
        #     print('-------')
        # quit()

        pts = []
        for i in range(N1):
            # print(ptsL[0,i], ptsR[0,torch.argmax(P[0,i])], M[0,i,torch.argmax(P[0,i])])
            idxL = i
            idxR = torch.argwhere(P[0,i]>0.2)
            idxR = torch.argwhere(M[0,i]>0.5)
            print(int(M[0,i].sum().item()), P[0,i, idxR].sum().item() / P[0,i].sum().item())
            # idxR = torch.argmax(P[0,i], dim=0).item()
            for idx in idxR:
                idx = idx.item()
                ptL = ptsL[0,idxL]
                ptR = ptsR[0,idx]
                pt3d = triangulate_point(ptL.cpu().numpy(), ptR.cpu().numpy(), K1[0].cpu().numpy(), R1[0].cpu().numpy(), t1[0].cpu().numpy(), K2[0].cpu().numpy(), R2[0].cpu().numpy(), t2[0].cpu().numpy())
                pts.append(pt3d)

        pts = np.array(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud("test.ply", pcd)
        quit()