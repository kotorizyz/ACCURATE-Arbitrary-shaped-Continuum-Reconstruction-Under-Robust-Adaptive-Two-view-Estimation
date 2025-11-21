import os
import math
import time
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def build_projection_matrix(K, R, t):
    Rt = np.concatenate([R, t.reshape(3,1)], axis=1)    # [3,4]
    P = K @ Rt                                          # [3,4]
    return P

def project_points(P, points3):
    N = points3.shape[0]
    hom = np.concatenate([points3, np.ones((N,1))], axis=1).T   # [4,N]
    proj = (P @ hom)                                            # [3,N]
    z = proj[2,:].copy()
    xy = (proj[:2,:] / (proj[2:3,:] + 1e-12)).T                 # [N,2]
    return xy, z                                                # [N,2], [N,]

def render_depth_and_mask_from_pointcloud(points3, K, R, t, H, W):
    P = build_projection_matrix(K,R,t)
    xy, z = project_points(P, points3)
    u = np.round(xy[:,0]).astype(int)
    v = np.round(xy[:,1]).astype(int)
    depth = np.zeros((H,W), dtype=np.float32)
    mask = np.zeros((H,W), dtype=np.uint8)
    for i in range(len(u)):
        ui, vi = u[i], v[i]
        if ui < 0 or ui >= W or vi < 0 or vi >= H: continue
        zi = z[i]
        if mask[vi, ui] == 0 or zi < depth[vi, ui] or depth[vi, ui] == 0:
            depth[vi, ui] = zi
            mask[vi, ui] = 1
    
    # depth_min = depth[mask>0].min()
    # depth_max = depth[mask>0].max()
    # depth_show = (depth - depth_min) / (depth_max - depth_min)
    # cv2.imwrite('depth.png', depth_show * 255)
    # cv2.imwrite('mask.png', mask * 255)

    return depth, mask                                          # [H,W], [H,W]

def compute_depth_range(cam_pairs, pc_list):
    depths_all = []
    num_data = len(pc_list)
    for i in range(num_data):
        pc = pc_list[i]
        K1= cam_pairs[i]["K1"]
        R1 = cam_pairs[i]["R1"]
        t1 = cam_pairs[i]["t1"].reshape(3, 1)
        pc = pc.T
        Xc = R1 @ pc + t1
        Z = Xc[2, :]
        depths_all.append(Z)

        pc = pc_list[i]
        K2= cam_pairs[i]["K2"]
        R2 = cam_pairs[i]["R2"]
        t2 = cam_pairs[i]["t2"].reshape(3, 1)
        pc = pc.T
        Xc = R2 @ pc + t2
        Z = Xc[2, :]
        depths_all.append(Z)

    depths_all = np.concatenate(depths_all)
    depths_all = depths_all[depths_all > 0]

    min_depth = np.min(depths_all)
    max_depth = np.max(depths_all)

    return float(min_depth), float(max_depth)

class PointCloudToDepthDataset(Dataset):
    def __init__(self, pc_list, cam_pairs, H=480, W=640):
        assert len(pc_list) == len(cam_pairs)
        self.pc_list = pc_list
        self.cam_pairs = cam_pairs
        self.H = H; self.W = W

    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx):
        pts = self.pc_list[idx]
        cams = self.cam_pairs[idx]
        K1, R1, t1 = cams['K1'], cams['R1'], cams['t1']
        K2, R2, t2 = cams['K2'], cams['R2'], cams['t2']
        DL, maskL = render_depth_and_mask_from_pointcloud(pts, K1, R1, t1, self.H, self.W)
        DR, maskR = render_depth_and_mask_from_pointcloud(pts, K2, R2, t2, self.H, self.W)
        sample = {
            'depthL': torch.from_numpy(DL).unsqueeze(0).float(),   # [1,H,W]
            'depthR': torch.from_numpy(DR).unsqueeze(0).float(),
            'maskL': torch.from_numpy(maskL).unsqueeze(0).float(), # [1,H,W]
            'maskR': torch.from_numpy(maskR).unsqueeze(0).float(),
            'K1': torch.from_numpy(K1).float(),
            'R1': torch.from_numpy(R1).float(),
            't1': torch.from_numpy(t1).float(),
            'K2': torch.from_numpy(K2).float(),
            'R2': torch.from_numpy(R2).float(),
            't2': torch.from_numpy(t2).float()
        }
        return sample

class Simple3DReg(nn.Module):
    def __init__(self, in_channels=1, base_channels=24):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(base_channels, 1, 3, padding=1)
        )

    def forward(self, cost):
        x = cost.unsqueeze(1)                                       # [B,1,D,H,W]
        x = self.conv3d(x)
        x = x.squeeze(1)
        return x

def soft_argmin(cost_volume, values=None, beta=1.0):
    prob = F.softmax(-beta * cost_volume, dim=1)
    v = values.view(1,-1,1,1).to(cost_volume.device)
    return (prob * v).sum(dim=1, keepdim=True)

class StereoMaskDepthNet(nn.Module):
    def __init__(self, method='epipolar', max_disp=64, n_samples=64, base_channels=24):
        super().__init__()
        self.method = method
        self.max_disp = max_disp
        self.n_samples = n_samples
        self.reg = Simple3DReg(base_channels=base_channels)

    def build_rectified_cost_volume(self, maskL, maskR, max_disp):
        B, C, H, W = maskL.shape
        D = max_disp
        paddedR = F.pad(maskR, (D,0,0,0), mode='constant', value=0)
        cost_list = []
        for d in range(D):
            shifted = paddedR[:, :, :, D-d:W + D - d]
            cost_list.append((maskL - shifted).abs())
        cost = torch.cat(cost_list, dim=1)  # [B,D,H,W]
        return cost

    def build_epipolar_cost_volume_gpu(self, maskL, maskR, K1, R1, t1, K2, R2, t2, n_samples, min_depth, max_depth):
        B,C,H,W = maskL.shape
        device = maskL.device
        depths = torch.linspace(min_depth, max_depth, n_samples, device=device)
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        u = xs.float()
        v = ys.float()
        ones = torch.ones_like(u)
        pix = torch.stack([u, v, ones], dim=-1).view(-1,3).T  # [3,H*W]
        invK1 = torch.from_numpy(np.linalg.inv(K1)).float().to(device)
        K2_t = torch.from_numpy(K2).float().to(device)
        R_rel = torch.from_numpy((R2 @ R1.T)).float().to(device)
        t_rel = torch.from_numpy((t2 - R_rel.cpu().numpy() @ t1)).float().to(device)
        Npix = pix.shape[1]
        pix = pix.to(device)
        K1inv_pix = invK1 @ pix  # [3,H*W]
        cost_slices = []
        for z in depths:
            X1 = K1inv_pix * z
            X2 = R_rel @ X1 + t_rel.view(3,1)
            proj = K2_t @ X2
            proj = proj / (proj[2:3,:] + 1e-8)
            x2 = proj[0,:].view(H,W); y2 = proj[1,:].view(H,W)
            nx = (x2 / (W-1) - 0.5) * 2.0
            ny = (y2 / (H-1) - 0.5) * 2.0
            grid = torch.stack([nx, ny], dim=-1).unsqueeze(0)
            sampled = F.grid_sample(maskR, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            c = (maskL - sampled).abs()
            cost_slices.append(c)
        cost = torch.cat(cost_slices, dim=1)  # [1,n_samples,H,W]
        return cost, depths

    def forward(self, maskL, maskR, geometry):
        cost, depth_values = self.build_epipolar_cost_volume_gpu(maskL, maskR,
                                                                    geometry['K1_np'], geometry['R1_np'], geometry['t1_np'],
                                                                    geometry['K2_np'], geometry['R2_np'], geometry['t2_np'],
                                                                    self.n_samples, geometry['min_depth'], geometry['max_depth'])
        cost_ref = self.reg(cost)
        depth = soft_argmin(cost_ref, values=depth_values)
        return depth, cost_ref

def depth_to_point_cam_coords(depth, K):
    B,_,H,W = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u = xs.reshape(-1).float()
    v = ys.reshape(-1).float()
    ones = torch.ones_like(u)
    pix = torch.stack([u, v, ones], dim=0)  # [3,H*W]
    Kinv = torch.from_numpy(np.linalg.inv(K)).float().to(device)
    Kinv_pix = Kinv @ pix  # [3, H*W]
    d = depth.view(B, -1)  # [B, H*W]
    Xcam = Kinv_pix.unsqueeze(0) * d.unsqueeze(1)  # [B,3,H*W]
    return Xcam

def warp_depth_to_other_view(depth_src, K_src, R_src, t_src, K_tgt, R_tgt, t_tgt, H, W):
    B = depth_src.shape[0]
    device = depth_src.device
    Ksrc_inv = torch.from_numpy(np.linalg.inv(K_src)).float().to(device)
    Ktgt = torch.from_numpy(K_tgt).float().to(device)
    R_src_t = torch.from_numpy(R_src).float().to(device)
    R_tgt_t = torch.from_numpy(R_tgt).float().to(device)
    t_src_t = torch.from_numpy(t_src).float().to(device).view(3,1)
    t_tgt_t = torch.from_numpy(t_tgt).float().to(device).view(3,1)

    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u = xs.reshape(-1).float()
    v = ys.reshape(-1).float()
    ones = torch.ones_like(u)
    pix = torch.stack([u, v, ones], dim=0).to(device)                   # [3,H*W]
    Kinv_pix = Ksrc_inv @ pix                                           # [3,H*W]
    d = depth_src.view(B, -1)                                           # [B,H*W]
    X_cam = Kinv_pix.unsqueeze(0) * d.unsqueeze(1)                      # [B,3,H*W]
    R_src_inv = R_src_t.T.to(device)
    X_world = R_src_inv.unsqueeze(0) @ (X_cam - t_src_t.unsqueeze(0))
    
    X_tgt = R_tgt_t.unsqueeze(0) @ X_world + t_tgt_t.unsqueeze(0)
    proj = Ktgt.unsqueeze(0) @ X_tgt                                    # [B,3,H*W]
    proj = proj / (proj[:,2:3,:] + 1e-8)
    x = proj[:,0,:].view(B,H,W); y = proj[:,1,:].view(B,H,W)
    z_tgt = X_tgt[:,2,:].view(B,H,W)
    nx = (x / (W-1) - 0.5) * 2.0
    ny = (y / (H-1) - 0.5) * 2.0
    grid = torch.stack([nx, ny], dim=-1)                                # [B,H,W,2]
    inside = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1))
    mask_proj = inside.float().unsqueeze(1)
    depth_proj = z_tgt.unsqueeze(1)                                     # [B,1,H,W]
    return depth_proj, mask_proj, grid

def depth_l1_loss(pred_depth, gt_depth, mask):
    valid = (mask > 0.5) & (gt_depth > 1e-6)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)
    diff = (pred_depth - gt_depth).abs()
    return diff[valid].mean()

def smoothness_loss(depth, mask):
    dx = torch.abs(depth[:,:,:,1:] - depth[:,:,:,:-1])
    dy = torch.abs(depth[:,:,1:,:] - depth[:,:,:-1,:])
    m_x = mask[:,:,:,1:] * mask[:,:,:,:-1]
    m_y = mask[:,:,1:,:] * mask[:,:,:-1,:]
    loss_x = (dx * m_x).sum() / (m_x.sum() + 1e-8)
    loss_y = (dy * m_y).sum() / (m_y.sum() + 1e-8)
    return loss_x + loss_y

def geo_consistency_loss(depthL_pred, depthR_pred, maskL, K1, R1, t1, K2, R2, t2):
    B = depthL_pred.shape[0]
    device = depthL_pred.device
    H = depthL_pred.shape[2]
    W = depthL_pred.shape[3]
    depth_proj, mask_proj, grid = warp_depth_to_other_view(depthL_pred, K1, R1, t1, K2, R2, t2, H, W)
    sampled_R = F.grid_sample(depthR_pred, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    valid = (mask_proj > 0.5) & (sampled_R > 1e-8)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    diff = (sampled_R - depth_proj).abs()
    return (diff[valid]).mean()

def train(model, train_loader, val_loader, device, epochs=20, lr=1e-3, save_dir='./checkpoints', min_depth = 0, max_depth = 100):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = 1e9
    for epoch in range(epochs):
        model.train()
        running = 0.0
        iters = 0
        t0 = time.time()
        for batch in train_loader:
            maskL = batch['maskL'].to(device)  # [B,1,H,W]
            maskR = batch['maskR'].to(device)
            depthL = batch['depthL'].to(device)
            depthR = batch['depthR'].to(device)
            geo = {
                'K1_np': batch['K1'].numpy()[0], 'R1_np': batch['R1'].numpy()[0], 't1_np': batch['t1'].numpy()[0],
                'K2_np': batch['K2'].numpy()[0], 'R2_np': batch['R2'].numpy()[0], 't2_np': batch['t2'].numpy()[0],
                'min_depth': min_depth, 'max_depth': max_depth
            }
            pred_depthL, _ = model(maskL, maskR, geo)  # [B,1,H,W]

            l_depth = depth_l1_loss(pred_depthL, depthL, maskL)
            l_smooth = smoothness_loss(pred_depthL, maskL)
            l_geo = geo_consistency_loss(pred_depthL, depthR, maskL, batch['K1'].numpy()[0], batch['R1'].numpy()[0], batch['t1'].numpy()[0],
                                         batch['K2'].numpy()[0], batch['R2'].numpy()[0], batch['t2'].numpy()[0])
            loss = l_depth + 0.1 * l_geo + 0.01 * l_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            iters += 1
        t1 = time.time()
        print(f"Epoch {epoch} train loss {running/iters:.6f} time {t1-t0:.1f}s")

        model.eval()
        with torch.no_grad():
            v_loss = 0.0; v_it=0
            for batch in val_loader:
                maskL = batch['maskL'].to(device)
                maskR = batch['maskR'].to(device)
                depthL = batch['depthL'].to(device)
                depthR = batch['depthR'].to(device)
                geo = {
                    'K1_np': batch['K1'].numpy()[0], 'R1_np': batch['R1'].numpy()[0], 't1_np': batch['t1'].numpy()[0],
                    'K2_np': batch['K2'].numpy()[0], 'R2_np': batch['R2'].numpy()[0], 't2_np': batch['t2'].numpy()[0],
                    'min_depth': min_depth, 'max_depth': max_depth
                }
                pred_depthL, _ = model(maskL, maskR, geo)
                l_depth = depth_l1_loss(pred_depthL, depthL, maskL)
                l_smooth = smoothness_loss(pred_depthL, maskL)
                l_geo = geo_consistency_loss(pred_depthL, depthR, maskL, batch['K1'].numpy()[0], batch['R1'].numpy()[0], batch['t1'].numpy()[0],
                                             batch['K2'].numpy()[0], batch['R2'].numpy()[0], batch['t2'].numpy()[0])
                loss = l_depth + 0.1*l_geo + 0.01*l_smooth
                v_loss += loss.item(); v_it += 1
            v_loss = v_loss / max(1, v_it)
            print(f"Val loss {v_loss:.6f}")

            if v_loss < best_val:
                best_val = v_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
                print("Saved best checkpoint")
            
            depthL_min = depthL[0,0][maskL[0,0]>0.5].min()
            depthL_max = depthL[0,0][maskL[0,0]>0.5].max()
            depthL_show = ((depthL[0,0] - depthL_min) / (depthL_max - depthL_min + 1e-8))
            depthL_show[maskL[0,0]<0.5] = 0
            cv2.imwrite('gt.png', depthL_show.cpu().detach().numpy()*255)
            pred_depthL_min = pred_depthL[0,0][maskL[0,0]>0.5].min()
            pred_depthL_max = pred_depthL[0,0][maskL[0,0]>0.5].max()
            pred_depthL_show = ((pred_depthL[0,0] - pred_depthL_min) / (pred_depthL_max - pred_depthL_min + 1e-8))
            pred_depthL_show[maskL[0,0]<0.5] = 0
            cv2.imwrite('pred.png', pred_depthL_show.cpu().detach().numpy()*255)

if __name__ == "__main__":
    # def make_synthetic_wire(n_pts=200, noise=0):
    #     t = np.linspace(0, 2*np.pi, n_pts)
    #     x = 0.02 * np.cos(2*t)
    #     y = 0.02 * np.sin(3*t)
    #     z = 0.1 + 0.02 * np.sin(t)
    #     pts = np.stack([x,y,z], axis=1)
    #     pts += np.random.randn(*pts.shape)*noise
    #     return pts

    # H, W = 240, 320
    # fx = fy = 400.0
    # cx = W/2.0; cy = H/2.0
    # K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], dtype=np.float64)

    # cam_pairs = []
    # pc_list = []
    # for i in range(200):
    #     pts = make_synthetic_wire(300)
    #     # left camera: front
    #     R1 = np.eye(3); t1 = np.array([0.0,0.0,0.0])
    #     # right camera: slightly to right and rotated
    #     angle = np.deg2rad(30.0)  # wide baseline example
    #     Rr = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    #     tr = np.array([0.04, 0.0, 0.0])  # baseline 4cm
    #     cam_pairs.append({'K1':K,'R1':R1,'t1':t1,'K2':K,'R2':Rr,'t2':tr})
    #     pc_list.append(pts)

    
    data_dir = './dataset/processed_data/'
    num_train = 100
    num_eval = 20
    cam_pairs = []
    pc_list = []
    H, W = 2048, 512
    for i in range(num_train+num_eval):
        data_i = torch.load(data_dir + f'data_{i+1}.pt')
        pts_i = data_i['points'].numpy()
        K1 = data_i['K1'].numpy()
        R1 = data_i['RT1'].numpy()[:3,:3]
        t1 = data_i['RT1'].numpy()[:3,3]
        K2 = data_i['K2'].numpy()
        R2 = data_i['RT2'].numpy()[:3,:3]
        t2 = data_i['RT2'].numpy()[:3,3]

        cam_pairs.append({'K1':K1,'R1':R1,'t1':t1,'K2':K2,'R2':R2,'t2':t2})
        pc_list.append(pts_i)
    
    min_depth, max_depth = compute_depth_range(cam_pairs, pc_list)

    ds = PointCloudToDepthDataset(pc_list[:num_train], cam_pairs[:num_train], H=H, W=W)
    ds_val = PointCloudToDepthDataset(pc_list[num_train:], cam_pairs[num_train:], H=H, W=W)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    vloader = DataLoader(ds_val, batch_size=1, shuffle=False)

    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    model = StereoMaskDepthNet(method='epipolar', n_samples=64, base_channels=12).to(device)
    train(model, loader, vloader, device, epochs=100, lr=1e-3, save_dir='./', min_depth=int(min_depth).__float__(), max_depth=int(max_depth+1).__float__())