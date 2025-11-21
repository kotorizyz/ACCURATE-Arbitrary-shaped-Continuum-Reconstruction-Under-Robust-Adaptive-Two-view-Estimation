import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import cv2

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
    return depth, mask                                          # [H,W], [H,W]

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

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Encoder2D(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = ConvBlock2d(in_ch, base)            # [base,H,W]
        self.enc2 = ConvBlock2d(base, base*2)           # [base*2,H/2,W/2]
        self.enc3 = ConvBlock2d(base*2, base*4)         # [base*4,H/4,W/4]
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Conv2d(base + base*2 + base*4, base, 1)

    def forward(self, x):
        f1 = self.enc1(x)                               # [B,base,H,W]
        f2 = self.enc2(self.pool(f1))                   # [B,base*2,H/2,W/2]
        f3 = self.enc3(self.pool(f2))                   # [B,base*4,H/4,W/4]
        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([f1, f2_up, f3_up], dim=1)
        out = self.proj(fused)                          # [B,base,H,W]
        return out

class RefineUNet2D(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.enc1 = ConvBlock2d(in_ch, base)
        self.enc2 = ConvBlock2d(base, base*2)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = ConvBlock2d(base*2, base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = F.interpolate(e2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        return out

def build_epipolar_grids(K1, R1, t1, K2, R2, t2, depth_values, H, W, device):
    invK1 = torch.from_numpy(np.linalg.inv(K1)).float().to(device)                                              # [3,3]
    K2_t = torch.from_numpy(K2).float().to(device)
    R_rel = torch.from_numpy((R2 @ R1.T)).float().to(device)
    t_rel = torch.from_numpy((t2 - (R_rel.cpu().numpy() @ t1))).float().to(device)                              # [3,]

    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u = xs.float()
    v = ys.float()
    ones = torch.ones_like(u)
    pix = torch.stack([u, v, ones], dim=-1).view(-1,3).T                                                        # [3,H*W]
    K1inv_pix = invK1 @ pix                                                                                     # [3,H*W]

    grids = []
    for z in depth_values:
        X1 = K1inv_pix * z                              # [3,H*W]
        X2 = R_rel @ X1 + t_rel.view(3,1)               # [3,H*W]
        proj = K2_t @ X2                                # [3,H*W]
        proj = proj / (proj[2:3,:] + 1e-8)
        x2 = proj[0,:].view(H,W)
        y2 = proj[1,:].view(H,W)
        nx = (x2 / (W-1) - 0.5) * 2.0
        ny = (y2 / (H-1) - 0.5) * 2.0
        grid = torch.stack([nx, ny], dim=-1)            # [H,W,2]
        grids.append(grid)
    grids = torch.stack(grids, dim=0)                   # [D,H,W,2]
    return grids

class EpipolarAttStereo(nn.Module):
    def __init__(self, in_ch=1, base_feat=32, n_depth=64):
        super().__init__()
        self.n_depth = n_depth
        self.encoder = Encoder2D(in_ch=in_ch, base=base_feat)
        self.refiner = RefineUNet2D(in_ch=base_feat + 2, base=max(16, base_feat//2)).to(device)

    def forward(self, left_mask, right_mask, geometry, depth_values):
        device = left_mask.device
        B,_,H,W = left_mask.shape
        D = depth_values.shape[0]

        left_feat = self.encoder(left_mask)                                 # [B,C,H,W]
        right_feat = self.encoder(right_mask)                               # [B,C,H,W]
        C = left_feat.shape[1]

        grids = build_epipolar_grids(geometry['K1'], geometry['R1'], geometry['t1'],
                                     geometry['K2'], geometry['R2'], geometry['t2'],
                                     depth_values, H, W, device=device)     # [D,H,W,2]

        right_rep = right_feat.unsqueeze(1).repeat(1, D, 1, 1, 1)           # [B,D,C,H,W]
        right_rep = right_rep.view(B*D, C, H, W)                            # [B*D,C,H,W]
        grids_exp = grids.unsqueeze(0).repeat(B,1,1,1,1)                    # [B,D,H,W,2]
        grids_exp = grids_exp.view(B*D, H, W, 2)
        sampled = F.grid_sample(right_rep, grids_exp, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.view(B, D, C, H, W)                               # [B,D,C,H,W]

        left_exp = left_feat.unsqueeze(1).expand(-1, D, -1, -1, -1)         # [B,D,C,H,W]
        S = (left_exp * sampled).sum(dim=2)                                 # [B,D,H,W]

        prob = F.softmax(S, dim=1)
        dv = depth_values.view(1, D, 1, 1).to(device)
        init_depth = (prob * dv).sum(dim=1, keepdim=True)                   # [B,1,H,W]

        concat = torch.cat([left_feat, left_mask, init_depth], dim=1)       # [B,C+2,H,W]
        refined = self.refiner(concat)                                      # [B,1,H,W]

        return refined, S, prob

def warp_depth_to_other_view(depth_src, K_src, R_src, t_src, K_tgt, R_tgt, t_tgt):
    B = depth_src.shape[0]
    device = depth_src.device
    _,_,H,W = depth_src.shape
    Ksrc_inv = torch.from_numpy(np.linalg.inv(K_src)).float().to(device)
    Ktgt = torch.from_numpy(K_tgt).float().to(device)
    R_src_t = torch.from_numpy(R_src).float().to(device)
    R_tgt_t = torch.from_numpy(R_tgt).float().to(device)
    t_src_t = torch.from_numpy(t_src).float().to(device).view(3,1)
    t_tgt_t = torch.from_numpy(t_tgt).float().to(device).view(3,1)

    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u = xs.reshape(-1).float(); v = ys.reshape(-1).float()
    ones = torch.ones_like(u)
    pix = torch.stack([u, v, ones], dim=0).to(device)
    Kinv_pix = Ksrc_inv @ pix
    d = depth_src.view(B, -1)
    X_cam = Kinv_pix.unsqueeze(0) * d.unsqueeze(1)
    R_src_inv = R_src_t.T.to(device)
    X_world = R_src_inv.unsqueeze(0) @ (X_cam - t_src_t.unsqueeze(0))
    X_tgt = R_tgt_t.unsqueeze(0) @ X_world + t_tgt_t.unsqueeze(0)
    proj = Ktgt.unsqueeze(0) @ X_tgt
    proj = proj / (proj[:,2:3,:] + 1e-8)
    x = proj[:,0,:].view(B,H,W); y = proj[:,1,:].view(B,H,W)
    z_tgt = X_tgt[:,2,:].view(B,H,W)
    nx = (x / (W-1) - 0.5) * 2.0
    ny = (y / (H-1) - 0.5) * 2.0
    grid = torch.stack([nx, ny], dim=-1)
    inside = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1))
    mask_proj = inside.float().unsqueeze(1)
    depth_proj = z_tgt.unsqueeze(1)
    return depth_proj, mask_proj, grid


def depth_l2_loss(pred_depth, gt_depth, mask):
    valid = (mask > 0.5) & (gt_depth > 1e-8)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)
    return ((pred_depth - gt_depth)**2)[valid].mean()

def smoothness_loss(depth, mask):
    dx = torch.abs(depth[:,:,:,1:] - depth[:,:,:,:-1])
    dy = torch.abs(depth[:,:,1:,:] - depth[:,:,:-1,:])
    m_x = mask[:,:,:,1:] * mask[:,:,:,:-1]
    m_y = mask[:,:,1:,:] * mask[:,:,:-1,:]
    loss_x = (dx * m_x).sum() / (m_x.sum() + 1e-8)
    loss_y = (dy * m_y).sum() / (m_y.sum() + 1e-8)
    return loss_x + loss_y

# def geo_consistency_loss(depthL_pred, depthR_pred, maskL, K1, R1, t1, K2, R2, t2):
#     depth_proj, mask_proj, grid = warp_depth_to_other_view(depthL_pred, K1, R1, t1, K2, R2, t2)
#     sampled_R = F.grid_sample(depthR_pred, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#     valid = (mask_proj > 0.5) & (sampled_R > 1e-8)
#     if valid.sum() == 0:
#         return torch.tensor(0.0, device=depthL_pred.device)
#     return (sampled_R - depth_proj).abs()[valid].mean()

if __name__ == "__main__":
    device = 'cuda:7'
    data_dir = './dataset/processed_data/'
    num_train = 100
    num_test = 10
    cam_pairs = []
    pc_list = []
    H, W = 2048, 512
    D = 32
    TRAIN = False

    for i in range(num_train+num_test):
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
    depth_values = torch.linspace(min_depth, max_depth, D).to(device)

    ds = PointCloudToDepthDataset(pc_list[:num_train], cam_pairs[:num_train], H=H, W=W)
    ds_test = PointCloudToDepthDataset(pc_list[num_train:], cam_pairs[num_train:], H=H, W=W)
    tloader = DataLoader(ds, batch_size=1, shuffle=True)
    tloader_test = DataLoader(ds_test, batch_size=1, shuffle=False)

    model = EpipolarAttStereo(in_ch=1, base_feat=16, n_depth=D).to(device)

    if TRAIN:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
        for epoch in range(100):
            for data in tloader:
                left_mask = data['maskL'].to(device)
                right_mask = data['maskR'].to(device)
                geometry_L = {
                    'K1': data['K1'].numpy()[0], 'R1': data['R1'].numpy()[0], 't1': data['t1'].numpy()[0],
                    'K2': data['K2'].numpy()[0], 'R2': data['R2'].numpy()[0], 't2': data['t2'].numpy()[0]
                }
                geometry_R = {
                    'K1': data['K2'].numpy()[0], 'R1': data['R2'].numpy()[0], 't1': data['t2'].numpy()[0],
                    'K2': data['K1'].numpy()[0], 'R2': data['R1'].numpy()[0], 't2': data['t1'].numpy()[0]
                }
                gt_depth_L = data['depthL'].to(device)
                gt_mask_L = (gt_depth_L > 0.05).float()
                gt_depth_R = data['depthR'].to(device)
                gt_mask_R = (gt_depth_R > 0.05).float()

                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    pred_depth_L, S, prob = model(left_mask, right_mask, geometry_L, depth_values)
                    # pred_depth_R, S, prob = model(right_mask, left_mask, geometry_R, depth_values)
                    pred_depth_R, _, __ = warp_depth_to_other_view(pred_depth_L, geometry_L['K1'], geometry_L['R1'], geometry_L['t1'], geometry_L['K2'], geometry_L['R2'], geometry_L['t2'])

                    l_depth_L = depth_l2_loss(pred_depth_L, gt_depth_L, gt_mask_L)
                    l_depth_R = depth_l2_loss(pred_depth_R, gt_depth_R, gt_mask_R)
                    l_smooth = smoothness_loss(pred_depth_L, left_mask) + smoothness_loss(pred_depth_R, right_mask)
                    loss = l_depth_L + l_depth_R + 0.1 * l_smooth

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            print(f"epoch {epoch} loss {loss.item():.6f}")
            torch.save(model.state_dict(), f'./unet/{epoch}.pt')

            gt_depth_max = gt_depth_L[gt_mask_L > 0.5].max().item()
            gt_depth_min = gt_depth_L[gt_mask_L > 0.5].min().item()
            gt_depth_show = torch.zeros((H,W)).to(device)
            gt_depth_show[gt_mask_L[0,0]>0.5] = ((gt_depth_L[0,0,gt_mask_L[0,0]>0.5] - gt_depth_min) / (gt_depth_max - gt_depth_min) * 255.0)
            cv2.imwrite("gt_depth.png", gt_depth_show.cpu().numpy().astype(np.uint8))

            pred_depth_max = pred_depth_L[gt_mask_L > 0.5].max().item()
            pred_depth_min = pred_depth_L[gt_mask_L > 0.5].min().item()
            pred_depth_show = torch.zeros((H,W)).to(device)
            pred_depth_show[gt_mask_L[0,0]>0.5] = ((pred_depth_L[0,0,gt_mask_L[0,0]>0.5] - pred_depth_min) / (pred_depth_max - pred_depth_min) * 255.0)
            cv2.imwrite("pred_depth.png", pred_depth_show.detach().cpu().numpy().astype(np.uint8))
        
    else:
        model.load_state_dict(torch.load('./unet/99.pt'))
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tloader_test):
                left_mask = data['maskL'].to(device)
                right_mask = data['maskR'].to(device)
                geometry_L = {
                    'K1': data['K1'].numpy()[0], 'R1': data['R1'].numpy()[0], 't1': data['t1'].numpy()[0],
                    'K2': data['K2'].numpy()[0], 'R2': data['R2'].numpy()[0], 't2': data['t2'].numpy()[0]
                }
                gt_depth_L = data['depthL'].to(device)
                gt_mask_L = (gt_depth_L > 0.05).float()

                pred_depth_L, S, prob = model(left_mask, right_mask, geometry_L, depth_values)

                pred_pts = []
                pts = depth_to_point_cam_coords(gt_depth_L * gt_mask_L, geometry_L['K1']).squeeze(0).T.reshape(H,W,3)
                # Xcam_valid = Xcam[0:3, gt_mask_L[0,0]>0.5].T.cpu().numpy()
                pts = pts[gt_mask_L[0,0]>0.5].cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                o3d.io.write_point_cloud(f"./output/gt_{i}.ply", pcd)

                pred_pts = []
                pts = depth_to_point_cam_coords(pred_depth_L * gt_mask_L, geometry_L['K1']).squeeze(0).T.reshape(H,W,3)
                # Xcam_valid = Xcam[0:3, gt_mask_L[0,0]>0.5].T.cpu().numpy()
                pts = pts[gt_mask_L[0,0]>0.5].cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                o3d.io.write_point_cloud(f"./output/pred_{i}.ply", pcd)

                gt_depth_max = gt_depth_L[gt_mask_L > 0.5].max().item()
                gt_depth_min = gt_depth_L[gt_mask_L > 0.5].min().item()
                gt_depth_show = torch.zeros((H,W)).to(device)
                gt_depth_show[gt_mask_L[0,0]>0.5] = ((gt_depth_L[0,0,gt_mask_L[0,0]>0.5] - gt_depth_min) / (gt_depth_max - gt_depth_min) * 255.0)
                cv2.imwrite(f"./output/gt_{i}.png", gt_depth_show.cpu().numpy().astype(np.uint8))

                pred_depth_max = pred_depth_L[gt_mask_L > 0.5].max().item()
                pred_depth_min = pred_depth_L[gt_mask_L > 0.5].min().item()
                pred_depth_show = torch.zeros((H,W)).to(device)
                pred_depth_show[gt_mask_L[0,0]>0.5] = ((pred_depth_L[0,0,gt_mask_L[0,0]>0.5] - pred_depth_min) / (pred_depth_max - pred_depth_min) * 255.0)
                cv2.imwrite(f"./output/pred_{i}.png", pred_depth_show.detach().cpu().numpy().astype(np.uint8))