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
    def __init__(self, pts):
        self.pts = pts

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        uv = self.pts[idx]       # [N,2]
        uv = torch.tensor(uv, dtype=torch.float32)

        N = uv.shape[0]
        t = torch.linspace(0, 1, N)

        perm = torch.randperm(N)
        uv = uv[perm]
        t = t[perm]

        return uv, t


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    uvs, ts = zip(*batch)
    max_len = max(x.shape[0] for x in uvs)

    uv_pad, t_pad, mask = [], [], []

    for uv, t in zip(uvs, ts):
        N = uv.shape[0]
        pad_len = max_len - N

        uv_pad.append(torch.cat([uv, torch.zeros(pad_len, 2)], dim=0))
        t_pad.append(torch.cat([t, torch.zeros(pad_len)], dim=0))

        mask.append(torch.cat([torch.ones(N), torch.zeros(pad_len)], dim=0))

    uv_pad = torch.stack(uv_pad)        # [B,max_len,2]
    t_pad = torch.stack(t_pad)          # [B,max_len]
    mask = torch.stack(mask)            # [B,max_len]

    return uv_pad, t_pad, mask

# class OrderNet(nn.Module):
#     def __init__(self, dim=128, depth=6):
#         super().__init__()
#         self.in_fc = nn.Linear(2, dim)

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=dim, nhead=8,
#             dim_feedforward=256,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

#         self.head = nn.Linear(dim, 1)

#     def forward(self, uv, mask):
#         x = self.in_fc(uv)
#         key_padding_mask = (mask == 0)

#         x = self.transformer(x, src_key_padding_mask=key_padding_mask)

#         s = self.head(x).squeeze(-1)    # [B,N]
#         return s

def pairwise_ranking_loss(pred, gt, mask=None):
    B, N = pred.shape
    loss = 0.0
    total_pairs = 0

    for b in range(B):
        valid_idx = torch.arange(N) if mask is None else torch.where(mask[b])[0]
        p = pred[b, valid_idx]       # [n]
        g = gt[b, valid_idx]         # [n]
        
        diff_pred = p[:, None] - p[None, :]   # [n, n]
        diff_gt   = g[:, None] - g[None, :]   # [n, n]
        tril_mask = torch.tril(torch.ones_like(diff_gt), diagonal=0)
        diff_pred = diff_pred[tril_mask==0]
        diff_gt   = diff_gt[tril_mask==0]

        pair_loss = torch.relu(-diff_pred * diff_gt)
        loss += pair_loss.mean()
        total_pairs += 1

    return loss / total_pairs

class OrderNet(nn.Module):
    def __init__(self, dim=256, depth=8, nhead=8, ff_dim=512):
        super().__init__()
        self.in_fc = nn.Linear(5, dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Linear(dim, 1)

    def forward(self, uv, mask):
        diff = torch.zeros_like(uv)
        diff[:,1:,:] = uv[:,1:,:] - uv[:,:-1,:]
        cum_len = torch.cumsum(torch.norm(diff, dim=-1, keepdim=True), dim=1)
        feat = torch.cat([uv, diff, cum_len], dim=-1)  # [B,N,5]

        x = self.in_fc(feat)
        key_padding_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        s = self.head(x).squeeze(-1)  # [B, N]
        return s

def pairwise_sampled_loss(pred, gt, mask, num_neg=10):
    B, N = pred.shape
    device = pred.device
    loss_total = 0.0
    valid_counts = 0
    T = 5

    for b in range(B):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        if len(valid_idx) < 2:
            continue
        s = pred[b, valid_idx]
        y = gt[b, valid_idx]

        for i in range(len(valid_idx)):
            neg_idx = torch.where(y > y[i])[0]
            if len(neg_idx) == 0:
                continue
            if len(neg_idx) > num_neg:
                neg_idx = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]
            diff = (s[neg_idx] - s[i]) * T
            loss_total += -(F.logsigmoid(diff)).mean()
            valid_counts += 1

    if valid_counts > 0:
        return loss_total / valid_counts
    else:
        return torch.tensor(0., device=device)

def regression_loss(pred, gt, mask):
    return F.mse_loss(pred[mask==1], gt[mask==1])

def ranking_loss(s, t, mask):
    B, N = s.shape
    total = 0
    cnt = 0

    for b in range(B):
        valid = mask[b] == 1
        sb = s[b][valid]
        tb = t[b][valid]

        idx = torch.argsort(tb)
        sb = sb[idx]

        diff = sb.unsqueeze(0) - sb.unsqueeze(1)  # [N,N]
        triu = torch.triu(torch.ones_like(diff), diagonal=1)

        loss = F.relu(diff) * triu
        total += loss.mean()
        cnt += 1

    return total / cnt


def reg_loss(s, t, mask):
    return ((s - t).abs() * mask).sum() / mask.sum()

# def ranking_loss(pred, gt, mask=None, margin=1.0):
#     B, N = pred.shape
#     loss_all = []

#     for b in range(B):
#         p = pred[b]   # [N]
#         g = gt[b]     # [N]

#         if mask is not None:
#             valid = mask[b] == 1
#             p = p[valid]
#             g = g[valid]

#         # 得到所有i<j的 pair
#         idx = torch.argsort(g)
#         p_sorted = p[idx]
#         g_sorted = g[idx]

#         # 构建pairwise
#         # diff_p[j] - diff_p[i] 希望 > margin
#         diff_p = p_sorted.unsqueeze(0) - p_sorted.unsqueeze(1)
#         diff_g = g_sorted.unsqueeze(0) - g_sorted.unsqueeze(1)

#         # 只保留 gt[i] < gt[j] 的 pair
#         mask_pos = diff_g < 0

#         # hinge ranking loss
#         loss = torch.relu(margin - (-diff_p[mask_pos]))  # s_j - s_i
#         loss_all.append(loss.mean())

#     return torch.stack(loss_all).mean()


# def reg_loss(pred, gt, mask=None):
#     B, N = pred.shape
#     reg_all = []

#     for b in range(B):
#         p = pred[b]
#         g = gt[b]

#         if mask is not None:
#             valid = mask[b] == 1
#             p = p[valid]
#             g = g[valid]

#         idx = torch.argsort(g)
#         p_sorted = p[idx]

#         diff = p_sorted[1:] - p_sorted[:-1]
#         reg = (diff ** 2).mean()
#         reg_all.append(reg)

#     return torch.stack(reg_all).mean()

@torch.no_grad()
def predict_order(model, uv_np):
    model.eval()

    uv = torch.tensor(uv_np, dtype=torch.float32).unsqueeze(0).cuda()
    N = uv.shape[1]
    mask = torch.ones(1, N).cuda()

    s = model(uv, mask)[0].cpu().numpy()  # [N,]

    idx = np.argsort(s)
    return uv_np[idx], idx

if __name__ == "__main__":
    epochs = 10
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

        dataset = CurveOrderDataset(pts)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        model = OrderNet().to(device)
        # model.load_state_dict(torch.load("ordernet_l2.pth"))
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            loss_epoch = 0
            loss_epoch_rank = 0
            loss_epoch_reg = 0
            loss_epoch_pair = 0
            for uv, t, mask in loader:
                uv = uv.to(device)
                t = t.to(device)
                mask = mask.to(device)

                s = model(uv, mask)

                # loss_pair = pairwise_sampled_loss(s, t, mask, num_neg=100)
                # loss_reg = regression_loss(s, t, mask)
                # loss = loss_pair + loss_pair

                loss = 

                optim.zero_grad()
                loss.backward()
                optim.step()

                loss_epoch += loss.item()
                # loss_epoch_rank += L_rank.item()

            print(f"Epoch {epoch:03d}  Loss={loss_epoch:.6f}")

        torch.save(model.state_dict(), "ordernet_pair.pth")

    else:
        pts = []
        for i in range(num_train, num_train+num_test):
            data_i = torch.load(f'./dataset/processed_data/data_{i+1}.pt')
            img = np.zeros((H, W))

            uv1 = np.rint(data_i['uv1']).numpy().astype(np.int32)
            uv1 = remove_duplicates_keep_last(uv1)
            uv2 = np.rint(data_i['uv2']).numpy().astype(np.int32)
            uv2 = remove_duplicates_keep_last(uv2)

            pts.append(uv1)
            pts.append(uv2)

        dataset = CurveOrderDataset(pts)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        model = OrderNet().to(device)
        model.load_state_dict(torch.load("ordernet_pair.pth"))
        model.eval()
        for uv, t, mask in loader:
            uv = uv.to(device)
            mask = mask.to(device)

            s = model(uv, mask)

            img_gt = np.zeros((H,W))
            img_gt[np.int64(uv[0,:,1].cpu().numpy()), np.int64(uv[0,:,0].cpu().numpy())] = 255 * t[0].cpu().numpy()
            cv2.imwrite('gt.png', img_gt)
            
            img_pred = np.zeros((H,W))
            img_pred[np.int64(uv[0,:,1].cpu().numpy()), np.int64(uv[0,:,0].cpu().numpy())] = 255 * (s[0].detach().cpu().numpy() - s[0].detach().cpu().numpy().min()) / (s[0].detach().cpu().numpy().max() - s[0].detach().cpu().numpy().min())
            cv2.imwrite('pred.png', img_pred)
            
            t, idx = torch.sort(t[0])
            s = s[0][idx]
            s_sort, idx_s = torch.sort(s)
            diff_idx = idx_s - torch.arange(s.shape[0]).to(device)
            print(diff_idx.max(), diff_idx.min())
            quit()