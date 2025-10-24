# Retry: smaller demo to avoid environment issues.
# Re-run a compact version of the training script on tiny synthetic data (1 epoch, small sizes).
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossMatchModel(nn.Module):
    def __init__(self, d_model=64, use_view_encoders=False, epi_bias_scale=1.0):
        super().__init__()
        self.input_embed = nn.Sequential(nn.Linear(2, d_model),
                                        nn.ReLU(), 
                                        nn.Linear(d_model, d_model))
        self.use_view_encoders = use_view_encoders
        if use_view_encoders:
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2, batch_first=True)
            self.encoder_L = nn.TransformerEncoder(enc, num_layers=1)
            self.encoder_R = nn.TransformerEncoder(enc, num_layers=1)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model)
        self.epi_bias_scale = epi_bias_scale

    def forward(self, ptsL, ptsR, maskL, maskR, epi_dist=None):
        B, N1, _ = ptsL.shape
        _, N2, _ = ptsR.shape
        eL = self.input_embed(ptsL)
        eR = self.input_embed(ptsR)
        if self.use_view_encoders:
            eL = self.encoder_L(eL, src_key_padding_mask=(~maskL))
            eR = self.encoder_R(eR, src_key_padding_mask=(~maskR))
        Q = self.query_proj(eL)
        K = self.key_proj(eR)
        sim = torch.bmm(Q, K.transpose(1,2)) / (self.scale + 1e-9)
        if epi_dist is not None:
            sim = sim - self.epi_bias_scale * epi_dist.to(sim.dtype)
        if maskL is not None:
            sim = sim.masked_fill((~maskL).unsqueeze(-1), float("-1e9"))
        if maskR is not None:
            sim = sim.masked_fill((~maskR).unsqueeze(1), float("-1e9"))
        return sim

def compute_F_batch(K1, R1, t1, K2, R2, t2):
    B = K1.shape[0]
    Fm = torch.zeros((B,3,3), device=K1.device, dtype=K1.dtype)
    for b in range(B):
        Rrel = R2[b] @ R1[b].T
        trel = (t2[b] - Rrel @ t1[b]).reshape(3)
        tx = torch.tensor([[0.0, -trel[2], trel[1]],[trel[2], 0.0, -trel[0]],[-trel[1], trel[0], 0.0]], device=K1.device, dtype=K1.dtype)
        Fm[b] = torch.inverse(K2[b]).T @ tx @ Rrel @ torch.inverse(K1[b])
    return Fm

def epipolar_loss_batch(ptsL, ptsR, M, K1, R1, t1, K2, R2, t2):
    B, N1, _ = ptsL.shape
    _, N2, _ = ptsR.shape
    Fm = compute_F_batch(K1, R1, t1, K2, R2, t2)
    total = 0.0
    count = 0
    for b in range(B):
        idxs = (M[b] > 0.5).nonzero(as_tuple = False)
        if idxs.numel() == 0: continue
        i_idx = idxs[:, 0]
        j_idx = idxs[:, 1]
        x1 = ptsL[b, i_idx]
        x2 = ptsR[b, j_idx]
        x1h = torch.cat([x1, torch.ones((x1.shape[0], 1), device = x1.device)], dim=1)
        x2h = torch.cat([x2, torch.ones((x2.shape[0], 1), device = x2.device)], dim=1)
        Fx1 = (Fm[b] @ x1h.T).T
        e = torch.abs((x2h * Fx1).sum(dim = 1))
        total += e.sum()
        count += e.shape[0]
    return total / (count + 1e-9)

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
    for b,s in enumerate(batch):
        n1 = s['ptsL'].shape[0]
        n2 = s['ptsR'].shape[0]
        if n1>0: 
            ptsL[b,:n1] = s['ptsL']
            maskL[b,:n1] = True
        if n2>0: 
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

# small synthetic data
# def make_sample(N1=16, N2=18):
#     ptsL = torch.rand((N1,2))*127.0
#     ptsR = torch.rand((N2,2))*127.0
#     K = torch.tensor([[120.0,0.0,64.0],[0.0,120.0,64.0],[0.0,0.0,1.0]])
#     R = torch.eye(3); t = torch.zeros((3,1))
#     M = torch.zeros((N1,N2))
#     for i in range(min(N1,N2)//2):
#         j = random.randrange(N2); M[i,j]=1.0
#     return {'ptsL':ptsL, 'ptsR':ptsR, 'K1':K, 'R1':R, 't1':t, 'K2':K, 'R2':R, 't2':t, 'M':M}

def make_sample(dir = ''):
    data_i = torch.load(dir)
    return {'ptsL':data_i['x1'], 'ptsR':data_i['x2'], 'K1':data_i['K1'], 'R1':data_i['RT1'][:,:3], 't1':data_i['RT1'][:,3], 'K2':data_i['K2'], 'R2':data_i['RT2'][:,:3], 't2':data_i['RT2'][:,3], 'M':data_i['M_ij']}
    

# samples = [make_sample(20,22) for _ in range(8)]
samples = [make_sample(f'dataset/processed_data/data_{i+1}.pt') for i in range(100)]
ds = PairsDataset(samples)
dl = DataLoader(ds, batch_size = 2, collate_fn = collate_fn, shuffle = True)

model = CrossMatchModel(d_model = 64, use_view_encoders = False, epi_bias_scale = 0.3).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
bce = nn.BCEWithLogitsLoss()

# one epoch demo training
for i in range(100):
    loss_epo = 0
    for batch in dl:
        ptsL = batch['ptsL'].to(DEVICE)
        ptsR = batch['ptsR'].to(DEVICE)
        maskL = batch['maskL'].to(DEVICE)
        maskR = batch['maskR'].to(DEVICE)
        M = batch['M'].to(DEVICE)
        K1 = batch['K1'].to(DEVICE)
        R1 = batch['R1'].to(DEVICE)
        t1 = batch['t1'].to(DEVICE)
        K2 = batch['K2'].to(DEVICE)
        R2 = batch['R2'].to(DEVICE)
        t2 = batch['t2'].to(DEVICE)

        # compute epi_dist per batch element (normalized)
        with torch.no_grad():
            Fm = compute_F_batch(K1,R1,t1,K2,R2,t2)
            B, N1, _ = ptsL.shape
            _, N2, _ = ptsR.shape
            epi = torch.zeros((B,N1,N2), device=DEVICE)
            for b in range(B):
                x1 = ptsL[b]; x2 = ptsR[b]
                x1h = torch.cat([x1, torch.ones((N1,1), device=DEVICE)], dim = 1)
                x2h = torch.cat([x2, torch.ones((N2,1), device=DEVICE)], dim = 1)
                Fx1 = (Fm[b] @ x1h.T).T  # (N1,3)
                Mpair = torch.abs(x2h @ Fx1.T)  # (N2,N1)
                epi[b] = Mpair.T
            epi = epi / (epi.mean(dim = (1,2), keepdim = True)+1e-9)

        sim = model(ptsL, ptsR, maskL, maskR, epi)
        loss_match = bce(sim, M)
        loss_epi = epipolar_loss_batch(ptsL, ptsR, M, K1, R1, t1, K2, R2, t2)
        loss = loss_match + 0.1 * loss_epi
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_epo += loss.item()
    print("Epoch loss:", loss_epo)

test_sample = make_sample('dataset/processed_data/data_101.pt')
model.eval()
with torch.no_grad():
    ptsL = test_sample['ptsL'].unsqueeze(0).to(DEVICE)
    ptsR = test_sample['ptsR'].unsqueeze(0).to(DEVICE)
    maskL = torch.ones((1, ptsL.shape[1]), dtype=torch.bool).to(DEVICE)
    maskR = torch.ones((1, ptsR.shape[1]), dtype=torch.bool).to(DEVICE)
    K1 = test_sample['K1'].unsqueeze(0).to(DEVICE)
    R1 = test_sample['R1'].unsqueeze(0).to(DEVICE)
    t1 = test_sample['t1'].unsqueeze(0).to(DEVICE)
    K2 = test_sample['K2'].unsqueeze(0).to(DEVICE)
    R2 = test_sample['R2'].unsqueeze(0).to(DEVICE)
    t2 = test_sample['t2'].unsqueeze(0).to(DEVICE)

    Fm = compute_F_batch(K1,R1,t1,K2,R2,t2)
    B, N1, _ = ptsL.shape
    _, N2, _ = ptsR.shape
    epi = torch.zeros((B,N1,N2), device=DEVICE)
    for b in range(B):
        x1 = ptsL[b]; x2 = ptsR[b]
        x1h = torch.cat([x1, torch.ones((N1,1), device=DEVICE)], dim=1)
        x2h = torch.cat([x2, torch.ones((N2,1), device=DEVICE)], dim=1)
        Fx1 = (Fm[b] @ x1h.T).T  # (N1,3)
        Mpair = torch.abs(x2h @ Fx1.T)  # (N2,N1)
        epi[b] = Mpair.T
    epi = epi / (epi.mean(dim=(1,2), keepdim=True)+1e-9)

    sim = model(ptsL, ptsR, maskL, maskR, epi)
    pred_M = (torch.sigmoid(sim) > 0.5).float()
    print("Predicted matches:\n", pred_M[0])