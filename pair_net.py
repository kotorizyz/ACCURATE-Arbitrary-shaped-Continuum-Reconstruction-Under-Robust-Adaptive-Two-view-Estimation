import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

    samples = [make_sample(f'dataset/processed_data/data_{i+1}.pt') for i in range(100)]
    dataset = PairsDataset(samples)
    dataloader = DataLoader(dataset, batch_size = 1, collate_fn = collate_fn, shuffle = True)

    model = UNetMatcher(in_ch=5, base_ch=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(100):
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
            loss = criterion(hat_M.squeeze(0).squeeze(0), M)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch += loss.item()
        print(f"Epoch {epoch+1}, Loss: {loss_epoch/len(dataloader)}")
    torch.save(model.state_dict(), 'pair_net.pth')