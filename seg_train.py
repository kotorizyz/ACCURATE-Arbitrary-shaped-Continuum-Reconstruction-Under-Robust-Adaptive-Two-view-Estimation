import torch
from torch.utils.data import Dataset, DataLoader

from model import SelfAttentionUNet, Dataset2D

num_train = 100
num_eval = 10
H, W = 2048, 512

batch_size = 1
num_epochs = 300
# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

img_train = torch.zeros((2*num_train, 1, H, W), dtype=torch.float32)
img_eval = torch.zeros((2*num_eval, 1, H, W), dtype=torch.float32)
mask_train = torch.zeros((2*num_train, 1, H, W), dtype=torch.float32)
mask_eval = torch.zeros((2*num_eval, 1, H, W), dtype=torch.float32)

for i in range(num_train+num_eval):
    data_i = torch.load(f'./dataset/processed_data/data_{i+1}.pt')
    if i < num_train:
        img_train[2*i,0] = data_i['img1']
        img_train[2*i+1,0] = data_i['img2']
        mask_train[2*i,0] = data_i['mask1']
        mask_train[2*i+1,0] = data_i['mask2']
    else:
        img_eval[2*(i-num_train),0] = data_i['img1']
        img_eval[2*(i-num_train)+1,0] = data_i['img2']
        mask_eval[2*(i-num_train),0] = data_i['mask1']
        mask_eval[2*(i-num_train)+1,0] = data_i['mask2']

train_dataset = Dataset2D(img_train, mask_train)
val_dataset = Dataset2D(img_eval, mask_eval)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
)

model = SelfAttentionUNet(in_channels=1, out_channels=1, base_ch=16).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(num_epochs):
    loss_epoch_train = 0
    loss_epoch_eval = 0
    model.train()
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_epoch_train += loss.item()

    model.eval()
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            val_loss = criterion(pred, mask)

            loss_epoch_eval += val_loss.item()
    print(f"Epoch {epoch}:", loss_epoch_train / num_train, loss_epoch_eval / num_eval)

torch.save(model.state_dict(), 'unet.pth')