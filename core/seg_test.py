import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from model import SelfAttentionUNet, Dataset2D

num_train = 100
num_eval = 10
num_test = 20
H, W = 2048, 512

batch_size = 1
num_epochs = 3
# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

img_test = torch.zeros((2*num_test, 1, H, W), dtype=torch.float32)
mask_test = torch.zeros((2*num_test, 1, H, W), dtype=torch.float32)

for i in range(num_test):
    data_i = torch.load(f'./dataset/processed_data/data_{i+1+num_train+num_eval}.pt')
    img_test[2*i,0] = data_i['img1']
    img_test[2*i+1,0] = data_i['img2']
    mask_test[2*i,0] = data_i['mask1']
    mask_test[2*i+1,0] = data_i['mask2']

test_dataset = Dataset2D(img_test, mask_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
)

model = SelfAttentionUNet(in_channels=1, out_channels=1, base_ch=16).to(device)
model.load_state_dict(torch.load('./dataset/unet.pth'))
criterion = torch.nn.BCEWithLogitsLoss()

with torch.no_grad():
    loss_test = 0
    for i, (img, mask) in enumerate(test_loader):
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        pred_img = (torch.sigmoid(pred[0,0]) > 0.5).cpu().numpy()
        gt_img = mask[0,0].cpu().numpy()
        res = i - (i // 2)*2
        cv2.imwrite(f'dataset/processed_data/pred_{(i //2)+num_train+num_eval+1}_{i - (i // 2)*2}.png', pred_img*255)