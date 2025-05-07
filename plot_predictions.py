import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from approach.ResEmoteNet import ResEmoteNet
from get_dataset import Four4All

# Thiết bị
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Transform cho grayscale
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load tập test
test_dataset = Four4All(csv_file='/Users/nguyenquangtruong/Desktop/HocTap2/Python/ResEmoteNet/data/test_labels.csv',
                        img_dir='/Users/nguyenquangtruong/Desktop/HocTap2/Python/ResEmoteNet/data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Load mô hình
model = ResEmoteNet().to(device)
model.load_state_dict(torch.load('/Users/nguyenquangtruong/Desktop/HocTap2/Python/ResEmoteNet/best_model_6emotions.pth', map_location=device))
model.eval()

# Lấy một batch ảnh từ test_loader
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Dự đoán
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# Chuyển dữ liệu về CPU và numpy
images = images.cpu().numpy()
labels = labels.cpu().numpy()
preds = preds.cpu().numpy()

# Tên lớp
class_names = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Fear']

# Vẽ 10 ảnh đầu tiên
plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = images[i].squeeze()  # Loại bỏ kênh (1, 48, 48) -> (48, 48)
    img = img * 0.5 + 0.5  # Đảo ngược chuẩn hóa
    plt.imshow(img, cmap='gray')
    plt.title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('/Users/nguyenquangtruong/Desktop/HocTap2/Python/ResEmoteNet/predictions_plot.png')
plt.show()