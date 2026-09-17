import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# 1. Setup Feature Extractor
resnet = models.resnet18(pretrained=True)
# Remove the random FC layer entirely!
extractor = nn.Sequential(*(list(resnet.children())[:-1]))
extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_binary_code(img_path):
    img = Image.open(img_path).convert('RGB') # Fix channel issue
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = extractor(img_tensor).squeeze().numpy()  # 512-dim
    # Simple binarization: above median = 1, else 0 -> balances 1s and 0s
    binary = (features > np.median(features)).astype(int)
    return binary

def fhd(c1, c2):
    md = 1.0
    for s in range(-8,9):
        md = min(md, np.sum(c1 != np.roll(c2, s))/len(c1))
    return md

db_dir = "/Users/suryanallaparaju/Desktop/Surya's Project/data/Iris/CASIA-Iris-Interval"
i1_1 = os.path.join(db_dir, "001", "L", os.listdir(os.path.join(db_dir, "001", "L"))[0])
i1_2 = os.path.join(db_dir, "001", "L", os.listdir(os.path.join(db_dir, "001", "L"))[1])
i2_1 = os.path.join(db_dir, "002", "L", os.listdir(os.path.join(db_dir, "002", "L"))[0])

c1_1 = get_binary_code(i1_1)
c1_2 = get_binary_code(i1_2)
c2_1 = get_binary_code(i2_1)

print("Lens:", len(c1_1))
print(f"Genuine Match FHD: {fhd(c1_1, c1_2):.4f}")
print(f"Impostor Match FHD: {fhd(c1_1, c2_1):.4f}")
