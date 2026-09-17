import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from tqdm import tqdm

def create_dl_extractor():
    # Load pretrained ResNet18 and modify to output 256-bit floating point vector
    model = models.resnet18(pretrained=True)
    # Strip the final classification layer and replace with a 256-out linear layer
    model.fc = nn.Linear(model.fc.in_features, 256)
    
    # Set to evaluation mode
    model.eval()
    
    # Define standard image transforms for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, transform

def extract_and_quantize(model, transform, img_path):
    try:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            features = model(img_tensor).squeeze().numpy()
            
        # 1. Magnitude-Based Masking
        # Get absolute values of neural activations
        abs_features = np.abs(features)
        
        # Find the threshold for the top 128 most confident bits
        threshold_val = np.percentile(abs_features, 50)  # Top 50% = 128 bits
        
        # 2. Binary Extraction (Only for confident bits, others set to '0')
        binary_string = ""
        for feat in features:
            if abs(feat) >= threshold_val:
                binary_string += "1" if feat > 0 else "0"
            else:
                binary_string += "0"  # Blanked out (Ignored by math later)
                
        return binary_string
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_iris_database(db_path, output_path):
    print("Initializing Deep Learning Iris Extractor (ResNet-18)...")
    model, transform = create_dl_extractor()
    print(f"Reading database from: {db_path}")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    image_paths = []
    for root, _, files in os.walk(db_path):
        for file in files:
            if file.lower().endswith(('.bmp', '.png', '.jpg', '.tiff')):
                image_paths.append(os.path.join(root, file))
                
    if not image_paths:
        print("No images found in the dataset path!")
        return

    print(f"Found {len(image_paths)} images. Starting extraction...")
    
    # Process with visible progress bar
    for img_path in tqdm(image_paths, desc="Extracting 256-bit Iris Features"):
        binary_code = extract_and_quantize(model, transform, img_path)
        
        if binary_code:
            # Reconstruct relative path to save in same structure
            rel_path = os.path.relpath(img_path, db_path)
            out_file = os.path.join(output_path, rel_path)
            out_file = os.path.splitext(out_file)[0] + '.txt'
            
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w') as f:
                f.write(binary_code)

if __name__ == "__main__":
    # Adjust these paths to your CASIA-Iris-Interval database
    DB_DIR = "../../../data/Iris/CASIA-Iris-Interval"
    OUT_DIR = "../../../data/Iris/DL_Extracted_Codes"
    process_iris_database(DB_DIR, OUT_DIR)
    print(f"Extraction complete! Features saved to: {OUT_DIR}")
