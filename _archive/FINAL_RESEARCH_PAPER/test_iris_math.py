# --- A Quick test to visually see the output of the Iris Extractor ---
import sys, cv2, os, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/suryanallaparaju/Desktop/Surya\'s Project/code/multimodal_implementation/iris_engine')
import iris_extractor_v3

e = iris_extractor_v3.IrisExtractorV3()
img_path = '/Users/suryanallaparaju/Desktop/Surya\'s Project/data/Iris/CASIA-Iris-Interval/001/R/S1001R01.jpg'
img = cv2.imread(img_path)

if img is None:
    print('Image not found')
    sys.exit()

segmented = e.segment_iris_robust(img)
if segmented:
    print('Segmentation SUCCESS:', segmented)
else:
    print('Segmentation FAILED. Using dummy random noise fallback...?')

vec = e.process_single_image(img_path)
print(f'Feature vector: {len(vec) if vec is not None else None}')
if vec is not None:
    print('Sum of 1s in feature vector:', np.sum(vec))
    
# Let's extract the same image twice and compare FHD directly to see if randomness is baked in
vec2 = e.process_single_image(img_path)
md = 1.0
for s in range(-8,9): md = min(md, np.sum(np.array(vec) != np.roll(np.array(vec2), s))/len(vec))
print(f'FHD of identical image to itself: {md:.4f}')

