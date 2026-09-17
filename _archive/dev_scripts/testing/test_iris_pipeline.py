#!/usr/bin/env python3
"""Quick test of iris extraction pipeline on a few sample images."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_extraction.segmentation import segment_iris
from iris_extraction.normalization import normalize_iris
from iris_extraction.encoding import encode_iris
from iris_extraction.matching import fractional_hamming_distance
import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "Iris" / "CASIA-Iris-Interval"

# Test on more subjects for broader validation
test_images = []
for subj in ["001", "002", "003", "050", "100", "150", "200"]:
    d = DATA / subj / "L"
    if d.exists():
        imgs = sorted(d.glob("*.jpg"))[:3]  # Up to 3 images per subject
        test_images.extend(imgs)

if not test_images:
    print("No test images found!")
    sys.exit(1)

codes = {}
for img_path in test_images:
    if not img_path.exists():
        print(f"MISSING: {img_path.name}")
        continue

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    seg = segment_iris(img)
    if not seg.success:
        print(f"SEG FAIL: {img_path.name} - {seg.failure_reason}")
        continue

    norm, nmask = normalize_iris(
        img, seg.pupil_center, seg.pupil_radius,
        seg.iris_center, seg.iris_radius, seg.noise_mask
    )
    code, cmask = encode_iris(norm, nmask)
    mask_cov = np.mean(cmask) * 100

    print(f"OK: {img_path.name}  pupil=({seg.pupil_center}, r={seg.pupil_radius})  "
          f"iris=({seg.iris_center}, r={seg.iris_radius})  "
          f"code_len={len(code)}  mask={mask_cov:.1f}%")
    codes[img_path.name] = (code, cmask)

# Test matching
print("\n--- FHD Matching ---")
names = list(codes.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        c1, m1 = codes[names[i]]
        c2, m2 = codes[names[j]]
        fhd = fractional_hamming_distance(c1, m1, c2, m2, max_shift=16)
        same_subj = names[i][1:5] == names[j][1:5]  # e.g., "1001" == "1001"
        label = "GENUINE" if same_subj else "IMPOSTOR"
        print(f"  {names[i]} vs {names[j]}: FHD={fhd:.4f} [{label}]")
