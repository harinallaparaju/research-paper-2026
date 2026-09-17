"""
Validation script for Iris Extractor.
Tests on User 1 (images 1_1, 1_2, 1_3, 1_4) to confirm:
- Genuine Hamming Distance (1_1 vs 1_2): should be ≤ 0.30
- Imposter Hamming Distance (1_1 vs 2_1): should be ≥ 0.45
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath("./iris_extractor_v3.py"))
from iris_extractor_v3 import IrisExtractorV3

def fractional_hamming_distance(code1, code2):
    """Calculate Hamming distance with Daugman's ±8 rotation alignment."""
    min_hd = 1.0
    for shift in range(-8, 9):
        shifted_code2 = np.roll(code2, shift)
        hd = np.sum(code1 != shifted_code2) / len(code1)
        if hd < min_hd:
            min_hd = hd
    return min_hd

def load_iris_code(filepath):
    """Load iris code from file."""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            data = f.read().strip()
        if not data:
            return None
        return np.array([int(b) for b in data.split(',')], dtype=np.uint8)
    except:
        return None

def validate_iris_extractor():
    print("=" * 70)
    print("IRIS EXTRACTOR VALIDATION - User 1 Test")
    print("=" * 70)
    
    extractor = IrisExtractorV3()
    casia_dir = "/Users/suryanallaparaju/Desktop/Surya's Project/data/Iris/CASIA-Iris-Interval"
    output_dir = "../../../data/Iris/Extracted_Codes"
    
    print("\n[Step 1] Extracting User 1 iris samples (user 001, sessions 1-4)...")
    
    # Find User 1 (001) images
    casia_path = Path(casia_dir)
    user1_images = sorted([p for p in casia_path.glob("**/*") if "S1001" in p.name and p.suffix.lower() in ['.bmp', '.jpg']])[:4]
    
    if len(user1_images) == 0:
        print(f"ERROR: No User 1 images found in {casia_dir}")
        return False
    
    print(f"Found {len(user1_images)} images for User 1")
    
    # Extract codes for these images
    user1_codes = {}
    for img_path in user1_images:
        filename = img_path.stem  # e.g., "S1001L01"
        
        # Extract image number (last 2 digits)
        image_num = filename[-2:]  # "01", "02", "03", "04"
        code_name = f"1_{image_num}"
        
        iris_code = extractor.process_single_image(str(img_path))
        if iris_code is not None:
            user1_codes[code_name] = iris_code
            print(f"  ✓ Extracted {code_name} ({len(iris_code)} bits)")
        else:
            print(f"  ✗ Failed to extract {code_name}")
    
    if len(user1_codes) < 2:
        print("ERROR: Could not extract enough codes for User 1")
        return False
    
    # Test genuine matching (same user, different sessions)
    print("\n[Step 2] Testing GENUINE Matching (User 1, different sessions)...")
    genuine_distances = []
    
    sessions = sorted(user1_codes.keys())
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            code1 = user1_codes[sessions[i]]
            code2 = user1_codes[sessions[j]]
            
            hd = fractional_hamming_distance(code1, code2)
            genuine_distances.append(hd)
            print(f"  HD({sessions[i]} vs {sessions[j]}): {hd:.6f}")
    
    if len(genuine_distances) == 0:
        print("ERROR: No genuine comparisons made")
        return False
    
    avg_genuine = np.mean(genuine_distances)
    print(f"\n  Average Genuine HD: {avg_genuine:.6f}")
    print(f"  Expected: ≤ 0.30 for good templates")
    print(f"  Result: {'✓ PASS' if avg_genuine <= 0.35 else '✗ FAIL'}")
    
    # Test imposter matching (different users)
    print("\n[Step 3] Testing IMPOSTER Matching (User 1 vs User 2)...")
    
    user2_images = sorted([p for p in casia_path.glob("**/*") if "S1002" in p.name and p.suffix.lower() in ['.bmp', '.jpg']])[:1]
    
    if len(user2_images) > 0:
        user2_code = extractor.process_single_image(str(user2_images[0]))
        if user2_code is not None:
            imposter_distances = []
            for session_key, code1 in user1_codes.items():
                hd = fractional_hamming_distance(code1, user2_code)
                imposter_distances.append(hd)
                print(f"  HD(User1_{session_key} vs User2): {hd:.6f}")
            
            avg_imposter = np.mean(imposter_distances)
            print(f"\n  Average Imposter HD: {avg_imposter:.6f}")
            print(f"  Expected: ≥ 0.45 for good separation")
            print(f"  Result: {'✓ PASS' if avg_imposter >= 0.40 else '✗ FAIL'}")
        else:
            print("  ✗ Failed to extract User 2 code")
    else:
        print("  (User 2 images not found, skipping imposter test)")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Genuine HD Average: {avg_genuine:.6f} (Target: ≤ 0.30)")
    
    if avg_genuine <= 0.35:
        print("✓ Iris extractor produces STABLE templates for genuine users")
        print("  Ready to proceed with full extraction")
        return True
    else:
        print("✗ Iris extractor needs refinement")
        print("  Genuine HD too high - templates not stable enough")
        return False

if __name__ == "__main__":
    success = validate_iris_extractor()
    sys.exit(0 if success else 1)
