"""
Test improved iris encoding parameters to boost d'.
Current: 2 wavelengths [18,36], rows 20-44 → d'=1.97
Goal: Higher d' to make stabilizer-based architectures viable.
"""
import json, numpy as np, time, cv2
from pathlib import Path
from iris_extraction.encoding import encode_iris
from iris_extraction.matching import fractional_hamming_distance
from iris_extraction.segmentation import segment_iris
from iris_extraction.normalization import normalize_iris

db = 'fvc2002_1'
with open(f'chimeric_db/chimeric_{db}_min2.json') as f:
    data = json.load(f)

IRIS_ROOT = Path('../data/Iris/CASIA-Iris-Interval')

def load_normalized_for_subject(entry, n_impressions=3):
    """Load normalized iris strips for a chimeric subject."""
    results = {}
    for pair in entry['pairs'][:n_impressions]:
        imp = pair['impression'] - 1
        iris_file = pair['iris_file']
        # iris_file: .../iris_codes/S007_L_S1007L01.npz
        npz_name = Path(iris_file).stem  # S007_L_S1007L01
        parts = npz_name.split('_')      # ['S007', 'L', 'S1007L01']
        subj_dir = parts[0][1:]           # '007'
        img_name = parts[2] + '.jpg'      # S1007L01.jpg
        img_path = IRIS_ROOT / subj_dir / 'L' / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        seg = segment_iris(img)
        if seg is None:
            continue
        norm, mask = normalize_iris(
            img, seg.pupil_center, seg.pupil_radius,
            seg.iris_center, seg.iris_radius, seg.noise_mask
        )
        results[imp] = {'norm': norm, 'mask': mask}
    return results

# Test configurations
configs = [
    {'name': 'Current (2λ, rows 20-44)',  'wavelengths': [18, 36],          'code_rows': (20, 44)},
    {'name': '4λ, rows 20-44',             'wavelengths': [9, 18, 36, 72],   'code_rows': (20, 44)},
    {'name': '2λ, rows 10-54',             'wavelengths': [18, 36],          'code_rows': (10, 54)},
    {'name': '4λ, rows 10-54',             'wavelengths': [9, 18, 36, 72],   'code_rows': (10, 54)},
    {'name': '3λ, rows 12-52',             'wavelengths': [12, 24, 48],      'code_rows': (12, 52)},
    {'name': '4λ, rows 8-56',              'wavelengths': [9, 18, 36, 72],   'code_rows': (8, 56)},
]

# Load normalized iris strips for first 40 subjects
print("Loading normalized iris strips...")
t0 = time.time()
norm_data = []
for entry in data['mapping'][:40]:
    norms = load_normalized_for_subject(entry, n_impressions=3)
    norm_data.append(norms)
print(f"Loaded {len(norm_data)} subjects in {time.time()-t0:.1f}s")

# For each config, encode and compute FHD statistics
for cfg in configs:
    wl = cfg['wavelengths']
    cr = cfg['code_rows']
    name = cfg['name']
    
    # Encode all
    codes = []  # codes[subj_idx][imp] = (code, mask)
    for norms in norm_data:
        subj_codes = {}
        for imp, nm in norms.items():
            code, cmask = encode_iris(nm['norm'], nm['mask'], wavelengths=wl, code_rows=cr)
            subj_codes[imp] = (code, cmask)
        codes.append(subj_codes)
    
    # Genuine FHD (imp0 vs imp1)
    gen_fhds = []
    for sc in codes:
        if 0 in sc and 1 in sc:
            fhd = fractional_hamming_distance(sc[0][0], sc[0][1], sc[1][0], sc[1][1], max_shift=16)
            gen_fhds.append(fhd)
    
    # Impostor FHD (first 25 subjects, imp0 vs imp0)
    imp_fhds = []
    for i in range(min(25, len(codes))):
        for j in range(i+1, min(25, len(codes))):
            if 0 in codes[i] and 0 in codes[j]:
                fhd = fractional_hamming_distance(codes[i][0][0], codes[i][0][1], codes[j][0][0], codes[j][0][1], max_shift=16)
                imp_fhds.append(fhd)
    
    gen = np.array(gen_fhds) if gen_fhds else np.array([0.5])
    imp = np.array(imp_fhds) if imp_fhds else np.array([0.5])
    dprime = (imp.mean() - gen.mean()) / np.sqrt((gen.std()**2 + imp.std()**2)/2)
    
    code_len = len(codes[0][0][0]) if 0 in codes[0] else 0
    mask_cov = np.mean([sc[0][1].mean() for sc in codes if 0 in sc])
    
    # FRR estimate: fraction of genuine pairs with FHD > 0.35
    frr_35 = (gen >= 0.35).mean() * 100
    frr_40 = (gen >= 0.40).mean() * 100
    
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"  Code length: {code_len} bits")
    print(f"  Mask coverage: {mask_cov:.3f}")
    print(f"  Genuine FHD:  mean={gen.mean():.4f} std={gen.std():.4f}")
    print(f"  Impostor FHD: mean={imp.mean():.4f} std={imp.std():.4f}")
    print(f"  d' = {dprime:.2f}")
    print(f"  Genuine FHD >= 0.35: {frr_35:.1f}%")
    print(f"  Genuine FHD >= 0.40: {frr_40:.1f}%")
