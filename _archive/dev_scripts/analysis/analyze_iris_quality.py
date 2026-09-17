"""Quick analysis of iris quality and the fundamental bottleneck."""
import json, numpy as np
from pathlib import Path
from iris_extraction.matching import fractional_hamming_distance

db = 'fvc2002_1'
with open(f'chimeric_db/chimeric_{db}_min2.json') as f:
    data = json.load(f)

# Load all subjects, first 3 impressions each
subjects = []
for entry in data['mapping']:
    iris_data = {}
    for pair in entry['pairs'][:3]:
        imp = pair['impression'] - 1
        d = np.load(pair['iris_file'])
        iris_data[imp] = {'code': d['code'], 'mask': d['mask']}
    subjects.append(iris_data)

# Genuine FHD: imp 0 vs imp 1
gen_fhds = []
for s in subjects:
    if 0 in s and 1 in s:
        fhd = fractional_hamming_distance(
            s[0]['code'], s[0]['mask'],
            s[1]['code'], s[1]['mask'], max_shift=16
        )
        gen_fhds.append(fhd)

# Genuine FHD: imp 0 vs imp 2
gen_fhds_02 = []
for s in subjects:
    if 0 in s and 2 in s:
        fhd = fractional_hamming_distance(
            s[0]['code'], s[0]['mask'],
            s[2]['code'], s[2]['mask'], max_shift=16
        )
        gen_fhds_02.append(fhd)

# Impostor FHD: subject i imp 0 vs subject j imp 0
imp_fhds = []
for i in range(min(30, len(subjects))):
    for j in range(i+1, min(30, len(subjects))):
        if 0 in subjects[i] and 0 in subjects[j]:
            fhd = fractional_hamming_distance(
                subjects[i][0]['code'], subjects[i][0]['mask'],
                subjects[j][0]['code'], subjects[j][0]['mask'], max_shift=16
            )
            imp_fhds.append(fhd)

gen = np.array(gen_fhds)
gen02 = np.array(gen_fhds_02)
imp = np.array(imp_fhds)

print("=" * 60)
print("IRIS QUALITY ANALYSIS")
print("=" * 60)
print(f"\nGenuine FHD (imp0 vs imp1): n={len(gen)}")
print(f"  mean={gen.mean():.4f} std={gen.std():.4f} min={gen.min():.4f} max={gen.max():.4f}")
print(f"\nGenuine FHD (imp0 vs imp2): n={len(gen02)}")
print(f"  mean={gen02.mean():.4f} std={gen02.std():.4f} min={gen02.min():.4f} max={gen02.max():.4f}")
print(f"\nImpostor FHD: n={len(imp)}")
print(f"  mean={imp.mean():.4f} std={imp.std():.4f} min={imp.min():.4f} max={imp.max():.4f}")

dprime = (imp.mean() - gen.mean()) / np.sqrt((gen.std()**2 + imp.std()**2)/2)
print(f"\nd-prime = {dprime:.2f}")

print("\nGenuine FHD distribution:")
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    pct = (gen >= t).sum() / len(gen) * 100
    print(f"  FHD >= {t:.2f}: {pct:5.1f}% ({(gen >= t).sum()}/{len(gen)})")

print("\nImpostor FHD distribution:")
for t in [0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
    pct = (imp <= t).sum() / len(imp) * 100
    print(f"  FHD <= {t:.2f}: {pct:5.1f}% ({(imp <= t).sum()}/{len(imp)})")

# Check mask coverage
mask_fracs = []
for s in subjects:
    if 0 in s:
        mask_fracs.append(s[0]['mask'].mean())
print(f"\nMask coverage: mean={np.mean(mask_fracs):.3f} min={np.min(mask_fracs):.3f} max={np.max(mask_fracs):.3f}")

# Identify the problematic subjects (high genuine FHD)
print("\nProblematic subjects (genuine FHD > 0.40):")
for idx, fhd in enumerate(gen_fhds):
    if fhd > 0.40:
        # Check mask coverage for this subject
        m0 = subjects[idx][0]['mask'].mean() if 0 in subjects[idx] else 0
        m1 = subjects[idx][1]['mask'].mean() if 1 in subjects[idx] else 0
        print(f"  Subject {idx}: FHD={fhd:.4f}, mask0={m0:.3f}, mask1={m1:.3f}")

# What if we use multi-instance enrollment (best of imp0, imp2)?
print("\n" + "=" * 60)
print("MULTI-INSTANCE ENROLLMENT ANALYSIS")
print("=" * 60)
# For subjects with 3 impressions, enroll with imp0, test with imp1
# vs. enroll with best-of(imp0, imp2), test with imp1
improved = 0
for idx, s in enumerate(subjects):
    if 0 in s and 1 in s and 2 in s:
        fhd_01 = gen_fhds[idx]
        fhd_21 = fractional_hamming_distance(
            s[2]['code'], s[2]['mask'],
            s[1]['code'], s[1]['mask'], max_shift=16
        )
        if fhd_21 < fhd_01:
            improved += 1
print(f"Subjects where imp2 enrollment gives lower FHD than imp0: {improved}/{len([s for s in subjects if 0 in s and 1 in s and 2 in s])}")
