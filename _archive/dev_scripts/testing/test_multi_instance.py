"""
Test multi-instance iris enrollment: majority-vote over N enrollment
codes to create a lower-noise reference template.

If single-impression BER ≈ 0.26, then majority of 3 independent 
impressions gives BER ≈ 3*(0.26^2)*(0.74) + (0.26^3) = 0.168.
This should dramatically drop genuine FHD and reduce FRR.
"""
import json, numpy as np
from pathlib import Path
from iris_extraction.matching import fractional_hamming_distance

db = 'fvc2002_1'
with open(f'chimeric_db/chimeric_{db}_min2.json') as f:
    data = json.load(f)

# Load ALL impressions for each subject
subjects = []
for entry in data['mapping']:
    iris_data = {}
    for pair in entry['pairs']:  # all 8 impressions
        imp = pair['impression'] - 1
        d = np.load(pair['iris_file'])
        iris_data[imp] = {'code': d['code'], 'mask': d['mask']}
    subjects.append(iris_data)

print(f"Loaded {len(subjects)} subjects")
print(f"Impressions per subject: {[len(s) for s in subjects[:5]]}")

def majority_vote_code(codes_and_masks):
    """Create a majority-vote averaged code from N impressions."""
    codes = np.array([c for c, m in codes_and_masks])  # (N, L)
    masks = np.array([m for c, m in codes_and_masks])  # (N, L)
    
    # Weighted majority vote: only count valid bits
    weighted_sum = (codes * masks).sum(axis=0)  # sum of 1s at each position
    total_valid = masks.sum(axis=0)             # how many valid at each position
    
    # Majority: 1 if more than half the valid samples say 1
    avg_code = (weighted_sum > total_valid / 2).astype(np.uint8)
    # Mask: valid if at least 1 sample was valid
    avg_mask = (total_valid > 0).astype(np.uint8)
    
    return avg_code, avg_mask

print("\n" + "=" * 70)
print("EXPERIMENT 1: Single enrollment (imp0) vs test (imp1)")
print("=" * 70)
gen_fhds_single = []
for s in subjects:
    if 0 in s and 1 in s:
        fhd = fractional_hamming_distance(
            s[0]['code'], s[0]['mask'],
            s[1]['code'], s[1]['mask'], max_shift=16
        )
        gen_fhds_single.append(fhd)
gen_s = np.array(gen_fhds_single)
print(f"Genuine FHD: mean={gen_s.mean():.4f} std={gen_s.std():.4f}")
for t in [0.30, 0.35, 0.40, 0.42, 0.44]:
    print(f"  FHD >= {t}: {(gen_s >= t).sum()}/{len(gen_s)} = {(gen_s >= t).mean()*100:.1f}%")

# Multi-instance enrollment: use imps 0,2,4 → test with imp 1
for n_enroll in [2, 3, 4]:
    print(f"\n{'=' * 70}")
    enroll_imps = list(range(0, 2*n_enroll, 2))  # [0,2], [0,2,4], [0,2,4,6]
    test_imp = 1
    print(f"EXPERIMENT: {n_enroll}-instance enrollment (imps {enroll_imps}) vs test (imp {test_imp})")
    print("=" * 70)
    
    gen_fhds_multi = []
    for s in subjects:
        # Check all enrollment impressions exist
        if not all(i in s for i in enroll_imps):
            continue
        if test_imp not in s:
            continue
        
        # Majority-vote enrollment code
        pairs = [(s[i]['code'], s[i]['mask']) for i in enroll_imps]
        avg_code, avg_mask = majority_vote_code(pairs)
        
        fhd = fractional_hamming_distance(
            avg_code, avg_mask,
            s[test_imp]['code'], s[test_imp]['mask'], max_shift=16
        )
        gen_fhds_multi.append(fhd)
    
    gen_m = np.array(gen_fhds_multi)
    print(f"Genuine FHD: mean={gen_m.mean():.4f} std={gen_m.std():.4f} (n={len(gen_m)})")
    for t in [0.30, 0.35, 0.40, 0.42, 0.44]:
        print(f"  FHD >= {t}: {(gen_m >= t).sum()}/{len(gen_m)} = {(gen_m >= t).mean()*100:.1f}%")

# Also check impostor FHD with multi-instance enrollment
print(f"\n{'=' * 70}")
print("IMPOSTOR CHECK: 3-instance enrollment, first 30 subjects")
print("=" * 70)
enroll_imps = [0, 2, 4]
n_check = min(30, len(subjects))
avg_codes = {}
for idx in range(n_check):
    s = subjects[idx]
    if all(i in s for i in enroll_imps):
        pairs = [(s[i]['code'], s[i]['mask']) for i in enroll_imps]
        avg_code, avg_mask = majority_vote_code(pairs)
        avg_codes[idx] = (avg_code, avg_mask)

imp_fhds_multi = []
for i in range(n_check):
    for j in range(i+1, n_check):
        if i in avg_codes and j in avg_codes:
            fhd = fractional_hamming_distance(
                avg_codes[i][0], avg_codes[i][1],
                avg_codes[j][0], avg_codes[j][1], max_shift=16
            )
            imp_fhds_multi.append(fhd)

imp_m = np.array(imp_fhds_multi)
print(f"Impostor FHD (enrolled vs enrolled): mean={imp_m.mean():.4f} std={imp_m.std():.4f}")

# Also do imp enrolled vs single probe
imp_fhds_cross = []
for i in range(n_check):
    for j in range(i+1, n_check):
        s_j = subjects[j]
        if i in avg_codes and 1 in s_j:
            fhd = fractional_hamming_distance(
                avg_codes[i][0], avg_codes[i][1],
                s_j[1]['code'], s_j[1]['mask'], max_shift=16
            )
            imp_fhds_cross.append(fhd)

imp_c = np.array(imp_fhds_cross)
print(f"Impostor FHD (enrolled vs probe): mean={imp_c.mean():.4f} std={imp_c.std():.4f}")

# d' comparison
dprime_single = (np.mean(imp_fhds_cross[:len(gen_fhds_single)]) - gen_s.mean()) / np.sqrt((gen_s.std()**2 + imp_c.std()**2)/2)
# Find the 3-instance gen FHDs matching these subjects
gen_m3 = np.array([gen_fhds_multi[i] for i in range(min(len(gen_fhds_multi), n_check))])
dprime_multi = (imp_c.mean() - gen_m3.mean()) / np.sqrt((gen_m3.std()**2 + imp_c.std()**2)/2)

print(f"\nd' comparison:")
print(f"  Single enrollment: d' = {dprime_single:.2f}")
print(f"  3-instance enrollment: d' = {dprime_multi:.2f}")
