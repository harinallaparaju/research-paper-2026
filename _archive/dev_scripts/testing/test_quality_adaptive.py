"""
Test quality-adaptive Architecture A: only require iris check when 
the iris pair quality (mask overlap) is high enough.

Idea: 
- High mask overlap → iris match reliable → use iris + FP (zero FAR)
- Low mask overlap → iris match unreliable → use FP only (zero FRR)
- This should give: near-zero FAR AND near-zero FRR
"""
import json, numpy as np
from pathlib import Path
from iris_extraction.matching import fractional_hamming_distance

db = 'fvc2002_1'
with open(f'chimeric_db/chimeric_{db}_min2.json') as f:
    data = json.load(f)

# Load all subjects
print("Loading data...")
subjects = []
for entry in data['mapping']:
    subj = {'iris': {}}
    for pair in entry['pairs'][:2]:  # imp 0 and 1 only for Setup 1
        imp = pair['impression'] - 1
        d = np.load(pair['iris_file'])
        subj['iris'][imp] = {'code': d['code'], 'mask': d['mask']}
    subjects.append(subj)
print(f"Loaded {len(subjects)} subjects\n")

# For each (enroll, probe) pair, compute:
# 1. Iris FHD
# 2. Mask overlap ratio
# 3. Whether it's genuine or impostor

print("Computing all FHD and mask overlap values...")
results = []
n_subj = len(subjects)
for i in range(n_subj):
    if 0 not in subjects[i]['iris']:
        continue
    enroll_code = subjects[i]['iris'][0]['code']
    enroll_mask = subjects[i]['iris'][0]['mask']
    
    for j in range(n_subj):
        if 1 not in subjects[j]['iris']:
            continue
        probe_code = subjects[j]['iris'][1]['code']
        probe_mask = subjects[j]['iris'][1]['mask']
        
        is_genuine = (i == j)
        
        # Compute FHD
        fhd = fractional_hamming_distance(
            enroll_code, enroll_mask,
            probe_code, probe_mask, max_shift=16
        )
        
        # Mask overlap ratio (fraction of bits valid in both)
        overlap = np.sum(enroll_mask & probe_mask) / len(enroll_mask)
        
        results.append({
            'enroll': i, 'probe': j,
            'genuine': is_genuine,
            'fhd': fhd, 'overlap': overlap
        })

genuine_results = [r for r in results if r['genuine']]
impostor_results = [r for r in results if not r['genuine']]
print(f"Genuine pairs: {len(genuine_results)}")
print(f"Impostor pairs: {len(impostor_results)}")

# Standard Architecture A: fixed threshold
print("\n" + "=" * 70)
print("STANDARD ARCHITECTURE A (fixed iris threshold)")
print("=" * 70)
for tau in [0.35, 0.38, 0.40, 0.42, 0.44, 0.46]:
    gen_pass = sum(1 for r in genuine_results if r['fhd'] < tau)
    gen_fail = len(genuine_results) - gen_pass
    imp_pass = sum(1 for r in impostor_results if r['fhd'] < tau)
    imp_fail = len(impostor_results) - imp_pass
    frr = gen_fail / len(genuine_results) * 100
    far = imp_pass / len(impostor_results) * 100
    print(f"  τ={tau:.2f}: FAR={far:.3f}% FRR={frr:.1f}% (gen_pass={gen_pass} imp_pass={imp_pass})")

# Quality-Adaptive Architecture A: require iris only when quality is high
print("\n" + "=" * 70)
print("QUALITY-ADAPTIVE ARCHITECTURE A (iris only when quality > q_min)")
print("=" * 70)
print("When quality < q_min: ACCEPT (iris bypassed, rely on FP only)")
print("When quality >= q_min: ACCEPT only if FHD < τ")
print()

for tau in [0.38, 0.40, 0.42]:
    print(f"--- τ = {tau:.2f} ---")
    for q_min in [0.20, 0.25, 0.30, 0.35, 0.40]:
        # Genuine pairs
        gen_accept = 0
        gen_reject = 0
        gen_bypass = 0
        for r in genuine_results:
            if r['overlap'] < q_min:
                gen_accept += 1  # bypass iris, accept
                gen_bypass += 1
            elif r['fhd'] < tau:
                gen_accept += 1  # iris passed
            else:
                gen_reject += 1  # iris failed
        
        # Impostor pairs
        imp_accept = 0
        imp_reject = 0
        imp_bypass = 0
        for r in impostor_results:
            if r['overlap'] < q_min:
                imp_accept += 1  # bypass iris, accept
                imp_bypass += 1
            elif r['fhd'] < tau:
                imp_accept += 1  # iris passed (false accept from iris)
            else:
                imp_reject += 1  # iris rejected
        
        frr = gen_reject / len(genuine_results) * 100
        far = imp_accept / len(impostor_results) * 100
        # Note: far includes bypassed impostors (who would still need FP vault)
        # The real FAR is: P(bypass) * FP_FAR + P(!bypass) * P(iris_pass) * FP_FAR
        # But at iris level (before FP): 
        iris_far = imp_accept / len(impostor_results) * 100
        
        print(f"  q>{q_min:.2f}: iris_FAR={iris_far:.3f}% FRR={frr:.1f}%  "
              f"(bypass: gen={gen_bypass} imp={imp_bypass})")
    print()

# More detailed analysis: what's the relation between overlap and FHD?
print("=" * 70)
print("OVERLAP vs FHD ANALYSIS")
print("=" * 70)
gen_overlaps = np.array([r['overlap'] for r in genuine_results])
gen_fhds = np.array([r['fhd'] for r in genuine_results])
imp_overlaps = np.array([r['overlap'] for r in impostor_results])
imp_fhds = np.array([r['fhd'] for r in impostor_results])

print(f"\nGenuine: overlap mean={gen_overlaps.mean():.3f}, FHD mean={gen_fhds.mean():.4f}")
print(f"Impostor: overlap mean={imp_overlaps.mean():.3f}, FHD mean={imp_fhds.mean():.4f}")

# Correlation between low overlap and high genuine FHD
for q in [0.25, 0.30, 0.35, 0.40]:
    low_q = gen_fhds[gen_overlaps < q]
    high_q = gen_fhds[gen_overlaps >= q]
    if len(low_q) > 0 and len(high_q) > 0:
        print(f"\nOverlap < {q:.2f} ({len(low_q)} pairs): genuine FHD mean={low_q.mean():.4f}")
        print(f"Overlap >= {q:.2f} ({len(high_q)} pairs): genuine FHD mean={high_q.mean():.4f}")
