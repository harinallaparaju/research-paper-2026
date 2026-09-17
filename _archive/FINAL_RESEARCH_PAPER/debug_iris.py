import os, numpy as np, sys

sys.path.insert(0, '/Users/suryanallaparaju/Desktop/Surya\'s Project/code/FINAL_RESEARCH_PAPER/architectures')
import Architecture_A

iris_dir = '/Users/suryanallaparaju/Desktop/Surya\'s Project/data/Iris/Extracted_Codes'
files = sorted([f for f in os.listdir(iris_dir) if f.endswith('.txt')])

codes = {}
for f in files:
    codes[f] = np.array([int(x) for x in open(os.path.join(iris_dir, f)).read().strip().split(',')])

users = sorted(list(set([f.split('_')[0] for f in files])))

genuine_dists = []
for u in users:
    u_files = [f for f in files if f.split('_')[0] == u]
    for i in range(len(u_files)):
        for j in range(i+1, len(u_files)):
            genuine_dists.append(Architecture_A.fhd(codes[u_files[i]], codes[u_files[j]]))

imposter_dists = []
# Match user 1 against user 2, 3, 4, etc
valid_users = [u for u in users if u + '_1.txt' in codes]
count = 0
for i in range(len(valid_users)):
    for j in range(i+1, min(i+20, len(valid_users))):
        imposter_dists.append(Architecture_A.fhd(codes[valid_users[i]+'_1.txt'], codes[valid_users[j]+'_1.txt']))
        count += 1

print(f'\n--- IRIS ONLY VALIDATION ---')
print(f'Genuine Dists (Mean): {np.mean(genuine_dists):.4f} Min: {np.min(genuine_dists):.4f} Max: {np.max(genuine_dists):.4f} (Count: {len(genuine_dists)})')
print(f'Imposter Dists (Mean): {np.mean(imposter_dists):.4f} Min: {np.min(imposter_dists):.4f} Max: {np.max(imposter_dists):.4f} (Count: {len(imposter_dists)})')
print(f'\nThreshold 0.20')
print(f'  Genuines correctly passing (< 0.20): {sum(d < 0.20 for d in genuine_dists)} / {len(genuine_dists)}')
print(f'  Imposters incorrectly passing (< 0.20): {sum(d < 0.20 for d in imposter_dists)} / {len(imposter_dists)}')
print(f'\nThreshold 0.35')
print(f'  Genuines correctly passing (< 0.35): {sum(d < 0.35 for d in genuine_dists)} / {len(genuine_dists)}')
print(f'  Imposters incorrectly passing (< 0.35): {sum(d < 0.35 for d in imposter_dists)} / {len(imposter_dists)}')
