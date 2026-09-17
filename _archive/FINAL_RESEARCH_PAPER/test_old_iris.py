import os, numpy as np, sys
sys.path.insert(0, '/Users/suryanallaparaju/Desktop/Surya\'s Project/code/FINAL_RESEARCH_PAPER/architectures')
import Architecture_A

iris_dir = '/Users/suryanallaparaju/Desktop/Surya\'s Project/data/Iris/Extracted_Codes'
files = [f for f in os.listdir(iris_dir) if f.endswith('.txt') and '_' in f and not f.split('_')[1].startswith('0')]

codes = {}
for f in files:
    try:
        data = np.array([int(float(x)) for x in open(os.path.join(iris_dir, f)).read().strip().split(',')])
        if len(data) == 512:
            codes[f] = data
    except Exception as e:
        pass

users = sorted(list(set([f.split('_')[0] for f in codes])), key=int)

genuine_dists = []
for u in users:
    u_files = [f for f in codes if f.split('_')[0] == u]
    for i in range(len(u_files)):
        for j in range(i+1, len(u_files)):
            genuine_dists.append(Architecture_A.fhd(codes[u_files[i]], codes[u_files[j]]))

imposter_dists = []
valid_users = [u for u in users if u + '_1.txt' in codes]
for i in range(len(valid_users)):
    for j in range(i+1, min(i+20, len(valid_users))):
        imposter_dists.append(Architecture_A.fhd(codes[valid_users[i]+'_1.txt'], codes[valid_users[j]+'_1.txt']))

print('\n--- IRIS ONLY VALIDATION (512-BIT CODES) ---')
print(f'Genuine Dists (Mean): {np.mean(genuine_dists):.4f}')
print(f'Imposter Dists (Mean): {np.mean(imposter_dists):.4f}')
print('Genuines under 0.20:', sum(d < 0.20 for d in genuine_dists), '/', len(genuine_dists))
print('Imposters under 0.20:', sum(d < 0.20 for d in imposter_dists), '/', len(imposter_dists))
print('Genuines under 0.40:', sum(d < 0.40 for d in genuine_dists), '/', len(genuine_dists))
print('Imposters under 0.40:', sum(d < 0.40 for d in imposter_dists), '/', len(imposter_dists))
print('Genuines under 0.46:', sum(d < 0.46 for d in genuine_dists), '/', len(genuine_dists))
print('Imposters under 0.46:', sum(d < 0.46 for d in imposter_dists), '/', len(imposter_dists))
