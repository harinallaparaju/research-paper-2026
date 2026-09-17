import os, numpy as np
base = '/Users/suryanallaparaju/Desktop/Surya\'s Project/data/Iris/Extracted_Codes'
files = [f for f in os.listdir(base) if f.endswith('.txt')][:200]
codes = {}
for f in files:
    lines = open(os.path.join(base, f)).read().strip().split('\n')
    v = lines[0].split(',') if ',' in lines[0] else lines
    codes[f] = np.array([int(float(x)) for x in v])

gen_dists = []
users = sorted(list(set([f.split('_')[0] for f in codes])))
for u in users:
    u_files = [f for f in codes if f.split('_')[0] == u]
    for i in range(len(u_files)):
        for j in range(i+1, len(u_files)):
            c1, c2 = codes[u_files[i]], codes[u_files[j]]
            if len(c1) == len(c2):
                gen_dists.append(np.sum(c1 != c2) / len(c1))

print("Genuine FHD:", np.mean(gen_dists) if gen_dists else "N/A")

imp_dists = []
for i in range(len(users)):
    for j in range(i+1, len(users)):
        f1, f2 = users[i]+'_1.txt', users[j]+'_1.txt'
        if f1 in codes and f2 in codes:
            c1, c2 = codes[f1], codes[f2]
            if len(c1) == len(c2):
                imp_dists.append(np.sum(c1 != c2) / len(c1))

print("Imposter FHD:", np.mean(imp_dists) if imp_dists else "N/A")
