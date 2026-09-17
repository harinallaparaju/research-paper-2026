"""Block size sweep: find optimal B for stabilizer-based architectures."""
import sys, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import load_chimeric_data
from architectures.architecture_b import lock_vault_b, unlock_vault_b

subjects = load_chimeric_data('fvc2002_1', min_iris=2)
n_subj = len(subjects)
print(f'Loaded {n_subj} subjects')

f, k = 24, 10

for B in [127, 255, 511, 1023]:
    print(f'\n=== Block size B={B} ===')

    # Monkey-patch the default block size
    import iris_stabilizer
    old_default = iris_stabilizer.DEFAULT_BLOCK_SIZE
    iris_stabilizer.DEFAULT_BLOCK_SIZE = B
    
    n_blocks = 49152 // B
    print(f'  {n_blocks} blocks, {n_blocks} key bits')

    locked = 0
    gen_ok = gen_fail = 0
    imp_ok = imp_total = 0
    vaults = {}

    # Lock all subjects
    for s in subjects:
        mins0 = s['minutiae'].get(0, [])
        iris0 = s['iris'].get(0)
        if not mins0 or iris0 is None:
            continue
        vault = lock_vault_b(mins0, iris0['code'], iris0['mask'], f, k, block_size=B)
        if vault is not None:
            vaults[s['chimeric_id']] = (vault, s)
            locked += 1

    # Genuine test (imp 0 → imp 1)
    for cid, (vault, s) in vaults.items():
        mins1 = s['minutiae'].get(1, [])
        iris1 = s['iris'].get(1)
        if not mins1 or iris1 is None:
            continue
        ok = unlock_vault_b(mins1, iris1['code'], iris1['mask'], vault)
        if ok:
            gen_ok += 1
        else:
            gen_fail += 1

    # Impostor test (all pairs)
    vault_list = list(vaults.items())
    for i in range(n_subj):
        s = subjects[i]
        mins1 = s['minutiae'].get(1, [])
        iris1 = s['iris'].get(1)
        if not mins1 or iris1 is None:
            continue
        for cid_j, (vault_j, _) in vault_list:
            if s['chimeric_id'] == cid_j:
                continue
            imp_total += 1
            ok = unlock_vault_b(mins1, iris1['code'], iris1['mask'], vault_j)
            if ok:
                imp_ok += 1

    ga = gen_ok + gen_fail
    far = imp_ok / imp_total * 100 if imp_total > 0 else 0
    frr = gen_fail / ga * 100 if ga > 0 else 0
    
    print(f'  Locked: {locked}')
    print(f'  Genuine: {gen_ok}/{ga} accept, FRR={frr:.2f}%')
    print(f'  Impostor: {imp_ok}/{imp_total} accept, FAR={far:.4f}%')
    print(f'  EER estimate: {(far+frr)/2:.4f}%')

    iris_stabilizer.DEFAULT_BLOCK_SIZE = old_default
